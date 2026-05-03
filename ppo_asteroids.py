# ppo_asteroids.py
#
# Self-contained PPO agent for the Asteroids game.
# Sim infrastructure is identical to dqn_asteroids.py so both files run independently.
# Only imports asteroids.py for watch mode (pygame rendering).
#
import sys
import os
import argparse
import random
from collections import deque
from math import sin, cos, pi, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# ---------------------------------------------------------------------------
# Game constants (must match asteroids.py)
# ---------------------------------------------------------------------------

NUM_ROCKS = 10
WIDTH = 900
HEIGHT = 700
winWidth = WIDTH + 1
winHeight = HEIGHT + 1
FPS = 60
REFERENCE_FPS = 150

WHITE = (255, 255, 255)
BLUE = (100, 149, 237)
RED = (220, 20, 60)

# ---------------------------------------------------------------------------
# Physics scaling
# ---------------------------------------------------------------------------

SIM_DT = REFERENCE_FPS / FPS  # ~2.5: all step() calls use this

# ---------------------------------------------------------------------------
# Toroidal distance helpers
# ---------------------------------------------------------------------------

def wrap_dist(a, b, mod):
    d = abs(a - b)
    return min(d, mod - d)

def wrap_delta(a, b, mod):
    """Signed shortest-path delta from a to b on a toroidal axis."""
    d = b - a
    if d > mod / 2:
        d -= mod
    elif d < -mod / 2:
        d += mod
    return d

def torus_dist(x1, y1, x2, y2):
    dx = wrap_dist(x1, x2, winWidth)
    dy = wrap_dist(y1, y2, winHeight)
    return sqrt(dx * dx + dy * dy)

# ---------------------------------------------------------------------------
# Lightweight simulation state (no pygame)
# ---------------------------------------------------------------------------

class SimShip:
    def __init__(self, x, y, dx, dy, theta, accel=0.02):
        self.x = x % winWidth
        self.y = y % winHeight
        self.dx = dx
        self.dy = dy
        self.theta = theta
        self.accel = accel
        self.d_theta = 0.0

    def copy(self):
        s = SimShip(self.x, self.y, self.dx, self.dy, self.theta, self.accel)
        s.d_theta = self.d_theta
        return s

    def step(self, thrust, left, right, dt=SIM_DT):
        self.d_theta = -1.5 if right else (1.5 if left else 0.0)
        if thrust:
            self.dx += self.accel * dt * -sin(self.theta * pi / 180)
            self.dy += self.accel * dt * -cos(self.theta * pi / 180)
        self.theta += self.d_theta * dt
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight


class SimRock:
    def __init__(self, x, y, dx, dy, radius):
        self.x = x % winWidth
        self.y = y % winHeight
        self.dx = dx
        self.dy = dy
        self.radius = radius

    def copy(self):
        return SimRock(self.x, self.y, self.dx, self.dy, self.radius)

    def step(self, dt=SIM_DT):
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight


class SimBullet:
    def __init__(self, x, y, dx, dy, dist_left):
        self.x = x % winWidth
        self.y = y % winHeight
        self.dx = dx
        self.dy = dy
        self.dist_left = dist_left
        self.speed = sqrt(dx * dx + dy * dy)

    def copy(self):
        return SimBullet(self.x, self.y, self.dx, self.dy, self.dist_left)

    def step(self, dt=SIM_DT):
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight
        self.dist_left -= self.speed * dt


def make_sim_bullet(ship_sim):
    speed = 5
    tdx = -sin(ship_sim.theta * pi / 180)
    tdy = -cos(ship_sim.theta * pi / 180)
    bdx = speed * tdx + ship_sim.d_theta * tdy + ship_sim.dx
    bdy = speed * tdy - ship_sim.d_theta * tdx + ship_sim.dy
    return SimBullet(ship_sim.x, ship_sim.y, bdx, bdy, 6 * winHeight / 7)


# ---------------------------------------------------------------------------
# Build sim state from live pygame sprites (used in watch mode)
# ---------------------------------------------------------------------------

def build_sim_state(ship, rocks, bullets):
    """Convert live pygame game objects into lightweight sim objects."""
    import asteroids as _astro

    s = SimShip(ship.p.x, ship.p.y, ship.dx, ship.dy, ship._theta)
    s.d_theta = ship.d_theta
    rs = []
    for r in rocks:
        radius = 50
        if isinstance(r, _astro.SmallRock):
            radius = 15
        elif isinstance(r, _astro.MediumRock):
            radius = 30
        elif isinstance(r, _astro.BigRock):
            radius = 50
        rs.append(SimRock(r.p.x, r.p.y, r.dx, r.dy, radius))
    bs = []
    for b in bullets:
        remaining = b.distance - b.distance_travelled
        bs.append(SimBullet(b.p.x, b.p.y, b.dx, b.dy, remaining))
    return s, rs, bs


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

# (thrust, left, right, shoot)
ACTIONS = [
    (False, False, False, False),  # drift
    (True,  False, False, False),  # thrust
    (False, True,  False, False),  # left
    (False, False, True,  False),  # right
    (True,  True,  False, False),  # thrust+left
    (True,  False, True,  False),  # thrust+right
    (False, False, False, True),   # shoot
    (False, True,  False, True),   # left+shoot
    (False, False, True,  True),   # right+shoot
    (True,  False, False, True),   # thrust+shoot
    (True,  True,  False, True),   # thrust+left+shoot
    (True,  False, True,  True),   # thrust+right+shoot
]

SHIP_RADIUS = 18
SIM_STEPS_PER_ACTION = 4

# ---------------------------------------------------------------------------
# Observation constants
# ---------------------------------------------------------------------------

MAX_ROCKS = 30          # observe up to this many rocks
MAX_BULLETS = 4         # observe up to this many bullets
SHIP_FEATURES = 13      # ship state + cooldown + bullet count + rotation rate + wave context
ROCK_FEATURES = 10      # pos(2) + vel(2) + radius + aim_dot + aim_cross + t_cpa + cpa_dist + occupied
BULLET_FEATURES = 5     # pos(2) + vel(2) + nearest_rock_dist
STATE_DIM = SHIP_FEATURES + MAX_ROCKS * ROCK_FEATURES + MAX_BULLETS * BULLET_FEATURES
N_ACTIONS = len(ACTIONS)

MAX_EPISODE_STEPS = 3500

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

LR              = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENT_COEF        = 0.015
VALUE_COEF      = 0.5
N_STEPS         = 2048      # transitions collected per PPO update
MINI_BATCH_SIZE = 256
PPO_EPOCHS      = 4
GRAD_CLIP       = 0.5

# Curriculum (same thresholds as DQN)
CURRICULUM_THRESHOLD = 220.0
CURRICULUM_MIN_EPS   = 5000

# ---------------------------------------------------------------------------
# Danger key (module-level factory; used in build_observation and AsteroidsEnv)
# ---------------------------------------------------------------------------

_DIAG = sqrt((winWidth / 2) ** 2 + (winHeight / 2) ** 2)


def _danger_key(ship):
    """Return a sort key that scores each rock by CPA threat (most dangerous = most negative)."""
    def key(r):
        dx = wrap_delta(ship.x, r.x, winWidth)
        dy = wrap_delta(ship.y, r.y, winHeight)
        rvx = (r.dx - ship.dx) * SIM_DT
        rvy = (r.dy - ship.dy) * SIM_DT
        rs2 = rvx * rvx + rvy * rvy
        if rs2 > 0.001:
            t = max(0.0, min(-(dx * rvx + dy * rvy) / rs2, 60.0))
            cd = sqrt((dx + rvx * t) ** 2 + (dy + rvy * t) ** 2) - 0.7 * (r.radius + SHIP_RADIUS)
        else:
            t = 60.0
            cd = sqrt(dx * dx + dy * dy) - 0.7 * (r.radius + SHIP_RADIUS)
        return -(r.radius / 30.0) * (1.0 + max(0.0, 1.0 - t / 20.0)) / max(1.0, cd + 8.0)
    return key


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_observation(ship, rocks, bullets, shoot_cooldown=0, wave_rocks=1, waves_cleared=0):
    """Convert sim state to a fixed-size numpy float32 vector."""
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    fwd_x = -sin(ship.theta * pi / 180)
    fwd_y = -cos(ship.theta * pi / 180)

    # Actual bullet direction: heading + rotation deflection + ship velocity.
    _blt_dx = 5.0 * fwd_x + ship.d_theta * fwd_y + ship.dx
    _blt_dy = 5.0 * fwd_y - ship.d_theta * fwd_x + ship.dy
    _blt_spd = sqrt(_blt_dx * _blt_dx + _blt_dy * _blt_dy)
    if _blt_spd > 1e-6:
        blt_x, blt_y = _blt_dx / _blt_spd, _blt_dy / _blt_spd
    else:
        blt_x, blt_y = fwd_x, fwd_y

    # Ship features (13)
    obs[idx]     = ship.x / winWidth - 0.5
    obs[idx + 1] = ship.y / winHeight - 0.5
    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
    obs[idx + 2] = float(np.tanh(ship.dx / 15.0))
    obs[idx + 3] = float(np.tanh(ship.dy / 15.0))
    obs[idx + 4] = float(np.tanh(speed / 15.0))
    obs[idx + 5] = sin(ship.theta * pi / 180)
    obs[idx + 6] = cos(ship.theta * pi / 180)
    obs[idx + 7] = min(len(rocks), MAX_ROCKS) / MAX_ROCKS
    obs[idx + 8] = max(0.0, shoot_cooldown) / 15.0
    obs[idx + 9] = len(bullets) / MAX_BULLETS
    obs[idx + 10] = ship.d_theta / 1.5
    obs[idx + 11] = len(rocks) / max(1, 7 * wave_rocks)
    obs[idx + 12] = min(waves_cleared, 5) / 5.0
    idx += SHIP_FEATURES

    # Sort rocks by CPA danger: most threatening first.
    rock_list = sorted(rocks, key=_danger_key(ship))

    for i in range(MAX_ROCKS):
        if i < len(rock_list):
            r = rock_list[i]
            dx_to = wrap_delta(ship.x, r.x, winWidth)
            dy_to = wrap_delta(ship.y, r.y, winHeight)
            dist_to = sqrt(dx_to ** 2 + dy_to ** 2)
            obs[idx]     = dx_to / _DIAG
            obs[idx + 1] = dy_to / _DIAG
            obs[idx + 2] = float(np.tanh((r.dx - ship.dx) / 15.0))
            obs[idx + 3] = float(np.tanh((r.dy - ship.dy) / 15.0))
            obs[idx + 4] = r.radius / 50.0
            if dist_to > 1e-6:
                t_flight = dist_to / max(1e-6, _blt_spd * SIM_DT)
                lead_x = dx_to + (r.dx - ship.dx) * SIM_DT * t_flight
                lead_y = dy_to + (r.dy - ship.dy) * SIM_DT * t_flight
                lead_dist = sqrt(lead_x * lead_x + lead_y * lead_y)
                if lead_dist > 1e-6:
                    nx, ny = lead_x / lead_dist, lead_y / lead_dist
                else:
                    nx, ny = dx_to / dist_to, dy_to / dist_to
                obs[idx + 5] = blt_x * nx + blt_y * ny
                obs[idx + 6] = blt_x * ny - blt_y * nx
            rel_vx = (r.dx - ship.dx) * SIM_DT
            rel_vy = (r.dy - ship.dy) * SIM_DT
            rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
            if rel_speed_sq > 0.001:
                t_cpa = -(dx_to * rel_vx + dy_to * rel_vy) / rel_speed_sq
                t_cpa = max(0.0, min(t_cpa, 60.0))
                cpa_dx = dx_to + rel_vx * t_cpa
                cpa_dy = dy_to + rel_vy * t_cpa
                cpa_dist = sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy) - 0.7 * (r.radius + SHIP_RADIUS)
            else:
                t_cpa = 60.0
                cpa_dist = dist_to - 0.7 * (r.radius + SHIP_RADIUS)
            obs[idx + 7] = t_cpa / 60.0
            obs[idx + 8] = max(-1.0, min(1.0, cpa_dist / 200.0))
            obs[idx + 9] = 1.0
        idx += ROCK_FEATURES

    if rocks:
        bullet_list = sorted(bullets, key=lambda b: min(torus_dist(b.x, b.y, r.x, r.y) for r in rocks))
    else:
        bullet_list = list(bullets)

    for i in range(MAX_BULLETS):
        if i < len(bullet_list):
            b = bullet_list[i]
            obs[idx]     = wrap_delta(ship.x, b.x, winWidth) / _DIAG
            obs[idx + 1] = wrap_delta(ship.y, b.y, winHeight) / _DIAG
            obs[idx + 2] = float(np.tanh((b.dx - ship.dx) / 5.0))
            obs[idx + 3] = float(np.tanh((b.dy - ship.dy) / 5.0))
            if rocks:
                min_br = min(torus_dist(b.x, b.y, r.x, r.y) for r in rocks)
                obs[idx + 4] = min_br / _DIAG
            else:
                obs[idx + 4] = 1.0
        idx += BULLET_FEATURES

    return obs


# ---------------------------------------------------------------------------
# Asteroids Environment (identical reward function to dqn_asteroids.py)
# ---------------------------------------------------------------------------

class AsteroidsEnv:
    """Gym-like environment wrapping the lightweight sim classes."""

    def __init__(self, num_rocks=1):
        self.num_rocks = num_rocks
        self.wave_rocks = num_rocks
        self.ship = None
        self.rocks = []
        self.bullets = []
        self.steps = 0
        self.score = 0
        self.alive = True
        self.shoot_cooldown = 0
        self.waves_cleared_this_ep = 0

    def reset(self):
        self.wave_rocks = self.num_rocks
        self.ship = SimShip(winWidth / 2, winHeight / 2, 0, 0, 0)
        self.rocks = []
        for _ in range(self.wave_rocks):
            self._spawn_big_rock()
        self.bullets = []
        self.steps = 0
        self.score = 0
        self.alive = True
        self.shoot_cooldown = 0
        self.waves_cleared_this_ep = 0
        return build_observation(self.ship, self.rocks, self.bullets, self.shoot_cooldown,
                                 self.wave_rocks, self.waves_cleared_this_ep)

    def _spawn_big_rock(self):
        if random.randint(0, 1):
            x = random.randint(-int(winWidth / 20), int(winWidth / 20))
            y = random.randint(0, winHeight)
        else:
            x = random.randint(0, winWidth)
            y = random.randint(-int(winHeight / 20), int(winHeight / 20))
        dx = dy = 0
        while dx == 0 and dy == 0:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        dx *= 0.2
        dy *= 0.2
        self.rocks.append(SimRock(x, y, dx, dy, 50))

    def _spawn_children(self, rock):
        """Spawn 2 child rocks when a rock is destroyed."""
        if rock.radius >= 50:
            child_radius = 30
            speed_range = 4
        elif rock.radius >= 30:
            child_radius = 15
            speed_range = 6
        else:
            return

        for _ in range(2):
            dx = dy = 0
            while dx == 0 and dy == 0:
                dx = random.randint(-speed_range, speed_range)
                dy = random.randint(-speed_range, speed_range)
            dx *= 0.2
            dy *= 0.2
            self.rocks.append(SimRock(rock.x, rock.y, dx, dy, child_radius))

    def _cpa_danger_top3(self):
        """Sum of the 3 highest per-rock CPA threat scores."""
        threats = []
        ship = self.ship
        for r in self.rocks:
            dx_to = wrap_delta(ship.x, r.x, winWidth)
            dy_to = wrap_delta(ship.y, r.y, winHeight)
            center_dist = sqrt(dx_to * dx_to + dy_to * dy_to)
            rel_vx = (r.dx - ship.dx) * SIM_DT
            rel_vy = (r.dy - ship.dy) * SIM_DT
            rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
            if rel_speed_sq > 0.001:
                t_cpa = -(dx_to * rel_vx + dy_to * rel_vy) / rel_speed_sq
                t_cpa = max(0.0, min(t_cpa, 60.0))
                cpa_dx = dx_to + rel_vx * t_cpa
                cpa_dy = dy_to + rel_vy * t_cpa
                cpa_dist = sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy) - 0.7 * (r.radius + SHIP_RADIUS)
            else:
                t_cpa = 60.0
                cpa_dist = center_dist - 0.7 * (r.radius + SHIP_RADIUS)
            if cpa_dist < 200:
                urgency = 1.0 + max(0.0, 1.0 - t_cpa / 20.0)
                size_mult = r.radius / 30.0
                closing = (dx_to * rel_vx + dy_to * rel_vy) / max(1.0, center_dist)
                speed_mult = 1.0 + max(0.0, -closing) * 0.8
                threats.append(size_mult * urgency * speed_mult / max(1.0, cpa_dist + 3.0))
        threats.sort(reverse=True)
        return sum(threats[:3])

    def step(self, action_idx):
        """Advance simulation by SIM_STEPS_PER_ACTION frames. Returns (obs, reward, done)."""
        action = ACTIONS[action_idx]
        thrust, left, right, shoot = action

        reward = 0.0
        shot_fired = False

        fire_fwd_x = -sin(self.ship.theta * pi / 180)
        fire_fwd_y = -cos(self.ship.theta * pi / 180)
        fire_speed_dt = 5.0 * SIM_DT

        if shoot and self.shoot_cooldown <= 0 and len(self.bullets) < MAX_BULLETS:
            _nb = make_sim_bullet(self.ship)
            self.bullets.append(_nb)
            self.shoot_cooldown = 15
            shot_fired = True
            if _nb.speed > 1e-6:
                fire_fwd_x = _nb.dx / _nb.speed
                fire_fwd_y = _nb.dy / _nb.speed
                fire_speed_dt = _nb.speed * SIM_DT

        for _ in range(SIM_STEPS_PER_ACTION):
            self.ship.step(thrust, left, right)
            for r in self.rocks:
                r.step()

            remaining_bullets = []
            for b in self.bullets:
                b.step()
                if b.dist_left > 0:
                    remaining_bullets.append(b)
            self.bullets = remaining_bullets

            if self.shoot_cooldown > 0:
                self.shoot_cooldown -= SIM_DT

            hit_rocks = set()
            hit_bullets = set()
            for bi, b in enumerate(self.bullets):
                for ri, r in enumerate(self.rocks):
                    if ri not in hit_rocks:
                        if torus_dist(b.x, b.y, r.x, r.y) < r.radius + 5:
                            hit_rocks.add(ri)
                            hit_bullets.add(bi)
                            if r.radius <= 15:
                                reward += 12.0
                            elif r.radius <= 30:
                                reward += 8.0
                            else:
                                reward += 5.0
                            break

            killed = [self.rocks[ri] for ri in hit_rocks]
            if hit_rocks:
                self.rocks = [r for i, r in enumerate(self.rocks) if i not in hit_rocks]
                self.bullets = [b for i, b in enumerate(self.bullets) if i not in hit_bullets]

            for r in self.rocks:
                if torus_dist(self.ship.x, self.ship.y, r.x, r.y) < 0.7 * (r.radius + SHIP_RADIUS):
                    self.alive = False
                    self.steps += 1
                    obs = build_observation(self.ship, self.rocks, self.bullets, self.shoot_cooldown,
                                            self.wave_rocks, self.waves_cleared_this_ep)
                    death_penalty = 15.0 + 5.0 * self.waves_cleared_this_ep
                    self.score += reward - death_penalty
                    return obs, reward - death_penalty, True

            for r in killed:
                self._spawn_children(r)

        self.steps += 1
        reward += 0.01  # survival: small keep-alive signal; killing rocks must dominate

        dist_to_center = torus_dist(
            self.ship.x, self.ship.y, winWidth / 2, winHeight / 2
        )
        reward += 0.02 * max(0.0, 1.0 - dist_to_center / 400.0)

        if len(self.rocks) == 0:
            self.waves_cleared_this_ep += 1
            reward += 8.0 * self.wave_rocks
            dist_to_center = torus_dist(
                self.ship.x, self.ship.y, winWidth / 2, winHeight / 2
            )
            reward += 10.0 * max(0.0, 1.0 - dist_to_center / 300.0)
            _speed = sqrt(self.ship.dx ** 2 + self.ship.dy ** 2)
            reward += 3.0 * max(0.0, 1.0 - _speed / 8.0)
            self.wave_rocks = min(self.wave_rocks + 1, self.num_rocks + 1)
            for _ in range(self.wave_rocks):
                self._spawn_big_rock()

        # Penalize shooting while spinning: deliberate aiming requires stopping rotation.
        if shot_fired:
            spin_frac = abs(self.ship.d_theta) / 1.5
            reward -= 1.5 * max(0.0, spin_frac - 0.3)

        # On-fire bonus: best aim reward across all rocks.
        # Only granted at low angular velocity — spinning must not substitute for aiming.
        # Wasted shots (best fire_dot < 0.5) are penalized to prevent spray strategies.
        if shot_fired:
            best_aim_r = 0.0
            best_fire_dot = -1.0
            for r in self.rocks:
                dx_to = wrap_delta(self.ship.x, r.x, winWidth)
                dy_to = wrap_delta(self.ship.y, r.y, winHeight)
                dist_to = sqrt(dx_to * dx_to + dy_to * dy_to)
                if dist_to > r.radius:
                    t_flight = dist_to / max(1e-6, fire_speed_dt)
                    lead_x = dx_to + (r.dx - self.ship.dx) * SIM_DT * t_flight
                    lead_y = dy_to + (r.dy - self.ship.dy) * SIM_DT * t_flight
                    lead_dist = sqrt(lead_x * lead_x + lead_y * lead_y)
                    if lead_dist > 1e-6:
                        fire_dot = fire_fwd_x * lead_x / lead_dist + fire_fwd_y * lead_y / lead_dist
                        best_fire_dot = max(best_fire_dot, fire_dot)
                        if abs(self.ship.d_theta) < 0.3:
                            cos_hit = sqrt(max(0.0, 1.0 - (r.radius / dist_to) ** 2))
                            approach = (fire_dot - cos_hit) / max(1e-6, 1.0 - cos_hit)
                            aim_r = max(0.0, approach) * 1.5 + max(0.0, fire_dot) * 0.2
                            best_aim_r = max(best_aim_r, aim_r)
            reward += best_aim_r
            reward -= 0.5 * max(0.0, 0.5 - best_fire_dot)
            if self.rocks:
                min_dist = min(
                    torus_dist(self.ship.x, self.ship.y, r.x, r.y) for r in self.rocks
                )
                reward -= 2.0 * max(0.0, 1.0 - min_dist / 120.0)

        if self.rocks:
            reward -= self._cpa_danger_top3() * 0.35

        _speed = sqrt(self.ship.dx ** 2 + self.ship.dy ** 2)
        if _speed > 4.0:
            reward -= (_speed - 4.0) * 0.04

        reward -= abs(self.ship.d_theta) * 0.02 * SIM_STEPS_PER_ACTION

        done = self.steps >= MAX_EPISODE_STEPS
        obs = build_observation(self.ship, self.rocks, self.bullets, self.shoot_cooldown,
                                self.wave_rocks, self.waves_cleared_this_ep)
        self.score += reward
        return obs, reward, done


# ---------------------------------------------------------------------------
# PPO Actor-Critic network
# ---------------------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """Shared 3-layer MLP backbone with separate policy and value heads."""

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, obs_t):
        """Sample action stochastically; return (action_int, log_prob_float, value_float)."""
        with torch.no_grad():
            logits, value = self.forward(obs_t.unsqueeze(0))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_t, actions_t):
        """Batch evaluate for PPO update. Returns (log_probs, values, entropy)."""
        logits, values = self.forward(obs_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_t), values, dist.entropy()


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, n_steps, state_dim):
        self.n   = n_steps
        self.dim = state_dim
        self.reset()

    def reset(self):
        self.states    = np.zeros((self.n, self.dim), dtype=np.float32)
        self.actions   = np.zeros(self.n, dtype=np.int64)
        self.rewards   = np.zeros(self.n, dtype=np.float32)
        self.dones     = np.zeros(self.n, dtype=np.float32)
        self.log_probs = np.zeros(self.n, dtype=np.float32)
        self.values    = np.zeros(self.n, dtype=np.float32)
        self.ptr       = 0

    def store(self, state, action, reward, done, log_prob, value):
        i = self.ptr
        self.states[i]    = state
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = float(done)
        self.log_probs[i] = log_prob
        self.values[i]    = value
        self.ptr += 1

    def compute_gae(self, next_value):
        """Compute GAE advantages and lambda-returns in-place."""
        self.advantages = np.zeros(self.n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n)):
            nv  = next_value if t == self.n - 1 else self.values[t + 1]
            nnt = 1.0 - self.dones[t]
            delta    = self.rewards[t] + GAMMA * nv * nnt - self.values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * nnt * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------

def ppo_update(net, optimizer, buffer, device, ent_coef=ENT_COEF):
    states   = torch.tensor(buffer.states,    device=device)
    actions  = torch.tensor(buffer.actions,   device=device)
    old_lps  = torch.tensor(buffer.log_probs, device=device)
    returns  = torch.tensor(buffer.returns,   device=device)
    advs     = torch.tensor(buffer.advantages, device=device)
    advs     = (advs - advs.mean()) / (advs.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        idxs = np.random.permutation(N_STEPS)
        for start in range(0, N_STEPS, MINI_BATCH_SIZE):
            b = idxs[start:start + MINI_BATCH_SIZE]
            lp, v, ent = net.evaluate(states[b], actions[b])
            ratio  = torch.exp(lp - old_lps[b])
            adv_b  = advs[b]
            surr   = torch.min(
                ratio * adv_b,
                torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b,
            )
            loss = -surr.mean() + VALUE_COEF * F.mse_loss(v, returns[b]) - ent_coef * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_episodes=50000, save_every=100, model_path="ppo_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net       = PPOActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    cur_rocks    = 1
    total_steps  = 0
    episode      = 0
    eps_at_level = 0  # episodes completed at the current curriculum level

    if os.path.exists(model_path):
        print(f"Resuming from {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        try:
            net.load_state_dict(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            total_steps  = ckpt.get("total_steps", 0)
            episode      = ckpt.get("episode", 0)
            cur_rocks    = ckpt.get("cur_rocks", 1)
            eps_at_level = ckpt.get("eps_at_level", 0)
            print(f"  total_steps={total_steps}  episode={episode}  cur_rocks={cur_rocks}")
        except RuntimeError as e:
            print(f"Checkpoint incompatible: {e}")
            print("Delete old .pt files to start fresh.")

    env    = AsteroidsEnv(num_rocks=cur_rocks)
    buffer = RolloutBuffer(N_STEPS, STATE_DIM)
    state  = env.reset()
    done   = False

    reward_history     = deque(maxlen=100)
    length_history     = deque(maxlen=100)
    wave_clear_history = deque(maxlen=100)
    best_avg           = -float("inf")

    ep_reward      = 0.0
    ep_steps       = 0
    update         = 0
    target_episode = episode + num_episodes   # always train num_episodes more, regardless of checkpoint

    while episode < target_episode:
        buffer.reset()

        for _ in range(N_STEPS):
            obs_t  = torch.tensor(state, device=device)
            action, log_prob, value = net.act(obs_t)
            next_state, reward, done = env.step(action)
            buffer.store(state, action, reward, done, log_prob, value)
            state      = next_state
            ep_reward += reward
            ep_steps  += 1
            total_steps += 1

            if done:
                reward_history.append(ep_reward)
                length_history.append(ep_steps)
                wave_clear_history.append(env.waves_cleared_this_ep)
                episode      += 1
                eps_at_level += 1
                ep_reward = 0.0
                ep_steps  = 0
                state = env.reset()
                done  = False

        # Bootstrap value for the last (possibly mid-episode) state
        with torch.no_grad():
            _, boot_val = net.forward(torch.tensor(state, device=device).unsqueeze(0))
        buffer.compute_gae(boot_val.item())
        frac = min(1.0, total_steps / 8_000_000)
        for pg in optimizer.param_groups:
            pg['lr'] = LR * (1.0 - 0.8 * frac)
        cur_ent_coef = ENT_COEF * (1.0 - 0.5 * frac)
        ppo_update(net, optimizer, buffer, device, ent_coef=cur_ent_coef)
        update += 1

        # Logging (every 10 updates = ~20k env steps)
        if update % 10 == 0 and len(reward_history) > 0:
            avg       = np.mean(reward_history)
            avg_len   = np.mean(length_history)
            avg_waves = sum(wave_clear_history) / max(1, len(wave_clear_history))
            print(
                f"Upd {update:5d} | Ep {episode:6d} | avg100 {avg:6.1f}/{CURRICULUM_THRESHOLD:.0f} | "
                f"len {avg_len:5.0f} | rocks {cur_rocks} | "
                f"waves {avg_waves:.2f} | "
                f"clears {sum(wave_clear_history):3d} | "
                f"3x {sum(1 for x in wave_clear_history if x >= 3):3d} | "
                f"steps {total_steps}"
            )

        # Curriculum promotion
        if (len(reward_history) >= 100
                and cur_rocks < NUM_ROCKS
                and eps_at_level >= CURRICULUM_MIN_EPS):
            avg       = np.mean(reward_history)
            avg_len   = np.mean(length_history)
            avg_waves = sum(wave_clear_history) / max(1, len(wave_clear_history))
            if (avg > CURRICULUM_THRESHOLD
                    and avg_len >= 600
                    and avg_waves >= 2.0
                    and sum(wave_clear_history) >= 50
                    and sum(1 for x in wave_clear_history if x >= 3) >= 20):
                cur_rocks += 1
                env.num_rocks = cur_rocks
                reward_history.clear()
                length_history.clear()
                wave_clear_history.clear()
                eps_at_level = 0
                best_avg     = -float("inf")
                print(f"=== CURRICULUM: now training with {cur_rocks} rocks ===")

        # Periodic checkpoint
        if episode > 0 and episode % save_every < (N_STEPS // MAX_EPISODE_STEPS + 1):
            _save(net, optimizer, model_path, total_steps, episode, cur_rocks, eps_at_level)
            if len(reward_history) >= 100:
                avg = np.mean(reward_history)
                if avg > best_avg:
                    best_avg = avg
                    _save(net, optimizer, model_path.replace(".pt", "_best.pt"),
                          total_steps, episode, cur_rocks, eps_at_level)
                    print(f"  -> new best avg: {best_avg:.1f}")
                elif best_avg > 5.0 and avg < 0.4 * best_avg:
                    print(f"  !! DEGRADATION: avg {avg:.1f} << best {best_avg:.1f}")

    _save(net, optimizer, model_path, total_steps, episode, cur_rocks, eps_at_level)
    print("Training complete.")


def _save(net, optimizer, path, total_steps, episode, cur_rocks, eps_at_level):
    torch.save({
        "net":          net.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "total_steps":  total_steps,
        "episode":      episode,
        "cur_rocks":    cur_rocks,
        "eps_at_level": eps_at_level,
    }, path)


# ---------------------------------------------------------------------------
# Watch mode: load trained model and play with pygame rendering
# ---------------------------------------------------------------------------

def watch(model_path="ppo_model.pt"):
    import pygame
    from pygame.locals import QUIT, KEYUP, K_q
    import asteroids as _astro

    if not os.path.exists(model_path):
        print(f"No model found at '{model_path}'. Run 'python ppo_asteroids.py train' first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PPOActorCritic().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    try:
        net.load_state_dict(ckpt["net"])
    except RuntimeError as e:
        print(f"Checkpoint incompatible: {e}")
        sys.exit(1)
    net.eval()

    pygame.init()
    fpsClock = pygame.time.Clock()
    _astro.fpsClock = fpsClock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroids - PPO Agent")
    pygame.mouse.set_visible(0)

    Ship     = _astro.Ship
    Bullet   = _astro.Bullet
    BigRock  = _astro.BigRock
    Score    = _astro.Score
    textBlit = _astro.textBlit

    ship    = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks   = pygame.sprite.Group()
    num_rocks = NUM_ROCKS
    Score.reset()

    while len(rocks) < num_rocks:
        BigRock(screen, rocks)

    shoot_cooldown      = 0
    action_frames_left  = 0
    thrust = left = right = False
    fire_this_frame = False

    while True:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYUP and event.key == K_q:
                pygame.quit()
                sys.exit()

        if action_frames_left <= 0:
            s, rs, bs = build_sim_state(ship, rocks, bullets)
            obs = build_observation(s, rs, bs, shoot_cooldown)
            with torch.no_grad():
                logits, _ = net.forward(torch.tensor(obs, device=device).unsqueeze(0))
            action_idx = torch.distributions.Categorical(logits=logits).sample().item()
            thrust, left, right, shoot = ACTIONS[action_idx]
            action_frames_left = SIM_STEPS_PER_ACTION

            if shoot and shoot_cooldown <= 0 and len(bullets) < MAX_BULLETS:
                fire_this_frame = True
                shoot_cooldown  = 15

        action_frames_left -= 1

        if shoot_cooldown > 0:
            shoot_cooldown -= dt

        screen.fill(WHITE)
        ship.update(thrust, left, right, dt)

        if fire_this_frame:
            Bullet(screen, ship, bullets)
            fire_this_frame = False

        if bullets:
            bullets.update(bullets, rocks, dt)
        rocks.update(dt)

        if pygame.sprite.spritecollideany(ship, rocks, pygame.sprite.collide_circle_ratio(0.7)):
            Score.delLife()
            if Score.getLives() == 0:
                textBlit(screen, f"Final score: {Score.get()}", "Arial", 60, RED,
                         "center", winWidth / 2, winHeight / 2)
                pygame.display.update()
                pygame.time.wait(3000)
                Score.reset()
                num_rocks = NUM_ROCKS
            bullets.empty()
            rocks.empty()
            del ship
            pygame.event.clear()
            ship = Ship(screen)
            while len(rocks) < num_rocks:
                BigRock(screen, rocks)
            shoot_cooldown     = 0
            action_frames_left = 0
            thrust = left = right = False
            fire_this_frame = False

        Score.draw(screen, rocks)
        textBlit(screen, "PPO Agent", "Arial", 30, BLUE,
                 "bottomleft", winWidth / 20, 18 * winHeight / 20, False)

        if len(rocks) == 0:
            num_rocks += 1
            bullets.empty()
            pygame.event.clear()
            while len(rocks) < num_rocks:
                BigRock(screen, rocks)

        pygame.display.update()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO agent for Asteroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser(
        "train",
        help="Train the PPO agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_p.add_argument("-episodes", type=int, default=50000, help="number of training episodes")
    train_p.add_argument("-save-every", type=int, default=100, help="save checkpoint every N episodes")
    train_p.add_argument("-model", default="ppo_model.pt", help="checkpoint file path")

    watch_p = sub.add_parser(
        "watch",
        help="Watch a trained agent play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    watch_p.add_argument("-model", default="ppo_model.pt", help="checkpoint file to load")

    args = parser.parse_args()

    if args.command == "train":
        train(num_episodes=args.episodes, save_every=args.save_every, model_path=args.model)
    elif args.command == "watch":
        watch(model_path=args.model)
    else:
        parser.print_help()
