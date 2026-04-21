# dqn_asteroids.py
#
# Self-contained DQN agent for the Asteroids game.
# Only imports asteroids.py for watch mode (pygame rendering).
# Training runs fully headless using lightweight sim classes defined here.
#
import sys
import os
import argparse
import random
from collections import deque
from math import sin, cos, pi, sqrt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------------------------------------------
# Game constants (must match asteroids.py)
# ---------------------------------------------------------------------------

NUM_ROCKS = 5
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
    # Import here to avoid circular / early pygame init
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
# Hyperparameters
# ---------------------------------------------------------------------------

MAX_ROCKS = 12          # observe up to this many rocks
MAX_BULLETS = 2         # observe up to this many bullets
SHIP_FEATURES = 11      # ship state + cooldown + bullet count + rotation rate
ROCK_FEATURES = 10      # pos(2) + vel(2) + radius + aim_dot + aim_cross + t_cpa + cpa_dist + occupied
BULLET_FEATURES = 5     # pos(2) + vel(2) + nearest_rock_dist
STATE_DIM = SHIP_FEATURES + MAX_ROCKS * ROCK_FEATURES + MAX_BULLETS * BULLET_FEATURES
N_ACTIONS = len(ACTIONS)

GAMMA = 0.995
LR = 1e-4
BATCH_SIZE = 256
REPLAY_SIZE = 400_000
TARGET_TAU = 0.01           # Polyak soft-update rate for target network
GRAD_STEPS_PER_ENV = 2      # gradient updates per environment step
GRAD_CLIP = 1.0             # max gradient norm
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1_000_000       # linear decay over this many steps
EPS_LEVEL_DECAY = 500_000   # per-level exploration budget; resets on curriculum promotion
MAX_EPISODE_STEPS = 16000   # ~267 seconds of game time at 60fps

# Prioritized Experience Replay
PER_ALPHA = 0.4
PER_BETA_START = 0.6
PER_BETA_END = 1.0
PER_BETA_FRAMES = 100_000
PER_EPSILON = 1e-6

# N-step returns
N_STEPS = 5

# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

_DIAG = sqrt((winWidth / 2) ** 2 + (winHeight / 2) ** 2)


def build_observation(ship, rocks, bullets, shoot_cooldown=0):
    """Convert sim state to a fixed-size numpy float32 vector."""
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    fwd_x = -sin(ship.theta * pi / 180)
    fwd_y = -cos(ship.theta * pi / 180)

    # Ship features (11)
    obs[idx]     = ship.x / winWidth - 0.5
    obs[idx + 1] = ship.y / winHeight - 0.5
    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
    obs[idx + 2] = float(np.tanh(ship.dx / 5.0))
    obs[idx + 3] = float(np.tanh(ship.dy / 5.0))
    obs[idx + 4] = float(np.tanh(speed / 5.0))
    obs[idx + 5] = sin(ship.theta * pi / 180)
    obs[idx + 6] = cos(ship.theta * pi / 180)
    obs[idx + 7] = min(len(rocks), MAX_ROCKS) / MAX_ROCKS
    obs[idx + 8] = max(0.0, shoot_cooldown) / 15.0   # shot availability
    obs[idx + 9] = len(bullets) / 6.0                 # bullet count
    obs[idx + 10] = ship.d_theta / 1.5                # rotation rate: -1=full-right, +1=full-left
    idx += SHIP_FEATURES

    # Sort rocks by CPA danger: most threatening (soonest, closest approach) first.
    # This puts the primary evasion target at a consistent observation index.
    def _danger_key(r):
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

    rock_list = sorted(rocks, key=_danger_key)

    for i in range(MAX_ROCKS):
        if i < len(rock_list):
            r = rock_list[i]
            dx_to = wrap_delta(ship.x, r.x, winWidth)
            dy_to = wrap_delta(ship.y, r.y, winHeight)
            dist_to = sqrt(dx_to ** 2 + dy_to ** 2)
            obs[idx]     = dx_to / _DIAG
            obs[idx + 1] = dy_to / _DIAG
            obs[idx + 2] = float(np.tanh((r.dx - ship.dx) / 5.0))
            obs[idx + 3] = float(np.tanh((r.dy - ship.dy) / 5.0))
            obs[idx + 4] = r.radius / 50.0
            # Aim features: dot and cross of heading with lead-targeted rock direction.
            # Uses same lead formula as the reward function so obs aligns with reward.
            if dist_to > 1e-6:
                t_flight = dist_to / (5.0 * SIM_DT)
                lead_x = dx_to + (r.dx - ship.dx) * SIM_DT * t_flight
                lead_y = dy_to + (r.dy - ship.dy) * SIM_DT * t_flight
                lead_dist = sqrt(lead_x * lead_x + lead_y * lead_y)
                if lead_dist > 1e-6:
                    nx, ny = lead_x / lead_dist, lead_y / lead_dist
                else:
                    nx, ny = dx_to / dist_to, dy_to / dist_to
                obs[idx + 5] = fwd_x * nx + fwd_y * ny   # aim_dot (-1 to 1)
                obs[idx + 6] = fwd_x * ny - fwd_y * nx   # aim_cross (turn direction)
            # CPA features: time-to-closest-approach and miss distance
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
            obs[idx + 7] = t_cpa / 60.0                           # 0=now, 1=far future
            obs[idx + 8] = max(-1.0, min(1.0, cpa_dist / 200.0)) # 0=collision boundary, neg=lethal
            obs[idx + 9] = 1.0                                     # occupied flag
        idx += ROCK_FEATURES

    # Sort bullets by proximity to nearest rock (closest bullet-rock pair most relevant).
    if rocks:
        bullet_list = sorted(bullets, key=lambda b: min(torus_dist(b.x, b.y, r.x, r.y) for r in rocks))
    else:
        bullet_list = list(bullets)

    for i in range(MAX_BULLETS):
        if i < len(bullet_list):
            b = bullet_list[i]
            obs[idx]     = wrap_delta(ship.x, b.x, winWidth) / _DIAG
            obs[idx + 1] = wrap_delta(ship.y, b.y, winHeight) / _DIAG
            obs[idx + 2] = b.dx / 10.0
            obs[idx + 3] = b.dy / 10.0
            # Distance from this bullet to nearest rock
            if rocks:
                min_br = min(torus_dist(b.x, b.y, r.x, r.y) for r in rocks)
                obs[idx + 4] = min_br / _DIAG
            else:
                obs[idx + 4] = 1.0
        idx += BULLET_FEATURES

    return obs


# ---------------------------------------------------------------------------
# Asteroids Environment
# ---------------------------------------------------------------------------

class AsteroidsEnv:
    """Gym-like environment wrapping the lightweight sim classes."""

    def __init__(self, num_rocks=1):
        self.num_rocks = num_rocks
        self.wave_rocks = num_rocks  # rocks in current wave (grows each wave)
        self.ship = None
        self.rocks = []
        self.bullets = []
        self.steps = 0
        self.score = 0
        self.alive = True
        self.shoot_cooldown = 0
        self.cleared_wave_this_ep = False

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
        self.cleared_wave_this_ep = False
        return build_observation(self.ship, self.rocks, self.bullets, self.shoot_cooldown)

    def _spawn_big_rock(self):
        if random.randint(0, 1):
            x = random.randint(-int(winWidth / 20), int(winWidth / 20))
            y = random.randint(0, winHeight)
        else:
            x = random.randint(0, winWidth)
            y = random.randint(-int(winWidth / 20), int(winHeight / 20))
        dx = dy = 0
        while dx == 0 and dy == 0:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        dx *= 0.2
        dy *= 0.2
        self.rocks.append(SimRock(x, y, dx, dy, 50))

    def _spawn_children(self, rock):
        """Spawn 2 child rocks when a rock is destroyed."""
        if rock.radius >= 50:  # big -> 2 medium
            child_radius = 30
            speed_range = 4
        elif rock.radius >= 30:  # medium -> 2 small
            child_radius = 15
            speed_range = 6
        else:
            return  # small rocks just disappear

        for _ in range(2):
            dx = dy = 0
            while dx == 0 and dy == 0:
                dx = random.randint(-speed_range, speed_range)
                dy = random.randint(-speed_range, speed_range)
            dx *= 0.2
            dy *= 0.2
            self.rocks.append(SimRock(rock.x, rock.y, dx, dy, child_radius))

    def _cpa_danger(self):
        """Compute CPA-based danger score: max single-rock threat.
        Using max rather than sum/N ensures the evasion signal stays constant
        regardless of rock count — the most dangerous rock always generates full
        pressure without overwhelming the aim bonus when many rocks are present."""
        best = 0.0
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
                threat = size_mult * urgency * speed_mult / max(1.0, cpa_dist + 3.0)
                if threat > best:
                    best = threat
        return best

    def step(self, action_idx):
        """Advance simulation by SIM_STEPS_PER_ACTION frames. Returns (obs, reward, done)."""
        action = ACTIONS[action_idx]
        thrust, left, right, shoot = action

        reward = 0.0
        shot_fired = False

        # Handle shooting with cooldown
        if shoot and self.shoot_cooldown <= 0 and len(self.bullets) < MAX_BULLETS:
            self.bullets.append(make_sim_bullet(self.ship))
            self.shoot_cooldown = 15  # reference frames
            shot_fired = True

        # Capture heading at fire time for aim reward.
        fire_fwd_x = -sin(self.ship.theta * pi / 180)
        fire_fwd_y = -cos(self.ship.theta * pi / 180)

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

            # Bullet-rock collisions
            hit_rocks = set()
            hit_bullets = set()
            for bi, b in enumerate(self.bullets):
                for ri, r in enumerate(self.rocks):
                    if ri not in hit_rocks:
                        if torus_dist(b.x, b.y, r.x, r.y) < r.radius + 5:
                            hit_rocks.add(ri)
                            hit_bullets.add(bi)
                            reward += 5.0
                            break

            killed = [self.rocks[ri] for ri in hit_rocks]
            if hit_rocks:
                self.rocks = [r for i, r in enumerate(self.rocks) if i not in hit_rocks]
                self.bullets = [b for i, b in enumerate(self.bullets) if i not in hit_bullets]

            # Ship-rock collision checked BEFORE spawning children: a close-range kill
            # cannot produce a child rock that immediately overlaps the ship.
            for r in self.rocks:
                if torus_dist(self.ship.x, self.ship.y, r.x, r.y) < 0.7 * (r.radius + SHIP_RADIUS):
                    self.alive = False
                    self.steps += 1
                    obs = build_observation(self.ship, self.rocks, self.bullets, self.shoot_cooldown)
                    return obs, reward - 15.0, True

            for r in killed:
                self._spawn_children(r)

        self.steps += 1
        reward += 0.05  # survival: reward every step the ship stays alive

        # Wave cleared
        if len(self.rocks) == 0:
            self.cleared_wave_this_ep = True
            reward += 5.0 * self.wave_rocks
            dist_to_center = torus_dist(
                self.ship.x, self.ship.y, winWidth / 2, winHeight / 2
            )
            reward += 5.0 * max(0.0, 1.0 - dist_to_center / 300.0)
            self.wave_rocks += 1
            for _ in range(self.wave_rocks):
                self._spawn_big_rock()

        # On-fire bonus: evaluated against the single most CPA-dangerous rock.
        # Threshold is physics-based: cos(arcsin(radius/dist)) — the minimum alignment
        # for the bullet to physically intersect the rock collision circle.
        if shot_fired:
            if self.rocks:
                ship = self.ship
                def _primary_key(r):
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
                primary = min(self.rocks, key=_primary_key)
                dx_to = wrap_delta(ship.x, primary.x, winWidth)
                dy_to = wrap_delta(ship.y, primary.y, winHeight)
                dist_to = sqrt(dx_to * dx_to + dy_to * dy_to)
                if dist_to > primary.radius:
                    t_flight = dist_to / (5.0 * SIM_DT)
                    lead_x = dx_to + (primary.dx - ship.dx) * SIM_DT * t_flight
                    lead_y = dy_to + (primary.dy - ship.dy) * SIM_DT * t_flight
                    lead_dist = sqrt(lead_x * lead_x + lead_y * lead_y)
                    if lead_dist > 1e-6:
                        fire_dot = fire_fwd_x * lead_x / lead_dist + fire_fwd_y * lead_y / lead_dist
                        cos_hit = sqrt(max(0.0, 1.0 - (primary.radius / dist_to) ** 2))
                        if fire_dot > cos_hit:
                            reward += (fire_dot - cos_hit) / max(1e-6, 1.0 - cos_hit) * 2.5

        # Per-step CPA danger penalty: dense signal mirrors MCTS static_eval.
        # Provides gradient on every step, not just at death.
        if self.rocks:
            reward -= self._cpa_danger() * 0.25

        reward -= abs(self.ship.d_theta) * 0.004 * SIM_STEPS_PER_ACTION

        done = self.steps >= MAX_EPISODE_STEPS
        obs = build_observation(self.ship, self.rocks, self.bullets, self.shoot_cooldown)
        self.score += reward
        return obs, reward, done


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class DuelingQNetwork(nn.Module):
    """Dueling DQN: separate value and advantage streams."""

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=512):
        super().__init__()
        half = hidden // 2
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, half),
            nn.ReLU(),
            nn.Linear(half, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, half),
            nn.ReLU(),
            nn.Linear(half, n_actions),
        )

    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each parent is the sum of its children.
    Enables O(log n) proportional-priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self._update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update(self, idx, priority):
        self._update(idx, priority)

    def get(self, s):
        """Find the leaf whose cumulative priority covers value s."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority ** self.alpha,
                      (state, action, reward, next_state, done))

    def sample(self, batch_size, beta):
        indices = []
        priorities = []
        batch = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, prio, data = self.tree.get(s)
            if data is None:
                # Edge case: sample again from this segment
                s = random.uniform(0, self.tree.total())
                idx, prio, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(prio)
            batch.append(data)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Importance-sampling weights
        priorities = np.array(priorities, dtype=np.float64)
        probs = priorities / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + PER_EPSILON) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, abs(td) + PER_EPSILON)

    def __len__(self):
        return self.tree.size


class NStepBuffer:
    """Accumulates transitions and emits n-step returns."""

    def __init__(self, n=N_STEPS, gamma=GAMMA):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if done:
            return self._flush()
        if len(self.buffer) >= self.n:
            t = self._nstep_from_front()
            self.buffer.popleft()
            return [t]
        return []

    def _nstep_from_front(self):
        """Compute n-step return from buffer[0]."""
        state, action = self.buffer[0][0], self.buffer[0][1]
        R = 0.0
        for i, (_, _, r, ns, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:
                return (state, action, R, ns, True)
        last = self.buffer[-1]
        return (state, action, R, last[3], False)

    def _flush(self):
        """Flush all remaining transitions at episode end."""
        transitions = []
        while self.buffer:
            transitions.append(self._nstep_from_front())
            self.buffer.popleft()
        return transitions


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.q_net = DuelingQNetwork(state_dim, n_actions, hidden=512).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, n_actions, hidden=512).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay = PrioritizedReplayBuffer()
        self.nstep = NStepBuffer()
        self.total_steps = 0       # gradient steps (for target net sync)
        self.env_steps = 0         # environment steps (for schedules)
        self._gamma_n = GAMMA ** N_STEPS

    def epsilon(self):
        frac = min(1.0, self.env_steps / EPS_DECAY)
        return EPS_START + (EPS_END - EPS_START) * frac

    def beta(self):
        frac = min(1.0, self.env_steps / PER_BETA_FRAMES)
        return PER_BETA_START + (PER_BETA_END - PER_BETA_START) * frac

    def act(self, state, eval_mode=False, eps_override=None):
        if not eval_mode:
            eps = eps_override if eps_override is not None else self.epsilon()
            if random.random() < eps:
                return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_net(t)
            return q_vals.argmax(dim=1).item()

    def push_transition(self, state, action, reward, next_state, done):
        """Push raw reward through n-step buffer into replay."""
        transitions = self.nstep.push(state, action, reward, next_state, done)
        for t in transitions:
            self.replay.push(*t)
        self.env_steps += 1

    def train_step(self):
        """Perform GRAD_STEPS_PER_ENV gradient updates."""
        if len(self.replay) < BATCH_SIZE:
            return None

        last_loss = None
        for _ in range(GRAD_STEPS_PER_ENV):
            last_loss = self._one_gradient_step()
        return last_loss

    def _one_gradient_step(self):
        beta = self.beta()
        states, actions, rewards, next_states, dones, indices, is_weights = \
            self.replay.sample(BATCH_SIZE, beta)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self._gamma_n * next_q * (1.0 - dones_t)

        td_errors = (q_values - target).detach()
        loss = (weights_t * nn.functional.smooth_l1_loss(q_values, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors.cpu().numpy())

        # Polyak soft update of target network
        with torch.no_grad():
            for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - TARGET_TAU).add_(op.data, alpha=TARGET_TAU)

        self.total_steps += 1
        return loss.item()

    def save(self, path="dqn_mlp_model.pt", cur_rocks=1):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "env_steps": self.env_steps,
            "cur_rocks": cur_rocks,
        }, path)

    def load(self, path="dqn_mlp_model.pt"):
        """Load checkpoint. Returns cur_rocks on success, 0 if incompatible."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        try:
            self.q_net.load_state_dict(ckpt["q_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.total_steps = ckpt["total_steps"]
            self.env_steps = ckpt["env_steps"]
            return ckpt.get("cur_rocks", 1)
        except RuntimeError as e:
            print(f"Checkpoint incompatible with current architecture: {e}")
            print(f"Expected STATE_DIM={STATE_DIM}, N_ACTIONS={N_ACTIONS}.")
            print("Delete old .pt files and run 'train' to build a fresh model.")
            return 0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_episodes=50000, save_every=100, model_path="dqn_mlp_model.pt", clear_buffer=False):
    agent = DQNAgent()
    print(f"Device: {agent.device}")

    # Resume from checkpoint if it exists
    cur_rocks = 1
    if os.path.exists(model_path):
        print(f"Resuming from {model_path}")
        restored = agent.load(model_path)
        if restored == 0:
            print("Checkpoint incompatible — starting fresh. Delete .pt files to suppress.")
        else:
            cur_rocks = restored
            print(f"  env_steps={agent.env_steps}  eps={agent.epsilon():.3f}  cur_rocks={cur_rocks}")
            if clear_buffer:
                agent.replay = PrioritizedReplayBuffer(REPLAY_SIZE)
                print("  Replay buffer cleared.")

    env = AsteroidsEnv(num_rocks=cur_rocks)
    best_reward = -float("inf")
    reward_history = deque(maxlen=100)
    length_history = deque(maxlen=100)
    wave_clear_history = deque(maxlen=100)

    # Curriculum: promote when avg100 exceeds this threshold.
    # With kill=+1, wave_bonus=5*wave_rocks, death=-30:
    #   clearing wave 1 (1 big rock, 7 kills + bonus 5) = +12
    #   dying in wave 2 after a few kills: roughly -14
    # avg100 = 0 means kills roughly offset the death penalty — agent reliably
    # clears most of the first wave. Sufficient bar to practice 2-rock scenarios.
    CURRICULUM_THRESHOLD = 30.0  # requires ~50%+ wave clear rate at each level
    CURRICULUM_MIN_EPS = 1000    # don't promote before this many episodes at current level

    eps_at_level = 0
    level_steps = 0

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            eps = EPS_END + (EPS_START - EPS_END) * max(0.0, 1.0 - level_steps / EPS_LEVEL_DECAY)
            action = agent.act(state, eps_override=eps)
            next_state, reward, done = env.step(action)
            agent.push_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            state = next_state
            ep_reward += reward
            ep_steps += 1
            level_steps += 1

        reward_history.append(ep_reward)
        length_history.append(ep_steps)
        wave_clear_history.append(env.cleared_wave_this_ep)
        avg = np.mean(reward_history)
        avg_len = np.mean(length_history)
        eps_at_level += 1

        if ep % 10 == 0:
            cur_eps = EPS_END + (EPS_START - EPS_END) * max(0.0, 1.0 - level_steps / EPS_LEVEL_DECAY)
            print(
                f"Ep {ep:5d} | R {ep_reward:7.1f} | avg100 {avg:6.1f}/{CURRICULUM_THRESHOLD:.0f} | "
                f"len {ep_steps:4d} | avglen {avg_len:5.0f} | "
                f"eps {cur_eps:.3f} | rocks {cur_rocks} | "
                f"clears {sum(wave_clear_history):3d} | buf {len(agent.replay)}"
            )

        # Curriculum: promote only when agent has genuinely mastered the current level.
        # avg >= 30 requires ~50%+ wave clear rate (lucky-kill episodes score ~2.5).
        # avglen >= 400 is the key evasion gate: an agent dying at 150 steps has not
        # learned to evade regardless of avg reward.
        # clears >= 20 requires a 20% clear rate in the last 100 episodes.
        if (cur_rocks < NUM_ROCKS
                and eps_at_level >= CURRICULUM_MIN_EPS
                and len(reward_history) >= 100
                and avg > CURRICULUM_THRESHOLD
                and avg_len >= 400
                and sum(wave_clear_history) >= 20):
            cur_rocks += 1
            env.num_rocks = cur_rocks
            reward_history.clear()
            length_history.clear()
            wave_clear_history.clear()
            eps_at_level = 0
            level_steps = 0
            best_reward = -float("inf")
            agent.replay = PrioritizedReplayBuffer()  # discard stale easy-level data
            print(f"=== CURRICULUM: now training with {cur_rocks} rocks ===")

        if ep % save_every == 0:
            agent.save(model_path, cur_rocks=cur_rocks)
            if avg > best_reward:
                best_reward = avg
                agent.save(model_path.replace(".pt", "_best.pt"), cur_rocks=cur_rocks)
                print(f"  -> new best avg reward: {best_reward:.1f}")
            elif (best_reward > 5.0
                    and len(reward_history) >= 100
                    and avg < 0.5 * best_reward):
                print(f"  !! DEGRADATION: avg {avg:.1f} << best {best_reward:.1f}")

    agent.save(model_path, cur_rocks=cur_rocks)
    print("Training complete.")


# ---------------------------------------------------------------------------
# Watch mode: load trained model and play with pygame rendering
# ---------------------------------------------------------------------------

def watch(model_path="dqn_mlp_model.pt"):
    import pygame
    from pygame.locals import QUIT, KEYUP, K_q
    import asteroids as _astro

    if not os.path.exists(model_path):
        print(f"No model found at '{model_path}'. Run 'python dqn_asteroids.py train' first.")
        sys.exit(1)
    agent = DQNAgent()
    if not agent.load(model_path):
        sys.exit(1)
    agent.q_net.eval()

    pygame.init()
    fpsClock = pygame.time.Clock()
    _astro.fpsClock = fpsClock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroids - DQN Agent")
    pygame.mouse.set_visible(0)

    Ship      = _astro.Ship
    Bullet    = _astro.Bullet
    BigRock   = _astro.BigRock
    Score     = _astro.Score
    textBlit  = _astro.textBlit

    ship = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks = pygame.sprite.Group()
    num_rocks = NUM_ROCKS
    Score.reset()

    while len(rocks) < num_rocks:
        BigRock(screen, rocks)

    shoot_cooldown = 0

    while True:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYUP and event.key == K_q:
                pygame.quit()
                sys.exit()

        # Build observation from live game state
        s, rs, bs = build_sim_state(ship, rocks, bullets)
        obs = build_observation(s, rs, bs, shoot_cooldown)
        action_idx = agent.act(obs, eval_mode=True)
        thrust, left, right, shoot = ACTIONS[action_idx]

        if shoot_cooldown > 0:
            shoot_cooldown -= dt
        if shoot and shoot_cooldown <= 0 and len(bullets) < MAX_BULLETS:
            shoot = True
            shoot_cooldown = 15
        else:
            shoot = False

        screen.fill(WHITE)
        ship.update(thrust, left, right, dt)

        if shoot:
            Bullet(screen, ship, bullets)

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
            shoot_cooldown = 0

        Score.draw(screen, rocks)
        textBlit(screen, "DQN Agent", "Arial", 30, BLUE,
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
        description="DQN agent for Asteroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser(
        "train",
        help="Train the DQN agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_p.add_argument("-episodes", type=int, help="number of training episodes", default=20000)
    train_p.add_argument("-save-every", type=int, help="save checkpoint every N episodes", default=100)
    train_p.add_argument("-model", help="checkpoint file path", default="dqn_mlp_model.pt")
    train_p.add_argument("--clear-buffer", action="store_true",
                         help="discard replay buffer on resume (use after reward function changes)")

    watch_p = sub.add_parser(
        "watch",
        help="Watch a trained agent play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    watch_p.add_argument("-model", help="checkpoint file to load", default="dqn_mlp_model.pt")

    args = parser.parse_args()

    if args.command == "train":
        train(num_episodes=args.episodes, save_every=args.save_every, model_path=args.model,
              clear_buffer=args.clear_buffer)
    elif args.command == "watch":
        watch(model_path=args.model)
    else:
        parser.print_help()
