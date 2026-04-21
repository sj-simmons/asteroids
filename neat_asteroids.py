# neat_asteroids.py
#
# NEAT (NeuroEvolution of Augmenting Topologies) AI for Asteroids.
#
# Trains a neural network via NEAT to play the asteroids game.
#
# Usage:
#   python neat_asteroids.py              # train from scratch
#   python neat_asteroids.py --play best_genome.pkl   # watch best genome play
#
import pygame, sys, pickle, argparse, os
from pygame.locals import *
from math import sin, cos, pi, sqrt, atan2
from random import randint, seed

import neat
import asteroids as _astro

Space    = _astro.Space
Ship     = _astro.Ship
Bullet   = _astro.Bullet
BigRock  = _astro.BigRock
Score    = _astro.Score
Fader    = _astro.Fader
textBlit = _astro.textBlit
NUM_ROCKS = _astro.NUM_ROCKS
WIDTH = _astro.WIDTH
HEIGHT = _astro.HEIGHT
winWidth = _astro.winWidth
winHeight = _astro.winHeight
FPS = _astro.FPS
REFERENCE_FPS = _astro.REFERENCE_FPS
WHITE = _astro.WHITE
RED = _astro.RED
BLUE = _astro.BLUE

# ---------------------------------------------------------------------------
# Toroidal geometry helpers
# ---------------------------------------------------------------------------

def wrap_dist(a, b, mod):
    d = abs(a - b)
    return min(d, mod - d)


def wrap_delta(a, b, mod):
    """Signed shortest delta from a to b on a wrapped axis."""
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
# Lightweight simulation state (no pygame rendering needed during training)
# ---------------------------------------------------------------------------
# The real game runs at 60 FPS with dt = tick_ms * REFERENCE_FPS / 1000.
# At a steady 60 FPS that gives dt ≈ 150/60 = 2.5.  Every game object's
# movement, acceleration, and rotation are scaled by dt.  The simulation
# must use the same dt so the agent experiences the same physics it will
# encounter during playback.
# ---------------------------------------------------------------------------

SIM_DT = REFERENCE_FPS / FPS  # 150 / 60 = 2.5


class SimShip:
    def __init__(self, x, y, dx, dy, theta, accel=0.02):
        self.x = x % winWidth
        self.y = y % winHeight
        self.dx = dx
        self.dy = dy
        self.theta = theta
        self.accel = accel
        self.d_theta = 0

    def step(self, thrust, turn, dt=SIM_DT):
        """
        turn: proportional value in [-1, 1].
              +1 = full left (d_theta = +1.5), -1 = full right (d_theta = -1.5).
        """
        self.theta_dx = -sin(self.theta * pi / 180)
        self.theta_dy = -cos(self.theta * pi / 180)
        self.d_theta = turn * 1.5       # proportional turn rate
        if thrust:
            self.dx += self.accel * dt * self.theta_dx
            self.dy += self.accel * dt * self.theta_dy
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

    def step(self, dt=SIM_DT):
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight
        self.dist_left -= self.speed * dt


def make_sim_bullet(ship):
    speed = 5
    # Match real Bullet.__init__ (asteroids.py:377-378):
    #   dx = speed * theta_dx + d_theta * theta_dy + ship.dx
    #   dy = speed * theta_dy - d_theta * theta_dx + ship.dy
    # When spinning, d_theta deflects bullets sideways.
    bdx = speed * ship.theta_dx + ship.d_theta * ship.theta_dy + ship.dx
    bdy = speed * ship.theta_dy - ship.d_theta * ship.theta_dx + ship.dy
    return SimBullet(ship.x, ship.y, bdx, bdy, 6 * winHeight / 7)


def spawn_rocks(num):
    """Create initial big rocks matching BigRock spawn logic."""
    rocks = []
    for _ in range(num):
        slow = 0.2
        if randint(0, 1):
            x = randint(-int(winWidth / 20), int(winWidth / 20))
            y = randint(0, winHeight)
        else:
            x = randint(0, winWidth)
            y = randint(-int(winWidth / 20), int(winHeight / 20))
        dx = dy = 0
        while dx == 0 and dy == 0:
            dx = randint(-3, 3)
            dy = randint(-3, 3)
        dx *= slow
        dy *= slow
        rocks.append(SimRock(x, y, dx, dy, radius=50))
    return rocks


def destroy_rock(rock):
    """Return child rocks when a rock is destroyed."""
    children = []
    if rock.radius == 50:  # big -> 2 medium
        for _ in range(2):
            slow = 0.2
            dx = dy = 0
            while dx == 0 and dy == 0:
                dx = randint(-4, 4)
                dy = randint(-4, 4)
            children.append(SimRock(rock.x, rock.y, slow * dx, slow * dy, radius=30))
    elif rock.radius == 30:  # medium -> 2 small
        for _ in range(2):
            slow = 0.2
            dx = dy = 0
            while dx == 0 and dy == 0:
                dx = randint(-6, 6)
                dy = randint(-6, 6)
            children.append(SimRock(rock.x, rock.y, slow * dx, slow * dy, radius=15))
    # small rocks just vanish
    return children


SCORE_BY_RADIUS = {50: 5, 30: 10, 15: 20}
SHIP_RADIUS = 20  # approximate collision radius for the ship
BULLET_RADIUS = 5


def collides(x1, y1, r1, x2, y2, r2):
    return torus_dist(x1, y1, x2, y2) < (r1 + r2)


# ---------------------------------------------------------------------------
# Headless game simulation
# ---------------------------------------------------------------------------

MAX_TICKS = 1500  # max ticks per evaluation (each tick = SIM_DT ≈ 2.5 game-time units)
SHOOT_COOLDOWN = 2  # min ticks between shots
MAX_BULLETS = 1   # one bullet at a time — every miss is costly, forces aimed shooting
TICKS_PER_LEVEL = 300  # hard time limit per level — game over if exceeded
GRACE_TICKS = 60  # ticks between levels to reposition


def dist_from_center(ship):
    """Normalized distance from screen center (0 = center, 1 = corner)."""
    dx = ship.x - winWidth / 2
    dy = ship.y - winHeight / 2
    return sqrt(dx * dx + dy * dy) / sqrt((winWidth / 2) ** 2 + (winHeight / 2) ** 2)


def simulate_game(net, num_rocks=NUM_ROCKS):
    """Run one game with the given NEAT neural network. Returns fitness."""
    # Start with a small random velocity and heading so the agent can't
    # exploit a static starting position — it must learn to manage motion.
    init_speed = 0.3
    init_theta = randint(0, 359)
    init_dx = init_speed * -sin(init_theta * pi / 180)
    init_dy = init_speed * -cos(init_theta * pi / 180)
    ship = SimShip(winWidth / 2, winHeight / 2, init_dx, init_dy, init_theta)
    rocks = spawn_rocks(num_rocks)
    bullets = []
    shoot_timer = 0
    rocks_destroyed = 0
    shots_fired = 0
    level = num_rocks
    fitness = 0.0
    level_tick = 0  # ticks spent on current level
    grace_tick = 0  # countdown for between-level grace period

    for tick in range(MAX_TICKS):
        if not rocks and grace_tick == 0:
            # Level cleared — reward with speed bonus
            speed_bonus = max(0, TICKS_PER_LEVEL - level_tick) / TICKS_PER_LEVEL
            fitness += 1000 + 1000 * speed_bonus
            level += 1
            grace_tick = GRACE_TICKS  # start grace period before next level
            bullets = []

        # Grace period — agent can reposition, no rocks yet
        if grace_tick > 0:
            inputs = build_inputs(ship, rocks, bullets)
            output = net.activate(inputs)
            thrust = output[0] > 0.5
            turn  = output[1]           # proportional: -1 = full right, +1 = full left
            ship.step(thrust, turn)
            grace_tick -= 1
            # Per-tick gradient: reward moving toward center and bleeding off speed.
            _gcx = wrap_delta(ship.x, winWidth / 2, winWidth)
            _gcy = wrap_delta(ship.y, winHeight / 2, winHeight)
            _gcd_abs = sqrt(_gcx * _gcx + _gcy * _gcy)
            _gcd = _gcd_abs / DIAG
            _gsp = sqrt(ship.dx ** 2 + ship.dy ** 2)
            fitness += 3.0 * max(0.0, 1.0 - _gcd / 0.4)   # proximity
            fitness += 3.0 * max(0.0, 1.0 - _gsp / 0.4)   # low speed
            # Reward velocity aimed directly at centre
            if _gcd > 0.05 and _gsp > 0.1:
                _g_toward = (_gcx * ship.dx + _gcy * ship.dy) / (_gcd_abs * _gsp)
                fitness += 2.0 * max(0.0, _g_toward)
            if grace_tick == 0:
                # Lump-sum bonus for arriving centred and nearly stopped.
                _gcxf = wrap_delta(ship.x, winWidth / 2, winWidth)
                _gcyf = wrap_delta(ship.y, winHeight / 2, winHeight)
                center_d = sqrt(_gcxf * _gcxf + _gcyf * _gcyf) / DIAG
                speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
                fitness += 800 * max(0, 1.0 - center_d / 0.35)  # tight target
                fitness += 600 * max(0, 1.0 - speed / 1.0)
                rocks = spawn_rocks(level)
                level_tick = 0
            continue

        level_tick += 1

        # Hard time limit — too slow means death
        if level_tick > TICKS_PER_LEVEL:
            break

        # --- Build neural network inputs ---
        inputs = build_inputs(ship, rocks, bullets, shoot_timer)

        # --- Query the network ---
        output = net.activate(inputs)
        thrust = output[0] > 0.5
        turn  = output[1]               # proportional: -1 = full right, +1 = full left
        shoot = output[2] > 0.5

        # --- Update ship ---
        ship.step(thrust, turn)

        # Penalise continuous spinning — nudges the agent toward deliberate aim.
        # Kept small (0.08) so random genomes still earn net-positive fitness;
        # the bullet cap (MAX_BULLETS=1) is the primary anti-spray pressure.
        fitness -= 0.08 * abs(turn)

        speed_now = sqrt(ship.dx ** 2 + ship.dy ** 2)
        threat_score_now = inputs[12]

        # Speed penalty: quadratic above 0.8 so every increment of speed above
        # the target is progressively more costly.  At speed 2.0 the penalty is
        # 0.60*(1.2)^2 = 0.86/tick; at speed 3.0 it is 0.60*(2.2)^2 = 2.90/tick.
        # This must dominate the toward-centre reward at high speed.
        if speed_now > 0.8:
            _excess = speed_now - 0.8
            fitness -= 0.60 * _excess * _excess

        # Always-on centre-proximity gradient: reward being near the centre at
        # every tick regardless of threat level.  Uses a tight 0.4-DIAG radius
        # so only genuinely central positions score well.
        _cx = wrap_delta(ship.x, winWidth / 2, winWidth)
        _cy = wrap_delta(ship.y, winHeight / 2, winHeight)
        _cd = sqrt(_cx * _cx + _cy * _cy)
        _centre_frac = _cd / DIAG
        fitness += 0.50 * max(0.0, 1.0 - _centre_frac / 0.4)

        # Three-state positioning reward:
        #   threatened           → small reward peaking at speed 0.8
        #   safe + off-centre    → reward velocity aimed toward centre, but only
        #                          at low speed (fades to zero at speed 1.5 so the
        #                          ship cannot earn rewards by thrusting indefinitely)
        #   safe + at centre     → reward holding still
        if threat_score_now > 0.2:
            fitness += 0.08 * min(1.0, speed_now / 0.8)
        else:
            if _centre_frac > 0.05 and speed_now > 0.1:
                toward_dot = (_cx * ship.dx + _cy * ship.dy) / (_cd * speed_now)
                # Speed taper: reward halves by speed 0.75, gone by speed 1.5.
                # The ship has no incentive to accelerate beyond 1.5 to reach centre.
                speed_taper = max(0.0, 1.0 - speed_now / 1.5)
                fitness += 0.60 * speed_taper * max(0.0, toward_dot)
            else:
                # Already at centre: peaked reward for near-zero speed.
                fitness += 0.25 * max(0.0, 1.0 - speed_now / 0.4)

        # End-of-level repositioning: graded centre-pull that grows as rocks
        # dwindle.  With 1 rock left the centre-proximity reward dominates.
        #   3 rocks → weight 0.3   2 rocks → weight 0.6   1 rock → weight 1.0
        if 0 < len(rocks) <= 3:
            _lr_w = 1.0 if len(rocks) == 1 else (0.6 if len(rocks) == 2 else 0.3)
            _ld = _cd  # reuse already-computed distance from centre
            if _ld / DIAG > 0.05 and speed_now > 0.1:
                _toward_lr = (_cx * ship.dx + _cy * ship.dy) / (_ld * speed_now)
                _lr_taper = max(0.0, 1.0 - speed_now / 1.5)
                fitness += _lr_w * 1.50 * _lr_taper * max(0.0, _toward_lr)
            fitness += _lr_w * 4.00 * max(0.0, 1.0 - (_ld / DIAG) / 0.35)
            fitness += _lr_w * 0.50 * max(0.0, 1.0 - speed_now / 0.5)

        # Rock-count slow reward: fewer rocks = less threat = more reason to slow down.
        if rocks:
            calm = max(0.0, 1.0 - len(rocks) / 6.0)
            if calm > 0:
                fitness += 0.20 * calm * max(0.0, 1.0 - speed_now / 0.5)

        # Retrograde burn: much stronger than before so braking genuinely competes
        # with any remaining forward-thrust incentives at high speed.
        if thrust and speed_now > 0.1:
            retro_dot = -(ship.theta_dx * (ship.dx / speed_now) +
                          ship.theta_dy * (ship.dy / speed_now))
            if retro_dot > 0:
                fitness += 2.5 * retro_dot * min(1.0, speed_now / MAX_SPEED_NORM)

        # Predicted clearance + center preference:
        # Project ship and rocks 25 ticks ahead; reward being clear of future rock
        # positions, weighted by how close the future position is to screen centre.
        # This steers the agent toward the safe central home-base rather than any
        # random open edge of the screen.
        min_pred_dist = 0.0
        if rocks:
            fx = (ship.x + ship.dx * SIM_DT * 25) % winWidth
            fy = (ship.y + ship.dy * SIM_DT * 25) % winHeight
            min_pred_dist = min(
                torus_dist(fx, fy,
                           (r.x + r.dx * SIM_DT * 25) % winWidth,
                           (r.y + r.dy * SIM_DT * 25) % winHeight)
                for r in rocks
            )
            clearance_score = min(1.0, min_pred_dist / 200.0)
            fut_dx = fx - winWidth / 2
            fut_dy = fy - winHeight / 2
            fut_center_d = sqrt(fut_dx * fut_dx + fut_dy * fut_dy) / DIAG
            center_weight = 0.5 + 0.5 * max(0.0, 1.0 - fut_center_d * 2.5)
            fitness += 0.12 * clearance_score * center_weight

        # Global open-space reward: scan a 4×3 grid to find the point with the
        # greatest minimum distance to any rock (centre of the largest empty
        # circle).  Strongly reward the ship for being near that point — this
        # is the primary positional driver each tick.
        if rocks:
            _best_d = -1.0
            _best_x = winWidth / 2
            _best_y = winHeight / 2
            for _gi in range(4):
                for _gj in range(3):
                    _gx = (_gi + 0.5) * winWidth / 4
                    _gy = (_gj + 0.5) * winHeight / 3
                    _gd = min(torus_dist(_gx, _gy, r.x, r.y) for r in rocks)
                    if _gd > _best_d:
                        _best_d = _gd
                        _best_x = _gx
                        _best_y = _gy
            _ship_to_safe = torus_dist(ship.x, ship.y, _best_x, _best_y)
            # Normalise to DIAG*0.4 (~335 px): must be genuinely close to score.
            fitness += 0.35 * max(0.0, 1.0 - _ship_to_safe / (DIAG * 0.4))

        # Evasion reward: reward thrusting away from threat, but only while
        # speed is still reasonable — fades to zero above speed 2.5 so the
        # agent can't justify indefinite acceleration by pointing away from rocks.
        # The compulsory-thrust penalty is removed; death is punishment enough
        # for failing to evade, and the old −0.8×threat was forcing non-stop thrust.
        flee_dot_now = inputs[15]   # +1 = thrust direction faces away from threat
        if threat_score_now > 0.1 and thrust and flee_dot_now > 0:
            speed_factor = max(0.0, 1.0 - speed_now / 2.5)
            fitness += 0.8 * threat_score_now * flee_dot_now * speed_factor

        # --- Shooting (capped at MAX_BULLETS in flight) ---
        shoot_timer = max(0, shoot_timer - 1)
        if shoot and shoot_timer == 0 and len(bullets) < MAX_BULLETS:
            bullets.append(make_sim_bullet(ship))
            shoot_timer = SHOOT_COOLDOWN
            shots_fired += 1

        # --- Update bullets ---
        new_bullets = []
        killed_rocks = []
        for b in bullets:
            b.step()
            if b.dist_left > 0:
                hit = False
                for i, r in enumerate(rocks):
                    if collides(b.x, b.y, BULLET_RADIUS, r.x, r.y, r.radius):
                        rocks_destroyed += 1
                        killed_rocks.append(r)
                        rocks.pop(i)
                        hit = True
                        break
                if not hit:
                    new_bullets.append(b)
        bullets = new_bullets

        # --- Update rocks ---
        for r in rocks:
            r.step()

        # Ship-rock collision checked BEFORE spawning children: a close-range
        # kill cannot produce a child rock that immediately overlaps the ship.
        ship_hit = False
        for r in rocks:
            if torus_dist(ship.x, ship.y, r.x, r.y) < 0.7 * (SHIP_RADIUS + r.radius):
                ship_hit = True
                break

        if ship_hit:
            break

        for r in killed_rocks:
            rocks.extend(destroy_rock(r))

    # Per-rock reward gives a gradient even within a level
    fitness += rocks_destroyed * 100
    # Accuracy bonus — reward aimed shooting
    if shots_fired > 0:
        fitness += (rocks_destroyed / shots_fired) * 200
    return max(0.0, fitness)


# ---------------------------------------------------------------------------
# Neural network input construction
# ---------------------------------------------------------------------------
# Inputs (38 total):
#   0:  ship.x / winWidth
#   1:  ship.y / winHeight
#   2:  ship.dx (clamped, normalized)
#   3:  ship.dy (clamped, normalized)
#   4:  sin(heading)
#   5:  cos(heading)
#   6:  ship speed (normalized)
#   7:  intercept aim_angle, coarse (/ 180)
#   8:  intercept aim_angle, fine (/ 15, clamped ±1)
#   9:  d_theta normalized — current rotation rate
#  10:  rocks remaining (normalized)
#  11:  distance from screen center (normalized)
#  12:  threat_score — aggregate danger level [0, 1]
#  13:  threat_sin — x-component of net incoming threat direction
#  14:  threat_cos — y-component of net incoming threat direction
#  15:  flee_dot — dot(thrust_dir, -threat_dir): +1=thrust flees, -1=thrust toward threat
#  16:  bullets in flight / MAX_BULLETS (0=none, 1=max)
#  17:  shoot_timer / SHOOT_COOLDOWN (0=ready, 1=just fired)
# For 4 closest rocks (indices 18-37, 5 per rock):
#   rel_dx, rel_dy (normalized), distance (normalized), t_cpa (/60), cpa_dist (/200)
# Total: 18 + 20 = 38

NUM_CLOSEST_ROCKS = 4
MAX_SPEED_NORM = 5.0
HALF_W = winWidth / 2
HALF_H = winHeight / 2
DIAG = sqrt(HALF_W ** 2 + HALF_H ** 2)


def aim_angle(ship, tx, ty):
    """Signed angle (degrees) from ship heading to target, wrapped toroidally."""
    dx = wrap_delta(ship.x, tx, winWidth)
    dy = wrap_delta(ship.y, ty, winHeight)
    target_angle = atan2(-dx, -dy) * 180 / pi  # same convention as ship theta
    diff = (target_angle - ship.theta + 180) % 360 - 180
    return diff


def closing_speed(ship, rock, rel_dx, rel_dy, dist):
    """Rate at which a rock approaches the ship (positive = closing)."""
    if dist < 1e-6:
        return 0.0
    # Unit vector from ship to rock
    ux, uy = rel_dx / dist, rel_dy / dist
    # Relative velocity (rock minus ship) projected onto that unit vector
    # Negative projection = closing
    rvx = rock.dx - ship.dx
    rvy = rock.dy - ship.dy
    return -(rvx * ux + rvy * uy)


MAX_ROCKS_NORM = 21.0  # normalizer for rock count (level 3 big rocks = 21 total fragments)
BULLET_SPEED = 5.0    # raw bullet speed (pixels per unit time, before dt)


def intercept_aim_angle(ship, rock, rel_dx, rel_dy):
    """
    Angle from ship heading to the predicted intercept point where a bullet
    fired now will meet the rock, accounting for both rock and ship velocities.

    Uses the quadratic intercept formula:
        (vrel² - s²)t² + 2(p·vrel)t + |p|² = 0
    where p = rock position relative to ship, vrel = rock vel - ship vel,
    s = bullet speed. Returns the direct aim_angle as fallback if unsolvable.
    """
    # Relative velocity of rock with respect to ship
    vx = rock.dx - ship.dx
    vy = rock.dy - ship.dy

    px, py = rel_dx, rel_dy

    a = vx*vx + vy*vy - BULLET_SPEED*BULLET_SPEED
    b = 2.0 * (px*vx + py*vy)
    c = px*px + py*py

    t = None
    if abs(a) < 1e-9:
        # Linear: bullet exactly matches rock speed component
        if abs(b) > 1e-9:
            t_cand = -c / b
            if t_cand > 0:
                t = t_cand
    else:
        disc = b*b - 4.0*a*c
        if disc >= 0:
            sqrt_disc = sqrt(disc)
            for t_cand in ((-b - sqrt_disc) / (2*a), (-b + sqrt_disc) / (2*a)):
                if t_cand > 0:
                    if t is None or t_cand < t:
                        t = t_cand

    if t is not None:
        # Intercept point relative to ship
        ix = px + vx * t
        iy = py + vy * t
        target_angle = atan2(-ix, -iy) * 180 / pi
    else:
        # No intercept solution — fall back to aiming at current position
        target_angle = atan2(-px, -py) * 180 / pi

    return (target_angle - ship.theta + 180) % 360 - 180


def build_inputs(ship, rocks, bullets, shoot_timer=0):
    # Sort rocks by distance to ship
    rock_data = []
    for r in rocks:
        dist = torus_dist(ship.x, ship.y, r.x, r.y)
        rel_dx = wrap_delta(ship.x, r.x, winWidth)
        rel_dy = wrap_delta(ship.y, r.y, winHeight)
        cs = closing_speed(ship, r, rel_dx, rel_dy, dist)
        rock_data.append((dist, rel_dx, rel_dy, cs, r))

    rock_data.sort(key=lambda t: t[0])

    # Lead-aim angle to nearest rock — points at predicted intercept, not current position
    if rock_data:
        dist0, rel_dx0, rel_dy0, _, nearest = rock_data[0]
        raw_aim = intercept_aim_angle(ship, nearest, rel_dx0, rel_dy0)
    else:
        raw_aim = 0.0

    aim_coarse = raw_aim / 180.0                         # [-1, 1] full range
    aim_fine   = max(-1, min(1, raw_aim / 15.0))         # saturates at ±15°, fine near target

    # Current rotation rate: -1 = full right, 0 = none, +1 = full left
    d_theta_norm = max(-1.0, min(1.0, ship.d_theta / 1.5))

    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)

    # --- Threat aggregate ---
    # For each approaching rock (positive closing speed), accumulate a
    # weighted direction vector pointing from the ship toward that rock.
    # Weight = closing_speed / distance  (high = fast and close).
    # threat_score: total danger level (clamped 0-1).
    # threat_sin/cos: heading of the net incoming threat in world space,
    #   so the ship knows which way to flee.
    tx, ty = 0.0, 0.0
    total_weight = 0.0
    for dist, rel_dx, rel_dy, cs, _ in rock_data:
        if cs > 0 and dist > 1e-6:
            w = cs / dist
            tx += (rel_dx / dist) * w
            ty += (rel_dy / dist) * w
            total_weight += w
    threat_score = min(1.0, total_weight / 0.01)  # normalise: 0.01 calibrated to typical cs/dist
    threat_mag = sqrt(tx*tx + ty*ty)
    if threat_mag > 1e-6:
        threat_sin_val = tx / threat_mag      # x-component of unit threat direction
        threat_cos_val = ty / threat_mag      # y-component
    else:
        threat_sin_val = 0.0
        threat_cos_val = 0.0

    # Dot product of ship's thrust direction with the flee direction (opposite of threat).
    # +1 = already aimed away from threat (thrust now to flee),
    # -1 = aimed toward threat (turn around first).
    thrust_dx = -sin(ship.theta * pi / 180)
    thrust_dy = -cos(ship.theta * pi / 180)
    flee_dot = -(thrust_dx * threat_sin_val + thrust_dy * threat_cos_val)

    inputs = [
        ship.x / winWidth,
        ship.y / winHeight,
        max(-1, min(1, ship.dx / MAX_SPEED_NORM)),
        max(-1, min(1, ship.dy / MAX_SPEED_NORM)),
        sin(ship.theta * pi / 180),
        cos(ship.theta * pi / 180),
        min(1.0, speed / MAX_SPEED_NORM),
        aim_coarse,
        aim_fine,
        d_theta_norm,
        min(1.0, len(rocks) / MAX_ROCKS_NORM),  # rocks remaining
        dist_from_center(ship),                   # distance from center
        threat_score,                             # aggregate danger level [0,1]
        threat_sin_val,                           # x of net threat direction
        threat_cos_val,                           # y of net threat direction
        flee_dot,                                 # +1 = thrust flees threat, -1 = thrust toward it
        len(bullets) / MAX_BULLETS,               # bullets in flight
        shoot_timer / SHOOT_COOLDOWN,             # cooldown fraction (0=ready, 1=just fired)
    ]

    for i in range(NUM_CLOSEST_ROCKS):
        if i < len(rock_data):
            dist, rel_dx, rel_dy, cs, r = rock_data[i]
            inputs.append(rel_dx / HALF_W)
            inputs.append(rel_dy / HALF_H)
            inputs.append(dist / DIAG)
            # CPA features: time to closest approach and miss distance
            rel_vx = (r.dx - ship.dx) * SIM_DT
            rel_vy = (r.dy - ship.dy) * SIM_DT
            rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
            if rel_speed_sq > 0.001:
                t_cpa = -(rel_dx * rel_vx + rel_dy * rel_vy) / rel_speed_sq
                t_cpa = max(0.0, min(t_cpa, 60.0))
                cpa_dx = rel_dx + rel_vx * t_cpa
                cpa_dy = rel_dy + rel_vy * t_cpa
                cpa_dist = sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy) - 0.7 * (r.radius + SHIP_RADIUS)
            else:
                t_cpa = 60.0
                cpa_dist = dist - 0.7 * (r.radius + SHIP_RADIUS)
            inputs.append(t_cpa / 60.0)
            inputs.append(max(-1.0, min(1.0, cpa_dist / 200.0)))
        else:
            inputs.append(0.0)
            inputs.append(0.0)
            inputs.append(1.0)
            inputs.append(1.0)   # t_cpa: far future
            inputs.append(1.0)   # cpa_dist: far away

    assert len(inputs) == 38, f"Expected 38 inputs, got {len(inputs)}"
    return inputs


# ---------------------------------------------------------------------------
# NEAT evaluation function
# ---------------------------------------------------------------------------

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Average over a few trials to reduce variance from random rock spawns
        total = 0
        trials = 3
        for _ in range(trials):
            total += simulate_game(net)
        genome.fitness = total / trials


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path, generations=300, checkpoint_dir="neat_checkpoints"):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    os.makedirs(checkpoint_dir, exist_ok=True)
    pop.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(checkpoint_dir, "neat-checkpoint-"),
        )
    )

    winner = pop.run(eval_genomes, generations)

    # Save the best genome
    with open("best_genome.pkl", "wb") as f:
        pickle.dump((winner, config), f)

    print(f"\nBest genome fitness: {winner.fitness}")
    print(f"Saved to best_genome.pkl")

    return winner, config


# ---------------------------------------------------------------------------
# Playback — watch the best genome play with full rendering
# ---------------------------------------------------------------------------

def play(genome_path):
    with open(genome_path, "rb") as f:
        winner, config = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    pygame.init()
    fpsClock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroids — NEAT AI")
    pygame.mouse.set_visible(0)

    # Inject fpsClock into asteroids module (needed by Fader.use_a_life)
    _astro.fpsClock = fpsClock

    Score.reset()
    ship = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks = pygame.sprite.Group()

    num_rocks = NUM_ROCKS
    while len(rocks) < num_rocks:
        BigRock(screen, rocks)

    shoot_timer = 0
    fader = Fader(screen)

    running = True
    while running:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYUP and event.key == K_q:
                running = False

        # --- Build inputs from live game state ---
        sim_rocks = []
        for r in rocks:
            sim_rocks.append(SimRock(r.p.x, r.p.y, r.dx, r.dy, r.rect.width / 2))

        sim_ship = SimShip(ship.p.x, ship.p.y, ship.dx, ship.dy, ship._theta)

        sim_bullets = []
        for b in bullets:
            remaining = b.distance - b.distance_travelled
            sim_bullets.append(SimBullet(b.p.x, b.p.y, b.dx, b.dy, remaining))
        inputs = build_inputs(sim_ship, sim_rocks, sim_bullets, shoot_timer / SIM_DT)
        output = net.activate(inputs)

        thrust = output[0] > 0.5
        turn   = output[1]              # proportional: -1 = full right, +1 = full left
        shoot  = output[2] > 0.5

        # Convert proportional turn to the binary left/right the real Ship expects.
        # Thresholding at ±0.1 gives a small dead zone to prevent jitter.
        left  = turn >  0.1
        right = turn < -0.1

        screen.fill(WHITE)

        ship.update(thrust, left, right, dt)

        shoot_timer = max(0, shoot_timer - dt)
        if shoot and shoot_timer <= 0:
            Bullet(screen, ship, bullets)
            shoot_timer = SHOOT_COOLDOWN * dt

        if bullets:
            bullets.update(bullets, rocks, dt)
        rocks.update(dt)

        # Ship-rock collision
        if pygame.sprite.spritecollideany(
            ship, rocks, pygame.sprite.collide_circle_ratio(0.7)
        ):
            fader.use_a_life()
            fader.reset()
            if Score.getLives() == 0:
                fader.lose()
                num_rocks = NUM_ROCKS
                Score.reset()
            bullets.empty()
            rocks.empty()
            del ship
            pygame.event.clear()
            ship = Ship(screen)
            while len(rocks) < num_rocks:
                BigRock(screen, rocks)

        Score.draw(screen, rocks)

        # Level cleared
        if len(rocks) == 0:
            if fader.frames > 0:
                fader.lifeBonus(dt)
            else:
                fader.reset()
                bullets.empty()
                pygame.event.clear()
                num_rocks += 1
                while len(rocks) < num_rocks:
                    BigRock(screen, rocks)

        # Draw AI info
        _draw_ai_info(screen, output)

        pygame.display.update()

    pygame.quit()


def _draw_ai_info(screen, output):
    """Show what the AI is doing in the corner."""
    labels = ["Thrust", "Turn", "Shoot"]
    y = 60
    for i, label in enumerate(labels):
        active = output[i] > 0.5
        color = RED if active else (180, 180, 180)
        textBlit(screen, f"{label}: {output[i]:.2f}", "Arial", 20, color,
                 "topleft", winWidth - 160, y + i * 25)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEAT AI for Asteroids")
    parser.add_argument("--play", type=str, default=None,
                        help="Path to a saved genome .pkl to watch it play")
    parser.add_argument("--generations", type=int, default=300,
                        help="Number of generations to train (default: 300)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume training from a checkpoint file")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neat_config.ini")

    if args.play:
        # Need pygame initialized for rendering
        play(args.play)
    elif args.checkpoint:
        # Resume from checkpoint
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        pop = neat.Checkpointer.restore_checkpoint(args.checkpoint)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(
            neat.Checkpointer(
                generation_interval=10,
                filename_prefix=os.path.join("neat_checkpoints", "neat-checkpoint-"),
            )
        )
        winner = pop.run(eval_genomes, args.generations)
        with open("best_genome.pkl", "wb") as f:
            pickle.dump((winner, config), f)
        print(f"\nBest genome fitness: {winner.fitness}")
        print(f"Saved to best_genome.pkl")
    else:
        # Initialize pygame with a dummy video driver so no window appears
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.display.set_mode((1, 1))
        _astro.fpsClock = pygame.time.Clock()
        train(config_path, generations=args.generations)
        pygame.quit()
