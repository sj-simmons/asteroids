# asteroids_mcts.py
#
# Adds an AI mode (MCTS) to the asteroids game without modifying asteroids.py.
# Run normally for human play; press <a> on the splash screen to watch the
# MCTS agent play.
#
# Usage:
#   python asteroids_mcts.py                  # splash screen, then play or AI
#   python asteroids_mcts.py --budget-ms 15   # more search time per move
#
import pygame, sys, time, argparse, random
from pygame.locals import *
from math import sin, cos, pi, sqrt, log, atan2

import asteroids as _astro

Space         = _astro.Space
Ship          = _astro.Ship
Bullet        = _astro.Bullet
BigRock       = _astro.BigRock
MediumRock    = _astro.MediumRock
SmallRock     = _astro.SmallRock
Score         = _astro.Score
Fader         = _astro.Fader
Burst         = _astro.Burst
textBlit      = _astro.textBlit
NUM_ROCKS     = _astro.NUM_ROCKS
WIDTH         = _astro.WIDTH
HEIGHT        = _astro.HEIGHT
winWidth      = _astro.winWidth
winHeight     = _astro.winHeight
FPS           = _astro.FPS
REFERENCE_FPS = _astro.REFERENCE_FPS
WHITE         = _astro.WHITE
GREY          = _astro.GREY
BLACK         = _astro.BLACK
BLUE          = _astro.BLUE
RED           = _astro.RED

# ---------------------------------------------------------------------------
# Physics scaling
# ---------------------------------------------------------------------------

SIM_DT  = REFERENCE_FPS / FPS   # ~2.5: all step() calls use this
MAX_DT  = SIM_DT * 1.5          # cap dt so physics stays stable if search runs long
_DIAG   = sqrt((winWidth / 2) ** 2 + (winHeight / 2) ** 2)
_SIN8   = [sin(i * pi / 4) for i in range(8)]
_COS8   = [cos(i * pi / 4) for i in range(8)]
CPA_HORIZON = 60  # frames to look ahead for closest-point-of-approach

# ---------------------------------------------------------------------------
# Toroidal helpers
# ---------------------------------------------------------------------------

def wrap_delta(a, b, mod):
    """Signed shortest-path delta from a to b on a toroidal axis."""
    d = b - a
    if d > mod / 2:
        d -= mod
    elif d < -mod / 2:
        d += mod
    return d

def torus_dist(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dx = min(dx, winWidth  - dx)
    dy = abs(y1 - y2)
    dy = min(dy, winHeight - dy)
    return sqrt(dx * dx + dy * dy)

# ---------------------------------------------------------------------------
# Lightweight sim objects (no pygame sprites)
# ---------------------------------------------------------------------------

SHIP_RADIUS = 18


class SimShip:
    __slots__ = ('x', 'y', 'dx', 'dy', 'theta', 'accel', 'd_theta')

    def __init__(self, x, y, dx, dy, theta, accel=0.02):
        self.x      = x % winWidth
        self.y      = y % winHeight
        self.dx     = dx
        self.dy     = dy
        self.theta  = theta
        self.accel  = accel
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
    __slots__ = ('x', 'y', 'dx', 'dy', 'radius')

    def __init__(self, x, y, dx, dy, radius):
        self.x      = x % winWidth
        self.y      = y % winHeight
        self.dx     = dx
        self.dy     = dy
        self.radius = radius

    def copy(self):
        return SimRock(self.x, self.y, self.dx, self.dy, self.radius)

    def step(self, dt=SIM_DT):
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight


class SimBullet:
    __slots__ = ('x', 'y', 'dx', 'dy', 'dist_left', 'speed')

    def __init__(self, x, y, dx, dy, dist_left):
        self.x         = x % winWidth
        self.y         = y % winHeight
        self.dx        = dx
        self.dy        = dy
        self.dist_left = dist_left
        self.speed     = sqrt(dx * dx + dy * dy)

    def copy(self):
        return SimBullet(self.x, self.y, self.dx, self.dy, self.dist_left)

    def step(self, dt=SIM_DT):
        self.x         = (self.x + self.dx * dt) % winWidth
        self.y         = (self.y + self.dy * dt) % winHeight
        self.dist_left -= self.speed * dt


def make_sim_bullet(ship):
    speed = 5
    tdx = -sin(ship.theta * pi / 180)
    tdy = -cos(ship.theta * pi / 180)
    bdx = speed * tdx + ship.d_theta * tdy + ship.dx
    bdy = speed * tdy - ship.d_theta * tdx + ship.dy
    return SimBullet(ship.x, ship.y, bdx, bdy, 6 * winHeight / 7)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ACTIONS = [
    (False, False, False, False),  # 0  drift
    (True,  False, False, False),  # 1  thrust
    (False, True,  False, False),  # 2  left
    (False, False, True,  False),  # 3  right
    (True,  True,  False, False),  # 4  thrust+left
    (True,  False, True,  False),  # 5  thrust+right
    (False, False, False, True),   # 6  shoot
    (False, True,  False, True),   # 7  left+shoot
    (False, False, True,  True),   # 8  right+shoot
    (True,  False, False, True),   # 9  thrust+shoot
]

SIM_STEPS_PER_ACTION = 4

# ---------------------------------------------------------------------------
# Game state for MCTS forward simulation
# ---------------------------------------------------------------------------

class GameState:
    __slots__ = ('ship', 'rocks', 'bullets', 'shoot_cooldown', 'alive', 'rocks_killed')

    def __init__(self, ship, rocks, bullets, shoot_cooldown=0):
        self.ship           = ship
        self.rocks          = rocks
        self.bullets        = bullets
        self.shoot_cooldown = shoot_cooldown
        self.alive          = True
        self.rocks_killed   = 0

    def copy(self):
        gs = GameState(
            self.ship.copy(),
            [r.copy() for r in self.rocks],
            [b.copy() for b in self.bullets],
            self.shoot_cooldown,
        )
        gs.rocks_killed = self.rocks_killed
        return gs

    def step(self, action_idx):
        """Advance SIM_STEPS_PER_ACTION physics frames. Returns immediate reward."""
        thrust, left, right, shoot = ACTIONS[action_idx]
        reward = 0.0

        if shoot and self.shoot_cooldown <= 0 and len(self.bullets) < 6:
            self.bullets.append(make_sim_bullet(self.ship))
            self.shoot_cooldown = 15

        for _ in range(SIM_STEPS_PER_ACTION):
            self.ship.step(thrust, left, right)
            for r in self.rocks:
                r.step()

            # step() returns None; `or True` keeps the comprehension truthy
            self.bullets = [b for b in self.bullets
                            if (b.step() or True) and b.dist_left > 0]

            if self.shoot_cooldown > 0:
                self.shoot_cooldown -= SIM_DT

            # Bullet-rock collisions
            hit_rocks   = set()
            hit_bullets = set()
            for bi, b in enumerate(self.bullets):
                for ri, r in enumerate(self.rocks):
                    if ri not in hit_rocks:
                        if torus_dist(b.x, b.y, r.x, r.y) < r.radius + 5:
                            hit_rocks.add(ri)
                            hit_bullets.add(bi)
                            self.rocks_killed += 1
                            reward += 50.0
                            break

            if hit_rocks:
                new_rocks = []
                for ri in hit_rocks:
                    rock = self.rocks[ri]
                    if rock.radius >= 50:
                        cr, sr = 30, 4
                    elif rock.radius >= 30:
                        cr, sr = 15, 6
                    else:
                        continue
                    for _ in range(2):
                        ddx = ddy = 0
                        while ddx == 0 and ddy == 0:
                            ddx = random.randint(-sr, sr)
                            ddy = random.randint(-sr, sr)
                        new_rocks.append(
                            SimRock(rock.x, rock.y, ddx * 0.2, ddy * 0.2, cr))
                self.rocks   = [r for i, r in enumerate(self.rocks)
                                if i not in hit_rocks] + new_rocks
                self.bullets = [b for i, b in enumerate(self.bullets)
                                if i not in hit_bullets]

            # Ship-rock collision
            for r in self.rocks:
                if torus_dist(self.ship.x, self.ship.y, r.x, r.y) < r.radius * 0.7 + SHIP_RADIUS:
                    self.alive = False
                    return reward - 500.0

        return reward

# ---------------------------------------------------------------------------
# Static evaluation (replaces expensive rollouts)
# ---------------------------------------------------------------------------

def static_eval(state):
    """Evaluate a game state cheaply. Higher = better for the agent."""
    if not state.alive:
        return -1000.0

    ship  = state.ship
    score = 0.0

    score += state.rocks_killed * 50.0

    if not state.rocks:
        # Wave cleared: brake, drift to centre, arrive ready.
        cx = winWidth / 2
        cy = winHeight / 2
        center_dist   = sqrt((ship.x - cx) ** 2 + (ship.y - cy) ** 2)
        center_bonus  = 80.0 * max(0.0, 1.0 - center_dist / (min(winWidth, winHeight) / 2))
        speed         = sqrt(ship.dx ** 2 + ship.dy ** 2)
        speed_penalty = max(0.0, speed - 1.0) ** 2 * 8.0
        fwd_x = -sin(ship.theta * pi / 180)
        fwd_y = -cos(ship.theta * pi / 180)
        brake_bonus = 0.0
        if speed > 0.8:
            brake_dot   = -(fwd_x * ship.dx + fwd_y * ship.dy) / speed
            brake_bonus = max(0.0, brake_dot) * 8.0
        return score + 200.0 + center_bonus - speed_penalty + brake_bonus

    fwd_x = -sin(ship.theta * pi / 180)
    fwd_y = -cos(ship.theta * pi / 180)
    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)

    # -- Danger from current position and closest-point-of-approach --
    min_edge_dist = float('inf')
    total_danger  = 0.0
    rock_info     = []  # cache for reuse below

    for r in state.rocks:
        dx_to       = wrap_delta(ship.x, r.x, winWidth)
        dy_to       = wrap_delta(ship.y, r.y, winHeight)
        center_dist = sqrt(dx_to * dx_to + dy_to * dy_to)
        edge_dist   = center_dist - r.radius

        rel_vx  = (r.dx - ship.dx) * SIM_DT
        rel_vy  = (r.dy - ship.dy) * SIM_DT
        closing = (dx_to * rel_vx + dy_to * rel_vy) / max(1.0, center_dist)

        rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
        if rel_speed_sq > 0.001:
            t_cpa    = -(dx_to * rel_vx + dy_to * rel_vy) / rel_speed_sq
            t_cpa    = max(0.0, min(t_cpa, CPA_HORIZON))
            cpa_dx   = dx_to + rel_vx * t_cpa
            cpa_dy   = dy_to + rel_vy * t_cpa
            cpa_dist = sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy) - r.radius
        else:
            t_cpa    = CPA_HORIZON
            cpa_dist = edge_dist

        rock_info.append((r, dx_to, dy_to, center_dist, edge_dist, closing, t_cpa, cpa_dist))

        if edge_dist < min_edge_dist:
            min_edge_dist = edge_dist

        # Danger from current proximity
        if edge_dist < 250:
            proximity  = max(0.0, 1.0 - edge_dist / 250.0) ** 2
            speed_mult = 1.0 + max(0.0, -closing) * 0.8
            size_mult  = r.radius / 30.0
            total_danger += proximity * speed_mult * size_mult

        # CPA danger
        if cpa_dist < 200:
            size_mult = r.radius / 30.0
            if center_dist > 1:
                aim_dot = (fwd_x * dx_to + fwd_y * dy_to) / center_dist
            else:
                aim_dot = 1.0
            blindside_mult = 1.0 if aim_dot > 0.3 else (1.8 - aim_dot)
            urgency        = 1.0 + max(0.0, 1.0 - t_cpa / 20.0)
            cpa_penalty    = size_mult * blindside_mult * urgency / (cpa_dist + 8.0)
            total_danger  += cpa_penalty

    score -= total_danger * 80.0
    score += min(min_edge_dist, 300) * 0.08

    # -- Escape corridor scoring --
    corridor_dists    = []
    best_corridor_raw = 0.0
    for i in range(8):
        ray_dx, ray_dy = _SIN8[i], _COS8[i]
        min_clear = 400.0
        for r, dx_to, dy_to, cdist, edist, closing, t_cpa, cpa_dist in rock_info:
            t = dx_to * ray_dx + dy_to * ray_dy
            if t < 0:
                continue
            perp = abs(dx_to * ray_dy - dy_to * ray_dx)
            if perp < r.radius + SHIP_RADIUS:
                clear = max(0.0, t - r.radius - SHIP_RADIUS)
                if clear < min_clear:
                    min_clear = clear
        corridor_dists.append(min_clear)
        if min_clear > best_corridor_raw:
            best_corridor_raw = min_clear

    corridor_dists.sort()
    best_escape   = corridor_dists[-1]
    second_escape = corridor_dists[-2] if len(corridor_dists) > 1 else 0
    if best_escape < 100:
        score -= 60.0
    elif best_escape < 200:
        score -= 25.0 * (1.0 - best_escape / 200.0)
    score += min(second_escape, 200) * 0.05

    # -- Close-range kill danger --
    for b in state.bullets:
        if b.dist_left <= 0:
            continue
        for r, dx_to, dy_to, cdist, edist, closing, t_cpa, cpa_dist in rock_info:
            if r.radius < 30:
                continue
            bdx = wrap_delta(b.x, r.x, winWidth)
            bdy = wrap_delta(b.y, r.y, winHeight)
            bd  = sqrt(bdx * bdx + bdy * bdy)
            if bd < r.radius + 20 and edist < 130:
                score -= 40.0 * (1.0 - edist / 130.0)

    # -- Bullets in flight: CPA check for hits --
    for b in state.bullets:
        if b.dist_left <= 0:
            continue
        for r, dx_to_r, dy_to_r, cdist_r, edist_r, closing_r, t_cpa_r, cpa_dist_r in rock_info:
            bdx = wrap_delta(b.x, r.x, winWidth)
            bdy = wrap_delta(b.y, r.y, winHeight)
            brvx = (b.dx - r.dx) * SIM_DT
            brvy = (b.dy - r.dy) * SIM_DT
            br_speed_sq = brvx * brvx + brvy * brvy
            if br_speed_sq < 0.001:
                continue
            t_hit = (bdx * brvx + bdy * brvy) / br_speed_sq
            if t_hit < 0:
                continue
            miss_dist = sqrt((bdx - brvx * t_hit) ** 2 + (bdy - brvy * t_hit) ** 2)
            if miss_dist < r.radius + 8:
                safe_mult = 1.0
                if r.radius >= 30 and edist_r < 130:
                    safe_mult = 0.2
                score += 15.0 * safe_mult

    # -- Aim alignment with lead targeting --
    any_collision_course = any(cpa_dist < 120
                               for _, _, _, _, _, _, _, cpa_dist in rock_info)
    suppress_aim = any_collision_course
    if len(state.rocks) == 1:
        cx_end   = winWidth  / 2
        cy_end   = winHeight / 2
        end_dist = sqrt((ship.x - cx_end) ** 2 + (ship.y - cy_end) ** 2)
        if end_dist > 110 or speed > 0.3:
            suppress_aim = True

    bullet_speed = 5.0 * SIM_DT
    best_aim = 0.0
    if not suppress_aim:
        for r, dx_to, dy_to, cdist, edist, closing, t_cpa, cpa_dist in rock_info:
            if cdist < 1 or closing > 0:
                continue
            t_flight  = cdist / bullet_speed if bullet_speed > 0 else 0
            lead_x    = dx_to + (r.dx - ship.dx) * SIM_DT * t_flight
            lead_y    = dy_to + (r.dy - ship.dy) * SIM_DT * t_flight
            lead_dist = sqrt(lead_x * lead_x + lead_y * lead_y)
            if lead_dist < 1:
                continue
            dot = fwd_x * (lead_x / lead_dist) + fwd_y * (lead_y / lead_dist)
            if dot > 0.7:
                aim_val = dot * 6.0
                if edist < 350:
                    aim_val *= 1.3
                if r.radius < 30:
                    aim_val *= 1.5
                elif edist < 130:
                    aim_val *= 0.3
                if aim_val > best_aim:
                    best_aim = aim_val
    score += best_aim

    # -- Centre tendency (last two rocks only) --
    cx  = winWidth  / 2
    cy  = winHeight / 2
    cdx_c = ship.x - cx
    cdy_c = ship.y - cy
    center_dist_c = sqrt(cdx_c * cdx_c + cdy_c * cdy_c)
    max_r   = min(winWidth, winHeight) / 2
    n_rocks = len(state.rocks)
    if n_rocks == 2:
        center_weight = 12.0
    elif n_rocks == 1:
        center_weight = 26.0
    else:
        center_weight = 0.0
    if center_weight > 0.0:
        score += center_weight * max(0.0, 1.0 - center_dist_c / max_r)

    if n_rocks == 1:
        center_fraction = min(1.0, center_dist_c / max_r)
        score -= speed * speed * 18.0 * (0.15 + 0.85 * (1.0 - center_fraction))
        if center_dist_c > 30 and speed > 0.1:
            to_cx = -cdx_c / center_dist_c
            to_cy = -cdy_c / center_dist_c
            vel_dot_center = (ship.dx * to_cx + ship.dy * to_cy) / speed
            score += max(0.0, vel_dot_center) * center_fraction * 30.0

    # -- Speed management --
    if speed < 0.5:
        score += 15.0
    elif speed < 1.5:
        score += 8.0 * (1.5 - speed)

    if speed > 1.5:
        rock_count_mult = 1.0 + min(len(state.rocks), 8) * 0.3
        score -= (speed - 1.5) ** 2 * 4.0 * rock_count_mult

    if speed > 0.8:
        brake_dot = -(fwd_x * ship.dx + fwd_y * ship.dy) / speed
        score += max(0.0, brake_dot) * 8.0

    return score

# ---------------------------------------------------------------------------
# Action pruning
# ---------------------------------------------------------------------------

def prune_actions(state):
    """Return action indices worth considering in this state."""
    can_shoot = state.shoot_cooldown <= 0 and len(state.bullets) < 6
    ship = state.ship

    if not state.rocks:
        return [0]

    speed        = sqrt(ship.dx ** 2 + ship.dy ** 2)
    danger       = False
    shoot_threat = False

    for r in state.rocks:
        dx_to       = wrap_delta(ship.x, r.x, winWidth)
        dy_to       = wrap_delta(ship.y, r.y, winHeight)
        center_dist = sqrt(dx_to * dx_to + dy_to * dy_to)
        edge_dist   = center_dist - r.radius

        if edge_dist < 120:
            danger = True
        if edge_dist < 75:
            shoot_threat = True

        rel_vx       = (r.dx - ship.dx) * SIM_DT
        rel_vy       = (r.dy - ship.dy) * SIM_DT
        rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
        if rel_speed_sq > 0.001:
            t_cpa    = max(0.0, min(-(dx_to * rel_vx + dy_to * rel_vy) / rel_speed_sq, 60.0))
            cpa_dx   = dx_to + rel_vx * t_cpa
            cpa_dy   = dy_to + rel_vy * t_cpa
            cpa_dist = sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy) - r.radius
            if cpa_dist < 90:
                danger = True
            if cpa_dist < 55 and t_cpa < 30:
                shoot_threat = True

        if danger and shoot_threat:
            break

    if danger or speed > 2.0:
        actions = [0, 1, 2, 3, 4, 5]
    else:
        actions = [0, 2, 3]
        if speed > 1.0:
            actions.append(1)
        if len(state.rocks) == 1 and 1 not in actions:
            actions.append(1)

    # Last rock: block shooting until centred and nearly stopped.
    if len(state.rocks) == 1 and can_shoot:
        cx_end = winWidth  / 2
        cy_end = winHeight / 2
        dx_c   = ship.x - cx_end
        dy_c   = ship.y - cy_end
        if sqrt(dx_c * dx_c + dy_c * dy_c) > 110 or speed > 0.3:
            imminent = False
            r_last   = state.rocks[0]
            lx = wrap_delta(ship.x, r_last.x, winWidth)
            ly = wrap_delta(ship.y, r_last.y, winHeight)
            if sqrt(lx * lx + ly * ly) - r_last.radius < 55:
                imminent = True
            else:
                rvx = (r_last.dx - ship.dx) * SIM_DT
                rvy = (r_last.dy - ship.dy) * SIM_DT
                rsq = rvx * rvx + rvy * rvy
                if rsq > 0.001:
                    tc   = max(0.0, min(-(lx * rvx + ly * rvy) / rsq, 60.0))
                    cdx2 = lx + rvx * tc
                    cdy2 = ly + rvy * tc
                    if sqrt(cdx2 * cdx2 + cdy2 * cdy2) - r_last.radius < 45:
                        imminent = True
            if not imminent:
                can_shoot = False

    if shoot_threat:
        can_shoot = False

    if can_shoot:
        fwd_x = -sin(ship.theta * pi / 180)
        fwd_y = -cos(ship.theta * pi / 180)
        suppress_shoot = False
        for r in state.rocks:
            if r.radius < 30:
                continue
            dx_to = wrap_delta(ship.x, r.x, winWidth)
            dy_to = wrap_delta(ship.y, r.y, winHeight)
            d = sqrt(dx_to * dx_to + dy_to * dy_to)
            if d < 1:
                continue
            dot       = (fwd_x * dx_to + fwd_y * dy_to) / d
            edge_dist = d - r.radius
            if dot > 0.85 and edge_dist < 100:
                suppress_shoot = True
                break
        if not suppress_shoot:
            actions.append(6)
            actions.append(7)
            actions.append(8)
            if 1 in actions:
                actions.append(9)

    return actions

# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

UCB_C = 1.0


class MCTSNode:
    __slots__ = ('state', 'action', 'parent', 'children',
                 'visits', 'total_value', 'untried')

    def __init__(self, state, action=None, parent=None, available_actions=None):
        self.state       = state
        self.action      = action
        self.parent      = parent
        self.children    = []
        self.visits      = 0
        self.total_value = 0.0
        if available_actions is not None:
            self.untried = list(available_actions)
        else:
            self.untried = prune_actions(state)
        random.shuffle(self.untried)

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def is_terminal(self):
        return not self.state.alive

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return (self.total_value / self.visits
                + UCB_C * sqrt(log(self.parent.visits) / self.visits))

    def best_child_ucb(self):
        return max(self.children, key=lambda c: c.ucb1())

    def best_child_visits(self):
        return max(self.children, key=lambda c: c.visits)

    def expand(self):
        action      = self.untried.pop()
        child_state = self.state.copy()
        child_state.step(action)
        child = MCTSNode(child_state, action=action, parent=self)
        self.children.append(child)
        return child

    def backpropagate(self, value):
        node = self
        while node is not None:
            node.visits      += 1
            node.total_value += value
            node = node.parent


def mcts_search(root_state, budget_sec=0.010):
    """Time-budgeted MCTS. Returns best action index."""
    root     = MCTSNode(root_state.copy())
    deadline = time.monotonic() + budget_sec

    while time.monotonic() < deadline:
        node = root

        # Selection
        while node.is_fully_expanded() and node.children and not node.is_terminal():
            node = node.best_child_ucb()

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # Evaluation (static — no rollout)
        value = static_eval(node.state)

        # Backpropagation
        node.backpropagate(value)

    if not root.children:
        return _heuristic_action(root_state)

    return root.best_child_visits().action


def _heuristic_action(state):
    """Fallback used only when MCTS has zero iterations."""
    ship = state.ship
    if not state.rocks:
        return 0

    nearest = min(state.rocks,
                  key=lambda r: torus_dist(ship.x, ship.y, r.x, r.y))
    dist    = torus_dist(ship.x, ship.y, nearest.x, nearest.y)

    if dist < 100:
        dx_to  = wrap_delta(ship.x, nearest.x, winWidth)
        dy_to  = wrap_delta(ship.y, nearest.y, winHeight)
        escape = atan2(dx_to, dy_to) * 180 / pi
        diff   = (escape - ship.theta + 180) % 360 - 180
        if abs(diff) < 45:
            return 1
        return 4 if diff > 0 else 5

    dx_to = wrap_delta(ship.x, nearest.x, winWidth)
    dy_to = wrap_delta(ship.y, nearest.y, winHeight)
    aim   = atan2(-dx_to, -dy_to) * 180 / pi
    diff  = (aim - ship.theta + 180) % 360 - 180
    if abs(diff) < 10 and state.shoot_cooldown <= 0:
        return 6
    return 2 if diff > 0 else 3

# ---------------------------------------------------------------------------
# Build simulation state from live pygame sprites
# ---------------------------------------------------------------------------

def build_sim_state(ship, rocks, bullets, shoot_cooldown=0.0):
    s = SimShip(ship.p.x, ship.p.y, ship.dx, ship.dy, ship._theta)
    s.d_theta = ship.d_theta
    rs = []
    for r in rocks:
        radius = 50
        if isinstance(r, SmallRock):
            radius = 15
        elif isinstance(r, MediumRock):
            radius = 30
        rs.append(SimRock(r.p.x, r.p.y, r.dx, r.dy, radius))
    bs = []
    for b in bullets:
        remaining = b.distance - b.distance_travelled
        bs.append(SimBullet(b.p.x, b.p.y, b.dx, b.dy, remaining))
    return GameState(s, rs, bs, shoot_cooldown)

# ---------------------------------------------------------------------------
# Throttled AI controller
# ---------------------------------------------------------------------------

class AIController:
    def __init__(self, budget_ms=10):
        self.budget_sec         = budget_ms / 1000.0
        self.frame_count        = 0
        self.shoot_cooldown     = 0.0
        self.current_action_idx = 0

    def reset(self):
        self.frame_count        = 0
        self.shoot_cooldown     = 0.0
        self.current_action_idx = 0

    def get_action(self, ship, rocks, bullets, dt=1):
        """Returns (thrust, left, right, fire).

        `fire` is True only on the frame the controller decides to fire
        (respects shoot cooldown and the 6-bullet cap).
        """
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= dt

        if self.frame_count % SIM_STEPS_PER_ACTION == 0:
            gs = build_sim_state(ship, rocks, bullets,
                                 max(0.0, self.shoot_cooldown))
            self.current_action_idx = mcts_search(gs, self.budget_sec)

        self.frame_count += 1

        thrust, left, right, shoot = ACTIONS[self.current_action_idx]
        fire = False
        if shoot and self.shoot_cooldown <= 0 and len(bullets) < 6:
            fire = True
            self.shoot_cooldown = 15.0

        return thrust, left, right, fire

# ---------------------------------------------------------------------------
# Modified main with AI mode option on splash screen
# ---------------------------------------------------------------------------

def main(budget_ms=10):
    global fpsClock

    pygame.init()
    fpsClock        = pygame.time.Clock()
    _astro.fpsClock = fpsClock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Asteroids')
    pygame.mouse.set_visible(0)

    fader   = Fader(screen)
    ai_mode = start_up_with_ai(fader, screen)

    ship    = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks   = pygame.sprite.Group()
    burst   = Burst()
    pause   = False

    while len(rocks) < NUM_ROCKS:
        BigRock(screen, rocks)

    num_rocks = NUM_ROCKS
    ai = AIController(budget_ms=budget_ms) if ai_mode else None

    while True:
        raw_dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0
        # Cap dt so physics stays stable if MCTS search runs long on a frame
        dt = min(raw_dt, MAX_DT)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if not ai_mode:
                    if event.key == K_b:
                        burst.shoot = True
                    if event.key == K_SPACE:
                        Bullet(screen, ship, bullets)
            elif event.type == KEYUP:
                if event.key == K_q:
                    pygame.quit()
                    sys.exit()
                if event.key == K_p and not ai_mode:
                    pause = True

        if ai_mode:
            thrust, left, right, fire = ai.get_action(ship, rocks, bullets, dt)
            shoot = fire
        else:
            shoot  = False
            thrust = left = right = False
            keys = pygame.key.get_pressed()
            if keys[K_UP]:
                thrust = True
            if keys[K_LEFT]:
                left = True
            if keys[K_RIGHT]:
                right = True

        screen.fill(WHITE)
        ship.update(thrust, left, right, dt)

        if ai_mode:
            if shoot:
                Bullet(screen, ship, bullets)
        elif burst.update(dt):
            Bullet(screen, ship, bullets)

        if bullets:
            bullets.update(bullets, rocks, dt)
        rocks.update(dt)

        if pygame.sprite.spritecollideany(ship, rocks, pygame.sprite.collide_circle_ratio(.7)):
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
            if ai_mode:
                ai.reset()

        Score.draw(screen, rocks)

        if ai_mode:
            textBlit(screen, "AI MODE (MCTS)", "Arial", 30, BLUE,
                     "bottomleft", winWidth / 20, 18 * winHeight / 20, False)

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

        if pause:
            pause = fader.info(pause)

        pygame.display.update()


def start_up_with_ai(fader, screen):
    """Run the startup screen with an AI mode option. Returns True if AI mode selected."""
    global fpsClock

    screen.fill(WHITE)
    fader.info_blit(False, True)
    textBlit(screen, "or hit <a> for AI mode (MCTS)", "Arial", 40, RED,
             "center", winWidth / 2, 23 * winHeight / 24)
    infoSurf = screen.subsurface(pygame.Rect(0, 0, WIDTH, HEIGHT)).copy()

    ship    = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks   = pygame.sprite.Group()
    burst   = Burst()
    ai_mode = False

    run         = True
    starting_up = True

    def rebuild_info_surf():
        screen.fill(WHITE)
        fader.info_blit(False, True)
        textBlit(screen, "or hit <a> for AI mode (MCTS)", "Arial", 40, RED,
                 "center", winWidth / 2, 23 * winHeight / 24)
        return screen.subsurface(pygame.Rect(0, 0, WIDTH, HEIGHT)).copy()

    while run:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_q:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_b:
                    burst.shoot = True
                elif event.key == K_p:
                    print('ship is at', ship.p.x, ship.p.y)
            elif event.type == KEYUP:
                if event.key in (K_1, K_2, K_3, K_4):
                    space_map = {K_1: (1, 1), K_2: (1, -1), K_3: (-1, 1), K_4: (-1, -1)}
                    Space.set_space(*space_map[event.key])
                    if event.key != K_1:
                        infoSurf = rebuild_info_surf()
                    del ship
                    ship = Ship(screen)
                elif event.key == K_c:
                    run = False
                elif event.key == K_a:
                    ai_mode = True
                    run = False
                elif event.key == K_SPACE:
                    Bullet(screen, ship, bullets)

        thrust = left = right = False
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            thrust = True
        if keys[K_LEFT]:
            left = True
        if keys[K_RIGHT]:
            right = True
        if keys[K_DOWN]:
            ship.dx = ship.dy = 0

        screen.fill(WHITE)

        if starting_up and fader.frames > 0:
            fader.title_banner(dt)
        else:
            starting_up = False

        if not starting_up:
            screen.blit(infoSurf, (0, 0, winWidth, winHeight))

        ship.update(thrust, left, right, dt)

        if burst.update(dt):
            Bullet(screen, ship, bullets)

        bullets.update(bullets, rocks, dt)

        pygame.display.update()

    fader.reset()
    bullets.empty()
    rocks.empty()
    del ship

    return ai_mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Asteroids with optional MCTS AI mode")
    parser.add_argument("--budget-ms", type=int, default=10,
                        help="MCTS search budget per decision in ms (default: 10)")
    args = parser.parse_args()
    main(budget_ms=args.budget_ms)
