# asteroids_ppo.py
#
# Adds a PPO AI mode to the asteroids game without modifying asteroids.py.
# Loads the best PPO checkpoint and optionally plays the game.
#
# Usage:
#   python asteroids_ppo.py                          # splash screen, then play or AI
#   python asteroids_ppo.py --model ppo_model.pt     # load a specific checkpoint
#
import pygame, sys, argparse, os, torch
from pygame.locals import *

import asteroids as _astro
from ppo_asteroids import (
    build_observation, build_sim_state,
    PPOActorCritic, ACTIONS, SIM_STEPS_PER_ACTION, MAX_BULLETS,
)

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

DEFAULT_MODEL = "ppo_model_best.pt"

# ---------------------------------------------------------------------------
# AI controller: observation → policy network forward pass → greedy action
# ---------------------------------------------------------------------------

class AIController:
    def __init__(self, model_path=DEFAULT_MODEL):
        if not os.path.exists(model_path):
            print(f"No PPO model found at '{model_path}'.")
            print("Run 'python ppo_asteroids.py train' to train one first.")
            sys.exit(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PPOActorCritic().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        try:
            self.net.load_state_dict(ckpt["net"])
        except RuntimeError as e:
            print(f"Checkpoint incompatible with current architecture: {e}")
            print("Delete old .pt files and run 'python ppo_asteroids.py train' to rebuild.")
            sys.exit(1)
        self.net.eval()
        self.shoot_cooldown     = 0.0
        self.frame_count        = 0
        self.current_action_idx = 0

    def reset(self):
        self.shoot_cooldown     = 0.0
        self.frame_count        = 0
        self.current_action_idx = 0

    def get_action(self, ship, rocks, bullets, dt=1):
        """Returns (thrust, left, right, fire).

        Recomputes the policy decision every SIM_STEPS_PER_ACTION frames,
        matching the temporal granularity the network was trained on.
        `fire` is True only on frames the controller decides to shoot
        (respects shoot cooldown and the bullet cap).
        """
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= dt

        if self.frame_count % SIM_STEPS_PER_ACTION == 0:
            s, rs, bs = build_sim_state(ship, rocks, bullets)
            obs = build_observation(s, rs, bs, max(0.0, self.shoot_cooldown))
            obs_t = torch.tensor(obs, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.net.forward(obs_t)
            self.current_action_idx = logits.argmax(dim=-1).item()

        self.frame_count += 1

        thrust, left, right, shoot = ACTIONS[self.current_action_idx]
        fire = False
        if shoot and self.shoot_cooldown <= 0 and len(bullets) < MAX_BULLETS:
            fire = True
            self.shoot_cooldown = 15.0

        return thrust, left, right, fire


# ---------------------------------------------------------------------------
# Modified main with AI mode option on splash screen
# ---------------------------------------------------------------------------

def main(model_path=DEFAULT_MODEL):
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
    ai = AIController(model_path=model_path) if ai_mode else None

    while True:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0

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
            textBlit(screen, "AI MODE (PPO)", "Arial", 30, BLUE,
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
    textBlit(screen, "or hit <a> for AI mode (PPO)", "Arial", 40, RED,
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
        textBlit(screen, "or hit <a> for AI mode (PPO)", "Arial", 40, RED,
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
    parser = argparse.ArgumentParser(description="Asteroids with optional PPO AI mode")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"PPO checkpoint to load (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    main(model_path=args.model)
