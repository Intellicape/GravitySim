from __future__ import annotations
import pygame as pg
import numpy as np
from dataclasses import dataclass, field
from .integrators import INTEGRATORS
from .model import accelerations, total_energy, handle_collisions
from .io_scenes import Scene, Body, save_scene, load_scene

@dataclass
class SimState:
    G: float = 1.0
    eps: float = 0.005
    dt: float = 0.005
    integrator: str = "leapfrog"
    running: bool = False
    zoom: float = 200.0      # пикс/ед. длины
    offset = np.array([500.0, 350.0]) # центр экрана
    v_scale: float = 0.5
    new_mass: float = 10.0
    trails: bool = True
    collisions: bool = False

class NBodyUI:
    def __init__(self, width=1000, height=700):
        pg.init()
        self.screen = pg.display.set_mode((width, height))
        pg.display.set_caption("N-Body Simulator")
        self.font = pg.font.SysFont("consolas", 16)
        self.clock = pg.time.Clock()
        self.state = SimState()
        self.pos = np.zeros((0,2), dtype=float)
        self.vel = np.zeros((0,2), dtype=float)
        self.mass = np.zeros((0,), dtype=float)
        self.radius = np.zeros((0,), dtype=float)
        self.drag_start = None
        self.trails_points = []

    # --- координатные преобразования ---
    def to_screen(self, p):
        p = np.asarray(p)
        return (p * int(self.state.zoom) + self.state.offset).astype(int)
    def to_world(self, s):
        return (np.array(s, dtype=float) - self.state.offset) / self.state.zoom

    # --- Симуляция ---
    def step(self):
        if len(self.pos) == 0: return
        if self.state.collisions:
            handle_collisions(self.pos, self.vel, self.mass, self.radius)
        stepper = INTEGRATORS[self.state.integrator]
        self.pos, self.vel = stepper(self.pos, self.vel, self.mass,
                                     self.state.dt, self.state.G, self.state.eps)
        if self.state.trails:
            self.trails_points.append(self.pos.copy())
            if len(self.trails_points) > 400: self.trails_points.pop(0)

    def draw(self):

        # 1. Очистка экрана
        self.screen.fill((10, 10, 18))

        # 2. Рисование следов
        if self.state.trails and len(self.trails_points) > 0:
            for pts in self.trails_points:

                # гарантируем numpy-массив формы (N, 2)
                pts = np.asarray(pts, dtype=float)

                if pts.ndim != 2 or pts.shape[1] != 2:
                    continue # пропускаем некорректные данные

                # перевод в экранные координаты
                scr = pts * self.state.zoom + self.state.offset

                for i in range(scr.shape[0]):
                    x = int(scr[i, 0])
                    y = int(scr[i, 1])
                    pg.draw.circle(self.screen, (70, 120, 200), (x, y), 1)

        # 3. Рисование тел
        pos = np.asarray(self.pos, dtype=float)
        mass = np.asarray(self.mass, dtype=float)
        if pos.ndim == 2 and pos.shape[1] == 2:
            scr_pos = pos * self.state.zoom + self.state.offset
            for i in range(scr_pos.shape[0]):

                x = int(scr_pos[i, 0])
                y = int(scr_pos[i, 1])

                pg.draw.circle(self.screen, (255, 0, 0), (x, y), max(int(self.radius[i]*self.state.zoom), 1))

        # 4. Линия задания скорости
        if self.drag_start is not None:
            mpos = pg.mouse.get_pos()
            pg.draw.line(self.screen, (200,80,80),
                         self.to_screen(self.drag_start), mpos, 2)

        # 5. HUD
        E = total_energy(self.pos, self.vel, self.mass, self.state.G, self.state.eps) if len(self.pos)>1 else 0.0
        hud = [
            f"N={len(self.pos)}  int={self.state.integrator}  dt={self.state.dt:.4f}  eps={self.state.eps:.3f}",
            f"mass_new={self.state.new_mass:.3f}  v_scale={self.state.v_scale:.2f}  running={self.state.running}",
            f"Energy≈{E:.5f}   zoom={self.state.zoom:.0f}   trails={self.state.trails}   collisions={self.state.collisions}",
        ]
        y=8
        for line in hud:
            surf = self.font.render(line, True, (200,220,255))
            self.screen.blit(surf, (8,y)); y+=18

        # 6. Обновление экрана
        pg.display.flip()

    # --- Сцены ---
    def clear(self):
        self.pos = np.zeros((0,2)); self.vel = np.zeros((0,2)); self.mass = np.zeros((0,))
        self.trails_points.clear()

    def add_body(self, world_pos, world_vel, m):
        if not self.state.running:
            self.pos = np.vstack([self.pos, world_pos[None,:]])
            self.vel = np.vstack([self.vel, world_vel[None,:]])
            self.mass = np.hstack([self.mass, np.array([m])])
            self.radius = np.array([max(0.01, (self.mass[i] ** 0.3)/100) for i in range(len(self.pos))])

    def save_current(self, path="current.json"):
        from .io_scenes import Body, Scene
        bodies = [Body(x=float(p[0]), y=float(p[1]), vx=float(v[0]), vy=float(v[1]), m=float(m))
                  for p,v,m in zip(self.pos, self.vel, self.mass)]
        save_scene(path, Scene(G=self.state.G, eps=self.state.eps, dt=self.state.dt, bodies=bodies))

    def load_from(self, path="scenes/two_body.json"):
        sc = load_scene(path)
        self.state.G, self.state.eps, self.state.dt = sc.G, sc.eps, sc.dt
        self.clear()
        for b in sc.bodies:
            self.add_body(np.array([b.x,b.y]), np.array([b.vx,b.vy]), b.m)


    # --- Цикл приложения ---
    def run(self):

        print("-" * 100)
        print("Добро пожаловать в β-версию GravitySim!")
        config = input("Введите имя файла, из которого вы собираетесь загрузить сцену(опционально): ")
        curr_name = input("Введите имя текущего файла: ")
        print("-" * 100)

        running_app = True
        while running_app:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running_app = False

                elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                    self.drag_start = self.to_world(e.pos)

                elif e.type == pg.MOUSEBUTTONUP and e.button == 1 and self.drag_start is not None:
                    end = self.to_world(e.pos)
                    v = (end - self.drag_start) * self.state.v_scale
                    self.add_body(self.drag_start, v, self.state.new_mass)
                    self.drag_start = None

                elif e.type == pg.KEYDOWN:
                    k = e.key
                    if k == pg.K_SPACE: self.state.running = not self.state.running
                    elif k == pg.K_c: self.clear()
                    elif k == pg.K_q:
                        if config != '' and config != ' ':
                            self.save_current(f'scenes/{curr_name}.json')
                    elif k == pg.K_l:
                        if config != '' and config != ' ':
                            self.load_from(f'scenes/{config}.json')
                    elif k == pg.K_1: self.state.integrator = "euler"
                    elif k == pg.K_2: self.state.integrator = "leapfrog"
                    elif k == pg.K_3: self.state.integrator = "rk4"
                    elif k == pg.K_LEFTBRACKET: self.state.dt = max(1e-5, self.state.dt/1.25)
                    elif k == pg.K_RIGHTBRACKET: self.state.dt *= 1.25
                    elif k == pg.K_MINUS: self.state.eps = max(0.0, self.state.eps/1.25)
                    elif k == pg.K_EQUALS: self.state.eps *= 1.25
                    elif k == pg.K_COMMA: self.state.new_mass = max(1e-3, self.state.new_mass/1.25)
                    elif k == pg.K_PERIOD: self.state.new_mass *= 1.25
                    elif k == pg.K_v: self.state.v_scale = max(0.05, self.state.v_scale/1.25)
                    elif k == pg.K_b: self.state.v_scale *= 1.25
                    elif k == pg.K_t: self.state.trails = not self.state.trails
                    elif k == pg.K_p: self.state.collisions = not self.state.collisions
                    elif k == pg.K_z: self.state.zoom *= 1.1
                    elif k == pg.K_x: self.state.zoom /= 1.1
                    elif k == pg.K_w: self.state.offset[1] += 30
                    elif k == pg.K_s: self.state.offset[1] -= 30
                    elif k == pg.K_a: self.state.offset[0] += 30
                    elif k == pg.K_d: self.state.offset[0] -= 30

            if self.state.running:
                # несколько шагов за кадр для ускорения
                for _ in range(2):
                    self.step()

            self.clock.tick(60)
            self.draw()

        pg.quit()