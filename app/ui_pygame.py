from __future__ import annotations
import os
import datetime
import pygame as pg
import numpy as np
from dataclasses import dataclass, field
from .integrators import INTEGRATORS
from .model import accelerations, total_energy, handle_collisions, _NUMBA
from .io_scenes import Scene, Body, save_scene, load_scene

_SCENES_DIR = os.path.join(os.path.dirname(__file__), "..", "scenes")


def _warmup_numba() -> None:
    """Первый вызов numba-функций вызывает JIT-компиляцию. Делаем это заранее."""
    if not _NUMBA:
        return
    dummy_pos  = np.zeros((2, 2), dtype=np.float64)
    dummy_vel  = np.zeros((2, 2), dtype=np.float64)
    dummy_mass = np.ones(2, dtype=np.float64)
    dummy_rad  = np.ones(2, dtype=np.float64) * 0.1
    dummy_pos[0] = [0.0, 0.0]
    dummy_pos[1] = [1.0, 0.0]
    accelerations(dummy_pos, dummy_mass, 1.0, 0.01)
    handle_collisions(dummy_pos, dummy_vel, dummy_mass, dummy_rad)

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
        self._scene_browser_open = False
        self._scene_browser_scroll = 0
        self._scene_browser_selected = 0

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
            f"N={len(self.pos)}  int={self.state.integrator}  dt={self.state.dt:.4f}  eps={self.state.eps:.3f}  numba={'on' if _NUMBA else 'off'}",
            f"mass_new={self.state.new_mass:.3f}  v_scale={self.state.v_scale:.2f}  running={self.state.running}",
            f"Energy≈{E:.5f}   zoom={self.state.zoom:.0f}   trails={self.state.trails}   collisions={self.state.collisions}   [O] сцены",
        ]
        y=8
        for line in hud:
            surf = self.font.render(line, True, (200,220,255))
            self.screen.blit(surf, (8,y)); y+=18

        # 6. Scene browser overlay
        if self._scene_browser_open:
            self._draw_scene_browser()

        # 7. Обновление экрана
        pg.display.flip()

    def _scene_list(self) -> list[str]:
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(_SCENES_DIR)
            if f.endswith(".json") and not f.startswith(".")
        )

    def _draw_scene_browser(self) -> None:
        scenes = self._scene_list()
        W, H = self.screen.get_size()
        PW, PH = 340, min(60 + len(scenes) * 24 + 40, H - 60)
        px = W - PW - 16
        py = 40

        # Полупрозрачный фон
        surf = pg.Surface((PW, PH), pg.SRCALPHA)
        surf.fill((10, 14, 30, 220))
        self.screen.blit(surf, (px, py))
        pg.draw.rect(self.screen, (60, 100, 180), (px, py, PW, PH), 1, border_radius=6)

        font_b = pg.font.SysFont("consolas", 16, bold=True)
        font_s = pg.font.SysFont("consolas", 15)
        COLOR_TEXT  = (200, 220, 255)
        COLOR_HINT  = (100, 120, 160)
        COLOR_SEL   = (40, 70, 130)
        COLOR_HOV   = (30, 50, 100)

        self.screen.blit(font_b.render("[ O ] Выбор сцены  (Enter — загрузить)", True, COLOR_TEXT), (px+10, py+8))
        self.screen.blit(font_s.render("↑ ↓ — навигация", True, COLOR_HINT), (px+10, py+28))

        MAX_VIS = (PH - 60) // 24
        scroll  = max(0, min(self._scene_browser_scroll, len(scenes) - MAX_VIS))
        self._scene_browser_scroll = scroll

        mx, my = pg.mouse.get_pos()
        for k, name in enumerate(scenes[scroll: scroll + MAX_VIS]):
            idx = scroll + k
            ry  = py + 52 + k * 24
            is_sel = (idx == self._scene_browser_selected)
            is_hov = (px <= mx <= px + PW and ry <= my <= ry + 22)
            if is_sel:
                pg.draw.rect(self.screen, COLOR_SEL, (px+4, ry, PW-8, 22), border_radius=3)
            elif is_hov:
                pg.draw.rect(self.screen, COLOR_HOV, (px+4, ry, PW-8, 22), border_radius=3)
            self.screen.blit(font_s.render(name, True, COLOR_TEXT), (px+12, ry+3))

        if len(scenes) > MAX_VIS:
            self.screen.blit(font_s.render(
                f"{scroll+1}–{scroll+min(MAX_VIS,len(scenes))}/{len(scenes)}",
                True, COLOR_HINT), (px+10, py+PH-22))

    # --- Сцены ---
    def clear(self):
        self.pos = np.zeros((0,2)); self.vel = np.zeros((0,2)); self.mass = np.zeros((0,))
        self.radius = np.zeros((0,), dtype=float)
        self.trails_points.clear()

    def add_body(self, world_pos, world_vel, m):
        if not self.state.running:
            self.pos = np.vstack([self.pos, world_pos[None,:]])
            self.vel = np.vstack([self.vel, world_vel[None,:]])
            self.mass = np.hstack([self.mass, np.array([m])])
            r_new = max(0.01, (m ** 0.3) / 100)
            self.radius = np.hstack([self.radius, [r_new]])

    def save_current(self, path="current.json"):
        from .io_scenes import Body, Scene
        bodies = [Body(x=float(p[0]), y=float(p[1]), vx=float(v[0]), vy=float(v[1]), m=float(m))
                  for p,v,m in zip(self.pos, self.vel, self.mass)]
        save_scene(path, Scene(G=self.state.G, eps=self.state.eps, dt=self.state.dt, bodies=bodies))

    def load_from(self, path="scenes/two_body.json"):
        try:
            sc = load_scene(path)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[load_from] ошибка загрузки '{path}': {exc}")
            return
        self.state.G, self.state.eps, self.state.dt = sc.G, sc.eps, sc.dt
        was_running = self.state.running
        self.state.running = False
        self.clear()
        for b in sc.bodies:
            self.add_body(np.array([b.x, b.y]), np.array([b.vx, b.vy]), b.m)
        self.state.running = was_running


    # --- Стартовый экран ---
    def _start_screen(self) -> tuple[str, str]:
        """Pygame-экран выбора сцены. Возвращает (scene_name, save_name)."""
        available = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(_SCENES_DIR)
            if f.endswith(".json") and not f.startswith(".")
        )

        font_big = pg.font.SysFont("consolas", 20, bold=True)
        font_sm  = pg.font.SysFont("consolas", 16)

        fields = {"scene": "", "name": "current"}
        active = "scene"
        COLOR_ACTIVE   = (100, 180, 255)
        COLOR_INACTIVE = (80, 100, 130)
        COLOR_BG       = (10, 10, 18)
        COLOR_TEXT     = (200, 220, 255)
        COLOR_HINT     = (100, 120, 160)
        COLOR_SEL      = (40, 60, 100)

        scroll_offset = 0
        MAX_VISIBLE = 10
        clock = pg.time.Clock()

        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    pg.quit()
                    raise SystemExit

                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_TAB:
                        active = "name" if active == "scene" else "scene"
                    elif e.key == pg.K_RETURN:
                        return fields["scene"].strip(), fields["name"].strip() or "current"
                    elif e.key == pg.K_BACKSPACE:
                        fields[active] = fields[active][:-1]
                    else:
                        if e.unicode.isprintable():
                            fields[active] += e.unicode

                elif e.type == pg.MOUSEBUTTONDOWN:
                    mx, my = e.pos
                    # клик по полю Scene
                    if 110 <= my <= 140:
                        active = "scene"
                    # клик по полю Save name
                    elif 200 <= my <= 230:
                        active = "name"
                    # клик по списку сцен
                    elif 270 <= my <= 270 + MAX_VISIBLE * 24:
                        idx = (my - 270) // 24 + scroll_offset
                        if 0 <= idx < len(available):
                            fields["scene"] = available[idx]
                            active = "name"

                    # скролл колесом
                    if e.button == 4:
                        scroll_offset = max(0, scroll_offset - 1)
                    elif e.button == 5:
                        scroll_offset = min(max(0, len(available) - MAX_VISIBLE), scroll_offset + 1)

            self.screen.fill(COLOR_BG)

            # Заголовок
            self.screen.blit(font_big.render("N-Body Simulator — выбор сцены", True, COLOR_TEXT), (30, 30))
            self.screen.blit(font_sm.render("Tab — переключить поле   Enter — запустить   ЛКМ — выбрать сцену из списка", True, COLOR_HINT), (30, 60))

            # Поле: Scene to load
            sc_col = COLOR_ACTIVE if active == "scene" else COLOR_INACTIVE
            pg.draw.rect(self.screen, sc_col, (30, 108, 500, 32), 2, border_radius=4)
            self.screen.blit(font_sm.render("Сцена для загрузки (опционально):", True, COLOR_HINT), (30, 90))
            self.screen.blit(font_sm.render(fields["scene"] or " ", True, COLOR_TEXT), (38, 116))

            # Поле: Save name
            sn_col = COLOR_ACTIVE if active == "name" else COLOR_INACTIVE
            pg.draw.rect(self.screen, sn_col, (30, 198, 500, 32), 2, border_radius=4)
            self.screen.blit(font_sm.render("Имя файла сохранения:", True, COLOR_HINT), (30, 180))
            self.screen.blit(font_sm.render(fields["name"] or " ", True, COLOR_TEXT), (38, 206))

            # Список доступных сцен
            self.screen.blit(font_sm.render("Доступные сцены:", True, COLOR_HINT), (30, 250))
            visible = available[scroll_offset: scroll_offset + MAX_VISIBLE]
            for k, name in enumerate(visible):
                y = 270 + k * 24
                if name == fields["scene"]:
                    pg.draw.rect(self.screen, COLOR_SEL, (30, y, 400, 22), border_radius=3)
                self.screen.blit(font_sm.render(name, True, COLOR_TEXT), (36, y + 2))

            if len(available) > MAX_VISIBLE:
                self.screen.blit(font_sm.render(
                    f"↑↓ колесо мыши  ({scroll_offset+1}–{scroll_offset+len(visible)}/{len(available)})",
                    True, COLOR_HINT), (30, 270 + MAX_VISIBLE * 24 + 4))

            pg.display.flip()
            clock.tick(30)

    # --- Цикл приложения ---
    def run(self, scene: str = "", save_name: str = "current"):
        _warmup_numba()

        # Если аргументы не переданы через CLI — показываем стартовый экран
        if not scene and save_name == "current":
            scene, save_name = self._start_screen()

        config = scene

        if config:
            self.load_from(f"scenes/{config}.json")

        running_app = True
        while running_app:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running_app = False

                elif e.type == pg.MOUSEBUTTONDOWN:
                    if self._scene_browser_open:
                        scenes = self._scene_list()
                        W, H = self.screen.get_size()
                        PW = 340
                        px = W - PW - 16
                        py = 40
                        PH = min(60 + len(scenes)*24 + 40, H - 60)
                        MAX_VIS = (PH - 60) // 24
                        mx, my = e.pos
                        # клик по строке списка
                        if e.button == 1 and px <= mx <= px+PW:
                            k2 = (my - py - 52) // 24
                            if 0 <= k2 < MAX_VIS:
                                idx = self._scene_browser_scroll + k2
                                if 0 <= idx < len(scenes):
                                    self._scene_browser_selected = idx
                                    name = scenes[idx]
                                    self.load_from(f"scenes/{name}.json")
                                    config = name
                                    self._scene_browser_open = False
                        # скролл колесом внутри панели
                        elif e.button == 4:
                            self._scene_browser_scroll = max(0, self._scene_browser_scroll - 1)
                        elif e.button == 5:
                            self._scene_browser_scroll += 1
                    elif e.button == 1:
                        self.drag_start = self.to_world(e.pos)

                elif e.type == pg.MOUSEBUTTONUP and e.button == 1 and self.drag_start is not None:
                    if not self._scene_browser_open:
                        end = self.to_world(e.pos)
                        v = (end - self.drag_start) * self.state.v_scale
                        self.add_body(self.drag_start, v, self.state.new_mass)
                    self.drag_start = None

                elif e.type == pg.KEYDOWN:
                    k = e.key
                    # ── навигация в браузере сцен ──
                    if self._scene_browser_open:
                        scenes = self._scene_list()
                        if k == pg.K_ESCAPE or k == pg.K_o:
                            self._scene_browser_open = False
                        elif k == pg.K_UP:
                            self._scene_browser_selected = max(0, self._scene_browser_selected - 1)
                            self._scene_browser_scroll = min(self._scene_browser_scroll,
                                                             self._scene_browser_selected)
                        elif k == pg.K_DOWN:
                            self._scene_browser_selected = min(len(scenes)-1,
                                                               self._scene_browser_selected + 1)
                            W, H = self.screen.get_size()
                            PH = min(60 + len(scenes)*24+40, H-60)
                            MAX_VIS = (PH-60)//24
                            if self._scene_browser_selected >= self._scene_browser_scroll + MAX_VIS:
                                self._scene_browser_scroll += 1
                        elif k == pg.K_RETURN:
                            if 0 <= self._scene_browser_selected < len(scenes):
                                name = scenes[self._scene_browser_selected]
                                self.load_from(f"scenes/{name}.json")
                                config = name
                                self._scene_browser_open = False
                        continue  # не обрабатываем остальные клавиши пока открыт браузер
                    # ── обычное управление ──
                    if k == pg.K_o:
                        self._scene_browser_open = True
                        self._scene_browser_scroll = 0
                    elif k == pg.K_SPACE: self.state.running = not self.state.running
                    elif k == pg.K_c: self.clear()
                    elif k == pg.K_q:
                        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                        base = config.strip() or "unnamed"
                        self.save_current(f"scenes/{base}_{ts}.json")
                    elif k == pg.K_l:
                        if config.strip():
                            self.load_from(f'scenes/{config.strip()}.json')
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