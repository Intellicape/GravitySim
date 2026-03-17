# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Activate the virtual environment first
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# Run the simulator (GUI start screen appears if no args given)
python main.py

# Run with CLI arguments to skip the start screen
python main.py --scene two_body --name my_save
```

## Dependencies

```bash
pip install numpy>=1.26 pygame>=2.5
# numba and matplotlib are listed in requirements.txt but unused in current code
```

There are no tests, no linter config, and no build step — it's a pure runtime Python project.

## Architecture

The app is split into four modules under `app/`:

- **`model.py`** — pure NumPy physics: `accelerations()` (vectorised O(N²) gravity with softening `eps`), `total_energy()`, `handle_collisions()` (impulse-based elastic collisions). No UI imports.
- **`integrators.py`** — three steppers (`euler_step`, `leapfrog_step`, `rk4_step`) that each call `accelerations()`. Selected at runtime via `INTEGRATORS` dict.
- **`io_scenes.py`** — `Body` and `Scene` dataclasses + `save_scene`/`load_scene` for JSON persistence. Scene files live in `scenes/`.
- **`ui_pygame.py`** — `SimState` (simulation parameters) + `NBodyUI` (Pygame window). `_start_screen()` is the file-picker shown before the main loop. `run(scene, save_name)` is the main loop.

`main.py` parses CLI args via `argparse` and calls `NBodyUI().run(scene=..., save_name=...)`.

## Key design decisions

- **Leapfrog is the default integrator** — it conserves energy far better than Euler for orbital mechanics.
- **Softening parameter `eps`** prevents gravitational singularities at close range; default is `0.005` everywhere (both `SimState` and `load_scene` fallback).
- **Bodies can only be added while paused** (`running=False`). `add_body()` silently ignores calls when running. `load_from()` temporarily sets `running=False` to work around this.
- **Radius is derived from mass**: `r = max(0.01, m**0.3 / 100)` — not stored in scene files, recomputed on load.
- **Trails** are stored as a capped deque of position snapshots (`trails_points`, max 400 frames).

## Coordinate system

World coordinates are centred around `state.offset` (default screen centre). Conversion:
- world → screen: `pos * zoom + offset`
- screen → world: `(screen_pos - offset) / zoom`

## Scene files

JSON format: `{ "G", "eps", "dt", "bodies": [{"x","y","vx","vy","m"}, ...] }`. Saved under `scenes/`. The `radius` field is not persisted.
