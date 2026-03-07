from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Body:
    x: float; y: float; vx: float; vy: float; m: float

@dataclass
class Scene:
    G: float = 1.0
    eps: float = 0.005
    dt: float = 0.005
    bodies: List[Body] = None

def save_scene(path, scene: Scene):
    data = asdict(scene)
    data["bodies"] = [asdict(b) for b in scene.bodies]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_scene(path) -> Scene:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bodies = [Body(**b) for b in data["bodies"]]
    return Scene(G=data.get("G",1.0), eps=data.get("eps",0.02),
                 dt=data.get("dt",0.005), bodies=bodies)
