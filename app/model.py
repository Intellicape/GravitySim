from __future__ import annotations
import numpy as np

import os

_NUMBA = False  # включается явно через enable_numba() из main.py

def enable_numba() -> bool:
    """Пытается загрузить numba. Возвращает True если успешно."""
    global _NUMBA, accelerations, handle_collisions
    try:
        from numba import njit as _njit

        @_njit(cache=True)
        def _acc_nb(pos, mass, G, eps):
            N = pos.shape[0]
            a = np.zeros((N, 2))
            eps2 = eps * eps
            for i in range(N):
                ax = 0.0; ay = 0.0
                for j in range(N):
                    if i == j: continue
                    dx = pos[j,0] - pos[i,0]; dy = pos[j,1] - pos[i,1]
                    inv_r3 = (dx*dx + dy*dy + eps2) ** -1.5
                    ax += mass[j] * inv_r3 * dx; ay += mass[j] * inv_r3 * dy
                a[i,0] = G * ax; a[i,1] = G * ay
            return a

        @_njit(cache=True)
        def _col_nb(pos, vel, mass, radius):
            N = pos.shape[0]
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0]-pos[j,0]; dy = pos[i,1]-pos[j,1]
                    dist2 = dx*dx + dy*dy
                    if dist2 == 0.0: continue
                    rs = radius[i]+radius[j]
                    if dist2 > rs*rs: continue
                    dist = dist2**0.5
                    nx = dx/dist; ny = dy/dist
                    rvx = vel[i,0]-vel[j,0]; rvy = vel[i,1]-vel[j,1]
                    rn = rvx*nx + rvy*ny
                    if rn >= 0.0: continue
                    if mass[i]==0.0 or mass[j]==0.0: continue
                    J = 2.0*rn/(1.0/mass[i]+1.0/mass[j])
                    vel[i,0] -= J*nx/mass[i]; vel[i,1] -= J*ny/mass[i]
                    vel[j,0] += J*nx/mass[j]; vel[j,1] += J*ny/mass[j]

        # прогрев — если LLVM крашится, abort случится здесь до старта окна
        _p = np.array([[0.0,0.0],[1.0,0.0]]); _m = np.ones(2); _v = np.zeros((2,2)); _r = np.full(2, 0.1)
        _acc_nb(_p, _m, 1.0, 0.01)
        _col_nb(_p, _v, _m, _r)

        accelerations = _acc_nb
        handle_collisions = _col_nb
        _NUMBA = True
        return True
    except Exception as exc:
        print(f"[numba] недоступна: {exc}")
        return False

# ---------------------------------------------------------------------------
# Ускорения (NumPy — дефолт; заменяется на njit-версию через enable_numba())
# ---------------------------------------------------------------------------

def accelerations(pos: np.ndarray, mass: np.ndarray, G: float, eps: float) -> np.ndarray:
    """
    pos: (N,2), mass: (N,), return a: (N,2)
    a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
    """
    r = pos[:, None, :] - pos[None, :, :]
    dist2 = np.sum(r**2, axis=-1) + eps**2
    mask = np.eye(len(pos), dtype=bool)
    dist2[mask] = np.inf
    inv_r3 = -dist2**-1.5
    a = (r * (mass[None, :] * inv_r3)[:, :, None]).sum(axis=1)
    return G * a

# ---------------------------------------------------------------------------
# Полная энергия (только для HUD, не горячий путь — оставляем NumPy)
# ---------------------------------------------------------------------------

def total_energy(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, G: float, eps: float) -> float:
    ke = 0.5 * (mass[:, None] * vel**2).sum()
    r = pos[None, :, :] - pos[:, None, :]
    dist = np.sqrt((r**2).sum(axis=2) + eps**2)
    i, j = np.triu_indices(len(pos), k=1)
    pe = -G * np.sum((mass[i] * mass[j]) / dist[i, j])
    return ke + pe

# ---------------------------------------------------------------------------
# Столкновения (NumPy — дефолт; заменяется на njit-версию через enable_numba())
# ---------------------------------------------------------------------------

def handle_collisions(pos: np.ndarray, vel: np.ndarray,
                      mass: np.ndarray, radius: np.ndarray) -> None:
    diff = pos[:, None, :] - pos[None, :, :]
    dist2 = np.sum(diff**2, axis=-1)
    rad_sum = radius[:, None] + radius[None, :]
    collision_mask = (dist2 > 0) & (dist2 <= rad_sum**2)
    i_idx, j_idx = np.where(np.triu(collision_mask))
    for i, j in zip(i_idx, j_idx):
        r = pos[i] - pos[j]
        dist = np.linalg.norm(r)
        if dist == 0:
            continue
        n = r / dist
        rel_vel = vel[i] - vel[j]
        rel_normal = np.dot(rel_vel, n)
        if rel_normal >= 0:
            continue
        if mass[i] == 0 or mass[j] == 0:
            continue
        J = 2.0 * rel_normal / (1.0 / mass[i] + 1.0 / mass[j])
        impulse = J * n
        vel[i] -= impulse / mass[i]
        vel[j] += impulse / mass[j]
