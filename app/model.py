from __future__ import annotations
import numpy as np

def accelerations(pos: np.ndarray, mass: np.ndarray, G: float, eps: float) -> np.ndarray:
    """
    pos: (N,2), mass: (N,), return a: (N,2)
    a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
    """
    # r_ij = r_j - r_i
    r = pos[:, None, :] - pos[None, :, :]
    dist2 = np.sum(r**2, axis=-1) + eps**2
    mask = np.eye(len(pos), dtype=bool)
    dist2[mask] = np.inf  # исключаем j=i
    inv_r3 = -dist2**-1.5 # 1 / |r|^3 с мягчением
    # взвешиваем по m_j и суммируем по j
    a = (r * (mass[None, :] * inv_r3)[:,:,None]).sum(axis=1)
    return G * a

def total_energy(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, G: float, eps: float) -> float:
    ke = 0.5 * (mass[:, None] * vel**2).sum()
    # потенциал: -G * sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2)
    r = pos[None, :, :] - pos[:, None, :]
    dist = np.sqrt((r**2).sum(axis=2) + eps**2)
    i, j = np.triu_indices(len(pos), k=1)
    pe = -G * np.sum((mass[i] * mass[j]) / dist[i, j])
    return ke + pe

def handle_collisions(pos, vel, mass, radius):

    # Разности координат (N,N,2)
    diff = pos[:, None, :] - pos[None, :, :]

    # Квадраты расстояний (N,N)
    dist2 = np.sum(diff**2, axis=-1)

    # Суммы радиусов (N,N)
    rad_sum = radius[:, None] + radius[None, :]
    rad_sum2 = rad_sum**2

    # Маска столкновений
    collision_mask = (dist2 > 0) & (dist2 <= rad_sum2)

    # Берём только верхний треугольник (i < j)
    i_idx, j_idx = np.where(np.triu(collision_mask))

    for i, j in zip(i_idx, j_idx):

        r = pos[i] - pos[j]
        dist = np.linalg.norm(r)

        if dist == 0:
            continue

        n = r / dist

        rel_vel = vel[i] - vel[j]
        rel_normal = np.dot(rel_vel, n)

        # def sigma(x): return 1/(1+np.exp(-x))
        # e = sigma(0.1*(np.linalg.norm(rel_vel)-50))

        e = 1

        # если тела расходятся — пропускаем
        if rel_normal >= 0:
            continue

        # Импульс
        if mass[i] == 0 or mass[j] == 0:
            continue
        J = (1 + e) * rel_normal / (1/mass[i] + 1/mass[j])

        impulse = J * n

        # Пересчёт скоростей
        vel[i] -= impulse / mass[i]
        vel[j] += impulse / mass[j]