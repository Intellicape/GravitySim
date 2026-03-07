from __future__ import annotations
from .model import accelerations

def euler_step(pos, vel, mass, dt, G, eps):
    a = accelerations(pos, mass, G, eps)
    pos_new = pos + dt * vel
    vel_new = vel + dt * a
    return pos_new, vel_new

def leapfrog_step(pos, vel, mass, dt, G, eps):
    # Kick-Drift-Kick (velocity Verlet)
    a = accelerations(pos, mass, G, eps)
    v_half = vel + 0.5 * dt * a
    pos_new = pos + dt * v_half
    a_new = accelerations(pos_new, mass, G, eps)
    vel_new = v_half + 0.5 * dt * a_new
    return pos_new, vel_new

def rk4_step(pos, vel, mass, dt, G, eps):
    def acc(p): return accelerations(p, mass, G, eps)
    k1p, k1v = vel, acc(pos)
    k2p, k2v = vel + 0.5*dt*k1v, acc(pos + 0.5*dt*k1p)
    k3p, k3v = vel + 0.5*dt*k2v, acc(pos + 0.5*dt*k2p)
    k4p, k4v = vel + dt*k3v, acc(pos + dt*k3p)
    pos_new = pos + (dt/6.0)*(k1p + 2*k2p + 2*k3p + k4p)
    vel_new = vel + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return pos_new, vel_new

INTEGRATORS = {
    "euler": euler_step,
    "leapfrog": leapfrog_step,  # рекомендованный по умолчанию
    "rk4": rk4_step,
}
