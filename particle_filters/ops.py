import numpy as np
from typing import  NamedTuple

RNG = np.random.default_rng()
class Range(NamedTuple):
    low: float
    high: float


def create_uniform_particles(x_range: Range, y_range: Range, heading_range: Range, n: int) -> np.ndarray:
    particles: np.ndarray = np.zeros((n, 3))
    particles[:, 0] = RNG.uniform(x_range.low, x_range.high, size=n)
    particles[:, 1] = RNG.uniform(x_range.low, x_range.high, size=n)
    particles[:, 2] = RNG.uniform(x_range.low, x_range.high, size=n)
    return particles


def predict(particles: np.ndarray, u: np.ndarray, std: np.ndarray , dt = 1):
    n = len(particles)
    particles = particles.copy()
    particles[:, 2] += u[0] + (RNG.random(n) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (RNG.random(n) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
    return  particles


if __name__ == "__main__":
    part_ = create_uniform_particles(Range(0, 1), Range(0, 1), Range(0, np.pi * 2), 4)
    print(predict(part_, np.array([1, 0]), np.array([1, 0])))