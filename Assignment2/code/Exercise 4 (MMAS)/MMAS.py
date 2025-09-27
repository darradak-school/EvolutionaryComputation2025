from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np


Objective = Callable[[List[int]], float]


@dataclass
class MMASConfig:
    n: int # problem dimension
    rho: float # evaporation and learning rate
    p_min: Optional[float] = None # lower pheromone bound (default = 1/n)
    p_max: Optional[float] = None # upper pheromone bound (default = 1 - 1/n)
    seed: Optional[int] = None # RNG seed

    def __post_init__(self):
        if self.p_min is None:
            self.p_min = 1.0 / self.n
        if self.p_max is None:
            self.p_max = 1.0 - 1.0 / self.n
        if not (0.0 < self.p_min <= 0.5 <= self.p_max < 1.0):
            raise ValueError("Require 0 < p_min ≤ 0.5 ≤ p_max < 1.")
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must be in (0, 1].")


def _sample_solution(rng: np.random.Generator, p: np.ndarray) -> List[int]:
    """Construct a solution by the chain/bitwise construction: x_i ~ Bernoulli(p_i)."""
    return (rng.random(p.size) < p).astype(np.int8).tolist()


def _update_pheromones(
    p: np.ndarray, x_best: np.ndarray, rho: float, p_min: float, p_max: float
) -> np.ndarray:
    """
    Pheromone (success prob) update toward the current best-so-far:
    p <- clip( (1 - rho) * p + rho * x_best, [p_min, p_max] ).
    """
    p = (1.0 - rho) * p + rho * x_best
    # Clamp to borders [p_min, p_max]
    np.minimum(p, p_max, out=p)
    np.maximum(p, p_min, out=p)
    return p


def mmas(
    objective: Objective,
    cfg: MMASConfig,
    budget: int,
) -> Tuple[List[float], List[int], float]:
    """
    Max-Min Ant System (MMAS): accepts non-worsening solutions (>=).
    Follows Figure 3 with the construction of Figures 1–2.
    Returns (trace of best-so-far fitness length=budget, best_solution, best_fitness).
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    p = np.full(n, 0.5, dtype=float) # initialize pheromones to 1/2

    # Initial construct & evaluate (counts toward budget)
    x_best = np.array(_sample_solution(rng, p), dtype=float)
    f_best = float(objective(x_best.tolist()))
    trace = [f_best]

    # Reinforce towards current best after the initial sample
    p = _update_pheromones(p, x_best, cfg.rho, cfg.p_min, cfg.p_max)

    # Main loop: one construction (evaluation) per iteration
    while len(trace) < budget:
        x = np.array(_sample_solution(rng, p), dtype=float)
        fx = float(objective(x.tolist()))
        if fx >= f_best:
            x_best, f_best = x, fx
        trace.append(f_best)
        p = _update_pheromones(p, x_best, cfg.rho, cfg.p_min, cfg.p_max)

    return trace, x_best.astype(int).tolist(), f_best


def mmas_star(
    objective: Objective,
    cfg: MMASConfig,
    budget: int,
) -> Tuple[List[float], List[int], float]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    p = np.full(n, 0.5, dtype=float)

    x_best = np.array(_sample_solution(rng, p), dtype=float)
    f_best = float(objective(x_best.tolist()))
    trace = [f_best]

    p = _update_pheromones(p, x_best, cfg.rho, cfg.p_min, cfg.p_max)

    while len(trace) < budget:
        x = np.array(_sample_solution(rng, p), dtype=float)
        fx = float(objective(x.tolist()))
        if fx > f_best:
            x_best, f_best = x, fx
        trace.append(f_best)
        p = _update_pheromones(p, x_best, cfg.rho, cfg.p_min, cfg.p_max)

    return trace, x_best.astype(int).tolist(), f_best

if __name__ == "__main__":
    # Example objective functions over {0,1}^n (maximization)
    def onemax(x: List[int]) -> float:
        return float(sum(x))

    def leading_ones(x: List[int]) -> float:
        s = 0
        for bit in x:
            if bit == 1:
                s += 1
            else:
                break
        return float(s)

    n = 50
    budget = 5_000

    # Try rho in {1, 1/sqrt(n), 1/n}
    rhos = [1.0, 1.0 / np.sqrt(n), 1.0 / n]

    print("MMAS on OneMax:")
    for rho in rhos:
        cfg = MMASConfig(n=n, rho=rho, seed=42)
        trace, xb, fb = mmas(onemax, cfg, budget)
        print(f"  rho={rho:.4f}  best={fb:.1f}")

    print("\nMMAS* on OneMax:")
    for rho in rhos:
        cfg = MMASConfig(n=n, rho=rho, seed=42)
        trace, xb, fb = mmas_star(onemax, cfg, budget)
        print(f"  rho={rho:.4f}  best={fb:.1f}")

    print("\nMMAS on LeadingOnes:")
    for rho in rhos:
        cfg = MMASConfig(n=n, rho=rho, seed=123)
        trace, xb, fb = mmas(leading_ones, cfg, budget)
        print(f"  rho={rho:.4f}  best={fb:.1f}")