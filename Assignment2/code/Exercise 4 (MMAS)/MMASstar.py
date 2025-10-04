from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np
from ioh import get_problem, ProblemClass, logger


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


def construct_solution(rng: np.random.Generator, p: np.ndarray) -> List[int]:
    """Construct solution"""
    solution = []
    for i in range(len(p)):
        # Each bit is chosen with probability p[i]
        bit = 1 if rng.random() < p[i] else 0
        solution.append(bit)
    return solution


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


def mmas_star(
    objective: Objective,
    cfg: MMASConfig,
    budget: int,
) -> Tuple[List[float], List[int], float]:
    """
    Max-Min Ant System* (MMAS*): accepts only improving solutions (>).
    Follows Figure 3 with the construction of Figures 1–2.
    Returns (trace of best-so-far fitness length=budget, best_solution, best_fitness).
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    
    # Initialize pheromones to 1/2 for all variables
    p = np.full(n, 0.5, dtype=float)
    
    # Create initial best solution using Construct(C, t)
    x_best = np.array(construct_solution(rng, p), dtype=int)
    f_best = float(objective(x_best.tolist()))
    trace = [f_best]
    
    # Update pheromones w.r.t. x*
    p = _update_pheromones(p, x_best, cfg.rho, cfg.p_min, cfg.p_max)
    
    # Iterative improvement
    while len(trace) < budget:
        # Create x using Construct(C, t)
        x = np.array(construct_solution(rng, p), dtype=int)
        fx = float(objective(x.tolist()))
        
        # If f(x) > f(x*) then x* := x
        if fx > f_best:
            x_best, f_best = x, fx
        
        trace.append(f_best)
        
        # Update pheromones w.r.t. x*
        p = _update_pheromones(p, x_best, cfg.rho, cfg.p_min, cfg.p_max)

    return trace, x_best.astype(int).tolist(), f_best


def mmas_star_algorithm(func, budget=100000):
    """MMAS* algorithm"""
    # Get problem dimension
    n = func.meta_data.n_variables
    
    # MMAS* configuration
    cfg = MMASConfig(n=n, rho=1.0/n, seed=None)
    
    # Run MMAS* and get results
    _, x_best, f_best = mmas_star(func, cfg, budget)
    
    # Reset function for next run
    func.reset()
    
    return f_best, x_best


# Create default logger compatible with IOHanalyzer
l = logger.Analyzer(root="data", 
    folder_name="mmas_star_run", 
    algorithm_name="max_min_ant_system_star", 
    algorithm_info="MMAS* with pheromone construction")

# List of problems to be tested
problems = [
    ("OneMax", get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO)),
    ("LeadingOnes", get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO)),
    ("LinearFunc", get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO)),
    ("LABS", get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO)),
    ("NQueens", get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO)),
    ("CTrap", get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO)),
    ("NKL", get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO))
]

# Run MMAS* on all problems (10 runs each)
for problem_name, problem in problems:
    print(f"\n{'='*50}")
    print(f"Running MMAS* on {problem_name}")
    print(f"{'='*50}")
    
    # Attach logger to the problem
    problem.attach_logger(l)
    
    # Run 10 independent runs
    for run in range(10):
        print(f"Run {run + 1}/10")
        f_opt, x_opt = mmas_star_algorithm(problem)
        print(f"Best fitness: {f_opt:.4f}")
    
    # Detach logger for next problem
    problem.detach_logger()

# This statement is necessary in case data is not flushed yet
del l
