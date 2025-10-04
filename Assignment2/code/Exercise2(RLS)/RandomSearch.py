from ioh import get_problem, ProblemClass, logger
import sys
import numpy as np

# --- Random Search Algorithm ---
def random_search(func, budget=None):

    # Run 10 independent trials
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None
        for i in range(budget):
            x = np.random.randint(2, size=func.meta_data.n_variables)
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
        print(f"Run {r+1} best fitness: {f_opt}")
        func.reset()

    return f_opt, x_opt

# --- Problem Setup ---
FUNCTION_IDS = [1, 2, 3, 18, 23, 24, 25]
DIMENSION = 100
INSTANCE = 1
ITERATIONS = 100000

# --- Logger Setup ---
log = logger.Analyzer(
    root="data",
    folder_name="random_search_runs",
    algorithm_name="random_search",
    algorithm_info="Basic random search for IOHprofiler benchmarks"
)

# --- Run Random Search on Each Problem ---
for fid in FUNCTION_IDS:
    problem = get_problem(fid=fid, dimension=DIMENSION, instance=INSTANCE, problem_class=ProblemClass.PBO)
    problem.attach_logger(log)
    print(f"\nRunning Random Search on F{fid}...")
    random_search(problem)

# --- Finalize Logging ---
del log