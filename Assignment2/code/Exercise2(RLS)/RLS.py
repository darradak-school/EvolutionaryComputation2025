# RLS (Randomised Local Search)

import ioh
import random
from ioh import logger, ProblemClass

# choose s ∈ {0, 1}ⁿ randomly
def random_solution(n):
    return [random.randint(0, 1) for _ in range(n)]

# flip exactly one bit randomly
def flip_one_bit(solution):
    s_prime = solution.copy()
    bit = random.randint(0, len(solution) - 1)
    s_prime[bit] = 1 - s_prime[bit]
    return s_prime

# Randomized Local Search (RLS)
# problem     : ioh problem object (fitness function)
# n           : dimension of the solution (number of bits)
# iterations  : number of iterations to run
def RLS(problem, n, iterations):
    # Step 1: Initialize with a random solution
    s = random_solution(n)
    f_s = problem(s)

    # Step 2: Iteratively improve the solution
    for _ in range(iterations):
        s_prime = flip_one_bit(s)
        f_s_prime = problem(s_prime)
        # Step 3: Accept new solution if it's at least as good
        if f_s_prime >= f_s:
            s, f_s = s_prime, f_s_prime

    # Step 4: Return result
    return f_s, s  # final fitness and solution

# --- Config ---
FUNCTION_IDS = [1, 2, 3, 18, 23, 24, 25]
DIMENSION = 100
INSTANCE = 1
ITERATIONS = 100000

# --- Logger Setup ---
log = logger.Analyzer(
    root="data",
    folder_name="rls_runs",
    algorithm_name="RLS",
    algorithm_info="Randomized Local Search for IOHprofiler benchmarks"
)

# Run RLS on each benchmark function
for fid in FUNCTION_IDS:
    problem = ioh.get_problem(fid=fid, dimension=DIMENSION, instance=INSTANCE, problem_class=ProblemClass.PBO)
    problem.attach_logger(log)
    print(f"\nRunning RLS on F{fid}...")

    # 10 independent runs for each algorithm on each problem
    for run in range(10):
        best_fitness, best_solution = RLS(problem, DIMENSION, ITERATIONS)
        print(f"Run {run+1} best fitness: {best_fitness}")
        problem.reset()

# This statement is necessary in case data is not flushed yet
del log