# RLS (Randomised Local Search)
import random

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
