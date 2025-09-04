# RLS (Randomised Local Search

import ioh
import random

# choose s ∈ {0, 1}n randomly
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
# return_trace: if True, record fitness after each iteration

def RLS(problem, n, iterations, return_trace=False):
    # Step 1: Initialize with a random solution
    s = [random.randint(0, 1) for _ in range(n)]
    f_s = problem(s)
    trace = [f_s]
    # Step 2: Iteratively improve the solution
    for _ in range(iterations):
        s_prime = s.copy()
        i = random.randrange(n)
        s_prime[i] = 1 - s_prime[i]
        f_s_prime = problem(s_prime)
        # Step 3: Accept new solution if it's at least as good
        if f_s_prime >= f_s:
            s, f_s = s_prime, f_s_prime
        # Record progress
        if return_trace:
            trace.append(f_s)
    # Step 4: Return result
    if return_trace:
        return trace, f_s  # return fitness trace and final fitness
    else:
        return f_s, None   # return only final fitness
    
