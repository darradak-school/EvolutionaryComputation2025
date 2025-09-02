# RLS (Randomised Local Search

# Choose s ∈ {0, 1}n randomly

# Copy the solution, 
# but flip exactly one bit (turn a 0 into 1 or a 1 into 0).

# If the new solution is just as good or better than the old one
# (based on the problem’s objective function), then keep the new solution.
# Otherwise, stick with the old one.

# Repeat forever 

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

def RLS(problem, n, iterations, return_trace=False):
    s = [random.randint(0, 1) for _ in range(n)]
    f_s = problem(s)
    trace = [f_s]

    for _ in range(iterations):
        s_prime = s.copy()
        i = random.randrange(n)
        s_prime[i] = 1 - s_prime[i]
        f_s_prime = problem(s_prime)

        if f_s_prime >= f_s:
            s, f_s = s_prime, f_s_prime

        if return_trace:
            trace.append(f_s)

    if return_trace:
        return trace, f_s  # return full trace
    else:
        return f_s, None
    
