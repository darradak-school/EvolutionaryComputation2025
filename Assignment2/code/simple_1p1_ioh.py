# simple_1p1_ioh_fixed.py
# Minimal (1+1) EA on OneMax using the latest IOH Python interface.

import os
import random
import numpy as np
from ioh import get_problem, logger, ProblemClass

def one_plus_one_EA_on_onemax(dim=50, budget=5000, folder_name="ioh_example"):
    # Create OneMax problem
    prob = get_problem("OneMax", instance=1, dimension=dim, problem_class=ProblemClass.PBO)

    # Create Analyzer logger
    ioh_logger = logger.Analyzer(
        root=os.getcwd(),
        folder_name=folder_name,
        algorithm_name="1p1EA",
    )

    # Attach logger to the problem
    prob.attach_logger(ioh_logger)

    # Initialize random solution
    x = np.random.randint(0, 2, dim).tolist()
    fx = prob(x)  # Evaluate initial solution

    for _ in range(budget):
        # Flip a random bit
        y = x.copy()
        i = random.randrange(dim)
        y[i] = 1 - y[i]

        fy = prob(y)  # Evaluate candidate

        # Accept if not worse (maximization)
        if fy >= fx:
            x, fx = y, fy

        # Stop if optimum is found
        if fx == dim:
            break

    print("Done. Logs written to folder:", os.path.join(os.getcwd(), folder_name))

if __name__ == "__main__":
    one_plus_one_EA_on_onemax(dim=50, budget=2000, folder_name="ioh_onemax_example")