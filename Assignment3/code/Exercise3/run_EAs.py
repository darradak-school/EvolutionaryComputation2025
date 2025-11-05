"""
Runs both Single-Objective and Multi-Objective EAs on all problem instances
Does 30 runs, 10,000 evaluations, population sizes 10/20/50
"""

import random
import numpy as np
import ioh
from ioh import logger
from SingleEA import single
from MultiEA import multi


def run_single():
    """Run Single-Objective EA."""
    problem_instances = {
        "MaxCoverage": [2100, 2101, 2102, 2103],
        "MaxInfluence": [2200, 2201, 2202, 2203],
    }

    pop_sizes = [10, 20, 50]
    runs = 3
    max_evals = 10000

    # Set up logging for each population size
    loggers = {}
    for pop_size in pop_sizes:
        loggers[pop_size] = logger.Analyzer(
            root="data",
            folder_name=f"singleea_pop{pop_size}_exercise3",
            algorithm_name=f"SingleEA_pop{pop_size}",
            algorithm_info=f"Single-Objective EA (pop={pop_size}) for submodular optimization",
        )

    for problem_type, problem_ids in problem_instances.items():
        print(f"\n{'='*60}")
        print(f"Running experiments on {problem_type}")
        print(f"{'='*60}")

        for problem_id in problem_ids:
            print(f"\nProblem ID: {problem_id}")
            print("-" * 40)

            # Get the problem instance
            problem = ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)

            for pop_size in pop_sizes:
                print(f"\nRunning SingleEA (pop={pop_size}) on problem {problem_id}...")

                # Attach logger
                problem.attach_logger(loggers[pop_size])

                # Run 30 independent runs
                for run in range(runs):
                    try:
                        # Set seed for reproducibility
                        random.seed(42 + run)
                        np.random.seed(42 + run)

                        # Run algorithm
                        best_fitness, _ = single(
                            problem=problem,
                            pop_size=pop_size,
                            max_evals=max_evals,
                        )

                        print(f"  Run {run+1:2d}: Best fitness = {best_fitness:.4f}")

                    except Exception as e:
                        print(f"  Run {run+1:2d}: Error - {str(e)}")

                    # Reset problem for next run
                    problem.reset()

                # Detach logger for this population size
                problem.detach_logger()


def run_multi():
    """Run Multi-Objective EA."""
    problem_instances = {
        "MaxCoverage": [2100, 2101, 2102, 2103],
        "MaxInfluence": [2200, 2201, 2202, 2203],
    }

    pop_sizes = [10, 20, 50]
    runs = 3
    max_evals = 10000

    # Set up logging for each population size
    loggers = {}
    for pop_size in pop_sizes:
        loggers[pop_size] = logger.Analyzer(
            root="data",
            folder_name=f"multiea_pop{pop_size}_exercise3",
            algorithm_name=f"MultiEA_pop{pop_size}",
            algorithm_info=f"Multi-Objective EA (pop={pop_size}) for submodular optimization",
        )

    for problem_type, problem_ids in problem_instances.items():
        print(f"\n{'='*60}")
        print(f"Running experiments on {problem_type}")
        print(f"{'='*60}")

        for problem_id in problem_ids:
            print(f"\nProblem ID: {problem_id}")
            print("-" * 40)

            # Get the problem instance
            problem = ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)

            for pop_size in pop_sizes:
                print(f"\nRunning MultiEA (pop={pop_size}) on problem {problem_id}...")

                # Attach logger
                problem.attach_logger(loggers[pop_size])

                # Run 30 independent runs
                for run in range(runs):
                    try:
                        # Set seed for reproducibility
                        random.seed(42 + run)
                        np.random.seed(42 + run)

                        # Run algorithm
                        best_f, _ = multi(
                            problem=problem,
                            pop_size=pop_size,
                            max_evals=max_evals,
                        )

                        print(f"  Run {run+1:2d}: Best fitness = {best_f:.4f}")

                    except Exception as e:
                        print(f"  Run {run+1:2d}: Error - {str(e)}")

                    # Reset problem for next run
                    problem.reset()

                # Detach logger for this population size
                problem.detach_logger()


def main():
    # Run Multi-Objective EA
    run_multi()

    # Run Single-Objective EA
    run_single()




if __name__ == "__main__":
    # Set global seed
    random.seed(42)
    np.random.seed(42)

    main()
