import ioh
from ioh import logger
import numpy as np
import random

# Import the algorithms from the Algorithms folder
from Algorithms.RLS import RLS
from Algorithms.EA import evolutionary_algorithm
from Algorithms.ga import ga


def comp():
    """Compare RLS, (1+1) EA, and GA on submodular optimization problems"""
    # Problem instances as specified in the assignment
    instances = {
        "MaxCoverage": [2100, 2101, 2102, 2103],
        "MaxInfluence": [2200, 2201, 2202, 2203],
        #"PackWhileTravel": [2300, 2301, 2302],
    }

    # Algorithm configurations
    algorithms = {"RLS": RLS, "EA": evolutionary_algorithm, "GA": ga}

    # Fixed budget of 10,000 fitness evaluations
    budget = 100000
    runs = 1

    # Set up logging for each algorithm
    loggers = {}
    for alg_name in algorithms.keys():
        loggers[alg_name] = logger.Analyzer(
            root="data",
            folder_name=f"{alg_name.lower()}_submodular",
            algorithm_name=alg_name,
            algorithm_info=f"{alg_name} for submodular optimization problems",
        )

    # Run experiments for each problem category
    for problem, ids in instances.items():
        print(f"\n{'='*60}")
        print(f"Running experiments on {problem}")
        print(f"{'='*60}")

        for id in ids:
            print(f"\nProblem ID: {id}")
            print("-" * 40)

            # Get the problem instance
            problem = ioh.get_problem(id, problem_class=ioh.ProblemClass.GRAPH)

            # Run each algorithm
            for alg_name, alg_func in algorithms.items():
                print(f"\nRunning {alg_name} on problem {id}...")

                # Attach logger
                problem.attach_logger(loggers[alg_name])

                # Run 30 independent runs
                for run in range(runs):
                    try:
                        if alg_name == "RLS":
                            # RLS expects (problem, n, iterations)
                            best_fitness, _ = alg_func(
                                problem, problem.meta_data.n_variables, budget
                            )
                        else:
                            # EA and GA expect (problem, budget)
                            best_fitness, _ = alg_func(problem, budget)

                        print(f"  Run {run+1:2d}: Best fitness = {best_fitness:.4f}")

                    except Exception as e:
                        print(f"  Run {run+1:2d}: Error - {str(e)}")

                    # Reset problem for next run
                    problem.reset()

                # Detach logger for this algorithm
                problem.detach_logger()

    # Clean up loggers
    for logger_obj in loggers.values():
        del logger_obj


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run the comparison
    comp()
