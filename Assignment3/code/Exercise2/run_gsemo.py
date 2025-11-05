import ioh
from ioh import logger
import numpy as np
import random
from GSEMO import GSEMO


def main():
    """Run GSEMO on multiple problem instances with logging"""
    # Problem instances as specified in the assignment
    instances = {
        "MaxCoverage": [2100, 2101, 2102, 2103],
        "MaxInfluence": [2200, 2201, 2202, 2203],
        "PackWhileTravel": [2300, 2301, 2302],
    }

    # Fixed budget of 10,000 fitness evaluations
    budget = 10000
    runs = 5

    # Set up logging
    gsemo_logger = logger.Analyzer(
        root="data",
        folder_name="gsemo_submodular",
        algorithm_name="GSEMO",
        algorithm_info="GSEMO for multi-objective submodular optimization",
    )

    # Run experiments for each problem category
    for problem_name, ids in instances.items():
        print(f"\n{'='*60}")
        print(f"Running experiments on {problem_name}")
        print(f"{'='*60}")

        for id in ids:
            print(f"\nProblem ID: {id}")
            print("-" * 40)

            # Get the problem instance to determine dimension
            problem = ioh.get_problem(id, problem_class=ioh.ProblemClass.GRAPH)
            dimension = problem.meta_data.n_variables
            
            print(f"Dimension: {dimension}")

            # Run 30 independent runs
            for run in range(runs):
                try:
                    # Create GSEMO instance with logger
                    gsemo = GSEMO(
                        problem_id=id,
                        dimension=dimension,
                        budget=budget,
                        logger_obj=gsemo_logger
                    )

                    # Run GSEMO
                    pareto_front = gsemo.run()

                    # Get best solution (highest f1)
                    best_sol, best_obj = gsemo.get_best_solution()
                    
                    if best_obj:
                        print(f"  Run {run+1:2d}: Best f1 = {best_obj[0]:.4f}, "
                              f"Cost (f2) = {best_obj[1]}, "
                              f"Pareto size = {len(pareto_front)}")

                    # Cleanup
                    gsemo.cleanup()

                    # Reset problem for next run
                    problem.reset()

                except Exception as e:
                    print(f"  Run {run+1:2d}: Error - {str(e)}")

    # Clean up logger
    del gsemo_logger


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run the experiments
    main()

