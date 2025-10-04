import random
import numpy as np
from ioh import get_problem, logger, ProblemClass

def Uniform_Mutation(parent, mutation_rate=None):
    """Perform uniform mutation on a binary parent solution"""
    mutated = np.copy(parent)
    mutation_rate = mutation_rate or len(parent)
    return np.array([1 - mutated[i] if random.random() < (1 / mutation_rate) else mutated[i] for i in range(len(parent))])

def Fitness_Evaluate(problem, solution):
    """Evaluate solution"""
    return problem(solution)

def evolutionary_algorithm(func, budget=100000):
    """Evolutionary Algorithm adapted to fit the problem_example.py template"""
    # Initialize with random solution
    parent = np.random.randint(0, 2, size=func.meta_data.n_variables)
    parent_fitness = Fitness_Evaluate(func, parent)
    evaluations = 1
    
    # Get optimum for early stopping
    optimum = func.optimum.y
    if hasattr(optimum, "item"):
        optimum = optimum.item()
    
    # Run EA loop
    while evaluations < budget:
        # Create offspring by mutation
        child = Uniform_Mutation(parent, func.meta_data.n_variables)
        child_fitness = Fitness_Evaluate(func, child)
        evaluations += 1
        
        # Accept if better or equal (maximization)
        if child_fitness >= parent_fitness:
            parent = child
            parent_fitness = child_fitness
        
        # Early stopping if optimum reached
        if not np.isinf(optimum) and parent_fitness >= optimum:
            break
    
    func.reset()
    return parent_fitness, parent

# Create default logger compatible with IOHanalyzer
l = logger.Analyzer(root="data", 
    folder_name="ea_run", 
    algorithm_name="evolutionary_algorithm", 
    algorithm_info="EA with uniform mutation and (1+1) selection")

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

# Run EA on all problems (10 runs each)
for problem_name, problem in problems:
    print(f"\n{'='*50}")
    print(f"Running EA on {problem_name}")
    print(f"{'='*50}")
    
    # Attach logger to the problem
    problem.attach_logger(l)
    
    # Do 10 runs
    for run in range(10):
        print(f"Run {run + 1}/10")
        f_opt, x_opt = evolutionary_algorithm(problem)
        print(f"Best fitness: {f_opt:.4f}")
    
    # Detach logger for next problem
    problem.detach_logger()

# This statement is necessary in case data is not flushed yet
del l
