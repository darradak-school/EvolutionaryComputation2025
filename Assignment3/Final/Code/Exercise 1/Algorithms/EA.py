import random
import numpy as np

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

