from ioh import get_problem, ProblemClass
from ioh import logger
import numpy as np

def eval(func, indv):
    """Helper function to evaluate fitness"""
    fitness = func(indv)
    return fitness.item() if hasattr(fitness, "item") else fitness

def crossover(parent1, parent2):
    """Uniform crossover, each bit comes from either parent with 50% probability"""
    mask = np.random.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

def mutation(indv, generation, rate=0.1):
    """Adaptive mutation rate that decreases over generations"""
    mutation_rate = max(rate * (1.0 - generation / 100.0), 0.01)
    mask = np.random.random(len(indv)) < mutation_rate
    return np.where(mask, 1 - indv, indv)

def selection(pop, fitness, size=5):
    """Tournament selection"""
    idxs = np.random.choice(len(pop), size, replace=False)
    winner_idx = idxs[np.argmax([fitness[i] for i in idxs])]
    return pop[winner_idx]

def ga(func, budget=100000):
    """Genetic algorithm with uniform crossover and adaptive mutation"""
    pop_size = 20
    elite = 2
    
    # Initialize population
    pop = []
    fitness = []
    for _ in range(pop_size):
        indv = np.random.randint(2, size=func.meta_data.n_variables)
        fit_val = eval(func, indv)
        pop.append(indv)
        fitness.append(fit_val)

    pop = np.array(pop)
    fitness = np.array(fitness)

    # Track best solution
    best_idx = np.argmax(fitness)
    best_fitness = fitness[best_idx]
    best_indv = pop[best_idx].copy()

    # Get optimum for early stopping
    optimum = func.optimum.y
    if hasattr(optimum, "item"):
        optimum = optimum.item()

    evals = pop_size
    generation = 0

    while evals < budget:
        generation += 1
        new_pop = []
        new_fitness = []

        # Keep best individuals
        elite_idxs = np.argsort(fitness)[-elite:]
        new_pop.extend(pop[elite_idxs])
        new_fitness.extend(fitness[elite_idxs])

        # Generate offspring
        while len(new_pop) < pop_size:
            # Selection
            parent1 = selection(pop, fitness)
            parent2 = selection(pop, fitness)

            # Crossover and mutation
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, generation)
            child2 = mutation(child2, generation)

            # Evaluate offspring
            fitness1 = eval(func, child1)
            fitness2 = eval(func, child2)
            evals += 2

            new_pop.extend([child1, child2])
            new_fitness.extend([fitness1, fitness2])

        # Update population
        pop = np.array(new_pop[:pop_size])
        fitness = np.array(new_fitness[:pop_size])

        # Update best solution
        cur_idx = np.argmax(fitness)
        cur_fitness = fitness[cur_idx]
        if cur_fitness > best_fitness:
            best_fitness = cur_fitness
            best_indv = pop[cur_idx].copy()

        # Early stopping for finite optima
        if not np.isinf(optimum) and best_fitness >= optimum:
            break

    func.reset()
    return best_fitness, best_indv

# Create default logger compatible with IOHanalyzer
l = logger.Analyzer(root="ga_data", 
    folder_name="ga_run", 
    algorithm_name="genetic_algorithm", 
    algorithm_info="GA with tournament selection, uniform crossover, and adaptive mutation")

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

for problem_name, problem in problems:
    print(f"\n{'='*50}")
    print(f"Running GA on {problem_name}")
    print(f"{'='*50}")
    
    # Attach logger to the problem
    problem.attach_logger(l)
    
    # Run 10 independent runs
    for run in range(10):
        print(f"Run {run + 1}/10")
        f_opt, x_opt = ga(problem)
        print(f"Best fitness: {f_opt:.4f}")
    
    # Detach logger for next problem
    problem.detach_logger()

# This statement is necessary in case data is not flushed yet.
del l
