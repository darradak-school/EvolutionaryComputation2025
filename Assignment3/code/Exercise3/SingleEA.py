#Population-based single-objective evolutionary algorithm
#assumes a uniform constraint (e.g., selecting exactly k elements).
#It uses a (μ+λ) EA with diversity maintenance via Hamming distance.

import random 
import numpy as np
from ioh import get_problem, ProblemClass

def initialize_population(pop_size, dimension, budget):
    population = []
    for _ in range(pop_size):
        individual = np.zeros(dimension, dtype=int)
        ones = random.sample(range(dimension), budget)
        individual[ones] = 1
        population.append(individual)
    return population

def mutate(individual):
    offspring = individual.copy()
    idx = random.randint(0, len(individual) - 1)
    offspring[idx] = 1 - offspring[idx]
    return offspring

def repair(individual, budget):
    """Ensure exactly 'budget' ones in the solution."""
    ones = np.where(individual == 1)[0]
    zeros = np.where(individual == 0)[0]
    if len(ones) > budget:
        to_flip = random.sample(list(ones), len(ones) - budget)
        individual[to_flip] = 0
    elif len(ones) < budget:
        to_flip = random.sample(list(zeros), budget - len(ones))
        individual[to_flip] = 1
    return individual

def hamming_distance(a, b):
    return np.sum(a != b)

def select_diverse(population, fitnesses, mu):
    selected = [population[0]]
    while len(selected) < mu:
        candidates = [ind for ind in population if ind.tolist() not in [s.tolist() for s in selected]]
        if not candidates:
            break
        next_ind = max(candidates, key=lambda x: min(hamming_distance(x, s) for s in selected))
        selected.append(next_ind)
    return selected

def single_objective_ea(problem_id=2100, pop_size=20, budget=10):
    problem = get_problem(problem_id, problem_class=ProblemClass.GRAPH)
    dimension = problem.meta_data.n_variables
    max_evals = 10000

    population = initialize_population(pop_size, dimension, budget)
    fitnesses = [problem(ind) for ind in population]
    evals = len(population)

    fitness_log = []

    while evals < max_evals:
        offspring = []
        for parent in population:
            child = mutate(parent)
            child = repair(child, budget)
            fitness = problem(child)
            offspring.append((child, fitness))
            evals += 1
            if evals >= max_evals:
                break

        combined = population + [ind for ind, _ in offspring]
        combined_fitnesses = fitnesses + [fit for _, fit in offspring]
        fitness_pairs = [(float(fit), ind.tolist()) for fit, ind in zip(combined_fitnesses, combined)]
        fitness_pairs.sort(reverse=True, key=lambda x: x[0])
        sorted_combined = [np.array(ind) for _, ind in fitness_pairs]
        population = select_diverse(sorted_combined, combined_fitnesses, pop_size)

        #track per generation 
        best_fitness = max(combined_fitnesses)
        fitness_log.append(best_fitness)
        print("Fitness log per generation:")
        for gen, fit in enumerate(fitness_log):
            print(f"Generation {gen}: {fit}")


    return population, fitness_log
