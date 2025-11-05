# opulation-based multi-objective evolutionary algorithm

# Multi-objective evolutionary algorithm for Exercise 3
# Two objectives: maximize f(S), minimize |S|
# Tournament selection (size 3)
# Uniform crossover (90% rate)
# Bit-flip mutation (1/n rate)
# Simple non-dominated sorting for selection
# Population sizes: 10, 20, 50

import random
import numpy as np
from ioh import get_problem, ProblemClass


def initialize(pop_size, dimension):
    """Initialize population with random binary solutions."""
    population = []
    for _ in range(pop_size):
        # Random cardinality between 0 and dimension
        num_ones = random.randint(0, dimension)
        individual = np.zeros(dimension, dtype=int)
        if num_ones > 0:
            indices = random.sample(range(dimension), num_ones)
            individual[indices] = 1
        population.append(individual)
    return population


def mutate(individual, mutation_rate):
    """Bit-flip mutation with rate 1/n."""
    mutated = individual.copy()
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated


def dominates(obj1, obj2):
    """
    Check if obj1 dominates obj2.
    obj = (f(S), -|S|) where we maximize f(S) and maximize -|S| (minimize |S|)
    """
    return (
        obj1[0] >= obj2[0]
        and obj1[1] >= obj2[1]
        and (obj1[0] > obj2[0] or obj1[1] > obj2[1])
    )


def select(population, objectives, tournament_size=3):
    """Tournament selection for multi-objective."""
    indices = random.sample(
        range(len(population)), min(tournament_size, len(population))
    )
    best_idx = indices[0]
    for idx in indices[1:]:
        if dominates(objectives[idx], objectives[best_idx]):
            best_idx = idx
    return population[best_idx].copy()


def crossover(parent1, parent2, crossover_rate=0.9):
    """Uniform crossover."""
    if random.random() > crossover_rate:
        return parent1.copy()

    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
    return child


def sorting(population, objectives):
    """
    Non-dominated sorting for diversity maintenance.
    Returns fronts (list of lists of indices).
    """
    fronts = []
    remaining = set(range(len(population)))

    while remaining:
        front = []
        for i in remaining:
            is_dominated = False
            for j in remaining:
                if i != j and dominates(objectives[j], objectives[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                front.append(i)

        if not front:
            break

        fronts.append(front)
        remaining -= set(front)

    return fronts


def generation(population, objectives, pop_size):
    """
    Select next generation using non-dominated sorting.
    Maintains diversity by selecting from multiple fronts.
    """
    fronts = sorting(population, objectives)

    selected_pop = []
    selected_objs = []

    # Fill from fronts
    for front in fronts:
        if len(selected_pop) + len(front) <= pop_size:
            # Add entire front
            for idx in front:
                selected_pop.append(population[idx])
                selected_objs.append(objectives[idx])
        else:
            # Fill remaining slots from this front
            remaining = pop_size - len(selected_pop)
            # Sort by second objective (cardinality) for diversity
            front_sorted = sorted(front, key=lambda i: objectives[i][1])
            for idx in front_sorted[:remaining]:
                selected_pop.append(population[idx])
                selected_objs.append(objectives[idx])
            break

        if len(selected_pop) >= pop_size:
            break

    # Fill remaining slots randomly if needed
    while len(selected_pop) < pop_size:
        idx = random.choice(range(len(population)))
        selected_pop.append(population[idx])
        selected_objs.append(objectives[idx])

    return selected_pop[:pop_size], selected_objs[:pop_size]


def multi(problem, pop_size, max_evals=10000):
    """
    Multi-objective EA for submodular optimization.
    Objectives: maximize f(S) and minimize |S|

    Args:
        problem: IOH problem instance (logger should be attached externally)
        pop_size: Population size
        max_evals: Maximum evaluations (10000)

    Returns:
        best_f, best_individual
    """

    # Setup
    n = problem.meta_data.n_variables
    mutation_rate = 1.0 / n

    # Initialize population
    population = initialize(pop_size, n)
    objectives = []
    for ind in population:
        f_val = problem(ind)
        # Objectives: (f(S), -|S|) - maximize both
        cardinality = -np.sum(ind)  # Negative for minimization
        objectives.append((f_val, cardinality))

    evals = len(population)

    # Main EA loop
    while evals < max_evals:
        offspring = []
        offspring_objs = []

        # Generate offspring
        for _ in range(pop_size):
            if evals >= max_evals:
                break

            # Selection
            parent1 = select(population, objectives)
            parent2 = select(population, objectives)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child, mutation_rate)

            # Evaluate
            f_val = problem(child)
            cardinality = -np.sum(child)  # Negative for minimization

            offspring.append(child)
            offspring_objs.append((f_val, cardinality))
            evals += 1

        # Combine parents and offspring
        combined_pop = population + offspring
        combined_objs = objectives + offspring_objs

        # Select next generation using non-dominated sorting
        population, objectives = generation(combined_pop, combined_objs, pop_size)

    # Extract best f(S) from final population
    best_f = max(obj[0] for obj in objectives) if objectives else 0
    best_idx = max(range(len(objectives)), key=lambda i: objectives[i][0]) if objectives else None
    best_individual = population[best_idx] if best_idx is not None else None

    return best_f, best_individual


if __name__ == "__main__":
    # Test run
    random.seed(42)
    np.random.seed(42)

    problem = get_problem(2100, problem_class=ProblemClass.GRAPH)
    
    best_f, best_individual = multi(
        problem=problem,
        pop_size=20,
        max_evals=10000,
    )

    print(f"Best f(S): {best_f:.2f}")
