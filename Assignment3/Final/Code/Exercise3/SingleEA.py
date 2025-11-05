import random
import numpy as np
from ioh import get_problem, ProblemClass


def initialize_population(pop_size, dimension):
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


def select(population, fitnesses, tournament_size=3):
    """Tournament selection."""
    indices = random.sample(
        range(len(population)), min(tournament_size, len(population))
    )
    best_idx = indices[0]
    for idx in indices[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:
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


def diverse(population, fitnesses, pop_size):
    """
    Diversity-aware selection using Hamming distance.
    Greedily selects diverse solutions starting from best fitness.
    """
    if len(population) <= pop_size:
        return population.copy()

    # Sort by fitness
    idx_sorted = sorted(
        range(len(population)), key=lambda i: fitnesses[i], reverse=True
    )

    selected = [population[idx_sorted[0]]]
    selected_indices = {idx_sorted[0]}

    while len(selected) < pop_size:
        best_candidate = None
        best_score = -1

        for idx in idx_sorted:
            if idx in selected_indices:
                continue

            # Calculate minimum Hamming distance to already selected
            min_dist = min(np.sum(population[idx] != s) for s in selected)

            if min_dist > best_score:
                best_score = min_dist
                best_candidate = idx

        if best_candidate is None:
            break

        selected.append(population[best_candidate])
        selected_indices.add(best_candidate)

    return selected


def single(problem, pop_size, max_evals=10000):
    """
    Single-objective EA for submodular optimization with uniform constraint.

    Args:
        problem: IOH problem instance (logger should be attached externally)
        pop_size: Population size
        max_evals: Maximum evaluations (10000)

    Returns:
        best_fitness, best_individual
    """
    # Setup
    n = problem.meta_data.n_variables
    mutation_rate = 1.0 / n

    # Initialize population
    population = initialize_population(pop_size, n)
    fitnesses = [problem(ind) for ind in population]

    evals = len(population)
    best_fitness = max(fitnesses) if fitnesses else 0
    best_individual = population[fitnesses.index(best_fitness)] if fitnesses else None

    # Main EA loop
    while evals < max_evals:
        offspring = []
        offspring_fits = []

        # Generate offspring
        for _ in range(pop_size):
            if evals >= max_evals:
                break

            # Selection
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child, mutation_rate)

            # Evaluate
            fit = problem(child)

            offspring.append(child)
            offspring_fits.append(fit)
            evals += 1

            # Update best
            if fit > best_fitness:
                best_fitness = fit
                best_individual = child.copy()

        # Combine parents and offspring
        combined_pop = population + offspring
        combined_fits = fitnesses + offspring_fits

        # Diversity-aware selection
        population = diverse(combined_pop, combined_fits, pop_size)
        # Recalculate fitnesses for selected population
        fitnesses = []
        for ind in population:
            # Find matching fitness from combined
            found = False
            for j, combined_ind in enumerate(combined_pop):
                if np.array_equal(ind, combined_ind):
                    fitnesses.append(combined_fits[j])
                    found = True
                    break
            if not found:
                # If not found, evaluate
                fit = problem(ind)
                fitnesses.append(fit)

    return best_fitness, best_individual


if __name__ == "__main__":
    # Test run
    random.seed(42)
    np.random.seed(42)

    from ioh import get_problem, ProblemClass
    problem = get_problem(2100, problem_class=ProblemClass.GRAPH)
    
    best_fitness, best_individual = single(
        problem=problem,
        pop_size=20,
        max_evals=10000,
    )

    print(f"Final best fitness: {best_fitness:.2f}")
