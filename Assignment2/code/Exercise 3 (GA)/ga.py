import numpy as np
import ioh
import matplotlib.pyplot as plt


def evaluate_fitness(func, individual):
    """Helper function to evaluate fitness and ensure scalar value"""
    fitness = func(individual)
    return fitness.item() if hasattr(fitness, "item") else fitness


def uniform_crossover(parent1, parent2):
    """Uniform crossover - each bit comes from either parent with 50% probability"""
    mask = np.random.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def adaptive_mutation(individual, generation, base_rate=0.1):
    """Adaptive mutation rate that decreases over generations"""
    mutation_rate = max(base_rate * (1.0 - generation / 100.0), 0.01)
    mask = np.random.random(len(individual)) < mutation_rate
    return np.where(mask, 1 - individual, individual)


def tournament_selection(population, fitness, tournament_size=5):
    """Tournament selection"""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    winner_idx = indices[np.argmax([fitness[i] for i in indices])]
    return population[winner_idx]


def ga(func, budget, population_size=20, elite_size=2):
    """Improved Genetic Algorithm with elitism and adaptive mutation"""
    # Initialize population
    population = []
    fitness = []
    for _ in range(population_size):
        individual = np.random.randint(2, size=func.meta_data.n_variables)
        fitness_val = evaluate_fitness(func, individual)
        population.append(individual)
        fitness.append(fitness_val)

    population = np.array(population)
    fitness = np.array(fitness)

    # Track best solution
    best_idx = np.argmax(fitness)
    best_fitness = fitness[best_idx]
    best_individual = population[best_idx].copy()

    # Get optimum for early stopping
    optimum = func.optimum.y
    if hasattr(optimum, "item"):
        optimum = optimum.item()

    # Track convergence
    fitness_history = [best_fitness]
    evaluations = population_size
    generation = 0

    while evaluations < budget:
        generation += 1
        new_population = []
        new_fitness = []

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness)[-elite_size:]
        new_population.extend(population[elite_indices])
        new_fitness.extend(fitness[elite_indices])

        # Generate offspring
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)

            # Crossover and mutation
            child1, child2 = uniform_crossover(parent1, parent2)
            child1 = adaptive_mutation(child1, generation)
            child2 = adaptive_mutation(child2, generation)

            # Evaluate offspring
            fitness1 = evaluate_fitness(func, child1)
            fitness2 = evaluate_fitness(func, child2)
            evaluations += 2

            new_population.extend([child1, child2])
            new_fitness.extend([fitness1, fitness2])

        # Update population
        population = np.array(new_population[:population_size])
        fitness = np.array(new_fitness[:population_size])

        # Update best solution
        current_best_idx = np.argmax(fitness)
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()

        fitness_history.append(best_fitness)

        # Early stopping for finite optima
        if not np.isinf(optimum) and best_fitness >= optimum:
            break

    func.reset()
    return best_fitness, best_individual, fitness_history


def run_ga_experiment(func, func_name, runs, budget):
    """Run GA multiple times and create convergence plots"""
    print(f"Running GA on {func_name}...")
    results = []
    all_histories = []

    for run in range(runs):
        print(f"  Run {run + 1}/{runs}")
        best_fitness, _, fitness_history = ga(func, budget=budget)
        results.append(best_fitness)
        all_histories.append(fitness_history)
        print(f"    Best fitness: {best_fitness}")

    # Create convergence plot
    plt.figure(figsize=(10, 6))

    # Plot all runs
    for history in all_histories:
        plt.plot(history, alpha=0.3, color="lightblue")

    # Plot mean convergence
    max_gens = max(len(h) for h in all_histories)
    mean_history = [
        np.mean([h[gen] for h in all_histories if gen < len(h)])
        for gen in range(max_gens)
    ]

    plt.plot(mean_history, color="red", linewidth=2, label="Mean")
    plt.plot(
        all_histories[np.argmax(results)], color="green", linewidth=2, label="Best Run"
    )

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"{func_name} Convergence - Improved GA")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{func_name.lower()}_convergence.png", dpi=150, bbox_inches="tight")

    return results


def main():
    """Run GA on all required functions"""
    problems = {
        "F1": ioh.get_problem(
            1, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO
        ),
        "F2": ioh.get_problem(
            2, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO
        ),
        "F3": ioh.get_problem(
            3, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO
        ),
        "F18": ioh.get_problem(
            18, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO
        ),
        "F23": ioh.get_problem(
            23, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO
        ),
        "F24": ioh.get_problem(
            24, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO
        ),
        "F25": ioh.get_problem(
            25, instance=1, dimension=100, problem_class=ioh.ProblemClass.PBO
        ),
    }

    all_results = {}

    for func_name, func in problems.items():
        print(f"\n{'='*50}")
        print(f"Testing GA on {func_name}")
        print(f"{'='*50}")

        results = run_ga_experiment(func, func_name, runs=10, budget=100000)
        all_results[func_name] = results

        # Print summary
        mean_fitness = np.mean(results)
        std_fitness = np.std(results)
        best_fitness = np.max(results)
        worst_fitness = np.min(results)

        print(f"\n{func_name} Results Summary:")
        print(f"  Mean fitness: {mean_fitness:.4f}")
        print(f"  Std fitness:  {std_fitness:.4f}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Worst fitness: {worst_fitness:.4f}")
        print(f"  Optimum: {func.optimum.y}")

    # Save results
    with open("ga_results.txt", "w") as f:
        f.write("GA Results - Exercise 3\n")
        f.write("=" * 50 + "\n\n")

        for func_name, results in all_results.items():
            f.write(f"{func_name} Results:\n")
            f.write(f"  Mean: {np.mean(results):.4f}\n")
            f.write(f"  Std:  {np.std(results):.4f}\n")
            f.write(f"  Best: {np.max(results):.4f}\n")
            f.write(f"  Worst: {np.min(results):.4f}\n")
            f.write(f"  All runs: {results}\n\n")


if __name__ == "__main__":
    main()
