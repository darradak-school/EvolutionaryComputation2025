# Basic demo for Evolutionary Algorithm
from evolutionary_tyler import EvolutionaryAlgorithm
import time

### CONFIGURATION VARIABLES ###
PROBLEMS = [
    "eil51",
    "st70",
    "eil76",
    "kroA100",
    "kroC100",
    "kroD100",
    "eil101",
    "lin105",
    "pcb442",
    "pr2392",
    "usa13509",
]
POPULATION = 200  # Number of individuals in the population.
GENERATIONS = 20000  # Number of generations to run.
MUTATION = 0.2  # Probability of a mutation occuring.
CROSSOVER = 0.9  # Probability of a crossover occuring.
TOURNAMENT = 5  # Number of parents to select for tournament selection.
REPLACEMENT = 0.05  # Percentage of the population to replace with offspring.

with open("../results/murad_ea_results.txt", "a") as f:
    f.write("######### EVOLUTIONARY ALGORITHM #########\n")
    f.write("Results from running the EA on TSP problems\n")
    f.write(f"Running with population {POPULATION}.\n\n")

    # Create the ea
    for problem in PROBLEMS:
        f.write(f"######### {problem} #########\n")

        ea = EvolutionaryAlgorithm(
            f"tsplib/{problem}.tsp",
            POPULATION,
            GENERATIONS,
            MUTATION,
            CROSSOVER,
            TOURNAMENT,
            REPLACEMENT,
        )

        print("=" * 20)
        print(f"Running {problem}...")

        # Track fitness at specific generations.
        fitness_log = {}
        gen_report = [2000, 5000, 10000, 20000]

        # Run evolutionary algorithm.
        start_time = time.time()
        for generation in range(GENERATIONS):
            ea.evolution()

            # Log fitness at target generations.
            if generation + 1 in gen_report:
                _, best = ea.population.best()
                fitness_values = [ind.fitness for ind in ea.population.individuals]
                avg_fitness = sum(fitness_values) / len(fitness_values)
                fitness_log[generation + 1] = (best.fitness, avg_fitness)

            # Print progress to console.
            if (generation+1) % 1000 == 0:
                _, best = ea.population.best()
                fitness_values = [ind.fitness for ind in ea.population.individuals]
                avg_fitness = sum(fitness_values) / len(fitness_values)
                print(
                    f"Generation {generation+1}: Best = {best.fitness:.2f}, Average = {avg_fitness:.2f}, Time elapsed: {time.time() - start_time:.2f}s"
                )

        # Write logged data to file.
        for gen, (best_fitness, avg_fitness) in fitness_log.items():
            f.write(f"===== Generation {gen} =====\n")
            f.write(f"Best fitness: {best_fitness:.2f}\n")
            f.write(f"Average fitness: {avg_fitness:.2f}\n")
            f.write(f"Time elapsed: {time.time() - start_time:.2f}s\n")
