import time
from ea_steadystate import EvolutionaryAlgorithm
from ea_generational import GenerationalElitistEA
from ea_crowding import SteadyStateDeterministicCrowdingEA

### CONFIGURATION ###
# Problems to run.
PROBLEMS = [
    # "eil51",
    # "st70",
    # "eil76",
    # "kroA100",
    # "kroC100",
    # "kroD100",
    # "eil101",
    # "lin105",
    "pcb442",
    # "pr2392",
    # "usa13509",
]

# Arguments for the algorithms.
POPULATION_SIZE = 200
MAX_GENERATIONS = 20000
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.9
STAG_LIMIT = 4000

# Algorithms to run.
ALGORITHMS = [
    (
        "SteadyState (Tourn + OX + Swap)",
        "steadystate",
        lambda tsp: EvolutionaryAlgorithm(
            tsp_file=tsp,
            population_size=POPULATION_SIZE,
            generations=MAX_GENERATIONS,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
        ),
    ),
    (
        "Generational+Elitism (FPS+PMX+INV)",
        "generational",
        lambda tsp: GenerationalElitistEA(
            tsp_file=tsp,
            population_size=POPULATION_SIZE,
            generations=MAX_GENERATIONS,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
        ),
    ),
    (
        "SteadyState+Crowding (Tourn+ERX+INS)",
        "crowding",
        lambda tsp: SteadyStateDeterministicCrowdingEA(
            tsp_file=tsp,
            population_size=POPULATION_SIZE,
            generations=MAX_GENERATIONS,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
        ),
    ),
]


def run():
    # Open results file
    with open("../results/ea_test_results.txt", "a") as results_file:
        results_file.write("Algorithm,Problem,Best_Fitness,Time_Seconds\n")
        
        for problem in PROBLEMS:
            tsp_path = f"tsplib/{problem}.tsp"
            print(f"\n===== Problem: {problem} =====")

            for name, key, factory in ALGORITHMS:
                print(f"\n--- Algorithm: {name} ---")
                algo = factory(tsp_path)
                start_time = time.time()
                
                # Track best fitness for stagnation detection
                best_fitness = float('inf')
                stagnation = 0

                for gen in range(1, MAX_GENERATIONS + 1):
                    algo.step()
                    
                    # Check for improvement
                    _, current_best = algo.best()
                    if current_best.fitness < best_fitness:
                        best_fitness = current_best.fitness
                        stagnation = 0
                    else:
                        stagnation += 1

                    if gen % 1000 == 0 or gen == MAX_GENERATIONS:
                        _, best = algo.best()
                        population = algo.population
                        fitness_values = [ind.fitness for ind in population.individuals]
                        avg_fitness = sum(fitness_values) / len(fitness_values)

                        elapsed = time.time() - start_time
                        print(
                            f"Gen {gen:>5}: Best={best.fitness:.2f}  Avg={avg_fitness:.2f}  Time={elapsed:.2f}s"
                        )

                    # Check stagnation limit
                    if stagnation >= STAG_LIMIT:
                        print(f"Stagnation limit reached.")
                        break

                # Final result
                _, best = algo.best()
                total_time = time.time() - start_time
                print(
                    f">>> DONE: {name} on {problem}: Best={best.fitness:.2f}  Time={total_time:.2f}s  Generations={gen}"
                )
                
                # Write to results file
                results_file.write(f"{name},{problem},{best.fitness:.2f},{total_time:.2f}\n")
                results_file.flush()  # Ensure data is written immediately


if __name__ == "__main__":
    run()
