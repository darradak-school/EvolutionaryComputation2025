import time
import statistics
from ea_steadystate import SteadyStateEA
from ea_generational import GenerationalEA
from ea_crowding import CrowdingEA

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
POPULATION_SIZE = 50
MAX_GENERATIONS = 20000
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.9
STAG_LIMIT = 4000

# Algorithms to run.
ALGORITHMS = [
    (
        "SteadyState (Tourn + OX + Swap + Elite5)",
        "steadystate",
        lambda tsp: SteadyStateEA(
            tsp_file=tsp,
            population_size=POPULATION_SIZE,
            generations=MAX_GENERATIONS,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
        ),
    ),
    (
        "Generational (FPS + PMX + Inversion)",
        "generational",
        lambda tsp: GenerationalEA(
            tsp_file=tsp,
            population_size=POPULATION_SIZE,
            generations=MAX_GENERATIONS,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
        ),
    ),
    (
        "Crowding (Tourn + ERX + Insert)",
        "crowding",
        lambda tsp: CrowdingEA(
            tsp_file=tsp,
            population_size=POPULATION_SIZE,
            generations=MAX_GENERATIONS,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
        ),
    ),
]


def run():
    # Store results for all problems to write summary later
    all_results = {}
    
    for problem in PROBLEMS:
        tsp_path = f"tsplib/{problem}.tsp"
        print(f"\n===== Problem: {problem} =====")

        for name, key, factory in ALGORITHMS:
            print(f"\n--- Algorithm: {name} ---")
            
            # Run algorithm 30 times
            fitness_results = []
            time_results = []
            
            for run_num in range(3):
                print(f"  Run {run_num + 1}", end=" ")
                
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

                    # Check stagnation limit
                    if stagnation >= STAG_LIMIT:
                        break

                # Final result
                _, best = algo.best()
                total_time = time.time() - start_time
                
                fitness_results.append(best.fitness)
                time_results.append(total_time)
                
                print(f"Fitness: {best.fitness:.2f}, Time: {total_time:.2f}s")
            
            # Calculate statistics
            avg_fitness = statistics.mean(fitness_results)
            std_fitness = statistics.stdev(fitness_results) if len(fitness_results) > 1 else 0
            avg_time = statistics.mean(time_results)
            std_time = statistics.stdev(time_results) if len(time_results) > 1 else 0
            
            print(f"\n  Summary for {problem}:")
            print(f"    Average Best Fitness: {avg_fitness:.2f} +/- {std_fitness:.2f}")
            print(f"    Average Time: {avg_time:.2f}s +/- {std_time:.2f}s")
            
            # Store results for summary
            if key not in all_results:
                all_results[key] = {}
            all_results[key][problem] = {
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'avg_time': avg_time,
                'std_time': std_time
            }
    
    # Write summary files for each algorithm
    for key, problem_results in all_results.items():
        algorithm_name = next(name for name, k, _ in ALGORITHMS if k == key)
        with open(f"../results/{key}_results.txt", "a") as summary_file:
            # summary_file.write(f"{algorithm_name} Results\n")
            # summary_file.write("=" * 50 + "\n")
            
            for problem in PROBLEMS:
                if problem in problem_results:
                    results = problem_results[problem]
                    # summary_file.write(f"Problem: {problem}\n")
                    # summary_file.write(f"Average Best Fitness: {results['avg_fitness']:.2f} +/- {results['std_fitness']:.2f}\n")
                    # summary_file.write(f"Average Time: {results['avg_time']:.2f}s +/- {results['std_time']:.2f}s\n")
                    # summary_file.write("-" * 30 + "\n\n")


if __name__ == "__main__":
    run()
