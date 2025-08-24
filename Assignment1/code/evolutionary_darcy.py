"""
Simple Evolutionary Algorithm for TSP Assignment
Modular design allowing easy configuration of operators and parameters
"""

from tsp import TSP
from individual_population import Individual, Population
from crossovers import Crossovers
from mutations import Mutations
from selection import Selection
import random
import numpy as np


class SimpleEvolutionaryAlgorithm:
    """
    Simple EA for TSP that can be easily configured and understood
    """
    
    def __init__(self, tsp_file, population_size=50, generations=1000, crossover_rate=0.8, mutation_rate=0.1, tournament_size=3, elitism_count=2, selection_method="tournament", crossover_method="order_crossover", mutation_method="swap"):
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.generations = generations
        self.population = None
        
        # Parameters (easy to modify)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism_count
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        
    def create_initial_population(self):
        """Create random starting population"""
        self.population = Population.random(self.tsp, self.population_size)
        
    def select_parents(self):
        """Select parents using configured selection method"""
        parents = []
        
        if self.selection_method == "tournament":
            # Use Selection class for tournament selection
            fitness_values = np.array([ind.fitness for ind in self.population.individuals])
            selected_indices = Selection.Tournament_Selection(
                fitness_values, self.population_size, self.tournament_size
            )
            parents = [self.population.individuals[i] for i in selected_indices]
            
        elif self.selection_method == "fitness_proportional":
            # Use Selection class for fitness proportional selection
            # Convert Individual objects to tour arrays for the Selection class
            parent_tours = np.array([ind.tour for ind in self.population.individuals])
            selected_indices = Selection.Fitness_Proportional_Selection(
                self.tsp, parent_tours
            )
            parents = [self.population.individuals[i] for i in selected_indices]
            
        else:
            # Default to simple tournament selection
            for _ in range(self.population_size):
                tournament = random.sample(self.population.individuals, min(self.tournament_size, len(self.population.individuals)))
                winner = min(tournament, key=lambda x: x.fitness)
                parents.append(winner)
                
        return parents
    
    def create_offspring(self, parents):
        """Create offspring through crossover and mutation"""
        offspring = []
        
        # Get crossover and mutation functions
        crossover_func = getattr(Crossovers, self.crossover_method)
        mutation_func = getattr(Mutations, self.mutation_method)
        
        # Process parents in pairs
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    # Handle edge recombination differently (returns single child)
                    if self.crossover_method == "edge_recombination":
                        child1_tour = crossover_func(parent1.tour, parent2.tour)
                        # Create second child by applying mutation to first
                        child2_tour, _, _ = mutation_func(child1_tour)
                    else:
                        # Normal crossovers return two children
                        child1_tour, child2_tour = crossover_func(
                            parent1.tour, parent2.tour
                        )
                else:
                    # No crossover - copy parents
                    child1_tour = parent1.tour[:]
                    child2_tour = parent2.tour[:]
                
                # Create children
                child1 = Individual(self.tsp, child1_tour)
                child2 = Individual(self.tsp, child2_tour)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1.tour, _, _ = mutation_func(child1.tour)
                    child1.fitness = child1.evaluate()
                    
                if random.random() < self.mutation_rate:
                    child2.tour, _, _ = mutation_func(child2.tour)
                    child2.fitness = child2.evaluate()
                
                offspring.extend([child1, child2])
        
        return offspring
    
    def create_next_generation(self, parents, offspring):
        """Combine parents and offspring to create next generation"""
        # Elitism - keep best individuals from current population
        all_individuals = parents + offspring
        all_individuals.sort(key=lambda x: x.fitness)
        
        # Take best individuals up to population size
        next_gen = Population.empty(self.tsp)
        for i in range(min(self.population_size, len(all_individuals))):
            next_gen.add(all_individuals[i])
            
        # Fill remaining spots with random individuals if needed
        while len(next_gen.individuals) < self.population_size:
            next_gen.add(Individual.random(self.tsp))
            
        return next_gen
    
    def get_best_individual(self):
        """Return best individual in current population"""
        _, best = self.population.best()
        return best
    
    def run(self, print_progress=True):
        """Run the evolutionary algorithm"""
        if print_progress:
            print(f"Starting EA on {len(self.tsp.location_ids)} city TSP")
            print(f"Population: {self.population_size}, Generations: {self.generations}")
            print(f"Crossover: {self.crossover_method}, Mutation: {self.mutation_method}")
            print(f"Selection: {self.selection_method}")
        
        # Initialize
        self.create_initial_population()
        
        # Evolution loop
        for generation in range(self.generations):
            # Selection
            parents = self.select_parents()
            
            # Create offspring
            offspring = self.create_offspring(parents)
            
            # Next generation
            self.population = self.create_next_generation(
                self.population.individuals, offspring
            )
            
            # Print progress
            if print_progress and generation % 200 == 0:
                best = self.get_best_individual()
                print(f"Generation {generation}: Best = {best.fitness:.2f}")
        
        # Final result
        best_individual = self.get_best_individual()
        if print_progress:
            print(f"Final best: {best_individual.fitness:.2f}")
        
        return best_individual


def test_different_operators():
    """Test the EA with different operator combinations"""
    print("Testing Different Operator Combinations")
    print("=" * 50)
    
    # Test configurations
    configs = [
        ("Order + Swap", "order_crossover", "swap"),
        ("PMX + Insert", "pmx_crossover", "insert"),
        ("Cycle + Inversion", "cycle_crossover", "inversion")
    ]
    
    results = []
    
    for name, crossover, mutation in configs:
        print(f"\nTesting {name}...")
        
        # Create EA with specified operators
        ea = SimpleEvolutionaryAlgorithm(
            "tsplib/eil51.tsp", 
            population_size=50, 
            generations=1000,
            crossover_method=crossover,
            mutation_method=mutation
        )
        
        best = ea.run(print_progress=False)
        results.append((name, best.fitness))
        print(f"Result: {best.fitness:.2f}")
    
    # Summary
    print(f"\nSummary:")
    for name, fitness in results:
        print(f"{name:20s}: {fitness:.2f}")
    
    best_config = min(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best_config[0]} with {best_config[1]:.2f}")


def test_selection_methods():
    """Test different selection methods using team member's Selection class"""
    print("Testing Different Selection Methods")
    print("=" * 50)
    
    selection_methods = ["tournament", "fitness_proportional"]
    results = []
    
    for method in selection_methods:
        print(f"Testing {method} selection...")
        
        ea = SimpleEvolutionaryAlgorithm(
            tsp_file="tsplib/eil51.tsp",
            population_size=50,
            generations=1000,
            selection_method=method,
            crossover_method="pmx_crossover",
            mutation_method="inversion"
        )
        
        best = ea.run(print_progress=False)
        results.append((method, best.fitness))
        print(f"  Result: {best.fitness:.2f}")
    
    print(f"\nSelection Method Summary:")
    for method, fitness in results:
        print(f"{method:20s}: {fitness:.2f}")


def comprehensive_algorithm_test():
    """Comprehensive test using all team member's components"""
    print("Comprehensive Algorithm Test")
    print("=" * 50)
    print("Testing all combinations of crossover, mutation, and selection")
    
    crossovers = ["order_crossover", "pmx_crossover", "cycle_crossover"]
    mutations = ["swap", "insert", "inversion"]
    selections = ["tournament", "fitness_proportional"]
    
    results = []
    test_count = 0
    
    for crossover in crossovers:
        for mutation in mutations:
            for selection in selections:
                test_count += 1
                config_name = f"{crossover.split('_')[0].title()}+{mutation.title()}+{selection.split('_')[0].title()}"
                
                print(f"Test {test_count}: {config_name}")
                
                ea = SimpleEvolutionaryAlgorithm(
                    tsp_file="tsplib/eil51.tsp",
                    population_size=30,
                    generations=500,
                    crossover_rate=0.8,
                    mutation_rate=0.1,
                    tournament_size=3,
                    selection_method=selection,
                    crossover_method=crossover,
                    mutation_method=mutation
                )
                
                best = ea.run(print_progress=False)
                results.append((config_name, best.fitness, crossover, mutation, selection))
                print(f"  Result: {best.fitness:.2f}")
    
    print(f"\nComprehensive Test Results (sorted by performance):")
    print("-" * 60)
    results.sort(key=lambda x: x[1])  # Sort by fitness
    
    for i, (name, fitness, cx, mut, sel) in enumerate(results[:10]):  # Top 10
        print(f"{i+1:2d}. {name:25s}: {fitness:8.2f}")
    
    best = results[0]
    print(f"\nBest combination: {best[0]} with {best[1]:.2f}")
    print(f"   Crossover: {best[2]}")
    print(f"   Mutation:  {best[3]}")
    print(f"   Selection: {best[4]}")


def run_multiple_experiments():
    """Run multiple experiments for statistics (quick demo)"""
    print("\nRunning Multiple Experiments (Demo)")
    print("=" * 40)
    
    runs = 5  # Reduced for demo
    results = []
    
    print(f"Running {runs} experiments...")
    
    for i in range(runs):
        ea = SimpleEvolutionaryAlgorithm(
            "tsplib/eil51.tsp", 
            population_size=50, 
            generations=500  # Reduced for demo
        )
        best = ea.run(print_progress=False)
        results.append(best.fitness)
        print(f"Run {i+1}: {best.fitness:.2f}")
    
    # Statistics
    avg = sum(results) / len(results)
    best_result = min(results)
    worst = max(results)
    
    print(f"\nStatistics over {runs} runs:")
    print(f"Average: {avg:.2f}")
    print(f"Best:    {best_result:.2f}")
    print(f"Worst:   {worst:.2f}")


def test_population_sizes_and_generations():
    """Exercise 6 Part 2: Test different population sizes and generations"""
    print("\nExercise 6 - Population Sizes and Generations Test")
    print("=" * 60)
    
    # TSP instances as specified in assignment
    tsp_instances = [
        "tsplib/eil51.tsp", "tsplib/eil76.tsp", "tsplib/eil101.tsp", 
        "tsplib/st70.tsp", "tsplib/kroa100.tsp", "tsplib/kroc100.tsp", 
        "tsplib/krod100.tsp", "tsplib/lin105.tsp", "tsplib/pcb442.tsp", 
        "tsplib/pr2392.tsp", "tsplib/usa13509.tsp"
    ]
    
    population_sizes = [20, 50, 100, 200]
    generation_checkpoints = [2000, 5000, 10000, 20000]
    
    # Three different algorithms as designed
    algorithms = [
        ("Algorithm1_PMX_Inversion_Tournament", "pmx_crossover", "inversion", "tournament"),
        ("Algorithm2_Cycle_Insert_Tournament", "cycle_crossover", "insert", "tournament"),
        ("Algorithm3_Order_Swap_FitnessProp", "order_crossover", "swap", "fitness_proportional")
    ]
    
    results = []
    
    for instance_file in tsp_instances:
        print(f"\nTesting on {instance_file}")
        print("-" * 40)
        
        for alg_name, crossover, mutation, selection in algorithms:
            print(f"Algorithm: {alg_name}")
            
            for pop_size in population_sizes:
                print(f"  Population size: {pop_size}")
                
                # Create EA with current configuration
                ea = SimpleEvolutionaryAlgorithm(
                    tsp_file=instance_file,
                    population_size=pop_size,
                    generations=max(generation_checkpoints),  # Run for maximum generations
                    crossover_method=crossover,
                    mutation_method=mutation,
                    selection_method=selection
                )
                
                # Modified run method to capture results at checkpoints
                ea.create_initial_population()
                
                checkpoint_results = {}
                for generation in range(max(generation_checkpoints)):
                    parents = ea.select_parents()
                    offspring = ea.create_offspring(parents)
                    ea.population = ea.create_next_generation(ea.population.individuals, offspring)
                    
                    # Check if this generation is a checkpoint
                    current_gen = generation + 1
                    if current_gen in generation_checkpoints:
                        best = ea.get_best_individual()
                        checkpoint_results[current_gen] = best.fitness
                        print(f"    Gen {current_gen}: {best.fitness:.2f}")
                
                # Store results
                for gen, fitness in checkpoint_results.items():
                    results.append({
                        'instance': instance_file,
                        'algorithm': alg_name,
                        'population_size': pop_size,
                        'generation': gen,
                        'fitness': fitness
                    })
    
    # Save results to file
    with open('results/population_generation_test.txt', 'w') as f:
        f.write("Instance,Algorithm,PopSize,Generation,Fitness\n")
        for result in results:
            f.write(f"{result['instance']},{result['algorithm']},{result['population_size']},{result['generation']},{result['fitness']:.2f}\n")
    
    print(f"\nResults saved to results/population_generation_test.txt")



def design_three_algorithms():
    """Exercise 6 Part 1: Design and justify three different evolutionary algorithms"""
    
    algorithms = {
        "Algorithm 1 - xxx": {
            "crossover": "pmx_crossover",
            "mutation": "inversion", 
            "selection": "tournament",
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "tournament_size": 3,
            "justification": "PMX preserves position info well, inversion maintains adjacency, tournament provides good selection pressure"
        },
        "Algorithm 2 - xxx": {
            "crossover": "cycle_crossover",
            "mutation": "insert",       
            "selection": "tournament",
            "crossover_rate": 0.75,
            "mutation_rate": 0.2,
            "tournament_size": 4,
            "justification": "Cycle crossover maintains absolute positions, insert mutation is less disruptive, tournament with size 4 increases selection pressure"
        },
        
        "Algorithm 3 - Balanced": {
            "crossover": "order_crossover",
            "mutation": "swap",
            "selection": "fitness_proportional", 
            "crossover_rate": 0.7,
            "mutation_rate": 0.15,
            "tournament_size": 2,
            "justification": "Order crossover preserves relative order, swap is gentle mutation, fitness proportional allows diversity"
        }
        
    }
    
    return algorithms

import concurrent.futures # Working with parallel processing - able to make use of multiple CPU cores

# Best algorithm configuration 
def single_run(instance_file):
    ea = SimpleEvolutionaryAlgorithm(
        tsp_file=instance_file,
        crossover_method = "order_crossover",
        mutation_method = "swap",
        selection_method = "fitness_proportional", 
        crossover_rate = 0.7,
        mutation_rate= 0.15,
        tournament_size= 2,
        population_size=50,
        generations=20000
    )
    best = ea.run(print_progress=False)
    return best.fitness

def run_best_algorithm_30_times():
    """Exercise 6 Part 3: Run best algorithm 30 times for statistics"""
    print("\nExercise 6 - Best Algorithm 30 Runs Test")
    print("=" * 50)
    
    # TSP instances as specified
    tsp_instances = [
        "tsplib/eil51.tsp", "tsplib/eil76.tsp", "tsplib/eil101.tsp", 
        "tsplib/st70.tsp", "tsplib/kroa100.tsp", "tsplib/kroc100.tsp", 
        "tsplib/krod100.tsp", "tsplib/lin105.tsp", "tsplib/pcb442.tsp", 
        "tsplib/pr2392.tsp", "tsplib/usa13509.tsp"
    ]
    
    
    
    all_results = []
    
    for instance_file in tsp_instances:
        print(f"\nTesting {instance_file} - 30 runs")
        print("-" * 30)
        
        # Parallelize the 30 runs
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(single_run, instance_file) for _ in range(30)]
            instance_results = [f.result() for f in concurrent.futures.as_completed(futures)]
            instance_results.sort()  # Optional: sort for reporting
        
        for i, fitness in enumerate(instance_results):
            print(f"Run {i+1}: {fitness:.2f}")
        
        avg_fitness = np.mean(instance_results)
        std_fitness = np.std(instance_results)
        min_fitness = min(instance_results)
        max_fitness = max(instance_results)
        
        print(f"Average: {avg_fitness:.2f} ± {std_fitness:.2f}")
        print(f"Best: {min_fitness:.2f}, Worst: {max_fitness:.2f}")
        
        all_results.append({
            'instance': instance_file,
            'average': avg_fitness,
            'std_dev': std_fitness,
            'best': min_fitness,
            'worst': max_fitness,
            'all_runs': instance_results
        })
    
    # Save results to file as required
    with open('results/your_EA.txt', 'w') as f:
        f.write("TSP Instance,Average Cost,Standard Deviation,Best Cost,Worst Cost\n")
        for result in all_results:
            f.write(f"{result['instance']},{result['average']:.2f},{result['std_dev']:.2f},{result['best']:.2f},{result['worst']:.2f}\n")
    
    print(f"\nResults saved to results/your_EA.txt")
    return all_results


# Example usage and testing
if __name__ == "__main__":
    """
    print("Modular Evolutionary Algorithm - Final Version")
    print("=" * 70)
    
    best = ea.run()
    print(f"Basic test result: {best.fitness:.2f}")
    
    # Test different operators
    print(f"\n2. Operator Comparison Test")
    test_different_operators()
    
    # Test selection methods
    print(f"\n3. Selection Method Test")
    test_selection_methods()
    
    # Multiple runs
    print(f"\n5. Multiple Runs")
    run_multiple_experiments()
    """
    # Exercise 6 - Complete Assignment Requirements
    print(f"\n" + "="*70)
    print("EXERCISE 6 - EVOLUTIONARY ALGORITHMS AND BENCHMARKING")
    print("="*70)
    
    # Part 1: Design three algorithms
    print("\nPart 1: Designing Three Different Algorithms")
   # algorithms = design_three_algorithms()
    
    # Part 2: Test with different population sizes and generations
    print("\nPart 2: Testing Population Sizes and Generations")
    print("This will take a very long time to complete!")
    
    #test_population_sizes_and_generations()
    
    # Part 3: Run best algorithm 30 times
    print("\nPart 3: Running Best Algorithm 30 Times")
    print("This will also take a long time - comment out if needed")
    
    run_best_algorithm_30_times()
    
