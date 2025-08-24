"""
evolutionary.py attempt by Darcy - Final Version 
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
import concurrent.futures # Working with parallel processing - able to make use of multiple CPU cores



class SimpleEvolutionaryAlgorithm:
    """
    Simple EA for TSP that can be easily configured and understood
    """
    
    def __init__(self, tsp_file, population_size=50, generations=1000, crossover_rate=0.8, mutation_rate=0.1, tournament_size=3, elitism_count=2, selection_method="tournament", crossover_method="order_crossover", mutation_method="swap", algorithm_type="generational"):
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.generations = generations
        self.population = None
        
        # Parameters 
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism_count
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        self.algorithm_type = algorithm_type  # "generational" or "steady_state"

        # Steady-state specific parameters
        self.replacement_rate = 0.05  #  steady-state
        self.elite_size = max(1, int(self.population_size * 0.05))  #  steady-state


        
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
            parent_tours = np.array([ind.tour for ind in self.population.individuals])
            selected_indices = Selection.Fitness_Proportional_Selection(
                self.tsp, parent_tours
            )
            parents = [self.population.individuals[i] for i in selected_indices]
            
        else:
            # Default tournament selection
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
    
    def create_next_gen(self, parents, offspring):
        """Combine parents and offspring to create next generation (generational only)"""
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
    
    def steady_state_step(self):
        """Steady-state evolution step (replaces a few individuals per generation)"""
        num_replacements = max(1, int(self.population_size * self.replacement_rate))
        
        # Get parents using configured selection method
        parents = []
        if self.selection_method == "tournament":
            fitness_values = np.array([ind.fitness for ind in self.population.individuals])
            selected_indices = Selection.Tournament_Selection(
                fitness_values, num_replacements, self.tournament_size
            )
            parents = [self.population.individuals[i] for i in selected_indices]
        else:
            # Fallback to simple tournament
            for _ in range(num_replacements):
                tournament = random.sample(self.population.individuals, min(self.tournament_size, len(self.population.individuals)))
                winner = min(tournament, key=lambda x: x.fitness)
                parents.append(winner)
        
        # Create offspring
        offspring = []
        crossover_func = getattr(Crossovers, self.crossover_method)
        mutation_func = getattr(Mutations, self.mutation_method)
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    if self.crossover_method == "edge_recombination":
                        child1_tour = crossover_func(parent1.tour, parent2.tour)
                        child2_tour, _, _ = mutation_func(child1_tour)
                    else:
                        child1_tour, child2_tour = crossover_func(parent1.tour, parent2.tour)
                else:
                    child1_tour, child2_tour = parent1.tour[:], parent2.tour[:]
                
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
            else:
                # Handle odd number of parents
                child = Individual(self.tsp, parents[i].tour[:])
                if random.random() < self.mutation_rate:
                    child.tour, _, _ = mutation_func(child.tour)
                    child.fitness = child.evaluate()
                offspring.append(child)
        
        # Replacement with elite preservation
        # Sort population by fitness
        self.population.individuals.sort(key=lambda x: x.fitness)
        offspring.sort(key=lambda x: x.fitness)
        
        # Replace worst individuals with best offspring
        num_offspring = min(len(offspring), num_replacements)
        replaceable_start = self.elite_size
        
        for i in range(num_offspring):
            if replaceable_start + i < len(self.population.individuals):
                self.population.individuals[replaceable_start + i] = offspring[i]
        
        # Keep population sorted
        self.population.individuals.sort(key=lambda x: x.fitness)
    
    def get_best_individual(self):
        """Return best individual in current population"""
        _, best = self.population.best()
        return best
    
    def run(self, print_progress=True):
        """Run the evolutionary algorithm"""
        if print_progress:
            print(f"Starting EA on {len(self.tsp.location_ids)} city TSP")
            print(f"Population: {self.population_size}, Generations: {self.generations}")
            print(f"Algorithm Type: {self.algorithm_type}")
            print(f"Crossover: {self.crossover_method}, Mutation: {self.mutation_method}")
            print(f"Selection: {self.selection_method}")
        
        # Init
        self.create_initial_population()
        
        # Evolution loop - different based on algorithm type
        if self.algorithm_type == "steady_state":
            # Steady-state evolution
            for generation in range(self.generations):
                self.steady_state_step()
                
                # Print progress
                if print_progress and generation % 200 == 0:
                    best = self.get_best_individual()
                    print(f"Generation {generation}: Best = {best.fitness:.2f}")
        else:
            # Generational evolution 
            for generation in range(self.generations):
                # Selection
                parents = self.select_parents()
                
                # Create offspring
                offspring = self.create_offspring(parents)
                
                # Next generation
                self.population = self.create_next_gen(
                    self.population.individuals, offspring
                )
                
                # Print progress
                if print_progress and generation % 200 == 0:
                    best = self.get_best_individual()
                    print(f"Generation {generation}: Best = {best.fitness:.2f}")
        
        # Result
        best_individual = self.get_best_individual()
        if print_progress:
            print(f"Final best: {best_individual.fitness:.2f}")
        
        return best_individual


def basic_test():
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
    
    # Print Summary
    print(f"\nSummary:")
    for name, fitness in results:
        print(f"{name:20s}: {fitness:.2f}")
    
    best_config = min(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best_config[0]} with {best_config[1]:.2f}")



# Exercise 6 stuff

def population_generation_worker(instance_file, algorithms, population_sizes, generation_checkpoints):
    """Worker function for a single TSP instance."""
    results = []
    for alg_name, algorithm_type, crossover, mutation, selection, crossover_rate, mutation_rate, tournament_size in algorithms:
        for pop_size in population_sizes:
            ea = SimpleEvolutionaryAlgorithm(
                tsp_file=instance_file,
                population_size=pop_size,
                generations=max(generation_checkpoints),
                algorithm_type=algorithm_type,
                crossover_method=crossover,
                mutation_method=mutation,
                selection_method=selection,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                tournament_size=tournament_size
            )
            ea.create_initial_population()
            checkpoint_results = {}
            for generation in range(max(generation_checkpoints)):
                parents = ea.select_parents()
                offspring = ea.create_offspring(parents)
                ea.population = ea.create_next_gen(ea.population.individuals, offspring)
                current_gen = generation + 1
                if current_gen in generation_checkpoints:
                    best = ea.get_best_individual()
                    checkpoint_results[current_gen] = best.fitness
            for gen, fitness in checkpoint_results.items():
                results.append({
                    'instance': instance_file,
                    'algorithm': alg_name,
                    'population_size': pop_size,
                    'generation': gen,
                    'fitness': fitness
                })
    return results

def test_popsizes_generations():
    """Exercise 6 Step 2: Test different population sizes and generations (parallelized by TSP instance)"""
    print("\nExercise 6 - Population Sizes and Generations Test")
    print("=" * 60)
    tsp_instances = [
        "tsplib/eil51.tsp", "tsplib/eil76.tsp", "tsplib/eil101.tsp", 
        "tsplib/st70.tsp", "tsplib/kroa100.tsp", "tsplib/kroc100.tsp", 
        "tsplib/krod100.tsp", "tsplib/lin105.tsp", "tsplib/pcb442.tsp", 
        "tsplib/pr2392.tsp", "tsplib/usa13509.tsp"
    ]
    population_sizes = [20, 50, 100, 200]
    generation_checkpoints = [2000, 5000, 10000, 20000]
    algorithms = [
        ("Algorithm1_Generational_PMX_Inversion_Tournament", "generational", "pmx_crossover", "inversion", "tournament", 0.8, 0.1, 3),
        ("Algorithm2_SteadyState_Order_Inversion_Tournament", "steady_state", "order_crossover", "inversion", "tournament", 0.9, 0.05, 5),
        ("Algorithm3_Generational_Order_Swap_FitnessProp", "generational", "order_crossover", "swap", "fitness_proportional", 0.7, 0.15, 2)
    ]
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                population_generation_worker,
                instance_file,
                algorithms,
                population_sizes,
                generation_checkpoints
            )
            for instance_file in tsp_instances
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    # Save results to file
    with open('results/population_generation_test.txt', 'w') as f:
        f.write("Instance,Algorithm,PopSize,Generation,Fitness\n")
        for result in results:
            f.write(f"{result['instance']},{result['algorithm']},{result['population_size']},{result['generation']},{result['fitness']:.2f}\n")
    print(f"\nResults saved to results/population_generation_test.txt")



# Best algorithm configuration 
def single_run(instance_file):
    # Steady-state EA
    ea = SimpleEvolutionaryAlgorithm(
    tsp_file="tsplib/eil51.tsp",
    population_size=50,
    generations=1000,
    algorithm_type="steady_state",    
    crossover_method="order_crossover",
    mutation_method="inversion",
    crossover_rate=0.9,
    mutation_rate=0.05
    )
    best = ea.run(print_progress=False)
    return best.fitness

def best_alg_30_times():
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
        
        # Parallelize 30 runs
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(single_run, instance_file) for _ in range(30)]
            instance_results = [f.result() for f in concurrent.futures.as_completed(futures)]
            instance_results.sort()  
        
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
    
    # Save results to file
    with open('results/your_EA.txt', 'w') as f:
        f.write("TSP Instance,Average Cost,Standard Deviation,Best Cost,Worst Cost\n")
        for result in all_results:
            f.write(f"{result['instance']},{result['average']:.2f},{result['std_dev']:.2f},{result['best']:.2f},{result['worst']:.2f}\n")
    
    print(f"\nResults saved to results/your_EA.txt")
    return all_results


# Example usage and testing
if __name__ == "__main__":
    # Exercise 6 - Complete Assignment Requirements
    print(f"\n" + "="*70)
    print("EXERCISE 6 - EVOLUTIONARY ALGORITHMS AND BENCHMARKING")
    print("="*70)
    
    
    # Part 2: Test with different population sizes and generations
    print("\nPart 2: Testing Population Sizes and Generations")
    print("This will take a very long time to complete!")
    
    test_popsizes_generations()

    
    # Part 3: Run best algorithm 30 times
    print("\nPart 3: Running Best Algorithm 30 Times")
    print("This will also take a long time - comment out if needed")
    
    best_alg_30_times()
    
