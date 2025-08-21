from tsp import TSP
from individual_population import Individual, Population
from crossovers import Crossovers
from mutations import Mutations
from selection import Selection
import random
import numpy as np


class SimpleEvolutionaryAlgorithm:
    """EA Class"""
    def __init__(self, tsp_file, population_size=50, generations=1000):
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.generations = generations
        self.population = None
        
        # Parameters (easy to modify)
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.elitism = 2  # Keep best 2 individuals
        
    def create_initial_population(self):
        """Create random starting population"""
        self.population = Population.random(self.tsp, self.population_size)
        
    def select_parents(self):
        """Select parents using tournament selection"""
        # Simple tournament selection without using Selection class
        parents = []
        
        for _ in range(self.population_size):
            # Pick random individuals for tournament
            tournament_size = 3
            tournament = random.sample(self.population.individuals, tournament_size)
            
            # Find best in tournament (lowest fitness = best tour)
            winner = min(tournament, key=lambda x: x.fitness)
            parents.append(winner)
            
        return parents
    
    def create_offspring(self, parents):
        """Create offspring through crossover and mutation"""
        offspring = []
        
        # Process parents in pairs
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1_tour, child2_tour = Crossovers.order_crossover(
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
                    child1.tour, _, _ = Mutations.swap(child1.tour)
                    child1.fitness = child1.evaluate()
                    
                if random.random() < self.mutation_rate:
                    child2.tour, _, _ = Mutations.swap(child2.tour)
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
        
        # Create modified EA class for this test
        class TestEA(SimpleEvolutionaryAlgorithm):
            def create_offspring(self, parents):
                offspring = []
                crossover_func = getattr(Crossovers, crossover)
                mutation_func = getattr(Mutations, mutation)
                
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        parent1 = parents[i]
                        parent2 = parents[i + 1]
                        
                        if random.random() < self.crossover_rate:
                            child1_tour, child2_tour = crossover_func(
                                parent1.tour, parent2.tour
                            )
                        else:
                            child1_tour = parent1.tour[:]
                            child2_tour = parent2.tour[:]
                        
                        child1 = Individual(self.tsp, child1_tour)
                        child2 = Individual(self.tsp, child2_tour)
                        
                        if random.random() < self.mutation_rate:
                            child1.tour, _, _ = mutation_func(child1.tour)
                            child1.fitness = child1.evaluate()
                            
                        if random.random() < self.mutation_rate:
                            child2.tour, _, _ = mutation_func(child2.tour)
                            child2.fitness = child2.evaluate()
                        
                        offspring.extend([child1, child2])
                
                return offspring
        
        # Run test
        ea = TestEA("tsplib/eil51.tsp", population_size=50, generations=1000)
        best = ea.run(print_progress=False)
        results.append((name, best.fitness))
        print(f"Result: {best.fitness:.2f}")
    
    # Summary
    print(f"\nSummary:")
    for name, fitness in results:
        print(f"{name:20s}: {fitness:.2f}")
    
    best_config = min(results, key=lambda x: x[1])
    print(f"\nBest configuration: {best_config[0]} with {best_config[1]:.2f}")


# Example usage and testing
if __name__ == "__main__":
    print("Simple Evolutionary Algorithm:")
    print("=" * 60)
    
    # Basic test
    print("1. Basic Test")
    ea = SimpleEvolutionaryAlgorithm(
        tsp_file="tsplib/eil51.tsp",
        population_size=50,
        generations=1000
    )
    
    best = ea.run()
    print(f"Best tour found: {best.fitness:.2f}")
    
    # Test different operators
    print(f"\n2. Operator Comparison")
    test_different_operators()
