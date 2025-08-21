# inverover.py - First Implementation of Inver-over Operator for TSP
# Based on paper "Inver-over Operator for the TSP" by Guo Tao and Zbigniew Michalewicz

from tsp import TSP
from individual_population import Individual, Population
import random
import numpy as np
import time
import os


class InverOver:
    """
    Implementation of Inver-over evolutionary algorithm for TSP
    
    Algorithm combines features of inversion (mutation) and crossover, 
    using population-driven adaptive inversions to solve TSP instances
    """
    
    def __init__(self, tsp_file, population_size=100, p=0.02, max_stagnation=10):
        """
        Initialize algorithm
        
        Args:
            tsp_file: Path to TSP problem file
            population_size: Size of population (default: 100 as per paper)
            p: Probability of random inversion (default: 0.02 as per paper)
            max_stagnation: Number of iterations without improvement before termination
        """
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.p = p  # Probability of random city selection
        self.max_stagnation = max_stagnation
        
        # Initialize population with random tours
        self.population = self._initialize_population()
        self.best_tour = None
        self.best_fitness = float('inf')
        self.stagnation_counter = 0
        self.generation = 0
        
    def _initialize_population(self):
        """Create initial population of random tours"""
        population = []
        for _ in range(self.population_size):
            tour = self.tsp.random_tour()
            population.append(tour)
        return population
    
    def _inversion(self, tour, start_idx, end_idx):
        """
        Perform inversion on a tour segment
        
        Args:
            tour: Current tour
            start_idx: Start index of segment to invert (exclusive - starts after this)
            end_idx: End index of segment to invert (inclusive)
            
        Returns:
            New tour with inverted segment
        """
        new_tour = tour.copy()
        n = len(tour)
        
        # Ensure indices are within bounds
        start_idx = start_idx % n
        end_idx = end_idx % n
        
        # Collect the segment to be inverted
        segment = []
        current_idx = (start_idx + 1) % n  # Start after start_idx
        
        # Collect cities in the segment
        while current_idx != (end_idx + 1) % n:
            segment.append(new_tour[current_idx])
            current_idx = (current_idx + 1) % n
        
        # Reverse the segment
        segment.reverse()
        
        # Put the reversed segment back
        current_idx = (start_idx + 1) % n
        for city in segment:
            new_tour[current_idx] = city
            current_idx = (current_idx + 1) % n
            
        return new_tour
    
    def _get_next_city(self, tour, city):
        """Get the city that follows the given city in a tour"""
        try:
            idx = tour.index(city)
            return tour[(idx + 1) % len(tour)]
        except ValueError:
            # If city not found, return a random city
            return random.choice(tour)
    
    def _get_prev_city(self, tour, city):
        """Get the city that precedes the given city in a tour"""
        try:
            idx = tour.index(city)
            return tour[(idx - 1) % len(tour)]
        except ValueError:
            # If city not found, return a random city
            return random.choice(tour)
    
    def _apply_inverover_operator(self, tour):
        """
        Apply the inver-over operator to a single individual
        
        This is the core operator that combines inversion with 
        population-based guidance
        
        Args:
            tour: Current tour to be modified
            
        Returns:
            Modified tour after applying inver-over operator
        """
        n = len(tour)
        S_prime = tour.copy()  # S' in the paper
        
        # Select random city c from S'
        c = random.choice(S_prime)
        
        # Repeat until exit condition
        max_iterations = n * 2  # Safety limit to prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Select city c'
            if random.random() <= self.p:
                # With probability p, select c' randomly from remaining cities
                remaining_cities = [city for city in S_prime if city != c]
                if not remaining_cities:  # Safety check
                    break
                c_prime = random.choice(remaining_cities)
            else:
                # Otherwise, select random individual and get city next to c
                random_individual = random.choice(self.population)
                # Make sure c exists in the random individual
                if c in random_individual:
                    c_prime = self._get_next_city(random_individual, c)
                else:
                    # If c doesn't exist in random individual, select randomly
                    remaining_cities = [city for city in S_prime if city != c]
                    if not remaining_cities:
                        break
                    c_prime = random.choice(remaining_cities)
            
            # Check if c_prime is in current tour (safety check)
            if c_prime not in S_prime:
                # This shouldn't happen, but if it does, select a valid city
                remaining_cities = [city for city in S_prime if city != c]
                if not remaining_cities:
                    break
                c_prime = random.choice(remaining_cities)
            
            # Check exit condition: if c' is next or previous to c in S'
            c_idx = S_prime.index(c)
            c_prime_idx = S_prime.index(c_prime)
            
            next_idx = (c_idx + 1) % n
            prev_idx = (c_idx - 1) % n
            
            if c_prime_idx == next_idx or c_prime_idx == prev_idx:
                break  # Exit from repeat loop
            
            # Perform inversion from the position after c to position of c'
            # This inverts the segment (c+1, c+2, ..., c')
            S_prime = self._inversion(S_prime, c_idx, c_prime_idx)
            
            # Update c to c'
            c = c_prime
        
        return S_prime
    
    def evolve_generation(self):
        """
        Perform one generation of evolution
        
        Each individual competes only with its offspring
        """
        improved = False
        
        for i in range(self.population_size):
            # Get current individual
            current_tour = self.population[i]
            current_fitness = self.tsp.tour_length(current_tour)
            
            # Apply inver-over operator to create offspring
            offspring = self._apply_inverover_operator(current_tour)
            offspring_fitness = self.tsp.tour_length(offspring)
            
            # Replace parent if offspring is better or equal
            if offspring_fitness <= current_fitness:
                self.population[i] = offspring
                
                # Update best solution if necessary
                if offspring_fitness < self.best_fitness:
                    self.best_fitness = offspring_fitness
                    self.best_tour = offspring.copy()
                    improved = True
        
        # Update stagnation counter
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
        self.generation += 1
        
        return improved
    
    def run(self, max_generations=20000, verbose=False):
        """
        Run Inver-over algorithm
        
        Args:
            max_generations: Max number of generations
            verbose: Whether to print information
            
        Returns:
            Tuple of (best_tour, best_fitness, generations_run, time_elapsed)
        """
        start_time = time.time()
        
        # Initialize best solution
        for tour in self.population:
            fitness = self.tsp.tour_length(tour)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_tour = tour.copy()
        
        # Main evolution loop
        while self.generation < max_generations:
            improved = self.evolve_generation()
            
            # Print progress if verbose
            if verbose and self.generation % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Generation {self.generation}: Best = {self.best_fitness:.2f}, "
                      f"Stagnation = {self.stagnation_counter}, Time = {elapsed:.2f}s")
            
            # Check termination condition (stagnation)
            if self.stagnation_counter >= self.max_stagnation:
                if verbose:
                    print(f"Terminated due to stagnation at generation {self.generation}")
                break
        
        elapsed_time = time.time() - start_time
        
        return self.best_tour, self.best_fitness, self.generation, elapsed_time
    
    def get_population_stats(self):
        """Get statistics about current population."""
        fitness_values = [self.tsp.tour_length(tour) for tour in self.population]
        return {
            'best': min(fitness_values),
            'worst': max(fitness_values),
            'average': np.mean(fitness_values),
            'std': np.std(fitness_values)
        }


def run_inverover_experiments():
    """
    Full test run experiments with Inver-over algorithm on TSPlib instances.
    """
    
    # Test instances as specified in assignment
    test_instances = [
        "eil51", "eil76", "eil101", "st70", 
        "kroA100", "kroC100", "kroD100", "lin105",
        "pcb442", "pr2392", "usa13509"
    ]
    
    # Parameters from research paper
    POPULATION_SIZE = 50
    P = 0.02
    MAX_GENERATIONS = 20000
    RUNS_PER_INSTANCE = 30
    
    results = {}
    
    print("=" * 60)
    print("INVER-OVER ALGORITHM EXPERIMENTS")
    print("=" * 60)
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Random Inversion Probability (p): {P}")
    print(f"Max Generations: {MAX_GENERATIONS}")
    print(f"Runs per Instance: {RUNS_PER_INSTANCE}")
    print("=" * 60)
    
    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)
    
    # Open results file
    with open("../results/inverover.txt", "w") as f:
        f.write("INVER-OVER ALGORITHM RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Population Size: {POPULATION_SIZE}\n")
        f.write(f"Random Inversion Probability (p): {P}\n")
        f.write(f"Max Generations: {MAX_GENERATIONS}\n")
        f.write(f"Runs per Instance: {RUNS_PER_INSTANCE}\n")
        f.write("=" * 60 + "\n\n")
        
        for instance in test_instances:
            print(f"\nProcessing {instance}...")
            instance_results = []
            
            try:
                # Run multiple times for each instance
                for run in range(RUNS_PER_INSTANCE):
                    # Create algorithm instance
                    algo = InverOver(
                        f"tsplib/{instance}.tsp",
                        population_size=POPULATION_SIZE,
                        p=P,
                        max_stagnation=10
                    )
                    
                    # Run algorithm
                    best_tour, best_fitness, generations, elapsed_time = algo.run(
                        max_generations=MAX_GENERATIONS,
                        verbose=(run == 0)  # Only verbose for first run
                    )
                    
                    instance_results.append(best_fitness)
                    
                    if run == 0:
                        print(f"  First run: fitness = {best_fitness:.2f}, "
                              f"generations = {generations}, time = {elapsed_time:.2f}s")
                
                # Calculate statistics
                avg_fitness = np.mean(instance_results)
                std_fitness = np.std(instance_results)
                min_fitness = np.min(instance_results)
                max_fitness = np.max(instance_results)
                
                results[instance] = {
                    'average': avg_fitness,
                    'std': std_fitness,
                    'min': min_fitness,
                    'max': max_fitness,
                    'all_results': instance_results
                }
                
                # Write to file
                f.write(f"Instance: {instance}\n")
                f.write(f"  Average: {avg_fitness:.2f}\n")
                f.write(f"  Std Dev: {std_fitness:.2f}\n")
                f.write(f"  Min: {min_fitness:.2f}\n")
                f.write(f"  Max: {max_fitness:.2f}\n")
                f.write("-" * 40 + "\n")
                
                print(f"  Results: avg = {avg_fitness:.2f} ± {std_fitness:.2f}, "
                      f"min = {min_fitness:.2f}, max = {max_fitness:.2f}")
                
            except Exception as e:
                print(f"  Error processing {instance}: {e}")
                f.write(f"Instance: {instance} - ERROR: {e}\n")
                f.write("-" * 40 + "\n")
    
    print("\n" + "=" * 60)
    print("Experiments completed. Results saved to ../results/inverover.txt")
    
    return results

if __name__ == "__main__":

    run_inverover_experiments()