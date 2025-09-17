import numpy as np
import ioh
import random
import matplotlib.pyplot as plt

class Population:
    def __init__(self, func, population_size=20):
        self.func = func
        self.population_size = population_size
        self.population = []
        self.fitness = []
        
    def initialize(self):
        """Initialize population with random individuals"""
        self.population = []
        self.fitness = []
        for i in range(self.population_size):
            x = np.random.randint(2, size=self.func.meta_data.n_variables)
            self.population.append(x)
            fitness_val = self.func(x)
            # Ensure fitness is a scalar value
            if hasattr(fitness_val, 'item'):
                fitness_val = fitness_val.item()
            self.fitness.append(fitness_val)
    
    def uniform_crossover(self, parent1, parent2):
        """Uniform crossover - each bit comes from either parent with 50% probability"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        
        return child1, child2
    
    def mutation(self, individual, mutation_rate=0.01):
        """Bit-flip mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated
    
    def tournament_selection(self, tournament_size=3):
        """Tournament selection"""
        tournament_indices = random.sample(range(self.population_size), tournament_size)
        tournament_fitness = [self.fitness[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]
    
    def get_best_individual(self):
        """Get the best individual and its fitness"""
        best_idx = np.argmax(self.fitness)
        best_individual = self.population[best_idx]
        best_fitness = self.fitness[best_idx]
        # Ensure fitness is a scalar value
        if hasattr(best_fitness, 'item'):
            best_fitness = best_fitness.item()
        return best_fitness, best_individual

def ga(func, budget, population_size=20, mutation_rate=0.01):
    """
    Genetic Algorithm with uniform crossover and mutation
    """
    pop = Population(func, population_size)
    pop.initialize()
    
    evaluations = population_size  # Initial population evaluation
    best_fitness, best_individual = pop.get_best_individual()
    optimum = func.optimum.y
    # Ensure optimum is a scalar value
    if hasattr(optimum, 'item'):
        optimum = optimum.item()
    
    generation = 0
    
    while evaluations < budget:
        generation += 1
        new_population = []
        new_fitness = []
        
        # Generate offspring
        for _ in range(population_size // 2):
            # Selection
            parent1 = pop.tournament_selection()
            parent2 = pop.tournament_selection()
            
            # Crossover
            child1, child2 = pop.uniform_crossover(parent1, parent2)
            
            # Mutation
            child1 = pop.mutation(child1, mutation_rate)
            child2 = pop.mutation(child2, mutation_rate)
            
            # Evaluate offspring
            fitness1 = func(child1)
            fitness2 = func(child2)
            # Ensure fitness values are scalars
            if hasattr(fitness1, 'item'):
                fitness1 = fitness1.item()
            if hasattr(fitness2, 'item'):
                fitness2 = fitness2.item()
            evaluations += 2
            
            new_population.extend([child1, child2])
            new_fitness.extend([fitness1, fitness2])
        
        # Replace population (generational replacement)
        pop.population = new_population
        pop.fitness = new_fitness
        
        # Update best
        current_best_fitness, current_best_individual = pop.get_best_individual()
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
        
        # Early stopping if optimum found
        if best_fitness >= optimum:
            break
    
    func.reset()
    return best_fitness, best_individual

def run_ga_experiment(func, func_name, runs, budget):
    """
    Run GA multiple times on a function and record results
    """
    print(f"Running GA on {func_name}...")
    results = []
    
    for run in range(runs):
        print(f"  Run {run + 1}/{runs}")
        best_fitness, best_individual = ga(func, budget=budget)
        results.append(best_fitness)
        print(f"    Best fitness: {best_fitness}")
    
    
    return results

def main():
    """
    Run GA on all required functions and record results
    """
    # Problem setup - using PBO (Pseudo-Boolean Optimization) versions
    problems = {
        'F1': ioh.get_problem(1, instance = 1, dimension = 100, problem_class=ioh.ProblemClass.PBO),
        'F2': ioh.get_problem(2, instance = 1, dimension = 100, problem_class=ioh.ProblemClass.PBO), 
        'F3': ioh.get_problem(3, instance = 1, dimension = 100, problem_class=ioh.ProblemClass.PBO),
        'F18': ioh.get_problem(18, instance = 1, dimension = 100, problem_class=ioh.ProblemClass.PBO),
        'F23': ioh.get_problem(23, instance = 1, dimension = 100, problem_class=ioh.ProblemClass.PBO),
        'F24': ioh.get_problem(24, instance = 1, dimension = 100, problem_class=ioh.ProblemClass.PBO),
        'F25': ioh.get_problem(25, instance = 1, dimension = 100, problem_class=ioh.ProblemClass.PBO)
    }
    
    # Run experiments
    all_results = {}
    
    for func_name, func in problems.items():
        print(f"\n{'='*50}")
        print(f"Testing GA on {func_name}")
        print(f"{'='*50}")
        
        results = run_ga_experiment(func, func_name, runs=10, budget=100000)
        all_results[func_name] = results
        
        # Print summary statistics
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
    
    # Save results to file
    print(f"\n{'='*50}")
    print("Saving results to file...")
    
    with open('ga_results.txt', 'w') as f:
        f.write("GA Results - Exercise 3\n")
        f.write("="*50 + "\n\n")
        
        for func_name, results in all_results.items():
            f.write(f"{func_name} Results:\n")
            f.write(f"  Mean: {np.mean(results):.4f}\n")
            f.write(f"  Std:  {np.std(results):.4f}\n")
            f.write(f"  Best: {np.max(results):.4f}\n")
            f.write(f"  Worst: {np.min(results):.4f}\n")
            f.write(f"  All runs: {results}\n\n")

if __name__ == "__main__":
    main()