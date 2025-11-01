import random
import numpy as np
from Multi_Objective_Fitness_Function import MultiObjectiveFitness

class GSEMO:
    def __init__(self, problem_id, dimension, budget=10000):
        """
        Initialize GSEMO
        Args:
            problem_id: IOH problem ID (e.g., 2100)
            dimension: length of the solution (number of nodes)
            budget: budget for fitness evaluations
            """
        self.fitness_func = MultiObjectiveFitness(problem_id)
        self.dimension = dimension
        self.budget = budget
        self.population = []
        self.evaluations = 0

    def initialize(self):
        """
        Initial population with one Random solution
        """
        # Create an initial solution (all zeros)
        initial_solution = [0] * self.dimension
        # Evaluate initial solution
        f1, f2 = self.fitness_func.evaluate(initial_solution)
        self.evaluations += 1

        # Add population
        self.population.append((initial_solution.copy(), (f1, f2)))

    def mutate(self, solution):
        """
        Standard bit-flip mutation
        Each bit has a 1/n chance to be flipped.

        Args:
            solution: binary list

        Returns:
            mutated solution (copy)
        """
        offspring = solution.copy()
        mutation_rate = 1.0 / self.dimension

        for i in range(self.dimension):
            if random.random() < mutation_rate:
                offspring[i] = 1 - offspring[i]  # flip bit

        return offspring
    
    def is_dominated(self, obj1, obj2):
        """
        Check if obj1 is dominated by obj2
        Returns:
            True if obj2 dominates obj1
        """
        return self.fitness_func.dominates(obj2, obj1)
    
    def update_population(self, offspring, offspring_obj):
        """
        Update population with new offspring
        Args:
            offspring: solution
            offspring_obj: (f1, f2) tuple
        """
        # Check Offspring is dominated by any in population
        is_dominated_by_any = False
        for sol, obj in self.population:
            if self.is_dominated(offspring_obj, obj):
                is_dominated_by_any = True
                break

        # if Offspring is not dominated by any in population, add it to population
        if not is_dominated_by_any:
            # Remove all solutions dominated by offspring
            self.population = [
                (sol, obj) for sol, obj in self.population
                if not self.is_dominated(obj, offspring_obj)
            ]

            # add offspring
            self.population.append((offspring, offspring_obj))

    def run(self):
        """
        Run GSEMO algorithm

        Returns:
            population: the lastest Pareto front
        """
        # Initialize population
        self.initialize()

        # Main loop
        while self.evaluations < self.budget:
            # 1. parent
            parent, parent_obj = random.choice(self.population)

            # 2. Mutate
            offspring = self.mutate(parent)

            # 3. Evaluation offspring
            f1, f2 = self.fitness_func.evaluate(offspring)
            self.evaluations += 1
            offspring_obj = (f1, f2)

            # 4. Update population
            self.update_population(offspring, offspring_obj)

            if self.evaluations % 1000 == 0:
                print(f"Evaluations: {self.evaluations}, "
                      f"Population size: {len(self.population)}")

        return self.population
    
    def get_best_solution(self):
        """
        Returns the highest f1 solution (pure coverage/influence)
        """
        if not self.population:
            return None, None

        best_sol, best_obj = max(self.population, key=lambda x: x[1][0])
        return best_sol, best_obj

if __name__ == "__main__":
    # Create GSEMO instance
    gsemo = GSEMO(
        problem_id=2100,
        dimension=100,
        budget=10000
    )

    # Run GSEMO
    pareto_front = gsemo.run()

    # show results
    print(f"Final Pareto front size: {len(pareto_front)}")
    best_sol, best_obj = gsemo.get_best_solution()
    print(f"Best solution: f1={best_obj[0]}, f2={best_obj[1]}")

    # Trade-off plot
    f1_values = [obj[0] for sol, obj in pareto_front]
    f2_values = [obj[1] for sol, obj in pareto_front]

    import matplotlib.pyplot as plt
    plt.scatter(f2_values, f1_values)
    plt.xlabel('Cost (number of nodes)')
    plt.ylabel('Coverage/Influence')
    plt.title('Pareto Front')
    plt.show()