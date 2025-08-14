from tsp import TSP
from localsearch import LocalSearch
from mutations import Mutations
from individual_population import Individual, Population
from crossovers import order_crossover
from selection import Selection
import random
import numpy as np


class EvolutionaryAlgorithm:
    def __init__(self, tsp_file, population_size=50, max_generations=100):
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.max_generations = max_generations
        self.population = Population.random(self.tsp, self.population_size)

    # Select parents
    def select_parents(self):
        tours = np.array([ind.tour for ind in self.population.individuals])
        n = len(self.tsp.location_ids)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = self.tsp.dist(
                        self.tsp.location_ids[i], self.tsp.location_ids[j]
                    )

        selected_indices = Selection.Tournament_Selection(distance_matrix, tours, 3)

        # Convert back to individuals
        selected_parents = []
        for tour in selected_indices:
            for ind in self.population.individuals:
                if ind.tour == tour.tolist():
                    selected_parents.append(ind)
                    break

        return selected_parents

    # Crossover
    def crossover(self, parent1, parent2):
        child1_tour, child2_tour = order_crossover(parent1.tour, parent2.tour)

        child1 = Individual(self.tsp, child1_tour)
        child2 = Individual(self.tsp, child2_tour)

        return [child1, child2]

    # Mutation
    def mutate(self, individual):
        if random.random() < 0.1:  # 10% mutation rate
            mutated_tour, _, _ = Mutations.swap(individual.tour)
            individual.tour = mutated_tour
            individual.fitness = individual.evaluate()

    # Evolve generation
    def evolve_generation(self):
        parents = self.select_parents()
        offspring = []

        # Process parents in pairs
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                children = self.crossover(parents[i], parents[i + 1])

                # Mutate children
                for child in children:
                    self.mutate(child)
                    offspring.append(child)

        # Simple replacement
        new_population = Population.empty(self.tsp)

        # Add offspring
        for i in range(min(self.population_size, len(offspring))):
            new_population.add(offspring[i])

        # Fill with random if needed
        while len(new_population) < self.population_size:
            random_ind = Individual.random(self.tsp)
            new_population.add(random_ind)

        self.population = new_population

    # Get best individual
    def get_best_individual(self):
        best_idx, best_ind = self.population.best()
        return best_ind

    # Run the algorithm
    def run(self):
        print(f"Starting EA for TSP with {len(self.tsp.location_ids)} cities")
        print(
            f"Population: {self.population_size}, Generations: {self.max_generations}"
        )

        for generation in range(self.max_generations):
            self.evolve_generation()

            if generation % 20 == 0:
                best = self.get_best_individual()
                print(f"Generation {generation}: Best = {best.fitness:.2f}")

        # Final result
        best_individual = self.get_best_individual()
        print(f"Final best: {best_individual.fitness:.2f}")

        return best_individual
