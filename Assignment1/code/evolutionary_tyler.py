from tsp import TSP
from mutations import Mutations
from individual_population import Individual, Population
from crossovers import Crossovers
from selection import Selection
import random
import numpy as np

# EA based on edge recombination and inversion

class EvolutionaryAlgorithm:
    def __init__(
        self,
        tsp_file,
        population_size,
        max_generations,
        mutation_rate,
        crossover_rate,
        tournament_size,
        replacement_rate,
    ):
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.max_generations = max_generations
        self.population = Population.random(self.tsp, self.population_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.replacement_rate = replacement_rate

    # Get parents using tournament selection on fitness values.
    def get_parents(self, num_parents=2):
        """Select parents using tournament selection based on fitness"""
        num_parents_to_select = min(num_parents, len(self.population.individuals))
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        selected_parent_indices = Selection.Tournament_Selection(
            fitness_values, num_parents_to_select, self.tournament_size
        )
        return [self.population.individuals[i] for i in selected_parent_indices]

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents with a given probability"""
        if random.random() < self.crossover_rate:
            # Apply cycle crossover if crossover rate is met
            child1_tour, child2_tour = Crossovers.cycle_crossover(
                parent1.tour, parent2.tour
            )
            child1 = Individual(self.tsp, child1_tour)
            child2 = Individual(self.tsp, child2_tour)
            return [child1, child2]
        else:
            # Otherwise, return clones of the parents
            return [
                Individual(self.tsp, parent1.tour[:]),  # Use slicing for copying
                Individual(self.tsp, parent2.tour[:]),
            ]

    # Mutate individual with given probability.
    def mutate(self, individual):
        """Randomly apply inversion mutation to an individual"""
        if random.random() < self.mutation_rate:
            # Invert a segment of the tour with inversion mutation
            individual.tour = Mutations.inversion(individual.tour)[0]
            # Re-evaluate fitness after mutation
            individual.fitness = individual.evaluate()

    # Evolutionary algorithm.
    def evolution(self):
        num_replacements = max(1, int(self.population_size * self.replacement_rate))
        parents = self.get_parents(num_replacements)
        offspring = []

        # Generate offspring
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                children = self.crossover(parents[i], parents[i + 1])
                for child in children:
                    self.mutate(child)
                    offspring.append(child)
            else:
                # Handle odd number of parents
                child = Individual(self.tsp, parents[i].tour.copy())
                self.mutate(child)
                offspring.append(child)

        # Sort population by fitness (best first) and replace worst individuals
        self.population.individuals.sort(key=lambda x: x.fitness)
        num_offspring = min(len(offspring), num_replacements)
        for i in range(num_offspring):
            self.population.individuals[-(i + 1)] = offspring[i]
