from tsp import TSP
from mutations import Mutations
from individual_population import Individual, Population
from crossovers import Crossovers
from selection import Selection
import random
import numpy as np


class EvolutionaryAlgorithm:
    """
    Steady-state GA with:
      - Tournament selection (k)
      - Order Crossover (OX)
      - Swap mutation
      - Replacement: child vs most similar parent (edge overlap)
    """
    def __init__(
        self,
        tsp_file,
        population_size,
        generations,
        mutation_rate,
        crossover_rate,
        tournament_size=3,
        replacement_rate=0.05,
    ):
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.generations = generations
        self.population = Population.random(self.tsp, self.population_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.replacement_rate = max(1, int(self.population_size * replacement_rate))

    def get_parents(self, num_parents=2):
        """ Tournament selection. """
        num_to_select = min(num_parents, len(self.population.individuals))
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        selected_indices = Selection.Tournament_Selection(
            fitness_values, num_to_select, self.tournament_size
        )
        return [self.population.individuals[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        """ Order Crossover (OX). """
        if random.random() < self.crossover_rate:
            child1_tour, child2_tour = Crossovers.order_crossover(
                parent1.tour, parent2.tour
            )
            child1 = Individual(self.tsp, child1_tour)
            child2 = Individual(self.tsp, child2_tour)
            return [child1, child2]
        else:
            return [
                Individual(self.tsp, parent1.tour.copy()),
                Individual(self.tsp, parent2.tour.copy()),
            ]

    def mutate(self, individual):
        """ Swap mutation. """
        if random.random() < self.mutation_rate:
            mutated_tour, _, _ = Mutations.swap(individual.tour)
            individual.tour = mutated_tour
            individual.fitness = individual.evaluate()

    def step(self):
        """ Steady-state GA with replacement. """
        num_replacements = max(1, int(self.population_size * self.replacement_rate))
        parents = self.get_parents(num_replacements)
        offspring = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                children = self.crossover(parents[i], parents[i + 1])
                for child in children:
                    self.mutate(child)
                    offspring.append(child)
            else:
                child = Individual(self.tsp, parents[i].tour.copy())
                self.mutate(child)
                offspring.append(child)

        self.population.individuals.sort(key=lambda x: x.fitness)
        num_offspring = min(len(offspring), num_replacements)
        for i in range(num_offspring):
            self.population.individuals[-(i + 1)] = offspring[i]

    def best(self):
        """ Return the best individual from the population. """
        return self.population.best()
