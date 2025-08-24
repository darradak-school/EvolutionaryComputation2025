from tsp import TSP
from mutations import Mutations
from individual_population import Individual, Population
from crossovers import Crossovers
from selection import Selection
import random
import numpy as np


def overlap(t1, t2):
    """Calculates the edge overlap distance between two tours."""
    n = len(t1)

    def edges(t):
        """Returns the edges of a tour."""
        return {(t[i], t[(i + 1) % n]) for i in range(n)}

    return n - len(edges(t1) & edges(t2))


class CrowdingEA:
    """
    Steady-state EA with Deterministic Crowding:
      - Tournament selection (k)
      - Edge Recombination crossover (ERX)
      - Insertion mutation
      - Replacement: child vs most similar parent (edge overlap)
    """

    def __init__(
        self,
        tsp_file,
        population_size,
        generations,
        mutation_rate,
        crossover_rate,
        tournament_k=5,
    ):
        self.tsp = TSP(tsp_file)
        self.n = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k
        self.population = Population.random(self.tsp, self.n)

    def get_parents(self):
        """Tournament selection."""
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        idxs = Selection.Tournament_Selection(fitness_values, 2, self.tournament_k)
        return (
            self.population.individuals[idxs[0]],
            self.population.individuals[idxs[1]],
        )

    def crossover(self, p1, p2):
        """Edge Recombination crossover."""
        if random.random() < self.crossover_rate:
            c1_tour = Crossovers.edge_recombination(p1.tour, p2.tour)
            c2_tour = Crossovers.edge_recombination(p2.tour, p1.tour)
            return Individual(self.tsp, c1_tour), Individual(self.tsp, c2_tour)
        return p1.copy(), p2.copy()

    def mutate(self, ind):
        """Insertion mutation."""
        if random.random() < self.mutation_rate:
            mutated, _, _ = Mutations.insert(ind.tour)
            ind.tour = mutated
            ind.fitness = ind.evaluate()

    def step(self):
        """Steady-state GA with Deterministic Crowding."""
        p1, p2 = self.get_parents()
        idx1 = self.population.individuals.index(p1)
        idx2 = self.population.individuals.index(p2)

        c1, c2 = self.crossover(p1, p2)
        self.mutate(c1)
        self.mutate(c2)

        d11 = overlap(c1.tour, p1.tour)
        d12 = overlap(c1.tour, p2.tour)
        d21 = overlap(c2.tour, p1.tour)
        d22 = overlap(c2.tour, p2.tour)

        if d11 + d22 <= d12 + d21:
            pairs = [(c1, idx1), (c2, idx2)]
        else:
            pairs = [(c1, idx2), (c2, idx1)]

        for child, parent_idx in pairs:
            if child.fitness < self.population.individuals[parent_idx].fitness:
                self.population.individuals[parent_idx] = child

    def best(self):
        """Return the best individual from the population."""
        return self.population.best()
