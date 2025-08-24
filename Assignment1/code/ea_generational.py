from tsp import TSP
from mutations import Mutations
from individual_population import Individual, Population
from crossovers import Crossovers
from selection import Selection
import random


class GenerationalEA:
    """
    Generational EA with:
      - Fitness-Proportional (roulette) selection
      - PMX crossover
      - Inversion mutation
      - Elitist replacement (μ best from parents+offspring)
    """

    def __init__(
        self,
        tsp_file,
        population_size,
        generations,
        mutation_rate,
        crossover_rate,
        elite_fraction=0.02,
    ):
        self.tsp = TSP(tsp_file)
        self.n = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_k = max(1, int(self.n * elite_fraction))
        self.population = Population.random(self.tsp, self.n)

    def get_parents(self):
        """ Fitness-Proportional (roulette) selection. """
        tours = [ind.tour for ind in self.population.individuals]
        idxs = Selection.Fitness_Proportional_Selection(self.tsp, tours)
        return [self.population.individuals[i] for i in idxs]

    def crossover(self, p1, p2):
        """ PMX crossover. """
        if random.random() < self.crossover_rate:
            c1_tour, c2_tour = Crossovers.pmx_crossover(p1.tour, p2.tour)
            return Individual(self.tsp, c1_tour), Individual(self.tsp, c2_tour)
        return Individual(self.tsp, p1.tour.copy()), Individual(
            self.tsp, p2.tour.copy()
        )

    def mutate(self, ind):
        """ Inversion mutation. """
        if random.random() < self.mutation_rate:
            mutated, _, _ = Mutations.inversion(ind.tour)
            ind.tour = mutated
            ind.fitness = ind.evaluate()

    def step(self):
        """ Generational EA with elitist replacement. """
        self.population.individuals.sort(key=lambda x: x.fitness)
        elites = [ind.copy() for ind in self.population.individuals[: self.elite_k]]

        offspring = []
        parents = self.get_parents()

        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                child = parents[i].copy()
                self.mutate(child)
                offspring.append(child)
            else:
                c1, c2 = self.crossover(parents[i], parents[i + 1])
                self.mutate(c1)
                self.mutate(c2)
                offspring.extend([c1, c2])
            if len(offspring) >= self.n:
                break

        while len(offspring) < self.n:
            offspring.append(Individual.random(self.tsp))
        if len(offspring) > self.n:
            offspring = offspring[: self.n]

        combined = Population(self.tsp, elites + offspring)
        combined.individuals.sort(key=lambda x: x.fitness)
        self.population = Population(self.tsp, combined.individuals[: self.n])

    def best(self):
        """ Return the best individual from the population. """
        return self.population.best()
