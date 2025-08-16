from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple, Callable
import random
import numpy as np
from tsp import TSP


# The individual class represents one possible solution to the TSP
class Individual:
    def __init__(self, tsp: TSP, tour: Optional[list[int]] = None):

        # Create a TSP solution (individual)
        # If tour is not provided, a random tour is generated
        self.tsp = tsp
        if tour is None:
            self.tour = tsp.random_tour()
        else:
            self.tour = tour
        # The fitness is the total length of the tour
        self.fitness: Optional[float] = self.evaluate()

    def evaluate(self) -> float:
        return self.tsp.tour_length(self.tour)

    @classmethod
    def random(cls, tsp: TSP, rng: Optional[random.Random] = None) -> "Individual":
        rng = rng or random
        return cls(tsp=tsp, tour=tsp.random_tour())

    def copy(self) -> "Individual":
        copy_tour = Individual(tsp=self.tsp, tour=self.tour)
        return copy_tour

    def __str__(self) -> str:
        return f"Tour: {self.tour}, Fitness: {self.fitness:.2f}"

    def __repr__(self) -> str:
        return f"fitness={self.fitness:.2f}"


# The population class is a wrapper for a list of individuals.
FitnessFn = Callable[[float], float]


@dataclass
class Population:
    """
    A population is a collection of Individuals tied to a specific TSP instance.
    Provides random construction, evaluation, and 'best' query.
    """

    tsp: TSP
    individuals: List[Individual]

    @classmethod
    def empty(cls, tsp: TSP) -> "Population":
        return cls(tsp=tsp, individuals=[])

    @classmethod
    def random(
        cls,
        tsp: TSP,
        size: int,
        rng: Optional[random.Random] = None,
        fitness: Optional[FitnessFn] = None,
    ) -> "Population":
        rng = rng or random
        inds = [Individual.random(tsp, rng) for _ in range(size)]
        pop = cls(tsp=tsp, individuals=inds)
        pop.evaluate_all(fitness)
        return pop

    def evaluate_all(self, fitness: Optional[FitnessFn] = None) -> None:
        # Calculate fitness scores directly using tours
        n = len(self.individuals)
        fitness_scores = np.empty(n, dtype=np.float64)

        for i in range(n):
            fitness_scores[i] = self.tsp.tour_length(self.individuals[i].tour)

        # Apply fitness transformation if provided
        if fitness is not None:
            fitness_scores = np.array([fitness(score) for score in fitness_scores])

        # Update individual fitness values
        for i, ind in enumerate(self.individuals):
            ind.fitness = fitness_scores[i]

    def best(self) -> Tuple[int, Individual]:
        # Find the best individual in the population.
        for ind in self.individuals:
            if ind.fitness is None:
                ind.fitness = ind.evaluate()

        # Find the best individual in the population.
        fitness_values = np.array([ind.fitness for ind in self.individuals])
        best_idx = np.argmin(fitness_values)
        return best_idx, self.individuals[best_idx]

    def add(
        self,
        ind: Individual,
        evaluate: bool = True,
        fitness: Optional[FitnessFn] = None,
    ) -> None:
        if evaluate:
            cost = ind.evaluate()
            if fitness is not None:
                ind.fitness = fitness(cost)
            else:
                ind.fitness = cost
        self.individuals.append(ind)

    def extend(
        self,
        inds: Iterable[Individual],
        evaluate: bool = False,
        fitness: Optional[FitnessFn] = None,
    ) -> None:
        if evaluate:
            for ind in inds:
                cost = ind.evaluate()
                if fitness is not None:
                    ind.fitness = fitness(cost)
                else:
                    ind.fitness = cost
        self.individuals.extend(inds)

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, i: int) -> Individual:
        return self.individuals[i]
