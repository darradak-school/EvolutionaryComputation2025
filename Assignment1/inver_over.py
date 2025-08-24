from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional

from tsp import TSP
from individual_population import Individual, Population

# Build a dictionary mapping each city to its position in the tour
def positions_of_city(tour: List[int]) -> Dict[int, int]:
    pos = {}
    for idx, city in enumerate(tour):
        pos[city] = idx
    return pos

# Check if two cities are adjacent
def are_adjacent(
    tour: List[int],
    pos: Dict[int, int],
    c: int, c_selected: int) -> bool:
    
    n = len(tour)
    idx_c = pos[c]  # find the position of c in the tour

    # check if c_selected is adjacent to c
    if tour[(idx_c + 1) % n] == c_selected:
        return True
    if tour[(idx_c - 1) % n] == c_selected:
        return True
    
    return False


def invert_function(
    tour: List[int], 
    start_inclusive: int, 
    end_inclusive: int) -> None:
   
    n = len(tour)
    # calculate segment length
    m = (end_inclusive - start_inclusive) % n + 1 

    for t in range(m // 2):
        # the two cities to swap
        a = (start_inclusive + t) % n
        b = (start_inclusive + (m - 1 - t)) % n
        tour[a], tour[b] = tour[b], tour[a]


# Create a child tour using the inver-over loop
def inver_over_loop(
    parent_tour: List[int], 
    child_tour: List[int],
    p: float,
    rng: Optional[random.Random] = None
    ) -> List[int]:       
  
    # set random number generator
    if rng is None:
        rng = random

    # initialize
    S = parent_tour[:]                  
    n = len(S)
    # find positions of cities in S
    pos_S = positions_of_city(S)       
    pos_child = positions_of_city(child_tour)

    c = rng.choice(S)

    while True:
        # select c'
        if rng.random() <= p:
            # look for a new c' which is not c
            candidates = [x for x in S if x != c]
            c_selected = rng.choice(candidates)
        else:
            # look for a new c' which is adjacent to c
            c_in_child = pos_child[c]
            c_selected = child_tour[(c_in_child + 1) % n]

        # check if c_selected is adjacent to c
        if are_adjacent(S, pos_S, c, c_selected):
            break

        # otherwise, swap c and c'
        idx_c  = pos_S[c]
        idx_cp = pos_S[c_selected]
        start = (idx_c + 1) % n
        end   = idx_cp
        invert_function(S, start, end)

        # positions changed
        pos_S = positions_of_city(S)
        c = c_selected

    return S

# Run inver-over
def run_inver_over(
    tsp: TSP,
    # popluation size required
    pop_size: int = 50,
    # probability to pick c' at random           
    p: float = 0.02,
    # number of generations
    generations: int = 20000,
    seed: int | None = None,
    verbose: bool = False,
    elitist_generation: bool = False,
) -> Tuple[Individual, float]:
   
    # set random number generator
    rng = random.Random(seed)

    # initialize random population
    population = Population.random(tsp=tsp, size=pop_size, rng=rng)
    best_ind = population.best()
    
    if best_ind.fitness is None:
        raise ValueError("fitness not calculated for best_ind")
    best_cost = float(best_ind.fitness)


    if verbose:
        print(f"[Init] best = {best_cost:.2f}")

    # generational loop
    for g in range(1, generations + 1):
        improved = False

        if not elitist_generation:
            # steady-state per individual 
            for i in range(len(population)):
                parent = population[i]
                if parent.fitness is None:
                    parent.fitness = parent.evaluate()

                Individual = population[rng.randrange(len(population))]
                child_tour = inver_over_loop(parent.tour, Individual.tour, p, rng)
                child_cost = tsp.tour_length(child_tour)

                # replace if child is better
                if child_cost <= parent.fitness:
                    population.individuals[i] = Individual(tsp=tsp, tour=child_tour)
                    population.individuals[i].fitness = child_cost

                    # update best
                    if child_cost + 1e-12 < best_cost:
                        best_cost = child_cost
                        best_ind  = population.individuals[i]
                        improved  = True
 
        if verbose and (g % 1000 == 0 or improved):
            msg = "improved" if improved else "no change"
            print(f"[Gen {g:5d}] {msg}, best = {best_cost:.6f}")

    return best_ind, best_cost
