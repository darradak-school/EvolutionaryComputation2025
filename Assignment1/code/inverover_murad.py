import time
import random
import numpy as np
from individual_population import Individual, Population
from tsp import TSP


class InverOverAlgorithm:
    """
    Inver-Over algorithm for TSP with single inver-over operator and mate-guided successor.
    Uses O(1) delta-cost updates and stops after consecutive passes without improvement.
    """

    def __init__(
        self,
        tsp_file,
        population_size=100,
        generations=20000,
        inversion_prob=0.02,
        no_improve_limit=1000,
        rng_seed=None,
        log_every=1000,
    ):
        self.tsp = TSP(tsp_file)
        self.pop_size = int(population_size)
        self.max_passes = int(max(1, generations))
        self.p = float(inversion_prob)
        self.no_improve_limit = int(max(1, no_improve_limit))
        self.rng = random.Random(rng_seed)
        self.log_every = int(max(0, log_every))

        # Initialize population and track best
        self.population = Population.random(self.tsp, self.pop_size, rng=self.rng)
        _, best_ind = self.population.best()
        self.best_individual = best_ind.copy()
        self.best_fitness = float(self.best_individual.fitness)

    def invert_cyclic(self, arr, i_after, j):
        """Reverse cyclic segment from i_after..j, wrapping if needed."""
        n = len(arr)
        if n <= 1:
            return
        a = i_after % n
        b = j % n
        if a <= b:
            arr[a : b + 1] = reversed(arr[a : b + 1])
        else:
            # wrap-around: segment is arr[a:] + arr[:b+1]
            seg = list(reversed(arr[a:] + arr[: b + 1]))
            k = n - a
            arr[a:] = seg[:k]
            arr[: b + 1] = seg[k:]

    def inver_over_offspring(self, parent_tour, next_maps):
        """Apply inver-over transformation using O(1) delta updates."""
        n = len(parent_tour)
        if n <= 2:
            L = self.tsp.tour_length(parent_tour)
            return parent_tour[:], L

        S = parent_tour[:]
        pos = {city: idx for idx, city in enumerate(S)}
        total_len = self.tsp.tour_length(S)
        c = self.rng.choice(S)
        max_steps = 3 * n
        steps = 0

        while steps < max_steps:
            steps += 1

            # Choose c' randomly or as mate's successor
            if self.rng.random() <= self.p or not next_maps:
                cprime = self.rng.choice([x for x in S if x != c])
            else:
                nm = self.rng.choice(next_maps)
                cprime = nm[c]

            i = pos[c]
            left_neighbor = S[(i - 1) % n]
            right_neighbor = S[(i + 1) % n]

            # Stop if c' already neighbors c
            if cprime == left_neighbor or cprime == right_neighbor:
                break

            j = pos[cprime]
            a = (i + 1) % n

            # Delta cost for inverting segment a..j
            Sa = S[a]
            Sj = S[j]
            Sj1 = S[(j + 1) % n]

            old = self.tsp.dist(c, Sa) + self.tsp.dist(Sj, Sj1)
            new = self.tsp.dist(c, Sj) + self.tsp.dist(Sa, Sj1)
            total_len += new - old

            # Perform inversion and update position map
            self.invert_cyclic(S, a, j)
            if a <= j:
                affected = range(a, j + 1)
            else:
                affected = list(range(a, n)) + list(range(0, j + 1))
            for idx in affected:
                pos[S[idx]] = idx

            c = cprime

        return S, total_len

    def outer_pass(self):
        """Apply inver-over operator once to each individual."""
        improved = False
        inds = self.population.individuals

        # Build next city maps for all tours
        all_tours = [ind.tour for ind in inds]
        next_maps = []
        for tour in all_tours:
            nm = {}
            n = len(tour)
            for i, c in enumerate(tour):
                nm[c] = tour[(i + 1) % n]
            next_maps.append(nm)

        for i, ind in enumerate(inds):
            mates_next = next_maps[:i] + next_maps[i + 1 :]
            child_tour, child_len = self.inver_over_offspring(ind.tour, mates_next)

            # Replace if not worse
            parent_len = (
                ind.fitness
                if ind.fitness is not None
                else self.tsp.tour_length(ind.tour)
            )
            if child_len <= parent_len:
                ind.tour = child_tour
                ind.fitness = child_len
                if child_len < self.best_fitness:
                    self.best_fitness = child_len
                    self.best_individual = ind.copy()
                    improved = True

        return improved

    def run(self):
        """Execute EA until no improvement for no_improve_limit passes."""
        start_time = time.time()
        fitness_history = []

        stale = 0
        outer_pass = 0

        while stale < self.no_improve_limit and outer_pass < self.max_passes:
            outer_pass += 1
            improved = self.outer_pass()

            # Compute avg fitness for logging
            fitness_values = np.array(
                [
                    (
                        ind.fitness
                        if ind.fitness is not None
                        else self.tsp.tour_length(ind.tour)
                    )
                    for ind in self.population.individuals
                ],
                dtype=float,
            )
            avg_fitness = float(np.mean(fitness_values))
            fitness_history.append((outer_pass, float(self.best_fitness), avg_fitness))

            # Periodic logging
            if self.log_every and (outer_pass % self.log_every == 0):
                elapsed = time.time() - start_time
                print(
                    f"[{outer_pass:7d}] best={self.best_fitness:.6f}  avg={avg_fitness:.6f}  elapsed={elapsed:.2f}s"
                )

            stale = 0 if improved else stale + 1

        total_time = time.time() - start_time

        if self.log_every:
            print(
                f"Finished after {outer_pass} passes, best={self.best_fitness:.6f}, elapsed={total_time:.2f}s"
            )

        return self.best_individual.copy(), fitness_history, total_time


def run_inverover_test():
    """Run Inver-Over on TSPLIB instances."""
    problems = ["eil51", "st70", "eil76", "kroA100"]
    for problem in problems:
        tsp_path = f"tsplib/{problem}.tsp"
        print(f"\n### {problem} ###")
        algo = InverOverAlgorithm(
            tsp_path,
            population_size=200,
            generations=20000,
            inversion_prob=0.02,
            no_improve_limit=1000,
            log_every=1000,
        )
        best_individual, _, total_time = algo.run()
        print(f"Best fitness: {best_individual.fitness:.6f}  time={total_time:.2f}s")


if __name__ == "__main__":
    run_inverover_test()
