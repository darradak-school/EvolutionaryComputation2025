import time
import random
import numpy as np
from individual_population import Population
from tsp import TSP


class InverOverAlgorithm:
    """Inverover algorithm, parents only compete with offspring, uses inverover operator."""

    def __init__(
        self,
        tsp,
        pop_size,
        generations,
        inversion,
        stag_limit,
    ):
        self.tsp = TSP(tsp)
        self.pop_size = pop_size
        self.generations = generations
        self.p = inversion
        self.stag_limit = stag_limit if stag_limit else float("inf")

        # Initialize population and track the best individual.
        self.population = Population.random(self.tsp, self.pop_size)
        _, best_ind = self.population.best()
        self.best_individual = best_ind.copy()
        self.best_fitness = float(best_ind.fitness)

    def invert(self, arr, ia, j):
        """Reverse cyclic segment from location after i -> j, wrapping if needed."""
        n = len(arr)
        if n <= 1:
            return
        # Ensure that the indicies wrap correctly.
        a = ia % n
        b = j % n
        if a <= b:
            arr[a : b + 1] = reversed(arr[a : b + 1])
        else:
            # wrap-around: segment is arr[a:] + arr[:b+1].
            seg = list(reversed(arr[a:] + arr[: b + 1]))
            k = n - a
            arr[a:] = seg[:k]
            arr[: b + 1] = seg[k:]

    def offspring(self, p_tour, next_maps):
        """Apply inverover transformation using delta updates."""
        n = len(p_tour)
        if n <= 2:
            return p_tour[:], self.tsp.tour_length(p_tour)
        s = p_tour[:]  # Copy the parent tour into s.
        pos = {city: idx for idx, city in enumerate(s)}
        total_len = self.tsp.tour_length(s)
        c = random.choice(s)  # Choose a random location to start with.

        while True:
            # Choose c' randomly or as mate's successor.
            if random.random() <= self.p or not next_maps:
                cprime = random.choice([x for x in s if x != c])
            else:
                nm = random.choice(next_maps)
                cprime = nm[c]

            i = pos[c]
            left = s[(i - 1) % n]
            right = s[(i + 1) % n]

            # Stop if c' already neighbors c.
            if cprime == left or cprime == right:
                break

            j = pos[cprime]
            a = (i + 1) % n

            # Delta cost for inverting segment.
            # BEFORE: c -> sa ... sj -> sj1
            # AFTER:  c -> sj ... sa -> sj1
            sa, sj, sj1 = s[a], s[j], s[(j + 1) % n]
            total_len += (
                self.tsp.dist(c, sj)
                + self.tsp.dist(sa, sj1)
                - self.tsp.dist(c, sa)
                - self.tsp.dist(sj, sj1)
            )

            # Perform inversion and update position map.
            self.invert(s, a, j)
            if a <= j:
                affected = range(a, j + 1)
            else:
                affected = list(range(a, n)) + list(range(0, j + 1))
            for idx in affected:
                pos[s[idx]] = idx

            c = cprime

        return s, total_len

    def outer_pass(self):
        """Apply inverover operator once to each individual."""
        improved = False
        inds = self.population.individuals

        # Build next city maps for all tours.
        next_maps = []
        for tour in inds:
            n = len(tour.tour)
            nm = {tour.tour[i]: tour.tour[(i + 1) % n] for i in range(n)}
            next_maps.append(nm)

        # Apply inverover operator to each individual.
        for i, ind in enumerate(inds):
            mates_next = next_maps[:i] + next_maps[i + 1 :]
            child_tour, child_len = self.offspring(ind.tour, mates_next)

            # Replace if not worse.
            parent_len = ind.fitness or self.tsp.tour_length(ind.tour)
            if child_len <= parent_len:
                ind.tour = child_tour
                ind.fitness = child_len
                if child_len < self.best_fitness:
                    self.best_fitness = child_len
                    self.best_individual = ind.copy()
                    improved = True

        return improved

    def run(self):
        """Execute EA until no improvement for stag_limit passes."""
        start_time = time.time()
        stagnation = 0
        gen = 0

        # Execute EA until no improvement for ammount of generations determined by stag_limit or hit generation limit.
        while stagnation < self.stag_limit and gen < self.generations:
            gen += 1
            improved = self.outer_pass()

            # Print progress every 1000 generations.
            if gen % 1000 == 0:
                # Calculate current average fitness.
                fitness_vals = [
                    ind.fitness or self.tsp.tour_length(ind.tour)
                    for ind in self.population.individuals
                ]
                avg_fitness = float(np.mean(fitness_vals))
                # Calculate elapsed time.
                elapsed = time.time() - start_time
                print(
                    f"Generation: {gen} | Best: {self.best_fitness:.2f} | Avg: {avg_fitness:.2f} | Elapsed: {elapsed:.2f}s"
                )

            stagnation = 0 if improved else stagnation + 1

        total_time = time.time() - start_time

        print(
            f"Finished after {gen} generations. Best: {self.best_fitness:.2f} Elapsed: {total_time:.2f}s"
        )

        return self.best_individual.copy(), total_time


## MAIN TESTING FUNCTION FOR THE INVEROVER ALGORITHM ##
def main():
    """Main function to run and test the inverover algorithm."""
    problems = [
        "eil51",
        "st70",
        "eil76",
        "kroA100",
        "kroC100",
        "kroD100",
        "eil101",
        "lin105",
        "pcb442",
        "pr2392",
        "usa13509",
    ]

    # Dictionary to store results for each problem.
    results = {}

    for problem in problems:
        print(f"\n### {problem} ###")

        # Collect results.
        best_fitnesses = []
        run_times = []

        for run in range(30):
            print(f"Run {run + 1}")
            tsp = f"tsplib/{problem}.tsp"

            algo = InverOverAlgorithm(
                tsp,  # TSP problem to run on
                pop_size=50,  # Population size
                generations=20000,  # Number of generations
                inversion=0.02,  # Inversion probability
                stag_limit=2000,  # Stagnation limit (None for no limit)
            )

            best_individual, total_time = algo.run()
            best_fitnesses.append(best_individual.fitness)
            run_times.append(total_time)

        # Calculate statistics.
        avg_fitness = np.mean(best_fitnesses)
        std_fitness = np.std(best_fitnesses)
        avg_time = np.mean(run_times)
        std_time = np.std(run_times)

        results[problem] = {
            "avg_fitness": avg_fitness,
            "std_fitness": std_fitness,
            "avg_time": avg_time,
            "std_time": std_time,
        }

        print(f"Problem: {problem}")
        print(f"Average Best Fitness: {avg_fitness:.2f} +/- {std_fitness:.2f}")
        print(f"Average Time: {avg_time:.2f}s +/- {std_time:.2f}s")

    # Write results to file.
    with open("../results/inverover.txt", "a") as f:
        f.write("InverOver Algorithm Results\n")
        f.write("=" * 50 + "\n")

        for problem, result in results.items():
            f.write(f"Problem: {problem}\n")
            f.write(
                f"Average Best Fitness: {result['avg_fitness']:.2f} +/- {result['std_fitness']:.2f}\n"
            )
            f.write(
                f"Average Time: {result['avg_time']:.2f}s +/- {result['std_time']:.2f}s\n"
            )
            f.write("-" * 30 + "\n\n")


if __name__ == "__main__":
    main()
