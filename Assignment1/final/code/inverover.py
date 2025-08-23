import time
import random
import numpy as np
from individual_population import Population
from tsp import TSP


class InverOverAlgorithm:
    """Inver-Over algorithm for TSP with single inver-over operator."""

    def __init__(
        self,
        tsp,
        pop_size,
        generations,
        inversion_p,
        stag_limit,
        max_steps,
    ):
        self.tsp = TSP(tsp)
        self.pop_size = int(pop_size)
        self.pass_limit = int(max(1, generations))
        self.p = float(inversion_p)
        self.stag_limit = int(max(1, stag_limit)) if stag_limit else float("inf")
        self.max_steps = int(max_steps)  # Store max_steps

        # Initialize population and track best
        self.population = Population.random(self.tsp, self.pop_size)
        _, best_ind = self.population.best()
        self.best_individual = best_ind.copy()
        self.best_fitness = float(best_ind.fitness)

    def invert(self, arr, i_after, j):
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

    def offspring(self, parent_tour, next_maps):
        """Apply inver-over transformation using delta updates."""
        n = len(parent_tour)
        if n <= 2:
            return parent_tour[:], self.tsp.tour_length(parent_tour)

        S = parent_tour[:]
        pos = {city: idx for idx, city in enumerate(S)}
        total_len = self.tsp.tour_length(S)
        c = random.choice(S)
        max_steps = self.max_steps

        for _ in range(max_steps):
            # Choose c' randomly or as mate's successor
            if random.random() <= self.p or not next_maps:
                cprime = random.choice([x for x in S if x != c])
            else:
                nm = random.choice(next_maps)
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
            Sa, Sj, Sj1 = S[a], S[j], S[(j + 1) % n]
            total_len += (
                self.tsp.dist(c, Sj)
                + self.tsp.dist(Sa, Sj1)
                - self.tsp.dist(c, Sa)
                - self.tsp.dist(Sj, Sj1)
            )

            # Perform inversion and update position map
            self.invert(S, a, j)
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
        next_maps = []
        for tour in inds:
            n = len(tour.tour)
            nm = {tour.tour[i]: tour.tour[(i + 1) % n] for i in range(n)}
            next_maps.append(nm)

        for i, ind in enumerate(inds):
            mates_next = next_maps[:i] + next_maps[i + 1 :]
            child_tour, child_len = self.offspring(ind.tour, mates_next)

            # Replace if not worse
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

        while stagnation < self.stag_limit and gen < self.pass_limit:
            gen += 1
            improved = self.outer_pass()

            # Periodic logging
            if gen % 1000 == 0:
                # Calculate current average fitness
                fitness_vals = [
                    ind.fitness or self.tsp.tour_length(ind.tour)
                    for ind in self.population.individuals
                ]
                avg_fitness = float(np.mean(fitness_vals))

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

    # Dictionary to store results for each problem
    results = {}

    for problem in problems:
        print(f"\n### {problem} ###")
        problem_size = int(
            "".join(c for c in problem if c.isdigit())
        )  # Get the problem size

        if problem_size < 1000:  # If problem size is less than 1000, use 20 steps
            steps = 20
        elif problem_size < 10000:  # If problem size is greater than 1000, use 5 steps
            steps = 5
        else:  # If problem size is greater than 10000, use 1 step
            steps = 1

        # Collect results over 30 runs
        best_fitnesses = []
        run_times = []

        for run in range(30):
            print(f"Run {run + 1}")
            tsp = f"tsplib/{problem}.tsp"

            algo = InverOverAlgorithm(
                tsp,  # TSP problem to run on
                pop_size=50,  # Population size
                generations=20000,  # Number of generations
                inversion_p=0.02,  # Inversion probability
                stag_limit=None,  # Stagnation limit (None for no limit)
                max_steps=steps,  # Maximum inversion steps per offspring
            )

            best_individual, total_time = algo.run()
            best_fitnesses.append(best_individual.fitness)
            run_times.append(total_time)

        # Calculate statistics
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

    # Write results to file
    with open("../results/inverover.txt", "w") as f:
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
