import time
import math
import random
from tsp import TSP


class LocalSearch:
    ### TOUR HELPERS ###
    def copy(self, t):
        """Copy the tour so we dont alter the original."""
        return t[:]

    def prev(self, i, n):
        """Get the previous index in the tour."""
        return (i - 1) % n

    def next(self, i, n):
        """Get the next index in the tour."""
        return (i + 1) % n

    ### NEIGHBOURHOOD APPLIERS ###
    def exchange(self, tour, i, j):
        """Get the new tour after applying exchange move."""
        new_tour = self.copy(tour)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def jump(self, tour, i, j):
        """Get the new tour after applying jump move."""
        new_tour = self.copy(tour)
        a = new_tour.pop(i)
        new_tour.insert(j, a)
        return new_tour

    def two_opt(self, tour, i, j):
        """Get the new tour after applying two opt move."""
        new_tour = self.copy(tour)
        new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
        return new_tour

    ### TOUR DIFFERENCE CALCULATORS ###
    def diff_two_opt(self, tour, i, j, tsp):
        """Calculate the difference in tour length for two opt move."""
        n = len(tour)
        dist = tsp.dist

        # Get locations of i and j, and their previous and next location using circular indexing.
        li = tour[i]
        lj = tour[j]
        ipre = tour[self.prev(i, n)]
        jnext = tour[self.next(j, n)]

        # BEFORE: ipre -> li ... lj -> jnext
        # AFTER:  ipre -> lj ... li -> jnext
        return (dist(ipre, lj) + dist(li, jnext)) - (dist(ipre, li) + dist(lj, jnext))

    def diff_exchange(self, tour, i, j, tsp):
        """Calculate the difference in tour length for exchange move."""
        n = len(tour)
        dist = tsp.dist

        # Check if i and j are adjacent in the tour, includes wrap around cases.
        adj = ((i + 1) % n == j) or ((j + 1) % n == i)

        # If adjacent but in the reverse orientation, flip them so j is i's successor.
        if adj and (j + 1) % n == i:
            i, j = j, i

        # Get locations of i and j, and their previous and next locations using circular indexing.
        li, lj = tour[i], tour[j]
        ipre = tour[(i - 1) % n]
        inext = tour[(i + 1) % n]
        jpre = tour[(j - 1) % n]
        jnext = tour[(j + 1) % n]

        if adj:
            # BEFORE: ipre -> li -> lj -> jnext
            # AFTER : ipre -> lj -> li -> jnext
            return (dist(ipre, lj) + dist(li, jnext)) - (
                dist(ipre, li) + dist(lj, jnext)
            )

        # BEFORE: ipre -> li -> inext ... jpre -> lj -> jnext
        # AFTER : ipre -> lj -> inext ... jpre -> li -> jnext
        before = dist(ipre, li) + dist(li, inext) + dist(jpre, lj) + dist(lj, jnext)
        after = dist(ipre, lj) + dist(lj, inext) + dist(jpre, li) + dist(li, jnext)
        return after - before

    def diff_jump(self, tour, i, j, tsp):
        """Calculate the difference in tour length for a jump move (remove i, insert at j)."""
        n = len(tour)
        dist = tsp.dist
        li = tour[i]

        # Cost change from removing i.
        ipre = tour[self.prev(i, n)]
        inext = tour[self.next(i, n)]
        diff_r = dist(ipre, inext) - dist(ipre, li) - dist(li, inext)

        # Cost change from inserting i at j.
        # Function to map new index to original index.
        def imap(x):
            return x if x < i else x + 1  # Skipping i.

        n -= 1  # New length of tour.
        jpre = tour[imap((j - 1) % n)]  # Location before j position.
        jnext = tour[imap(j % n)]  # Location after j position.

        # BEFORE: jpre -> jnext
        # AFTER : jpre -> li -> jnext
        diff_i = -dist(jpre, jnext) + dist(jpre, li) + dist(li, jnext)

        return diff_r + diff_i  # New edges - old edges.

    ### SIMULATED ANNEALING ###
    def random_ij(self, search_type, n):
        """Generate two random indices for the given search type."""
        # Random i and j for two opt.
        if search_type == "two_opt":
            while True:
                # Random i and j for two opt.
                i = random.randrange(0, n - 1)
                j_low = i + 2
                j_high = n if i > 0 else n - 1
                if j_low < j_high:
                    j = random.randrange(j_low, j_high)
                    return i, j

        # Random i and j for exchange.
        elif search_type == "exchange":
            i = random.randrange(0, n - 1)
            j = random.randrange(i + 1, n)
            return i, j

        # Random i and j for jump.
        elif search_type == "jump":
            while True:
                i = random.randrange(0, n)
                j = random.randrange(0, n)
                if i != j:
                    return i, j

    def search(
        self,
        tour,
        search_type,
        tsp,
        cooling,
        target,
        stag_limit=1000,  # Max cycles without improvement before stopping
    ):
        """Run the searching algorithm with the provided parameters."""
        start = time.time()
        n = len(tour)

        # Get the functions for the selected search type.
        diff = getattr(self, f"diff_{search_type}")
        move = getattr(self, search_type)

        # Copy tour and get its length.
        t = self.copy(tour)
        t_len = tsp.tour_length(t)

        # Calibrate temperature from a sample of positive diffs.
        pos_diffs = []
        samples = max(50, int(1000))
        # Sample random neighbours and record positive diffs, then calculate mean.
        for i in range(samples):
            i, j = self.random_ij(search_type, n)
            d = diff(t, i, j, tsp)
            if d > 0:
                pos_diffs.append(d)
        dmean = sum(pos_diffs) / len(pos_diffs)
        temp = -dmean / math.log(target)  # Calculate temperature.

        # Variables for the search.
        cycles = 0
        stag = 0
        best_length = t_len

        # Run the search until the temperature drops to 0 or the stagnation limit is reached.
        while True:
            i, j = self.random_ij(search_type, n)
            d = diff(t, i, j, tsp)

            # Accept tour if better, otherwise accept with probability determined by temp.
            accept = False
            if d <= 0:
                accept = True
            elif temp > 0:
                z = d / temp
                if z <= 700 and random.random() < math.exp(-z):
                    accept = True

            # If the move is accepted, update the tour and length.
            if accept:
                t = move(t, i, j)
                t_len += d

                # Reset stagnation counter if we found an improvement.
                if t_len < best_length:
                    best_length = t_len
                    stag = 0
                else:
                    stag += 1
            else:
                stag += 1

            cycles += 1

            # Cooling each cycle.
            if cooling < 1 and temp > 0:
                temp *= cooling
                # Set temp to 0 if it's too small.
                if temp < 1e-12:
                    temp = 0.0

            # Stop when temp drops to 0 or no improvements for a while
            if temp == 0.0 or stag >= stag_limit:
                return t, cycles, time.time() - start


## MAIN TESTING FUNCTION FOR THE LOCAL SEARCH ALGORITHM ##
def main():
    """Main function to run and test the local search algorithm, runs with the following configuration variables."""
    ### CONFIGURATION VARIABLES ###
    PROBLEMS = [
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
    ]  # Problems to test.

    TYPES = ["jump", "exchange", "two_opt"]  # Search types to test.
    INSTANCES = 30  # Number of instances to run for each problem.
    STAG_LIMIT = 10000  # Max number of cycles without an improvement before stopping.

    ### SIMULATED ANNEALING VARIABLES ###
    COOLING = 0.9999  # How fast to cool the temp, 1.0 disables cooling.
    TARGET = 0.3  # Target chance to accept for an average tour with apositive diff.

    # Calculate statistics for results.
    def calc(data_list):
        values = [r[0] for r in data_list]
        cycles = [r[1] for r in data_list]
        times = [r[2] for r in data_list]
        return (
            sum(values) / len(values),  # Mean tour length.
            min(values),  # Minimum tour length.
            sum(cycles) / len(cycles),  # Mean number of cycles.
            sum(times) / len(times),  # Mean time taken.
        )

    # Write results to file.
    def write_results(f, section_name, results, search_types):
        f.write(f"===== {section_name} =====\n")
        for search_type in search_types:
            data = results[search_type]
            avg, min_v, avg_cycles, avg_time = calc(data)
            if section_name == "MEAN":
                f.write(f"{search_type.title()}: {avg:.2f}\n")
            elif section_name == "MINIMUM":
                f.write(f"{search_type.title()}: {min_v:.2f}\n")
            elif section_name == "MEAN TIMES":
                f.write(f"{search_type.title()} cycles: {avg_cycles:.2f}\n")
                f.write(f"{search_type.title()} time: {avg_time:.2f}s\n")
                f.write("---------------------\n")

    # Run tests on all problems.
    for problem in PROBLEMS:
        start = time.time()
        print(f"Running {problem}...")

        # Create the problem instance.
        tsp = TSP(f"tsplib/{problem}.tsp")

        # Create random tours to get average unoptimised tour length.
        tours = []
        for _ in range(10):
            t = tsp.random_tour()
            tours.append(tsp.tour_length(t))

        # Results dictionary for each search type.
        results = {search_type: [] for search_type in TYPES}

        # Create local search object.
        local_search = LocalSearch()

        for search_type in TYPES:
            # Run a few times to get average results.
            for _ in range(INSTANCES):
                r_tour = tsp.random_tour()  # Create a random starting tour.
                result = local_search.search(
                    r_tour, search_type, tsp, COOLING, TARGET, STAG_LIMIT
                )
                # Append the result to the results dictionary.
                tour_length = tsp.tour_length(result[0])
                results[search_type].append((tour_length, result[1], result[2]))

        print(f"Time taken: {time.time() - start:.2f} seconds")

        # Calculate average unoptimised tour length.
        baseline_avg = sum(tours) / len(tours)

        # Write results to file.
        with open("../results/local_search.txt", "a") as f:
            f.write(f"\n######### {problem} #########\n")
            f.write(f"Average tour length before optimisation: {baseline_avg:.2f}\n")
            write_results(f, "MEAN", results, TYPES)
            write_results(f, "MINIMUM", results, TYPES)
            write_results(f, "MEAN TIMES", results, TYPES)


if __name__ == "__main__":
    main()
