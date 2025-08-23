import time
import math
import random


class LocalSearch:
    ### TOUR HELPERS ###
    # Copy the tour so we dont alter the original.
    def copy(self, t):
        return t[:]

    # Get the previous index in the tour.
    def prev(self, i, n):
        return (i - 1) % n

    # Get the next index in the tour.
    def next(self, i, n):
        return (i + 1) % n

    ### NEIGHBOURHOOD APPLIERS ###
    # Get the new tour after applying exchange move.
    def exchange(self, tour, i, j):
        new_tour = self.copy(tour)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    # Get the new tour after applying jump move.
    def jump(self, tour, i, j):
        new_tour = self.copy(tour)
        a = new_tour.pop(i)
        new_tour.insert(j, a)
        return new_tour

    # Get the new tour after applying two opt move.
    def two_opt(self, tour, i, j):
        new_tour = self.copy(tour)
        new_tour[i : j + 1] = reversed(new_tour[i : j + 1])
        return new_tour

    ### TOUR DIFFERENCE CALCULATORS ###
    # Calculate the difference in tour length for two opt move.
    def diff_two_opt(self, tour, i, j, tsp):
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

    # Calculate the difference in tour length for exchange move.
    def diff_exchange(self, tour, i, j, tsp):
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

    # Calculate the difference in tour length for a jump move (remove i, insert at j).
    def diff_jump(self, tour, i, j, tsp):
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
    # Generate two random indices for the given search type.
    def random_ij(self, search_type, n):
        # Random i and j for two opt.
        if search_type == "two_opt":
            while True:
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

    # Run the searching algorithm.
    def search(
        self,
        tour,
        search_type,
        tsp,
        cooling,
        target,
        stag_limit=1000,  # Max cycles without improvement before stopping
    ):
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
