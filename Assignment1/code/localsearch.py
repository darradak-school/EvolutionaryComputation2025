from tsp import TSP
import time

class LocalSearch:
    def __init__(self):
        pass

    # Swap two locations in the tour.
    def exchange(self, tour, i, j):
        new_tour = tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
    
    # Jump two locations in the tour.
    def jump(self, tour, i, j):
        if i == j:
            return tour[:]
        new_tour = tour[:]
        location = new_tour.pop(i)
        new_tour.insert(j, location)
        return new_tour
    
    # Invert the order of locations between two indices.
    def inversion(self, tour, i, j):
        new_tour = tour[:]
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour
    
    # Apply the two-opt algorithm to switch two edges.
    def two_opt(self, tour, i, j):
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour
    
    # Generate a list of neighbours based on the search type.
    def neighbourhood(self, tour, search_type):
        neighbours = []
        n = len(tour)
        if search_type == 'exchange':
            for i in range(n):
                for j in range(i + 1, n):
                    neighbours.append(self.exchange(tour, i, j))
        elif search_type == 'jump':
            for i in range(n):
                for j in range(i + 1, n):
                    neighbours.append(self.jump(tour, i, j))
        elif search_type == 'inversion':
            for i in range(n):
                for j in range(i + 1, n):
                    neighbours.append(self.inversion(tour, i, j))
        elif search_type == 'two_opt':
            for i in range(n):
                for j in range(i + 1, n):
                    # Avoid invalid two-opt moves, adjacent edges or inverting the entire tour.
                    if j - i > 1 and not (i == 0 and j == n - 1):
                        neighbours.append(self.two_opt(tour, i, j))
        return neighbours

    # Find the best neighbour using the tour length.
    def best(self, tour, search_type, tsp):
        neighbours = self.neighbourhood(tour, search_type)
        tour_score = tsp.tour_length(tour)
        best_tour = tour
        for n_tour in neighbours:
            n_score = tsp.tour_length(n_tour)
            if n_score < tour_score:
                best_tour = n_tour
        return best_tour

    # Find the first better neighbour using the tour length, returns the first better neighbour.
    def first(self, tour, search_type, tsp):
        neighbours = self.neighbourhood(tour, search_type)
        tour_score = tsp.tour_length(tour)
        for n_tour in neighbours:
            n_score = tsp.tour_length(n_tour)
            if n_score < tour_score:
                return n_tour
        return tour

    # Run the local search algorithm until no better neighbour is found, tracks cycles taken. Can be limited to a certain number of cycles.
    def local_search(self, tour, search_type, search_method, tsp, limit=None):
        start_time = time.time()
        cycles_taken = 0
        current_tour = tour
        while True:
            if search_method == 'best':
                best_neighbour = self.best(current_tour, search_type, tsp)
            elif search_method == 'first':
                best_neighbour = self.first(current_tour, search_type, tsp)
            if best_neighbour == current_tour:
                return current_tour, cycles_taken, time.time() - start_time
            current_tour = best_neighbour
            cycles_taken += 1
            if limit is not None and cycles_taken > limit:
                return current_tour, cycles_taken, time.time() - start_time