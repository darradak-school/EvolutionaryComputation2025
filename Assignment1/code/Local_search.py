class LocalSearch:
    def __init__(self):
        pass

    # Swap two locations in the tour
    def exchange(self, tour, i, j):
        new_tour = tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour
    
    # Jump two locations in the tour
    def jump(self, tour, i, j):
        if i == j:
            return tour[:]
        new_tour = tour[:]
        location = new_tour.pop(i)
        new_tour.insert(j, location)
        return new_tour
    
    # Invert the order of locations between two indices
    def inversion(self, tour, i, j):
        new_tour = tour[:]
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour
    
    # Apply the two-opt algorithm to remove crossings
    def two_opt(self, tour, i, j):
        pass
    
    # Generate a list of neighbours based on the search type
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
                    neighbours.append(self.two_opt(tour, i, j))
        return neighbours