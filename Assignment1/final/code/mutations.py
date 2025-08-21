import random

class Mutations:
    @staticmethod
    def insert(tour):
        """Pick two random alleles, move second to follow first, shifting others right."""
        i, j = random.sample(range(len(tour)), 2)
        new_tour = tour[:]

        if i > j:
            to_move = new_tour.pop(i)
            insert_pos = j + 1
        else:
            to_move = new_tour.pop(j)
            insert_pos = i + 1

        new_tour.insert(insert_pos, to_move)
        return new_tour, i, j

    @staticmethod
    def swap(tour):
        """Swap two random alleles in the tour."""
        i, j = random.sample(range(len(tour)), 2)
        new_tour = tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour, i, j

    @staticmethod
    def inversion(tour):
        """Invert the substring between two random indices (inclusive)."""
        i, j = sorted(random.sample(range(len(tour)), 2))
        mutated = tour[:]
        while i < j:
            mutated[i], mutated[j] = mutated[j], mutated[i]
            i += 1
            j -= 1
        return mutated, i, j