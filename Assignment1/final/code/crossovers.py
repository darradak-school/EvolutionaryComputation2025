# Crossover file to be completed by Darcy

from tsp import TSP

import random

# setup as class so more can be added

class Crossovers:

    # Crossover functions implemented as per assignment specs
    # Order Crossover
    @staticmethod 
    def order_crossover(parent1, parent2):
        """ Preserves relative order of cities from parents. """
        size = len(parent1)
        
        # Choose random segment from parent1
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        # Create children
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy segment from parents to children
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # Fill child1 with remaining cities from parent2
        segment1 = set(parent1[start:end+1])
        pointer = (end + 1) % size
        
        for city in parent2[end+1:] + parent2[:end+1]:
            if city not in segment1:
                child1[pointer] = city
                pointer = (pointer + 1) % size
        
        # Fill child2 with remaining cities from parent1
        segment2 = set(parent2[start:end+1])
        pointer = (end + 1) % size
        
        for city in parent1[end+1:] + parent1[:end+1]:
            if city not in segment2:
                child2[pointer] = city
                pointer = (pointer + 1) % size
        
        return child1, child2

    # PMX Crossover
    @staticmethod
    def pmx_crossover(parent1, parent2):
        """ Creates mapping between segment positions. """
        size = len(parent1)
        
        # Choose random segment 
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        # Initialize children with -1 (empty positions)
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy the segment from parents to opposite children
        child1[start:end+1] = parent2[start:end+1]
        child2[start:end+1] = parent1[start:end+1]
        
        # Create mapping dictionaries for the segments
        mapping1 = {}  # Maps from parent2 to parent1 in segment
        mapping2 = {}  # Maps from parent1 to parent2 in segment
        
        for i in range(start, end + 1):
            mapping1[parent2[i]] = parent1[i]
            mapping2[parent1[i]] = parent2[i]
        
        # Fill remaining positions for child1
        for i in range(size):
            if i < start or i > end:  # Outside the crossover segment
                # Try to place parent1[i] in child1[i]
                value = parent1[i]
                
                # Check if this value conflicts (already in segment)
                while value in parent2[start:end+1]:
                    # Find non-conflicting value
                    value = mapping1[value]
                
                child1[i] = value
        
        # Fill remaining positions for child2
        for i in range(size):
            if i < start or i > end:  # Outside the crossover segment
                # Try to place parent2[i] in child2[i]
                value = parent2[i]
                
                # Check if this value conflicts (already in segment)
                while value in parent1[start:end+1]:
                    # Find non-conflicting value
                    value = mapping2[value]
                
                child2[i] = value
        
        return child1, child2

    @staticmethod
    def cycle_crossover(parent1, parent2):
        """ Finds cycles in parents and copies them to children. """
        size = len(parent1)

        child1 = [None] * size
        child2 = [None] * size
        
        # Track which positions have been assigned
        visited = [False] * size
        
        while not all(visited):
            # Find first unvisited position
            start = visited.index(False)
            cycle = []
            current = start
            
            # Build the cycle
            while current not in cycle:
                cycle.append(current)
                visited[current] = True
                # Find position of parent1[current] in parent2
                value = parent1[current]
                current = parent2.index(value)
                # If back to start, cycle is complete
                if current == start:
                    break
            
            # Alternate between parents for different cycles
            if len([c for c in cycle if child1[c] is not None]) == 0:
                # First cycle or even cycle - copy from parent1 to child1
                use_parent1_for_child1 = len([i for i in range(size) if child1[i] is not None]) % (size * 2) < size
            else:
                use_parent1_for_child1 = not use_parent1_for_child1
            
            # Simpler approach: alternate cycles
            cycle_num = sum(1 for i in range(start) if visited[i])
            
            if cycle_num % 2 == 0:
                # Even cycle: parent1 -> child1, parent2 -> child2
                for pos in cycle:
                    child1[pos] = parent1[pos]
                    child2[pos] = parent2[pos]
            else:
                # Odd cycle: vice versa
                for pos in cycle:
                    child1[pos] = parent2[pos]
                    child2[pos] = parent1[pos]
        
        return child1, child2


    # Edge Recombination
    @staticmethod
    def edge_recombination(parent1, parent2):
        """ Builds edge table and uses it to create offspring. """
        size = len(parent1)

        # Build edge table
        edge_table = {}
        
        # Initialize edge table for all cities
        for city in parent1:
            edge_table[city] = set()
        
        # Add edges from parent1
        for i in range(size):
            current = parent1[i]
            prev_city = parent1[(i - 1) % size]
            next_city = parent1[(i + 1) % size]
            edge_table[current].add(prev_city)
            edge_table[current].add(next_city)
        
        # Add edges from parent2 and mark common edges
        common_edges = {}
        for city in parent1:
            common_edges[city] = set()
        
        for i in range(size):
            current = parent2[i]
            prev_city = parent2[(i - 1) % size]
            next_city = parent2[(i + 1) % size]
            
            # Check if edges are common (already in table)
            if prev_city in edge_table[current]:
                common_edges[current].add(prev_city)
            else:
                edge_table[current].add(prev_city)
                
            if next_city in edge_table[current]:
                common_edges[current].add(next_city)
            else:
                edge_table[current].add(next_city)
        


        # Build offspring
        offspring = []
        
        # Pick initial element at random
        current = random.choice(parent1)
        offspring.append(current)
        
        # Remove current from all edge lists
        for city in edge_table:
            edge_table[city].discard(current)
            common_edges[city].discard(current)
        
        # Build rest of tour
        while len(offspring) < size:
            # Get candidates from current's edge list
            candidates = list(edge_table.get(current, []))
            
            if candidates:
                # Check for common edges (marked earlier)
                common = [c for c in candidates if c in common_edges.get(current, [])]
                
                if common:
                    # Prefer common edge
                    next_city = random.choice(common)
                else:
                    # Choose entry with shortest edge list
                    min_edges = min(len(edge_table.get(c, [])) for c in candidates)
                    shortest = [c for c in candidates if len(edge_table.get(c, [])) == min_edges]
                    next_city = random.choice(shortest)
            else:
                # Empty list - choose random unvisited city
                remaining = [city for city in parent1 if city not in offspring]
                if not remaining:
                    break
                next_city = random.choice(remaining)
            
            # Add to offspring
            offspring.append(next_city)
            
            # Remove from all edge lists
            for city in edge_table:
                edge_table[city].discard(next_city)
                common_edges[city].discard(next_city)
            
            current = next_city
        
        return offspring




# Testing current crossover functions
def is_valid_tour(tour, original_tour):
    """ Check if tour is valid permutation. """
    return sorted(tour) == sorted(original_tour) and len(tour) == len(original_tour)


def test_crossovers():
    """ Test the crossover functions. """
    # Sample parent tours
    parent1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    parent2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    
    print("Testing Crossover Operators")
    print("=" * 40)
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print()
    
    # Test Order Crossover
    print("Order Crossover (OX):")
    child1, child2 = Crossovers.order_crossover(parent1[:], parent2[:])
    print(f"  Child 1: {child1} - Valid: {is_valid_tour(child1, parent1)}")
    print(f"  Child 2: {child2} - Valid: {is_valid_tour(child2, parent1)}")
    print()
    
    # Test PMX Crossover
    print("PMX Crossover:")
    child1, child2 = Crossovers.pmx_crossover(parent1[:], parent2[:])
    print(f"  Child 1: {child1} - Valid: {is_valid_tour(child1, parent1)}")
    print(f"  Child 2: {child2} - Valid: {is_valid_tour(child2, parent1)}")
    print()

    # Test Cycle Crossover
    print("Cycle Crossover:")
    child1, child2 = Crossovers.cycle_crossover(parent1[:], parent2[:])
    print(f"  Child 1: {child1} - Valid: {is_valid_tour(child1, parent1)}")
    print(f"  Child 2: {child2} - Valid: {is_valid_tour(child2, parent1)}")
    print()
    
    # Test Edge Recombination
    print("Edge Recombination:")
    child = Crossovers.edge_recombination(parent1[:], parent2[:])
    print(f"  Child: {child} - Valid: {is_valid_tour(child, parent1)}")
    print()
    
    print("Running 100 tests for each operator...")
    operators = [
        ("Order Crossover", Crossovers.order_crossover, True),
        ("PMX Crossover", Crossovers.pmx_crossover, True),
        ("Cycle Crossover", Crossovers.cycle_crossover, True),
        ("Edge Recombination", Crossovers.edge_recombination, False)
    ]
    
    for name, func, returns_pair in operators:
        success_count = 0
        for _ in range(100):
            if returns_pair:
                c1, c2 = func(parent1[:], parent2[:])
                if is_valid_tour(c1, parent1) and is_valid_tour(c2, parent1):
                    success_count += 1
            else:
                c = func(parent1[:], parent2[:])
                if is_valid_tour(c, parent1):
                    success_count += 1
        
        print(f"  {name}: {success_count}/100 valid tours")


# Run tests if executed as file
if __name__ == "__main__":
    test_crossovers()
