# Crossover file to be completed by Darcy

from tsp import TSP

import random


# Crossover functions implemented as per assignment specs
# Order Crossover 
def order_crossover(parent1, parent2):
    """
    Order Crossover (OX)
    Preserves relative order of cities from parents
    """
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
def pmx_crossover(parent1, parent2):
    """
    Partially Mapped Crossover (PMX)
    Creates mapping between segment positions
    """
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
                # Follow the mapping to find non-conflicting value
                value = mapping1[value]
            
            child1[i] = value
    
    # Fill remaining positions for child2
    for i in range(size):
        if i < start or i > end:  # Outside the crossover segment
            # Try to place parent2[i] in child2[i]
            value = parent2[i]
            
            # Check if this value conflicts (already in segment)
            while value in parent1[start:end+1]:
                # Follow the mapping to find non-conflicting value
                value = mapping2[value]
            
            child2[i] = value
    
    return child1, child2


# Cycle Crossover
# Basic idea: Each allele comes from one parent together with its position.
# • Informal procedure:
# 1. Make a cycle of alleles from P1 in the following way.
#   (a) Start with the first allele of P1.
#   (b) Look at the allele at the same position in P2.
#   (c) Go to the position with the same allele in P1.
#   (d) Add this allele to the cycle.
#   (e) Repeat step b through d until you arrive at the first allele of P1.
# 2. Put the alleles of the cycle in the first child on the positions theyhave in the first parent.
# 3. Take next cycle from second parent


# Edge Recombination
# First Contrust Edge Table -- Refer to Lecture notes


# Informal procedure once edge table is constructed
# 1. Pick an initial element at random and put it in the offspring
# 2. Set the variable current element = entry
# 3. Remove all references to current element from the table
# 4. Examine list for current element:
#   – If there is a common edge, pick that to be next element
#   – Otherwise pick the entry in the list which itself has the shortest list
#   – Ties are split at random
# 5. In the case of reaching an empty list:
#   – Examine the other end of the offspring is for extension
#   – Otherwise a new element is chosen at random




# Testing current crossover functions
# Test functions for verification
def is_valid_tour(tour, original_tour):
    """Check if tour is valid permutation"""
    return sorted(tour) == sorted(original_tour) and len(tour) == len(original_tour)


def test_crossovers():
    """Test all crossover operators"""
    # Create sample parent tours
    parent1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    parent2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    
    print("Testing Crossover Operators")
    print("=" * 40)
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print()
    
    # Test Order Crossover
    print("Order Crossover (OX):")
    child1, child2 = order_crossover(parent1[:], parent2[:])
    print(f"  Child 1: {child1} - Valid: {is_valid_tour(child1, parent1)}")
    print(f"  Child 2: {child2} - Valid: {is_valid_tour(child2, parent1)}")
    print()
    
    # Test PMX Crossover
    print("PMX Crossover:")
    child1, child2 = pmx_crossover(parent1[:], parent2[:])
    print(f"  Child 1: {child1} - Valid: {is_valid_tour(child1, parent1)}")
    print(f"  Child 2: {child2} - Valid: {is_valid_tour(child2, parent1)}")
    print()
    
    # Run multiple tests to check consistency
    print("Running 100 tests for each operator...")
    operators = [
        ("Order Crossover", order_crossover, True),
        ("PMX Crossover", pmx_crossover, True)
        # Add Cycle and Edge funcs to test
        #("Cycle Crossover", cycle_crossover, True),
        #("Edge Recombination", edge_recombination, False)
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


# Run tests if this file is executed directly
if __name__ == "__main__":
    test_crossovers()
