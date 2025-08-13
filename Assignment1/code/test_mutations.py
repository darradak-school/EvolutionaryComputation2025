from mutations import swap_mutation, insert_mutation
import random

def test_swap():
    original_tour = list(range(1, 11))  # Example tour [1..10]

    success_count = 0
    iterations = 20  # Reduced so output isn't overwhelming

    print("Testing swap mutation operator...\n")

    for test_num in range(1, iterations + 1):
        mutated_tour = swap_mutation(original_tour.copy())

        # Check 1: Same elements, no duplicates or missing
        elements_ok = sorted(mutated_tour) == sorted(original_tour)
        
        # Check 2: Exactly two positions differ
        diffs = [i for i in range(len(original_tour)) if original_tour[i] != mutated_tour[i]]
        two_positions_changed = (len(diffs) == 2)
        
        # Check 3: Elements at those positions swapped
        swapped_correctly = False
        if two_positions_changed:
            i, j = diffs
            swapped_correctly = (original_tour[i] == mutated_tour[j] and original_tour[j] == mutated_tour[i])
        
        if elements_ok and two_positions_changed and swapped_correctly:
            success_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"Test {test_num}: {status}")
        print(f"  Original: {original_tour}")
        print(f"  Mutated : {mutated_tour}")
        print(f"  Differences at positions: {diffs}\n")

    print(f"Swap mutation passed {success_count}/{iterations} tests.")

import random
from mutations import inversion_mutation  # adjust import as needed

def test_inversion_mutation():
    original_tour = list(range(1, 11))  # Example tour [1..10]
    tests = 20
    success_count = 0

    print("Testing inversion mutation operator...\n")

    for test_num in range(1, tests + 1):
        # Seed random for reproducibility (optional)
        # random.seed(test_num)

        mutated_tour = inversion_mutation(original_tour)
        
        # We can't control i,j here because random inside function, so extract them by comparing
        # To find i and j, compare mutated and original:
        diffs = [idx for idx, (o, m) in enumerate(zip(original_tour, mutated_tour)) if o != m]

        if len(diffs) == 0:
            # No change means i==j (should not happen because random.sample picks two distinct)
            status = "FAIL (no mutation)"
            print(f"Test {test_num}: {status}\n")
            continue
        
        i = diffs[0]
        j = diffs[-1]

        # Check substring reversed:
        original_sub = original_tour[i:j+1]
        mutated_sub = mutated_tour[i:j+1]
        reversed_correctly = (mutated_sub == original_sub[::-1])

        # Check rest unchanged:
        unchanged_before = (original_tour[:i] == mutated_tour[:i])
        unchanged_after = (original_tour[j+1:] == mutated_tour[j+1:])

        # Check elements same:
        elements_ok = sorted(original_tour) == sorted(mutated_tour)

        if reversed_correctly and unchanged_before and unchanged_after and elements_ok:
            success_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"Test {test_num}: {status}")
        print(f"  Selected indices i={i}, j={j}")
        print(f"  Original: {original_tour}")
        print(f"  Mutated : {mutated_tour}")
        print(f"  Differences at positions: {diffs}\n")

    print(f"Inversion mutation passed {success_count}/{tests} tests.")

# Run the test
if __name__ == "__main__":
    test_inversion_mutation()
    test_swap()
