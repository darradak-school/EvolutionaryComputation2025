# Basic demo for Evolutionary Algorithm
from evolutionary import EvolutionaryAlgorithm

print("EA Test")
print("=" * 20)

# Create the ea
ea = EvolutionaryAlgorithm(
    tsp_file="tsplib/eil51.tsp", population_size=15, max_generations=15
)

# Run the ea
print("Running...")
best = ea.run()

# Show result
print(f"Done! Best tour length: {best.fitness:.2f}")