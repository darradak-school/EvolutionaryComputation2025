# Basic demo for Evolutionary Algorithm
from evolutionary import EvolutionaryAlgorithm

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
]
POPULATION = 100
GENERATIONS = 100

# Create the ea
for problem in PROBLEMS:
    ea = EvolutionaryAlgorithm(
        f"tsplib/{problem}.tsp", POPULATION, GENERATIONS
    )
    print("=" * 20)
    print(f"Running {problem} with population {POPULATION} and generations {GENERATIONS}")
    evolve = ea.run()
    