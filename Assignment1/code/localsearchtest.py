from localsearch import LocalSearch
from tsp import TSP
import time

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
def write_results(file, section_name, results, search_types):
    file.write(f"===== {section_name} =====\n")
    for search_type in search_types:
        data = results[search_type]
        avg, min_v, avg_cycles, avg_time = calc(data)
        if section_name == "MEAN":
            file.write(f"{search_type.title()}: {avg:.2f}\n")
        elif section_name == "MINIMUM":
            file.write(f"{search_type.title()}: {min_v:.2f}\n")
        elif section_name == "MEAN TIMES":
            file.write(f"{search_type.title()} cycles: {avg_cycles:.2f}\n")
            file.write(f"{search_type.title()} time: {avg_time:.2f}s\n")
            file.write("---------------------\n")


# Run tests on all problems.
for problem in PROBLEMS:
    start = time.time()
    print(f"Running {problem}...")

    # Create the problem instance.
    tsp = TSP(f"tsplib/{problem}.tsp")
    location_count = len(tsp.location_ids)

    # Create random tours to get average unoptimised tour length.
    tours = []
    for i in range(10):
        t = tsp.random_tour()
        tours.append(tsp.tour_length(t))

    # Results dictionary for each search type.
    results = {search_type: [] for search_type in TYPES}

    # Create local search object.
    local_search = LocalSearch()

    for search_type in TYPES:
        # Run a few times to get average results.
        for i in range(INSTANCES):
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
    with open("../results/local_search.txt", "a") as file:
        file.write(f"\n######### {problem} #########\n")
        file.write(f"Average tour length before optimisation: {baseline_avg:.2f}\n")
        write_results(file, "MEAN", results, TYPES)
        write_results(file, "MINIMUM", results, TYPES)
        write_results(file, "MEAN TIMES", results, TYPES)
