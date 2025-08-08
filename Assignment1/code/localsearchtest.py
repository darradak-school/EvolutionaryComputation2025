from localsearch import LocalSearch
from tsp import TSP
import time

# Change the problem to use a different TSP file.
# problem = "eil51"
problem = "st70"
# problem = "eil76"
# problem = "kroA100"
# problem = "korC100"
# problem = "kroD100"
# problem = "eil101"
# problem = "lin105"
# problem = "pcb442"
# problem = "pr2392"
# problem = "usa13509"

tsp = TSP(f"tsplib/{problem}.tsp")

# Initialize the local search algorithm.
local_search = LocalSearch()

# Initialize variables to track minimum and average tour lengths.
min_jump_best = float('inf')
min_jump_first = float('inf')
min_exchange_best = float('inf')
min_exchange_first = float('inf')
min_2opt_best = float('inf')
min_2opt_first = float('inf')
# Total tour lengths for calculating average
total_jump_best = 0
total_jump_first = 0
total_exchange_best = 0
total_exchange_first = 0
total_2opt_best = 0
total_2opt_first = 0
# Total cycle times for calculating average
total_jump_best_cycles = 0
total_jump_first_cycles = 0
total_exchange_best_cycles = 0
total_exchange_first_cycles = 0
total_2opt_best_cycles = 0
total_2opt_first_cycles = 0
# Total time taken for each search type
total_jump_best_time = 0
total_jump_first_time = 0
total_exchange_best_time = 0
total_exchange_first_time = 0
total_2opt_best_time = 0
total_2opt_first_time = 0

# Start time tracker
start_time = time.time()

# Running local search with best and first improvement 30 times, using exchange, jump, and 2-opt.
# Tracking minimum and average tour lengths and cycle times for each search type.
for i in range(30):
    # Generate a random tour.
    random_tour = tsp.random_tour()

    # Iteration tracker
    print(f"Iteration {i + 1} of 30")

    # Run local search with best and first improvement for each search type.
    local_search_best_jump = local_search.local_search(random_tour, 'jump', 'best', tsp)
    local_search_first_jump = local_search.local_search(random_tour, 'jump', 'first', tsp)
    local_search_best_exchange = local_search.local_search(random_tour, 'exchange', 'best', tsp)
    local_search_first_exchange = local_search.local_search(random_tour, 'exchange', 'first', tsp)
    local_search_best_2opt = local_search.local_search(random_tour, 'two_opt', 'best', tsp)
    local_search_first_2opt = local_search.local_search(random_tour, 'two_opt', 'first', tsp)

    # Tracking minimum tour length for each search type
    if tsp.tour_length(local_search_best_jump[0]) < min_jump_best:
        min_jump_best = tsp.tour_length(local_search_best_jump[0])
    if tsp.tour_length(local_search_first_jump[0]) < min_jump_first:
        min_jump_first = tsp.tour_length(local_search_first_jump[0])
    if tsp.tour_length(local_search_best_exchange[0]) < min_exchange_best:
        min_exchange_best = tsp.tour_length(local_search_best_exchange[0])
    if tsp.tour_length(local_search_first_exchange[0]) < min_exchange_first:
        min_exchange_first = tsp.tour_length(local_search_first_exchange[0])
    if tsp.tour_length(local_search_best_2opt[0]) < min_2opt_best:
        min_2opt_best = tsp.tour_length(local_search_best_2opt[0])
    if tsp.tour_length(local_search_first_2opt[0]) < min_2opt_first:
        min_2opt_first = tsp.tour_length(local_search_first_2opt[0])
    
    # Tracking total tour length for each search type
    total_jump_best += tsp.tour_length(local_search_best_jump[0])
    total_jump_first += tsp.tour_length(local_search_first_jump[0])
    total_exchange_best += tsp.tour_length(local_search_best_exchange[0])
    total_exchange_first += tsp.tour_length(local_search_first_exchange[0])
    total_2opt_best += tsp.tour_length(local_search_best_2opt[0])
    total_2opt_first += tsp.tour_length(local_search_first_2opt[0])

    # Tracking total cycle times for each search type
    total_jump_best_cycles += local_search_best_jump[1]
    total_jump_first_cycles += local_search_first_jump[1]
    total_exchange_best_cycles += local_search_best_exchange[1]
    total_exchange_first_cycles += local_search_first_exchange[1]
    total_2opt_best_cycles += local_search_best_2opt[1]
    total_2opt_first_cycles += local_search_first_2opt[1]

    # Tracking total time taken for each search type
    total_jump_best_time += local_search_best_jump[2]
    total_jump_first_time += local_search_first_jump[2]
    total_exchange_best_time += local_search_best_exchange[2]
    total_exchange_first_time += local_search_first_exchange[2]
    total_2opt_best_time += local_search_best_2opt[2]
    total_2opt_first_time += local_search_first_2opt[2]

    # Print time taken for each iteration
    print(f'time taken: {time.time() - start_time:.2f} seconds')

# Calculate average tour length for each search type
avg_jump_best = total_jump_best / (i + 1)
avg_jump_first = total_jump_first / (i + 1)
avg_exchange_best = total_exchange_best / (i + 1)
avg_exchange_first = total_exchange_first / (i + 1)
avg_2opt_best = total_2opt_best / (i + 1)
avg_2opt_first = total_2opt_first / (i + 1)

# Calculate average cycle time for each search type
avg_jump_best_cycles = total_jump_best_cycles / (i + 1)
avg_jump_first_cycles = total_jump_first_cycles / (i + 1)
avg_exchange_best_cycles = total_exchange_best_cycles / (i + 1)
avg_exchange_first_cycles = total_exchange_first_cycles / (i + 1)
avg_2opt_best_cycles = total_2opt_best_cycles / (i + 1)
avg_2opt_first_cycles = total_2opt_first_cycles / (i + 1)

# Calculate average time taken for each search type
avg_jump_best_time = total_jump_best_time / (i + 1)
avg_jump_first_time = total_jump_first_time / (i + 1)
avg_exchange_best_time = total_exchange_best_time / (i + 1)
avg_exchange_first_time = total_exchange_first_time / (i + 1)
avg_2opt_best_time = total_2opt_best_time / (i + 1)
avg_2opt_first_time = total_2opt_first_time / (i + 1)

# Print results to local_search.txt
with open("../results/local_search.txt", "a") as file:
    file.write(f"\n######### {problem} #########\n")
    file.write(f"===== MEAN =====\n")
    file.write(f"== Jump ==\n")
    file.write(f"Best-improvement: {avg_jump_best:.2f} | First-improvement: {avg_jump_first:.2f}\n")
    file.write(f"== Exchange ==\n")
    file.write(f"Best-improvement: {avg_exchange_best:.2f} | First-improvement: {avg_exchange_first:.2f}\n")
    file.write(f"== 2-opt ==\n")
    file.write(f"Best-improvement: {avg_2opt_best:.2f} | First-improvement: {avg_2opt_first:.2f}\n")
    file.write(f"===== MINIMUM =====\n")
    file.write(f"== Jump ==\n")
    file.write(f"Best-improvement: {min_jump_best:.2f} | First-improvement: {min_jump_first:.2f}\n")
    file.write(f"== Exchange ==\n")
    file.write(f"Best-improvement: {min_exchange_best:.2f} | First-improvement: {min_exchange_first:.2f}\n")
    file.write(f"== 2-opt ==\n")
    file.write(f"Best-improvement: {min_2opt_best:.2f} | First-improvement: {min_2opt_first:.2f}\n")
    file.write(f"===== MEAN TIMES =====\n")
    file.write(f"== Jump ==\n")
    file.write(f"Best-improvement cycles: {avg_jump_best_cycles:.2f} | First-improvement cycles: {avg_jump_first_cycles:.2f}\n")
    file.write(f"Best-improvement time: {avg_jump_best_time:.2f} | First-improvement time: {avg_jump_first_time:.2f}\n")
    file.write(f"== Exchange ==\n")
    file.write(f"Best-improvement cycles: {avg_exchange_best_cycles:.2f} | First-improvement cycles: {avg_exchange_first_cycles:.2f}\n")
    file.write(f"Best-improvement time: {avg_exchange_best_time:.2f} | First-improvement time: {avg_exchange_first_time:.2f}\n")
    file.write(f"== 2-opt ==\n")
    file.write(f"Best-improvement cycles: {avg_2opt_best_cycles:.2f} | First-improvement cycles: {avg_2opt_first_cycles:.2f}\n")
    file.write(f"Best-improvement time: {avg_2opt_best_time:.2f} | First-improvement time: {avg_2opt_first_time:.2f}\n")