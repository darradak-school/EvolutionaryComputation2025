from localsearch import LocalSearch
from tsp import TSP

# Example usage of the TSP class using the eil51.tsp file, all other files are commented out, can be uncommented to test other files.
tsp = TSP("tsplib/eil51.tsp")
# tsp = TSP("tsplib/st70.tsp")
# tsp = TSP("tsplib/eil76.tsp")
# tsp = TSP("tsplib/kroA100.tsp")
# tsp = TSP("tsplib/korC100.tsp")
# tsp = TSP("tsplib/kroD100.tsp")
# tsp = TSP("tsplib/eil101.tsp")
# tsp = TSP("tsplib/lin105.tsp")
# tsp = TSP("tsplib/pcb442.tsp")
# tsp = TSP("tsplib/pr2392.tsp")
# tsp = TSP("tsplib/usa13509.tsp")

# Print the coordinates of a specific location
print(tsp.location_coords[1])

# Calculate and print the Euclidean distance between two locations
print(tsp.euclidian_distance(1, 2))

# Example tour to calculate its length
tour = [1, 2, 3, 4, 5]
print("Tour length:", tsp.tour_length(tour))

# Example usage of the LocalSearch class
local_search = LocalSearch()

# Generate neighbours using different search types
exchange_neighbours = local_search.neighbourhood(tour, 'exchange')
jump_neighbours = local_search.neighbourhood(tour, 'jump')
inversion_neighbours = local_search.neighbourhood(tour, 'inversion')
two_opt_neighbours = local_search.neighbourhood(tour, 'two_opt')
print("Exchange Neighbours:", exchange_neighbours)
print("Jump Neighbours:", jump_neighbours)
print("Inversion Neighbours:", inversion_neighbours)
print("Two Opt Neighbours:", two_opt_neighbours)

# Finding best neighbour using exchange, with coordinates from tsp
best_neighbour = local_search.best(tour, 'exchange', tsp)
print("Best Neighbour, Exchange:", best_neighbour)

# Finding first better neighbour using exchange, with coordinates from tsp
first_neighbour = local_search.first(tour, 'exchange', tsp)
print("First Neighbour, Exchange:", first_neighbour)

# Running local search with best improvement
local_search_best = local_search.local_search(tour, 'exchange', 'best', tsp)
print("Local Search Best, Exchange:", local_search_best[0], "Cycles taken:", local_search_best[1])

# Running local search with first improvement
local_search_first = local_search.local_search(tour, 'exchange', 'first', tsp)
print("Local Search First, Exchange:", local_search_first[0], "Cycles taken:", local_search_first[1])

# Generate a random tour
random_tour = tsp.random_tour()
print("Random Tour:", random_tour)

# Finding best neighbour using exchange, with coordinates from tsp
best_neighbour = local_search.best(random_tour, 'exchange', tsp)
print("Best Neighbour, Exchange:", best_neighbour)

# Finding first better neighbour using exchange, with coordinates from tsp
first_neighbour = local_search.first(random_tour, 'exchange', tsp)
print("First Neighbour, Exchange:", first_neighbour)

# Running local search with best improvement
local_search_best = local_search.local_search(random_tour, 'exchange', 'best', tsp)
print(f'''
Local search: Best, Exchange: {local_search_best[0]}
Cycles taken: {local_search_best[1]}
Original tour length: {tsp.tour_length(random_tour)}
Tour length: {tsp.tour_length(local_search_best[0])}''')
    
# Running local search with first improvement
local_search_first = local_search.local_search(random_tour, 'exchange', 'first', tsp)
print(f'''
Local search: First, Exchange: {local_search_first[0]}
Cycles taken: {local_search_first[1]}
Original tour length: {tsp.tour_length(random_tour)}
Tour length: {tsp.tour_length(local_search_first[0])}''')

# Printing the distance between two locations
print(tsp.distance(1, 2))

# Print the distances between all locations
print(tsp.distances)