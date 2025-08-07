from Local_search import LocalSearch
from TSP import TSP

# Example usage of the TSP class using the eil51.tsp file.
tsp = TSP("eil51.tsp")

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