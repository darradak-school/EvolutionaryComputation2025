import math
import random

class TSP:
    def __init__(self, filepath):
        # Read the TSP file and store the coordinates of locations.
        self.location_coords = self.read_tsp_file(filepath)
        # Extract location IDs from the coordinates dictionary.
        self.location_ids = list(self.location_coords.keys())
        # Calculates the distances between all locations and stores them in a dictionary.
        self.distances = self.calculate_distances()

    # This function reads a TSP file and extracts the coordinates of the nodes in a dictionary as {location id : coordinates in tuple}.
    def read_tsp_file(self, file_path):
        locations = {}
        with open(file_path, 'r') as tsp:
            read = False
            for line in tsp:
                line = line.strip()
                if line == 'EOF' or line == '':
                    break
                if read:
                    parts = line.split()
                    loc_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    locations[loc_id] = (x_coord, y_coord)
                if line == 'NODE_COORD_SECTION':
                    read = True
        return locations

    # Calculate the distances between all locations, storing them in a dictionary.
    def calculate_distances(self):
        distances = {}
        for i in self.location_ids:
            for j in self.location_ids:
                if i == j:
                    distances[i, j] = 0.0
                elif (j, i) in distances:
                    distances[i, j] = distances[j, i]
                else:
                    distances[i, j] = self.euclidian_distance(i, j)
        return distances
    
    # Get the distance between two locations.
    def distance(self, location1, location2):
        return self.distances[location1, location2]
    
    # Calculate the Euclidean distance between two points.
    def euclidian_distance(self, location1, location2):
        x1, y1 = self.location_coords[location1]
        x2, y2 = self.location_coords[location2]
        return math.hypot(x2 - x1, y2 - y1)

    # Calculate the tour length given a list of locations.
    def tour_length(self, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += self.distance(tour[i], tour[i + 1])
        length += self.distance(tour[-1], tour[0])
        return length

    # Generate a random tour.
    def random_tour(self):
        return random.sample(self.location_ids, len(self.location_ids))