import math
import random

class TSP:
    def __init__(self, filepath):
        # Read the TSP file and store the coordinates of locations.
        self.location_coords = self.read_tsp_file(filepath)
        # Extract location IDs from the coordinates dictionary.
        self.location_ids = list(self.location_coords.keys())

    # This function reads a TSP file and extracts the coordinates of the nodes in a dictionary as {location id : coordinates in tuple}.
    def read_tsp_file(self, file_path):
        locations = {}
        with open(file_path, 'r') as tsp:
            read = False
            for line in tsp:
                line = line.strip()
                if line == 'EOF':
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

    # Calculate the Euclidean distance between two points.
    def euclidian_distance(self, location1, location2):
        x1, y1 = self.location_coords[location1]
        x2, y2 = self.location_coords[location2]
        return math.hypot(x2 - x1, y2 - y1)

    # Calculate the tour length given a list of locations.
    def tour_length(self, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += self.euclidian_distance(tour[i], tour[i + 1])
        length += self.euclidian_distance(tour[-1], tour[0])
        return length

    # Generate a random tour.
    def random_tour(self):
        return random.sample(self.location_ids, len(self.location_ids))