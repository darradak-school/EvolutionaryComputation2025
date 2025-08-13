import math
import random


class TSP:
    def __init__(self, filepath):
        # Read the TSP file using read_tsp and store the information in a dictionary {key: location id, value: coordinates}.
        self.locations = self.read_tsp(filepath)
        # Extract location IDs from the coordinates dictionary.
        self.location_ids = list(self.locations.keys())

    # Read TSP file and extract coordinates, puts results into a dictionary that it returns.
    def read_tsp(self, file_path):
        locations = {}
        with open(file_path, "r") as tsp:
            read = False
            for line in tsp:
                line = line.strip()
                if line == "EOF":  # End of file, break loop.
                    break
                if line == "NODE_COORD_SECTION":  # Start of coordinates section.
                    read = True
                    continue
                if read and line:
                    loc_parts = (
                        line.split()
                    )  # Splitting the line and storing in locations dictionary.
                    if len(loc_parts) >= 3:
                        loc_id, x_coord, y_coord = (
                            int(loc_parts[0]),
                            float(loc_parts[1]),
                            float(loc_parts[2]),
                        )
                        locations[loc_id] = (x_coord, y_coord)  # Store the coordinates.
        return locations

    # Calculate euclidean distance between two points.
    def dist(self, location1, location2):
        x1, y1 = self.locations[location1]  # Get coordinates of first location.
        x2, y2 = self.locations[location2]  # Get coordinates of second location.
        return math.hypot(x2 - x1, y2 - y1)  # Calculate euclidean distance.

    # Calculate entire tour length.
    def tour_length(self, tour):
        total = sum(self.dist(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
        total += self.dist(tour[-1], tour[0])  # Add distance to return to start.
        return total

    # Create a random tour.
    def random_tour(self):
        return random.sample(self.location_ids, len(self.location_ids))
