import math
import random
import numpy as np


class TSP:
    def __init__(self, file):
        """ Initialize the TSP object with the given file. """
        self.locations = self.read_tsp(file)
        self.location_ids = list(self.locations.keys())

    def read_tsp(self, file):
        """ Read the TSP file and extract coordinates, puts results into a dictionary that it returns. """
        locations = {}
        with open(file, "r") as tsp:
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

    def dist(self, location1, location2):
        """ Calculate euclidean distance between two points. """
        x1, y1 = self.locations[location1]  # Get coordinates of first location.
        x2, y2 = self.locations[location2]  # Get coordinates of second location.
        return math.hypot(x2 - x1, y2 - y1)  # Calculate euclidean distance.

    def tour_length(self, tour):
        """ Calculate entire tour length. """
        total = sum(self.dist(tour[i], tour[i + 1]) for i in range(len(tour) - 1))
        total += self.dist(tour[-1], tour[0])  # Add distance to return to start.
        return total

    def random_tour(self):
        """ Create a random tour. """
        return random.sample(self.location_ids, len(self.location_ids))

    def get_idx(self, loc_id):
        """ Get 0 based index of a location from its ID. """
        return self.location_ids.index(loc_id)

    def tour_idxs(self, tour):
        """ Convert tour of location IDs to 0 based index for distance matrix. """
        return [self.get_idx(loc_id) for loc_id in tour]
