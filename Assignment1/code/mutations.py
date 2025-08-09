# Mutations file to be completed by Tyler


# insert: pick two random alleles, move second to follow first.
# shift rest along. preserves order and adjacency info. 

def insert(self, tour, i,  j):
    if i == j:              # no change if i, j are equal
        return tour[:]
    if i > j:               # Validate which is first and which is second. 
        i, j = j, i
    new_tour = tour[:]
    city = new_tour.pop(j)
    insert_index = (i + 1) % len(new_tour)
    new_tour.insert(insert_index, city)
    


# swap: pick two alleles at random.
# swap positions. I.e. exchange 

def swap(self, tour, i, j):
    if i == j:    
        return tour[:] # no change if same position
    new_tour = tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# inversion: pick two alleles at random, invert substring (inclusive)
# preserves adjacency info, disruptive to order. 

