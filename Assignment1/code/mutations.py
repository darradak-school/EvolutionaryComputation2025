# Mutations file to be completed by Tyler

import random


# insert: pick two random alleles, move second to follow first.
# shift rest along. preserves order and adjacency info. 

def insert_mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    new_tour = tour[:]

    if i > j:
        to_move = new_tour.pop(i)
        insert_pos = j + 1
    else:
        to_move = new_tour.pop(j)
        insert_pos = i + 1

    new_tour.insert(insert_pos, to_move)

    return new_tour, i, j



# swap: pick two alleles at random.
# swap positions. I.e. exchange 

def swap_mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

# inversion: pick two alleles at random, invert substring (inclusive)
# preserves adjacency info, disruptive to order. 
