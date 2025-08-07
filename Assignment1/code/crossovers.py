# Crossover file to be completed by Darcy


from TSP import TSP


# • Implement the different crossover operators 
# (Order Crossover, PMX Crossover, Cycle Crossover, Edge Recombination) 
# for permutations given in the lecture.


# Order Crossover 
# • Informal procedure:
# 1. Choose an arbitrary part from the first parent
# 2. Copy this part to the first child
# 3. Copy the numbers that are not in the first part, to the first child:
#   • starting right from cut point of the copied part,
#   • using the order of the second parent
#   • and wrapping around at the end
# 4. Analogous for the second child, with parent roles reversed


# PMX Crossover
# • Informal procedure for parents P1 and P2:
# 1. Choose random segment and copy it from P1
# 2. Starting from the first crossover point look for elements in that segment of P2 that have not been copied
# 3. For each of these i look in the offspring to see what element j has been copied in its place from P1
# 4. Place i into the position occupied j in P2, since we know that we will not be putting j there (as is already in offspring)
# 5. If the place occupied by j in P2 has already been filled in the offspring k, put i in the position occupied by k in P2
# 6. Having dealt with the elements from the crossover segment, the rest of theoffspring can be filled from P2.
# Second child is created analogously


# Cycle Crossover
# Basic idea: Each allele comes from one parent together with its position.
# • Informal procedure:
# 1. Make a cycle of alleles from P1 in the following way.
#   (a) Start with the first allele of P1.
#   (b) Look at the allele at the same position in P2.
#   (c) Go to the position with the same allele in P1.
#   (d) Add this allele to the cycle.
#   (e) Repeat step b through d until you arrive at the first allele of P1.
# 2. Put the alleles of the cycle in the first child on the positions theyhave in the first parent.
# 3. Take next cycle from second parent


# Edge Recombination
# First Contrust Edge Table -- Refer to Lecture notes


# Informal procedure once edge table is constructed
# 1. Pick an initial element at random and put it in the offspring
# 2. Set the variable current element = entry
# 3. Remove all references to current element from the table
# 4. Examine list for current element:
#   – If there is a common edge, pick that to be next element
#   – Otherwise pick the entry in the list which itself has the shortest list
#   – Ties are split at random
# 5. In the case of reaching an empty list:
#   – Examine the other end of the offspring is for extension
#   – Otherwise a new element is chosen at random





