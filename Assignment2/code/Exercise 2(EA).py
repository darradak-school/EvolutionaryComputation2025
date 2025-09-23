import os, random
import numpy as np
from ioh import get_problem, logger, ProblemClass
def Uniform_Mutation(parent, mutation_rate=None):
    mutated = parent.copy()
    mutation_rate = mutation_rate or len(parent)
    return [1 - mutated[i] if random.random() < (1 / mutation_rate) else mutated[i] for i in range(len(parent))]
def Fitness_Evaluate(problem, solution):
    return problem(solution)
def Evolutionary_Algorithm(Total_Length=16, budget=100000):
    problem = get_problem("OneMax", instance=1, dimension=Total_Length, problem_class=ProblemClass.PBO)
    parent = [random.randint(0, 1) for i in range(Total_Length)]
    parent_fitness = Fitness_Evaluate(problem, parent)
    evaluations = 0
    while budget > evaluations:
        children = Uniform_Mutation(parent, Total_Length)
        children_fitness = Fitness_Evaluate(problem, children)
        if children_fitness >= parent_fitness:
            parent, parent_fitness = children, children_fitness
        evaluations += 1
    return parent, parent_fitness
Evolutionary_Algorithm()