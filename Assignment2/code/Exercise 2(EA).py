import os, random
import numpy as np
from ioh import get_problem, logger, ProblemClass
def Uniform_Mutation(parent, mutation_rate=None):
    mutated = np.copy(parent)
    mutation_rate = mutation_rate or len(parent)
    return np.array([1 - mutated[i] if random.random() < (1 / mutation_rate) else mutated[i] for i in range(len(parent))])
def Fitness_Evaluate(problem, solution):    return problem(solution)
def Evolutionary_Algorithm(Total_Length=16, budget=100000, Problem_Name="OneMax", Problem_ID=1):
    IOH_Logger = logger.Analyzer(root=os.getcwd(), folder_name=f"EA_{Problem_Name}", store_positions=True)
    # Create OneMax problem
    problem = get_problem(Problem_Name, instance=Problem_ID, dimension=Total_Length, problem_class=ProblemClass.PBO)
    problem.attach_logger(IOH_Logger)
    # Create Solution List
    parent = np.random.randint(0, 2, size=Total_Length)
    # Evaluate Initial Solution
    parent_fitness = Fitness_Evaluate(problem, parent)
    evaluations = 1
    # Loop until budget is exhausted
    while budget > evaluations:
        children = Uniform_Mutation(parent, Total_Length)
        children_fitness = Fitness_Evaluate(problem, children)
        if children_fitness >= parent_fitness:
            parent, parent_fitness = children, children_fitness
        evaluations += 1
    IOH_Logger.close()
    return parent, parent_fitness
if __name__ == "__main__":
    functions = []
    functions.append(("OneMax", 1))
    functions.append(("LeadingOnes", 2))
    functions.append(("BinaryValue", 3))
    functions.append(("Labs", 18))
    Evolutionary_Algorithm()
