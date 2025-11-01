import ioh
import numpy as np
class MultiObjectiveFitness:
    def __init__(self, problem_id):
        """
        Initialize multi-objective fitness function
        Problem ID examples: 2100, 2200, etc.
        1st objective: submodular function value (maximize)
        """
        self.problem = ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)
    def evaluate(self, solution):
        """
        Evaluate One Solution
        Solution: binary array (0 or 1)
        Returns:
            (f1, f2):
            f1 = submodular function value (need maximize)
            f2 = cost (number of selected nodes)
        """
        # calculate one objective: submodular function value
        f1 = self.problem(solution)

        # Calculate Second Objective: Cost (number of selected nodes)
        f2 = sum(solution)  # Because each node has cost=1

        return f1, f2
    def dominates(self, obj1, obj2):
        """
        Check obj1 is dominate obj2:
        for minimize f2 and maximize f1:
        obj1 dominates obj2 if:
        - obj1.f1 >= obj2.f1 AND obj1.f2 <= obj2.f2
        - And at least one of them is strictly better
        - and at least one objective is strictly better
        Args:
            obj1: (f1, f2) tuple
            obj2: (f1, f2) tuple

        Returns:
            True if obj1 dominates obj2
        """
        f1_1, f2_1 = obj1
        f1_2, f2_2 = obj2

        better_or_equal = (f1_1 >= f1_2) and (f2_1 <= f2_2)

        # and at least one objective is strictly better
        strictly_better = (f1_1 > f1_2) or (f2_1 < f2_2)

        return better_or_equal and strictly_better

if __name__ == "__main__":
    fitness_func = MultiObjectiveFitness(2100)
    solution = np.random.randint(0, 2, size=10000)
    f1, f2 = fitness_func.evaluate(solution)

    print(f"Coverage: {f1}, Cost: {f2}")