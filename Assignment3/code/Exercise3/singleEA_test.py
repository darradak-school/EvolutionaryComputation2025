import numpy as np
from ioh import get_problem, ProblemClass
from SingleEA import single_objective_ea  # Replace with actual module name

def test_population_size():
    pop, _ = single_objective_ea(problem_id=2100, pop_size=20, budget=10)
    assert len(pop) == 20, "Population size mismatch"

def test_uniform_constraint():
    pop, _ = single_objective_ea(problem_id=2100, pop_size=20, budget=10)
    for ind in pop:
        assert np.sum(ind) == 10, f"Uniform constraint violated: {np.sum(ind)} selected"

def test_fitness_values():
    problem = get_problem(2100, problem_class=ProblemClass.GRAPH)
    pop, _ = single_objective_ea(problem_id=2100, pop_size=20, budget=10)
    for ind in pop:
        fitness = problem(ind)
        assert isinstance(fitness, (int, float)), f"Invalid fitness type: {type(fitness)}"

def test_diversity():
    pop, _ = single_objective_ea(problem_id=2100, pop_size=20, budget=10)
    unique = {tuple(ind) for ind in pop}
    assert len(unique) > 1, "Population lacks diversity"

def run_all_tests():
    print("Running tests...")
    test_population_size()
    print("Population size test passed")
    test_uniform_constraint()
    print("Uniform constraint test passed")
    test_fitness_values()
    print("Fitness evaluation test passed")
    test_diversity()
    print("Diversity test passed")
   

if __name__ == "__main__":
    run_all_tests()