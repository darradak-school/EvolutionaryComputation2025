# opulation-based multi-objective evolutionary algorithm

# Multi-Objective EA for Exercise 3
# Simple implementation of a multi-objective EA
# two objectives: maximize f(S), minimize |S|
# Tournament selection (size 3)
# Uniform crossover (90% rate)
# Bit-flip mutation (1/n rate)
# Simple non-dominated sorting for selection
# Population sizes: 10, 20, 50 (20 works best)

import os
import random
import numpy as np
from ioh import get_problem, ProblemClass, logger


# === Core Algorithm ===
def multi_objective_ea(problem_id=2100, pop_size=20, run_index=0):
    """
    Multi-objective EA using f(S) and -|S| as objectives.
    Includes correct IOH logging setup for the current IOH version.
    """
    algo_name = f"MultiEA_pop{pop_size}_run{run_index+1}"

    # ✅ Create proper run folder: data/MultiEA_Exercise3/MultiEA_pop20_run1/
    run_folder = os.path.join("data", "MultiEA_Exercise3", algo_name)
    os.makedirs(run_folder, exist_ok=True)

    # ✅ Create logger with correct parameters
    l = logger.Analyzer(
        root=logger.Path(run_folder),
        folder_name="",           # prevents IOHProfiler from adding "-1" suffixes
        algorithm_name=algo_name,
        store_positions=True
    )

    # === Problem setup ===
    problem = get_problem(problem_id, problem_class=ProblemClass.GRAPH)
    problem.attach_logger(l)

    n = problem.meta_data.n_variables
    max_evals = 10000
    mutation_rate = 1.0 / n

    # Initialize population
    population, objectives = [], []
    for _ in range(pop_size):
        num_ones = random.randint(0, n)
        individual = np.zeros(n, dtype=int)
        if num_ones > 0:
            indices = random.sample(range(n), num_ones)
            individual[indices] = 1

        f_val = max(0, problem(individual))  # handle invalid negatives
        population.append(individual)
        objectives.append((f_val, -np.sum(individual)))

    evals = pop_size
    best_f_history = []

    # === Evolutionary Loop ===
    while evals < max_evals:
        offspring, offspring_objs = [], []
        for _ in range(pop_size):
            p1 = tournament_select(population, objectives)
            p2 = tournament_select(population, objectives)

            # Crossover & Mutation
            child = crossover(p1, p2) if random.random() < 0.9 else p1.copy()
            child = mutate(child, mutation_rate)

            f_val = max(0, problem(child))
            offspring.append(child)
            offspring_objs.append((f_val, -np.sum(child)))

            evals += 1
            if evals >= max_evals:
                break

        combined_pop = population + offspring
        combined_objs = objectives + offspring_objs
        population, objectives = select_next_generation(combined_pop, combined_objs, pop_size)

        best_f = max(obj[0] for obj in objectives)
        best_f_history.append(best_f)

    # === Final Pareto Front ===
    pareto_front, pareto_objs = [], []
    for i in range(len(population)):
        if not any(dominates(objectives[j], objectives[i]) for j in range(len(population)) if i != j):
            pareto_front.append(population[i])
            pareto_objs.append(objectives[i])

    # ✅ Correctly detach logger (no args)
    problem.detach_logger()

    return population, objectives, best_f_history, pareto_front, pareto_objs


# === Helper Functions ===
def dominates(obj1, obj2):
    return obj1[0] >= obj2[0] and obj1[1] >= obj2[1] and (obj1[0] > obj2[0] or obj1[1] > obj2[1])


def tournament_select(population, objectives, tournament_size=3):
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    best_idx = indices[0]
    for idx in indices[1:]:
        if dominates(objectives[idx], objectives[best_idx]):
            best_idx = idx
    return population[best_idx].copy()


def crossover(p1, p2):
    child = np.where(np.random.rand(len(p1)) < 0.5, p1, p2)
    return child


def mutate(individual, rate):
    child = individual.copy()
    flip_mask = np.random.rand(len(child)) < rate
    child[flip_mask] = 1 - child[flip_mask]
    return child


def select_next_generation(population, objectives, pop_size):
    selected_pop, selected_objs = [], []
    remaining = list(range(len(population)))

    while len(selected_pop) < pop_size and remaining:
        non_dom = []
        for i in remaining:
            if not any(dominates(objectives[j], objectives[i]) for j in remaining if i != j):
                non_dom.append(i)

        if non_dom:
            if len(selected_pop) + len(non_dom) > pop_size:
                sorted_idx = sorted(non_dom, key=lambda i: objectives[i][1])
                step = max(1, len(sorted_idx) // (pop_size - len(selected_pop)))
                for i in range(0, len(sorted_idx), step):
                    if len(selected_pop) < pop_size:
                        idx = sorted_idx[i]
                        selected_pop.append(population[idx])
                        selected_objs.append(objectives[idx])
            else:
                for idx in non_dom:
                    selected_pop.append(population[idx])
                    selected_objs.append(objectives[idx])

            for idx in non_dom:
                if idx in remaining:
                    remaining.remove(idx)
        else:
            break

    while len(selected_pop) < pop_size and remaining:
        idx = random.choice(remaining)
        selected_pop.append(population[idx])
        selected_objs.append(objectives[idx])
        remaining.remove(idx)

    return selected_pop, selected_objs


# === Experiment Runner ===
def run_experiments():
    print("Running Multi-Objective EA Experiments")
    print("=" * 50)

    problem_ids = {
        'MaxCoverage': [2100, 2101, 2102, 2103],
        'MaxInfluence': [2200, 2201, 2202, 2203]
    }

    pop_sizes = [10, 20, 50]
    results = {}

    for problem_type, ids in problem_ids.items():
        print(f"\nTesting {problem_type}:")
        results[problem_type] = {}

        for problem_id in ids:
            print(f"  Problem {problem_id}:")
            results[problem_type][problem_id] = {}

            for pop_size in pop_sizes:
                print(f"    Running Pop {pop_size} once (test)...")

                random.seed(42)
                np.random.seed(42)

                pop, objs, hist, pareto_front, pareto_objs = multi_objective_ea(
                    problem_id=problem_id,
                    pop_size=pop_size,
                    run_index=0
                )

                best_f = max(obj[0] for obj in objs)
                pareto_size = len(pareto_front)

                results[problem_type][problem_id][pop_size] = {
                    'best_f': best_f,
                    'pareto_size': pareto_size
                }

                print(f"      Best f={best_f:.1f}, Pareto size={pareto_size}")

    return results


if __name__ == "__main__":
    print("Multi-Objective EA for Exercise 3")
    print("-" * 50)

    print("\nQuick test on Problem 2100 with pop_size=20:")

    random.seed(42)
    np.random.seed(42)

    pop, objs, hist, pareto_front, pareto_objs = multi_objective_ea(problem_id=2100, pop_size=20)

    print(f"\nResults:")
    print(f"  Final population size: {len(pop)}")
    print(f"  Best f(S): {max(obj[0] for obj in objs):.1f}")
    print(f"  Pareto front size: {len(pareto_front)}")

    sorted_pareto = sorted(zip(pareto_objs, pareto_front), key=lambda x: x[0][0], reverse=True)
    print("\nTop 5 Pareto solutions:")
    for i, (obj, sol) in enumerate(sorted_pareto[:5]):
        print(f"  {i+1}. f(S)={obj[0]:.1f}, |S|={-obj[1]}")

    print("\nRunning full experiments (1 run each)...")
    results = run_experiments()
