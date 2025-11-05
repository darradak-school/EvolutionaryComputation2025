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

def multi_objective_ea(problem_id=2100, pop_size=20, run_index=0, problem_type="Unknown"):
    """
    Multi-objective EA using f(S) and -|S| as objectives.
    Structured IOH logging: data/MultiEA_Exercise3/<problem_type>/f<id>/pop<pop_size>/run<run_index>
    """
    algo_name = f"MultiEA_pop{pop_size}_run{run_index+1}"

    # Build hierarchical folder path
    run_folder = os.path.join(
        "data",
        "MultiEA_Exercise3",
        problem_type,
        f"f{problem_id}",
        f"pop{pop_size}",
        f"run{run_index+1}"
    )
    os.makedirs(run_folder, exist_ok=True)

    # Create logger
    l = logger.Analyzer(
        root=logger.Path(run_folder),
        folder_name="",  # prevent IOH from auto-appending "-1"
        algorithm_name=algo_name,
        store_positions=True
    )

    # Get problem and attach logger
    problem = get_problem(problem_id, problem_class=ProblemClass.GRAPH)
    problem.attach_logger(l)

    # Setup
    n = problem.meta_data.n_variables
    max_evals = 10000
    mutation_rate = 1.0 / n

    # Initialize population with random cardinalities
    population = []
    objectives = []
    for i in range(pop_size):
        num_ones = random.randint(0, n)
        individual = np.zeros(n, dtype=int)
        if num_ones > 0:
            indices = random.sample(range(n), num_ones)
            for idx in indices:
                individual[idx] = 1
        population.append(individual)
        f_val = problem(individual)
        if f_val < 0:  # Handle constraint violation
            f_val = 0
        objectives.append((f_val, -np.sum(individual)))

    evals = pop_size
    best_f_history = []

    # Main loop
    while evals < max_evals:
        offspring = []
        offspring_objs = []

        for _ in range(pop_size):
            parent1 = tournament_select(population, objectives)
            parent2 = tournament_select(population, objectives)

            # Crossover
            if random.random() < 0.9:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            child = mutate(child, mutation_rate)

            # Evaluate
            f_val = problem(child)
            if f_val < 0:
                f_val = 0

            offspring.append(child)
            offspring_objs.append((f_val, -np.sum(child)))
            evals += 1
            if evals >= max_evals:
                break

        # Combine parent and offspring
        combined_pop = population + offspring
        combined_objs = objectives + offspring_objs

        # Select next generation
        population, objectives = select_next_generation(combined_pop, combined_objs, pop_size)

        # Track best f(S)
        best_f = max(obj[0] for obj in objectives)
        best_f_history.append(best_f)

    # Get final Pareto front
    pareto_front = []
    pareto_objs = []
    for i in range(len(population)):
        is_dominated = False
        for j in range(len(population)):
            if i != j and dominates(objectives[j], objectives[i]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(population[i])
            pareto_objs.append(objectives[i])

    return population, objectives, best_f_history, pareto_front, pareto_objs


def dominates(obj1, obj2):
    """Check if obj1 dominates obj2."""
    return obj1[0] >= obj2[0] and obj1[1] >= obj2[1] and (obj1[0] > obj2[0] or obj1[1] > obj2[1])


def tournament_select(population, objectives, tournament_size=3):
    """Simple tournament selection."""
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    best_idx = indices[0]
    for idx in indices[1:]:
        if dominates(objectives[idx], objectives[best_idx]):
            best_idx = idx
    return population[best_idx].copy()


def crossover(parent1, parent2):
    """Uniform crossover."""
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
    return child


def mutate(individual, rate):
    """Bit flip mutation."""
    child = individual.copy()
    for i in range(len(individual)):
        if random.random() < rate:
            child[i] = 1 - child[i]
    return child


def select_next_generation(population, objectives, pop_size):
    """Select next generation based on non-dominated sorting."""
    selected_pop = []
    selected_objs = []
    remaining_indices = list(range(len(population)))

    while len(selected_pop) < pop_size and remaining_indices:
        non_dom_indices = []
        for i in remaining_indices:
            is_dominated = False
            for j in remaining_indices:
                if i != j and dominates(objectives[j], objectives[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dom_indices.append(i)

        if non_dom_indices:
            if len(selected_pop) + len(non_dom_indices) > pop_size:
                sorted_indices = sorted(non_dom_indices, key=lambda i: objectives[i][1])
                step = max(1, len(sorted_indices) // (pop_size - len(selected_pop)))
                for i in range(0, len(sorted_indices), step):
                    if len(selected_pop) < pop_size:
                        idx = sorted_indices[i]
                        selected_pop.append(population[idx])
                        selected_objs.append(objectives[idx])
            else:
                for idx in non_dom_indices:
                    selected_pop.append(population[idx])
                    selected_objs.append(objectives[idx])
            for idx in non_dom_indices:
                if idx in remaining_indices:
                    remaining_indices.remove(idx)
        else:
            break

    while len(selected_pop) < pop_size and remaining_indices:
        idx = random.choice(remaining_indices)
        selected_pop.append(population[idx])
        selected_objs.append(objectives[idx])
        remaining_indices.remove(idx)

    return selected_pop, selected_objs


def run_full_experiments(runs_per_instance=30):
    """Run full experiments with multiple runs per instance."""
    print("Running Multi-Objective EA Experiments")
    print("="*50)

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
            print(f"\n  Problem {problem_id}:")
            results[problem_type][problem_id] = {}
            for pop_size in pop_sizes:
                print(f"    Pop {pop_size} ({runs_per_instance} runs)...")
                results[problem_type][problem_id][pop_size] = []

                for run_index in range(runs_per_instance):
                    random.seed(42 + run_index)
                    np.random.seed(42 + run_index)
                    pop, objs, history, pareto_front, pareto_objs = multi_objective_ea(
                        problem_id=problem_id,
                        pop_size=pop_size,
                        run_index=run_index,
                        problem_type=problem_type
                    )
                    best_f = max(obj[0] for obj in objs)
                    pareto_size = len(pareto_front)
                    results[problem_type][problem_id][pop_size].append({
                        'run': run_index + 1,
                        'best_f': best_f,
                        'pareto_size': pareto_size
                    })
                    print(f"      Run {run_index+1}: Best f={best_f:.1f}, Pareto size={pareto_size}")

    return results


if __name__ == "__main__":
    print("Multi-Objective EA for Exercise 3")
    print("-" * 50)

    # Quick test on one problem (run 1)
    random.seed(42)
    np.random.seed(42)
    population, objectives, history, pareto_front, pareto_objs = multi_objective_ea(
        problem_id=2100,
        pop_size=20,
        run_index=0,
        problem_type="MaxCoverage"
    )

    print(f"\nResults of quick test:")
    print(f"  Final population size: {len(population)}")
    print(f"  Best f(S): {max(obj[0] for obj in objectives):.1f}")
    print(f"  Pareto front size: {len(pareto_front)}")

    print("\nRunning full experiments (30 runs each)...")
    full_results = run_full_experiments(runs_per_instance=30)
