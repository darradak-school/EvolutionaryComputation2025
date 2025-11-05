import os
import time
import json
import random
import numpy as np
from ioh import get_problem, ProblemClass, logger

# ---------------- utility IO helpers ----------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def save_best_so_far(run_folder, best_so_far):
    np.save(os.path.join(run_folder, "best_so_far.npy"), np.array(best_so_far, dtype=float))

def save_metadata(run_folder, meta):
    with open(os.path.join(run_folder, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

# ---------------- EA components (kept similar to your original) ----------------

def initialize_population(pop_size, dimension, budget):
    population = []
    for _ in range(pop_size):
        individual = np.zeros(dimension, dtype=int)
        ones = random.sample(range(dimension), budget)
        individual[ones] = 1
        population.append(individual)
    return population

def mutate(individual):
    offspring = individual.copy()
    idx = random.randint(0, len(individual) - 1)
    offspring[idx] = 1 - offspring[idx]
    return offspring

def repair(individual, budget):
    """Ensure exactly 'budget' ones in the solution."""
    ones = np.where(individual == 1)[0]
    zeros = np.where(individual == 0)[0]
    if len(ones) > budget:
        to_flip = random.sample(list(ones), len(ones) - budget)
        individual[to_flip] = 0
    elif len(ones) < budget:
        to_flip = random.sample(list(zeros), budget - len(ones))
        individual[to_flip] = 1
    return individual

def hamming_distance(a, b):
    return int(np.sum(a != b))

def select_diverse(population, fitnesses, mu):
    # Greedy Hamming diversity selection starting from best by fitness
    if not population:
        return []
    # Ensure fitness alignment
    idx_sorted = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    selected = [population[idx_sorted[0]]]
    selected_idx = {idx_sorted[0]}
    while len(selected) < mu:
        best_candidate = None
        best_score = -1
        for idx in idx_sorted:
            if idx in selected_idx:
                continue
            dist = min(hamming_distance(population[idx], s) for s in selected)
            if dist > best_score:
                best_score = dist
                best_candidate = idx
        if best_candidate is None:
            break
        selected.append(population[best_candidate])
        selected_idx.add(best_candidate)
    return selected

# ---------------- Single-objective EA with logging ----------------

def single_objective_ea(problem_id=2100, pop_size=20, budget=10,
                        run_index=0, problem_type="MaxCoverage",
                        max_evals=10000, run_folder_root="data"):
    """
    Population-based single-objective EA with (mu+lambda) style and Hamming diversity.
    Logs IOH output and saves best_so_far and metadata under:
      data/SingleEA_Exercise3/<problem_type>/f<problem_id>/pop<pop_size>/run<run_index+1>
    """

    # Prepare run folder and IOH analyzer
    run_folder = os.path.join(
        run_folder_root,
        "SingleEA_Exercise3",
        problem_type,
        f"f{problem_id}",
        f"pop{pop_size}",
        f"run{run_index+1}"
    )
    run_folder = ensure_dir(run_folder)
    print("SingleEA run folder:", run_folder)

    algo_name = f"SingleEA_pop{pop_size}_run{run_index+1}"
    analyzer = logger.Analyzer(root=logger.Path(run_folder), algorithm_name=algo_name, store_positions=True)

    # Get problem and attach logger (problem(...) will be logged)
    problem = get_problem(problem_id, problem_class=ProblemClass.GRAPH)
    problem.attach_logger(analyzer)

    # Reproducible seeds per run
    base_seed = 1000
    seed = base_seed + run_index
    random.seed(seed)
    np.random.seed(seed)

    # Initialization
    dimension = problem.meta_data.n_variables
    population = initialize_population(pop_size, dimension, budget)
    fitnesses = [float(problem(ind)) for ind in population]  # IOH logs each evaluation
    evals = len(population)

    # best_so_far tracking (length grows with each evaluation)
    best_so_far = []
    current_best = max(fitnesses) if fitnesses else -float("inf")
    for _ in range(evals):
        best_so_far.append(current_best)

    # Main loop: generate one offspring per parent (lambda = mu) and do mu+lambda selection
    while evals < max_evals:
        offspring = []
        offspring_fits = []

        for parent in population:
            if evals >= max_evals:
                break
            child = mutate(parent)
            child = repair(child, budget)
            fit = float(problem(child))  # IOH logs evaluation
            offspring.append(child)
            offspring_fits.append(fit)
            evals += 1
            # update best_so_far
            current_best = max(current_best, fit)
            best_so_far.append(current_best)

        # Combine parents and offspring and select next generation
        combined = population + offspring
        combined_fits = fitnesses + offspring_fits

        # Sort combined by fitness descending and produce aligned lists
        idx_sorted = sorted(range(len(combined)), key=lambda i: combined_fits[i], reverse=True)
        sorted_pop = [combined[i] for i in idx_sorted]
        sorted_fits = [combined_fits[i] for i in idx_sorted]

        # Diversity-aware selection (Hamming-based)
        population = select_diverse(sorted_pop, sorted_fits, pop_size)

        # Align fitnesses for new population (map from sorted lists)
        fit_map = {tuple(sorted_pop[i].tolist()): sorted_fits[i] for i in range(len(sorted_pop))}
        fitnesses = [fit_map.get(tuple(ind.tolist()), float(problem(ind))) for ind in population]

    # Detach logger and save results
    problem.detach_logger()

    save_best_so_far(run_folder, best_so_far)
    metadata = {
        "algorithm": "SingleEA",
        "problem_id": problem_id,
        "problem_type": problem_type,
        "pop_size": pop_size,
        "budget": budget,
        "max_evals": max_evals,
        "run_index": run_index,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_metadata(run_folder, metadata)

    return population, fitnesses, best_so_far, run_folder

# ---------------- quick demo ----------------

if __name__ == "__main__":
    # experiment config (adjust for full experiments)
    problem_id = 2100
    problem_type = "MaxCoverage"
    budget = 10
    runs = 1          # set to 30 for full experiment
    max_evals = 1000  # set to 10000 for full experiment
    pop_sizes = [10, 20, 50]

    for run_index in range(runs):
        for pop_size in pop_sizes:
            print(f"\n=== Running SingleEA: problem={problem_id}, pop={pop_size}, run={run_index+1} ===")
            population, fitnesses, best_so_far, folder = single_objective_ea(
                problem_id=problem_id,
                pop_size=pop_size,
                budget=budget,
                run_index=run_index,
                problem_type=problem_type,
                max_evals=max_evals,
                run_folder_root="data"
            )
            print(f"Finished pop={pop_size}, run={run_index+1}, run folder: {folder}")
            print(f"Best at end: {best_so_far[-1] if best_so_far else None}")