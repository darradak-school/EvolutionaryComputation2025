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
from ioh import get_problem, ProblemClass, logger, analyzer

def multi_objective_ea(problem_id=2100, pop_size=20, run_index=0):
    """
    Multi-objective EA using f(S) and -|S| as objectives.
    Includes correct IOH logging setup.
    """

    # Unique folder name for each run (prevents overwriting)
    algo_name = f"MultiEA_pop{pop_size}_run{run_index+1}"
    
    # Create logger
    l = logger.Analyzer(
        root="data",  # Folder where logs are stored
        folder_name="MultiEA_Exercise3",
        algorithm_name=algo_name,
        store_positions=True,
        merge_output=False
    )
    
    # Get problem and attach logger
    problem = get_problem(problem_id, problem_class=ProblemClass.GRAPH)
    problem.attach_logger(l)

    # Setup
    n = problem.meta_data.n_variables
    max_evals = 10000
    mutation_rate = 1.0 / n
    
    # Initialize population
    population = []
    objectives = []
    
    for i in range(pop_size):
        num_ones = random.randint(0, n)
        individual = np.zeros(n, dtype=int)
        if num_ones > 0:
            indices = random.sample(range(n), num_ones)
            individual[indices] = 1
        population.append(individual)
        
        # Evaluate and log automatically via problem()
        f_val = problem(individual)
        if f_val < 0:
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
            
            # Crossover and mutation
            child = crossover(parent1, parent2) if random.random() < 0.9 else parent1.copy()
            child = mutate(child, mutation_rate)
            
            # Evaluate through IOH problem
            f_val = problem(child)
            if f_val < 0:
                f_val = 0
            
            offspring.append(child)
            offspring_objs.append((f_val, -np.sum(child)))
            
            evals += 1
            if evals >= max_evals:
                break
        
        # Combine and select next generation
        population, objectives = select_next_generation(
            population + offspring, objectives + offspring_objs, pop_size
        )
        
        best_f = max(obj[0] for obj in objectives)
        best_f_history.append(best_f)
    
    # Compute Pareto front
    pareto_front, pareto_objs = [], []
    for i in range(len(population)):
        if not any(dominates(objectives[j], objectives[i]) for j in range(len(population)) if i != j):
            pareto_front.append(population[i])
            pareto_objs.append(objectives[i])

    # Detach logger to finalize .dat files
    problem.detach_logger(l)

    return population, objectives, best_f_history, pareto_front, pareto_objs


def dominates(obj1, obj2):
    """Check if obj1 dominates obj2."""
    return obj1[0] >= obj2[0] and obj1[1] >= obj2[1] and (obj1[0] > obj2[0] or obj1[1] > obj2[1])

def tournament_select(population, objectives, tournament_size=3):
    """Simple tournament selection."""
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    
    # Pick the one that dominates others, or random if none dominate
    best_idx = indices[0]
    for idx in indices[1:]:
        if dominates(objectives[idx], objectives[best_idx]):
            best_idx = idx
    
    return population[best_idx].copy()

def crossover(parent1, parent2):
    """Uniform crossover."""
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def mutate(individual, rate):
    """Bit flip mutation."""
    child = individual.copy()
    for i in range(len(individual)):
        if random.random() < rate:
            child[i] = 1 - child[i]
    return child

def select_next_generation(population, objectives, pop_size):
    """
    Select next generation based on non-dominated sorting.
    Simplified version that actually works.
    """
    # Find all non-dominated solutions first
    selected_pop = []
    selected_objs = []
    remaining_indices = list(range(len(population)))
    
    while len(selected_pop) < pop_size and remaining_indices:
        # Find non-dominated in remaining
        non_dom_indices = []
        for i in remaining_indices:
            is_dominated = False
            for j in remaining_indices:
                if i != j and dominates(objectives[j], objectives[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dom_indices.append(i)
        
        # If we found non-dominated solutions, add them
        if non_dom_indices:
            # If adding all would exceed pop_size, select based on diversity
            if len(selected_pop) + len(non_dom_indices) > pop_size:
                # Simple diversity: pick solutions with different cardinalities
                sorted_indices = sorted(non_dom_indices, 
                                      key=lambda i: objectives[i][1])  # Sort by cardinality
                # Take evenly spaced solutions
                step = max(1, len(sorted_indices) // (pop_size - len(selected_pop)))
                for i in range(0, len(sorted_indices), step):
                    if len(selected_pop) < pop_size:
                        idx = sorted_indices[i]
                        selected_pop.append(population[idx])
                        selected_objs.append(objectives[idx])
            else:
                # Add all non-dominated
                for idx in non_dom_indices:
                    selected_pop.append(population[idx])
                    selected_objs.append(objectives[idx])
            
            # Remove added solutions from remaining
            for idx in non_dom_indices:
                if idx in remaining_indices:
                    remaining_indices.remove(idx)
        else:
            # No non-dominated found (shouldn't happen), just take what's left
            break
    
    # If still need more, add random ones
    while len(selected_pop) < pop_size and remaining_indices:
        idx = random.choice(remaining_indices)
        selected_pop.append(population[idx])
        selected_objs.append(objectives[idx])
        remaining_indices.remove(idx)
    
    return selected_pop, selected_objs

def run_experiments():
    """Run the required experiments and export to CSV automatically."""
    print("Running Multi-Objective EA Experiments")
    print("=" * 50)
    
    # Test problems
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
                results[problem_type][problem_id][pop_size] = []
                
                for i in range(30):  # Run 30 times
                    print(f"    Run {i+1}/30 for Pop {pop_size}...")
                    
                    # Use slightly different seeds per run
                    random.seed(i + 42)
                    np.random.seed(i + 42)
                    
                    pop, objs, history, pareto_front, pareto_objs = multi_objective_ea(
                        problem_id=problem_id,
                        pop_size=pop_size,
                        run_index=i  # unique logger name per run
                    )
                    
                    best_f = max(obj[0] for obj in objs)
                    pareto_size = len(pareto_front)
                    
                    results[problem_type][problem_id][pop_size].append({
                        'best_f': best_f,
                        'pareto_size': pareto_size
                    })
                    
                    print(f"    Run {i+1}/30: Best f={best_f:.1f}, Pareto size={pareto_size}")
    
    # ---------------------------------------------------
    # 🧾 AUTOMATIC CSV EXPORT SECTION
    # ---------------------------------------------------
    print("\nExporting all IOH logs to CSV...")
    
    ioh_data_path = "data/MultiEA_Exercise3"
    csv_output_path = "data/MultiEA_Exercise3_csv"
    os.makedirs(csv_output_path, exist_ok=True)
    
    analyzer.IOHAnalyzer(
        root=ioh_data_path,
        output_directory=csv_output_path,
        convert_to_csv=True
    )
    
    print(f"\n✅ CSV files saved to: {os.path.abspath(csv_output_path)}")
    print("You can now open them in IOHAnalyzer or any spreadsheet tool.")
    
    return results


if __name__ == "__main__":
    print("Multi-Objective EA for Exercise 3")
    print("-"*50)
    
    # Quick test on one problem
    print("\nQuick test on Problem 2100 with pop_size=20:")
    
    random.seed(42)
    np.random.seed(42)
    
    population, objectives, history, pareto_front, pareto_objs = multi_objective_ea(
        problem_id=2100, 
        pop_size=20
    )
    
    print(f"\nResults:")
    print(f"  Final population size: {len(population)}")
    print(f"  Best f(S): {max(obj[0] for obj in objectives):.1f}")
    print(f"  Pareto front size: {len(pareto_front)}")
    
    print(f"\nTop 5 Pareto solutions:")
    sorted_pareto = sorted(zip(pareto_objs, pareto_front), key=lambda x: x[0][0], reverse=True)
    for i, (obj, sol) in enumerate(sorted_pareto[:5]):
        print(f"  {i+1}. f(S)={obj[0]:.1f}, |S|={-obj[1]}")
    
    print("\nRunning full experiments...")
    results = run_experiments()