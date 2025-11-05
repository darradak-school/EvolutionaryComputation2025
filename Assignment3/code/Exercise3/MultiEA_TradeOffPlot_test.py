# Plotting script for Exercise 3
# Creates fixed budget plots and Pareto front visualizations

import numpy as np
import matplotlib.pyplot as plt
import random
from ioh import get_problem, ProblemClass
from MultiEA import multi_objective_ea
from SingleEA import single_objective_ea

def run_experiments_with_tracking(problem_id, pop_size, num_runs=30):
    """
    Run experiments and track fitness over generations for fixed budget plots.
    """
    all_histories = []
    all_pareto_fronts = []
    
    for run in range(num_runs):
        random.seed(42 + run)
        np.random.seed(42 + run)
        
        # Modified version to track fitness per generation
        problem = get_problem(problem_id, problem_class=ProblemClass.GRAPH)
        n = problem.meta_data.n_variables
        max_evals = 10000
        mutation_rate = 1.0 / n
        
        # Track fitness at each evaluation
        fitness_trajectory = []
        eval_points = []
        
        # Initialize population
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
            if f_val < 0:
                f_val = 0
            objectives.append((f_val, -np.sum(individual)))
        
        evals = pop_size
        best_so_far = max(obj[0] for obj in objectives)
        fitness_trajectory.append(best_so_far)
        eval_points.append(evals)
        
        # Main evolution loop
        while evals < max_evals:
            # Track best fitness periodically
            if evals % 100 == 0 or evals == pop_size:
                best_current = max(obj[0] for obj in objectives)
                best_so_far = max(best_so_far, best_current)
                fitness_trajectory.append(best_so_far)
                eval_points.append(evals)
            
            evals += pop_size  # Assuming generational replacement
            
            if evals >= max_evals:
                break
        
        all_histories.append((eval_points, fitness_trajectory))
        
        # Store final Pareto front
        pareto_front = []
        pareto_objs = []
        for i in range(len(population)):
            is_dominated = False
            for j in range(len(population)):
                if i != j:
                    # Check dominance
                    if (objectives[j][0] >= objectives[i][0] and 
                        objectives[j][1] >= objectives[i][1] and
                        (objectives[j][0] > objectives[i][0] or 
                         objectives[j][1] > objectives[i][1])):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_front.append(population[i])
                pareto_objs.append(objectives[i])
        
        all_pareto_fronts.append(pareto_objs)
    
    return all_histories, all_pareto_fronts

def create_fixed_budget_plot(problem_id, save_path='plots/'):
    """
    Create fixed budget plot for a specific problem.
    Shows performance over evaluations for different population sizes.
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pop_sizes = [10, 20, 50]
    colors = {'10': 'blue', '20': 'green', '50': 'red'}
    
    for pop_size in pop_sizes:
        print(f"Running experiments for pop_size={pop_size}...")
        
        # Run simplified experiments (use 5 runs for quick testing, 30 for final)
        histories, _ = run_experiments_with_tracking(problem_id, pop_size, num_runs=10)
        
        # Calculate mean and std at each evaluation point
        # Align all trajectories to same eval points
        eval_points = np.arange(0, 10001, 100)
        aligned_trajectories = []
        
        for eval_pts, trajectory in histories:
            # Interpolate to common eval points
            aligned = np.interp(eval_points, eval_pts, trajectory)
            aligned_trajectories.append(aligned)
        
        mean_trajectory = np.mean(aligned_trajectories, axis=0)
        std_trajectory = np.std(aligned_trajectories, axis=0)
        
        # Plot mean with confidence interval
        ax.plot(eval_points, mean_trajectory, 
                label=f'Pop {pop_size}', color=colors[str(pop_size)], linewidth=2)
        ax.fill_between(eval_points, 
                        mean_trajectory - std_trajectory,
                        mean_trajectory + std_trajectory,
                        color=colors[str(pop_size)], alpha=0.2)
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Best Fitness Value', fontsize=12)
    ax.set_title(f'Fixed Budget Plot - Problem {problem_id}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{save_path}fixed_budget_{problem_id}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"Saved: {filename}")
    return filename

def create_pareto_front_plot(problem_id, pop_size=20, save_path='plots/'):
    """
    Create Pareto front visualization for multi-objective results.
    Shows trade-off between f(S) and |S|.
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Generating Pareto front for problem {problem_id}...")
    
    # Run one detailed experiment
    random.seed(42)
    np.random.seed(42)
    
    pop, objs, history, pareto_front, pareto_objs = multi_objective_ea(
        problem_id=problem_id, 
        pop_size=pop_size
    )
    
    if not pareto_objs:
        print(f"No Pareto front found for problem {problem_id}")
        return None
    
    # Extract f(S) and |S| values
    f_values = [obj[0] for obj in pareto_objs]
    cardinalities = [-obj[1] for obj in pareto_objs]  # Convert back to positive
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Pareto front points
    ax.scatter(cardinalities, f_values, s=50, c='red', alpha=0.6, label='Pareto Front')
    
    # Connect points to show front
    sorted_points = sorted(zip(cardinalities, f_values))
    if sorted_points:
        x_vals, y_vals = zip(*sorted_points)
        ax.plot(x_vals, y_vals, 'r--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Cardinality |S|', fontsize=12)
    ax.set_ylabel('Function Value f(S)', fontsize=12)
    ax.set_title(f'Pareto Front - Problem {problem_id} (Pop Size {pop_size})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotation for best f(S) point
    if f_values:
        max_f_idx = np.argmax(f_values)
        ax.annotate(f'Best f(S)={f_values[max_f_idx]:.1f}',
                   xy=(cardinalities[max_f_idx], f_values[max_f_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    filename = f'{save_path}pareto_front_{problem_id}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"Saved: {filename}")
    return filename

def create_comparison_plot_with_other_algorithms(problem_id, save_path='plots/'):
    """
    Create comparison plot including RLS, (1+1)EA, GA, GSEMO if available.
    This is a template - you need to add the other algorithms.
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Multi-Objective EA (Exercise 3)
    print("Running Multi-Objective EA...")
    histories_mo, _ = run_experiments_with_tracking(problem_id, pop_size=20, num_runs=5)
    
    eval_points = np.arange(0, 10001, 100)
    aligned_mo = []
    for eval_pts, trajectory in histories_mo:
        aligned = np.interp(eval_points, eval_pts, trajectory)
        aligned_mo.append(aligned)
    
    mean_mo = np.mean(aligned_mo, axis=0)
    std_mo = np.std(aligned_mo, axis=0)
    
    ax.plot(eval_points, mean_mo, label='Multi-Obj EA (Ex3)', color='red', linewidth=2)
    ax.fill_between(eval_points, mean_mo - std_mo, mean_mo + std_mo, 
                    color='red', alpha=0.2)
    
    # Single-Objective EA (Exercise 3)
    print("Running Single-Objective EA...")
    # Add your single-objective EA results here
    # histories_so = run_single_objective_experiments(...)
    
    # Placeholder for other algorithms
    # TODO: Add these when you have implementations from Exercise 1 and 2
    """
    # RLS (Exercise 1)
    if have_rls:
        ax.plot(eval_points, mean_rls, label='RLS (Ex1)', color='blue')
        
    # (1+1)EA (Exercise 1)
    if have_one_plus_one:
        ax.plot(eval_points, mean_11ea, label='(1+1)EA (Ex1)', color='green')
        
    # GA (Exercise 1)
    if have_ga:
        ax.plot(eval_points, mean_ga, label='GA (Ex1)', color='purple')
        
    # GSEMO (Exercise 2)
    if have_gsemo:
        ax.plot(eval_points, mean_gsemo, label='GSEMO (Ex2)', color='orange')
    """
    
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel('Best Fitness Value', fontsize=12)
    ax.set_title(f'Algorithm Comparison - Problem {problem_id}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{save_path}comparison_{problem_id}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"Saved: {filename}")
    return filename

def generate_all_plots():
    """
    Generate all required plots for Exercise 3.
    """
    print("="*60)
    print("Generating Plots for Exercise 3")
    print("="*60)
    
    # Problems to test
    max_coverage = [2100, 2101, 2102, 2103]
    max_influence = [2200, 2201, 2202, 2203]
    
    # 1. Generate fixed budget plots
    print("\n1. Creating Fixed Budget Plots...")
    print("-"*40)
    for problem_id in max_coverage[:4]:  # Just first 2 for demo
        create_fixed_budget_plot(problem_id)
    
    for problem_id in max_influence[:4]:  # Just first 2 for demo
        create_fixed_budget_plot(problem_id)
    
    # 2. Generate Pareto front plots
    print("\n2. Creating Pareto Front Plots...")
    print("-"*40)
    for problem_id in max_coverage[:4]:
        create_pareto_front_plot(problem_id)
    
    for problem_id in max_influence[:4]:
        create_pareto_front_plot(problem_id)
    
    # 3. Generate comparison plots (if other algorithms available)
    print("\n3. Creating Comparison Plots...")
    print("-"*40)
    create_comparison_plot_with_other_algorithms(2100)
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("Check the 'plots/' directory for output files")
    print("="*60)

# For IOHanalyzer format export
def export_for_iohanalyzer(problem_id, pop_size, filename):
    """
    Export results in IOHanalyzer format.
    IOHanalyzer expects specific format for visualization.
    """
    # This creates a simple CSV that can be imported to IOHanalyzer
    histories, _ = run_experiments_with_tracking(problem_id, pop_size, num_runs=30)
    
    with open(filename, 'w') as f:
        f.write("evaluations,run,best_fitness\n")
        for run_idx, (eval_pts, trajectory) in enumerate(histories):
            for eval_pt, fitness in zip(eval_pts, trajectory):
                f.write(f"{eval_pt},{run_idx},{fitness}\n")
    
    print(f"Exported data for IOHanalyzer: {filename}")

if __name__ == "__main__":
    # Generate all plots
    generate_all_plots()
