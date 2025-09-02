import numpy as np
import matplotlib.pyplot as plt
import ioh
import os
import numpy as np
from RLS import RLS 

# --- Config ---
FUNCTION_IDS = [1, 2, 3, 18, 23, 24, 25]
DIM = 100
RUNS = 10
ITERATIONS = 100000


def run_experiment(fid, dimension, iterations, runs):
    problem = ioh.get_problem(fid=fid, dimension=dimension)

    all_traces = []
    for run in range(runs):
        print(f"Running RLS on F{fid}, run {run+1}/{runs}...")
        trace, best = RLS(problem, dimension, iterations, return_trace=True)
        all_traces.append(trace)
        print(f"Run {run+1} best fitness: {best}\n")

    all_traces = np.array(all_traces)
    mean_trace = np.mean(all_traces, axis=0)
    std_trace = np.std(all_traces, axis=0)

    return mean_trace, std_trace

# automatically create the fixed budget plot in /plots_RLS
def plot_fixed_budget(fid, mean_trace, std_trace):
    save_dir = "plots_RLS"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x = np.arange(len(mean_trace))
    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_trace, label='Mean Fitness')
    plt.fill_between(x, mean_trace - std_trace, mean_trace + std_trace,
                     color='b', alpha=0.2, label='Std Dev')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'RLS Fixed-Budget Plot on F{fid}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/F{fid}_RLS_fixed_budget.png")
    plt.close()




def main():
    dimension = 100
    iterations = 100000
    runs = 10
    functions = [1, 2, 3, 18, 23, 24] # have not got 25 working yet

    for fid in functions:
        mean_trace, std_trace = run_experiment(fid, dimension, iterations, runs)
        plot_fixed_budget(fid, mean_trace, std_trace)
        print(f"Finished plotting F{fid}")

if __name__ == "__main__":
    main()