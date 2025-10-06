"""
Complete code for Exercise 5
"""

from ioh import get_problem, ProblemClass
from ioh import logger
import numpy as np
import os


class CustomACO:
    """Custom ACO following lecture specifications (slides 18-35)."""
    
    def __init__(self, n_ants=15, rho=0.1, alpha=1.0, beta=2.0, 
                 elitist_weight=5, local_search=True):
        self.n_ants = n_ants
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.elitist_weight = elitist_weight
        self.use_local_search = local_search
        
    def initialize_pheromones(self, n_vars):
        """Initialize pheromone values (slide 18)."""
        self.tau = np.ones(n_vars) * 0.5
        self.tau_min = 0.01
        self.tau_max = 0.99
        
    def greedy_function(self, bit_position, n_vars):
        """Greedy function η(c_i,j) - uniform for general PBO."""
        return 1.0
        
    def construct_solution(self, n_vars):
        """
        Construct solution using formula from slide 32:
        p(c_i,j | sp) = [τ_i,j]^α · [η(c_i,j)]^β / Σ [τ_k,l]^α · [η(c_k,l)]^β
        """
        solution = np.zeros(n_vars, dtype=int)
        
        for i in range(n_vars):
            tau_one = self.tau[i]
            tau_zero = 1.0 - self.tau[i]
            
            eta_one = self.greedy_function(i, n_vars)
            eta_zero = self.greedy_function(i, n_vars)
            
            prob_one = (tau_one ** self.alpha) * (eta_one ** self.beta)
            prob_zero = (tau_zero ** self.alpha) * (eta_zero ** self.beta)
            
            total = prob_one + prob_zero
            prob_one = prob_one / total
            
            solution[i] = 1 if np.random.random() < prob_one else 0
                
        return solution
    
    def local_search(self, solution, func, max_flips=None):
        """Optional local search"""
        if max_flips is None:
            max_flips = len(solution)
            
        current_fitness = func(solution)
        improved = True
        flips_done = 0
        
        while improved and flips_done < max_flips:
            improved = False
            for i in range(len(solution)):
                if flips_done >= max_flips:
                    break
                    
                solution[i] = 1 - solution[i]
                new_fitness = func(solution)
                flips_done += 1
                
                if new_fitness > current_fitness:
                    current_fitness = new_fitness
                    improved = True
                    break
                else:
                    solution[i] = 1 - solution[i]
                    
        return solution, current_fitness
    
    def apply_pheromone_update(self, S_iter, best_solution_ever, best_fitness_ever):
        """
        Pheromone update 
        """
        n_vars = len(self.tau)
        
        # Evaporation
        self.tau = (1 - self.rho) * self.tau
        
        # Reinforcement from iteration solutions
        for solution, fitness in S_iter:
            quality_normalized = fitness / 1000.0
            for i in range(n_vars):
                if solution[i] == 1:
                    self.tau[i] += self.rho * quality_normalized
        
        # Elitist reinforcement
        if best_solution_ever is not None:
            quality_best = best_fitness_ever / 1000.0
            for i in range(n_vars):
                if best_solution_ever[i] == 1:
                    self.tau[i] += self.rho * self.elitist_weight * quality_best
        
        # Enforce bounds
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
        
    def optimize(self, func, budget=None):
        """Main ACO algorithm"""
        if budget is None:
            budget = int(func.meta_data.n_variables ** 2 * 50)
            
        if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
            optimum = 8
        else:
            optimum = func.optimum.y
            
        n_vars = func.meta_data.n_variables
        self.initialize_pheromones(n_vars)
        
        best_solution = None
        best_fitness = -np.inf
        evaluations = 0
        
        while evaluations < budget:
            S_iter = []
            
            for ant in range(self.n_ants):
                if evaluations >= budget:
                    break
                    
                solution = self.construct_solution(n_vars)
                fitness = func(solution)
                evaluations += 1
                
                if self.use_local_search and evaluations < budget:
                    max_ls_evals = min(n_vars, budget - evaluations)
                    solution, fitness = self.local_search(solution, func, max_ls_evals)
                    evaluations += max_ls_evals
                
                S_iter.append((solution.copy(), fitness))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
                    
                if best_fitness >= optimum:
                    break
                    
            self.apply_pheromone_update(S_iter, best_solution, best_fitness)
            
            if best_fitness >= optimum:
                break
                
        return best_fitness, best_solution


def run_custom_aco(func, budget=None):
    """Run custom ACO with 10 independent runs."""
    if budget is None:
        budget = int(func.meta_data.n_variables ** 2 * 50)
        
    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
        
    for run in range(10):
        aco = CustomACO(
            n_ants=15,
            rho=0.1,
            alpha=1.0,
            beta=2.0,
            elitist_weight=5,
            local_search=True
        )
        
        f_opt, x_opt = aco.optimize(func, budget)
        func.reset()
        
    return f_opt, x_opt


def main():
    """Main execution function."""
    
    problems = [
        (1, "OneMax"),
        (2, "LeadingOnes"),
        (3, "BinaryValue"),
        (18, "LABS"),
        (23, "Ising_Ring"),
        (24, "Ising_Torus"),
        (25, "MIS")
    ]
    
    print("="*80)
    
    os.makedirs("data", exist_ok=True)
    
    l = logger.Analyzer(
        root="data",
        folder_name="custom_aco_lecture_compliant",
        algorithm_name="CustomACO_Ex5",
        algorithm_info="Lecture-compliant: elitist AS-update, α=1 β=2 ρ=0.1, 15 ants, LS"
    )
    
    results = {}
    for fid, name in problems:
        print(f"\n{'='*80}")
        print(f"Running on F{fid}: {name} (n=100, budget=100,000)")
        print(f"{'='*80}")
        
        problem = get_problem(
            fid=fid,
            dimension=100,
            instance=1,
            problem_class=ProblemClass.PBO
        )
        
        problem.attach_logger(l)
        budget = 100000
        
        f_opt, x_opt = run_custom_aco(problem, budget)
        results[fid] = f_opt
        
        print(f"✓ Completed F{fid} - Best fitness: {f_opt}")
    
    del l
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Function':<25} {'Best Fitness':>15}")
    print("-"*80)
    for fid, name in problems:
        print(f"F{fid:2d} {name:<22} {results[fid]:>15.2f}")
    

    print("\n" + "="*80)
    print("✓ EXECUTION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()