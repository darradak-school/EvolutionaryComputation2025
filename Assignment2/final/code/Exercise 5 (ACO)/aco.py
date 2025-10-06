"""
Complete code for Exercise 5: ACO Algorithm
"""

from ioh import get_problem, ProblemClass
from ioh import logger
import numpy as np


class StandardACO:
    """
    Standard ACO algorithm for Assignment 2.
    """
    
    def __init__(self, n_ants=10, rho=0.1, debug=False):
        self.n_ants = n_ants
        self.rho = rho
        self.debug = debug
        
    def initialize_pheromones(self, n_vars):
        """Initialize all pheromones to same value."""
        self.tau = np.ones(n_vars) * 1.0
        self.tau_min = 0.01
        self.tau_max = 10.0
        
        if self.debug:
            print(f"Initialized pheromones to 1.0")
        
    def construct_solution(self, n_vars):
        """
        Construct solution using pheromone values.
        Probability of bit i = 1 is proportional to tau[i].
        """
        solution = np.zeros(n_vars, dtype=int)
        
        for i in range(n_vars):
            # Probability bit i = 1, using tau[i] vs constant for 0
            tau_one = self.tau[i]
            tau_zero = 1.0  # Constant baseline for setting to 0
            
            prob_one = tau_one / (tau_one + tau_zero)
            
            if np.random.random() < prob_one:
                solution[i] = 1
                
        return solution
    
    def update_pheromones(self, solutions_with_fitness):
        """
        STANDARD ACO UPDATE for binary problems:
        1. Evaporate all pheromones (toward 0)
        2. Add pheromone ONLY to bits that are 1 in solutions
        3. Weight by fitness
        
        This is the key difference: NO subtraction!
        """
        n_vars = len(self.tau)
        
        # Step 1: Evaporation 
        self.tau = (1 - self.rho) * self.tau
        
        # Step 2: Add pheromone only to 1-bits in solutions
        for solution, fitness in solutions_with_fitness:
            # Deposit amount proportional to fitness
            deposit = fitness / 100.0  # Normalize
            
            for i in range(n_vars):
                if solution[i] == 1:
                    # Add pheromone to bits that are 1
                    self.tau[i] += deposit
                # If bit is 0: do nothing (evaporation already reduced tau[i])
        
        # Step 3: Enforce bounds (prevent stagnation/extinction)
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
        
        if self.debug:
            best_fit = max(f for _, f in solutions_with_fitness)
            print(f"  Best fitness: {best_fit}, Mean tau: {np.mean(self.tau):.3f}")
            print(f"  Tau[0-10]: {self.tau[:10]}")
    
    def optimize(self, func, budget=10000):
        """Main optimization loop."""
        n_vars = func.meta_data.n_variables
        self.initialize_pheromones(n_vars)
        
        if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
            optimum = 8
        else:
            optimum = func.optimum.y
        
        best_solution = None
        best_fitness = -np.inf
        evaluations = 0
        iteration = 0
        
        while evaluations < budget:
            iteration += 1
            solutions_with_fitness = []
            
            # Generate solutions with all ants
            for ant in range(self.n_ants):
                if evaluations >= budget:
                    break
                    
                solution = self.construct_solution(n_vars)
                fitness = func(solution)
                evaluations += 1
                
                solutions_with_fitness.append((solution, fitness))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
                
                if best_fitness >= optimum:
                    break
            
            # Update pheromones with all solutions
            if solutions_with_fitness:
                self.update_pheromones(solutions_with_fitness)
            
            # Debug output
            if self.debug and iteration % 100 == 0:
                print(f"\nIter {iteration}, Evals: {evaluations}, Best: {best_fitness}")
            
            if best_fitness >= optimum:
                break
        
        if self.debug:
            print(f"\n{'='*70}")
            print(f"FINAL: Fitness = {best_fitness} after {evaluations} evals")
            print(f"Pheromones (first 30):")
            print(self.tau[:30])
            print(f"Best solution (first 30): {best_solution[:30]}")
            print(f"{'='*70}\n")
        
        return best_fitness, best_solution


class FinalACO:
    """
    Final ACO for Exercise 5 with all features:
    - Standard pheromone update 
    - Elitist strategy
    - Local search
    """
    
    def __init__(self, n_ants=15, rho=0.1, elitist_weight=5, local_search=True):
        self.n_ants = n_ants
        self.rho = rho
        self.elitist_weight = elitist_weight
        self.use_local_search = local_search
        
    def initialize_pheromones(self, n_vars):
        self.tau = np.ones(n_vars) * 1.0
        self.tau_min = 0.01
        self.tau_max = 10.0
        
    def construct_solution(self, n_vars):
        solution = np.zeros(n_vars, dtype=int)
        
        for i in range(n_vars):
            tau_one = self.tau[i]
            tau_zero = 1.0
            prob_one = tau_one / (tau_one + tau_zero)
            
            if np.random.random() < prob_one:
                solution[i] = 1
                
        return solution
    
    def local_search(self, solution, func, max_evals):
        """First-improvement hill climbing."""
        current_fitness = func(solution)
        evals_used = 1
        improved = True
        
        while improved and evals_used < max_evals:
            improved = False
            
            for i in range(len(solution)):
                if evals_used >= max_evals:
                    break
                
                solution[i] = 1 - solution[i]
                new_fitness = func(solution)
                evals_used += 1
                
                if new_fitness > current_fitness:
                    current_fitness = new_fitness
                    improved = True
                    break
                else:
                    solution[i] = 1 - solution[i]
        
        return solution, current_fitness, evals_used
    
    def update_pheromones(self, solutions_with_fitness, best_ever_sol, best_ever_fit):
        """
        Elitist AS-update:
        - Update from all iteration solutions (weight=1)
        - Extra update from best-ever solution (weight=elitist_weight)
        """
        n_vars = len(self.tau)
        
        # Evaporation
        self.tau = (1 - self.rho) * self.tau
        
        # Add from iteration solutions
        for solution, fitness in solutions_with_fitness:
            deposit = fitness / 100.0
            for i in range(n_vars):
                if solution[i] == 1:
                    self.tau[i] += deposit
        
        # Elitist: add from best-ever (stronger weight)
        if best_ever_sol is not None:
            deposit_best = (best_ever_fit / 100.0) * self.elitist_weight
            for i in range(n_vars):
                if best_ever_sol[i] == 1:
                    self.tau[i] += deposit_best
        
        # Bounds
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)
    
    def optimize(self, func, budget=100000):
        """Main loop with local search."""
        n_vars = func.meta_data.n_variables
        self.initialize_pheromones(n_vars)
        
        if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
            optimum = 8
        else:
            optimum = func.optimum.y
        
        best_solution = None
        best_fitness = -np.inf
        
        # Budget split
        ls_budget = int(budget * 0.1) if self.use_local_search else 0
        construction_budget = budget - ls_budget
        
        evaluations = 0
        ls_evals_used = 0
        
        while evaluations < construction_budget:
            solutions_with_fitness = []
            
            # Construct solutions
            for ant in range(self.n_ants):
                if evaluations >= construction_budget:
                    break
                
                solution = self.construct_solution(n_vars)
                fitness = func(solution)
                evaluations += 1
                
                solutions_with_fitness.append((solution, fitness))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution.copy()
                
                if best_fitness >= optimum:
                    break
            
            # Local search on best of iteration
            if self.use_local_search and solutions_with_fitness and ls_evals_used < ls_budget:
                best_idx = max(range(len(solutions_with_fitness)), 
                              key=lambda i: solutions_with_fitness[i][1])
                sol, fit = solutions_with_fitness[best_idx]
                
                max_ls = min(n_vars // 2, ls_budget - ls_evals_used)
                if max_ls > 0:
                    improved_sol, improved_fit, used = self.local_search(
                        sol.copy(), func, max_ls
                    )
                    ls_evals_used += used
                    
                    if improved_fit > best_fitness:
                        best_fitness = improved_fit
                        best_solution = improved_sol.copy()
                    
                    solutions_with_fitness[best_idx] = (improved_sol, improved_fit)
            
            # Update pheromones
            self.update_pheromones(solutions_with_fitness, best_solution, best_fitness)
            
            if best_fitness >= optimum:
                break
        
        return best_fitness, best_solution


def test_single_run():
    """Test on F2 for debug issues."""
    
    problem = get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO)
    
    aco = StandardACO(n_ants=10, rho=0.1, debug=True)
    best_fit, best_sol = aco.optimize(problem, budget=10000)
    
    print("\nRESULT:")
    print(f"Fitness: {best_fit}/100")
    print(f"First 30 bits: {best_sol[:30]}")
    
    if best_fit >= 70:
        print("\nSUCCESS")
    

def run_full_benchmark():
    """Run final ACO on all benchmarks."""
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
    print("FINAL ACO FOR EXERCISE 5")
    print("="*80)
    print("Features: Standard update + Elitist + Local Search")
    print("="*80 + "\n")
    
    l = logger.Analyzer(
        root="data",
        folder_name="final_aco_ex5",
        algorithm_name="FinalACO_Ex5",
        algorithm_info="Standard update (no sub) + Elitist(w=5) + LS: 15 ants, rho=0.1"
    )
    
    results = {}
    
    for fid, name in problems:
        print(f"F{fid} ({name})...")
        
        problem = get_problem(
            fid=fid,
            dimension=100,
            instance=1,
            problem_class=ProblemClass.PBO
        )
        
        problem.attach_logger(l)
        
        for run in range(10):
            aco = FinalACO(n_ants=15, rho=0.1, elitist_weight=5, local_search=True)
            best_fit, best_sol = aco.optimize(problem, budget=100000)
            problem.reset()
        
        results[fid] = best_fit
        print(f"  Best: {best_fit}\n")
    
    del l
    
    print("="*80)
    print("FINAL RESULTS")
    print("="*80)
    for fid, name in problems:
        print(f"F{fid:2d} {name:<20} {results[fid]:>10.2f}")
    print("="*80)


if __name__ == "__main__":
    import sys
    run_full_benchmark()