# Exercise 2: Multi-Objective Evolutionary Submodular Optimisation
## GSEMO Algorithm Analysis and Results

---

## 1. Implementation Overview

The GSEMO (Global Simple Evolutionary Multi-Objective Optimizer) algorithm was implemented with the following multi-objective formulation:

- **Objective 1 (f1)**: Maximize submodular function value (coverage/influence)
- **Objective 2 (f2)**: Minimize cost (number of selected nodes, where each node has cost = 1)

The algorithm maintains a population of non-dominated solutions using Pareto dominance relationships, where solution A dominates solution B if:
- f1(A) ≥ f1(B) AND f2(A) ≤ f2(B)
- At least one inequality is strict

Key algorithmic components:
- **Initialization**: Single solution starting with all zeros
- **Mutation**: Standard bit-flip with probability 1/n per bit
- **Selection**: Uniform random selection from current Pareto front
- **Update**: Environmental selection based on Pareto dominance

---

## 2. Results Analysis

### 2.1 MaxCoverage Problems (Instances 2100-2103)

#### Performance Summary
The GSEMO algorithm demonstrates a characteristic performance profile across all MaxCoverage instances:

**Early Stage (0-500 evaluations):**
GSEMO exhibits exceptional performance in the initial optimization phase, achieving rapid fitness improvements that significantly outpace all baseline algorithms (EA, GA, RLS). The green curve in the plots shows steep ascent, indicating efficient exploration of high-quality regions in the search space.

**Mid Stage (500-2000 evaluations):**
The algorithm maintains competitive performance, with continued steady improvements. However, the rate of fitness gain begins to decelerate as the Pareto front becomes more populated and the search transitions from exploration to exploitation.

**Late Stage (2000-10000 evaluations):**
A clear convergence disadvantage emerges. Single-objective algorithms (particularly EA and RLS) gradually overtake GSEMO, achieving superior final fitness values. The performance gap at budget exhaustion ranges from approximately 5-10% across instances:

- **comp2100**: GSEMO ~400, EA/RLS ~410-420
- **comp2101**: GSEMO ~420, EA/RLS ~430-440  
- **comp2102**: GSEMO ~520, EA/RLS ~530-535
- **comp2103**: GSEMO ~660, EA/RLS ~670-680

#### Algorithmic Insights

The superior early performance stems from GSEMO's population-based approach. By maintaining multiple non-dominated solutions representing different cost-coverage trade-offs, the algorithm effectively explores diverse regions of the search space simultaneously. This diversity facilitates rapid discovery of promising solution components.

However, the late-stage disadvantage reveals a fundamental trade-off in the multi-objective formulation. The Pareto-based selection distributes selection pressure across the entire non-dominated front, reducing focus on maximizing f1 alone. In contrast, single-objective algorithms maintain concentrated selection pressure on coverage maximization, enabling more effective fine-tuning in later generations.

The monotone submodular property of MaxCoverage (adding nodes never decreases coverage) aligns well with GSEMO's optimization strategy, ensuring that the multi-objective formulation with cost minimization remains valid throughout optimization.

---

### 2.2 MaxInfluence Problems (Instances 2200-2203)

#### Performance Summary

MaxInfluence instances exhibit similar performance patterns to MaxCoverage, with some notable variations:

**Overall Trajectory:**
GSEMO again dominates early-stage optimization (0-500 evaluations), showing rapid improvement. The subsequent convergence pattern mirrors MaxCoverage, with single-objective algorithms eventually achieving superior final solutions.

**Instance-Specific Observations:**

- **comp2200**: GSEMO maintains closer proximity to top performers throughout optimization, with final gap of only 2-3%
- **comp2201**: Most competitive GSEMO performance observed, nearly matching EA and RLS at convergence
- **comp2202**: Moderate performance gap (~4-5%) emerges by final evaluation
- **comp2203**: Larger convergence gap similar to some MaxCoverage instances

#### Problem-Specific Analysis

The relatively stronger performance on MaxInfluence compared to MaxCoverage suggests that influence maximization's network structure may be more amenable to GSEMO's approach. Influence propagation involves complex dependencies where different node combinations can achieve similar influence through distinct propagation paths.

The diversity maintained by GSEMO's Pareto front may better capture these alternative high-quality regions, particularly when influence cascades create non-obvious synergies between node selections. However, the fundamental limitation persists: reduced selection pressure in late stages prevents optimal convergence compared to single-objective methods.

The algorithm successfully handles the monotone submodular nature of influence maximization, confirming that the multi-objective cost formulation appropriately models the problem structure.

---

### 2.3 PackWhileTravel Problems (Instances 2300-2302)

#### Performance Summary

**Critical Failure:** GSEMO encounters catastrophic failure across all PackWhileTravel instances, producing essentially zero fitness throughout the entire evaluation budget.

Observed behavior:
- **comp2300**: Flat line near zero, while other algorithms reach ~60,000
- **comp2301**: Minimal improvement to ~10,000, while GA achieves ~500,000
- **comp2302**: Remains near zero, while other algorithms exceed 1,000,000

#### Root Cause Analysis

Extensive investigation reveals the failure stems from a fundamental incompatibility between GSEMO's initialization strategy and PackWhileTravel's fitness landscape:

**Diagnostic Findings (Problem 2300):**
- All-zeros solution: f1 = -2851, f2 = 0
- Random solutions (~50% density): f1 ≈ -270,000, f2 ≈ 130-150
- Optimal region (10-20 nodes): f1 = 7,000-14,000 (positive!)

**The Domination Trap:**

1. **Problematic Initialization**: GSEMO starts with all-zeros solution having negative f1 (-2851) but zero cost (f2=0)

2. **Mutation Degradation**: Random bit-flips typically add nodes, increasing cost (f2) while making f1 more negative (from -2851 to approximately -270,000)

3. **Pareto Rejection**: All mutated solutions are dominated:
   ```
   All-zeros:  (f1 = -2851,    f2 = 0)
   Mutated:    (f1 = -270,000, f2 = 138)
   
   Dominance check:
   - f1: -2851 > -270,000 ✓ (all-zeros better)
   - f2: 0 < 138 ✓ (all-zeros better)
   → Mutated solution REJECTED
   ```

4. **Optimization Deadlock**: Population remains at all-zeros for all 10,000 evaluations as no offspring can pass dominance check

**Non-Monotone Submodularity:**

Unlike MaxCoverage and MaxInfluence, PackWhileTravel exhibits non-monotone submodularity:
- Adding arbitrary nodes can severely decrease objective value
- Optimal solutions exist in a "sweet spot" of moderate node selection
- Too few nodes: negative baseline penalty
- Optimal range: positive value (problem-dependent)
- Too many nodes: severe overloading penalty

This creates a fitness valley that GSEMO's strict Pareto dominance cannot traverse. Single-objective algorithms succeed because they accept worse solutions during search (via selection mechanisms or population dynamics), enabling escape from negative fitness regions.

#### Required Modifications

To successfully apply GSEMO to PackWhileTravel, substantial algorithmic modifications are needed:

1. **Intelligent Initialization**: Replace all-zeros with greedy construction or smart random initialization to start in feasible regions
2. **Modified Dominance**: Implement feasibility-aware dominance that prioritizes positive-fitness solutions
3. **Constraint Handling**: Add explicit mechanisms for constraint satisfaction and penalty management
4. **Adaptive Operators**: Develop mutation strategies that respect problem structure

Without these modifications, the current GSEMO implementation is fundamentally unsuitable for non-monotone submodular optimization with complex fitness landscapes.

---

## 3. Trade-off Analysis

### 3.1 Pareto Front Visualization

For each problem instance, the first run's Pareto front reveals the objective space structure:

**MaxCoverage and MaxInfluence:**
The Pareto fronts display characteristic concave trade-off curves between coverage/influence (f1) and cost (f2). Key observations:

- **Shape**: Monotonically increasing but with diminishing returns, reflecting submodular property
- **Coverage**: Fronts span from low-cost/low-coverage to high-cost/high-coverage solutions
- **Diversity**: Multiple non-dominated points represent distinct trade-off options for different budget constraints
- **Decision Support**: Fronts enable informed decision-making by revealing cost-benefit ratios at different operating points

The trade-offs demonstrate that GSEMO successfully identifies the spectrum of efficient solutions, providing valuable information for resource allocation decisions when budget constraints vary.

**PackWhileTravel:**
Due to algorithmic failure, Pareto fronts for PackWhileTravel (if any solutions exist beyond initialization) show degenerate behavior with solutions clustered near the origin or in negative fitness regions, confirming the implementation's inability to solve these instances.

### 3.2 Multi-Objective vs. Single-Objective Trade-offs

**Advantages of Multi-Objective Approach:**
- Provides comprehensive view of cost-benefit trade-offs
- Single run yields multiple solutions for different budgets
- Useful when budget constraints are uncertain or variable
- Enables post-hoc decision-making without re-optimization

**Disadvantages:**
- Lower convergence quality for any specific objective
- Increased computational overhead from population maintenance
- Requires careful dominance relation design for problem class
- May fail on non-monotone or constrained problems

---

## 4. Comparative Algorithm Performance

### 4.1 Algorithm Rankings by Problem Type

**MaxCoverage (Final Quality):**
1. EA / RLS (tied, highest final fitness)
2. GA (close second)
3. GSEMO (5-10% behind leaders)

**MaxCoverage (Anytime Performance):**
1. GSEMO (best for budgets < 1000)
2. EA / RLS
3. GA

**MaxInfluence (Final Quality):**
1. EA / RLS (tied or close)
2. GSEMO (competitive, 2-5% gap)
3. GA

**PackWhileTravel:**
1. GA (strongest on comp2301-2302)
2. EA / RLS
3. GSEMO (failed completely)

### 4.2 Algorithm Selection Guidelines

**Choose GSEMO when:**
- Problem exhibits monotone submodularity
- Evaluation budget is severely limited (< 1000 evaluations)
- Multiple trade-off solutions are desired from single run
- Early good solutions are more valuable than optimal convergence
- Decision flexibility regarding cost constraints is needed

**Choose EA/RLS when:**
- Convergence quality is paramount
- Full evaluation budget is available
- Single best solution is sufficient
- Problem may have non-monotone characteristics

**Choose GA when:**
- Population diversity benefits are expected
- Problem has complex constraint structure
- Crossover can exploit solution building blocks

---

## 5. Conclusions and Recommendations

### 5.1 Key Findings

1. **Problem-Dependent Efficacy**: GSEMO's effectiveness is highly dependent on problem structure, excelling on monotone submodular problems but failing on non-monotone variants.

2. **Exploration-Exploitation Trade-off**: The multi-objective formulation provides superior exploration (early-stage performance) at the cost of exploitation (late-stage convergence).

3. **Practical Value**: For monotone problems with budget constraints, GSEMO offers excellent anytime performance and trade-off visualization, making it valuable despite convergence limitations.

4. **Implementation Sensitivity**: Success requires careful problem analysis and potentially problem-specific modifications to initialization, dominance relations, and operators.

### 5.2 Future Research Directions

**For Monotone Problems:**
- Hybrid approaches combining GSEMO's early exploration with single-objective exploitation
- Adaptive selection pressure mechanisms that increase focus on primary objective over time
- Pareto front size limiting to maintain efficiency and increase selection pressure

**For Non-Monotone Problems:**
- Problem-aware initialization strategies (greedy, heuristic-based)
- Modified dominance relations incorporating feasibility and constraint satisfaction
- Constraint-handling techniques (penalties, repair operators, feasibility preservation)
- Integration with local search for refinement within feasible regions

### 5.3 Assignment Insights

The experimental results provide several important insights for evolutionary algorithm design:

1. **No Free Lunch**: Algorithm performance is fundamentally tied to problem characteristics; superior performance on one problem class does not guarantee success on others.

2. **Initialization Matters**: Particularly for constrained or non-monotone problems, initialization strategy can determine success or failure.

3. **Dominance Design**: Standard Pareto dominance works well for monotone multi-objective problems but requires adaptation for complex constraint structures or non-monotone objectives.

4. **Anytime vs. Convergence**: Algorithm selection should consider whether anytime performance or final convergence quality is more important for the application.

The GSEMO implementation successfully demonstrates multi-objective evolutionary optimization principles on monotone submodular problems while highlighting the challenges of extending such approaches to more complex problem classes. The PackWhileTravel failure, while representing unsuccessful optimization, provides valuable learning about algorithm limitations and the importance of problem-specific algorithm design.

---

## 6. Technical Implementation Notes

**Code Structure:**
- `GSEMO.py`: Main algorithm implementation with population management and Pareto dominance
- `Multi_Objective_Fitness_Function.py`: Fitness evaluation wrapper handling both objectives
- `run_gsemo.py`: Experimental framework for running multiple instances and independent runs

**Key Parameters:**
- Budget: 10,000 function evaluations (fixed)
- Independent runs: 30 per instance
- Mutation rate: 1/n per bit
- Population: Variable size based on Pareto front
- Initialization: Single all-zeros solution

**Logging:**
IOH logger integration enables standardized performance tracking and comparison with baseline algorithms.