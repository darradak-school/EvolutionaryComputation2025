import os, random,itertools, sys, gc
import numpy as np
class TSP:
    def __init__(self, File_Path=None):
        TSP.Clear_Memory()
        self.TSP_File_Path_List = [f"../Assignment·1/ALL_tsp/{i}" for i in os.listdir(File_Path) if i.endswith(".tsp")]
        # TSP.Evolution_Strategies_Process(self.TSP_File_Path_List[0], Generation_Iteration=1500, Selection_Operator="Fitness Proportional", Crossover_Operator='cycle', Mutation_Operator='Insert')
        TSP.Process_Benchmark_ALL_FILE(self.TSP_File_Path_List)
    @classmethod
    def Clear_Memory(cls): gc.collect()
    @staticmethod
    def Load_TSP_File(file_path):
        with open(file_path, "r") as File:  TSP_File = [line.replace("\n", "") for line in File.readlines()]
        data_lines = TSP_File[TSP_File.index('NODE_COORD_SECTION') + 1 : TSP_File.index('EOF')]
        TSP_Array = np.array([line.split() for line in data_lines], dtype=float)
        node, X_Y_Coordinates = TSP_Array[::,0].astype(int), TSP_Array[::,1:3]
        return node, X_Y_Coordinates
    @staticmethod
    def TSP_Distance_Cost(X_Y_Coordinates):
        Each_node_diff = X_Y_Coordinates[:, np.newaxis, :] - X_Y_Coordinates[np.newaxis, :, :]
        return np.sqrt(np.sum(Each_node_diff**2, axis=2))   # node_distances
    @staticmethod
    def Initial_different_Parent(TSP_Node, population_sizes=None):
        population_sizes = random.randint(1, 10)*10 if population_sizes is None else population_sizes
        Parent_population_diff = np.array([np.random.permutation(TSP_Node) for i in range(population_sizes)])
        return Parent_population_diff
    @classmethod
    def Evolution_Strategies_Process(cls, TSP_File_Path, Generation_Iteration=None, Selection_Operator=None, Crossover_Operator=None, Mutation_Operator=None, population_size=50, crossover_rate=0.8, mutation_rate=0.1):
        # ----------------------------------------
        cls.Selection_Operator = [method for method in dir(Selection) if not method.startswith('_')]
        cls.Crossover_Operator = [method for method in dir(Crossover) if not method.startswith('_')]
        cls.Mutation_Operator = [method for method in dir(Mutation) if not method.startswith('_')]
        Selection_Operator = random.choice(cls.Selection_Operator) if Selection_Operator is None else [i for i in cls.Selection_Operator for name in Selection_Operator.split() if name.lower() in i.replace("_Selection", "").lower()][0]
        Crossover_Operator = random.choice(cls.Crossover_Operator) if Crossover_Operator is None else [i for i in cls.Crossover_Operator for name in Crossover_Operator.split() if name.lower() in i.replace("_Crossover", "").lower()][0]
        Mutation_Operator = random.choice(cls.Mutation_Operator) if Mutation_Operator is None else [i for i in cls.Mutation_Operator for name in Mutation_Operator.split() if name.lower() in i.replace("_Mutation", "").lower()][0]
        Generation_Iteration, Generation_Iteration_Count = random.choice([2000, 5000, 10000, 20000]) if Generation_Iteration is None else Generation_Iteration, 0
        # ----------------------------------------
        TSP_city, TSP_locations = TSP.Load_TSP_File(TSP_File_Path)
        cls.TSP_City_distance_matrix = cls.TSP_Distance_Cost(TSP_locations)
        Parent_Population = TSP.Initial_different_Parent(TSP_city, population_sizes=population_size)
        Best_Solution, Best_Fitness, fitness_history = None, float('inf'), []
        while Generation_Iteration_Count < Generation_Iteration:
            parent_pool = getattr(Selection, Selection_Operator)(cls.TSP_City_distance_matrix, Parent_Population)
            # ----------------------------------------
            offspring_population = []
            for i in range(0, len(parent_pool) - 1, 2):
                parent1 = parent_pool[i]
                parent2 = parent_pool[i + 1] if i + 1 < len(parent_pool) else parent_pool[0]
                if random.random() < crossover_rate and Crossover_Operator:
                    try:
                        child1, child2 = getattr(Crossover, Crossover_Operator)(parent1, parent2)
                        offspring_population.extend([child1, child2])
                    except:
                        offspring_population.extend([parent1.copy(), parent2.copy()])
                else:
                    offspring_population.extend([parent1.copy(), parent2.copy()])
            # --------------------
            offspring_population = np.array(offspring_population)
            # --------------------
            if Mutation_Operator:
                for i in range(len(offspring_population)):
                    if random.random() < mutation_rate:
                        try:
                            offspring_population[i] = getattr(Mutation, Mutation_Operator)(offspring_population[i])
                        except:
                            pass
            # --------------------
            offspring_population = np.array(offspring_population[:population_size])
            if "Elitist" in Selection_Operator:
                Parent_Population = Selection.Elitist_Selection(cls.TSP_City_distance_matrix,Parent_Population, offspring_population)
            else:
                Parent_Population = offspring_population
            # --------------------
            current_fitness = Selection._Population_Group_Fitness(Parent_Population, cls.TSP_City_distance_matrix)
            min_fitness_idx = np.argmin(current_fitness)

            if current_fitness[min_fitness_idx] < Best_Fitness:
                Best_Fitness = current_fitness[min_fitness_idx]
                Best_Solution = Parent_Population[min_fitness_idx].copy()

            fitness_history.append(Best_Fitness)

            Generation_Iteration_Count += 1

            if Generation_Iteration_Count % 100 == 0:
                print(f"Generation {Generation_Iteration_Count}: Best Fitness = {Best_Fitness:.2f}")
            # ----------------------------------------
        print(f"\n=== Final Results ===")
        print(f"Best Path Length: {Best_Fitness:.2f}")
        print(f":Selection Operator {Selection_Operator}")
        print(f"Crossover Operator: {Crossover_Operator}")
        print(f"Mutation Operator: {Mutation_Operator}")
        return Best_Solution, Best_Fitness, fitness_history
    @classmethod
    def Process_Benchmark_ALL_FILE(cls, All_TSP_File_Path_List):
        def config(): return [
            {
                'name'          : 'EA1_Tournament_Order_Insert',
                'selection'     : 'Tournament',
                'crossover'     : 'Order',
                'mutation'      : 'Insert',
                'crossover_rate': 0.8,
                'mutation_rate' : 0.2
            },
            {
                'name'          : 'EA2_FitnessProportional_PMX_Swap',
                'selection'     : 'Fitness Proportional',
                'crossover'     : 'PMX',
                'mutation'      : 'Swap',
                'crossover_rate': 0.9,
                'mutation_rate' : 0.1
            },
            {
                'name'          : 'EA3_Elitist_Cycle_Inversion',
                'selection'     : 'Elitist',
                'crossover'     : 'Cycle',
                'mutation'      : 'Inversion',
                'crossover_rate': 0.7,
                'mutation_rate' : 0.3
            }
        ], [20, 50, 100, 200], [2000, 5000, 10000, 20000]
        Evolution_Strategies_config, population_sizes, generation_checkpoints = config()
        # ----------------------------------------
        Benchmark_results = {}
        print("=== Starting Benchmark Testing ===\n")
        for config_idx, config in enumerate(Evolution_Strategies_config, 1):
            print(f"Testing Configuration {config_idx}/{len(Evolution_Strategies_config)}: {config['name']}")
            config_results = {}
            for file_idx, file_path in enumerate(All_TSP_File_Path_List, 1):
                instance_name = file_path.split('/')[-1].replace('.tsp', '')
                print(f"  Instance {file_idx}/{len(All_TSP_File_Path_List)}: {instance_name}")
                instance_results = {}
                for pop_size in population_sizes:
                    print(f"    Population Size: {pop_size}")
                    # Run EA with max generations
                    best_solution, best_fitness, fitness_history = cls.Evolution_Strategies_Process(
                        TSP_File_Path=file_path,
                        Generation_Iteration=generation_checkpoints[-1],
                        Selection_Operator=config['selection'],
                        Crossover_Operator=config['crossover'],
                        Mutation_Operator=config['mutation'],
                        population_size=pop_size,
                        crossover_rate=config['crossover_rate'],
                        mutation_rate=config['mutation_rate']
                    )
                    # checkpoint_fitness = {}
                    # for checkpoint in generation_checkpoints:
                    #     checkpoint_fitness[f'gen_{checkpoint}'] = fitness_history[checkpoint - 1] if checkpoint <= len(fitness_history) else fitness_history[-1]
                    checkpoint_fitness = {f'gen_{checkpoint}': fitness_history[checkpoint - 1] if checkpoint <= len(fitness_history) else fitness_history[-1] for checkpoint in generation_checkpoints}


                    instance_results[f'pop_{pop_size}'] = {
                        'best_fitness'  : best_fitness,
                        'checkpoints'   : checkpoint_fitness,
                        'final_solution': best_solution.tolist()
                    }

                config_results[instance_name] = instance_results

            Benchmark_results[config['name']] = config_results
            print(f"  Configuration {config['name']} completed!\n")

        # cls._save_benchmark_results(Benchmark_results)

        print("=== Running Best Algorithm 30 times ===")
        cls._run_best_algorithm_multiple_times(All_TSP_File_Path_List, Evolution_Strategies_config)

        return Benchmark_results
    @classmethod
    def _run_best_algorithm_multiple_times(cls, ALL_TSP_File_Path, configs):
        best_config = configs[0]

        results_for_report = {}

        print("\nRunning best algorithm 30 times per instance...")

        for file_path in ALL_TSP_File_Path:
            instance_name = file_path.split('/')[-1].replace('.tsp', '')
            print(f"  Processing {instance_name}...")

            fitness_values = []

            for run in range(30):
                _, best_fitness, _ = cls.Evolution_Strategies_Process(
                    TSP_File_Path=file_path,
                    Generation_Iteration=20000,
                    Selection_Operator=best_config['selection'],
                    Crossover_Operator=best_config['crossover'],
                    Mutation_Operator=best_config['mutation'],
                    population_size=50,
                    crossover_rate=best_config['crossover_rate'],
                    mutation_rate=best_config['mutation_rate']
                )
                fitness_values.append(best_fitness)

                if (run + 1) % 10 == 0:
                    print(f"    Completed {run + 1}/30 runs")

            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)
            min_fitness = np.min(fitness_values)
            max_fitness = np.max(fitness_values)

            results_for_report[instance_name] = {
                'mean': mean_fitness,
                'std' : std_fitness,
                'min' : min_fitness,
                'max' : max_fitness
            }

        with open('results/your_EA.txt', 'w') as f:
            f.write("=== Best EA Results (30 runs per instance) ===\n")
            f.write(f"Algorithm: {best_config['name']}\n")
            f.write(f"Population Size: 50, Generations: 20000\n\n")

            f.write(f"{'Instance':<15} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}\n")
            f.write("-" * 65 + "\n")

            for instance, stats in results_for_report.items():
                f.write(f"{instance:<15} {stats['mean']:<12.2f} {stats['std']:<12.2f} "
                        f"{stats['min']:<12.2f} {stats['max']:<12.2f}\n")

        print("\nBenchmark testing completed!")
        print("Results saved to:")
        print("  - results/benchmark_detailed.json")
        print("  - results/benchmark_summary.txt")
        print("  - results/your_EA.txt")
    @classmethod
    def _save_benchmark_results(cls, benchmark_results):
        import json

        with open('results/benchmark_detailed.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)

        with open('results/benchmark_summary.txt', 'w') as f:
            f.write("=== TSP Evolutionary Algorithm Benchmark Results ===\n\n")

            for ea_name, ea_results in benchmark_results.items():
                f.write(f"Algorithm: {ea_name}\n")
                f.write("-" * 50 + "\n")

                for instance, pop_results in ea_results.items():
                    f.write(f"\nInstance: {instance}\n")

                    for pop_size, results in pop_results.items():
                        f.write(f"  {pop_size}:\n")
                        f.write(f"    Best Fitness: {results['best_fitness']:.2f}\n")
                        f.write(f"    Checkpoints:\n")

                        for gen, fitness in results['checkpoints'].items():
                            f.write(f"      {gen}: {fitness:.2f}\n")

                f.write("\n" + "=" * 50 + "\n\n")
class Selection:
    @staticmethod
    def _Premutation_Total_Distance(Premutation, Premutation_distance_matrix):
        Premutation_idx = Premutation - 1
        Distances = Premutation_distance_matrix[Premutation_idx, np.roll(Premutation_idx, -1)]
        return np.sum(Distances)
    @staticmethod
    def _Population_Group_Fitness(Population_Group, Premutation_distance_matrix):
        All_Fitness_Scores = np.array([Selection._Premutation_Total_Distance(i, Premutation_distance_matrix) for i in Population_Group])
        return All_Fitness_Scores
    @classmethod
    def Elitist_Selection(cls, Premutation_distance_matrix,  Parent_Group, Children_Group=None, Elitist_Children=None):
        Elitist_Children = len(Parent_Group) if Elitist_Children is None else Elitist_Children
        if Children_Group is None: return Parent_Group
        Parent_Maxed_Children = np.concatenate((Parent_Group, Children_Group))
        Fitness_Scores = Selection._Population_Group_Fitness(Parent_Maxed_Children, Premutation_distance_matrix)
        Lower_Fitness_Order = np.argsort(Fitness_Scores)[:Elitist_Children]
        Elitist_Group = Parent_Maxed_Children[Lower_Fitness_Order]
        return Elitist_Group
    @classmethod
    def Tournament_Selection(cls, Premutation_distance_matrix, Parent_Group, k=3):
        Parent_Select_Pool = []
        for i in range(len(Parent_Group)):
            Parent_k_idx = np.random.choice([i for i in range(len(Parent_Group))], size=k, replace=False, p=None)
            Parent_k = [Parent_Group[i] for i in Parent_k_idx]
            Fitness_Scores = Selection._Population_Group_Fitness(Parent_k, Premutation_distance_matrix)
            Parent_Select_Pool.append(int(Parent_k_idx[np.argmin(Fitness_Scores)]))
        Parent_Select_Pool = np.array([Parent_Group[i] for i in Parent_Select_Pool])
        return Parent_Select_Pool
    @classmethod
    def Fitness_Proportional_Selection(cls, Premutation_distance_matrix, Parent_Group):
        Fitness_Scores = Selection._Population_Group_Fitness(Parent_Group, Premutation_distance_matrix)
        Fitness_Scores = 1 / Fitness_Scores
        Probabilities = Fitness_Scores / np.sum(Fitness_Scores)
        # ----------------------------------------
        parent_pool_idx = np.random.choice(np.array([i for i in range(len(Parent_Group))]), size=len(Parent_Group), replace=True, p=Probabilities)
        parent_pool = np.array([Parent_Group[i] for i in parent_pool_idx])
        return parent_pool
class Crossover:
    @staticmethod
    def _Check_Parents_Length(parent1, parent2):
        if len(parent1) != len(parent2): raise ValueError("Parents must be of the same length")
    @staticmethod
    def _numpyArray_to_list(array): return array.tolist()
    @staticmethod
    def _list_to_numpyArray(array): return np.array(array)
    @classmethod
    def One_Point_Crossover(cls, parent1, parent2):
        parent1, parent2 = cls._numpyArray_to_list(parent1), cls._numpyArray_to_list(parent2)
        Crossover._Check_Parents_Length(parent1, parent2)
        P_C = random.randint(int(len(parent1) * 0.6), int(len(parent1) * 0.9))
        return cls._list_to_numpyArray(parent1[:P_C] + parent2[P_C:]), cls._list_to_numpyArray(parent2[:P_C] + parent1[P_C:])
    @classmethod
    def N_Point_Crossover(cls, parent1, parent2, n=None):
        parent1, parent2 = cls._numpyArray_to_list(parent1), cls._numpyArray_to_list(parent2)
        n = random.randint(int(len(parent1) * 0.6), int(len(parent1) * 0.9)) if n is None else n
        crossover_points = sorted(random.sample(range(1, int((len(parent1) + len(parent2)) / 2)), n))
        children_1 = list(itertools.chain(
            *[parent1[:i] if (idx == 0) else parent1[crossover_points[idx - 1]:i] if idx % 2 == 0 else parent2[crossover_points[idx - 1]:i]
              for idx, i in enumerate(crossover_points)] + [
                 parent1[crossover_points[-1]::] if len(crossover_points) % 2 == 0 else parent2[crossover_points[-1]::]]))
        children_2 = list(itertools.chain(
            *[parent2[:i] if (idx == 0) else parent2[crossover_points[idx - 1]:i] if idx % 2 == 0 else parent1[crossover_points[idx - 1]:i]
              for idx, i in enumerate(crossover_points)] + [
                 parent2[crossover_points[-1]::] if len(crossover_points) % 2 == 0 else parent1[crossover_points[-1]::]]))
        return cls._list_to_numpyArray(children_1), cls._list_to_numpyArray(children_2)
    @classmethod
    def Uniform_Crossover(cls, parent1, parent2):
        children1, children2 = [], []
        for i in range(len(parent1)):
            n = random.randint(0, 1)
            children1.append(parent1[i] if n == 0 else parent2[i])
            children2.append(parent2[i] if n == 0 else parent1[i])
        return cls._list_to_numpyArray(children1), cls._list_to_numpyArray(children2)
    @classmethod
    def Order_Crossover(cls, parent1, parent2):
        parent1, parent2 = cls._numpyArray_to_list(parent1), cls._numpyArray_to_list(parent2)
        cls._Check_Parents_Length(parent1, parent2)
        P_C_start = random.randint(int(len(parent1) * 0.1), int(len(parent1) * 0.5))
        P_C_end = random.randint(int(len(parent1) * 0.6), int(len(parent1) * 1))
        # ----------------------------------------
        children1_fix, children2_fix = parent1[P_C_start:P_C_end], parent2[P_C_start:P_C_end]
        children1_releate = [i for i in parent2[P_C_end:] + parent2[:P_C_end] if i not in children1_fix]
        children2_releate = [i for i in parent1[P_C_end:] + parent1[:P_C_end] if i not in children2_fix]
        # ====================
        children1_head, children2_head = children1_releate[len(parent1) - P_C_end:], children2_releate[len(parent2) - P_C_end:]
        children1_tail, children2_tail = children1_releate[:len(parent1) - P_C_end], children2_releate[:len(parent2) - P_C_end]
        # ====================
        children1 = children1_head + children1_fix + children1_tail
        children2 = children2_head + children2_fix + children2_tail
        return cls._list_to_numpyArray(children1), cls._list_to_numpyArray(children2)
    @classmethod
    def Cycle_Crossover(cls, parent1, parent2):
        parent1, parent2 = cls._numpyArray_to_list(parent1), cls._numpyArray_to_list(parent2)
        cls._Check_Parents_Length(parent1, parent2)
        children1, children2 = [None] * len(parent1), [None] * len(parent2)
        children1[0], children2[0] = parent1[0], parent2[0]
        next_pos = parent1.index(parent2[0])
        while next_pos != 0: children1[next_pos], children2[next_pos], next_pos = parent1[next_pos], parent2[next_pos], parent1.index(parent2[next_pos])
        for i in range(len(parent1)):
            children1[i] = parent2[i] if children1[i] is None else children1[i]
            children2[i] = parent1[i] if children2[i] is None else children2[i]
        return cls._list_to_numpyArray(children1), cls._list_to_numpyArray(children2)
    @classmethod
    def PMX_Crossover(cls, parent1, parent2):
        total_length = int((len(parent1) + len(parent2)) // 2)
        # --------------------
        P_C_start, P_C_end = np.random.randint(int(len(parent1) * 0.1), int(len(parent1) * 0.5)), np.random.randint(int(len(parent1) * 0.6), int(len(parent1) * 0.9))
        total_length = int((len(parent1) + len(parent2)) // 2)
        # ----------------------------------------
        mask = np.arange(total_length)
        crossover_mask = (mask >= P_C_start) & (mask < P_C_end)
        # ----------------------------------------
        children1, children2 = np.where(crossover_mask, parent1, parent2), np.where(crossover_mask, parent2, parent1)
        for i in range(total_length):
            while children1[i] in parent1[P_C_start: P_C_end] and not crossover_mask[i] : children1[i] = parent2[np.where(parent1 == children1[i])[0]].tolist()[0]
            while children2[i] in parent2[P_C_start: P_C_end] and not crossover_mask[i]: children2[i] = parent1[np.where(parent2 == children2[i])[0]].tolist()[0]
        return children1, children2
class Mutation:
    @staticmethod
    def _numpyArray_to_list(array): return array.tolist()
    @staticmethod
    def _list_to_numpyArray(array): return np.array(array)
    @classmethod
    def Insert_mutation(cls, parent):
        P_C_start = random.randint(0, len(parent) - 2)
        P_C_end = random.randint(P_C_start + 1, len(parent) - 1)
        children_head = np.array([parent[i] for i in range(P_C_start)])
        children_tail = np.array([parent[i] for i in range(P_C_start, len(parent)) if not (i == P_C_end)])
        return np.concatenate([children_head, np.array([parent[P_C_end]]), children_tail])
    @classmethod
    def Swap_mutation(cls, parent):
        P_C_start = random.randint(0, len(parent) - 2)
        P_C_end = random.randint(P_C_start + 1, len(parent) - 1)
        children = parent.copy()
        children[P_C_start], children[P_C_end] = parent[P_C_end], parent[P_C_start]
        return children
    @classmethod
    def Inversion_mutation(cls, parent):
        P_C_start = random.randint(int(len(parent) * 0.1), int(len(parent) * 0.5))
        P_C_end = random.randint(int(len(parent) * 0.6), int(len(parent) * 0.9))
        Inversion = list(reversed(parent[P_C_start:P_C_end]))
        # children = parent[:P_C_start] + Inversion +parent[P_C_end:]
        # print((parent[:P_C_start], np.array(Inversion), parent[P_C_end:]))
        return np.concatenate((parent[:P_C_start], np.array(Inversion), parent[P_C_end:]))
    @classmethod
    def Scramble_mutation(cls, parent):
        P_C_start = random.randint(int(len(parent) * 0.1), int(len(parent) * 0.5))
        P_C_end = random.randint(int(len(parent) * 0.6), int(len(parent) * 1))
        Scramble = list(parent[P_C_start:P_C_end])
        random.shuffle(Scramble)
        # children = parent[:P_C_start] + Scramble +parent[P_C_end:]
        print(parent[:P_C_start], np.array(Scramble,dtype=int), parent[P_C_end:])
        return np.concatenate((parent[:P_C_start], np.array(Scramble,dtype=int), parent[P_C_end:]))


# ========================================
File_Path=""
TSP(File_Path)