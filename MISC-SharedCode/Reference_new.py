import os, random,itertools, sys, gc, math, datetime
import numpy as np
# ----------------------------------------
class CONFIG:
    def __init__(self, File_Path, Data_Type=np.ndarray):
        self.TSP_File_Path_List = [f"{File_Path}/{i}" for i in os.listdir(File_Path) if i.endswith(".tsp")]
        self.Data_Type = Data_Type
    def Load_TSP_File(self, file_path):
        with open(file_path, "r") as File:  TSP_File = [line.replace("\n", "") for line in File.readlines()]
        data_lines = TSP_File[TSP_File.index('NODE_COORD_SECTION') + 1 : TSP_File.index('EOF')]
        TSP_Array = np.array([line.split() for line in data_lines], dtype=float)
        node, X_Y_Coordinates = TSP_Array[::,0].astype(int), TSP_Array[::,1:3]
        return (node, X_Y_Coordinates) if self.Data_Type == np.ndarray else (node.tolist(), X_Y_Coordinates.tolist())
class TSP:
    def __init__(self):     self.Clear_Memory()
    def Clear_Memory(self):      gc.collect()
    @staticmethod
    def TSP_Distance_Cost(X_Y_Coordinates):
        if not (CONFIG.Data_Type == np.ndarray):
            return [[math.sqrt((X_Y_Coordinates[i][0] - X_Y_Coordinates[j][0]) ** 2 + (X_Y_Coordinates[i][1] - X_Y_Coordinates[j][1]) ** 2) for j in range(len(X_Y_Coordinates))] for i in range(len(X_Y_Coordinates))]
        Each_node_diff = X_Y_Coordinates[:, np.newaxis, :] - X_Y_Coordinates[np.newaxis, :, :]
        return np.sqrt(np.sum(Each_node_diff ** 2, axis=2))  # node_distances
    @staticmethod
    def Initial_different_Parent(TSP_Node, population_sizes=None):
        population_sizes = random.randint(1, 10) * 10 if population_sizes is None else population_sizes
        Parent_population_diff = np.array([np.random.permutation(TSP_Node) for i in range(population_sizes)])
        return Parent_population_diff if CONFIG.Data_Type == np.ndarray else Parent_population_diff.tolist()
# ========================================
class Selection:
    @staticmethod
    def _Premutation_Total_Distance(Premutation, Premutation_distance_matrix):
        Premutation_idx = Premutation - 1 if CONFIG.Data_Type == np.ndarray else [x - 1 for x in Premutation]
        if CONFIG.Data_Type == np.ndarray:
            Distances = Premutation_distance_matrix[Premutation_idx, np.roll(Premutation_idx, -1)]
            return np.sum(Distances) if CONFIG.Data_Type == np.ndarray else np.sum(Distances).tolist()
        else:
            rolled_idx = Premutation_idx[1:] + [Premutation_idx[0]]  # 手動實現 roll
            total_distance = 0
            for i in range(len(Premutation_idx)):
                total_distance += Premutation_distance_matrix[Premutation_idx[i]][rolled_idx[i]]
            return total_distance
        # Distances = Premutation_distance_matrix[Premutation_idx, np.roll(Premutation_idx, -1)]
        # return np.sum(Distances) if CONFIG.Data_Type == np.ndarray else np.sum(Distances).tolist()
    @staticmethod
    def _Population_Group_Fitness(Population_Group, Premutation_distance_matrix):
        # ----------------------------------------
        if CONFIG.Data_Type == np.ndarray:  All_Fitness_Scores = np.array([Selection._Premutation_Total_Distance(i, Premutation_distance_matrix) for i in Population_Group])
        else:                               All_Fitness_Scores = [Selection._Premutation_Total_Distance(i, Premutation_distance_matrix) for i in Population_Group]
        return All_Fitness_Scores
    @classmethod
    def Elitist_Selection(cls, Premutation_distance_matrix,  Parent_Group, Children_Group=None, Elitist_Children=None):
        Elitist_Children = len(Parent_Group) if Elitist_Children is None else Elitist_Children
        if Children_Group is None: return Parent_Group
        Parent_Maxed_Children = np.concatenate((Parent_Group, Children_Group)) if CONFIG.Data_Type == np.ndarray else Parent_Group + Children_Group
        Fitness_Scores = Selection._Population_Group_Fitness(Parent_Maxed_Children, Premutation_distance_matrix)
        Lower_Fitness_Order = np.argsort(Fitness_Scores)[:Elitist_Children] if CONFIG.Data_Type == np.ndarray else sorted(range(len(Fitness_Scores)), key=lambda i: Fitness_Scores[i])[:Elitist_Children]
        Elitist_Group = Parent_Maxed_Children[Lower_Fitness_Order] if CONFIG.Data_Type == np.ndarray else [Parent_Maxed_Children[i] for i in Lower_Fitness_Order]
        return Elitist_Group
    @classmethod
    def Tournament_Selection(cls, Premutation_distance_matrix, Parent_Group, k=3):
        Parent_Select_Pool = []
        for i in range(len(Parent_Group)):
            Parent_k_idx = np.random.choice([i for i in range(len(Parent_Group))], size=k, replace=False, p=None)
            Parent_k = [Parent_Group[i] for i in Parent_k_idx]
            Fitness_Scores = Selection._Population_Group_Fitness(Parent_k, Premutation_distance_matrix)
            Parent_Select_Pool.append(int(Parent_k_idx[np.argmin(Fitness_Scores)]) if CONFIG.Data_Type == np.ndarray else int(Parent_k_idx[min(enumerate(Fitness_Scores), key=lambda x: x[1])[0]]))
        Parent_Select_Pool = [Parent_Group[i] for i in Parent_Select_Pool]
        return Parent_Select_Pool if CONFIG.Data_Type != np.ndarray else np.array(Parent_Select_Pool)
    @classmethod
    def Fitness_Proportional_Selection(cls, Premutation_distance_matrix, Parent_Group):
        Fitness_Scores = Selection._Population_Group_Fitness(Parent_Group, Premutation_distance_matrix)
        Fitness_Scores = 1 / Fitness_Scores if CONFIG.Data_Type == np.ndarray else [1/i for i in Fitness_Scores]
        Probabilities = Fitness_Scores / np.sum(Fitness_Scores) if CONFIG.Data_Type == np.ndarray else [i/sum(Fitness_Scores) for i in Fitness_Scores]
        # ----------------------------------------
        parent_pool_idx = random.choices(range(len(Parent_Group)), weights=Probabilities, k=len(Parent_Group)) if CONFIG.Data_Type != np.ndarray else np.random.choice(np.array([i for i in range(len(Parent_Group))]), size=len(Parent_Group), replace=True, p=Probabilities)
        parent_pool = [Parent_Group[i] for i in parent_pool_idx]
        return parent_pool if CONFIG.Data_Type != np.ndarray else np.array(parent_pool)
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
        if CONFIG.Data_Type == np.ndarray:
            parent1, parent2 = cls._numpyArray_to_list(parent1), cls._numpyArray_to_list(parent2)
        Crossover._Check_Parents_Length(parent1, parent2)
        P_C = random.randint(int(len(parent1) * 0.6), int(len(parent1) * 0.9))
        return (parent1[:P_C] + parent2[P_C:], parent2[:P_C] + parent1[P_C:]) if CONFIG.Data_Type != np.ndarray else np.array(parent1[:P_C] + parent2[P_C:]), np.array(parent2[:P_C] + parent1[P_C:])
    @classmethod
    def N_Point_Crossover(cls, parent1, parent2, n=None):
        if CONFIG.Data_Type == np.ndarray:
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
        return (children_1, children_2) if CONFIG.Data_Type != np.ndarray else (np.array(children_1), np.array(children_2))
    @classmethod
    def Uniform_Crossover(cls, parent1, parent2):
        children1, children2 = [], []
        for i in range(len(parent1)):
            n = random.randint(0, 1)
            children1.append(parent1[i] if n == 0 else parent2[i])
            children2.append(parent2[i] if n == 0 else parent1[i])
        return (children1, children2) if CONFIG.Data_Type != np.ndarray else (np.array(children1), np.array(children2))
    @classmethod
    def Order_Crossover(cls, parent1, parent2):
        if CONFIG.Data_Type == np.ndarray:
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
        return (children1, children2) if CONFIG.Data_Type != np.ndarray else (np.array(children1), np.array(children2))
    @classmethod
    def Cycle_Crossover(cls, parent1, parent2):
        if CONFIG.Data_Type == np.ndarray: parent1, parent2 = cls._numpyArray_to_list(parent1), cls._numpyArray_to_list(parent2)
        cls._Check_Parents_Length(parent1, parent2)
        children1, children2 = [None] * len(parent1), [None] * len(parent2)
        children1[0], children2[0] = parent1[0], parent2[0]
        next_pos = parent1.index(parent2[0])
        while next_pos != 0: children1[next_pos], children2[next_pos], next_pos = parent1[next_pos], parent2[next_pos], parent1.index(parent2[next_pos])
        for i in range(len(parent1)):
            children1[i] = parent2[i] if children1[i] is None else children1[i]
            children2[i] = parent1[i] if children2[i] is None else children2[i]
        return (children1, children2) if CONFIG.Data_Type != np.ndarray else (np.array(children1), np.array(children2))
    @classmethod
    def PMX_Crossover(cls, parent1, parent2):
        total_length = int((len(parent1) + len(parent2)) // 2)
        # --------------------
        if CONFIG.Data_Type == np.ndarray:
            P_C_start = np.random.randint(int(len(parent1) * 0.1), int(len(parent1) * 0.5))
            P_C_end = np.random.randint(int(len(parent1) * 0.6), int(len(parent1) * 0.9))
        else:
            P_C_start = random.randint(int(len(parent1) * 0.1), int(len(parent1) * 0.5))
            P_C_end = random.randint(int(len(parent1) * 0.6), int(len(parent1) * 1))
        # ----------------------------------------
        if CONFIG.Data_Type == np.ndarray:
            mask = np.arange(total_length)
            crossover_mask = (mask >= P_C_start) & (mask < P_C_end)
            # ----------------------------------------
            children1, children2 = np.where(crossover_mask, parent1, parent2), np.where(crossover_mask, parent2, parent1)
            for i in range(total_length):
                while children1[i] in parent1[P_C_start: P_C_end] and not crossover_mask[i]: children1[i] = \
                parent2[np.where(parent1 == children1[i])[0]].tolist()[0]
                while children2[i] in parent2[P_C_start: P_C_end] and not crossover_mask[i]: children2[i] = \
                parent1[np.where(parent2 == children2[i])[0]].tolist()[0]
            return children1, children2
        else:
            children1, children2 = [None] * total_length, [None] * total_length
            children1_dict = {i: parent2[parent1.index(i)] for i in parent1[P_C_start:P_C_end]}
            children2_dict = {i: parent1[parent2.index(i)] for i in parent2[P_C_start:P_C_end]}
            fix_element = list(range(P_C_start, P_C_end))
            for i in range(total_length):
                children1_value, children2_value = parent1[i], parent2[i]
                if i not in fix_element:
                    children1_value, children2_value = parent2[i], parent1[i]
                    while children1_value in children1_dict.keys(): children1_value = children1_dict[children1_value]
                    while children2_value in children2_dict.keys(): children2_value = children2_dict[children2_value]
                children1[i], children2[i] = children1_value, children2_value
            return children1, children2
class Mutation:
    @classmethod
    def Insert_mutation(cls, parent):
        P_C_start = random.randint(0, len(parent) - 2)
        P_C_end = random.randint(P_C_start + 1, len(parent) - 1)
        children_head = [parent[i] for i in range(P_C_start)]
        children_tail = [parent[i] for i in range(P_C_start, len(parent)) if not (i == P_C_end)]
        if CONFIG.Data_Type == np.ndarray:  return np.concatenate([np.array(children_head + [parent[P_C_end]] + children_tail)])
        else:                               return children_head + [parent[P_C_end]] + children_tail
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
        if CONFIG.Data_Type == np.ndarray:      return np.concatenate((parent[:P_C_start], np.array(Inversion), parent[P_C_end:]))
        else:                                   return parent[:P_C_start] + Inversion + parent[P_C_end:]
    @classmethod
    def Scramble_mutation(cls, parent):
        P_C_start = random.randint(int(len(parent) * 0.1), int(len(parent) * 0.5))
        P_C_end = random.randint(int(len(parent) * 0.6), int(len(parent) * 1))
        Scramble = list(parent[P_C_start:P_C_end])
        random.shuffle(Scramble)
        if CONFIG.Data_Type == np.ndarray:      return np.concatenate((parent[:P_C_start], np.array(Scramble, dtype=int), parent[P_C_end:]))
        else:                                   return parent[:P_C_start] + Scramble +parent[P_C_end:]
# ========================================
File_Path="/Users/zebb/Library/Mobile Documents/com~apple~CloudDocs/Documents/University Of Adelaide/Error404/Semester 1/COMP SCI 1015/Interpreter/PyCharm (history)/Project Advantage/-Pycharm-uni-Examination:Testing/uni-Examination/ - Testing - COMP SCI 3316/Assignment·1/ALL_tsp"
# ====================
CONFIG = CONFIG(File_Path)
TSP = TSP()
# ====================
CONFIG.Data_Type = list
# CONFIG.Data_Type = np.ndarray
# ====================
def Process_Benchmark_ALL_FILE(All_TSP_File_Path_List):
    def TSP_config():
        return [
            {
                'name'          : 'EA1_Tournament_Order_Insert',
                'selection'     : Selection.Tournament_Selection,
                'crossover'     : Crossover.Order_Crossover,
                'mutation'      : Mutation.Insert_mutation,
                'crossover_rate': 0.8,
                'mutation_rate' : 0.2
            },
            {
                'name'          : 'EA2_FitnessProportional_PMX_Swap',
                'selection'     : Selection.Fitness_Proportional_Selection,
                'crossover'     : Crossover.PMX_Crossover,
                'mutation'      : Mutation.Swap_mutation,
                'crossover_rate': 0.9,
                'mutation_rate' : 0.1
            },
            {
                'name'          : 'EA3_Elitist_Cycle_Inversion',
                'selection'     : Selection.Elitist_Selection,
                'crossover'     : Crossover.Cycle_Crossover,
                'mutation'      : Mutation.Inversion_mutation,
                'crossover_rate': 0.7,
                'mutation_rate' : 0.3
            }
        ], [20, 50, 100, 200], [20, 50, 100, 200]
    Evolution_Strategies_config, population_sizes, generation_checkpoints = TSP_config()
    for file_idx, file_path in enumerate(All_TSP_File_Path_List):
        file_name = file_path.split("/")[-1]
        TSP_city, TSP_locations = CONFIG.Load_TSP_File(file_path)
        TSP_City_distance_matrix = TSP.TSP_Distance_Cost(TSP_locations)
        for config_idx, config in enumerate(Evolution_Strategies_config):
            for pop_size in population_sizes:
                Parent_Population = TSP.Initial_different_Parent(TSP_city, population_sizes=pop_size)
                Generation_Iteration, Generation_Iteration_Count = generation_checkpoints[0], 0
                # ----------------------------------------
                Best_Solution, Best_Fitness, fitness_history = None, float('inf'), []
                crossover_rate, mutation_rate = config['crossover_rate'], config['mutation_rate']
                # ----------------------------------------
                while Generation_Iteration_Count < Generation_Iteration:
                    parent_pool = config['selection'](TSP_City_distance_matrix, Parent_Population)
                    # ====================
                    offspring_population = []
                    for i in range(0, len(parent_pool) - 1, 2):
                        parent1 = parent_pool[i]
                        parent2 = parent_pool[i + 1] if i + 1 < len(parent_pool) else parent_pool[0]
                        if random.random() < crossover_rate:
                            try:
                                child1, child2 = config['crossover'](parent1, parent2)
                                offspring_population.extend([child1, child2])
                            except:
                                offspring_population.extend([parent1.copy(), parent2.copy()])
                        else:
                            offspring_population.extend([parent1.copy(), parent2.copy()])
                    # ====================
                    if CONFIG.Data_Type == np.ndarray:
                        offspring_population = np.array(offspring_population)
                    for i in range(len(offspring_population)):
                        if random.random() < mutation_rate:
                            offspring_population[i] = config['mutation'](offspring_population[i])
                    if CONFIG.Data_Type == np.ndarray:
                        offspring_population = np.array(offspring_population[:pop_size])
                    if "Elitist" in config['selection'].__name__:
                        Parent_Population = Selection.Elitist_Selection(TSP_City_distance_matrix, Parent_Population, offspring_population)
                    else:
                        Parent_Population = offspring_population
                    current_fitness = Selection._Population_Group_Fitness(Parent_Population, TSP_City_distance_matrix)
                    min_fitness_idx = np.argmin(current_fitness) if CONFIG.Data_Type == np.ndarray else min(enumerate(current_fitness), key=lambda x: x[1])[0]
                    if current_fitness[min_fitness_idx] < Best_Fitness:
                        Best_Fitness = current_fitness[min_fitness_idx]
                        Best_Solution = Parent_Population[min_fitness_idx].copy()
                    fitness_history.append(Best_Fitness)
                    Generation_Iteration_Count += 1
                    if Generation_Iteration_Count % 100 == 0:
                        print(f"Generation {Generation_Iteration_Count}: Best Fitness = {Best_Fitness:.2f}")
                print(f"\n=== 最終結果 ===")
                print(f"最佳路徑長度: {Best_Fitness:.2f}")
        break
Process_Benchmark_ALL_FILE(CONFIG.TSP_File_Path_List)
