# Selection file worked on by Ian
import numpy as np


# Selection methods.
class Selection:
    @classmethod
    def Elitist_Selection(
        cls, tsp_instance, parent_group, children_group=None, elitist_children=None
    ):
        if children_group is None:
            return parent_group

        elitist_children = (
            len(parent_group) if elitist_children is None else elitist_children
        )
        combined_group = np.concatenate((parent_group, children_group))

        # Calculate fitness scores directly
        n = len(combined_group)
        fitness_scores = np.empty(n, dtype=np.float64)

        for i in range(n):
            fitness_scores[i] = tsp_instance.tour_length(combined_group[i])
        best_indices = np.argsort(fitness_scores)[:elitist_children]

        return best_indices

    @classmethod
    def Tournament_Selection(cls, fitness_values, num_parents, tournament_size):
        n = len(fitness_values)
        selected_idxs = np.empty(num_parents, dtype=np.int32)

        for i in range(num_parents):
            # Select tournament participants
            t_indices = np.random.choice(n, tournament_size, replace=False)
            t_fitness = fitness_values[t_indices]

            # Find winner (best fitness - lowest distance)
            winner_idx = t_indices[np.argmin(t_fitness)]
            selected_idxs[i] = winner_idx

        return selected_idxs

    @classmethod
    def Fitness_Proportional_Selection(cls, tsp_instance, parent_group):
        # Calculate fitness scores directly
        n = len(parent_group)
        fitness_scores = np.empty(n, dtype=np.float64)

        for i in range(n):
            fitness_scores[i] = tsp_instance.tour_length(parent_group[i])

        # Convert distances to probabilities (lower distance = higher probability).
        inverse_fitness = 1.0 / fitness_scores
        probabilities = inverse_fitness / np.sum(inverse_fitness)

        # Select parents based on probabilities
        n_parents = len(parent_group)
        selected_indices = np.random.choice(
            n_parents, size=n_parents, replace=True, p=probabilities
        )

        return selected_indices
