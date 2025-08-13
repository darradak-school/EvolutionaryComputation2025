# Selection file worked on by Ian
import os, random,itertools, sys, gc
import numpy as np
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