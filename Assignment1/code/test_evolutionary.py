#!/usr/bin/env python3
"""
Test file for the Evolutionary Algorithm implementation
Tests the EA on different TSP instances and demonstrates usage
"""

import os
import sys
import time
from evolutionary import EvolutionaryAlgorithm


# Basic test for Evolutionary Algorithm
from evolutionary import EvolutionaryAlgorithm

def basic_test():
    print("Basic EA Test")
    print("=" * 30)
    
    # Test on small TSP
    ea = EvolutionaryAlgorithm(
        tsp_file="tsplib/eil51.tsp",
        population_size=20,
        max_generations=10
    )
    
    print("Running EA...")
    best = ea.run()
    print(f"Best tour length: {best.fitness:.2f}")

if __name__ == "__main__":
    basic_test() 