import random
import numpy as np
from tsp import TSP
from individual_population import Individual, Population
import os

class InverOver:
    def __init__(self, tsp_file, population_size=50, generations=20000, p=0.02):
        self.tsp = TSP(tsp_file)
        self.population_size = population_size
        self.generations = generations
        self.p = p
        self.population = Population.random(self.tsp, self.population_size)

    def inver_over(self, individual):
        S = individual.tour.copy()
        c = random.choice(S)

        while True:
            if random.random() <= self.p:
                c_prime = random.choice([city for city in S if city != c])
            else:
                mate = random.choice(self.population.individuals)
                idx = mate.tour.index(c)
                c_prime = mate.tour[(idx + 1) % len(mate.tour)]

            idx_c = S.index(c)
            idx_cp = S.index(c_prime)
            if S[(idx_c + 1) % len(S)] == c_prime or S[(idx_c - 1) % len(S)] == c_prime:
                break

            i, j = sorted([idx_c, idx_cp])
            S[i+1:j+1] = reversed(S[i+1:j+1])
            c = c_prime

        return Individual(self.tsp, S)

    def evolve(self):
        for _ in range(self.generations):
            for i in range(len(self.population)):
                offspring = self.inver_over(self.population[i])
                if offspring.fitness < self.population[i].fitness:
                    self.population.individuals[i] = offspring

    def run(self):
        self.evolve()
        costs = [ind.fitness for ind in self.population.individuals]
        return np.mean(costs), np.std(costs)


def run_inverover_all():
    instances = [
        "eil101", "eil51", "eil76", "kroA100", "kroC100", 
        "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"
    ]

    os.makedirs("../results", exist_ok=True)
    with open("../results/inverover.txt", "w") as f:
        f.write("Instance\tAvgCost\tStdDev\n")
        
        for name in instances:
            print(f"Running {name}...")
            results = []
            for _ in range(30):
                algo = InverOver(f"tsplib/{name}.tsp")
                avg, std = algo.run()
                results.append(avg)
            final_avg = np.mean(results)
            final_std = np.std(results)
            f.write(f"{name}\t{final_avg:.2f}\t{final_std:.2f}\n")


if __name__ == "__main__":
    run_inverover_all()