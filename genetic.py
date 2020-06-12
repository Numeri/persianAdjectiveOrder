#!/bin/python3 -i

import copy
import queue
import random

class Organism:
    def __init__(self, genes, gene_range):
        self.genes = genes
        self.gene_range = gene_range

    def __repr__(self):
        return ''.join(str(i) for i in self.genes)

    def produce_children(self, mate, mutation_chance):
        crossover = random.randint(0, len(self.genes)-1)
        genes_a = [(self.genes[i] if i <= crossover else mate.genes[i]) if random.random() > mutation_chance else random.randint(0, self.gene_range) for i in range(len(self.genes))] 
        genes_b = [(mate.genes[i] if i <= crossover else self.genes[i]) if random.random() > mutation_chance else random.randint(0, self.gene_range) for i in range(len(self.genes))] 
        child_a = Organism(genes_a, self.gene_range)
        child_b = Organism(genes_b, self.gene_range)

        return child_a, child_b

class Population:
    def __init__(self, num_organisms, gene_length, gene_range, cost_function, parents_selected=0.4, mutation_chance=0.03):
        self.num_organisms = num_organisms
        self.gene_length = gene_length
        self.gene_range = gene_range
        self.cost_function = cost_function
        self.parents_selected = parents_selected
        self.mutation_chance = mutation_chance
        
        self.organisms = [Organism([random.randint(0,gene_range) for _ in range(self.gene_length)], self.gene_range) for _ in range(self.num_organisms)]

    def simulate_generation(self):
        best_organisms = queue.PriorityQueue()

        for tie_breaker,o in enumerate(self.organisms):
            best_organisms.put((self.cost_function(o), tie_breaker, o))

        num_parents = int(len(self.organisms) * self.parents_selected)
        parents = [best_organisms.get() for _ in range(num_parents)]

        new_organisms = []
        for _ in range(self.num_organisms//2):
            a, b = random.sample(parents, 2)
            new_organisms.extend(a[2].produce_children(b[2], self.mutation_chance))

        self.organisms = new_organisms

    def simulate_n_generations(self, n):
        for i in range(n):
            print(f'{self.average_health()}:\t{self.take_n(3)}')
            self.simulate_generation()

        print(f'{self.average_health()}:\t{self.take_n(3)}')

    def average_health(self):
        return sum(map(self.cost_function, self.organisms)) / len(self.organisms)

    def take_n(self, n):
        if n > self.num_organisms:
            raise Exception("Trying to take more organisms than exist")

        best_organisms = queue.PriorityQueue()

        for tie_breaker,o in enumerate(self.organisms):
            best_organisms.put((self.cost_function(o), tie_breaker, o))

        return [best_organisms.get()[2] for _ in range(n)]

