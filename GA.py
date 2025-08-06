#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random


# In[3]:


# Parameters
POPULATION_SIZE = 50
MATRIX_SHAPE = (3, 3)
GENERATIONS = 100
MUTATION_RATE = 0.1

# Target matrix (the desired result)
target_matrix = np.array([
    [5, 2, 3],
    [1, 7, 4],
    [6, 9, 8]
])

# Generate an initial matrix with some error (random noise)
def generate_noisy_input(target):
    noise = np.random.randint(-3, 4, size=target.shape)
    return target + noise

# Initialize a random matrix (individual)
def generate_random_matrix(shape):
    return np.random.randint(0, 10, size=shape)

# Fitness function: lower error means better fitness
def calculate_fitness(matrix):
    return -np.mean((matrix - target_matrix) ** 2)

# Crossover: combine two matrices
def crossover(parent1, parent2):
    child = np.copy(parent1)
    rows, cols = parent1.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < 0.5:
                child[i][j] = parent2[i][j]
    return child

# Mutation: randomly modify values in the matrix
def mutate(matrix):
    mutated = np.copy(matrix)
    for i in range(mutated.shape[0]):
        for j in range(mutated.shape[1]):
            if random.random() < MUTATION_RATE:
                mutated[i][j] = random.randint(0, 9)
    return mutated

# Generate initial population
population = [generate_random_matrix(MATRIX_SHAPE) for _ in range(POPULATION_SIZE)]

# Evolution loop
for generation in range(GENERATIONS):
    # Evaluate fitness
    fitness_scores = [calculate_fitness(individual) for individual in population]
    
    # Select the top individuals (elitism)
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    population = sorted_population[:POPULATION_SIZE // 2]

    # Generate new population via crossover and mutation
    new_population = population.copy()
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = random.sample(population, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    
    population = new_population

    # Best solution so far
    best_matrix = population[0]
    best_fitness = calculate_fitness(best_matrix)

    print(f"Generation {generation+1} | Best Fitness: {best_fitness:.4f}")
    print(best_matrix)
    print()

# Final result
print("Target Matrix:")
print(target_matrix)
print("Evolved Matrix:")
print(best_matrix)


# In[ ]:


# What this code does:
# Creates a target matrix.
# Evolves random matrices using a genetic algorithm.
# Selects the best candidates based on mean squared error.
# Uses crossover and mutation to generate new solutions.
# After several generations, the evolved matrix approximates the target matrix.**/


# In[ ]:




