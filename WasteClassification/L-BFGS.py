import numpy as np
import random

#Using Genetic Algorithm
# Robot arm parameters
joint_limits = [(-180, 180), (-90, 90), (-90, 90), (-180, 180), (-90, 90), (-180, 180)]  # Joint angle limits for 6 DOF
arm_lengths = [0.2, 0.5, 0.7, 0.3, 0.6, 0.5]  # Lengths of the robot arm segments

# Inverse Kinematics Solver (simplified version)
def inverse_kinematics(target_position):
    # A simplified inverse kinematics solver (replace with your own IK solver)
    # Assuming a 3DOF arm for simplicity (just an example)
    x, y, z = target_position
    # Solve for joint angles (this would be more complex in a real scenario)
    joint_angles = np.random.uniform(-180, 180, 6)  # Random values for now
    return joint_angles

# Fitness function (objective function)
def fitness_function(individual, trajectory_points):
    total_energy = 0
    total_time = 0
    total_rotation = 0
    
    # Compute the energy, time, and rotation for each segment of the trajectory
    for i in range(len(trajectory_points) - 1):
        start, end = trajectory_points[i], trajectory_points[i + 1]
        total_energy += calculate_energy(start, end)
        total_time += calculate_time(start, end)
        total_rotation += calculate_rotation(start, end)
    
    # Combined objective function with weighting coefficients
    return total_time + 5 * total_energy + 10 * total_rotation

# Calculate energy consumed during movement
def calculate_energy(start, end):
    # Energy calculation based on joint rotation and robotic arm parameters
    energy = np.sum(np.abs(np.array(start) - np.array(end)))  # Simplified energy calculation
    return energy

# Calculate time required for movement between two points
def calculate_time(start, end):
    # Time calculation based on joint velocities (simplified)
    time = np.sum(np.abs(np.array(start) - np.array(end)))  # Placeholder, use actual joint velocity
    return time

# Calculate rotation angles between two points
def calculate_rotation(start, end):
    # Calculate the difference in joint angles
    rotation = np.sum(np.abs(np.array(start) - np.array(end)))
    return rotation

# Genetic Algorithm Functions
def initialize_population(population_size, num_joints, num_points):
    # Initialize population with random joint angles within their limits
    low_bounds = np.tile(np.array([limit[0] for limit in joint_limits]), num_points)
    high_bounds = np.tile(np.array([limit[1] for limit in joint_limits]), num_points)

    # Generate a population of random joint angles within the bounds
    population = np.random.uniform(low=low_bounds, high=high_bounds, size=(population_size, num_joints * num_points))
    
    return population


def select_parents(population, fitness_values, num_parents):
    # Select parents based on fitness (roulette wheel or tournament selection)
    selected_parents = []
    fitness_sum = np.sum(fitness_values)
    probabilities = fitness_values / fitness_sum
    cumulative_probabilities = np.cumsum(probabilities)
    
    for _ in range(num_parents):
        rand_val = random.random()
        selected_parents.append(population[np.searchsorted(cumulative_probabilities, rand_val)])
    
    return np.array(selected_parents)

def crossover_and_mutate(parents, crossover_rate=0.7, mutation_rate=0.1):
    # Crossover and mutation operations to generate offspring
    offspring = []
    num_parents = len(parents)
    num_children = num_parents // 2
    
    # Crossover
    for i in range(num_children):
        parent1, parent2 = parents[2*i], parents[2*i+1]
        crossover_point = random.randint(1, len(parent1)-1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        
        offspring.append(child1)
        offspring.append(child2)
    
    # Mutation
    for child in offspring:
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(child)-1)
            child[mutation_point] += np.random.uniform(-10, 10)  # Small mutation
    
    return np.array(offspring)

def genetic_algorithm(population_size, generations, trajectory_points):
    num_joints = len(joint_limits)
    num_points = len(trajectory_points)
    
    # Initialize population
    population = initialize_population(population_size, num_joints, num_points)
    
    for generation in range(generations):
        # Evaluate fitness of the population
        fitness_values = np.array([fitness_function(individual, trajectory_points) for individual in population])
        
        # Select parents for the next generation
        num_parents = population_size // 2
        parents = select_parents(population, fitness_values, num_parents)
        
        # Create offspring through crossover and mutation
        offspring = crossover_and_mutate(parents)
        
        # Create the next generation (parents + offspring)
        population[:num_parents] = parents
        population[num_parents:] = offspring
    
    # Return the best individual (solution) from the final population
    best_individual = population[np.argmin(fitness_values)]
    return best_individual

# Example trajectory points (in 3D space)
trajectory_points = np.array([
    [2.25, 1.1, 0.25],
    [0.9, 1.5, 0.25],
    [-0.85, 1.14, 2.22],
    [-1.8, 1.25, 1.17],
    [1.8, 1.25, 1.17],
    [-1.25, -1.1, 0.25],
    [-2.25, -1.48, 0.25],
    [0.45, -1.14, 2.22],
    [0.8, -1.25, 2.35],
    [0.8, -1.25, -1.35]
])

# Running the genetic algorithm to optimize the trajectory
population_size = 100
generations = 1000
best_trajectory = genetic_algorithm(population_size, generations, trajectory_points)

print("Optimized trajectory:", best_trajectory)
