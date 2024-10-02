import random
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def create_dataset(file_name):
    """
    Reads the dataset file and extracts the necessary information for the knapsack problem.

    Args:
        file_name (str): The name or path of the file containing the dataset.

    Returns:
        tuple: A tuple containing the following elements:
            - items_data (list): A list of dictionaries, where each dictionary represents an item with its weight and value.
            - max_capacity (int): The maximum capacity of the knapsack.
            - item_count (int): The total number of items in the dataset.
            - optimal_value (int): The optimal or best possible value that can be achieved for the given dataset.
    """
    weights = []
    values = []
    max_capacity = 0   # value of maximume weights
    item_count = 0     # number of the items for each individual
    optimal_value = 0  

    # Read file and extract data file
    full_path = os.path.abspath(__file__) # Get the full path of the script     
    script_directory = os.path.dirname(full_path) # Get the directory of the script
    data_file = os.path.join(script_directory,file_name) # Get the full path of the data file

    with open(data_file,'r') as file: 
        data = file.readlines()      

        for idx, line in enumerate(data): # extract weights and vlues and store it into list
            x = line.split()
            if idx == 0:
                max_capacity = int(x[1])
                item_count = int(x[0])
            else:
                weights.append(int(x[1]))
                values.append(int(x[0]))
        
        # Find the vlaue of optimal_value paramener. depend on value of (max_capacity) 
        if max_capacity == 269: optimal_value = 295
        elif max_capacity == 10000: optimal_value = 9767
        else: optimal_value = 1514
        
        item_dict = {"weights":weights ,"values":values}
    
    return item_dict, max_capacity, item_count, optimal_value

def initial_pop(population_size, num_items, seed_val):
    """
    Generate the initial population for the genetic algorithm.

    Args:
        population_size (int): The desired size of the initial population.
        num_items (int): The number of items in the knapsack problem.

    Returns:
        list: A list of individuals, where each individual is a binary vector
              representing a potential solution to the knapsack problem.
              The length of each individual is equal to `num_items`.
    """
    random.seed(seed_val)
    return [np.random.randint(2, size=num_items) for _ in range(population_size)]

def calculate_fitness(individual, items, max_capicity):
    """
        This function calculates the fitness of an individual. 
        The fitness is the total value of the items included in the knapsack.
    """
    total_weight = sum([items['weights'][i] * individual[i] for i in range(len(items['weights']))])
    total_value = sum([items['values'][i] * individual[i] for i in range(len(items['values']))])
    if total_weight > max_capicity:
        return 0
    else:
        return total_value

def calculate_marginal_probabilities(fitnesses, individuals):
        """
        Calculate the marginal probabilities of each individual based on their fitness values.
        
        Args:
            fitnesses (list): A list of fitness values for each individual in the population.
        
        Returns:
            list: A list of marginal probabilities for each individual in the population.
        """
        total_fitness = sum(fitnesses)
        total_fitness = sum(f for f in fitnesses if f > 0)
        normalized_weights = [(f / total_fitness if f > 0 else 0) for f in fitnesses]
        
        n_bits = len(individuals[0])  # Number of bits per individual
        weighted_probabilities = []

        # Calculate the probability for each bit position
        for bit_position in range(n_bits):
            weighted_sum_for_bit_1 = 0.0
            for i, individual in enumerate(individuals):
                if individual[bit_position] == 1:
                    weighted_sum_for_bit_1 += normalized_weights[i]
        
        weighted_probabilities.append(weighted_sum_for_bit_1)

        return weighted_probabilities


def generate_offspring(offspring_size, n_bits, probability_model):
        """
        Generate offspring for the next generation using the probability model.
        
        Args:
            offspring_size (int): The number of offspring to generate.
            n_bits (int): The number of bits in each individual.
            probability_model (float): The probability of each bit being set to 1.
        """
        offspring = np.zeros((offspring_size, n_bits), dtype=int)
        
        for i in range(offspring_size):
            offspring[i] = np.random.random(n_bits) < probability_model
        return offspring

def plot_curve(best_weights):
    """
    Plot the average fitness of the best solution weights over the generations.
    
    Args:
        best_weights (list): A list of the best solution weights for each generation.
    """
    average_fitness = []

    #for weights in best_weights:
    #    average_fitness.append(sum(weights) / len(best_weights[0]))
    #average_fitness.append(sum(best_weights) / len(best_weights))

    average_fitness = np.mean(best_weights, axis=0)

    generations = list(range(1, len(average_fitness) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, average_fitness, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Generations')
    plt.ylabel('Average Fitness of 5 Best Solution Weights')
    plt.title('Average Fitness Over Generations')
    plt.grid(True)
    plt.show()

def main():
    # parameter setting 
    population_size = 50   # Population size 
    generations = 100   # number of generations to run the Genetic Algorithm
    run_no = 5  # number of runs GA
    # load data 
    dataset_file = ['23_10000','10_269','100_995']  # 23_10000  10_269  100_995

    for file in dataset_file:
        print('--------------------------------------')
        print(f'DATASET NAME: {file}\n')
        knapsack_items, max_capacity, num_items, optimal_value = create_dataset(file)  # Obtain dataset values into parameter
    
        # run GA for 5 times
        best_value = None  # Store summation value of best individual each run 
        best_individuals = None  # Store best individual each run
        best_fitnesses_runs = [] # Store best fitnesses in genereation for each run
        elitism_size = 2  # Number of best individuals to keep for the next generation
        
        seed_value = [20, 40, 60, 80, 100]
        subset_size = population_size // 2  # Select half of the population for the next generation

        for run in range(run_no):
            print(f'Run {run+1} of {run_no}:')

            # Reset best wights (during generation) list for each run
            best_value_generations = []  # Store best wights in each generation for each run
            best_value = None
            best_individuals = None

            # Initialize populations
            populations = initial_pop(population_size, num_items, seed_value[run])

            # Apply Selection process
            for generation in range(generations):
                #print(f'\tGeneration {generation+1} of {generations} . . .', end='\r')

                # Calculate fitness of each individual
                fitnesses = [calculate_fitness(individual, knapsack_items, max_capacity) for individual in populations]

                # apply Elitism
                elitism_individual = np.argsort(fitnesses)[::-1][:elitism_size]

                # Select subset of individual 
                selected_indices = np.argsort(fitnesses)[::-1][:subset_size] 
                selected_fitnesses = [fitnesses[i] for i in selected_indices]
                selected_individuals = [populations[i] for i in selected_indices]

                # calculate probability of each gens
                probabilities = calculate_marginal_probabilities(selected_fitnesses, selected_individuals)
                #probabilities = np.array(probabilities)

                # Sampling new population based on the probability
                #new_population = [np.random.choice(selected_indices, p=probabilities) for _ in range(population_size - elitism_size)]

                # # Sampling new population based on the probability, Generate offspring.
                offspring = generate_offspring((population_size-elitism_size), num_items, probabilities)

                # Combine the new population with the elitism individual
                populations = np.vstack([[populations[i] for i in elitism_individual], offspring])

                # Evaluate the best individual in each generation
                population_fitness = [calculate_fitness(individual, knapsack_items, max_capacity) for individual in populations]

                # Store the best individual in each generation and get individual with the best value
                best_value_generations.append(max(population_fitness))
                best_individual_generations = populations[np.argmax(population_fitness)]

                # Track best value and individual in each generation
                if best_value is None or best_value_generations[generation] > best_value:
                    best_value = best_value_generations[generation]
                    best_individuals = best_individual_generations

                # Stopping criteria
                if best_value == optimal_value:
                    print(f'\tObtain optimal value: {best_value}, at generation: {generation+1}')
                    print(f'\tBest individual: {best_individuals}\n')
                    break

            # Store best value in each run
            best_fitnesses_runs.append(best_value_generations)

            if best_value != optimal_value:
                print(f'\tBest value over {generation} generations: {best_value}')
                print(f'\tBest individual: {best_individuals}\n')

        # Calculate the average best value over all runs
        best_value_each_run = [max(a) for a in best_fitnesses_runs]
        avg_best_value = sum(best_value_each_run) / run_no
        std_best_value = np.std(best_value_each_run)
        print(f'AVERAGE BEST VALUE OVER ({run_no}) RUNS: {avg_best_value}')
        print(f'STANDARD DEVIATION OF BEST VALUE OVER ({run_no}) RUNS: {std_best_value}\n')

        # Plot the average fitness of the best solution weights over the generations
        plot_curve(best_fitnesses_runs)
            

            

                 




if __name__ == "__main__":
    main()