import os
import random
import shutil

# Global variables
popsize = None
num_requests = None
num_edge_servers = None
num_cloud_servers = None
crossover_rate = None
mutation_rate = None
chromosome_length = None
max_num_generations = None
requests = []
edge_servers = []
cloud_servers = []

def cleanup():
    # Remove the Generations folder if it exists
    if os.path.exists("Generations"):
        shutil.rmtree("Generations")
    
    # Remove parameters.txt if it exists
    if os.path.exists("parameters.txt"):
        os.remove("parameters.txt")
    
    # Remove final.txt if it exists
    if os.path.exists("final.txt"):
        os.remove("final.txt")

def step_0_generate_parameters():
    """Generates and saves initial parameters to parameters.txt based on specified constraints."""
    
    global popsize, num_requests, num_edge_servers, num_cloud_servers
    global crossover_rate, mutation_rate, chromosome_length, max_num_generations
    global requests, edge_servers, cloud_servers
    
    # Define parameter values
    popsize = random.randint(10, 30)  # Small population size for smooth and quick execution
    num_requests = 10  # Fixed low number of requests
    num_edge_servers = random.randint(3, 7)  # Number of edge servers
    num_cloud_servers = random.randint(3, 7)  # Number of cloud servers
    crossover_rate = 0.5  # Fixed crossover rate
    mutation_rate = 0.1   # Fixed mutation rate
    chromosome_length = num_requests  # Each chromosome represents an offloading solution for each request
    max_num_generations = 100  # Maximum number of generations for the algorithm

    # Generate Edge and Cloud Servers with processing speeds
    edge_servers = []
    for i in range(1, num_edge_servers + 1):
        speed = random.randint(100, 500)  # Edge servers have lower processing speeds
        edge_servers.append({"id": i, "processing_speed": speed, "area_id": i})

    cloud_servers = []
    for i in range(1, num_cloud_servers + 1):
        speed = random.randint(600, 1500)  # Cloud servers have higher processing speeds
        cloud_servers.append({"id": i, "processing_speed": speed})

    # Generate Requests with an instruction length and assign area_id
    requests = []
    for i in range(num_requests):
        area_id = random.randint(1, num_edge_servers)  # Request area_id within number of RSUs
        instruction_length = random.randint(500, 5000)  # Request instruction length in MI
        max_processing_time = random.randint(1, 10)  # Max processing time in seconds (assumed 1/10 of instruction length)

        requests.append({
            "id": i + 1,
            "area_id": area_id,
            "instruction_length": instruction_length,
            "max_processing_time": max_processing_time
        })

    # Save all parameters to the parameters.txt file
    with open("parameters.txt", "w") as file:
        file.write(f"popsize: {popsize}\n")
        file.write(f"num_requests: {num_requests}\n")
        file.write(f"num_edge_servers: {num_edge_servers}\n")
        file.write(f"num_cloud_servers: {num_cloud_servers}\n")
        file.write(f"crossover_rate: {crossover_rate}\n")
        file.write(f"mutation_rate: {mutation_rate}\n")
        file.write(f"chromosome_length: {chromosome_length}\n")
        file.write(f"max_num_generations: {max_num_generations}\n\n")
        
        file.write("Edge Servers:\n")
        for es in edge_servers:
            file.write(f"  Edge Server {es['id']}: Processing Speed = {es['processing_speed']} MIPS, Area ID = {es['area_id']}\n")
        
        file.write("\nCloud Servers:\n")
        for cs in cloud_servers:
            file.write(f"  Cloud Server {cs['id']}: Processing Speed = {cs['processing_speed']} MIPS\n")
        
        file.write("\nRequests:\n")
        for req in requests:
            file.write(f"  Request {req['id']}: Instruction Length = {req['instruction_length']} MI, "
                       f"Max Processing Time = {req['max_processing_time']} seconds, Area ID = {req['area_id']}\n")
    
    print("Parameters generated and saved to parameters.txt")

def step_1_initialization(generation_folder):
    """Initializes a subset of offloading solutions and saves to step1.txt in the specified generation folder."""    
    offloading_solutions = []
    for l in range(popsize):  # For each solution in the population
        solution = []
        for req in requests:  # For each request in this solution
            # Decide between edge server and cloud server
            server_type = random.choice(["edge", "cloud"])
            if server_type == "edge":
                # Get edge servers in the area specified by the request
                area_id = req["area_id"]
                available_edge_servers = [es for es in edge_servers if es["area_id"] == area_id]
                
                # Choose an edge server from the available ones
                if available_edge_servers:
                    server = random.choice(available_edge_servers)
                    server_info = f"Edge Server {server['id']}"
                else:
                    # If no edge server is available, fallback to cloud server
                    server = random.choice(cloud_servers)
                    server_info = f"Cloud Server {server['id']}"
            else:
                server = random.choice(cloud_servers)
                server_info = f"Cloud Server {server['id']}"
            
            solution.append((req["id"], server_info))
        
        offloading_solutions.append(solution)

    # Save offloading solutions to step1.txt within the specified generation folder
    with open(os.path.join(generation_folder, "step1.txt"), "w") as file:
        file.write("Initialization of Offloading Solutions:\n")
        for idx, solution in enumerate(offloading_solutions, 1):
            file.write(f"Solution {idx}:\n")
            for req_id, server_info in solution:
                file.write(f"  Request {req_id} -> {server_info}\n")
            file.write("\n")
    
    print(f"Step 1: Initialization complete. Saved to {os.path.join(generation_folder, 'step1.txt')}")
    
    return offloading_solutions  # Return the initialized solutions for evaluation

def evaluate_solutions(offloading_solutions):
    """Evaluates all solutions and calculates their fitness scores."""
    violations_list = []

    # Calculate violations for each solution
    for solution in offloading_solutions:
        total_violation = 0
        
        for req_id, server_info in solution:
            # Get the request details
            request = next(req for req in requests if req['id'] == int(req_id))
            
            # Determine the server type and processing speed
            if "Edge Server" in server_info:
                server = next(es for es in edge_servers if es['id'] == int(server_info.split()[-1]))
                processing_speed = server['processing_speed']
            else:
                server = next(cs for cs in cloud_servers if cs['id'] == int(server_info.split()[-1]))
                processing_speed = server['processing_speed']
            
            # Calculate processing time
            processing_time = request['instruction_length'] / processing_speed
            
            # Calculate processing time violation
            violation = max(0, processing_time - request['max_processing_time'])
            total_violation += violation
        
        violations_list.append(total_violation)

    # Find the maximum violation among all solutions
    max_violation = max(violations_list) if violations_list else 0

    # Calculate fitness scores for each solution and return them
    fitness_scores = []
    for total_violation in violations_list:
        if max_violation > 0:
            fitness_score = 1 / (total_violation / max_violation + 1)
        else:
            fitness_score = 1.0  # No violations
        fitness_scores.append(fitness_score)

    return fitness_scores  # Return fitness scores for selection

def step_2_evaluate_solutions(offloading_solutions, gen_folder):
    """Evaluates all solutions and saves the results to step2.txt."""
    fitness_scores = evaluate_solutions(offloading_solutions)

    # Save fitness scores to step2.txt
    with open(os.path.join(gen_folder, "step2.txt"), "w") as file:
        file.write("Fitness Scores:\n")
        for idx, score in enumerate(fitness_scores, 1):
            file.write(f"Solution {idx}: Fitness Score = {score}\n")

    print(f"Step 2: Evaluation complete. Fitness scores saved to {os.path.join(gen_folder, 'step2.txt')}.")
    return fitness_scores  # Return fitness scores

def step_3_selection(offloading_solutions, fitness_scores, gen_folder):
    """Selects fittest solutions using Roulette Wheel Selection (RWS) method."""
    total_fitness = sum(fitness_scores)
    
    # Calculate fitness probabilities and cumulative probabilities
    fitness_probabilities = [fitness / total_fitness for fitness in fitness_scores]
    cumulative_probabilities = [sum(fitness_probabilities[:i + 1]) for i in range(len(fitness_probabilities))]
    
    # Generate random numbers for selection
    random_numbers = [random.random() for _ in range(len(offloading_solutions))]
    
    # Select solutions based on random numbers
    selected_solutions = []
    for rand in random_numbers:
        for idx, cumulative_probability in enumerate(cumulative_probabilities):
            if rand <= cumulative_probability:
                selected_solutions.append((idx + 1, offloading_solutions[idx]))
                break

    # Save selection results to step3.txt within the specified generation folder
    with open(os.path.join(gen_folder, "step3.txt"), "w") as file:
        file.write("Selection Results:\n")
        file.write("Solution  Fitness Probability  Cumulative Probability  Random Number  Selected Solution\n")
        for i in range(len(offloading_solutions)):
            file.write(f"{i + 1:<9}\t{fitness_probabilities[i]:<20.4f}\t{cumulative_probabilities[i]:<25.4f}"
                       f"{random_numbers[i]:<15.4f}\t{selected_solutions[i][0]}\n")
    
    print(f"Step 3: Selection complete. Results saved to {os.path.join(gen_folder, 'step3.txt')}.")
    return [solution[1] for solution in selected_solutions]  # Return only the selected solutions


def step_4_crossover(selected_solutions, gen_folder):
    """Performs single-point crossover on selected solutions and saves the results to step4.txt in the specified generation folder."""
    offspring_solutions = selected_solutions.copy()  # Copy of selected solutions for generating offspring
    
    # Step 1: Select solutions for crossover
    crossover_candidates = []
    random_numbers = [random.random() for _ in range(len(selected_solutions))]
    for i, rand in enumerate(random_numbers):
        if rand < crossover_rate:
            crossover_candidates.append(i)  # Select solution based on random number and crossover rate
    
    # Step 2: Randomly pair solutions for crossover
    random.shuffle(crossover_candidates)
    pairs = [(crossover_candidates[i], crossover_candidates[i + 1])
             for i in range(0, len(crossover_candidates) - 1, 2)]
    
    # Step 3: Perform single-point crossover for each pair
    for idx1, idx2 in pairs:
        parent1 = offspring_solutions[idx1]
        parent2 = offspring_solutions[idx2]
        
        # Generate a random cutoff point for single-point crossover
        cutoff = random.randint(1, chromosome_length - 1)
        
        # Create offspring by swapping allocations after the cutoff point
        offspring1 = parent1[:cutoff] + parent2[cutoff:]
        offspring2 = parent2[:cutoff] + parent1[cutoff:]
        
        # Step 4: Calculate fitness scores for parents and offspring
        parent1_violation = calculate_total_violation(parent1)
        parent2_violation = calculate_total_violation(parent2)
        offspring1_violation = calculate_total_violation(offspring1)
        offspring2_violation = calculate_total_violation(offspring2)
        
        # Maximum violation among all 4 solutions for relative fitness calculation
        max_violation = max(parent1_violation, parent2_violation, offspring1_violation, offspring2_violation)
        
        # Fitness scores with conditional check for max violation
        parent1_fitness = 1 if max_violation == 0 else 1 / (parent1_violation / max_violation + 1)
        parent2_fitness = 1 if max_violation == 0 else 1 / (parent2_violation / max_violation + 1)
        offspring1_fitness = 1 if max_violation == 0 else 1 / (offspring1_violation / max_violation + 1)
        offspring2_fitness = 1 if max_violation == 0 else 1 / (offspring2_violation / max_violation + 1)
        
        # Step 5: Replace parents with the two fittest solutions from parents and offspring
        fittest_two = sorted(
            [(parent1, parent1_fitness), (parent2, parent2_fitness), 
             (offspring1, offspring1_fitness), (offspring2, offspring2_fitness)],
            key=lambda x: x[1], reverse=True
        )[:2]
        
        offspring_solutions[idx1], offspring_solutions[idx2] = fittest_two[0][0], fittest_two[1][0]
    
    # Save results to step4.txt in the specified generation folder
    with open(os.path.join(gen_folder, "step4.txt"), "w") as file:
        file.write("Offloading Solutions after Crossover:\n")
        for idx, solution in enumerate(offspring_solutions, 1):
            file.write(f"Solution {idx}:\n")
            for req_id, server_info in solution:
                # Update to format "Request i -> Server"
                file.write(f" Request {req_id} -> {server_info}\n")
            file.write("\n")
    
    print(f"Step 4: Crossover complete. Results saved to {os.path.join(gen_folder, 'step4.txt')}.")
    return offspring_solutions

def calculate_total_violation(solution):
    """Calculates the total violation for a given solution."""
    total_violation = 0
    for req_id, server_info in solution:
        # Use req_id directly since it's already an integer
        request = requests[int(req_id) - 1]  # Access the request directly using the integer
        
        if "Edge Server" in server_info:
            server = next(es for es in edge_servers if es['id'] == int(server_info.split()[-1]))
        else:
            server = next(cs for cs in cloud_servers if cs['id'] == int(server_info.split()[-1]))
        
        processing_time = request['instruction_length'] / server['processing_speed']
        violation = max(0, processing_time - request['max_processing_time'])
        total_violation += violation
    return total_violation

def step_5_mutation(offspring_solutions, gen_folder):
    """Performs mutation on the offloading solutions and saves the results to step5.txt."""
    mutated_solutions = [solution.copy() for solution in offspring_solutions]  # Create a copy for mutation

    # Calculate the total number of allocations and mutations
    nallocations = chromosome_length * popsize
    nmut = int(nallocations * mutation_rate)  # Calculate the number of mutations

    for _ in range(nmut):
        rnd = random.randint(1, nallocations)  # Generate a random number to select a request for mutation
        rem = rnd % chromosome_length  # Determine which request to mutate

        # Determine which solution the random number corresponds to
        solution_index = (rnd - rem) // chromosome_length

        # Check that indices are within range
        if solution_index >= len(mutated_solutions) or rem >= len(mutated_solutions[solution_index]):
            continue  # Skip this iteration if indices are out of bounds

        # Identify the current request to reallocate
        current_request_id = rem + 1
        current_server_info = mutated_solutions[solution_index][rem][1]  # Access the server info correctly
        current_server_type = "edge" if "Edge Server" in current_server_info else "cloud"

        if current_server_type == "edge":
            # If the current server is an edge server, choose a different edge server or a cloud server
            new_server = random.choice(cloud_servers)  # Choose a cloud server
            mutated_solutions[solution_index][rem] = (f"{current_request_id}", f"Cloud Server {new_server['id']}")
        else:
            # Retrieve the area ID for the current request using the current_request_id
            current_request_area_id = None
            for request in requests:  # Assuming 'requests' is a list or dictionary containing your requests
                if request['id'] == current_request_id:  # Check if the current request matches
                    current_request_area_id = request['area_id']  # Get the area ID
                    break  # Exit the loop once the request is found

            # Proceed to choose an edge server based on the area ID
            if len(edge_servers) > 0 and random.random() < 0.5:  # Randomly choose to switch to an edge server
                # Filter edge servers to find those in the same area as the current request
                valid_edge_servers = [server for server in edge_servers if server['area_id'] == current_request_area_id]
                
                if valid_edge_servers:  # Check if there are any valid edge servers
                    new_server = random.choice(valid_edge_servers)  # Randomly select a valid edge server
                    mutated_solutions[solution_index][rem] = (f"{current_request_id}", f"Edge Server {new_server['id']}")
            else:
                # Otherwise, choose a different cloud server
                available_cloud_servers = [cs for cs in cloud_servers if cs['id'] != int(current_server_info.split()[-1])]
                if available_cloud_servers:
                    new_server = random.choice(available_cloud_servers)
                    mutated_solutions[solution_index][rem] = (f"{current_request_id}", f"Cloud Server {new_server['id']}")

    # Save mutated solutions to step5.txt in the specified generation folder
    with open(os.path.join(gen_folder, "step5.txt"), "w") as file:
        file.write("Mutated Offloading Solutions:\n")
        for idx, solution in enumerate(mutated_solutions, 1):
            file.write(f"Solution {idx}:\n")
            for req_id, server_info in solution:
                file.write(f" Request {req_id} -> {server_info}\n")
            file.write("\n")

    print(f"Step 5: Mutation complete. Results saved to {os.path.join(gen_folder, 'step5.txt')}.")
    return mutated_solutions  # Return the mutated solutions for further processing


def final_evaluation(offloading_solutions):
    """Evaluates all solutions and calculates their fitness scores."""
    violations_list = []
    
    # Calculate violations for each solution
    for solution in offloading_solutions:
        total_violation = 0
        
        for req_id, server_info in solution:
            # Get the request details
            request = next(req for req in requests if req['id'] == int(req_id))
            
            # Determine the server type and processing speed
            if "Edge Server" in server_info:
                server = next(es for es in edge_servers if es['id'] == int(server_info.split()[-1]))
                processing_speed = server['processing_speed']
            else:
                server = next(cs for cs in cloud_servers if cs['id'] == int(server_info.split()[-1]))
                processing_speed = server['processing_speed']
            
            # Calculate processing time
            processing_time = request['instruction_length'] / processing_speed
            
            # Calculate processing time violation
            violation = max(0, processing_time - request['max_processing_time'])
            total_violation += violation
        
        violations_list.append(total_violation)

    # Find the maximum violation among all solutions
    max_violation = max(violations_list) if violations_list else 0

    # Calculate fitness scores for each solution and return them as tuples
    fitness_scores = []
    for i, total_violation in enumerate(violations_list):
        if max_violation > 0:
            fitness_score = 1 / (total_violation / max_violation + 1)
        else:
            fitness_score = 1.0  # No violations
        fitness_scores.append((offloading_solutions[i], fitness_score))  # Append tuple (solution, fitness_score)

    return fitness_scores  # Return tuples of solutions and their fitness scores




def main():

    cleanup()

    print("\n")
    # Step 0: Generate parameters
    step_0_generate_parameters()
    print("\n")
    
    # Create main "Generations" folder and first generation folder "gen1"
    main_folder = "Generations"
    os.makedirs(main_folder, exist_ok=True)
    gen1_folder = os.path.join(main_folder, "gen1")
    os.makedirs(gen1_folder, exist_ok=True)

    print("---- Generation 1 ----")

    # Step 1: Initialization for generation 1
    offloading_solutions_gen1 = step_1_initialization(gen1_folder)

    # Step 2: Evaluate the solutions for generation 1
    fitness_scores_gen1 = step_2_evaluate_solutions(offloading_solutions_gen1, gen1_folder)

    # Perform selection for generation 1
    selected_solutions_gen1 = step_3_selection(offloading_solutions_gen1, fitness_scores_gen1, gen1_folder)

    # Save crossover results for generation 1
    offspring_solutions_gen1 = step_4_crossover(selected_solutions_gen1, gen1_folder)

    # Perform mutation on the offspring solutions for generation 1
    mutated_solutions_gen1 = step_5_mutation(offspring_solutions_gen1, gen1_folder)

    # Create folders for generations 2 to 10 and perform evaluation
    mutated_solutions = mutated_solutions_gen1  # Initialize for the loop
    for gen in range(2, max_num_generations + 1):
        gen_folder = os.path.join(main_folder, f"gen{gen}")
        os.makedirs(gen_folder, exist_ok=True)

        print(f"---- Generation {gen} ----")  # Print the generation header

        # Use the mutated solutions from the previous generation
        offloading_solutions = mutated_solutions

        # Evaluate the solutions for the current generation
        fitness_scores = step_2_evaluate_solutions(offloading_solutions, gen_folder)

        # Perform selection for the current generation
        selected_solutions = step_3_selection(offloading_solutions, fitness_scores, gen_folder)

        # Step 4: Perform crossover on selected solutions and save results
        offspring_solutions = step_4_crossover(selected_solutions, gen_folder)

        # In the loop for subsequent generations, perform mutation on the offspring solutions
        mutated_solutions = step_5_mutation(offspring_solutions, gen_folder)

    final_fitness_scores=final_evaluation(mutated_solutions)

    # Find the solution with the maximum fitness score
    best_solution, best_score = max(final_fitness_scores, key=lambda x: x[1])

    # Write the best solution and its score to final.txt
    with open("final.txt", 'w') as final_file:
        final_file.write(f"Best Solution (Score: {best_score}):\n")
        
        # Formatting the solution for output
        for req_id, server_info in best_solution:
            final_file.write(f"Request {req_id} -> {server_info}\n")

    print(f"\nBest Solution saved to final.txt with a score of {best_score}.")


if __name__ == "__main__":
    main()

