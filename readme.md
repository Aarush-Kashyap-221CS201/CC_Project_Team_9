# Energy-SLA-aware Genetic Algorithm for Edge-Cloud Integrated Computation Offloading in Vehicular Networks

## Team Members
- **Aarush Kashyap** (221CS201)
- **Sandeep Kumar** (221CS149)
- **Shashank Prabhakar** (221CS246)
- **Rahul Bhimakari** (221CS143)

## Project Aim
The aim of this project is to design and implement an **energy-efficient**, **Service Level Agreement (SLA)-aware** genetic algorithm for optimizing computation offloading in vehicular networks. By effectively balancing the load between edge and cloud servers, we strive to enhance overall system performance while minimizing energy consumption and ensuring compliance with SLAs.

## Methodology
Our methodology includes the following steps:
1. **Parameter Generation:** Generate necessary parameters for simulation.
2. **Initialization:** Set up initial offloading solutions.
3. **Evaluation:** Assess the fitness of the solutions based on processing time.
4. **Selection:** Choose the best-performing solutions for the next generation.
5. **Crossover:** Create new solutions by combining features from selected solutions.
6. **Mutation:** Introduce variability into the solutions to maintain diversity in the population.
7. **Termination:** Repeat the process for a predefined number of generations and compile results.

## Implementation
The project is implemented in Python. The main file is named **proj_code_final.py**. To run the file, follow these instructions:

### Instructions to Run
1. Ensure you have Python installed on your system.
2. Clone the repository to your local machine.
3. Navigate to the project directory in your terminal.
4. Run the file titled **proj_code_final.py**.

### Output
The program will generate the following outputs:

1. A **Generations** folder containing results for each generation.
2. A **parameters.txt** file with the simulation parameters.
3. A **final.txt** file with the best offloading solution based on fitness scores.

Each generation's folder will include:

1. Solutions evaluated for fitness.
2. Selected solutions for crossover.
3. Solutions after crossover.
4. Mutated solutions from the previous generation.

**Note:** Only the first generation will have the initial solutions, which will be randomly generated. After that, every generation uses the solutions from the previous generation

**Note:** The offloading decisions are based primarily on processing time to ensure optimal performance.

## Conclusion
This project demonstrates the potential of genetic algorithms in optimizing computation offloading in vehicular networks. By leveraging edge and cloud computing resources efficiently, we can enhance system performance and meet energy constraints. The findings contribute to ongoing research in cloud computing and intelligent vehicular networks, paving the way for smarter and more sustainable solutions in the future. The implementation shows how integrating genetic algorithms can effectively address the challenges of computation offloading while maintaining compliance with SLAs and optimizing energy usage.

The results indicate a significant improvement in resource allocation efficiency, as seen in the final output file, where the best solution yields a fitness score that reflects optimal processing time adherence and effective resource utilization.
