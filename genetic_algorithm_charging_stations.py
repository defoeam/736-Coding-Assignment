"""
Genetic Algorithm for Electric Vehicle Charging Station Optimization
CSCI 736 Computational Intelligence - Coding Assignment

This module implements a genetic algorithm to optimize the placement of
electric vehicle charging stations in a city, balancing coverage and cost.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import csv


class ChargingStationGA:
    """
    Genetic Algorithm for optimizing electric vehicle charging station locations.
    
    Attributes:
        candidate_locations: List of (x, y) coordinates for potential station sites
        demand_points: List of (x, y) coordinates for demand locations
        demand_weights: Weight/importance of each demand point
        installation_costs: Cost of installing/maintaining station at each location
        coverage_radius: Maximum distance for a station to cover a demand point
        budget: Maximum budget constraint
        population_size: Number of chromosomes in each generation
        num_generations: Maximum number of generations to run
        crossover_rate: Probability of crossover occurring
        mutation_rate: Probability of mutation for each gene
        lambda_weight: Trade-off parameter between coverage and cost
        alpha_penalty: Penalty weight for budget violations
    """
    
    def __init__(
        self,
        candidate_locations: List[Tuple[float, float]],
        demand_points: List[Tuple[float, float]],
        demand_weights: List[float],
        installation_costs: List[float],
        coverage_radius: float = 5.0,
        budget: float = 100000.0,
        population_size: int = 100,
        num_generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.02,
        lambda_weight: float = 0.3,
        alpha_penalty: float = 2.0
    ):
        self.candidate_locations = np.array(candidate_locations)
        self.demand_points = np.array(demand_points)
        self.demand_weights = np.array(demand_weights)
        self.installation_costs = np.array(installation_costs)
        self.coverage_radius = coverage_radius
        self.budget = budget
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.lambda_weight = lambda_weight
        self.alpha_penalty = alpha_penalty
        
        self.num_locations = len(candidate_locations)
        self.num_demand_points = len(demand_points)
        
        # Track best solutions and fitness history
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.avg_fitness_history = []
        self.best_fitness_history = []
        
    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)
    
    def calculate_coverage(self, chromosome: np.ndarray) -> Tuple[float, int]:
        """
        Calculate total weighted coverage for a given chromosome.
        
        Returns:
            Tuple of (weighted_coverage, num_covered_points)
        """
        selected_stations = self.candidate_locations[chromosome == 1]
        
        if len(selected_stations) == 0:
            return 0.0, 0
        
        covered_demand = np.zeros(self.num_demand_points, dtype=bool)
        
        # Check which demand points are covered
        for i, demand_point in enumerate(self.demand_points):
            for station in selected_stations:
                distance = self.calculate_distance(demand_point, station)
                if distance <= self.coverage_radius:
                    covered_demand[i] = True
                    break
        
        # Calculate weighted coverage
        weighted_coverage = np.sum(self.demand_weights[covered_demand])
        num_covered = np.sum(covered_demand)
        
        return weighted_coverage, num_covered
    
    def calculate_cost(self, chromosome: np.ndarray) -> float:
        """Calculate total installation and maintenance cost."""
        return np.sum(self.installation_costs[chromosome == 1])
    
    def fitness_function(self, chromosome: np.ndarray) -> float:
        """
        Calculate penalized fitness for GA evolution.

        Fitness = cov_norm - λ * cost_norm - α * max(0, cost_norm - 1)
        where:
        cov_norm  = (weighted_coverage / total_weight)
        cost_norm = (cost / budget)
        """
        # Coverage and cost
        weighted_coverage, _ = self.calculate_coverage(chromosome)
        cost = self.calculate_cost(chromosome)

        # Normalize coverage
        total_weight = float(np.sum(self.demand_weights))
        cov_norm = (weighted_coverage / total_weight) if total_weight > 0 else 0.0

        # Normalize cost by BUDGET (not max_cost)
        if getattr(self, "budget", None) is None or self.budget <= 0:
            # If no valid budget is set, make any cost extremely unfavorable
            cost_norm = 1e9
        else:
            cost_norm = cost / float(self.budget)

        # Penalty only if over budget
        overage = max(0.0, cost_norm - 1.0)
        penalty = self.alpha_penalty * overage

        # Final penalized fitness
        fitness = cov_norm - self.lambda_weight * cost_norm - penalty

        # Penalize empty solutions (no stations selected) to encourage coverage
        if np.sum(chromosome) == 0:
            fitness -= 0.1  # Small penalty for zero coverage
            
        return float(fitness)
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize population with random binary chromosomes.
        Each gene has 50% probability of being 1 or 0.
        """
        return np.random.randint(0, 2, size=(self.population_size, self.num_locations))
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for entire population."""
        fitness_values = np.array([self.fitness_function(chromosome) 
                                   for chromosome in population])
        return fitness_values
    
    def roulette_wheel_selection(self, population: np.ndarray, 
                                 fitness_values: np.ndarray) -> np.ndarray:
        """
        Perform roulette wheel selection.
        Probability of selection is proportional to fitness.
        """
        # Shift fitness values to be positive for roulette wheel
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            adjusted_fitness = fitness_values - min_fitness + 1e-6
        else:
            adjusted_fitness = fitness_values + 1e-6
        
        # Calculate selection probabilities
        total_fitness = np.sum(adjusted_fitness)
        probabilities = adjusted_fitness / total_fitness
        
        # Select two parents
        selected_indices = np.random.choice(
            self.population_size, 
            size=2, 
            p=probabilities,
            replace=False
        )
        
        return population[selected_indices]
    
    def single_point_crossover(self, parent1: np.ndarray, 
                              parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single-point crossover between two parents.
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Choose random crossover point
        crossover_point = random.randint(1, self.num_locations - 1)
        
        # Create offspring
        offspring1 = np.concatenate([parent1[:crossover_point], 
                                    parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], 
                                    parent1[crossover_point:]])
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform bit-flip mutation on chromosome.
        Each gene has mutation_rate probability of flipping.
        """
        mutated = chromosome.copy()
        for i in range(self.num_locations):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated
    
    def evolve(self, verbose: bool = True) -> Dict:
        """
        Run the genetic algorithm evolution process.
        
        Returns:
            Dictionary containing results and statistics
        """
        # Initialize population
        population = self.initialize_population()
        
        # Track convergence
        no_improvement_count = 0
        
        for generation in range(self.num_generations):
            # Evaluate fitness
            fitness_values = self.evaluate_population(population)
            
            # Track statistics
            avg_fitness = np.mean(fitness_values)
            max_fitness = np.max(fitness_values)
            best_idx = np.argmax(fitness_values)
            
            self.avg_fitness_history.append(avg_fitness)
            self.best_fitness_history.append(max_fitness)
            
            # Update best solution found
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_chromosome = population[best_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose and generation % 20 == 0:
                print(f"Generation {generation}: "
                      f"Avg Fitness = {avg_fitness:.4f}, "
                      f"Best Fitness = {max_fitness:.4f}")
            
            # Check for convergence
            if no_improvement_count > 50:
                if verbose:
                    print(f"Converged at generation {generation}")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism: Keep best individual
            new_population.append(self.best_chromosome.copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parents = self.roulette_wheel_selection(population, fitness_values)
                
                # Crossover
                offspring1, offspring2 = self.single_point_crossover(parents[0], parents[1])
                
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            population = np.array(new_population[:self.population_size])
        
        # Calculate final results
        results = self.get_results()
        
        if verbose:
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            self.print_results(results)
        
        return results
    
    def get_results(self) -> Dict:
        """Get detailed results of the optimization."""
        if self.best_chromosome is None:
            return {}
        
        selected_indices = np.where(self.best_chromosome == 1)[0]
        selected_locations = self.candidate_locations[selected_indices]
        selected_costs = self.installation_costs[selected_indices]
        
        weighted_coverage, num_covered = self.calculate_coverage(self.best_chromosome)
        total_cost = self.calculate_cost(self.best_chromosome)
        
        coverage_percentage = (num_covered / self.num_demand_points) * 100
        
        return {
            'best_fitness': self.best_fitness,
            'selected_indices': selected_indices,
            'selected_locations': selected_locations,
            'num_stations': len(selected_indices),
            'total_cost': total_cost,
            'weighted_coverage': weighted_coverage,
            'num_covered': num_covered,
            'coverage_percentage': coverage_percentage,
            'within_budget': total_cost <= self.budget,
            'budget_utilization': (total_cost / self.budget) * 100,
            'avg_fitness_history': self.avg_fitness_history,
            'best_fitness_history': self.best_fitness_history
        }
    
    def print_results(self, results: Dict):
        """Print formatted results."""
        print(f"Best Fitness: {results['best_fitness']:.4f}")
        print(f"Number of Stations Selected: {results['num_stations']}")
        print(f"Total Cost: ${results['total_cost']:,.2f}")
        print(f"Budget: ${self.budget:,.2f}")
        print(f"Within Budget: {'Yes' if results['within_budget'] else 'No'}")
        print(f"Budget Utilization: {results['budget_utilization']:.2f}%")
        print(f"Weighted Coverage: {results['weighted_coverage']:.2f}")
        print(f"Demand Points Covered: {results['num_covered']} / {self.num_demand_points}")
        print(f"Coverage Percentage: {results['coverage_percentage']:.2f}%")
        print(f"\nSelected Station Locations (indices): {results['selected_indices'].tolist()}")
        print(f"\nSelected Station Coordinates:")
        for i, (idx, loc) in enumerate(zip(results['selected_indices'], 
                                           results['selected_locations'])):
            cost = self.installation_costs[idx]
            print(f"  Station {i+1}: Location {idx} at ({loc[0]:.2f}, {loc[1]:.2f}), "
                  f"Cost: ${cost:,.2f}")
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Create visualization of the optimization results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Convergence graph
        ax1 = axes[0]
        generations = range(len(results['avg_fitness_history']))
        ax1.plot(generations, results['avg_fitness_history'], 
                label='Average Fitness', linewidth=2, alpha=0.7)
        ax1.plot(generations, results['best_fitness_history'], 
                label='Best Fitness', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('GA Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Station locations and coverage
        ax2 = axes[1]
        
        # Plot demand points
        ax2.scatter(self.demand_points[:, 0], self.demand_points[:, 1],
                   c='lightblue', s=self.demand_weights*10, alpha=0.6,
                   label='Demand Points', edgecolors='blue', linewidth=0.5)
        
        # Plot candidate locations (not selected)
        not_selected = np.ones(self.num_locations, dtype=bool)
        not_selected[results['selected_indices']] = False
        if np.any(not_selected):
            ax2.scatter(self.candidate_locations[not_selected, 0],
                       self.candidate_locations[not_selected, 1],
                       c='lightgray', s=100, alpha=0.3,
                       label='Candidate Locations (Not Selected)',
                       marker='s')
        
        # Plot selected stations
        selected_locs = results['selected_locations']
        ax2.scatter(selected_locs[:, 0], selected_locs[:, 1],
                   c='red', s=200, alpha=0.8, marker='*',
                   label='Selected Stations', edgecolors='darkred', linewidth=1.5)
        
        # Draw coverage circles
        for loc in selected_locs:
            circle = plt.Circle(loc, self.coverage_radius, 
                              color='red', fill=False, 
                              linestyle='--', alpha=0.3, linewidth=1)
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X Coordinate', fontsize=12)
        ax2.set_ylabel('Y Coordinate', fontsize=12)
        ax2.set_title('Charging Station Placement', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
    
    def export_results_csv(self, results: Dict, filename: str):
        """Export results to CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Station ID', 'Location Index', 'X Coordinate', 
                           'Y Coordinate', 'Installation Cost'])
            
            for i, (idx, loc) in enumerate(zip(results['selected_indices'], 
                                               results['selected_locations'])):
                cost = self.installation_costs[idx]
                writer.writerow([i+1, idx, loc[0], loc[1], cost])
        
        print(f"Results exported to {filename}")


def generate_test_dataset(num_candidates: int = 10, 
                         num_demand: int = 50,
                         grid_size: Tuple[float, float] = (50, 50),
                         seed: int = 42) -> Dict:
    """
    Generate a simplified, hypothetical dataset for testing.
    
    Args:
        num_candidates: Number of candidate station locations
        num_demand: Number of demand points
        grid_size: Size of the city grid (width, height)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing dataset parameters
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate candidate locations (e.g., parking lots, shopping centers)
    candidate_locations = np.random.uniform(0, grid_size[0], 
                                          size=(num_candidates, 2))
    
    # Generate demand points (e.g., residential areas)
    demand_points = np.random.uniform(0, grid_size[0], 
                                     size=(num_demand, 2))
    
    # Assign random weights to demand points (population density)
    demand_weights = np.random.uniform(1, 10, size=num_demand)
    
    # Assign random installation costs
    installation_costs = np.random.uniform(3000, 8000, size=num_candidates)
    
    return {
        'candidate_locations': candidate_locations.tolist(),
        'demand_points': demand_points.tolist(),
        'demand_weights': demand_weights.tolist(),
        'installation_costs': installation_costs.tolist()
    }


def save_dataset_csv(dataset: Dict, filename: str):
    """Save dataset to CSV files."""
    # Save candidate locations
    with open(f"{filename}_candidates.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Location_ID', 'X', 'Y', 'Installation_Cost'])
        for i, (loc, cost) in enumerate(zip(dataset['candidate_locations'], 
                                            dataset['installation_costs'])):
            writer.writerow([i, loc[0], loc[1], cost])
    
    # Save demand points
    with open(f"{filename}_demand.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Demand_ID', 'X', 'Y', 'Weight'])
        for i, (loc, weight) in enumerate(zip(dataset['demand_points'], 
                                              dataset['demand_weights'])):
            writer.writerow([i, loc[0], loc[1], weight])
    
    print(f"Dataset saved to {filename}_candidates.csv and {filename}_demand.csv")


if __name__ == "__main__":
    print("="*60)
    print("Electric Vehicle Charging Station Optimization")
    print("Using Genetic Algorithm")
    print("="*60)
    print()
    
    # Generate test dataset
    print("Generating test dataset...")
    dataset = generate_test_dataset(
        num_candidates=10,
        num_demand=50,
        grid_size=(50, 50),
        seed=42
    )
    
    # Save dataset
    save_dataset_csv(dataset, 'charging_station_dataset')
    print()
    
    # Initialize and run genetic algorithm
    print("Initializing Genetic Algorithm...")
    print(f"Population Size: 100")
    print(f"Generations: 200")
    print(f"Crossover Rate: 0.8")
    print(f"Mutation Rate: 0.02")
    print(f"Coverage Radius: 5 miles")
    print(f"Budget: $50,000")
    print()
    
    ga = ChargingStationGA(
        candidate_locations=dataset['candidate_locations'],
        demand_points=dataset['demand_points'],
        demand_weights=dataset['demand_weights'],
        installation_costs=dataset['installation_costs'],
        coverage_radius=5.0,
        budget=50000.0,
        population_size=100,
        num_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.02,
        lambda_weight=0.5,
        alpha_penalty=2.0
    )
    
    print("Running Genetic Algorithm...")
    print()
    results = ga.evolve(verbose=True)
    
    # Visualize results
    print("\nGenerating visualization...")
    ga.plot_results(results, save_path='ga_results.png')
    
    # Export results
    ga.export_results_csv(results, 'selected_stations.csv')
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
