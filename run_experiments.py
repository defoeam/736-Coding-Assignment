"""
Script to generate multiple test datasets and run GA experiments on each.
Results are organized in datasets/ and results/ folders.
"""

import os
import json
import numpy as np
from genetic_algorithm_charging_stations import (
    ChargingStationGA, 
    generate_test_dataset, 
    save_dataset_csv
)
import csv


def create_dataset_scenarios():
    """
    Create diverse test scenarios with different characteristics.
    """
    scenarios = [
        {
            'id': '01',
            'name': 'Small Urban Area',
            'num_candidates': 10,
            'num_demand': 50,
            'grid_size': (50, 50),
            'coverage_radius': 5.0,
            'budget': 50000,
            'seed': 42
        },
        {
            'id': '02',
            'name': 'Dense City Center',
            'num_candidates': 15,
            'num_demand': 100,
            'grid_size': (40, 40),
            'coverage_radius': 4.0,
            'budget': 75000,
            'seed': 123
        },
        {
            'id': '03',
            'name': 'Suburban Sprawl',
            'num_candidates': 20,
            'num_demand': 80,
            'grid_size': (80, 80),
            'coverage_radius': 8.0,
            'budget': 100000,
            'seed': 456
        },
        {
            'id': '04',
            'name': 'Budget Constrained',
            'num_candidates': 12,
            'num_demand': 60,
            'grid_size': (50, 50),
            'coverage_radius': 5.0,
            'budget': 30000,  # Low budget
            'seed': 789
        },
        {
            'id': '05',
            'name': 'Large Metro Area',
            'num_candidates': 25,
            'num_demand': 150,
            'grid_size': (100, 100),
            'coverage_radius': 10.0,
            'budget': 150000,
            'seed': 101112
        }
    ]
    
    return scenarios


def generate_and_save_datasets():
    """Generate datasets for all scenarios."""
    scenarios = create_dataset_scenarios()
    
    print("="*70)
    print("GENERATING TEST DATASETS")
    print("="*70)
    print()
    
    for scenario in scenarios:
        dataset_dir = f"datasets/{scenario['id']}"
        os.makedirs(dataset_dir, exist_ok=True)
        
        print(f"Creating Dataset {scenario['id']}: {scenario['name']}")
        print(f"  - Candidates: {scenario['num_candidates']}")
        print(f"  - Demand Points: {scenario['num_demand']}")
        print(f"  - Grid Size: {scenario['grid_size']}")
        print(f"  - Budget: ${scenario['budget']:,}")
        
        # Generate dataset
        dataset = generate_test_dataset(
            num_candidates=scenario['num_candidates'],
            num_demand=scenario['num_demand'],
            grid_size=scenario['grid_size'],
            seed=scenario['seed']
        )
        
        # Save to dataset folder
        save_dataset_csv(dataset, f"{dataset_dir}/charging_station_dataset")
        
        # Save scenario metadata
        scenario_info = {k: v for k, v in scenario.items() if k != 'seed'}
        with open(f"{dataset_dir}/scenario_info.json", 'w') as f:
            json.dump(scenario_info, f, indent=2)
        
        print(f"  ✓ Saved to {dataset_dir}/")
        print()
    
    print(f"Generated {len(scenarios)} datasets successfully!")
    print()


def load_dataset(dataset_id: str):
    """Load a dataset from the datasets folder."""
    dataset_dir = f"datasets/{dataset_id}"
    
    # Load candidates
    candidates = []
    costs = []
    with open(f"{dataset_dir}/charging_station_dataset_candidates.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append((float(row['X']), float(row['Y'])))
            costs.append(float(row['Installation_Cost']))
    
    # Load demand points
    demand_points = []
    weights = []
    with open(f"{dataset_dir}/charging_station_dataset_demand.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            demand_points.append((float(row['X']), float(row['Y'])))
            weights.append(float(row['Weight']))
    
    # Load scenario info
    with open(f"{dataset_dir}/scenario_info.json", 'r') as f:
        scenario_info = json.load(f)
    
    return {
        'candidate_locations': candidates,
        'installation_costs': costs,
        'demand_points': demand_points,
        'demand_weights': weights,
        'scenario_info': scenario_info
    }


def run_experiment(dataset_id: str, ga_params: dict = None):
    """Run GA experiment on a specific dataset."""
    
    # Load dataset
    data = load_dataset(dataset_id)
    scenario = data['scenario_info']
    
    # Create results directory
    results_dir = f"results/{dataset_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print(f"EXPERIMENT {dataset_id}: {scenario['name']}")
    print("="*70)
    print()
    
    # Default GA parameters
    default_params = {
        'population_size': 100,
        'num_generations': 200,
        'crossover_rate': 0.8,
        'mutation_rate': 0.02,
        'lambda_weight': 0.1,
        'alpha_penalty': 2.0
    }
    
    if ga_params:
        default_params.update(ga_params)
    
    print(f"Dataset: {scenario['name']}")
    print(f"Candidates: {len(data['candidate_locations'])}")
    print(f"Demand Points: {len(data['demand_points'])}")
    print(f"Coverage Radius: {scenario['coverage_radius']} miles")
    print(f"Budget: ${scenario['budget']:,}")
    print()
    print("GA Parameters:")
    for key, value in default_params.items():
        print(f"  - {key}: {value}")
    print()
    
    # Initialize and run GA
    ga = ChargingStationGA(
        candidate_locations=data['candidate_locations'],
        demand_points=data['demand_points'],
        demand_weights=data['demand_weights'],
        installation_costs=data['installation_costs'],
        coverage_radius=scenario['coverage_radius'],
        budget=scenario['budget'],
        **default_params
    )
    
    print("Running Genetic Algorithm...")
    print()
    results = ga.evolve(verbose=True)
    
    # Save results
    print(f"\nSaving results to {results_dir}/")
    
    # Export selected stations
    ga.export_results_csv(results, f"{results_dir}/selected_stations.csv")
    
    # Save visualization
    ga.plot_results(results, save_path=f"{results_dir}/ga_results.png")
    
    # Save detailed results as JSON
    results_summary = {
        'dataset_id': dataset_id,
        'scenario_name': scenario['name'],
        'best_fitness': float(results['best_fitness']),
        'num_stations_selected': int(results['num_stations']),
        'total_cost': float(results['total_cost']),
        'budget': scenario['budget'],
        'within_budget': bool(results['within_budget']),
        'budget_utilization': float(results['budget_utilization']),
        'weighted_coverage': float(results['weighted_coverage']),
        'num_covered': int(results['num_covered']),
        'total_demand_points': len(data['demand_points']),
        'coverage_percentage': float(results['coverage_percentage']),
        'selected_indices': results['selected_indices'].tolist(),
        'ga_parameters': default_params,
        'convergence_generation': len(results['avg_fitness_history'])
    }
    
    with open(f"{results_dir}/results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"  ✓ selected_stations.csv")
    print(f"  ✓ ga_results.png")
    print(f"  ✓ results_summary.json")
    print()
    
    return results_summary


def run_all_experiments():
    """Run experiments on all datasets."""
    scenarios = create_dataset_scenarios()
    
    print("="*70)
    print("RUNNING ALL EXPERIMENTS")
    print("="*70)
    print()
    
    all_results = []
    
    for scenario in scenarios:
        results_summary = run_experiment(scenario['id'])
        all_results.append(results_summary)
        print()
    
    # Create comparison report
    print("="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)
    print()
    
    print(f"{'ID':<6} {'Scenario':<25} {'Stations':<10} {'Cost':<15} {'Coverage':<10} {'Fitness':<10}")
    print("-"*86)
    
    for result in all_results:
        print(f"{result['dataset_id']:<6} "
              f"{result['scenario_name']:<25} "
              f"{result['num_stations_selected']:<10} "
              f"${result['total_cost']:>12,.0f} "
              f"{result['coverage_percentage']:>9.1f}% "
              f"{result['best_fitness']:>9.4f}")
    
    # Save comparison report
    with open("results/comparison_report.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print()
    print("Comparison report saved to results/comparison_report.json")
    print()
    
    return all_results


def run_parameter_sensitivity_analysis(dataset_id: str = '01'):
    """
    Run experiments with different GA parameters on the same dataset
    to analyze sensitivity.
    """
    print("="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"Dataset: {dataset_id}")
    print("="*70)
    print()
    
    parameter_configs = [
        {
            'name': 'Baseline',
            'params': {
                'mutation_rate': 0.02,
                'crossover_rate': 0.8,
                'population_size': 100
            }
        },
        {
            'name': 'High Mutation',
            'params': {
                'mutation_rate': 0.05,
                'crossover_rate': 0.8,
                'population_size': 100
            }
        },
        {
            'name': 'Low Mutation',
            'params': {
                'mutation_rate': 0.01,
                'crossover_rate': 0.8,
                'population_size': 100
            }
        },
        {
            'name': 'Large Population',
            'params': {
                'mutation_rate': 0.02,
                'crossover_rate': 0.8,
                'population_size': 200
            }
        },
        {
            'name': 'High Crossover',
            'params': {
                'mutation_rate': 0.02,
                'crossover_rate': 0.95,
                'population_size': 100
            }
        }
    ]
    
    sensitivity_results = []
    
    for i, config in enumerate(parameter_configs, 1):
        print(f"\n{'='*70}")
        print(f"Configuration {i}: {config['name']}")
        print(f"{'='*70}\n")
        
        # Create subdirectory for this configuration
        config_id = f"{dataset_id}_config{i:02d}"
        results_dir = f"results/sensitivity/{config_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Run experiment
        data = load_dataset(dataset_id)
        scenario = data['scenario_info']
        
        ga = ChargingStationGA(
            candidate_locations=data['candidate_locations'],
            demand_points=data['demand_points'],
            demand_weights=data['demand_weights'],
            installation_costs=data['installation_costs'],
            coverage_radius=scenario['coverage_radius'],
            budget=scenario['budget'],
            **config['params']
        )
        
        results = ga.evolve(verbose=False)
        
        # Save results
        ga.export_results_csv(results, f"{results_dir}/selected_stations.csv")
        ga.plot_results(results, save_path=f"{results_dir}/ga_results.png")
        
        summary = {
            'config_name': config['name'],
            'parameters': config['params'],
            'best_fitness': float(results['best_fitness']),
            'num_stations': int(results['num_stations']),
            'coverage_percentage': float(results['coverage_percentage']),
            'total_cost': float(results['total_cost']),
            'convergence_generation': len(results['avg_fitness_history'])
        }
        
        sensitivity_results.append(summary)
        
        print(f"Results: Fitness={summary['best_fitness']:.4f}, "
              f"Coverage={summary['coverage_percentage']:.1f}%, "
              f"Stations={summary['num_stations']}, "
              f"Converged at gen {summary['convergence_generation']}")
    
    # Save sensitivity analysis report
    with open("results/sensitivity/sensitivity_analysis.json", 'w') as f:
        json.dump(sensitivity_results, f, indent=2)
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<20} {'Fitness':<12} {'Coverage':<12} {'Stations':<10} {'Conv.Gen':<10}")
    print("-"*64)
    
    for result in sensitivity_results:
        print(f"{result['config_name']:<20} "
              f"{result['best_fitness']:<12.4f} "
              f"{result['coverage_percentage']:<11.1f}% "
              f"{result['num_stations']:<10} "
              f"{result['convergence_generation']:<10}")
    
    print()
    print("Sensitivity analysis saved to results/sensitivity/")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "generate":
            # Generate datasets only
            generate_and_save_datasets()
        
        elif command == "run":
            # Run single experiment
            if len(sys.argv) > 2:
                dataset_id = sys.argv[2]
                run_experiment(dataset_id)
            else:
                print("Usage: python run_experiments.py run <dataset_id>")
        
        elif command == "all":
            # Generate datasets and run all experiments
            generate_and_save_datasets()
            run_all_experiments()
        
        elif command == "sensitivity":
            # Run parameter sensitivity analysis
            dataset_id = sys.argv[2] if len(sys.argv) > 2 else '01'
            run_parameter_sensitivity_analysis(dataset_id)
        
        else:
            print("Unknown command. Available commands:")
            print("  generate  - Generate all datasets")
            print("  run <id>  - Run experiment on specific dataset")
            print("  all       - Generate datasets and run all experiments")
            print("  sensitivity [id] - Run parameter sensitivity analysis")
    
    else:
        # Default: generate and run all
        print("Generating datasets and running all experiments...")
        print()
        generate_and_save_datasets()
        run_all_experiments()
        print()
        print("Would you like to run sensitivity analysis? (requires more time)")
        print("Run: python run_experiments.py sensitivity")
