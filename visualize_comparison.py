"""
Visualization script to create comparison plots across all experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_comparison():
    """Create comparison visualizations across all experiments."""
    
    # Load comparison report
    with open('results/comparison_report.json', 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    scenarios = [r['scenario_name'] for r in results]
    scenario_ids = [r['dataset_id'] for r in results]
    fitness = [r['best_fitness'] for r in results]
    coverage = [r['coverage_percentage'] for r in results]
    costs = [r['total_cost'] for r in results]
    num_stations = [r['num_stations_selected'] for r in results]
    budget_util = [r['budget_utilization'] for r in results]
    convergence = [r['convergence_generation'] for r in results]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot 1: Fitness Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(scenario_ids, fitness, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Dataset ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('Fitness Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, fitness):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Coverage vs Cost
    ax2 = axes[0, 1]
    scatter = ax2.scatter(costs, coverage, s=[n*50 for n in num_stations], 
                         c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, (x, y, label) in enumerate(zip(costs, coverage, scenario_ids)):
        ax2.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Coverage vs Cost (bubble size = # stations)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Number of Stations
    ax3 = axes[1, 0]
    bars3 = ax3.bar(scenario_ids, num_stations, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Dataset ID', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Stations', fontsize=12, fontweight='bold')
    ax3.set_title('Selected Charging Stations per Scenario', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, num_stations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: Budget Utilization and Convergence
    ax4 = axes[1, 1]
    x = np.arange(len(scenario_ids))
    width = 0.35
    
    bars4_1 = ax4.bar(x - width/2, budget_util, width, label='Budget Utilization (%)',
                      color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax4_twin = ax4.twinx()
    bars4_2 = ax4_twin.bar(x + width/2, convergence, width, label='Convergence (gen)',
                          color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax4.set_xlabel('Dataset ID', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Budget Utilization (%)', fontsize=12, fontweight='bold', color='#3498db')
    ax4_twin.set_ylabel('Convergence Generation', fontsize=12, fontweight='bold', color='#e74c3c')
    ax4.set_title('Budget Utilization & Convergence Speed', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenario_ids)
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='#3498db')
    ax4_twin.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/comparison_visualization.png', dpi=300, bbox_inches='tight')
    print("Comparison visualization saved to results/comparison_visualization.png")
    plt.show()


def create_summary_table():
    """Create a formatted summary table."""
    
    with open('results/comparison_report.json', 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*100)
    print("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print()
    
    # Header
    print(f"{'ID':<4} {'Scenario':<22} {'Stations':<9} {'Cost':<14} "
          f"{'Budget':<14} {'Utilization':<12} {'Coverage':<10} {'Fitness':<10}")
    print("-"*100)
    
    # Data rows
    for r in results:
        print(f"{r['dataset_id']:<4} "
              f"{r['scenario_name']:<22} "
              f"{r['num_stations_selected']:<9} "
              f"${r['total_cost']:>11,.0f} "
              f"${r['budget']:>11,} "
              f"{r['budget_utilization']:>10.1f}% "
              f"{r['coverage_percentage']:>9.1f}% "
              f"{r['best_fitness']:>9.4f}")
    
    print()
    print("="*100)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 100)
    
    best_fitness = max(results, key=lambda x: x['best_fitness'])
    best_coverage = max(results, key=lambda x: x['coverage_percentage'])
    most_efficient = min(results, key=lambda x: x['total_cost'] / (x['coverage_percentage'] + 0.01))
    
    print(f"• Best Overall Fitness: {best_fitness['scenario_name']} "
          f"(ID: {best_fitness['dataset_id']}, Fitness: {best_fitness['best_fitness']:.4f})")
    
    print(f"• Highest Coverage: {best_coverage['scenario_name']} "
          f"(ID: {best_coverage['dataset_id']}, Coverage: {best_coverage['coverage_percentage']:.1f}%)")
    
    print(f"• Most Cost-Efficient: {most_efficient['scenario_name']} "
          f"(ID: {most_efficient['dataset_id']}, "
          f"${most_efficient['total_cost']:,.0f} for {most_efficient['coverage_percentage']:.1f}% coverage)")
    
    avg_convergence = np.mean([r['convergence_generation'] for r in results])
    print(f"• Average Convergence Generation: {avg_convergence:.1f}")
    
    print()


if __name__ == "__main__":
    print("Generating comparison visualizations...")
    print()
    
    create_summary_table()
    plot_experiment_comparison()
    
    print("\nAll visualizations complete!")
