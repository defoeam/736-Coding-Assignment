# Electric Vehicle Charging Station Optimization with Genetic Algorithm

## Overview
This project implements a Genetic Algorithm (GA) to optimize the placement of electric vehicle charging stations in a city. The algorithm balances two competing objectives: maximizing coverage of demand points while minimizing installation and operational costs.

## Project Structure

```
736-Coding-Assignment/
├── genetic_algorithm_charging_stations.py  # Core GA implementation
├── run_experiments.py                      # Experiment runner script
├── datasets/                               # Test datasets
│   ├── 01/                                 # Small Urban Area
│   ├── 02/                                 # Dense City Center
│   ├── 03/                                 # Suburban Sprawl
│   ├── 04/                                 # Budget Constrained
│   └── 05/                                 # Large Metro Area
│       ├── charging_station_dataset_candidates.csv
│       ├── charging_station_dataset_demand.csv
│       └── scenario_info.json
├── results/                                # Experiment results
│   ├── 01/
│   ├── 02/
│   ├── 03/
│   ├── 04/
│   └── 05/
│       ├── selected_stations.csv
│       ├── ga_results.png
│       └── results_summary.json
│   └── comparison_report.json
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage

### Running All Experiments

Generate all datasets and run experiments on each:

```bash
python run_experiments.py all
```

### Running Individual Experiments

Run experiment on a specific dataset:

```bash
python run_experiments.py run 01
```

### Generating Datasets Only

Generate test datasets without running experiments:

```bash
python run_experiments.py generate
```

### Parameter Sensitivity Analysis

Run sensitivity analysis on different GA parameters:

```bash
python run_experiments.py sensitivity 01
```

## Test Datasets

The project includes 5 diverse test scenarios:

| ID | Scenario Name          | Candidates | Demand Points | Grid Size | Coverage Radius | Budget    |
|----|------------------------|------------|---------------|-----------|-----------------|-----------|
| 01 | Small Urban Area       | 10         | 50            | 50×50     | 5.0 miles       | $50,000   |
| 02 | Dense City Center      | 15         | 100           | 40×40     | 4.0 miles       | $75,000   |
| 03 | Suburban Sprawl        | 20         | 80            | 80×80     | 8.0 miles       | $100,000  |
| 04 | Budget Constrained     | 12         | 60            | 50×50     | 5.0 miles       | $30,000   |
| 05 | Large Metro Area       | 25         | 150           | 100×100   | 10.0 miles      | $150,000  |

## Genetic Algorithm Parameters

Default GA configuration:

- **Population Size**: 100 chromosomes
- **Generations**: 200 maximum (with early stopping)
- **Crossover Rate**: 0.8 (80%)
- **Mutation Rate**: 0.02 (2%)
- **Lambda Weight**: 0.1 (coverage prioritizations)
- **Alpha Penalty**: 2.0 (budget violation penalty)

## Algorithm Components

### Chromosome Representation
- Binary encoding: each gene represents whether a candidate location is selected (1) or not (0)
- Example: `[1, 0, 0, 1, 1, 0]` means locations 0, 3, and 4 are selected

### Fitness Function
```
Fitness = (coverage / total_demand) - λ * (cost / max_cost) - α * budget_penalty
```

### Selection
- Roulette wheel selection (proportional to fitness)
- Probability of selection increases with fitness value

### Crossover
- Single-point crossover
- Random crossover point chosen
- Applied with 80% probability

### Mutation
- Bit-flip mutation
- Each gene has 2% probability of flipping
- Maintains population diversity

### Elitism
- Best individual always preserved
- Ensures best solution is never lost

## Output Files

### Per-Dataset Results (`results/XX/`)
- **selected_stations.csv**: List of selected charging stations with coordinates and costs
- **ga_results.png**: Visualization showing convergence graph and station placement map
- **results_summary.json**: Detailed metrics and parameters

### Comparison Report (`results/comparison_report.json`)
- Comprehensive comparison across all test scenarios
- Includes fitness, coverage, cost, and convergence statistics

## Customization

### Creating Custom Datasets

Manually create CSV files in `datasets/XX/`:

**charging_station_dataset_candidates.csv**:
```csv
Location_ID,X,Y,Installation_Cost
0,10.5,20.3,5000
1,30.2,45.8,6500
...
```

**charging_station_dataset_demand.csv**:
```csv
Demand_ID,X,Y,Weight
0,12.3,25.7,8.5
1,35.1,40.2,6.2
...
```

### Modifying GA Parameters

Edit parameters in `run_experiments.py` or create custom configurations in the `run_experiment()` function.

## Performance Considerations

- Convergence typically occurs within 50-100 generations
- Early stopping prevents unnecessary computation
- Larger populations provide better exploration but slower convergence
- Higher mutation rates increase diversity but may slow convergence

## Author

Anthony DeFoe  
CSCI 736 Computational Intelligence  
North Dakota State University

## License

This project is for educational purposes as part of a university coding assignment.
