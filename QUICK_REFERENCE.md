# Quick Reference Guide

## Running Experiments

### Generate All Datasets and Run All Experiments
```bash
python run_experiments.py all
```

### Run Single Experiment
```bash
python run_experiments.py run 01  # Replace 01 with dataset ID (01-05)
```

### Generate Datasets Only
```bash
python run_experiments.py generate
```

### Parameter Sensitivity Analysis
```bash
python run_experiments.py sensitivity 01  # Runs on dataset 01
```

### Create Comparison Visualizations
```bash
python visualize_comparison.py
```

## Dataset IDs

- **01**: Small Urban Area (10 candidates, 50 demand, $50k budget)
- **02**: Dense City Center (15 candidates, 100 demand, $75k budget)
- **03**: Suburban Sprawl (20 candidates, 80 demand, $100k budget)
- **04**: Budget Constrained (12 candidates, 60 demand, $30k budget)
- **05**: Large Metro Area (25 candidates, 150 demand, $150k budget)

## Output Files

### Per Dataset (`results/XX/`)
- `selected_stations.csv` - Selected charging station locations
- `ga_results.png` - Convergence graph + placement map
- `results_summary.json` - Detailed metrics

### Summary Files
- `results/comparison_report.json` - All results in JSON format
- `results/comparison_visualization.png` - Visual comparison charts

## Key Results

| Dataset | Stations | Cost    | Coverage | Fitness |
|---------|----------|---------|----------|---------|
| 01      | 3        | $10,903 | 12.0%    | 0.0490  |
| 02      | 3        | $18,673 | 17.0%    | 0.0822  |
| 03      | 6        | $31,519 | 23.8%    | 0.0793  |
| 04      | 1        | $5,071  | 6.7%     | 0.0086  |
| 05      | 11       | $66,043 | 35.3%    | 0.1744  |

## GA Parameters (Default)

- Population Size: 100
- Generations: 200 (with early stopping)
- Crossover Rate: 0.8
- Mutation Rate: 0.02
- Lambda (cost weight): 0.5
- Alpha (budget penalty): 2.0

## Customization

Edit `run_experiments.py` to:
- Add new test scenarios
- Modify GA parameters
- Change fitness function weights

Edit `genetic_algorithm_charging_stations.py` to:
- Modify fitness function
- Change selection/crossover/mutation operators
- Adjust convergence criteria
