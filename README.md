# Marine Pilot Scheduling Optimization

## İskenderun Gulf Region Case Study

This repository contains a comprehensive implementation of a Linear Mixed-Integer Programming (MILP) approach for optimizing marine pilot schedules in the İskenderun Gulf, Turkey.

## 📊 Key Results

- **946 pilot assignments** optimized across 3 scenarios
- **5,300 rest violations eliminated** through mathematical optimization
- **Up to 97.3% improvement** in workload equity distribution
- **Complete elimination** of rest period violations in feasible scenarios

## 🚀 Features

- **MILP Optimization**: Advanced mathematical optimization using PuLP and CBC solver
- **Scenario Analysis**: Three traffic scenarios (Standard, Peak Season, Disturbed Traffic)
- **Baseline Comparison**: Heuristic baseline vs. optimized scheduling
- **Professional Visualizations**: Academic-quality charts and analysis
- **Comprehensive Reporting**: Detailed performance metrics and recommendations

## 📁 Project Structure

```
marine_pilot_scheduling/
├── src/                          # Source code
│   ├── config.py                # Configuration parameters
│   ├── data_generation.py       # Ship arrival simulation
│   ├── baseline_heuristic.py    # Baseline scheduling algorithm
│   ├── simple_scheduler.py      # MILP optimization engine
│   └── run_analysis.py          # Main analysis runner
├── output/                       # Results and deliverables
│   ├── figures/                 # Visualization outputs
│   ├── tables/                  # Data tables (CSV)
│   ├── data/                    # Generated datasets
│   └── academic_report.md       # Comprehensive analysis report
└── README.md                    # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Required packages (install via pip):

```bash
pip install pandas numpy matplotlib seaborn pulp
```

### Clone Repository

```bash
git clone https://github.com/vahitcalisir-art/Marine-Pilot-Scheduling-Optimization.git
cd Marine-Pilot-Scheduling-Optimization
```

## 🏃‍♂️ Usage

### Quick Start

Run the complete analysis:

```bash
cd src
python run_analysis.py
```

This will:
1. Generate ship arrival data for all scenarios
2. Execute baseline heuristic scheduling
3. Perform MILP optimization
4. Create professional visualizations
5. Generate comprehensive academic report

### Individual Components

Generate data for specific scenario:
```python
from data_generation import generate_scenario_data
jobs_df = generate_scenario_data('A')  # 'A', 'B', or 'C'
```

Run baseline scheduling:
```python
from baseline_heuristic import solve_baseline_scheduling
baseline_results = solve_baseline_scheduling(jobs_df, pilot_ids)
```

Execute MILP optimization:
```python
from simple_scheduler import SimplePilotScheduler
scheduler = SimplePilotScheduler(jobs_df)
milp_results = scheduler.solve()
```

## 📈 Scenarios

### Scenario A: Standard Traffic
- 274 total jobs over 21 days
- 35.4% night operations
- **Results**: 96.2% workload equity improvement, 1,078 violations eliminated

### Scenario B: Peak Season
- 324 total jobs over 21 days
- 52.8% night operations
- **Results**: 97.3% workload equity improvement, 1,751 violations eliminated

### Scenario C: Disturbed Traffic
- 348 total jobs over 21 days
- 63.8% night operations
- **Results**: Required constraint relaxation, 2,471 violations eliminated

## ⚙️ Configuration

Key parameters in `config.py`:

```python
# Fleet Configuration
NUM_PILOTS = 12
PLANNING_HORIZON_DAYS = 21

# Working Time Constraints
MAX_DAILY_HOURS = 16.0
MAX_WEEKLY_HOURS = 84.0
MIN_REST_HOURS = 8.0

# Traffic Scenarios
SCENARIO_CONFIGS = {
    'A': {'daily_arrivals_mean': 13, 'night_bias': 1.0},
    'B': {'daily_arrivals_mean': 15, 'night_bias': 1.8},
    'C': {'daily_arrivals_mean': 17, 'night_bias': 2.5}
}
```

## 📊 Outputs

### Visualizations
- `comprehensive_comparison.png`: Main dashboard comparing all scenarios
- `workload_scenario_X.png`: Individual scenario workload analysis

### Reports
- `academic_report.md`: Complete academic analysis with findings and recommendations
- `summary_results.csv`: Quantitative results table

### Data
- `jobs_scenario_X.csv`: Generated ship arrival datasets for each scenario

## 🔬 Methodology

### Optimization Model

**Objective Function:**
Minimize workload inequality while penalizing night work assignments

**Key Constraints:**
- Each job assigned to exactly one pilot
- Daily working hour limits (16h max)
- Weekly working hour limits (84h max)
- Minimum rest periods between assignments (8h min)
- No overlapping assignments for same pilot

### Algorithms

1. **Baseline Heuristic**: "Next-available pilot" greedy assignment
2. **MILP Optimization**: Linear Mixed-Integer Programming with CBC solver
3. **Fallback Strategy**: Relaxed constraints for high-traffic scenarios

## 📋 Key Findings

1. **Significant Equity Improvements**: Up to 97% reduction in workload inequality
2. **Safety Enhancement**: Complete elimination of rest period violations
3. **Scalability Challenges**: Extreme traffic scenarios require constraint relaxation
4. **Practical Applicability**: Demonstrates feasibility of mathematical optimization for marine operations

## 🎯 Recommendations

1. **Hybrid Approach**: Combine MILP optimization with heuristic fallback
2. **Real-time Adaptation**: Dynamic constraint adjustment based on traffic volume
3. **Fleet Optimization**: Evaluate pilot staffing levels for peak periods
4. **System Integration**: Interface with vessel tracking and port management systems

## 📚 Academic Context

This work addresses the complex challenge of marine pilot scheduling in high-traffic port environments, specifically focusing on the İskenderun Gulf region. The study contributes to:

- Maritime operations research
- Transportation optimization
- Workforce scheduling theory
- Safety management in marine environments

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Vahit Çalışır**
- GitHub: [@vahitcalisir-art](https://github.com/vahitcalisir-art)

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact through GitHub.

---

*İskenderun Gulf Marine Pilot Scheduling Optimization Study - December 2025*
