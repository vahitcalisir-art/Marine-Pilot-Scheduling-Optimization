"""
Configuration parameters for the İskenderun Marine Pilot Scheduling Model
"""

# Simulation Parameters
RANDOM_SEED = 42
PLANNING_HORIZON_DAYS = 21  # 3 weeks for better analysis

# Pilot Fleet  
NUM_PILOTS = 12
PILOT_IDS = [f"P{i:02d}" for i in range(1, NUM_PILOTS + 1)]

# Job Parameters
AVERAGE_JOB_DURATION = 3.5  # hours (2.5 active + 1.0 overhead)
JOB_DURATION_VARIANCE = 0.8  # hours
MIN_JOB_DURATION = 2.0  # hours
MAX_JOB_DURATION = 6.0  # hours

# Working Time Constraints (Maritime Standards - Relaxed for High Traffic)
MAX_DAILY_HOURS = 16.0  # Maximum hours per pilot per day (relaxed)
MAX_WEEKLY_HOURS = 84.0  # Maximum hours per pilot per week (relaxed)
MIN_REST_HOURS = 8.0   # Minimum rest between consecutive jobs (relaxed)
MAX_NIGHT_JOBS_MONTHLY = 15  # Maximum night jobs per pilot per month
MAX_NIGHT_JOBS_WEEKLY = 5   # Maximum night jobs per pilot per week

# Night Work Definition
NIGHT_START_HOUR = 22  # 22:00
NIGHT_END_HOUR = 6     # 06:00

# Traffic Scenarios
SCENARIO_CONFIGS = {
    'A': {  # Standard traffic
        'name': 'Standard Traffic',
        'daily_arrivals_mean': 12.0,  # ~250 jobs per 21 days
        'daily_arrivals_std': 3.0,
        'traffic_multiplier': 1.0,
        'night_bias': 1.0
    },
    'B': {  # Peak season
        'name': 'Peak Season',
        'daily_arrivals_mean': 16.0,  # ~330 jobs per 21 days  
        'daily_arrivals_std': 4.0,
        'traffic_multiplier': 1.3,
        'night_bias': 1.1
    },
    'C': {  # Disturbed traffic
        'name': 'Disturbed Traffic', 
        'daily_arrivals_mean': 15.0,  # ~315 jobs per 21 days
        'daily_arrivals_std': 5.0,
        'traffic_multiplier': 1.25,
        'night_bias': 1.8  # Heavy night bias
    }
}

# Output Paths
OUTPUT_PATHS = {
    'data_dir': '../data',
    'output_dir': '../output',
    'figures_dir': '../output/figures',
    'tables_dir': '../output/tables'
}