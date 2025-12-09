"""
Data generation module for İskenderun marine pilot scheduling simulation.
Implements Poisson process-based ship arrival generation for three scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import config


class ShipArrivalGenerator:
    """
    Generates synthetic ship arrivals using non-homogeneous Poisson processes
    calibrated to İskenderun Gulf traffic patterns.
    """
    
    def __init__(self, scenario: str, random_seed: int = None):
        """
        Initialize the ship arrival generator.
        
        Args:
            scenario: 'A', 'B', or 'C' for different traffic scenarios
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.scenario = scenario
        self.scenario_config = config.SCENARIO_CONFIGS[scenario]
        self.planning_horizon = config.PLANNING_HORIZON_DAYS
        
    def generate_arrivals(self) -> pd.DataFrame:
        """
        Generate ship arrivals for the planning horizon.
        
        Returns:
            DataFrame with columns: job_id, start_time, duration, end_time, day, is_night
        """
        arrivals = []
        
        for day in range(self.planning_horizon):
            day_arrivals = self._generate_daily_arrivals(day)
            arrivals.extend(day_arrivals)
        
        # Convert to DataFrame
        df = pd.DataFrame(arrivals)
        
        # Add computed fields
        df['end_time'] = df['start_time'] + df['duration']
        df['is_night'] = df['start_time'].apply(self._is_night_job)
        df['day'] = df['start_time'].apply(lambda x: int(x // 24))
        
        # Sort by start time
        df = df.sort_values('start_time').reset_index(drop=True)
        df['job_id'] = [f"J{i:03d}" for i in range(len(df))]
        
        return df[['job_id', 'start_time', 'duration', 'end_time', 'day', 'is_night']]
    
    def _generate_daily_arrivals(self, day: int) -> List[Dict]:
        """Generate arrivals for a single day."""
        # Determine number of arrivals for this day
        if self.scenario == 'C':  # Disturbed traffic with clustering
            n_arrivals = self._get_clustered_daily_arrivals(day)
        else:
            n_arrivals = max(0, int(np.random.normal(
                self.scenario_config['daily_arrivals_mean'],
                self.scenario_config['daily_arrivals_std']
            )))
        
        arrivals = []
        for _ in range(n_arrivals):
            arrival_time = self._generate_arrival_time(day)
            duration = self._generate_job_duration()
            
            arrivals.append({
                'start_time': arrival_time,
                'duration': duration
            })
        
        return arrivals
    
    def _generate_arrival_time(self, day: int) -> float:
        """Generate arrival time within a day using appropriate distribution."""
        if self.scenario == 'C':  # Clustered/disturbed traffic
            return self._generate_clustered_arrival_time(day)
        else:
            # Uniform distribution with slight night bias for scenario B
            hour = np.random.uniform(0, 24)
            if self.scenario == 'B' and self.scenario_config['night_bias'] > 1.0:
                # Apply night bias
                if self._is_night_hour(hour):
                    hour = hour  # Keep night hours as is (already biased by selection)
                else:
                    # Reduce probability of day hours
                    if np.random.random() > 0.7:
                        hour = np.random.uniform(22, 24) if hour > 12 else np.random.uniform(0, 6)
            
            return day * 24 + hour
    
    def _generate_clustered_arrival_time(self, day: int) -> float:
        """Generate clustered arrival times for disturbed traffic scenario."""
        # Create clusters, especially during night hours
        cluster_centers = [2, 4, 14, 19, 22, 23.5]  # Peak times
        cluster_weights = [0.25, 0.2, 0.15, 0.1, 0.15, 0.15]  # Favor night clusters
        
        # Select cluster
        cluster_center = np.random.choice(cluster_centers, p=cluster_weights)
        
        # Add noise around cluster center
        noise = np.random.normal(0, 1.5)  # 1.5 hour standard deviation
        hour = max(0, min(24, cluster_center + noise))
        
        return day * 24 + hour
    
    def _get_clustered_daily_arrivals(self, day: int) -> int:
        """Get number of arrivals for clustered scenario with day-to-day variation."""
        base_arrivals = self.scenario_config['daily_arrivals_mean']
        
        # Add clustering: some days have significantly more traffic
        if day % 7 in [1, 3, 5]:  # Certain days are busier
            multiplier = np.random.uniform(1.3, 1.8)
        else:
            multiplier = np.random.uniform(0.7, 1.2)
        
        n_arrivals = max(0, int(np.random.normal(
            base_arrivals * multiplier,
            self.scenario_config['daily_arrivals_std']
        )))
        
        return n_arrivals
    
    def _generate_job_duration(self) -> float:
        """Generate job duration with realistic variation."""
        duration = np.random.normal(
            config.AVERAGE_JOB_DURATION,
            config.JOB_DURATION_VARIANCE
        )
        
        # Clamp to realistic bounds
        return max(config.MIN_JOB_DURATION, 
                  min(config.MAX_JOB_DURATION, duration))
    
    def _is_night_job(self, start_time: float) -> bool:
        """Check if a job is a night job based on start time."""
        hour = start_time % 24
        return self._is_night_hour(hour)
    
    def _is_night_hour(self, hour: float) -> bool:
        """Check if an hour is within night time range."""
        return hour >= config.NIGHT_START_HOUR or hour <= config.NIGHT_END_HOUR


def generate_scenario_data(scenario: str, random_seed: int = None) -> pd.DataFrame:
    """
    Convenience function to generate data for a specific scenario.
    
    Args:
        scenario: 'A', 'B', or 'C'
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with ship arrival data
    """
    generator = ShipArrivalGenerator(scenario, random_seed)
    return generator.generate_arrivals()


def preprocess_for_optimization(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Preprocess job data for optimization model.
    
    Args:
        df: DataFrame with job data
    
    Returns:
        Tuple of (overlapping_pairs, rest_incompatible_pairs)
    """
    jobs = df['job_id'].tolist()
    
    # Find overlapping job pairs
    overlapping_pairs = set()
    for i, job1 in df.iterrows():
        for j, job2 in df.iterrows():
            if i < j:  # Avoid duplicates
                if (job1['start_time'] < job2['end_time'] and 
                    job2['start_time'] < job1['end_time']):
                    overlapping_pairs.add((job1['job_id'], job2['job_id']))
    
    # Find rest-incompatible pairs
    rest_incompatible_pairs = set()
    for i, job1 in df.iterrows():
        for j, job2 in df.iterrows():
            if i != j:
                time_gap = job2['start_time'] - job1['end_time']
                if 0 < time_gap < config.MIN_REST_HOURS:
                    rest_incompatible_pairs.add((job1['job_id'], job2['job_id']))
    
    return overlapping_pairs, rest_incompatible_pairs


if __name__ == "__main__":
    # Test data generation
    print("Testing ship arrival generation...")
    
    for scenario in ['A', 'B', 'C']:
        print(f"\nScenario {scenario}:")
        df = generate_scenario_data(scenario, random_seed=42)
        
        print(f"  Total jobs: {len(df)}")
        print(f"  Average jobs per day: {len(df) / config.PLANNING_HORIZON_DAYS:.1f}")
        print(f"  Night jobs: {df['is_night'].sum()} ({df['is_night'].mean()*100:.1f}%)")
        print(f"  Average duration: {df['duration'].mean():.2f} hours")
        
        # Save sample data
        output_file = f"../experiments/sample_data_scenario_{scenario}.csv"
        df.to_csv(output_file, index=False)
        print(f"  Sample data saved to: {output_file}")