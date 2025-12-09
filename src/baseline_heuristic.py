"""
Baseline heuristic algorithm for marine pilot scheduling.
Implements the "next-available pilot" greedy assignment strategy
used as a comparison baseline for the MILP optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import config


class BaselineScheduler:
    """
    Greedy baseline scheduler implementing "next-available pilot" assignment.
    This represents typical manual scheduling practices in marine pilotage.
    """
    
    def __init__(self, jobs_df: pd.DataFrame, pilots: List[str] = None):
        """
        Initialize the baseline scheduler.
        
        Args:
            jobs_df: DataFrame with job data (job_id, start_time, duration, end_time, day, is_night)
            pilots: List of pilot IDs (defaults to config.PILOT_IDS)
        """
        self.jobs_df = jobs_df.copy().sort_values('start_time').reset_index(drop=True)
        self.pilots = pilots or config.PILOT_IDS
        
        # Initialize pilot state tracking
        self.pilot_state = {}
        for pilot in self.pilots:
            self.pilot_state[pilot] = {
                'last_job_end': 0.0,  # Time when last job ended
                'daily_hours': {},    # Hours worked per day
                'weekly_hours': 0.0,  # Total hours in current period
                'night_jobs': 0,      # Number of night jobs assigned
                'total_hours': 0.0,   # Total hours worked
                'jobs': []           # List of assigned jobs
            }
            
            # Initialize daily hours for all days
            for day in range(config.PLANNING_HORIZON_DAYS):
                self.pilot_state[pilot]['daily_hours'][day] = 0.0
    
    def schedule(self, enforce_strict_constraints: bool = True) -> Dict:
        """
        Generate schedule using greedy next-available pilot assignment.
        
        Args:
            enforce_strict_constraints: If True, strictly enforce all constraints.
                                      If False, allow some constraint violations
                                      (to simulate real-world flexibility).
        
        Returns:
            Dictionary with scheduling results and violations
        """
        assignments = []
        violations = {
            'rest_violations': [],
            'daily_hour_violations': [],
            'weekly_hour_violations': [],
            'night_work_violations': [],
            'delayed_jobs': []
        }
        
        for _, job in self.jobs_df.iterrows():
            pilot, delay = self._assign_job(job, enforce_strict_constraints, violations)
            
            if pilot is not None:
                # Record assignment
                assignment = {
                    'job_id': job['job_id'],
                    'pilot_id': pilot,
                    'planned_start': job['start_time'],
                    'actual_start': job['start_time'] + delay,
                    'end_time': job['start_time'] + delay + job['duration'],
                    'duration': job['duration'],
                    'day': job['day'],
                    'is_night': job['is_night'],
                    'delay': delay
                }
                assignments.append(assignment)
                
                # Update pilot state
                self._update_pilot_state(pilot, assignment)
                
                if delay > 0:
                    violations['delayed_jobs'].append({
                        'job_id': job['job_id'],
                        'delay': delay
                    })
            else:
                print(f"Warning: Could not assign job {job['job_id']}")
        
        # Create results
        results = {
            'assignments': pd.DataFrame(assignments),
            'pilot_hours': {p: self.pilot_state[p]['total_hours'] for p in self.pilots},
            'violations': violations,
            'pilot_state': self.pilot_state
        }
        
        return results
    
    def _assign_job(self, job: pd.Series, enforce_strict: bool, violations: Dict) -> Tuple[Optional[str], float]:
        """
        Assign a job to the first available pilot.
        
        Args:
            job: Job to assign
            enforce_strict: Whether to strictly enforce constraints
            violations: Dictionary to record violations
            
        Returns:
            Tuple of (selected_pilot, delay_hours)
        """
        best_pilot = None
        min_delay = float('inf')
        
        for pilot in self.pilots:
            pilot_available_time = self._get_pilot_availability(pilot, job, enforce_strict, violations)
            
            if pilot_available_time is not None:
                delay = max(0, pilot_available_time - job['start_time'])
                
                if delay < min_delay:
                    min_delay = delay
                    best_pilot = pilot
                    
                    # If no delay needed, assign immediately
                    if delay == 0:
                        break
        
        return best_pilot, min_delay if best_pilot else 0
    
    def _get_pilot_availability(self, pilot: str, job: pd.Series, 
                               enforce_strict: bool, violations: Dict) -> Optional[float]:
        """
        Determine when a pilot becomes available for a job.
        
        Args:
            pilot: Pilot ID
            job: Job to assign
            enforce_strict: Whether to enforce strict constraints
            violations: Dictionary to record violations
            
        Returns:
            Earliest time pilot is available, or None if unavailable
        """
        state = self.pilot_state[pilot]
        
        # Basic availability: after last job + minimum rest
        earliest_start = max(
            job['start_time'],
            state['last_job_end'] + config.MIN_REST_HOURS
        )
        
        # Check rest constraint
        if state['last_job_end'] > 0:  # Not first job
            rest_time = job['start_time'] - state['last_job_end']
            if rest_time < config.MIN_REST_HOURS:
                if enforce_strict:
                    earliest_start = state['last_job_end'] + config.MIN_REST_HOURS
                else:
                    # Allow violation but record it
                    violations['rest_violations'].append({
                        'pilot_id': pilot,
                        'job_id': job['job_id'],
                        'actual_rest': rest_time,
                        'required_rest': config.MIN_REST_HOURS
                    })
        
        # Check daily hours constraint
        job_day = job['day']
        if job_day >= config.PLANNING_HORIZON_DAYS:
            # Handle edge case where job extends beyond planning horizon
            return None
            
        current_daily_hours = state['daily_hours'].get(job_day, 0.0)
        
        if current_daily_hours + job['duration'] > config.MAX_DAILY_HOURS:
            if enforce_strict:
                return None  # Cannot assign without violating daily limit
            else:
                violations['daily_hour_violations'].append({
                    'pilot_id': pilot,
                    'job_id': job['job_id'],
                    'day': job_day,
                    'current_hours': current_daily_hours,
                    'job_duration': job['duration'],
                    'total_would_be': current_daily_hours + job['duration'],
                    'limit': config.MAX_DAILY_HOURS
                })
        
        # Check weekly/total hours constraint
        weekly_limit = config.MAX_WEEKLY_HOURS * (config.PLANNING_HORIZON_DAYS / 7)
        if state['weekly_hours'] + job['duration'] > weekly_limit:
            if enforce_strict:
                return None
            else:
                violations['weekly_hour_violations'].append({
                    'pilot_id': pilot,
                    'job_id': job['job_id'],
                    'current_hours': state['weekly_hours'],
                    'job_duration': job['duration'],
                    'total_would_be': state['weekly_hours'] + job['duration'],
                    'limit': weekly_limit
                })
        
        # Check night work constraint
        if job['is_night'] and state['night_jobs'] >= config.MAX_NIGHT_JOBS_MONTHLY:
            if enforce_strict:
                return None
            else:
                violations['night_work_violations'].append({
                    'pilot_id': pilot,
                    'job_id': job['job_id'],
                    'current_night_jobs': state['night_jobs'],
                    'limit': config.MAX_NIGHT_JOBS_MONTHLY
                })
        
        return earliest_start
    
    def _update_pilot_state(self, pilot: str, assignment: Dict):
        """Update pilot state after assignment."""
        state = self.pilot_state[pilot]
        
        # Update last job end time
        state['last_job_end'] = assignment['end_time']
        
        # Update hours tracking
        day = assignment['day']
        duration = assignment['duration']
        
        # Handle edge case for days beyond planning horizon
        if day < config.PLANNING_HORIZON_DAYS:
            state['daily_hours'][day] += duration
        
        state['weekly_hours'] += duration
        state['total_hours'] += duration
        
        # Update night jobs counter
        if assignment['is_night']:
            state['night_jobs'] += 1
        
        # Add to jobs list
        state['jobs'].append(assignment)
    
    def print_summary(self, results: Dict):
        """Print summary of baseline scheduling results."""
        assignments = results['assignments']
        violations = results['violations']
        pilot_hours = results['pilot_hours']
        
        print(f"\n=== Baseline Scheduling Summary ===")
        print(f"Total jobs assigned: {len(assignments)}")
        print(f"Total pilots: {len(self.pilots)}")
        
        if len(assignments) > 0:
            print(f"Average delay per job: {assignments['delay'].mean():.2f} hours")
            print(f"Jobs with delays: {(assignments['delay'] > 0).sum()}")
            
        # Workload distribution
        hours_list = list(pilot_hours.values())
        print(f"\nPilot Workload Distribution:")
        print(f"  Mean: {np.mean(hours_list):.2f} hours")
        print(f"  Std Dev: {np.std(hours_list):.2f} hours")
        print(f"  Min: {np.min(hours_list):.2f} hours")
        print(f"  Max: {np.max(hours_list):.2f} hours")
        
        # Violations summary
        print(f"\nConstraint Violations:")
        print(f"  Rest violations: {len(violations['rest_violations'])}")
        print(f"  Daily hour violations: {len(violations['daily_hour_violations'])}")
        print(f"  Weekly hour violations: {len(violations['weekly_hour_violations'])}")
        print(f"  Night work violations: {len(violations['night_work_violations'])}")
        print(f"  Delayed jobs: {len(violations['delayed_jobs'])}")


def solve_baseline_scheduling(jobs_df: pd.DataFrame, pilots: List[str] = None,
                            enforce_strict_constraints: bool = True) -> Dict:
    """
    Convenience function to solve using baseline heuristic.
    
    Args:
        jobs_df: DataFrame with job data
        pilots: List of pilot IDs
        enforce_strict_constraints: Whether to enforce all constraints strictly
        
    Returns:
        Dictionary with baseline scheduling results
    """
    scheduler = BaselineScheduler(jobs_df, pilots)
    return scheduler.schedule(enforce_strict_constraints)


if __name__ == "__main__":
    # Test the baseline scheduler
    from data_generation import generate_scenario_data
    
    print("Testing baseline scheduler...")
    
    # Generate test data
    jobs_df = generate_scenario_data('A', random_seed=42)
    print(f"Generated {len(jobs_df)} jobs for testing")
    
    # Test with reduced pilot count
    test_pilots = config.PILOT_IDS[:10]
    
    scheduler = BaselineScheduler(jobs_df, test_pilots)
    
    print("\nTesting strict constraint enforcement...")
    results_strict = scheduler.schedule(enforce_strict_constraints=True)
    scheduler.print_summary(results_strict)
    
    # Reset scheduler for second test
    scheduler = BaselineScheduler(jobs_df, test_pilots)
    
    print("\nTesting relaxed constraint enforcement...")
    results_relaxed = scheduler.schedule(enforce_strict_constraints=False)
    scheduler.print_summary(results_relaxed)