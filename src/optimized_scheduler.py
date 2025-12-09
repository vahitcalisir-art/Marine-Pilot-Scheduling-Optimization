"""
Optimized MILP model for marine pilot scheduling with improved constraints.
"""

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import config


class OptimizedPilotScheduler:
    """Enhanced MILP model for pilot scheduling with better constraint handling."""
    
    def __init__(self, jobs_df: pd.DataFrame):
        self.jobs_df = jobs_df.copy()
        self.pilots = config.PILOT_IDS
        self.jobs = jobs_df['job_id'].tolist()
        
        # Precompute job parameters
        self.job_params = {}
        for _, row in jobs_df.iterrows():
            self.job_params[row['job_id']] = {
                'duration': row['duration'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'day': int(row['day']),
                'is_night': row['is_night']
            }
        
        # Precompute constraint sets
        self.overlaps = self._find_overlapping_jobs()
        self.rest_conflicts = self._find_rest_conflicts()
        
        # Calculate target for fairness
        total_duration = jobs_df['duration'].sum()
        self.target_hours = total_duration / len(self.pilots)
        
    def _find_overlapping_jobs(self) -> List[tuple]:
        """Find pairs of jobs that overlap in time."""
        overlaps = []
        for i, job1 in self.jobs_df.iterrows():
            for j, job2 in self.jobs_df.iterrows():
                if i < j:
                    # Check for actual time overlap
                    if (job1['start_time'] < job2['end_time'] and 
                        job2['start_time'] < job1['end_time']):
                        overlaps.append((job1['job_id'], job2['job_id']))
        return overlaps
    
    def _find_rest_conflicts(self) -> List[tuple]:
        """Find pairs of jobs that violate minimum rest requirements."""
        conflicts = []
        for i, job1 in self.jobs_df.iterrows():
            for j, job2 in self.jobs_df.iterrows():
                if i != j:
                    gap = job2['start_time'] - job1['end_time']
                    if 0 < gap < config.MIN_REST_HOURS:
                        conflicts.append((job1['job_id'], job2['job_id']))
        return conflicts
    
    def solve(self, time_limit: int = 180) -> Optional[Dict]:
        """Solve the optimization problem."""
        print(f"Optimizing schedule for {len(self.jobs)} jobs with {len(self.pilots)} pilots...")
        
        # Create optimization model
        model = pulp.LpProblem("PilotScheduling", pulp.LpMinimize)
        
        # Decision variables
        x = pulp.LpVariable.dicts("assign", 
                                 [(j, p) for j in self.jobs for p in self.pilots],
                                 cat='Binary')
        
        # Total hours per pilot
        total_hours = pulp.LpVariable.dicts("hours", self.pilots, lowBound=0)
        
        # Deviation variables for fairness
        dev_plus = pulp.LpVariable.dicts("dev_plus", self.pilots, lowBound=0)
        dev_minus = pulp.LpVariable.dicts("dev_minus", self.pilots, lowBound=0)
        
        # Objective: minimize unfairness and night work penalties
        night_penalty = pulp.lpSum([
            (3.0 if self.job_params[j]['is_night'] else 1.0) * x[(j, p)]
            for j in self.jobs for p in self.pilots
        ])
        
        fairness_penalty = pulp.lpSum([dev_plus[p] + dev_minus[p] for p in self.pilots])
        
        model += 1.0 * night_penalty + 10.0 * fairness_penalty
        
        # Constraints
        
        # 1. Each job assigned to exactly one pilot
        for j in self.jobs:
            model += pulp.lpSum([x[(j, p)] for p in self.pilots]) == 1
        
        # 2. No overlapping assignments
        for (j1, j2) in self.overlaps:
            for p in self.pilots:
                model += x[(j1, p)] + x[(j2, p)] <= 1
        
        # 3. Rest requirements
        for (j1, j2) in self.rest_conflicts:
            for p in self.pilots:
                model += x[(j1, p)] + x[(j2, p)] <= 1
        
        # 4. Daily hour limits
        for p in self.pilots:
            for day in range(config.PLANNING_HORIZON_DAYS):
                day_jobs = [j for j in self.jobs if self.job_params[j]['day'] == day]
                if day_jobs:
                    model += pulp.lpSum([
                        self.job_params[j]['duration'] * x[(j, p)]
                        for j in day_jobs
                    ]) <= config.MAX_DAILY_HOURS
        
        # 5. Total hours calculation
        for p in self.pilots:
            model += total_hours[p] == pulp.lpSum([
                self.job_params[j]['duration'] * x[(j, p)] for j in self.jobs
            ])
        
        # 6. Weekly hour limits
        weekly_limit = config.MAX_WEEKLY_HOURS * (config.PLANNING_HORIZON_DAYS / 7.0)
        for p in self.pilots:
            model += total_hours[p] <= weekly_limit
        
        # 7. Night work limits
        for p in self.pilots:
            night_jobs = [j for j in self.jobs if self.job_params[j]['is_night']]
            if night_jobs:
                monthly_limit = config.MAX_NIGHT_JOBS_MONTHLY * (config.PLANNING_HORIZON_DAYS / 30.0)
                model += pulp.lpSum([x[(j, p)] for j in night_jobs]) <= monthly_limit
        
        # 8. Fairness constraints
        for p in self.pilots:
            model += (total_hours[p] - self.target_hours == 
                     dev_plus[p] - dev_minus[p])
        
        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=True)
        status = model.solve(solver)
        
        print(f"Solver status: {pulp.LpStatus[status]}")
        
        if status != pulp.LpStatusOptimal:
            return None
        
        # Extract solution
        assignments = []
        for j in self.jobs:
            for p in self.pilots:
                if x[(j, p)].varValue == 1:
                    job_data = self.job_params[j]
                    assignments.append({
                        'job_id': j,
                        'pilot_id': p,
                        'start_time': job_data['start_time'],
                        'end_time': job_data['end_time'],
                        'duration': job_data['duration'],
                        'day': job_data['day'],
                        'is_night': job_data['is_night']
                    })
                    break
        
        pilot_hours = {p: total_hours[p].varValue for p in self.pilots}
        
        return {
            'assignments': pd.DataFrame(assignments),
            'pilot_hours': pilot_hours,
            'objective_value': pulp.value(model.objective),
            'target_hours': self.target_hours
        }
    
    def _check_time_overlap(self, job_i: int, job_j: int) -> bool:
        """Check if two jobs overlap in time."""
        job_i_data = self.jobs_df.iloc[job_i]
        job_j_data = self.jobs_df.iloc[job_j]
        
        # Calculate end times
        end_time_i = job_i_data['start_hour'] + job_i_data['duration']
        end_time_j = job_j_data['start_hour'] + job_j_data['duration']
        
        # Check if they're on the same day and overlap
        if job_i_data['day'] == job_j_data['day']:
            return (job_i_data['start_hour'] < end_time_j and 
                    job_j_data['start_hour'] < end_time_i)
        
        # Check cross-day overlap
        if abs(job_i_data['day'] - job_j_data['day']) == 1:
            if job_i_data['day'] < job_j_data['day']:
                # job_i ends on day X, job_j starts on day X+1
                return end_time_i > 24 and (end_time_i - 24) > job_j_data['start_hour']
            else:
                # job_j ends on day X, job_i starts on day X+1
                return end_time_j > 24 and (end_time_j - 24) > job_i_data['start_hour']
        
        return False
    
    def _check_rest_conflict(self, job_i: int, job_j: int) -> bool:
        """Check if job_j violates rest after job_i."""
        job_i_data = self.jobs_df.iloc[job_i]
        job_j_data = self.jobs_df.iloc[job_j]
        
        # Calculate job_i end time in absolute hours
        job_i_end = job_i_data['day'] * 24 + job_i_data['start_hour'] + job_i_data['duration']
        
        # Calculate job_j start time in absolute hours
        job_j_start = job_j_data['day'] * 24 + job_j_data['start_hour']
        
        # Check if there's enough rest time between jobs
        rest_time = job_j_start - job_i_end
        return rest_time < config.MIN_REST_HOURS