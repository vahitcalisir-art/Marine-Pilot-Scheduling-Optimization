"""
Simplified MILP scheduler for marine pilot scheduling.
Focuses on essential constraints to ensure feasibility.
"""

import pandas as pd
import pulp
import config
from typing import Dict, Optional


class SimplePilotScheduler:
    """Simplified MILP-based pilot scheduler."""
    
    def __init__(self, jobs_df: pd.DataFrame):
        self.jobs_df = jobs_df.copy()
        
        # Add start_hour column (hour within the day)
        self.jobs_df['start_hour'] = self.jobs_df['start_time'] % 24
        
        # Sort by day and start hour
        self.jobs_df = self.jobs_df.sort_values(['day', 'start_hour']).reset_index(drop=True)
        
        self.jobs = list(range(len(jobs_df)))
        self.pilots = config.PILOT_IDS
        
        # Pre-calculate job parameters
        self.job_params = {}
        for idx, row in self.jobs_df.iterrows():
            self.job_params[idx] = {
                'start_hour': row['start_hour'],
                'duration': row['duration'],
                'day': row['day'],
                'is_night': row['is_night'],
                'start_time': row['start_time'],
                'end_time': row['end_time']
            }
        
        # Calculate target hours per pilot for fairness
        total_work_hours = self.jobs_df['duration'].sum()
        self.target_hours = total_work_hours / len(self.pilots)
    
    def solve(self, time_limit: int = 180) -> Optional[Dict]:
        """Solve the simplified optimization problem."""
        print(f"Optimizing schedule for {len(self.jobs)} jobs with {len(self.pilots)} pilots...")
        print(f"Target hours per pilot: {self.target_hours:.1f}")
        
        # Create optimization model
        model = pulp.LpProblem("SimplePilotScheduling", pulp.LpMinimize)
        
        # Decision variables
        x = pulp.LpVariable.dicts("assign", 
                                 [(j, p) for j in self.jobs for p in self.pilots],
                                 cat='Binary')
        
        # Total hours per pilot
        total_hours = pulp.LpVariable.dicts("hours", self.pilots, lowBound=0)
        
        # Deviation variables for fairness
        dev_plus = pulp.LpVariable.dicts("dev_plus", self.pilots, lowBound=0)
        dev_minus = pulp.LpVariable.dicts("dev_minus", self.pilots, lowBound=0)
        
        # Objective: minimize unfairness with small night work penalty
        fairness_penalty = pulp.lpSum([dev_plus[p] + dev_minus[p] for p in self.pilots])
        night_penalty = pulp.lpSum([
            x[(j, p)] for j in self.jobs for p in self.pilots
            if self.job_params[j]['is_night']
        ])
        
        model += 100.0 * fairness_penalty + 0.1 * night_penalty
        
        # CONSTRAINTS
        
        # 1. Each job assigned to exactly one pilot
        for j in self.jobs:
            model += pulp.lpSum([x[(j, p)] for p in self.pilots]) == 1
        
        # 2. Daily hour limits (essential constraint)
        for p in self.pilots:
            for day in range(config.PLANNING_HORIZON_DAYS):
                day_jobs = [j for j in self.jobs if self.job_params[j]['day'] == day]
                if day_jobs:
                    model += pulp.lpSum([
                        self.job_params[j]['duration'] * x[(j, p)]
                        for j in day_jobs
                    ]) <= config.MAX_DAILY_HOURS
        
        # 3. Simple overlap prevention for same day
        for p in self.pilots:
            for day in range(config.PLANNING_HORIZON_DAYS):
                day_jobs = [j for j in self.jobs if self.job_params[j]['day'] == day]
                
                # For each pair of jobs on the same day, prevent overlap
                for i, j1 in enumerate(day_jobs):
                    for j2 in day_jobs[i+1:]:
                        job1_start = self.job_params[j1]['start_hour']
                        job1_end = job1_start + self.job_params[j1]['duration']
                        job2_start = self.job_params[j2]['start_hour']
                        job2_end = job2_start + self.job_params[j2]['duration']
                        
                        # Check if jobs overlap
                        if (job1_start < job2_end and job2_start < job1_end):
                            model += x[(j1, p)] + x[(j2, p)] <= 1
        
        # 4. Total hours calculation
        for p in self.pilots:
            model += total_hours[p] == pulp.lpSum([
                self.job_params[j]['duration'] * x[(j, p)] for j in self.jobs
            ])
        
        # 5. Weekly hour limits (relaxed)
        weekly_limit = config.MAX_WEEKLY_HOURS * (config.PLANNING_HORIZON_DAYS / 7.0)
        for p in self.pilots:
            model += total_hours[p] <= weekly_limit * 1.1  # 10% relaxation
        
        # 6. Fairness constraints
        for p in self.pilots:
            model += (total_hours[p] - self.target_hours == 
                     dev_plus[p] - dev_minus[p])
        
        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
        status = model.solve(solver)
        
        print(f"Solver status: {pulp.LpStatus[status]}")
        
        if status != pulp.LpStatusOptimal:
            print("Optimization failed - trying relaxed version...")
            return self._solve_relaxed()
        
        # Extract solution
        assignments = []
        for j in self.jobs:
            for p in self.pilots:
                if x[(j, p)].varValue and x[(j, p)].varValue > 0.5:
                    job_data = self.job_params[j]
                    assignments.append({
                        'job_id': j,
                        'pilot_id': p,
                        'start_time': job_data['start_time'],
                        'end_time': job_data['end_time'],
                        'duration': job_data['duration'],
                        'day': job_data['day'],
                        'start_hour': job_data['start_hour'],
                        'is_night': job_data['is_night']
                    })
                    break
        
        pilot_hours = {p: total_hours[p].varValue for p in self.pilots}
        
        print(f"Successfully assigned {len(assignments)}/{len(self.jobs)} jobs")
        print(f"Pilot hours range: {min(pilot_hours.values()):.1f} - {max(pilot_hours.values()):.1f}")
        
        return {
            'assignments': pd.DataFrame(assignments),
            'pilot_hours': pilot_hours,
            'objective_value': pulp.value(model.objective),
            'target_hours': self.target_hours
        }
    
    def _solve_relaxed(self) -> Optional[Dict]:
        """Solve with further relaxed constraints."""
        print("Solving with maximum relaxation...")
        
        model = pulp.LpProblem("RelaxedPilotScheduling", pulp.LpMinimize)
        
        # Decision variables
        x = pulp.LpVariable.dicts("assign", 
                                 [(j, p) for j in self.jobs for p in self.pilots],
                                 cat='Binary')
        
        # Objective: just minimize night work
        model += pulp.lpSum([
            x[(j, p)] for j in self.jobs for p in self.pilots
            if self.job_params[j]['is_night']
        ])
        
        # Only essential constraints
        
        # 1. Each job assigned to exactly one pilot
        for j in self.jobs:
            model += pulp.lpSum([x[(j, p)] for p in self.pilots]) == 1
        
        # 2. Very relaxed daily limits
        for p in self.pilots:
            for day in range(config.PLANNING_HORIZON_DAYS):
                day_jobs = [j for j in self.jobs if self.job_params[j]['day'] == day]
                if day_jobs:
                    model += pulp.lpSum([
                        self.job_params[j]['duration'] * x[(j, p)]
                        for j in day_jobs
                    ]) <= config.MAX_DAILY_HOURS * 1.5  # 50% relaxation
        
        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=60, msg=False)
        status = model.solve(solver)
        
        print(f"Relaxed solver status: {pulp.LpStatus[status]}")
        
        if status != pulp.LpStatusOptimal:
            return None
        
        # Extract solution
        assignments = []
        for j in self.jobs:
            for p in self.pilots:
                if x[(j, p)].varValue and x[(j, p)].varValue > 0.5:
                    job_data = self.job_params[j]
                    assignments.append({
                        'job_id': j,
                        'pilot_id': p,
                        'start_time': job_data['start_time'],
                        'end_time': job_data['end_time'],
                        'duration': job_data['duration'],
                        'day': job_data['day'],
                        'start_hour': job_data['start_hour'],
                        'is_night': job_data['is_night']
                    })
                    break
        
        # Calculate pilot hours
        pilot_hours = {}
        assignments_df = pd.DataFrame(assignments)
        for p in self.pilots:
            pilot_assignments = assignments_df[assignments_df['pilot_id'] == p]
            pilot_hours[p] = pilot_assignments['duration'].sum()
        
        print(f"Successfully assigned {len(assignments)}/{len(self.jobs)} jobs (relaxed)")
        
        return {
            'assignments': assignments_df,
            'pilot_hours': pilot_hours,
            'objective_value': pulp.value(model.objective),
            'target_hours': self.target_hours
        }