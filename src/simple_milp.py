"""
Simplified MILP model with relaxed constraints for feasibility.
"""

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import config


class SimplifiedPilotScheduling:
    """Simplified MILP model for pilot scheduling."""
    
    def __init__(self, jobs_df: pd.DataFrame, pilots: List[str] = None):
        self.jobs_df = jobs_df.copy()
        self.pilots = pilots or config.PILOT_IDS[:10]  # Use fewer pilots
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
        
        # Only compute critical overlaps (same time exactly)
        self.overlapping_pairs = self._compute_critical_overlaps()
        
    def _compute_critical_overlaps(self):
        """Compute only direct time overlaps."""
        overlapping = set()
        for i, job1 in self.jobs_df.iterrows():
            for j, job2 in self.jobs_df.iterrows():
                if i < j and abs(job1['start_time'] - job2['start_time']) < 1.0:
                    overlapping.add((job1['job_id'], job2['job_id']))
        return overlapping
    
    def solve(self) -> Optional[Dict]:
        """Solve the simplified model."""
        model = pulp.LpProblem("SimplifiedPilotScheduling", pulp.LpMinimize)
        
        # Decision variables
        x = pulp.LpVariable.dicts("assign", 
                                 [(j, p) for j in self.jobs for p in self.pilots],
                                 cat='Binary')
        
        # Objective: minimize total cost
        model += pulp.lpSum([
            (2.0 if self.job_params[j]['is_night'] else 1.0) * x[(j, p)]
            for j in self.jobs for p in self.pilots
        ])
        
        # Constraint 1: Each job assigned to exactly one pilot
        for j in self.jobs:
            model += pulp.lpSum([x[(j, p)] for p in self.pilots]) == 1
        
        # Constraint 2: No direct overlaps
        for (j1, j2) in self.overlapping_pairs:
            for p in self.pilots:
                model += x[(j1, p)] + x[(j2, p)] <= 1
        
        # Constraint 3: Simple daily limits (relaxed)
        for p in self.pilots:
            for day in range(max(1, config.PLANNING_HORIZON_DAYS)):
                day_jobs = [j for j in self.jobs if self.job_params[j]['day'] == day]
                if day_jobs:
                    model += pulp.lpSum([
                        self.job_params[j]['duration'] * x[(j, p)]
                        for j in day_jobs
                    ]) <= 16.0  # Very relaxed daily limit
        
        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=60, msg=False)
        status = model.solve(solver)
        
        if status != pulp.LpStatusOptimal:
            return None
        
        # Extract solution
        assignments = []
        for j in self.jobs:
            for p in self.pilots:
                if x[(j, p)].varValue == 1:
                    job_params = self.job_params[j]
                    assignments.append({
                        'job_id': j,
                        'pilot_id': p,
                        'start_time': job_params['start_time'],
                        'end_time': job_params['end_time'],
                        'duration': job_params['duration'],
                        'day': job_params['day'],
                        'is_night': job_params['is_night']
                    })
                    break
        
        pilot_hours = {}
        for p in self.pilots:
            total = sum([a['duration'] for a in assignments if a['pilot_id'] == p])
            pilot_hours[p] = total
        
        return {
            'assignments': pd.DataFrame(assignments),
            'pilot_hours': pilot_hours,
            'objective_value': pulp.value(model.objective)
        }


def solve_simplified_pilot_scheduling(jobs_df: pd.DataFrame, pilots: List[str] = None) -> Dict:
    """Solve using simplified model."""
    solver = SimplifiedPilotScheduling(jobs_df, pilots)
    return solver.solve()