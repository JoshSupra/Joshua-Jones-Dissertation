import yaml
import numpy as np
import pyomo.environ as pyo
import pandas as pd

# ----------------------------
# LOAD CONFIG
# ----------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config("config.yaml")

# Extract global params
PERIODS = config['global']['periods']
PERIOD_SECONDS = config['global']['period_seconds']
TOKENS_PER_JOB = config['global']['tokens_per_job']
TOTAL_DAILY_JOBS = config['global']['total_daily_jobs']
MIN_SHARE = config['global']['min_share']

tiers = config['tiers']
demand_shape = np.array(config['demand_shape'])
demand_shape = demand_shape / demand_shape.sum()  # normalize just in case

# ----------------------------
# CREATE DEMAND PROFILE
# ----------------------------
demand_dict = {t+1: float(TOTAL_DAILY_JOBS * demand_shape[t]) for t in range(PERIODS)}

# ----------------------------
# CAPACITY CALCULATIONS
# ----------------------------
def calc_capacity(tier_params):
    return (
        tier_params['throughput'] *
        tier_params['utilization'] *
        tier_params['parallel_units'] *
        PERIOD_SECONDS
    ) / TOKENS_PER_JOB

capacities = {tier: calc_capacity(params) for tier, params in tiers.items()}
print(f'DC Capacity:',capacities['DC'])
print(f'SDC Capacity:',capacities['SDC'])
print(f'EDGE Zone 1 Capacity:',capacities['EDGE_ZONE1'])
print(f'EDGE Zone 2 Capacity:',capacities['EDGE_ZONE2'])
print(f'EDGE Zone 3 Capacity:',capacities['EDGE_ZONE3'])

# ----------------------------
# BUILD OPTIMIZATION MODEL
# ----------------------------
m = pyo.ConcreteModel(name="CO2_minimization")

# Sets
m.T = pyo.Set(initialize=range(1, PERIODS+1))
m.TIERS = pyo.Set(initialize=tiers.keys())

# Params
m.demand = pyo.Param(m.T, initialize=demand_dict)

# Variables
m.jobs = pyo.Var(m.T, m.TIERS, within=pyo.NonNegativeReals)
m.backlog = pyo.Var(m.T, within=pyo.NonNegativeReals)
#m.energy = pyo.Var(m.T, m.TIERS, within=pyo.NonNegativeReals)

# Demand constraint
def demand_bal(m, t):
    return sum(m.jobs[t, tier] for tier in tiers) == m.demand[t]
m.demand_constraint = pyo.Constraint(m.T, rule=demand_bal)

def backlog_balance_rule(m, t): #passes excess compute workload to next timestep
    if t == m.T.first():
        return m.backlog[t] == m.demand[t] - sum(m.jobs[t, tier] for tier in m.TIERS)
    else:
        return m.backlog[t] == m.backlog[t - 1] + m.demand[t] - sum(m.jobs[t, tier] for tier in m.TIERS)

m.backlog_balance = pyo.Constraint(m.T, rule=backlog_balance_rule)

m.final_clearance = pyo.Constraint(expr=m.backlog[m.T.last()] == 0) #ensures no jobs are left in backlog at end of run

# Capacity constraints
for tier, cap in capacities.items():
    def cap_rule(m, t, tier=tier, cap=cap):
        return m.jobs[t, tier] <= cap
    setattr(m, f"{tier}_cap_con", pyo.Constraint(m.T, rule=cap_rule))

# Optional minimum share
if MIN_SHARE > 0:
    for tier in tiers:
        def min_share_rule(m, t, tier=tier):
            return m.jobs[t, tier] >= MIN_SHARE * m.demand[t]
        setattr(m, f"{tier}_min_share", pyo.Constraint(m.T, rule=min_share_rule))


# Precompute constants
timestep_hours = PERIOD_SECONDS / 3600.0
job_time_h_per_token = 1 / 3600.0  # seconds to hours per token

# Auxiliary energy expression and constraints
m.energy_active = pyo.Var(m.T, m.TIERS, within=pyo.NonNegativeReals) #Defining the active energy decision variable 

def active_energy_rule(m, t, tier):
    job_time_h = TOKENS_PER_JOB * job_time_h_per_token / tiers[tier]['throughput'] #converts time taken for a job to hours
    return m.energy_active[t, tier] == tiers[tier]['power_draw'] * job_time_h * m.jobs[t, tier] #Constrains the active energy variable to always be equal to this equation

m.active_energy_calc = pyo.Constraint(m.T, m.TIERS, rule=active_energy_rule)

#Defining idle energy and constraining it to be equal to the expression
m.energy_idle = pyo.Var(m.T, m.TIERS, within=pyo.NonNegativeReals)

def idle_energy_rule(m, t, tier):
    job_time = TOKENS_PER_JOB * job_time_h_per_token /tiers[tier]['throughput']
    used_time = m.jobs[t, tier] * job_time
    total_time = tiers[tier]['parallel_units'] * timestep_hours
    return m.energy_idle[t, tier] == (total_time - used_time) * tiers[tier]['idle_power_kw']
    
m.idle_energy_constraint = pyo.Constraint(m.T, m.TIERS, rule=idle_energy_rule)

#Calculating total energy through an expression summing the two components per timestep
def total_energy_expr(m, t, tier):
    return m.energy_active[t, tier] + m.energy_idle[t, tier]

m.energy = pyo.Expression(m.T, m.TIERS, rule=total_energy_expr)

# Objective: total CO2 (operational + embodied + idle)
def total_CO2(m):
    total = 0
    for t in m.T:
        for tier in tiers:
            jobs = m.jobs[t, tier]
            grid_frac = tiers[tier]['grid_fraction'][t - 1]
            local_frac = 1 - grid_frac
            ci_grid = tiers[tier]['carbon_intensity'][t-1]
            ci_local = tiers[tier]['local_carbon_intensity'][t-1]
            op_CO2 = m.energy_active[t, tier] * (grid_frac * ci_grid) + (local_frac * ci_local)
            emb_CO2 = tiers[tier]['embodied_carbon'] * (jobs / capacities[tier]) * tiers[tier]['parallel_units']
            idle_co2 = m.energy_idle[t, tier] * (grid_frac * ci_grid) + (local_frac * ci_local)
            total += op_CO2 + emb_CO2 + idle_co2
    return total
m.total_CO2 = pyo.Expression(rule=total_CO2)
m.obj = pyo.Objective(expr=m.total_CO2, sense=pyo.minimize)

# ----------------------------
# SOLVE
# ----------------------------
solver = pyo.SolverFactory("glpk")
result = solver.solve(m, tee=False)

# ----------------------------
# REPORT
# ----------------------------
print("\nSolve Status:", result.solver.status)
print("Termination  :", result.solver.termination_condition)
print("\nTotal CO2 (kgCO2):", pyo.value(m.total_CO2))

print("\nPer-period allocation:")
for t in m.T:
    print(f"t={t:02d} demand={m.demand[t]:7.1f} " +
          " ".join(f"{tier}={m.jobs[t,tier].value:7.1f}" for tier in tiers))

# Create DataFrame (Note values are being calculated in post using the indexed outputs of the model anddoing the same calculations)
records = []
for t in m.T:
    row = {"period": t, "demand": pyo.value(m.demand[t])}
    total_energy = 0
    total_co2 = 0
    for tier in tiers:
        jobs = pyo.value(m.jobs[t, tier])
        grid_frac = tiers[tier]['grid_fraction'][t - 1]
        local_frac = 1 - grid_frac
        ci_grid = tiers[tier]['carbon_intensity'][t-1]
        energy = pyo.value(m.energy[t, tier])
        op_co2 = pyo.value(m.energy_active[t, tier]) * (grid_frac * ci_grid) + (local_frac * ci_local)
        emb_co2 = tiers[tier]['embodied_carbon'] * (jobs / capacities[tier]) * tiers[tier]['parallel_units']
        idle_co2 =pyo.value( m.energy_idle[t, tier]) * (grid_frac * ci_grid) + (local_frac * ci_local)
        row[f"{tier}_jobs"] = jobs
        row[f"{tier}_idle_energy"] = pyo.value(m.energy_idle[t, tier])
        row[f"{tier}_active_energy"] = pyo.value(m.energy_active[t, tier])
        row[f"{tier}_energy"] = energy
        row[f"{tier}_op_co2"] = op_co2
        row[f"{tier}_emb_co2"] = emb_co2
        row[f"{tier}_idle_co2"] = idle_co2
        total_co2 += op_co2 + emb_co2 + idle_co2
    row["total_energy"] = total_energy
    row["total_co2"] = total_co2
    records.append(row)

results_df = pd.DataFrame(records)
print("\n--- Results DataFrame Preview ---")
print(results_df.head())
results_df.to_csv("results.csv", index=False)
