import yaml
import numpy as np
import pyomo.environ as pyo

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
print(f'EDGE Capacity:',capacities['EDGE'])
# ----------------------------
# BUILD OPTIMIZATION MODEL
# ----------------------------
m = pyo.ConcreteModel(name="CO2_minimization")

# Sets
m.T = pyo.Set(initialize=range(1, PERIODS+1))

# Params
m.demand = pyo.Param(m.T, initialize=demand_dict)

# Variables
m.jobs = pyo.Var(m.T, tiers.keys(), within=pyo.NonNegativeReals)

# Demand constraint
def demand_bal(m, t):
    return sum(m.jobs[t, tier] for tier in tiers) == m.demand[t]
m.demand_constraint = pyo.Constraint(m.T, rule=demand_bal)

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

# Objective: total CO2 (operational + embodied)
def total_CO2(m):
    total = 0
    for t in m.T:
        for tier, params in tiers.items():
            jobs = m.jobs[t, tier]
            energy = params['comp_energy'] * jobs
            grid_fraction = params['grid_fraction'][t-1]
            ci = params['carbon_intensity']
            op_CO2 = energy * grid_fraction * ci
            emb_CO2 = params['embodied_carbon'] * (jobs / capacities[tier])
            total += op_CO2 + emb_CO2
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

print("\nTotal CO2 (gCO2):", pyo.value(m.total_CO2))
print("\nPer-period allocation:")
for t in m.T:
    print(f"t={t:02d} demand={m.demand[t]:7.1f} " +
          " ".join(f"{tier}={m.jobs[t,tier].value:7.1f}" for tier in tiers))
    
import pandas as pd

# Create lists to populate into the DataFrame
records = []

for t in m.T:
    row = {
        "period": t,
        "demand": pyo.value(m.demand[t]),
    }

    total_energy = 0
    total_co2 = 0

    for tier, params in tiers.items():
        jobs = pyo.value(m.jobs[t, tier])
        comp_energy = jobs * params['comp_energy']
        grid_frac = params['grid_fraction'][t - 1]
        op_co2 = comp_energy * grid_frac * params['carbon_intensity']
        emb_co2 = params['embodied_carbon'] * (jobs / capacities[tier])
        idle_co2 = params['idle_power_w'] * params['carbon_intensity']
        
        # Store per-tier values
        row[f"{tier}_jobs"] = jobs
        row[f"{tier}_energy"] = comp_energy
        row[f"{tier}_op_co2"] = op_co2
        row[f"{tier}_emb_co2"] = emb_co2
        row[f"{tier}_idle_co2"] = idle_co2

        total_energy += comp_energy
        total_co2 += op_co2 + emb_co2 + idle_co2

    row["total_energy"] = total_energy
    row["total_co2"] = total_co2

    records.append(row)

# Create the DataFrame
results_df = pd.DataFrame(records)

# Show a preview
print("\n--- Results DataFrame Preview ---")
print(results_df.head())

# Optionally export to CSV
results_df.to_csv("results.csv", index=False)
