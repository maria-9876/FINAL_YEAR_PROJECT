import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch

from src.env.seaird_env import KeralaCovidEnv
from src.agents.marl_controller import MARLController

app = FastAPI(title="Kerala COVID Simulation API")

# Reduce PyTorch thread startup overhead for small CPU-bound inference
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Cache a controller instance so we don't re-create neural nets on every HTTP request
_cached_controller = None

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAT_LON_MAP = {
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366},
    "Kollam": {"lat": 8.8932, "lon": 76.6141},
    "Pathanamthitta": {"lat": 9.2648, "lon": 76.7870},
    "Alappuzha": {"lat": 9.4981, "lon": 76.3388},
    "Kottayam": {"lat": 9.5916, "lon": 76.5222},
    "Idukki": {"lat": 9.8500, "lon": 76.9492},
    "Ernakulam": {"lat": 9.9816, "lon": 76.2999},
    "Thrissur": {"lat": 10.5276, "lon": 76.2144},
    "Palakkad": {"lat": 10.7867, "lon": 76.6548},
    "Malappuram": {"lat": 11.0733, "lon": 76.0740},
    "Kozhikode": {"lat": 11.2588, "lon": 75.7804},
    "Wayanad": {"lat": 11.6854, "lon": 76.1320},
    "Kannur": {"lat": 11.8745, "lon": 75.3704},
    "Kasaragod": {"lat": 12.4962, "lon": 74.9869}
}

@app.get("/api/simulate")
def simulate(max_steps: int = 100):
    """
    Runs a simulation of `max_steps` steps and returns the data aligned
    with the React frontend's expectations. `max_steps` can be lowered
    by the frontend to reduce latency.
    """
    env = KeralaCovidEnv(max_steps=max_steps)
    observations, _ = env.reset()

    # Initialize or reuse a cached controller so we don't rebuild networks per request
    global _cached_controller
    state_dim = 10
    action_dim = 4
    if _cached_controller is None or _cached_controller.agent_names != env.agents:
        _cached_controller = MARLController(env.agents, state_dim, action_dim)

    controller = _cached_controller
    
    outbreakCurveData = []
    
    finalDirectives = {}
    logisticsData = {}

    for step in range(max_steps):
        # Deterministic actions
        actions, _, _ = controller.get_actions(observations)
        observations, _, terminations, truncations, _ = env.step(actions)
        
        # Log globals for the curve
        total_inf = sum([env.state[agent]['I'] for agent in env.agents])
        outbreakCurveData.append({
            "day": step,
            "cases": int(float(total_inf))   # safer
        })
        
        # Capture final state information on the last step (or continuously update)
        for agent in env.agents:
            lockdown_idx = int(actions[agent][0] * 3.99)
            infected = int(env.state[agent]['I'])
            pop = env.district_data[agent]['population']
            
            # Form final directives logic
            finalDirectives[agent] = {
                "district": agent,
                "policy": f"Level {lockdown_idx}",
                "cases": infected,
                "pop": pop,
                "lat": LAT_LON_MAP.get(agent, {}).get("lat", 0),
                "lon": LAT_LON_MAP.get(agent, {}).get("lon", 0)
            }
            
            # Logistics logic
            icu_occupancy = float(env.state[agent]['icu_occupancy'])
            icu_capacity = float(env.district_data[agent]['icu_capacity'])
            cap_percent = (icu_occupancy / icu_capacity) * 100 if icu_capacity > 0 else 0
            
            o2_supply_demand = int(icu_capacity - icu_occupancy) # Mock metric for O2 logistics
            
            logisticsData[agent] = {
                "district": agent,
                "cap": round(cap_percent, 1),
                "used": int(icu_occupancy),
                "limit": int(icu_capacity),
                "o2": o2_supply_demand
            }
            
        if all(terminations.values()) or all(truncations.values()):
            break
            
    # List formats for frontend
    finalDirectives_list = list(finalDirectives.values())
    logisticsData_list = list(logisticsData.values())

    return {
        "outbreakCurveData": outbreakCurveData,
        "finalDirectives": finalDirectives_list,
        "logisticsData": logisticsData_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
