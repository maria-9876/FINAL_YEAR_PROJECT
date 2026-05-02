import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
from pettingzoo import ParallelEnv
import numpy as np
import json
import os
from .disease_dynamics import SEAIRDDynamics

class KeralaCovidEnv(ParallelEnv):
    """
    Multi-Agent Environment for Kerala COVID-19 Resource Allocation.
    Each district is an agent trying to minimize infections, hospital overload, and economic loss.
    """
    metadata = {"render_modes": ["human"], "name": "kerala_covid_v1"}

    def __init__(self, data_path="data/kerala_districts.json", max_steps=100):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        
        # Load district data
        with open(os.path.join(os.path.dirname(__file__), '..', '..', data_path), 'r') as f:
            data = json.load(f)
            
        self.districts = data['districts']
        self.global_params = data['global_parameters']
        self.agents = [d['name'] for d in self.districts]
        self.possible_agents = self.agents[:]
        
        # Mapping district names to indices and data
        self.agent_name_mapping = {name: i for i, name in enumerate(self.agents)}
        self.district_data = {d['name']: d for d in self.districts}
        
        # Initialize internal state dictionaries
        self.state = {agent: {} for agent in self.agents}
        self.dynamics_models = {}
        
        # Define Spaces
        # State: [S, E, A, I, R, D, icu_occupancy, vaccines_avail, compliance_rate, economic_health]
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32) 
            for agent in self.agents
        }
        
        # Action: [lockdown_level(0-3), vaccine_allocation(0-1), icu_request(0-1), staff_modifier(0-1)]
        self.action_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        observations = {}
        infos = {agent: {} for agent in self.agents}
        
        for agent in self.agents:
            pop = self.district_data[agent]['population']
            
            # Initial seed of infections
            initial_infections = int(pop * 0.0001)  # 0.01% initially infected
            
            self.state[agent] = {
                'S': pop - initial_infections,
                'E': initial_infections * 0.5,
                'A': initial_infections * 0.2,
                'I': initial_infections * 0.3,
                'R': 0,
                'D': 0,
                'icu_occupancy': 0,
                'vaccines_avail': 1000, # Initial vaccine stock
                'compliance_rate': self.district_data[agent]['compliance_rate'],
                'economic_health': 1.0
            }
            
            self.dynamics_models[agent] = SEAIRDDynamics(pop)
            observations[agent] = self._get_obs(agent)
            
        return observations, infos

    def step(self, actions):
        """
        Executes one step in the environment.
        """
        self.current_step += 1
        
        observations = {}
        rewards = {}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # First pass: compute new states based on actions
        for agent, action in actions.items():
            # Action decoding
            lockdown_idx = int(action[0] * 3.99) # 0 to 3
            vaccine_alloc = action[1]
            icu_request = action[2]
            staff_mod = action[3]
            
            # Map lockdown to transmission and economy modifiers
            lockdown_key = f"level_{lockdown_idx}"
            lockdown_fx = self.global_params['lockdown_effects'][lockdown_key]
            
            action_modifiers = {
                'transmission_modifier': lockdown_fx['transmission_modifier'],
                'vaccines_administered': int(vaccine_alloc * self.state[agent]['vaccines_avail']),
                'icu_capacity': self.district_data[agent]['icu_capacity'] * (1.0 + staff_mod * 0.5),
                'compliance_rate': self.state[agent]['compliance_rate']
            }
            
            # Step the SEAIRD model
            new_epi_state = self.dynamics_models[agent].step(self.state[agent], action_modifiers)
            
            # Update economic health based on lockdown
            economic_loss = lockdown_fx['economic_impact'] * self.district_data[agent]['economic_weight']
            new_economic_health = max(0.0, self.state[agent]['economic_health'] - economic_loss * 0.01)
            
            # Merge state
            self.state[agent].update(new_epi_state)
            self.state[agent]['economic_health'] = new_economic_health
            self.state[agent]['icu_occupancy'] = min(new_epi_state['I'] * 0.10, action_modifiers['icu_capacity'])
            
            observations[agent] = self._get_obs(agent)
            
            # Formulate reward (Placeholder logic, will be expanded in utils/reward_functions.py)
            new_infections = new_epi_state['E'] # Approximation
            hospital_overflow = max(0, new_epi_state['I'] * 0.10 - action_modifiers['icu_capacity'])
            
            # Penalize: infections, hospital overflow, and economic drop
            rewards[agent] = - (new_infections * 0.01) - (hospital_overflow * 1.0) - ((1.0 - new_economic_health) * 100)

        # Truncate if max steps reached
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
                
        # To handle multi-agent inter-district travel, a second pass would transfer a % of S, E, A, I 
        # across graph neighbors here.
        self._apply_travel_dynamics()

        if all(truncations.values()) or all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _apply_travel_dynamics(self):
        """
        Simulates inter-district travel of exposed and asymptomatic individuals.
        """
        base_travel = self.global_params['travel_probability_base']
        transfers = {agent: {'E': 0, 'A': 0} for agent in self.agents}
        
        for agent in self.agents:
            neighbors = self.district_data[agent]['neighbors']
            for neighbor in neighbors:
                if neighbor in self.agents:
                    # 1% of travel prob moves exposed/asymptomatic
                    e_move = self.state[agent]['E'] * base_travel * 0.01
                    a_move = self.state[agent]['A'] * base_travel * 0.01
                    
                    self.state[agent]['E'] -= e_move
                    self.state[agent]['A'] -= a_move
                    
                    transfers[neighbor]['E'] += e_move
                    transfers[neighbor]['A'] += a_move
                    
        for agent in self.agents:
            self.state[agent]['E'] += transfers[agent]['E']
            self.state[agent]['A'] += transfers[agent]['A']

    def _get_obs(self, agent):
        """
        Normalizes state values to [0, 1] for neural network input.
        """
        pop = self.district_data[agent]['population']
        s = self.state[agent]
        return np.array([
            s['S'] / pop,
            s['E'] / pop,
            s['A'] / pop,
            s['I'] / pop,
            s['R'] / pop,
            s['D'] / pop,
            s['icu_occupancy'] / self.district_data[agent]['icu_capacity'],
            s['vaccines_avail'] / 100000.0, # Arbitrary scaling factor for max vaccines
            s['compliance_rate'],
            s['economic_health']
        ], dtype=np.float32)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
