import numpy as np

class SEAIRDDynamics:
    """
    Mathematical SEAIRD model representing the epidemic dynamics for a single node (district).
    SEAIRD stands for:
        S: Susceptible
        E: Exposed (infected but not yet infectious)
        A: Asymptomatic (infectious but no symptoms)
        I: Infected (symptomatic and infectious)
        R: Recovered
        D: Deceased
    """
    def __init__(self, population, beta=0.3, alpha=0.2, gamma_a=0.1, gamma_i=0.05,
                 mu=0.01, asymptomatic_ratio=0.4):
        """
        Initialize the disease parameters.
        
        Args:
            population (int): Total population of the node.
            beta (float): Base transmission rate.
            alpha (float): Rate of progressing from Exposed to Asymptomatic/Infected (1/incubation_period).
            gamma_a (float): Recovery rate for asymptomatic individuals.
            gamma_i (float): Recovery rate for symptomatic individuals.
            mu (float): Base mortality rate (can increase if hospitals overflow).
            asymptomatic_ratio (float): Fraction of exposed that become asymptomatic.
        """
        self.N = population
        self.beta = beta
        self.alpha = alpha
        self.gamma_a = gamma_a
        self.gamma_i = gamma_i
        self.mu = mu
        self.asymptomatic_ratio = asymptomatic_ratio

    def step(self, state, action_modifiers):
        """
        Compute the next state using Euler method for a discretized time step (e.g., 1 day).
        
        Args:
            state (dict): Current counts of S, E, A, I, R, D.
            action_modifiers (dict): Modifiers applied by RL agent (lockdown_modifier, vaccine_doses, hospital_capacity_modifier).
            
        Returns:
            dict: Next state counts.
        """
        S, E, A, I, R, D = state['S'], state['E'], state['A'], state['I'], state['R'], state['D']
        
        # RL actions influence transmission and mortality
        transmission_mod = action_modifiers.get('transmission_modifier', 1.0)
        vaccine_doses = action_modifiers.get('vaccines_administered', 0)
        icu_capacity = action_modifiers.get('icu_capacity', float('inf'))
        compliance = action_modifiers.get('compliance_rate', 1.0)
        
        # Adjust effective beta based on lockdown and compliance
        effective_beta = self.beta * transmission_mod * (2.0 - compliance)  # If compliance is low, beta is higher
        
        # Compute new infections
        force_of_infection = effective_beta * (I + 0.5 * A) / self.N
        
        # CAP TRANSITIONS to strictly prevent mass/population creation out of thin air
        new_exposures = min(S, force_of_infection * S)
        
        # Vaccines move people directly from Susceptible to Recovered (immune)
        effective_vaccines = min(vaccine_doses, max(0, S - new_exposures))
        
        new_infections = min(E, self.alpha * E)
        new_asymptomatic = new_infections * self.asymptomatic_ratio
        new_symptomatic = new_infections * (1 - self.asymptomatic_ratio)
        
        recovered_a = min(A, self.gamma_a * A)
        recovered_i = min(I, self.gamma_i * I)
        
        icu_demand = I * 0.10
        if icu_demand > icu_capacity:
            effective_mu = self.mu * 3.0
        else:
            effective_mu = self.mu
            
        deaths = min(max(0, I - recovered_i), effective_mu * I)
        
        # Euler integration update
        next_S = S - new_exposures - effective_vaccines
        next_E = E + new_exposures - new_infections
        next_A = A + new_asymptomatic - recovered_a
        next_I = I + new_symptomatic - recovered_i - deaths
        next_R = R + recovered_a + recovered_i + effective_vaccines
        next_D = D + deaths
        
        # Ensure no negative populations
        next_state = {
            'S': max(0, next_S),
            'E': max(0, next_E),
            'A': max(0, next_A),
            'I': max(0, next_I),
            'R': max(0, next_R),
            'D': max(0, next_D)
        }
        
        return next_state
