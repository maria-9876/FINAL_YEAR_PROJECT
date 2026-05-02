class RewardCalculator:
    """
    Computes the multi-objective reward function for the RL agent.
    Goal: Minimize fatalities & infections, prevent hospital overflow, 
          and minimize economic disruption.
    """
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.5):
        """
        Args:
            alpha (float): Scaling factor penalty for new infections/deaths.
            beta (float): Exponential/linear penalty for exceeding hospital limits.
            gamma (float): Penalty for severe economic lockdowns.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_reward(self, old_state, new_state, action_modifiers):
        """
        Calculate step reward.
        """
        # 1. Health Penalty
        new_infections = max(0, new_state['E'] - old_state['E'])
        new_deaths = max(0, new_state['D'] - old_state['D'])
        
        # Heavy weight on actual deaths over just infections
        health_penalty = self.alpha * (new_infections * 0.1 + new_deaths * 10.0)
        
        # 2. Healthcare Overload Penalty
        icu_demand = new_state['I'] * 0.10 # Approximation of critical cases
        icu_capacity = action_modifiers['icu_capacity']
        
        if icu_demand > icu_capacity:
            # Non-linear penalty if hospitals collapse
            overflow = icu_demand - icu_capacity
            hospital_penalty = self.beta * (overflow ** 1.5)
        else:
            hospital_penalty = 0.0
            
        # 3. Economic Penalty
        # Action modifiers lower economic health depending on lockdown severity
        economic_health_loss = max(0, old_state['economic_health'] - new_state['economic_health'])
        economic_penalty = self.gamma * (economic_health_loss * 500.0) # Scaled appropriately
        
        # Total Reward is negative since we are minimizing these bad outcomes
        total_reward = -(health_penalty + hospital_penalty + economic_penalty)
        
        return total_reward, {
            'health_penalty': health_penalty,
            'hospital_penalty': hospital_penalty,
            'economic_penalty': economic_penalty
        }
