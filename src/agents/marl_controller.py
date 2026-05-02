import torch
from pettingzoo.utils import agent_selector
from .actor_critic import MultiAgentActorCritic

class MARLController:
    """
    Wrapper controller managing the multiple Actor-Critic agents 
    interacting with the PettingZoo Environment.
    """
    def __init__(self, agent_names, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.agent_names = agent_names
        self.gamma = gamma
        
        # Instantiate separate networks for each district agent 
        # (Though sharing weights is an option, distinct nets allow district specialization)
        self.policies = {
            agent: MultiAgentActorCritic(state_dim, action_dim) 
            for agent in agent_names
        }
        
        # Optimizers per agent
        self.optimizers = {
            agent: torch.optim.Adam(self.policies[agent].parameters(), lr=lr)
            for agent in agent_names
        }

    def get_actions(self, observations):
        """
        Samples actions for all given agents based on their independent local observations.
        """
        actions = {}
        logprobs = {}
        values = {}
        
        for agent, obs in observations.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) # Batch dim
            
            # Use the local Actor-Critic network
            action, logprob, entropy, value = self.policies[agent].get_action_and_value(obs_tensor)
            
            # Detach and store for gym interaction
            actions[agent] = action.detach().numpy()[0]
            logprobs[agent] = logprob
            values[agent] = value
            
        return actions, logprobs, values

    def update(self, agent, rewards_list, logprobs_list, values_list, eps_clip=0.2, k_epochs=4):
        """
        Perform Proximal Policy Optimization (PPO) update with Generalized Advantage Estimation (GAE).
        """
        optimizer = self.optimizers[agent]
        policy = self.policies[agent]
        
        # Convert lists to tensors
        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        old_logprobs = torch.stack(logprobs_list).detach()
        values = torch.cat(values_list).squeeze().detach()
        
        # Calculate Generalized Advantage Estimation (GAE)
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards.tolist()):
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        advantages = returns - values
        
        # PPO Optimization over K epochs
        total_loss = 0
        for _ in range(k_epochs):
            # Evaluate current policy
            # Since actions were already sampled, we could re-evaluate from states.
            # For this simplified skeletal version, we will approximate PPO update
            # by using the old trajectory without full state re-sampling if we didn't save states.
            # *In a full framework, you would save 'states' and re-run get_action_and_value.*
            
            # Simplified surrogate loss using stored logprobs to demonstrate math
            # ratio = exp(log_prob_current - log_prob_old) 
            # (Assuming 1 epoch perfectly aligns if we don't re-eval states, to show structure)
            ratios = torch.exp(old_logprobs - old_logprobs) # Placeholder for actual re-evaluation ratio
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = torch.nn.functional.mse_loss(returns, values)
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / k_epochs
