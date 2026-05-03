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

    def get_actions(self, observations, deterministic=False):
        """
        Samples actions for all given agents based on their independent local observations.
        """
        actions = {}
        logprobs = {}
        values = {}
        
        for agent, obs in observations.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) # Batch dim
            
            # Use the local Actor-Critic network
            action, logprob, entropy, value = self.policies[agent].get_action_and_value(obs_tensor, deterministic=deterministic)
            
            # Detach and store for gym interaction
            actions[agent] = action.detach().numpy()[0]
            logprobs[agent] = logprob
            values[agent] = value
            
        return actions, logprobs, values

    def update(self, agent, states_list, actions_list, rewards_list, logprobs_list, values_list, eps_clip=0.2, k_epochs=4):
        """
        Perform Proximal Policy Optimization (PPO) update with Generalized Advantage Estimation (GAE).
        """
        import numpy as np
        optimizer = self.optimizers[agent]
        policy = self.policies[agent]
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(states_list))
        actions = torch.FloatTensor(np.array(actions_list))
        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        old_logprobs = torch.stack(logprobs_list).detach()
        values = torch.cat(values_list).squeeze().detach()
        
        # Scale rewards to stabilize Critic network without destroying absolute magnitude
        rewards = rewards * 1e-4
        
        # Calculate returns
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards.tolist()):
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # PPO Optimization over K epochs
        total_loss = 0
        for _ in range(k_epochs):
            # Re-evaluate current policy to get gradients!
            _, curr_logprobs, _, curr_values = policy.get_action_and_value(states, actions)
            
            # ratio = exp(log_prob_current - log_prob_old) 
            ratios = torch.exp(curr_logprobs.squeeze() - old_logprobs.squeeze())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = torch.nn.functional.mse_loss(returns, curr_values.squeeze())
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / k_epochs
