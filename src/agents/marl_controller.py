import torch
import torch.nn as nn
import numpy as np
from src.agents.actor_critic import RecurrentActor, AttentionCritic

class MARLController:
    """
    Implements the Multi-Agent Recurrent Attention Actor-Critic (MARAAC) from the paper.
    Uses Parameter Sharing, GRU Actors, and an Attention Critic.
    """
    def __init__(self, agent_names, state_dim, action_dim, lr=3e-4, gamma=0.99, alpha=0.01):
        self.agent_names = agent_names
        self.gamma = gamma
        self.alpha = alpha # Temperature parameter for entropy
        
        # Shared Actor and Critic for all districts
        self.actor = RecurrentActor(state_dim, action_dim)
        self.critic = AttentionCritic(state_dim, action_dim)
        
        # Target Critic for stable Q-learning
        self.target_critic = AttentionCritic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)


    def get_actions(self, observations, hidden_states, deterministic=False):
        actions = {}
        next_hidden_states = {}
        
        for agent, obs in observations.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            h_tensor = hidden_states.get(agent, None)
            if h_tensor is not None:
                h_tensor = torch.FloatTensor(h_tensor).unsqueeze(0)
                
            action, _, h_next = self.actor.get_action(obs_tensor, h_tensor, deterministic=deterministic)
            
            actions[agent] = action.detach().numpy()[0]
            next_hidden_states[agent] = h_next.detach().numpy()[0]
            
        return actions, next_hidden_states


    def update_all(self, ep_states, ep_actions, ep_rewards):
        """
        Updates the MARAAC network using the stored episode trajectories of all agents.
        Satisfies Equation 4 and Equation 8 from the paper.
        """

        agents = list(ep_states.keys())
        seq_len = len(ep_states[agents[0]])
        
        total_loss = 0
        
        # Scale rewards to prevent network gradient explosion
        for a in agents:
            r = np.array(ep_rewards[a])
            r = r * 1e-4
            ep_rewards[a] = r.tolist()
            
        # Initialize hidden states for training
        h_states = {a: None for a in agents}
            
        # Process the trajectory sequentially to preserve GRU time-series dependencies
        for t in range(seq_len):
            # 1. Gather current step data across all agents
            obs_dict = {a: torch.FloatTensor(ep_states[a][t]).unsqueeze(0) for a in agents}
            act_dict = {a: torch.FloatTensor(ep_actions[a][t]).unsqueeze(0) for a in agents}
            reward_dict = {a: torch.tensor([ep_rewards[a][t]], dtype=torch.float32) for a in agents}
            
            for agent in agents:
                obs = obs_dict[agent]
                action = act_dict[agent]
                reward = reward_dict[agent]
                
                other_obs = [obs_dict[a] for a in agents if a != agent]
                other_acts = [act_dict[a] for a in agents if a != agent]
                
                # --- Critic Loss (Equation 8) ---
                current_q = self.critic(obs, action, other_obs, other_acts)
                
                with torch.no_grad():
                    if t < seq_len - 1:
                        next_obs = torch.FloatTensor(ep_states[agent][t+1]).unsqueeze(0)
                        other_next_obs = [torch.FloatTensor(ep_states[a][t+1]).unsqueeze(0) for a in agents if a != agent]
                        
                        # Get next actions and logprobs from actor
                        next_action, next_logprob, _ = self.actor.get_action(next_obs, h_states[agent])
                        
                        other_next_acts = []
                        for a in agents:
                            if a != agent:
                                n_a, _, _ = self.actor.get_action(other_next_obs[len(other_next_acts)], h_states[a])
                                other_next_acts.append(n_a)
                                
                        target_q = self.target_critic(next_obs, next_action, other_next_obs, other_next_acts)
                        y = reward.squeeze() + self.gamma * (target_q.squeeze() - self.alpha * next_logprob.squeeze())
                    else:
                        y = reward.squeeze()
                        
                critic_loss = nn.functional.mse_loss(current_q.squeeze(), y.squeeze())
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # --- Actor Loss (Equation 4 / Soft Actor-Critic PG) ---
                curr_act, curr_logprob, h_next = self.actor.get_action(obs, h_states[agent])
                
                q_val = self.critic(obs, curr_act, other_obs, other_acts)
                # Minimize: alpha * log(pi) - Q(o,a)
                actor_loss = (self.alpha * curr_logprob - q_val).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                h_states[agent] = h_next.detach()
                total_loss += (actor_loss.item() + critic_loss.item())
                
        # Soft update target network (Polyak averaging)
        tau = 0.005
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        return total_loss / (seq_len * len(agents))
