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
        """
        agents = list(ep_states.keys())
        seq_len = len(ep_states[agents[0]])
        
        # Scale rewards
        scaled_rewards = {}
        for a in agents:
            scaled_rewards[a] = np.array(ep_rewards[a]) * 1e-4
            
        h_states = {a: None for a in agents}
        total_actor_loss = 0
        total_critic_loss = 0
        
        # Accumulate losses over the episode
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        for t in range(seq_len):
            # Gather current step data across all agents
            obs_dict = {a: torch.FloatTensor(ep_states[a][t]).unsqueeze(0) for a in agents}
            act_dict = {a: torch.FloatTensor(ep_actions[a][t]).unsqueeze(0) for a in agents}
            
            for agent in agents:
                obs = obs_dict[agent]
                action = act_dict[agent]
                reward = torch.tensor([scaled_rewards[agent][t]], dtype=torch.float32)
                
                other_obs = [obs_dict[a] for a in agents if a != agent]
                other_acts = [act_dict[a] for a in agents if a != agent]
                
                # --- Critic Loss ---
                current_q = self.critic(obs, action, other_obs, other_acts)
                
                with torch.no_grad():
                    if t < seq_len - 1:
                        next_obs = torch.FloatTensor(ep_states[agent][t+1]).unsqueeze(0)
                        other_next_obs = [torch.FloatTensor(ep_states[a][t+1]).unsqueeze(0) for a in agents if a != agent]
                        
                        # Use target policy for SAC targets
                        next_action, next_logprob, _ = self.actor.get_action(next_obs, h_states[agent])
                        
                        other_next_acts = []
                        for idx, a in enumerate(agents):
                            if a != agent:
                                n_a, _, _ = self.actor.get_action(other_next_obs[len(other_next_acts)], h_states[a])
                                other_next_acts.append(n_a)
                                
                        target_q = self.target_critic(next_obs, next_action, other_next_obs, other_next_acts)
                        y = reward + self.gamma * (target_q.squeeze() - self.alpha * next_logprob.squeeze())
                    else:
                        y = reward
                        
                critic_loss = nn.functional.mse_loss(current_q.squeeze(), y.squeeze())
                (critic_loss / (seq_len * len(agents))).backward()
                total_critic_loss += critic_loss.item()
                
                # --- Actor Loss ---
                curr_act, curr_logprob, h_next = self.actor.get_action(obs, h_states[agent])
                q_val = self.critic(obs, curr_act, other_obs, other_acts)
                actor_loss = (self.alpha * curr_logprob - q_val).mean()
                
                (actor_loss / (seq_len * len(agents))).backward()
                total_actor_loss += actor_loss.item()
                
                h_states[agent] = h_next.detach()
                
        # Perform optimization once per episode
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        
        # Soft update target network
        tau = 0.005
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        return total_actor_loss / (seq_len * len(agents))
