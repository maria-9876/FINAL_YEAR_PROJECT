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
        Updates the MARAAC network using Proximal Policy Optimization (PPO) epochs.
        This drastically speeds up learning from small episode counts.
        """
        agents = list(ep_states.keys())
        seq_len = len(ep_states[agents[0]])
        
        # Scale rewards
        scaled_rewards = {}
        for a in agents:
            scaled_rewards[a] = np.array(ep_rewards[a]) * 1e-4
            
        # 1. First Pass: Collect Old Logprobs and Q-values for Advantages
        old_logprobs = []
        q_values = []
        h_states = {a: None for a in agents}
        
        with torch.no_grad():
            for t in range(seq_len):
                obs_dict = {a: torch.FloatTensor(ep_states[a][t]).unsqueeze(0) for a in agents}
                act_dict = {a: torch.FloatTensor(ep_actions[a][t]).unsqueeze(0) for a in agents}
                
                for agent in agents:
                    obs = obs_dict[agent]
                    action = act_dict[agent]
                    
                    other_obs = [obs_dict[a] for a in agents if a != agent]
                    other_acts = [act_dict[a] for a in agents if a != agent]
                    
                    current_q = self.critic(obs, action, other_obs, other_acts)
                    q_values.append(current_q.squeeze())
                    
                    action_logprob, _, h_next = self.actor.evaluate_action(obs, action, h_states[agent])
                    old_logprobs.append(action_logprob)
                    h_states[agent] = h_next
                    
        old_logprobs_tensor = torch.stack(old_logprobs).detach()
        q_tensor = torch.stack(q_values).detach()
        
        # Baseline = mean Q value. Advantage = Q - Baseline
        baseline = q_tensor.mean()
        advantages = q_tensor - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 2. PPO Epochs
        k_epochs = 4
        eps_clip = 0.2
        
        for epoch in range(k_epochs):
            critic_losses = []
            actor_logprobs = []
            actor_entropies = []
            h_states = {a: None for a in agents}
            
            for t in range(seq_len):
                obs_dict = {a: torch.FloatTensor(ep_states[a][t]).unsqueeze(0) for a in agents}
                act_dict = {a: torch.FloatTensor(ep_actions[a][t]).unsqueeze(0) for a in agents}
                
                for agent in agents:
                    obs = obs_dict[agent]
                    action = act_dict[agent]
                    reward = torch.tensor([scaled_rewards[agent][t]], dtype=torch.float32)
                    
                    other_obs = [obs_dict[a] for a in agents if a != agent]
                    other_acts = [act_dict[a] for a in agents if a != agent]
                    
                    # --- Critic ---
                    current_q = self.critic(obs, action, other_obs, other_acts)
                    
                    with torch.no_grad():
                        if t < seq_len - 1:
                            next_obs = torch.FloatTensor(ep_states[agent][t+1]).unsqueeze(0)
                            other_next_obs = [torch.FloatTensor(ep_states[a][t+1]).unsqueeze(0) for a in agents if a != agent]
                            
                            next_action = torch.FloatTensor(ep_actions[agent][t+1]).unsqueeze(0)
                            
                            other_next_acts = []
                            for idx, a in enumerate(agents):
                                if a != agent:
                                    n_a = torch.FloatTensor(ep_actions[a][t+1]).unsqueeze(0)
                                    other_next_acts.append(n_a)
                                    
                            target_q = self.target_critic(next_obs, next_action, other_next_obs, other_next_acts)
                            y = reward + self.gamma * target_q.squeeze()
                        else:
                            y = reward
                            
                    critic_loss = nn.functional.mse_loss(current_q.squeeze(), y.squeeze())
                    critic_losses.append(critic_loss)
                    
                    # --- Actor ---
                    action_logprob, entropy, h_next = self.actor.evaluate_action(obs, action, h_states[agent])
                    actor_logprobs.append(action_logprob)
                    actor_entropies.append(entropy)
                    h_states[agent] = h_next
                    
            # --- Backpropagate ---
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            # Critic Loss
            total_critic_loss = torch.stack(critic_losses).mean()
            total_critic_loss.backward()
            
            # Actor Loss (PPO Clipped Surrogate)
            new_logprobs_tensor = torch.stack(actor_logprobs)
            entropies_tensor = torch.stack(actor_entropies)
            
            ratios = torch.exp(new_logprobs_tensor - old_logprobs_tensor)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean() - (self.alpha * entropies_tensor).mean()
            actor_loss.backward()
            
            # Gradient Clipping & Step
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            self.actor_optimizer.step()
            
        # Soft update target network
        tau = 0.005
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        return actor_loss.item()
