import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCritic(nn.Module):
    """
    Centralized Critic with Attention Mechanism from the MARAAC paper.
    Calculates Q-value using local observation/action and attention over other agents' states and actions.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(AttentionCritic, self).__init__()
        
        # Local Embedding function g_i (encodes state and action)
        self.embedding = nn.Linear(state_dim + action_dim, hidden_dim)
        
        # Attention Weights (Keys, Queries, Values)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Final MLP f_i to calculate Q-value
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action, other_obs_list, other_action_list):
        """
        obs: (batch, state_dim)
        action: (batch, action_dim)
        other_obs_list: list of (batch, state_dim)
        other_action_list: list of (batch, action_dim)
        """
        # 1. Local embedding
        local_in = torch.cat([obs, action], dim=-1)
        e_i = F.relu(self.embedding(local_in)) # (batch, hidden)
        
        # 2. Query
        q_i = self.W_q(e_i) # (batch, hidden)
        
        # 3. Process other agents to compute Keys and Values
        if len(other_obs_list) > 0:
            other_embs = []
            for o_obs, o_act in zip(other_obs_list, other_action_list):
                o_in = torch.cat([o_obs, o_act], dim=-1)
                other_embs.append(F.relu(self.embedding(o_in)))
                
            # Stack to (batch, num_others, hidden)
            other_embs_stacked = torch.stack(other_embs, dim=1)
            
            K_j = self.W_k(other_embs_stacked)
            V_j = self.W_v(other_embs_stacked)
            
            # 4. Attention scores = Q * K^T / sqrt(d)
            scores = torch.bmm(q_i.unsqueeze(1), K_j.transpose(1, 2)) / (K_j.shape[-1] ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            
            # 5. Context vector x_i (Weighted sum of values)
            x_i = torch.bmm(attn_weights, V_j).squeeze(1) # (batch, hidden)
        else:
            x_i = torch.zeros_like(e_i)
            
        # 6. Final Q-value
        combined = torch.cat([e_i, x_i], dim=-1)
        out = F.relu(self.fc1(combined))
        q_value = self.q_out(out)
        
        return q_value


class RecurrentActor(nn.Module):
    """
    Decentralized Actor with GRU from the MARAAC paper.
    Captures temporal dynamics using a hidden state memory.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(RecurrentActor, self).__init__()
        
        # GRU Cell to replace standard FC layer
        self.gru = nn.GRUCell(state_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, hidden_state=None):
        """
        obs: (batch, state_dim)
        hidden_state: (batch, hidden_dim)
        """
        if hidden_state is None:
            hidden_state = torch.zeros(obs.shape[0], self.gru.hidden_size, device=obs.device)
            
        # Step the GRU
        h_next = self.gru(obs, hidden_state)
        
        # Output Action
        x = F.relu(self.fc1(h_next))
        action_mean = torch.sigmoid(self.action_out(x))
        
        return action_mean, h_next
        
    def get_action(self, obs, hidden_state=None, deterministic=False):
        action_mean, h_next = self.forward(obs, hidden_state)
        
        cov_var = torch.full(size=(action_mean.shape[-1],), fill_value=0.1, device=obs.device)
        cov_mat = torch.diag(cov_var)
        
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        
        if deterministic:
            action = action_mean
        else:
            # Use rsample() for the reparameterization trick (crucial for SAC/MARAAC gradient flow)
            action = dist.rsample()
            
        action = torch.clamp(action, 0.0, 1.0)
        action_logprob = dist.log_prob(action)
        
        return action, action_logprob, h_next
        
    def evaluate_action(self, obs, action, hidden_state=None):
        action_mean, h_next = self.forward(obs, hidden_state)
        
        cov_var = torch.full(size=(action_mean.shape[-1],), fill_value=0.1, device=obs.device)
        cov_mat = torch.diag(cov_var)
        
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action_logprob, entropy, h_next
