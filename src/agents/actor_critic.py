import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAgentActorCritic(nn.Module):
    """
    Centralized Training, Decentralized Execution (CTDE) Actor-Critic Network.
    Each agent shares these parameters (or can have independent ones in a wrapper).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initializes the shared parameter networks.
        
        Args:
            state_dim (int): Length of local observation vector (10).
            action_dim (int): Length of action vector (4 continuous variables).
            hidden_dim (int): Width of hidden layers.
        """
        super(MultiAgentActorCritic, self).__init__()
        
        # ACTOR: Evaluates solely the local state space
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        
        # CRITIC: In MAPPO/MADDPG, the critic sees the *global* state (concat of all agent states)
        # Here we define the critic for local state as a base skeleton. 
        # A full multi-agent controller overrides this with `state_dim * num_agents`.
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Pass a state vector to extract actions and values.
        Returns:
            action_mean (tensor): Bounded continuous actions [0, 1].
            state_value (tensor): Expected reward estimation.
        """
        # Actor Pathway
        x_a = F.relu(self.actor_fc1(state))
        x_a = F.relu(self.actor_fc2(x_a))
        # Use sigmoid to bound outputs between 0 and 1 for Gym spaces
        action_mean = torch.sigmoid(self.actor_out(x_a))
        
        # Critic Pathway
        x_c = F.relu(self.critic_fc1(state))
        x_c = F.relu(self.critic_fc2(x_c))
        state_value = self.critic_value(x_c)
        
        return action_mean, state_value

    def get_action_and_value(self, state, action=None, deterministic=False):
        """
        Samples an action probabilistically during training.
        """
        action_mean, value = self.forward(state)
        # Assuming a Gaussian policy for continuous action space exploration
        cov_var = torch.full(size=(action_mean.shape[-1],), fill_value=0.1)
        cov_mat = torch.diag(cov_var).to(state.device)
        
        # Multivariate Normal Distribution
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
            
        # Clamp action to valid space [0, 1] bounds
        action = torch.clamp(action, 0.0, 1.0)
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_logprob, entropy, value
