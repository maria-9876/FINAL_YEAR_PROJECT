import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import os

from src.env.seaird_env import KeralaCovidEnv
from src.agents.marl_controller import MARLController

def train(num_episodes=10, max_steps=100):
    """
    Skeletal training loop for the Multi-Agent framework.
    """
    print(f"Starting Training for {num_episodes} episodes...")
    env = KeralaCovidEnv(max_steps=max_steps)
    
    # Initialize MARL wrapper (CTDE structure)
    state_dim = 10
    action_dim = 4
    controller = MARLController(agent_names=env.agents, state_dim=state_dim, action_dim=action_dim)
    
    # Track metrics
    history = {
        'total_rewards': [],
        'total_infections': [],
        'total_deaths': []
    }
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        
        # Lists for computing advantages/returns per episode
        ep_rewards = {agent: [] for agent in env.possible_agents}
        ep_logprobs = {agent: [] for agent in env.possible_agents}
        ep_values = {agent: [] for agent in env.possible_agents}
        
        for step in range(max_steps):
            # 1. Get continuous actions from Policy Networks
            actions, logprobs, values = controller.get_actions(observations)
            
            # 2. Step Environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # 3. Store transitions
            for agent in env.agents:
                ep_rewards[agent].append(rewards[agent])
                ep_logprobs[agent].append(logprobs[agent])
                ep_values[agent].append(values[agent])
                episode_reward += rewards[agent]
                
            observations = next_obs
            
            if all(terminations.values()) or all(truncations.values()):
                break
                
        # End of episode policy update
        for agent in env.agents:
            # Perform PPO updates over the stored trajectory memory
            controller.update(
                agent=agent,
                rewards_list=ep_rewards[agent],
                logprobs_list=ep_logprobs[agent],
                values_list=ep_values[agent]
            )
        
        # Logging
        history['total_rewards'].append(episode_reward)
        if (episode + 1) % 2 == 0:
            print(f"Episode [{episode+1}/{num_episodes}] - Total Reward: {episode_reward:.2f}")
            
    print("Training Completed.")
    return controller, history

def evaluate(controller, max_steps=100):
    """
    Evaluates a trained policy without exploration noise and plots results.
    """
    print("Starting Evaluation...")
    env = KeralaCovidEnv(max_steps=max_steps)
    observations, _ = env.reset()
    
    # Track tracking metrics for plotting
    infection_curve = []
    icu_usage = []
    economic_health = []
    lockdown_levels = {agent: [] for agent in env.agents}
    
    for step in range(max_steps):
        # Deterministic actions (no sampling noise)
        actions, _, _ = controller.get_actions(observations)
        observations, _, terminations, truncations, _ = env.step(actions)
        
        # Log globals
        total_inf = sum([env.state[agent]['I'] for agent in env.agents])
        total_icu = sum([env.state[agent]['icu_occupancy'] for agent in env.agents])
        avg_econ = np.mean([env.state[agent]['economic_health'] for agent in env.agents])
        
        infection_curve.append(total_inf)
        icu_usage.append(total_icu)
        economic_health.append(avg_econ)
        
        # Log specifics
        for agent in env.agents:
            lockdown_idx = int(actions[agent][0] * 3.99)
            lockdown_levels[agent].append(lockdown_idx)
            
        if all(terminations.values()) or all(truncations.values()):
            break
            
    # Simple Visualization 
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(infection_curve, color='red')
    plt.title('Active Infections over Time')
    plt.xlabel('Days (Steps)')
    
    plt.subplot(2, 2, 2)
    plt.plot(icu_usage, color='orange')
    plt.title('ICU Occupancy over Time')
    plt.xlabel('Days (Steps)')
    
    plt.subplot(2, 2, 3)
    plt.plot(economic_health, color='green')
    plt.title('Average Economic Health')
    plt.xlabel('Days (Steps)')
    
    plt.subplot(2, 2, 4)
    # Plot lockdown levels for a few sample districts to keep it readable
    sample_agents = env.agents[:3]
    for agent in sample_agents:
        plt.plot(lockdown_levels[agent], label=agent, alpha=0.7)
    plt.title('Agent Policy: Lockdown Levels Chosen')
    plt.xlabel('Days (Steps)')
    plt.ylabel('Lockdown Strictness (0-3)')
    plt.yticks([0, 1, 2, 3])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print(f"Evaluation Complete. Results saved to {os.path.abspath('evaluation_results.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kerala COVID-19 Resource RL framework")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='both',
                        help="Mode to run the framework in")
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        trained_controller, hist = train(num_episodes=5) # Small for quick test
        
    if args.mode in ['eval', 'both']:
        if 'trained_controller' not in locals():
            print("Running untrained evaluation...")
            # Initialize a generic controller with ALL agents from JSON
            from src.env.seaird_env import KeralaCovidEnv
            dummy_env = KeralaCovidEnv()
            dummy_env.reset()
            trained_controller = MARLController(dummy_env.agents, 10, 4)
            
        evaluate(trained_controller)
