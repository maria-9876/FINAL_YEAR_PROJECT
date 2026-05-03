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
        ep_states = {agent: [] for agent in env.possible_agents}
        ep_actions = {agent: [] for agent in env.possible_agents}
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
                ep_states[agent].append(observations[agent])
                ep_actions[agent].append(actions[agent])
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
                states_list=ep_states[agent],
                actions_list=ep_actions[agent],
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
    district_data = {agent: [] for agent in env.agents}
    
    for step in range(max_steps):
        # Deterministic actions (no sampling noise)
        actions, _, _ = controller.get_actions(observations)
        observations, _, terminations, truncations, _ = env.step(actions)
        
        # Log globals
        total_inf = sum([env.state[agent]['I'] for agent in env.possible_agents])
        total_icu = sum([env.state[agent]['icu_occupancy'] for agent in env.possible_agents])
        avg_econ = np.mean([env.state[agent]['economic_health'] for agent in env.possible_agents])
        
        infection_curve.append(total_inf)
        icu_usage.append(total_icu)
        economic_health.append(avg_econ)
        
        # Log specifics
        for agent in env.possible_agents:
            lockdown_idx = int(actions[agent][0] * 3.99)
            lockdown_levels[agent].append(lockdown_idx)
            
            state_record = env.state[agent].copy()
            state_record['step'] = step
            state_record['lockdown_level'] = lockdown_idx
            district_data[agent].append(state_record)
            
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
    total_base_icu = sum([env.district_data[a]['icu_capacity'] for a in env.possible_agents])
    plt.axhline(y=total_base_icu, color='red', linestyle='--', label='Max Capacity')
    plt.legend()
    plt.title('ICU Occupancy over Time')
    plt.xlabel('Days (Steps)')
    
    plt.subplot(2, 2, 3)
    plt.plot(economic_health, color='green')
    plt.title('Average Economic Health')
    plt.xlabel('Days (Steps)')
    
    plt.subplot(2, 2, 4)
    # Plot lockdown levels for a few sample districts to keep it readable
    sample_agents = env.possible_agents[:3]
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
    
    output_dir = "district_evaluation_data"
    os.makedirs(output_dir, exist_ok=True)
    
    for agent, history in district_data.items():
        if not history: continue
        
        inf_curve = [r['I'] for r in history]
        icu_curve = [r['icu_occupancy'] for r in history]
        econ_curve = [r['economic_health'] for r in history]
        lock_curve = [r['lockdown_level'] for r in history]
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(inf_curve, color='red')
        plt.title(f'{agent} - Active Infections')
        plt.xlabel('Days (Steps)')
        
        plt.subplot(2, 2, 2)
        plt.plot(icu_curve, color='orange')
        agent_icu_cap = env.district_data[agent]['icu_capacity']
        plt.axhline(y=agent_icu_cap, color='red', linestyle='--', label='Max Capacity')
        plt.legend()
        plt.title(f'{agent} - ICU Occupancy')
        plt.xlabel('Days (Steps)')
        
        plt.subplot(2, 2, 3)
        plt.plot(econ_curve, color='green')
        plt.title(f'{agent} - Economic Health')
        plt.xlabel('Days (Steps)')
        
        plt.subplot(2, 2, 4)
        plt.plot(lock_curve, label=agent, color='blue', alpha=0.7)
        plt.title(f'{agent} - Lockdown Levels Chosen')
        plt.xlabel('Days (Steps)')
        plt.ylabel('Lockdown Strictness (0-3)')
        plt.yticks([0, 1, 2, 3])
        plt.legend()
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{agent}_evaluation.png")
        plt.savefig(filepath)
        plt.close()
        
    print(f"District graphs saved to {os.path.abspath(output_dir)}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kerala COVID-19 Resource RL framework")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='both',
                        help="Mode to run the framework in")
    parser.add_argument('--episodes', type=int, default=5,
                        help="Number of episodes to train for")
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        trained_controller, hist = train(num_episodes=args.episodes)
        
    if args.mode in ['eval', 'both']:
        if 'trained_controller' not in locals():
            print("Running untrained evaluation...")
            # Initialize a generic controller with ALL agents from JSON
            from src.env.seaird_env import KeralaCovidEnv
            dummy_env = KeralaCovidEnv()
            dummy_env.reset()
            trained_controller = MARLController(dummy_env.agents, 10, 4)
            
        evaluate(trained_controller)
