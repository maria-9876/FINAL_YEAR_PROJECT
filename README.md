# COVID-19 Multi-Agent Reinforcement Learning (MARL) for Resource Allocation in Kerala

This repository contains a research-level simulation and Multi-Agent Reinforcement Learning (MARL) framework designed to optimize the allocation of healthcare resources during infectious disease outbreaks. The framework focuses on the state of Kerala, India, treating each district as an independent, interacting agent within the environment.

## Overview

Based on the paper *"Reinforcement Learning Based Framework for COVID-19 Resource Allocation,"* this project models an epidemic spread using a **SEAIRD (Susceptible, Exposed, Asymptomatic, Infected, Recovered, Deceased)** compartment model. 

The framework employs an Actor-Critic MARL approach (Centralized Training, Decentralized Execution) to train district-level agents. These agents make daily or weekly policy decisions regarding:
1. Lockdown level
2. Vaccine allocation
3. ICU bed redistribution requests 
4. Medical staff allocation modifier

The learned policies seek an optimal balance between minimizing fatalities, preventing hospital overload, and reducing economic damage.

## File Structure

- `data/kerala_districts.json`: Synthetic/approximated district-level demographic, healthcare, and economic data used by the simulator.
- `docs/`: Academic documentation, report structuring, and ideas for future extensions.
- `src/env/`: The custom OpenAI Gym / PettingZoo environment (`seaird_env.py`) and pure SEAIRD mathematical dynamics (`disease_dynamics.py`).
- `src/agents/`: Neural network definitions for Actor-Critic agents and the wrapper controller managing multi-agent interaction.
- `src/utils/`: Custom reward function logic prioritizing healthcare capacity and economic stability.
- `main.py`: The entry point script to train the model, evaluate the results, and generate performance plots.

## Setup Instructions

1. Ensure you have Python 3.8+ installed.
2. Install the necessary dependencies (Gym, PyTorch, Pandas, NumPy, Matplotlib).
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training loop:
   ```bash
   python main.py --mode train
   ```
4. Evaluate a trained model:
   ```bash
   python main.py --mode eval
   ```

## Assumptions & Disclaimers

This represents a prototype designed for academic research. District demographics and hospital capacities are reasonable approximations to demonstrate the capabilities of the reinforcement learning framework, but are not precise real-time statistics.
