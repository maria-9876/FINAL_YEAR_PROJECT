# Chapter 3: System Architecture & Methodology

## 3.1 Overview
The proposed framework relies on two distinct, interacting modules: the Epidemiological Environment Simulator and the Multi-Agent Reinforcement Learning Controller. Unlike static historical models, the MARL structure relies on Centralized Training and Decentralized Execution (CTDE), enabling intelligent individual agents to continuously adjust their resource provisioning based on both local metrics and neighbor-state awareness.

## 3.2 The SEAIRD Simulator
The environment models an outbreak using a modified Susceptible (S), Exposed (E), Asymptomatic (A), Infected (I), Recovered (R), and Deceased (D) progression dynamic. 

### Multi-Disease Parameterization
A key novelty of this architecture is its detachment from COVID-19 specific constraints. The environment accepts configurable generic parameters:
- **$\beta$ (Transmission Rate)**: Modifiable to reflect high R0 diseases (e.g., Measles) vs. low R0 diseases (e.g., Ebola).
- **$\gamma_a, \gamma_i$ (Recovery Rates)**: Separate metrics for asymptomatic and symptomatic progression.
- **$\mu$ (Mortality Rate)**: Baseline lethality, which dynamically scales non-linearly when regional ICU capacities are breached.
- **Asymptomatic Ratio**: The fraction of the exposed transitioning to silent spreaders, a crucial variable for diseases like COVID-19.

### Behavioral Variability Integration
Real-world policy adoption is rarely uniform. The states explicitly map:
1. **Public Compliance Rates**: Affecting the real efficacy of implemented lockdown levels.
2. **Economic Health Indexes**: Reflecting regional dependencies on daily-wage labor versus service sectors.

## 3.3 State and Action Spaces
At each continuous time-step, each agent $i$ receives a state observation vector $S_i$:
$$ S_i = [S \%, E \%, A \%, I \%, R \%, D \%, \text{ICU\_Occupancy}, \text{Vaccines\_Available}, \text{Compliance}, \text{Economy} ] $$

Based on $S_i$, the agent outputs a continuous action vector $A_i \in [0, 1]^4$, subsequently mapped to:
1. **Lockdown Strictness**: Discretized into 4 levels (0: Open, to 3: Full Containment).
2. **Vaccine Distribution**: Percentage of available local stockpiles to mass-dispense.
3. **Medical Staff Movement**: Requesting or releasing surplus emergency personnel to modulate local ICU overflow limits.
4. **Test Kit Allocation**: Affecting the $\alpha$ transition rate by isolating identified Asymptomatic clusters.

## 3.4 Multi-Agent Reinforcement Learning Formulation
The controller utilizes a Proximal Policy Optimization (PPO) foundation augmented with Generalized Advantage Estimation (GAE). 

### Continuous Feedback Loop
While traditional systems analyze data in weekly batches, this architecture's Gym environment operates dynamically. The agent networks process the differential equations step-by-step, allowing rapid micro-adjustments in medical staff allocation prior to a catastrophic ICU breach.

### Multi-Objective Reward Architecture
The reward function $R_t$ minimizes three competing penalties simultaneously:
$$ R_t = -\Big( \alpha \cdot f(\text{Infections}, \text{Deaths}) + \beta \cdot g(\text{ICU\_Overflow}) + \gamma \cdot h(\text{Economic\_Loss}) \Big) $$

Wherein the function $g(x)$ applies an exponential penalty ($overflow \rightarrow \infty$) specifically targeting structural inequality in health access, ensuring the algorithm heavily penalizes policies that allow rural districts to collapse before urban ones.
