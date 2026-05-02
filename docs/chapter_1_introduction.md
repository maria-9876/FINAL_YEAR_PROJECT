# Chapter 1: Introduction

## 1.1 Background
The rapid global spread of infectious diseases has historically challenged robust healthcare infrastructure. Traditional epidemiological models—such as the Susceptible, Infected, Recovered (SIR) model—have served as the bedrock for predicting transmission dynamics. However, translating these theoretical predictions into actionable, real-time public policy remains extraordinarily complex. Lockdowns disrupt economic stability, while unchecked spread causes hospital infrastructure overload and increased mortality rates. 

## 1.2 Motivation
Static or purely reactive containment strategies—often updated weekly or monthly—struggle to keep pace with the exponential dynamic of disease transmission. Inspired by the paper *"Reinforcement Learning Based Framework for COVID-19 Resource Allocation"*, this thesis aims to bridge the gap between deterministic disease modeling and dynamic optimization via Multi-Agent Reinforcement Learning (MARL). 

Our system specifically replaces fixed-rule systems with continuous feedback loops. By extending the referenced MARAAC (Multi-Agent Recurrent Attention Actor-Critic) methodology, this project moves beyond standard state-wise lockdowns to incorporate fine-grained decisions across multiple resource dimensions (vaccines, test kits, ICU beds, medical personnel) targeted at a geographically modeled simulation of Kerala, India.

## 1.3 Problem Statement
The central problem addressed by this thesis is mapping optimal resource allocation across a networked system of agents (districts) to balance three competing objectives:
1. **Minimizing fatalities** and halting disease propagation.
2. **Preventing healthcare overload**, specifically tracking precise finite ICU bed capacities.
3. **Preserving economic stability**, mitigating the cascading financial impact of severe lockdowns.

Unlike traditional methods that assume homogenous behavior, our environment specifically models public compliance, misinformation spread, and regional inequalities in healthcare access.

## 1.4 Objectives
- **Develop a Multi-Disease Environment**: Implement a compartmental SEAIRD simulation capable of accepting parameter changes for different viral strains.
- **Formulate the MARL Algorithm**: Design an Actor-Critic architecture to map a high-dimensional continuous state space to multiple simultaneous actions.
- **Implement Continuous Feedback**: Upgrade traditional weekly-decision paradigms to daily continuous feedback loops to adapt instantly to emerging hotspots.
- **Evaluate Trade-offs**: Prove the stability of our model over baseline approaches via evaluation metrics capturing both economic and health statistics.

## 1.5 Organization of the Report
Chapter 2 explores the relevant literature concerning epidemic modeling and RL approaches in healthcare. Chapter 3 details the system architecture, mathematical formulations, and agent design. Chapter 4 documents the implementation and data scaling logic. Experimental results, including dynamic lockdown charting and baseline comparisons, are discussed in Chapter 5. Finally, Chapter 6 concludes the report and discusses limitations and future research trajectories.
