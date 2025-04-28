Gem5_DVFS_PPO_DRL
Adaptive Dynamic Voltage and Frequency Scaling (DVFS) Optimization using Proximal Policy Optimization (PPO) in Gem5 Simulations

Project Overview
This project integrates Dynamic Voltage and Frequency Scaling (DVFS) with Deep Reinforcement Learning (PPO algorithm) in the Gem5 simulator environment.
The goal is to optimize power consumption and performance in heterogeneous multicore systems by dynamically adjusting CPU frequencies based on workload characteristics.

This work is part of a Master's thesis focused on energy efficiency in modern computing architectures.

Repository Structure
Benchmark_rcS/ – Custom rcS scripts to automate benchmark execution inside the simulated Ubuntu environment.

Gem5_Collaterals/DTB/ – Device Tree Binary (DTB) files and kernel collaterals customized for DVFS control.

Main_Script/ – Python scripts managing environment simulation, parsing stats, and PPO interactions.

PPO_Model/ – Implementation of the PPO Deep Reinforcement Learning model.

Run_Shells/ – Shell scripts to launch Gem5 simulations.

README.md – Project description and usage instructions.

Main Features
Full-system simulation in Gem5 with Ubuntu and DVFS enabled.

Real-time PPO agent that controls CPU frequencies.

Automated benchmark execution using Splash-3 workloads.

Dynamic adaptation based on system load for energy/performance optimization.

Power modeling using Gemstone extensions.

Set up Gem5:

Ensure a working Gem5 full-system simulation environment (ARM architecture recommended).

Kernel must support DVFS and userspace frequency scaling.

Prepare the environment:

Place the kernel and disk images accordingly.

Update any paths inside Run_Shells/ scripts if needed.

Train or load PPO model:

The PPO agent trains automatically if no model is found.

Trained models are saved under PPO_Model/.


PPO Model Overview

State space: Selected system metrics including CPU frequency, voltage, power estimations, cache behavior, and memory access patterns.

Action space: Frequency scaling actions for each voltage/frequency domain (big and little clusters).

Reward function: Balances minimizing dynamic power consumption and maintaining system performance.

Gem5 Simulator (Ubuntu + DVFS enabled)
        ↓
Gem5 generates system statistics (stats.txt)
        ↓
Main Python Script processes stats and generates signals
        ↓
PPO Model predicts scaling actions
        ↓
Action commands written to shared files
        ↓
rcS scripts apply new CPU frequencies inside Ubuntu
        ↓
Cycle repeats until benchmark completion
Requirements
Gem5 simulator (with DVFS and ARM big.LITTLE support)

Python 3.8+

PyTorch

Numpy

Matplotlib (optional, for performance plots)

Thesis Context
This repository supports the research for the Master's thesis:
"Escalado Dinámico de Voltaje y Frecuencia (DVFS) Adaptativo impulsado por Aprendizaje Automático para Sistemas Heterogéneos Multinúcleo en Gem5."

The project explores ML-driven DVFS strategies in simulated heterogeneous computing systems.

Author
Heiner Solis Esquivel
Focus Area: Microelectronics

License
This project is licensed under the MIT License.
