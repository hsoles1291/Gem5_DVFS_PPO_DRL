Gem5_DVFS_PPO_DRL
Adaptive Dynamic Voltage and Frequency Scaling (DVFS) Optimization using Proximal Policy Optimization (PPO) in Gem5 Simulations

📚 Project Overview
This project integrates Dynamic Voltage and Frequency Scaling (DVFS) with Deep Reinforcement Learning (PPO algorithm) in the Gem5 simulator environment.
The goal is to intelligently optimize power consumption and performance in heterogeneous multicore systems by dynamically adjusting CPU frequencies based on workload characteristics.

This work is part of a Master's thesis focused on energy efficiency in modern computing architectures.

🛠️ Repository Structure
Benchmark_rcS/ – Custom rcS scripts to automate the execution of benchmarks inside the simulated Ubuntu environment.

Gem5_Collaterals/DTB/ – Device Tree Binary (DTB) files and kernel collaterals customized for DVFS control.

Main_Script/ – Python scripts managing the environment simulation, parsing stats, and coordinating PPO model interactions.

PPO_Model/ – Implementation of the PPO Deep Reinforcement Learning model adapted for real-time decision-making on DVFS.

Run_Shells/ – Shell scripts to launch Gem5 simulations with proper environment settings.

README.md – Project description and instructions (this file).

⚙️ Main Features
Full-System Gem5 Simulation with Ubuntu OS booted and DVFS enabled.

Real-time PPO Agent that receives system signals and controls CPU frequencies.

Custom Benchmark Automation using Splash-3 workloads.

Dynamic Adaptation based on system load to optimize energy and performance.

Custom Power Modeling using Gemstone power model extensions.

🚀 Quickstart
Clone the repository:

bash
Copy
Edit
git clone https://github.com/hsoles1291/Gem5_DVFS_PPO_DRL.git
cd Gem5_DVFS_PPO_DRL
Set up Gem5:

Ensure you have a working Gem5 full-system simulation environment (ARM architecture recommended).

Kernel must support DVFS and userspace frequency scaling.

Prepare the environment:

Place your kernel and disk images in the appropriate paths.

Update paths in Run_Shells/ scripts if necessary.

Train or Load PPO Model:

The PPO agent will train automatically if no pretrained model is found.

Trained models are stored in PPO_Model/.

Run the Simulation:

bash
Copy
Edit
bash Run_Shells/run_gem5.sh
(Adjust script if needed for your specific benchmarks.)

🧠 PPO Model Overview
State Space: Selected system metrics including CPU frequency, voltage, power estimations, cache behavior, and memory access patterns.

Action Space: Frequency scaling actions for each voltage/frequency domain (big and little clusters).

Reward Function: Optimizes a trade-off between minimizing dynamic power consumption and maintaining acceptable performance levels.

🖥️ System Architecture Overview
Here’s a simple high-level flow:

pgsql
Copy
Edit
Gem5 Simulator (Ubuntu + DVFS enabled)
        ↓
Gem5 generates system statistics (stats.txt)
        ↓
Main Python Script reads stats and processes signals
        ↓
PPO Model decides frequency scaling actions
        ↓
Action commands written to shared files
        ↓
rcS scripts inside Ubuntu apply new CPU frequencies
        ↓
Cycle repeats until the end of the benchmark
This cycle enables adaptive, intelligent frequency management based on real-time system load.

📋 Requirements
Gem5 (with DVFS and ARM big.LITTLE simulation support)

Python 3.8+

PyTorch (for PPO model training and inference)

Numpy

Matplotlib (optional, for plotting performance)

📑 Thesis Context
This repository is part of the research for the Master's thesis:
"Escalado Dinámico de Voltaje y Frecuencia (DVFS) Adaptativo impulsado por Aprendizaje Automático para Sistemas Heterogéneos Multinúcleo en Gem5."

The work demonstrates the feasibility of integrating AI-based DVFS strategies in simulation environments, contributing to the development of energy-efficient computing architectures.

👤 Author
Heiner Solis Esquivel
Master's Candidate in Modern Manufacturing Systems
Focus Area: Microelectronics and Energy Efficiency
GitHub: hsoles1291

📜 License
This project is licensed under the MIT License — feel free to use, modify, and build upon it with attribution.

