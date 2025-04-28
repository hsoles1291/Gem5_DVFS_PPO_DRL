import numpy as np
import argparse
from stable_baselines3 import PPO
from gym.vector import AsyncVectorEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv  # Import DummyVecEnv
import multiprocessing
import warnings
multiprocessing.set_start_method('spawn', force=True)
import time
from gem5_telnet_terminal_MEP_Final import set_userspace_governor, set_frequencies, run_script
from collections import deque
import random
from collections import deque
import threading
import builtins
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the level of detail (DEBUG captures everything)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format: timestamp, level, message
    handlers=[
        logging.FileHandler("debug_47ab.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

# Redirect print to logging.info
original_print = print
def print(*args, level="info", **kwargs):
    """
    Custom print function with adjustable log levels.
    """
    message = " ".join(map(str, args))
    if level.lower() == "debug":
        logging.debug(message)
    elif level.lower() == "warning":
        logging.warning(message)
    elif level.lower() == "error":
        logging.error(message)
    elif level.lower() == "critical":
        logging.critical(message)
    else:
        logging.info(message)

# Example: Restore original print if needed
# builtins.print = original_print


warnings.filterwarnings("ignore", message=".*TimeoutError.*")
# Configuración para guardar y cargar el modelo
MODEL_PATH = "ppo_dvfs_agent.zip"

# Experience Buffer
experience_buffer = deque(maxlen=10000)  # Store up to 10,000 experiences

# Define a batch size for training
BATCH_SIZE = 64

def ticks_to_hz(ticks):
    if ticks <= 0:
        print(f"[DEBUG] Invalid tick value: {ticks}")
        return 0
    return (1 / (ticks / 1e12)) / 1e6

def ticks_to_hz_freqchange(ticks):
    if ticks <= 0:
        print(f"[DEBUG] Invalid tick value: {ticks}")
        return 0
        # Convert ticks to Hz and scale it to match the expected format
    frequency_hz = (1 / (ticks / 1e12))  # Convert to Hz
    scaled_frequency = int(frequency_hz /1e3)  # Scale to match the expected format
    if scaled_frequency == 999999:
        scaled_frequency = 1000000
    return scaled_frequency

# Funciones para calcular la potencia dinámica y estática
def calculate_dynamic_power_a15(vars):
    if vars["simSeconds"] == 0:
        print("Warning: simSeconds is zero, assigning variable to 0.001")
        vars["simSeconds"] = 0.001

    dyn = (
            (((((vars["system.bigCluster.cpus0.numCycles"] + vars["system.bigCluster.cpus1.numCycles"]) / 2) / vars[
                "simSeconds"]) / (1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.bigCluster.voltage_domain.voltage"] ** 2 * 6.06992538845e-10) +
            (((((vars["system.bigCluster.cpus0.dcache.overallAccesses::total"] + vars[
                "system.bigCluster.cpus1.dcache.overallAccesses::total"]) / 2) / vars["simSeconds"]) /
              (1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.bigCluster.voltage_domain.voltage"] ** 2 * 2.32633723171e-10) +
            (((((vars["system.bigCluster.cpus0.iew.instsToCommit"] + vars[
                "system.bigCluster.cpus1.iew.instsToCommit"]) / 2) / vars["simSeconds"]) /
              ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) -
              (((vars["system.bigCluster.cpus0.statFuBusy::IntAlu"] + vars[
                  "system.bigCluster.cpus0.statFuBusy::IntMult"] + vars["system.bigCluster.cpus0.statFuBusy::IntDiv"] +
                 vars["system.bigCluster.cpus1.statFuBusy::IntAlu"] + vars[
                     "system.bigCluster.cpus1.statFuBusy::IntMult"] + vars[
                     "system.bigCluster.cpus1.statFuBusy::IntDiv"]) / 2) /
               vars["simSeconds"]) / (1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.bigCluster.voltage_domain.voltage"] ** 2 * 5.43933973638e-10) +
            (((((vars["system.bigCluster.cpus0.dcache.WriteReq.misses::total"] + vars[
                "system.bigCluster.cpus1.dcache.WriteReq.misses::total"]) / 2) /
               vars["simSeconds"]) / (1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.bigCluster.voltage_domain.voltage"] ** 2 * 4.79625288372e-08) +
            (((((vars["system.bigCluster.l2.overallAccesses::bigCluster.cpus0.data"] + vars["system.bigCluster.l2.overallAccesses::bigCluster.cpus1.data"]) / 2) / vars["simSeconds"]) /
              (1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.bigCluster.voltage_domain.voltage"] ** 2 * 5.72830963981e-09) +
            (((((vars["system.bigCluster.cpus0.icache.ReadReq.accesses::total"] + vars[
                "system.bigCluster.cpus1.icache.ReadReq.accesses::total"]) / 2) /
               vars["simSeconds"]) / (1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.bigCluster.voltage_domain.voltage"] ** 2 * 8.41332534886e-10) +
            (((((vars["system.bigCluster.cpus0.statFuBusy::IntAlu"] + vars[
                "system.bigCluster.cpus0.statFuBusy::IntMult"] + vars["system.bigCluster.cpus0.statFuBusy::IntDiv"] +
                 vars["system.bigCluster.cpus1.statFuBusy::IntAlu"] + vars[
                     "system.bigCluster.cpus1.statFuBusy::IntMult"] + vars[
                     "system.bigCluster.cpus1.statFuBusy::IntDiv"]) / 2) /
               vars["simSeconds"]) / (1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.bigCluster.voltage_domain.voltage"] ** 2 * 2.44859350364e-10)
    )
    #return max(dyn, 0)  # Ensure dynamic power is not negative
    return dyn

def calculate_static_power_a15(vars):
    if vars["simSeconds"] == 0:
        print("Warning: simSeconds is zero, assigning variable to 0.001")
        vars["simSeconds"] = 0.001
    st = (
            (1 * -681.604059986) +
            (((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) * 0.117551170367) +
            (vars["system.bigCluster.voltage_domain.voltage"] * 2277.16890778) +
            (((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) * vars[
                "system.bigCluster.voltage_domain.voltage"] * -0.491846201277) +
            (vars["system.bigCluster.voltage_domain.voltage"] * vars[
                "system.bigCluster.voltage_domain.voltage"] * -2528.1574686) +
            (((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) * vars[
                "system.bigCluster.voltage_domain.voltage"] * vars[
                 "system.bigCluster.voltage_domain.voltage"] * 0.645456768269) +
            (vars["system.bigCluster.voltage_domain.voltage"] * vars["system.bigCluster.voltage_domain.voltage"] * vars[
                "system.bigCluster.voltage_domain.voltage"] * 932.937276293) +
            (((1 / (vars["system.bigCluster.clk_domain.clock"] / 1e12)) / 1e6) * vars[
                "system.bigCluster.voltage_domain.voltage"] * vars["system.bigCluster.voltage_domain.voltage"] * vars[
                 "system.bigCluster.voltage_domain.voltage"] * -0.271180478671)
    )
    #return max(st, 0)  # Ensure static power is not negative
    return st

def calculate_dynamic_power_a7(vars):
    if vars["simSeconds"] == 0:
        print("Warning: simSeconds is zero, assigning variable to 0.001")
        vars["simSeconds"] = 0.001
    dyn = (
            (((((vars["system.littleCluster.cpus0.numCycles"] + vars["system.littleCluster.cpus1.numCycles"]) / 2) /
               vars["simSeconds"]) /
              (1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.littleCluster.voltage_domain.voltage"] ** 2 * 1.84132144059e-10) +
            (((((vars["system.littleCluster.cpus0.commitStats0.numInsts"] + vars[
                "system.littleCluster.cpus1.commitStats0.numInsts"]) / 2) /
               vars["simSeconds"]) /
              (1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.littleCluster.voltage_domain.voltage"] ** 2 * 1.11008189839e-10) +
            (((((vars["system.littleCluster.cpus0.dcache.overallAccesses::total"] + vars[
                "system.littleCluster.cpus1.dcache.overallAccesses::total"]) / 2) /
               vars["simSeconds"]) /
              (1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.littleCluster.voltage_domain.voltage"] ** 2 * 2.7252658216e-10) +
            (((((vars["system.littleCluster.cpus0.dcache.overallMisses::total"] + vars[
                "system.littleCluster.cpus1.dcache.overallMisses::total"]) / 2) /
               vars["simSeconds"]) /
              (1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.littleCluster.voltage_domain.voltage"] ** 2 * 2.4016441235e-09) +
            (((((vars["system.mem_ctrls.bytesRead::littleCluster.cpus0.data"] + vars["system.mem_ctrls.bytesRead::littleCluster.cpus1.data"] ) / 2) / vars["simSeconds"]) /
              (1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             ((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) *
             vars["system.littleCluster.voltage_domain.voltage"] ** 2 * -2.44881613234e-09)
    )
    #return max(dyn, 0)  # Ensure dynamic power is not negative
    return dyn

def calculate_static_power_a7(vars):
    if vars["simSeconds"] == 0:
        print("Warning: simSeconds is zero, assigning variable to 0.001")
        vars["simSeconds"] = 0.001
    st = (
            (1 * 31.0366448991) +
            (((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) * -0.0267126706228) +
            (vars["system.littleCluster.voltage_domain.voltage"] * -87.7978467067) +
            (((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) * vars[
                "system.littleCluster.voltage_domain.voltage"] * 0.0748426796784) +
            (vars["system.littleCluster.voltage_domain.voltage"] * vars[
                "system.littleCluster.voltage_domain.voltage"] * 82.5596011612) +
            (((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) * vars[
                "system.littleCluster.voltage_domain.voltage"] * vars[
                 "system.littleCluster.voltage_domain.voltage"] * -0.0696612748138) +
            (vars["system.littleCluster.voltage_domain.voltage"] * vars["system.littleCluster.voltage_domain.voltage"] *
             vars["system.littleCluster.voltage_domain.voltage"] * -25.8616662356) +
            (((1 / (vars["system.littleCluster.clk_domain.clock"] / 1e12)) / 1e6) * vars[
                "system.littleCluster.voltage_domain.voltage"] * vars["system.littleCluster.voltage_domain.voltage"] *
             vars["system.littleCluster.voltage_domain.voltage"] * 0.0216526889381)
    )
    #return max(st, 0)  # Ensure static power is not negative
    return st

# Mapping functions for frequency values
def map_big_core_frequency(freq):
    mapping = {
        1800: 1798561,
        1700: 1700680,
        1610: 1610305,
        1510: 1510574,
        1400: 1400560,
        1200: 1200480,
        1000: 1000000,
        667: 667111,
    }
    return mapping.get(freq, freq)  # Default to the input if no mapping is found

def map_little_core_frequency(freq):
    mapping = {
        1900: 1901140,
        1700: 1700680,
        1610: 1610305,
        1510: 1510574,
        1400: 1400560,
        1200: 1200480,
        1000: 1000000,
        667: 667111,
    }
    return mapping.get(freq, freq)  # Default to the input if no mapping is found

def update_ema(current_ema, new_value, alpha=0.2):
    """
    Update the EMA for a single value.

    Parameters:
    - current_ema (float): The current EMA value.
    - new_value (float): The new value to include in the EMA.
    - alpha (float): Smoothing factor (0 < alpha ≤ 1).

    Returns:
    - float: Updated EMA value.
    """
    return alpha * new_value + (1 - alpha) * current_ema

# Crear un entorno de gym personalizado para gem5 DVFS
class Gem5DVFSEnv(gym.Env):
    def __init__(self, stats_chunks, stats_generator ,big_cores, little_cores, threads, big_vf_curve, little_vf_curve, host, port):
        super(Gem5DVFSEnv, self).__init__()
        self.stats_chunks = stats_chunks
        self.stats_generator = stats_generator  # Add stats_generator
        self.big_cores = big_cores
        self.little_cores = little_cores
        self.threads = threads
        self.big_vf_curve = big_vf_curve  # NEW CHANGE
        self.little_vf_curve = little_vf_curve  # NEW CHANGE
        self.current_chunk_index = 0
        self.host = host  # Added to support Telnet communication
        self.port = port  # Added to support Telnet communication
        # Initialize history dictionaries for tracking signal deltas
        self.big_cluster_history = {
            "cpus0_dcache_misses": 0.0,
            "cpus1_dcache_misses": 0.0,
            "l2_overall_accesses": 0.0,
            "l2_cpus0_data": 0.0,
            "l2_cpus1_data": 0.0,
        }
        self.little_cluster_history = {
            "bytes_read_total": 0.0,
            "dcache_cpus0_accesses": 0.0,
            "dcache_cpus1_accesses": 0.0,
            "dcache_cpus0_misses": 0.0,
            "dcache_cpus1_misses": 0.0,
        }
        self.low_load_state = {"big": False, "little": False}
        self.low_load_applied = {"big": False, "little": False}
        self.run_mode = "train"
        #self.action_space = gym.spaces.Discrete(len(big_vf_curve))  # NEW CHANGE
        # Initialize frequency state tracking
        self.previous_big_freq = None
        self.previous_little_freq = None
        self.current_big_freq = None
        self.current_little_freq = None
        self.action_space = gym.spaces.MultiDiscrete([len(big_vf_curve), len(little_vf_curve)])
        self.accumulated_chunks = []  # Define it as a class variable

        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(stats_chunks[0]),), dtype=np.float32)

        # Track previous load states to avoid oscillation
        self.previous_load_state = {"big": None, "little": None}

        # Define transition margin to avoid instability
        self.transition_margin = 5  # Adjust this based on testing results
        # Initialize load state history tracking
        self.load_state_history = {
            "big": [],
            "little": []
        }
        self.simulation_time = 0.0  # Initialize the accumulated simulation time
        self.last_state = None  # Store last processed state for comparison
        self.prev_low_load_state = {"big": False, "little": False}
        self.low_load_transition_pending = {"big": False, "little": False}
        self.high_load_transition_pending = {"big": False, "little": False}
        self.dur_at_freq = 0
        self.chunk_user_var_to_wait = 0
        self.user_changes_to_check = 0

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.current_chunk_index = 0

            if not self.stats_chunks:
                print("[WARNING] stats_chunks is empty during reset. Returning default observation.")
                return np.zeros(self.observation_space.shape), {}

            observation = self.stats_chunks[self.current_chunk_index]
            observation_array = np.array(list(observation.values()), dtype=np.float32)

            if np.any(np.isnan(observation_array)):
                print("[ERROR] Reset observation contains NaN values!")
            if np.any(np.isinf(observation_array)):
                print("[ERROR] Reset observation contains infinite values!")

            return observation_array.reshape(1, -1), {}

        except Exception as e:
            print(f"[ERROR] Exception in reset function: {e}")
            return np.zeros(self.observation_space.shape), {}

    def evaluate_load_state(self, current_state, thresholds, cluster_type, threshold_type, factor):
        """
        Evaluates if the given cluster (big/little) is in the specified load state (low/high).
        Uses a robust 3-step confirmation mechanism to prevent rapid state switching.
        """

        # Ensure tracking variables exist
        if cluster_type not in self.load_state_history:
            self.load_state_history[cluster_type] = []
        if cluster_type not in self.previous_load_state:
            self.previous_load_state[cluster_type] = None

        # Ensure a default transition margin if not set
        if not hasattr(self, "transition_margin"):
            self.transition_margin = 3  # Default value if not set elsewhere

        current_values = {}
        high_metrics = []
        low_metrics = []
        Changes_To_Check = self.user_changes_to_check
        print(f"\n[INFO] Evaluating {cluster_type.capitalize()} Cluster for {threshold_type.capitalize()} Load Mode:")

        for key, threshold in thresholds[threshold_type][cluster_type].items():
            current_value = current_state.get(key, None)
            if current_value is None:
                print(f"[ERROR] Key {key} is missing in current_state. Defaulting to 0.")
                current_value = 0  # Explicitly set to 0 after warning
            current_values[key] = current_value

            if threshold_type == "low":
                if current_value <= threshold:
                    low_metrics.append(key)
                elif current_value > threshold:
                    high_metrics.append(key)
            else:  # High Load Mode
                if current_value >= threshold:
                    high_metrics.append(key)
                elif current_value < threshold:
                    low_metrics.append(key)

        # Debug Outputs
        print(
            f"[DEBUG] {cluster_type.capitalize()} Lower than Threshold Metrics: {len(low_metrics)} | Higher than Threshold Metrics: {len(high_metrics)}")
        print(f"[INFO] Len Low Metrics: {len(low_metrics)} | Len High Metrics: {len(high_metrics)}")
        for key in low_metrics:
            print(f"  [DEBUG] {key}: {current_values[key]} <= {thresholds[threshold_type][cluster_type][key]}")
        print(f"[DEBUG] {cluster_type.capitalize()} Higher than {threshold_type} load mode Threshold")
        for key in high_metrics:
            print(f"  [DEBUG] {key}: {current_values[key]} >= {thresholds[threshold_type][cluster_type][key]}")
        print(
            f"[INFO] {cluster_type} Cores - *** Tentative Load State ***: {'LOW' if len(low_metrics) > len(high_metrics) else 'HIGH'}")

        # Determine tentative load state
        in_low_load = len(low_metrics) > len(high_metrics)

        # If `previous_load_state` is None, set it to the initial detected value
        if self.previous_load_state[cluster_type] is None:
            print(f"[INFO] {cluster_type} Cluster: Initializing previous state to {'LOW' if in_low_load else 'HIGH'}")
            self.previous_load_state[cluster_type] = in_low_load
            return in_low_load

        # Print current history before any changes
        print(f"[DEBUG] Current {cluster_type} Load History: {self.load_state_history[cluster_type]}")

        
            # Add the new evaluation to the history
        if self.load_state_history[cluster_type] and self.load_state_history[cluster_type][-1] != in_low_load:
            # Detected a mismatch - Reset history
            print(f"[WARNING] {cluster_type} Cluster: Mismatch in state detected! Resetting history.")
            self.load_state_history[cluster_type] = []

        self.load_state_history[cluster_type].append(in_low_load)

        # Print updated history
        print(f"[DEBUG] Updated {cluster_type} Load History After Append: {self.load_state_history[cluster_type]}")

        # Keep only the last 5 evaluations
        if len(self.load_state_history[cluster_type]) > Changes_To_Check:
            self.load_state_history[cluster_type].pop(0)

        # Apply 3-step confirmation logic **only if the state is about to change**
        if in_low_load != self.previous_load_state[cluster_type]:
            if len(self.load_state_history[cluster_type]) < Changes_To_Check:
                print(f"[INFO] {cluster_type} Cluster: Awaiting more samples before confirming a load state change.")
                return self.previous_load_state[cluster_type]  # Maintain the previous state for now

            # If all last 5 evaluations match, confirm the change
            if all(state == in_low_load for state in self.load_state_history[cluster_type]):
                print(f"[INFO] {cluster_type} Cluster *** CONFIRMED LOAD CHANGE ***: {'LOW' if in_low_load else 'HIGH'}")
                self.previous_load_state[cluster_type] = in_low_load
                self.load_state_history[cluster_type] = []
            else:
                print(
                    f"[INFO] {cluster_type} Cluster *** FALSE TRANISTION NOT MATCHING 5 TIMES DETECTED - MAINTAINING PREVIOUS STATE ***: {'LOW' if self.previous_load_state[cluster_type] else 'HIGH'}")
                self.load_state_history[cluster_type] = []
                return self.previous_load_state[cluster_type]

        else:
            # No state change is occurring, update normally
            self.previous_load_state[cluster_type] = in_low_load
            print(f"[INFO] {cluster_type} Cluster: No state change detected, maintaining current state.")

        return in_low_load

    def apply_frequency_override(self, cluster, is_low, action, freq, min_freq, high_freq,duration_at_freq=None):
        """
        Applies frequency overrides based on load state.
        If in low load, sets frequency to min_freq.
        If in high load and the action is 7, sets frequency to high_freq.
        """

        action_val = 0

        if is_low:
            print(
                f"[INFO] {cluster.capitalize()} Low Load Detected Previous Action {action}")
            if not self.low_load_state[cluster]:
                self.low_load_applied[cluster] = False
                print(f"[INFO] {cluster.capitalize()} Low Load Applied - Resetting Applied Flag")
            self.low_load_state[cluster] = True
            print(f"[INFO] {cluster.capitalize()} Low Load Mode - Setting Freq to {min_freq}")
            action_val = 7
            min_freq = 667111
            duration_at_freq = 0
            print(
                f"[INFO] {cluster.capitalize()} Low Load Detected Adjusting to Action {action_val} Freq:{min_freq}")
            return action_val, min_freq , duration_at_freq
        else:
            if action <= 7:
                print(
                    f"[INFO] {cluster.capitalize()} High Load Detected with Action {action} >= 4 - Adjusting to Mid-High Freq")
                if cluster == "big":

                    action_val = 7
                    high_freq = 667111
                    print(
                        f"[INFO] {cluster.capitalize()} High Load Detected with Action {action}  >= 4- Adjusting to Action {action_val} Freq:{high_freq}")
                elif cluster == "little":

                    action_val = 5
                    high_freq = 1200480
                    print(
                        f"[INFO] {cluster.capitalize()} High Load Detected with Action {action}  >= 4 - Adjusting to Action {action_val} Freq:{high_freq}")
                if duration_at_freq and duration_at_freq > self.dur_at_freq:  # Example: 180 seconds threshold
                    print(f"[INFO] {cluster.capitalize()} Prolonged High Frequency Detected - Gradual Adjustment Down After:{self.dur_at_freq} times")
                    action_to_reduce = action_val
                    action_val = min(action_to_reduce + 1, 7)  # Move one step lower
                    high_freq = map_big_core_frequency(
                        int(self.big_vf_curve[action_val][0])) if cluster == "big" else map_little_core_frequency(
                        int(self.little_vf_curve[action_val][0]))

                    # Now print using the precomputed value
                    print(
                        f"High Load Prolonged Mode Change [INFO] {cluster.capitalize()} Previous Action {action_to_reduce} ---> New Action {action_val} Freq:{high_freq}")
                print(f"[DEBUG] Returning from apply_frequency_override: {action_val}, {high_freq}")

                # Ensure action stays an integer, not a list
                if isinstance(action_val, list):
                    print(f"[ERROR] Action_val became a list in apply_frequency_override: {action_val}")
                    raise ValueError("Action should not be a list here!")
                return action_val, high_freq , duration_at_freq

        return action, freq , duration_at_freq

    def check_load_state(self,current_state, big_action, little_action, big_freq, little_freq):
        """
        Determines whether the system is in low-load or high-load mode and applies frequency overrides accordingly.
        """
        # Before: Initialize duration_at_freq if it doesn't exist
        if not hasattr(self, 'duration_at_freq'):
            self.duration_at_freq = {"big": 0, "little": 0}

        # Ensure tracking variables exist
        if not hasattr(self, 'prev_low_load_state'):
            self.prev_low_load_state = {"big": False, "little": False}
        if not hasattr(self, 'low_load_transition_pending'):
            self.low_load_transition_pending = {"big": False, "little": False}

        # Define Low and High Thresholds
        thresholds = {
            "low": {
                "big": {
                    "system.bigCluster.cpus0.dcache.overallAccesses::total": 8000,
                    "system.bigCluster.cpus1.dcache.overallAccesses::total": 8000,
                    "system.bigCluster.cpus0.dcache.WriteReq.misses::total": 1000,
                    "system.bigCluster.cpus1.dcache.WriteReq.misses::total": 1000,
                    "system.bigCluster.l2.overallAccesses::total": 500,
                    "system.bigCluster.l2.overallAccesses::bigCluster.cpus0.data": 1000,
                    "system.bigCluster.l2.overallAccesses::bigCluster.cpus1.data": 1000,
                    "system.bigCluster.cpus0.icache.ReadReq.accesses::total": 10000,
                    "system.bigCluster.cpus1.icache.ReadReq.accesses::total": 10000,
                    "system.bigCluster.cpus0.decode.blockedCycles": 37000,
                    "system.bigCluster.cpus1.decode.blockedCycles": 37000,
                    "system.bigCluster.cpus0.rename.ROBFullEvents": 500,
                    "system.bigCluster.cpus1.rename.ROBFullEvents": 500,
                    "system.bigCluster.cpus0.dcache.demandAvgMissLatency::total": 25000,
                    "system.bigCluster.cpus1.dcache.demandAvgMissLatency::total": 25000,
                    "system.bigCluster.cpus0.icache.demandAvgMissLatency::total": 25000,
                    "system.bigCluster.cpus1.icache.demandAvgMissLatency::total": 25000,
                    "system.bigCluster.cpus0.dcache.demandAvgMshrMissLatency::total": 25000,
                    "system.bigCluster.cpus1.dcache.demandAvgMshrMissLatency::total": 25000,
                    "system.bigCluster.cpus0.mmu.itb_walker.walkServiceTime::mean": 10000,
                    "system.bigCluster.cpus1.mmu.itb_walker.walkServiceTime::mean": 10000,
                    "system.bigCluster.cpus0.fetch.cycles": 50000,
                    "system.bigCluster.cpus1.fetch.cycles": 50000,
                    "system.bigCluster.cpus0.lsq0.loadToUse::mean": 4,
                    "system.bigCluster.cpus1.lsq0.loadToUse::mean": 4,
                },
                "little": {
                    "system.littleCluster.cpus0.numCycles": 100000,
                    "system.littleCluster.cpus1.numCycles": 100000,
                    "system.mem_ctrls.bytesRead::total": 200000,
                    "system.mem_ctrls.bytesRead::littleCluster.cpus0.data": 60000,
                    "system.mem_ctrls.bytesRead::littleCluster.cpus1.data": 60000,
                    "system.littleCluster.cpus0.dcache.overallAccesses::total": 20000,
                    "system.littleCluster.cpus1.dcache.overallAccesses::total": 20000,
                    "system.littleCluster.cpus0.dcache.overallMisses::total": 3000,
                    "system.littleCluster.cpus1.dcache.overallMisses::total": 3000,
                    "system.littleCluster.cpus0.commitStats0.numInsts": 80000,
                    "system.littleCluster.cpus1.commitStats0.numInsts": 80000,
                    "system.littleCluster.cpus0.commitStats0.numOps": 100000,
                    "system.littleCluster.cpus1.commitStats0.numOps": 100000,
                    "system.littleCluster.cpus0.icache.overallMisses::total": 100,
                    "system.littleCluster.cpus1.icache.overallMisses::total": 100,
                    "system.littleCluster.cpus0.dcache.WriteReq.misses::total": 650,
                    "system.littleCluster.cpus1.dcache.WriteReq.misses::total": 650,
                    "system.littleCluster.l2.overallAccesses::total": 10000,
                    "system.littleCluster.l2.overallMisses::total": 2000,
                    "system.littleCluster.cpus0.fetch2.intInstructions": 10000,
                    "system.littleCluster.cpus1.fetch2.intInstructions": 10000,
                    "system.littleCluster.cpus0.fetch2.loadInstructions": 10000,
                    "system.littleCluster.cpus1.fetch2.loadInstructions": 10000,
                    "system.littleCluster.cpus0.fetch2.storeInstructions": 1000,
                    "system.littleCluster.cpus1.fetch2.storeInstructions": 1000,
                },
            },
        }

        # Check Load States
        big_low_load = self.evaluate_load_state(current_state, thresholds, "big", "low", 1)
        little_low_load = self.evaluate_load_state(current_state, thresholds, "little", "low", 1)
        print(f"[INFO] Check Load States Summary Low Load --> (Big:{big_low_load} Little:{little_low_load}) High Load --> (Big:{not big_low_load} Little:{not little_low_load})")
        # Where: Update duration based on high load state
        if not big_low_load:

            self.duration_at_freq["big"] += 1
            print(f"[INFO] Adding to the Duration At Freq Counter Big High Load Total:{self.duration_at_freq['big']}")
        else:
            self.duration_at_freq["big"] = 0
            print(f"[INFO] REsetting to 0 to the Duration At Freq Counter Big High Load Total:{self.duration_at_freq['big']}")

        if not little_low_load:
            self.duration_at_freq["little"] += 1
            print(f"[INFO] Adding to the Duration At Freq Counter Little High Load Total:{self.duration_at_freq['little']}")
        else:
            self.duration_at_freq["little"] = 0
            print(
                f"[INFO] Resetting to 0 to the Duration At Freq Counter Little High Load Total:{self.duration_at_freq['little']}")
        # Debug Summary
        print("\n[INFO] Load State Summary")
        print(f"[DEBUG] Big Low Load: {big_low_load}, Little Low Load: {little_low_load}")
        print(f"[DEBUG] Big High Load: {not big_low_load}, Little High Load: {not little_low_load}")

        # Apply Frequency Changes
        big_action, big_freq, self.duration_at_freq["big"] = self.apply_frequency_override("big", big_low_load,  big_action, big_freq,
                                                        667111, 1510574,self.duration_at_freq["big"])
        little_action, little_freq, self.duration_at_freq["little"] = self.apply_frequency_override("little", little_low_load,
                                                              little_action, little_freq, 667111, 1510574,self.duration_at_freq["little"])

        print(
            f"[INFO] Load State Transition Check: Big (Prev: {self.prev_low_load_state['big']} → New: {big_low_load}), "
            f"Little (Prev: {self.prev_low_load_state['little']} → New: {little_low_load})"
        )

        # Transition logic for big cluster (High → Low)
        if big_low_load and not self.prev_low_load_state["big"]:  # Transition detected
            self.low_load_transition_pending["big"] = True  # First step flag
            big_freq = 667111  # Step down first
            big_action = 7
            print("[INFO] Big Cluster Transition Started: Setting intermediate frequency (1200 MHz)")

        elif big_low_load and self.low_load_transition_pending["big"]:  # Second step in next iteration
            big_freq = 667111  # Final transition
            big_action = 7
            self.low_load_transition_pending["big"] = False  # Reset flag
            print("[INFO] Big Cluster Final Transition: Setting lowest frequency (667 MHz)")

        # Transition logic for little cluster (High → Low)
        if little_low_load and not self.prev_low_load_state["little"]:
            self.low_load_transition_pending["little"] = True
            little_freq = 1000000
            little_action = 6
            print("[INFO] Little Cluster Transition Started: Setting intermediate frequency (1200 MHz)")

        elif little_low_load and self.low_load_transition_pending["little"]:
            little_freq = 667111
            little_action = 7
            self.low_load_transition_pending["little"] = False
            print("[INFO] Little Cluster Final Transition: Setting lowest frequency (667 MHz)")

        # Transition logic for big cluster (Low → High)
        if not big_low_load and self.prev_low_load_state["big"]:  # Transition detected
            self.high_load_transition_pending["big"] = True  # First step flag
            big_freq = 667111  # Step up first
            big_action = 7
            print("[INFO] Big Cluster Transition Started: Setting intermediate frequency (1200 MHz)")

        elif not big_low_load and self.high_load_transition_pending["big"]:  # Second step in next iteration
            big_freq = 667111  # Final transition (Max Frequency)
            big_action = 7
            self.high_load_transition_pending["big"] = False  # Reset flag
            print("[INFO] Big Cluster Final Transition: Setting highest frequency (2000 MHz)")

        # Transition logic for little cluster (Low → High)
        if not little_low_load and self.prev_low_load_state["little"]:
            self.high_load_transition_pending["little"] = True
            little_freq = 1000000
            little_action = 6
            print("[INFO] Little Cluster Transition Started: Setting intermediate frequency (1200 MHz)")

        elif not little_low_load and self.high_load_transition_pending["little"]:
            little_freq = 1200480  # Final transition (Max Frequency for Little Cores)
            little_action = 5
            self.high_load_transition_pending["little"] = False
            print("[INFO] Little Cluster Final Transition: Setting highest frequency (1600 MHz)")

        # Update previous load states
        self.prev_low_load_state["big"] = big_low_load
        self.prev_low_load_state["little"] = little_low_load

        return big_action, little_action, big_freq, little_freq

    def wait_for_frequency_change(self, big_change_needed, little_change_needed, expected_big_freq,
                                  expected_little_freq,max_wait_time_input):
        """
        Wait dynamically until the new frequencies are reflected in the stats.txt file.

        This implementation continuously polls the stats generator for new chunks
        until the expected frequency changes are detected or the timeout is reached.
        """
        print("[INFO] Waiting for frequency changes to reflect in stats.txt...")
        #print(f"[DEBUG] Initial Chunk Count: {len(accumulated_chunks)}")
        start_poll_time = time.time()  # Track time for polling intervals
        max_wait_time = max_wait_time_input  # Maximum wait time in seconds
        start_time = time.time()
        polling_interval = 0.2  # Initial polling interval in seconds
        max_polling_interval = 1.0  # Max interval to avoid excessive waiting
        print(f"[INFO] Waiting for frequency changes. Max wait time: {max_wait_time}s")  # Added
        accumulated_chunks = []  # Initialize the variable at the beginning
        while time.time() - start_time < max_wait_time:
            try:
                # Dynamically fetch the next chunk from the generator
                #current_state = next(self.stats_generator) heiner removed next
                current_state = self.stats_generator[-1]
                #print(f"[DEBUG] Current State:{current_state}")
                try:
                    accumulated_chunks.append(current_state)
                except Exception as e:
                    print(f"[ERROR] Failed to append chunk: {e}")

                # Extract frequency values from the current chunk
                big_freq_ticks = current_state.get("system.bigCluster.clk_domain.clock", 0)
                little_freq_ticks = current_state.get("system.littleCluster.clk_domain.clock", 0)
                # Debugging: Log tick values
                print(f"[DEBUG] Ticks -> Big: {big_freq_ticks}, Little: {little_freq_ticks}")  # Added
                # Convert ticks to MHz
                #current_big_freq = ticks_to_hz(big_freq_ticks)
                #current_little_freq = ticks_to_hz(little_freq_ticks)

                current_big_freq = ticks_to_hz_freqchange(big_freq_ticks)
                current_little_freq = ticks_to_hz_freqchange(little_freq_ticks)


                # Debugging: Log the observed and expected frequencies
                time_passed = time.time() - start_time
                print(
                    f"[DEBUG] Current Frequencies -> Big: {current_big_freq} MHz, Little: {current_little_freq} MHz. Time:{time_passed}s"
                )
                print(
                    f"[DEBUG] Expected Frequencies -> Big: {expected_big_freq} MHz, Little: {expected_little_freq} MHz.Time:{time_passed}s"
                )

                # Check if the expected frequencies match the current values
                #print(f"[DEBUG] big_change_neeeded:{big_change_needed} little_change_needed:{little_change_needed}")
                #print(f"[DEBUG] abs(current_big_freq - expected_big_freq) : {abs(current_big_freq - expected_big_freq)}")
                #print(f"[DEBUG] abs(current_little_freq - expected_little_freq):{abs(current_little_freq - expected_little_freq)}")
                big_done = not big_change_needed or abs(current_big_freq - expected_big_freq) < 2
                little_done = not little_change_needed or abs(current_little_freq - expected_little_freq) < 2

                if big_done and little_done:
                    print(
                        f"[INFO] Frequency changes verified successfully: Big={expected_big_freq} MHz, "
                        f"Little={expected_little_freq} MHz."
                    )

                    # Define how many new chunks we need before proceeding
                    CHUNKS_TO_WAIT = self.chunk_user_var_to_wait
                    new_chunk_count = 0  # Tracks new unique chunks received

                    # Timer tracking
                    start_time2 = time.time()
                    waiting_printed = False  # Ensure waiting message prints only once

                    while new_chunk_count < CHUNKS_TO_WAIT:
                        if self.is_duplicate_chunk(current_state):
                            if not waiting_printed:
                                print("[WARNING] Duplicate chunk detected. Waiting for new chunks to move on.")
                                waiting_printed = True  # Prevents further printing

                            current_state = self.stats_generator[-1]  # Fetch latest state

                        else:
                            new_chunk_count += 1
                            print(f"[INFO] Received chunk {new_chunk_count} of {CHUNKS_TO_WAIT}, waiting for next...")

                            # Store the latest chunk to compare for the next loop iteration
                            self.last_state = current_state.copy()

                    # After exiting the loop, calculate the total wait time
                    total_wait_time2 = time.time() - start_time2
                    print(f"[INFO] {CHUNKS_TO_WAIT} chunks received successfully after {total_wait_time2:.4f} seconds.")
                    return accumulated_chunks  # Return collected stats
                    #return

            except StopIteration:
                print("[WARNING] Stats generator completed or no more data available.")
                break
            except Exception as e:
                print(f"[ERROR] Exception while waiting for frequency change: {e}")
                break

            # Adjust polling interval dynamically (optional) but check the frequency more often if needed
            time.sleep(0.2)  # Adjust this value based on your data generation speed

        # Timeout warning if frequency changes are not detected
        print("[WARNING] Frequency changes were not verified within the maximum wait time.")
        return accumulated_chunks  # Return whatever was collected


    def prepare_combined_state(self,accumulated_chunks, new_chunks, weight_accumulated=0.3, weight_new=0.7):
        """
        Combine accumulated and new chunks into a single state with weighted priority.

        Parameters:
        - accumulated_chunks: List of chunks accumulated during the wait time.
        - new_chunks: List of chunks collected after the frequency change.
        - weight_accumulated: Weight for accumulated chunks.
        - weight_new: Weight for new chunks.

        Returns:
        - combined_state: Weighted combination of accumulated and new chunks as a single state.
        """
        if not accumulated_chunks and not new_chunks:
            print("[ERROR] Both accumulated and new chunks are empty. Returning default zero state.")
            return np.zeros((1, len(next(iter(accumulated_chunks or new_chunks)).keys())), dtype=np.float32)

        combined_values = {}
        keys = accumulated_chunks[0].keys() if accumulated_chunks else new_chunks[0].keys()

        for key in keys:
            accumulated_avg = (
                sum(chunk.get(key, 0.0) for chunk in accumulated_chunks) / len(accumulated_chunks)
                if accumulated_chunks else 0.0
            )
            new_avg = (
                sum(chunk.get(key, 0.0) for chunk in new_chunks) / len(new_chunks)
                if new_chunks else 0.0
            )
            combined_values[key] = (weight_accumulated * accumulated_avg) + (weight_new * new_avg)

        # Convert combined values into a NumPy array
        combined_state = np.array(list(combined_values.values()), dtype=np.float32).reshape(1, -1)
        print(f"[DEBUG] Combined State: {combined_state}")
        if np.any(np.isnan(combined_state)):
            print("[ERROR] Combined state contains NaN values! Sanitizing...")
            print("[ERROR] Inside the prepare_combined_state")
            combined_state = np.nan_to_num(combined_state, nan=0.0, posinf=1e6, neginf=-1e6)

        if np.any(np.isinf(combined_state)):
            print("[ERROR] Combined state contains infinite values! Sanitizing...")
            combined_state = np.nan_to_num(combined_state, nan=0.0, posinf=1e6, neginf=-1e6)

        print(f"[DEBUG] Final Combined State: {combined_state}")
        return combined_state

    def step(self, action):
        mode = self.run_mode
        try:
            # Debugging the received action
            print(f"[DEBUG] Original action received: {action}")

            # Validate action before any processing
            if isinstance(action, np.ndarray):
                action = action.tolist()  # Ensure action is a Python list

            if isinstance(action, (list, np.ndarray)):
                # Flatten the action if it's nested (e.g., [[4, 5]])
                print(f"[DEBUG] Action BEFORE flattening in step(): {action}, Length: {len(action)}")
                action = np.array(action).flatten().tolist()
                print(f"[DEBUG] Received Flattened action: {action}")
                # Ensure action has exactly 2 elements
                if len(action) != 2:
                    print(f"[ERROR] Invalid action length in step() function: {len(action)}. Expected length: 2.")
                    raise ValueError(f"Invalid action length: {len(action)}. Expected length: 2.")

            # Ensure action format and separate into big and little actions
            if not isinstance(action, list) or len(action) != 2:
                print(f"[ERROR] Invalid action format: {action}. Flattening")
                print(f"[ERROR] Inside Step function")
                #action = [0, 0]
                print(f"[DEBUG] Action BEFORE flattening in step(): {action}, Length: {len(action)}")
                action = np.array(action).flatten().tolist()
                print(f"[DEBUG] Received Flattened action: {action}")
                # Ensure action has exactly 2 elements
                if len(action) != 2:
                    print(f"[ERROR] Invalid action length in step() function: {len(action)}. Expected length: 2.")
                    raise ValueError(f"Invalid action length: {len(action)}. Expected length: 2.")
                print(f"[ERROR] Invalid action length in step() function: {len(action)}. Expected length: 2.")

            # Separate actions for big and little cores
            big_action, little_action = action
            model_action = action

            # Validate individual actions
            if not isinstance(big_action, int) or big_action < 0 or big_action >= len(self.big_vf_curve):
                print(f"[ERROR] Big Action {big_action} is out of bounds! Defaulting to 0.")
                big_action = 0
            if not isinstance(little_action, int) or little_action < 0 or little_action >= len(self.little_vf_curve):
                print(f"[ERROR] Little Action {little_action} is out of bounds! Defaulting to 0.")
                little_action = 0

            # Extract frequency and voltage for the chosen actions
            big_freq, big_voltage = self.big_vf_curve[big_action]
            little_freq, little_voltage = self.little_vf_curve[little_action]

            # Monitor low-load signals
            #current_state = self.stats_chunks[self.current_chunk_index] hjsolise
            current_state = self.stats_generator[-1]
            # Extract simSeconds value for the current step
            sim_seconds = current_state.get("simSeconds", 0.0)

            # Skip time accumulation if the chunk is a duplicate
            if self.is_duplicate_chunk(current_state):
                print("[WARNING] Duplicate chunk detected. Skipping time accumulation.")
            else:
                self.simulation_time += sim_seconds  # Only accumulate if the chunk is new

            # Store the current state as last_state for next comparison
            self.last_state = current_state.copy()
            print(f"[TIME] Simulation Currently at {self.simulation_time:.4f} seconds")

            # Delta-based signal calculation for big and little clusters
            ###High/Low Load Mode Check
            # Timer tracking
            start_time2 = time.time()
            waiting_printed = False  # Ensure waiting message prints only once
            while self.is_duplicate_chunk(current_state):
                if not waiting_printed:
                    print("[WARNING] Duplicate chunk detected. Waiting for new chunk to move on.")
                    waiting_printed = True  # Prevents further printing

                current_state = self.stats_generator[-1]

            # After exiting the loop, calculate the total wait time
            total_wait_time2 = time.time() - start_time2

            # Print total wait time only if there was a delay
            if waiting_printed:
                print(f"[INFO] Fetching new data took {total_wait_time2:.4f} seconds.")

            # Reset timer variables (not necessary here but good practice)
            start_time = 0

            print(f"[DEBUG] Action BEFORE check_load_state(): {big_action}, {little_action}")
            big_action, little_action, big_freq, little_freq = self.check_load_state(current_state, big_action,
                                                                                little_action, big_freq, little_freq)
            print(f"[DEBUG] Action AFTER check_load_state(): {big_action}, {little_action}")

            # Validate action size
            if isinstance(big_action, list) or isinstance(little_action, list):
                print(f"[ERROR] Actions became lists unexpectedly! Big: {big_action}, Little: {little_action}")
                raise ValueError("Actions should not be lists at this stage!")

            # Map frequencies to the correct format
            mapped_big_freq = map_big_core_frequency(int(big_freq))
            mapped_little_freq = map_little_core_frequency(int(little_freq))

            print(f"[DEBUG] Big Cluster -> Model Expected Freq: {mapped_big_freq}, Voltage: {big_voltage}")
            print(f"[DEBUG] Little Cluster -> Model Expected Freq: {mapped_little_freq}, Voltage: {little_voltage}")

            # Extract the current state
            current_state = self.stats_generator[-1]
            current_big_freq_ticks = current_state.get("system.bigCluster.clk_domain.clock", 0)
            current_little_freq_ticks = current_state.get("system.littleCluster.clk_domain.clock", 0)

            # Convert ticks to MHz for debugging
            current_big_freq = ticks_to_hz_freqchange(current_big_freq_ticks)
            current_little_freq = ticks_to_hz_freqchange(current_little_freq_ticks)

            print(f"[DEBUG] Current Big Cluster Freq (MHz): {current_big_freq}")
            print(f"[DEBUG] Current Little Cluster Freq (MHz): {current_little_freq}")
            if mode == "live":
                # Determine if frequency change is needed
                big_freq_change_needed = current_big_freq != mapped_big_freq
                little_freq_change_needed = current_little_freq != mapped_little_freq

                # Store chunks during the wait
                accumulated_chunks = []

                # Send Telnet commands if frequency change is needed
                if big_freq_change_needed or little_freq_change_needed:
                    print(f"[INFO] #####Freq Change Happening#####")
                    #print(f"[INFO] Frequency change required. Sending commands...")
                    print(f"[INFO] Frequency change required. BigCores:{big_freq_change_needed} LittleCores:{little_freq_change_needed}")
                    print(f"[INFO] BigCore Current:{current_big_freq} BigCore Expected:{mapped_big_freq}")
                    print(f"[INFO] LittleCore Current:{current_little_freq} LittleCore Expected:{mapped_little_freq}")

                    try:
                        if big_freq_change_needed:
                            print(f"[INFO] Sending frequency change for Big Cores: {mapped_big_freq} MHz")
                            #set_frequencies(self.host, self.port, [mapped_big_freq] * self.big_cores, [])
                            set_frequencies(
                                self.host,
                                self.port,
                                simulation_path,
                                big_frequency=mapped_big_freq,
                                little_frequency=mapped_little_freq,
                                set_big=big_freq_change_needed,
                                set_little=None
                            )
                            print(f"[INFO] Sent new frequency for Big Cores: {mapped_big_freq} MHz")
                        if little_freq_change_needed:
                            print(f"[INFO] Sending frequency change for Little Cores: {mapped_little_freq} MHz")
                            #set_frequencies(self.host, self.port, [], [mapped_little_freq] * self.little_cores)
                            set_frequencies(
                                self.host,
                                self.port,
                                simulation_path,
                                big_frequency=mapped_big_freq,
                                little_frequency=mapped_little_freq,
                                set_big=None,
                                set_little=little_freq_change_needed
                            )
                            print(f"[INFO] Sent new frequency for Little Cores: {mapped_little_freq} MHz")
                    except Exception as telnet_error:
                        print(f"[ERROR] Failed to send frequency commands: {telnet_error}")

                    # Wait dynamically for frequency change and collect accumulated chunks
                    #self.wait_for_frequency_change(big_freq_change_needed, little_freq_change_needed, mapped_big_freq,mapped_little_freq,args.freqwait)
                    # Wait dynamically for frequency change and collect accumulated chunks

                    accumulated_chunks = self.wait_for_frequency_change(
                        big_freq_change_needed, little_freq_change_needed, mapped_big_freq, mapped_little_freq,
                        args.freqwait
                    )
                else:
                    print(f"[INFO] No Freq Chnage Needed Big_Freq_Change_Flag-->{big_freq_change_needed} Little_Freq_Change_Flag-->{little_freq_change_needed}")
                # Process additional chunks after frequency change
                new_chunks = []
                for _ in range(args.process_chunks_num):
                    try:
                        #new_chunk = next(self.stats_generator)
                        new_chunk = self.stats_generator[-1]
                        new_chunks.append(new_chunk)
                    except StopIteration:
                        print("[WARNING] Not enough chunks available for the requested count.")
                        break

                # Combine accumulated chunks and new chunks
                all_chunks = accumulated_chunks + new_chunks
                # Combine accumulated chunks and new chunks
                self.stats_chunks.extend(accumulated_chunks + new_chunks)
                # Validate chunk index
                if self.current_chunk_index >= len(self.stats_chunks):
                    print(
                        f"[WARNING] Current chunk index {self.current_chunk_index} exceeds stats_chunks size {len(self.stats_chunks)}. Resetting.")
                    self.current_chunk_index = len(self.stats_chunks) - 1
                print(
                    f"[INFO] {len(all_chunks)} chunks processed: {len(accumulated_chunks)} from wait + {len(new_chunks)} post-frequency.")
                print(f"[INFO] {len(all_chunks)} chunks processed in total.")
                print(f"[DEBUG] Accumulated Chunks: {len(accumulated_chunks)}")
                print(f"[DEBUG] Post-Frequency Change Chunks: {len(new_chunks)}")


                # Combine accumulated and new chunks into a single state
                combined_state = self.prepare_combined_state(accumulated_chunks, new_chunks)

                # Validate combined state
                if np.any(np.isnan(combined_state)):
                    print("[ERROR] Combined state contains NaN values! Sanitizing...")
                    print("[ERROR] Inside the step function")
                    combined_state = np.nan_to_num(combined_state, nan=0.0, posinf=1e6, neginf=-1e6)

                if np.any(np.isinf(combined_state)):
                    print("[ERROR] Combined state contains infinite values! Sanitizing...")
                    combined_state = np.nan_to_num(combined_state, nan=0.0, posinf=1e6, neginf=-1e6)

                print(f"[DEBUG] Combined State Prepared: {combined_state}")

            # Calculate the reward based on the actions and the processed chunks
            action_taken = [big_action, little_action]
            big_model_action, little_model_action = model_action
            reward = self.calculate_reward(big_action, little_action)
            # Apply a penalty if the model's actions do not match the overridden actions
            if big_model_action != big_action or little_model_action != little_action:
                penalty = -1  # Add a configurable penalty
                print(f"[DEBUG] Applying penalty for mismatched actions. Penalty: {penalty}")
                print(f"[DEBUG] Model_Big_Action:{big_model_action} Big_Action_Taken:{big_action}")
                print(f"[DEBUG] Model_Big_Action:{little_model_action} Big_Action_Taken:{little_action}")
                reward += penalty
            print(f"[DEBUG] Calculated Reward: {reward}")

            # Update the environment state
            self.current_chunk_index += 1
            print("[DEBUG] Update the environment state")
            print(f"[DEBUG] self.current_chunk_index:{self.current_chunk_index} len(self.stats_chunks):{len(self.stats_chunks)}")
            terminated = self.current_chunk_index >= len(self.stats_chunks)
            if mode == "live":
                print(
                    f"[DEBUG] Terminated is :{terminated} going to episode done inside Step function")
                truncated = not terminated and len(new_chunks) == 0 and len(accumulated_chunks) == 0
            else:
                truncated = not terminated


            # Prepare the next state
            if terminated:
                next_state = np.zeros(len(current_state))  # Terminal state as a NumPy array
                print("[ERROR] Episode done. Terminal state returned.")
            else:
                next_state = np.array(list(self.stats_chunks[self.current_chunk_index].values()), dtype=np.float32)

            # Validate next state for NaN or Inf
            if np.any(np.isnan(next_state)):
                print("[ERROR] Next state contains NaN values! Sanitizing...")
                next_state = np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)
            if np.any(np.isinf(next_state)):
                print("[ERROR] Next state contains infinite values! Sanitizing...")
                next_state = np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)



            print(f"[DEBUG] Next State: {next_state}")
            if mode == "live":
                print(f"[DEBUG] Terminated: {terminated}, Truncated: {truncated}")

            # [AFTER CHANGE] Decide whether to use combined_state or next_state

                final_state = combined_state if not terminated else next_state
            else:
                final_state = next_state
            final_state = np.array(final_state, dtype=np.float32)  # Ensure correct format
            if np.any(np.isnan(final_state)) or np.any(np.isinf(final_state)):
                print("[ERROR] Next state contains NaN or infinite values! Replacing with zeros.")
                final_state = np.nan_to_num(final_state, nan=0.0, posinf=1e6, neginf=-1e6)
            #action_taken = [big_action,little_action]
            print(f"[DEBUG] Finishing step variables final_state:{final_state} reward:{reward} terminated:{terminated} truncated:{truncated} action_taken:{action_taken}")
            #return final_state, reward, terminated, truncated, action_taken, {}
            # return final_state, reward, terminated, truncated, action_taken, {}
            #return final_state, reward, terminated, truncated, action_taken
            return final_state, reward, terminated, truncated, {"action_taken": action_taken}

        except Exception as e:
            print(f"[ERROR] Exception in step function: {e}")
            # Capture and print the full traceback
            traceback_str = traceback.format_exc()
            print(f"[ERROR] Full Traceback:\n{traceback_str}")
            exit()
            return np.zeros(self.observation_space.shape), 0.0, True, False, {"error": str(e)}

    def calculate_reward(self, big_action, little_action):
        try:
            #current_state = self.stats_chunks[self.current_chunk_index]
            current_state =self.stats_generator[-1]
            #print(f"[DEBUG] Current State in calculate_reward: {current_state}")

            # Dynamic Performance Thresholds
            if not hasattr(self, 'big_performance_threshold'):
                self.big_performance_threshold = np.mean([
                    (chunk.get("system.bigCluster.cpus0.iew.instsToCommit", 0) +
                     chunk.get("system.bigCluster.cpus1.iew.instsToCommit", 0)) /
                    chunk.get("simSeconds", 1)
                    for chunk in self.stats_chunks if chunk.get("simSeconds", 1) > 0
                ])
                self.little_performance_threshold = np.mean([
                    (chunk.get("system.littleCluster.cpus0.commitStats0.numInsts", 0) +
                     chunk.get("system.littleCluster.cpus1.commitStats0.numInsts", 0)) /
                    chunk.get("simSeconds", 1)
                    for chunk in self.stats_chunks if chunk.get("simSeconds", 1) > 0
                ])
                print(f"[DEBUG] Calculated Big Performance Threshold: {self.big_performance_threshold}")
                print(f"[DEBUG] Calculated Little Performance Threshold: {self.little_performance_threshold}")

            # Performance metrics
            big_performance = (
                                      current_state.get("system.bigCluster.cpus0.iew.instsToCommit", 0) +
                                      current_state.get("system.bigCluster.cpus1.iew.instsToCommit", 0)
                              ) / current_state.get("simSeconds", 1)
            little_performance = (
                                         current_state.get("system.littleCluster.cpus0.commitStats0.numInsts", 0) +
                                         current_state.get("system.littleCluster.cpus1.commitStats0.numInsts", 0)
                                 ) / current_state.get("simSeconds", 1)

            # Power metrics
            big_dynamic_power = calculate_dynamic_power_a15(current_state)
            big_static_power = calculate_static_power_a15(current_state)
            big_total_power = big_dynamic_power + big_static_power
            print(
                f"[DEBUG] Power -> Big: Dynamic={big_dynamic_power}, Static={big_static_power}, Total={big_total_power}")  # Added

            little_dynamic_power = calculate_dynamic_power_a7(current_state)
            little_static_power = calculate_static_power_a7(current_state)
            little_total_power = little_dynamic_power + little_static_power
            print(
                f"[DEBUG] Power -> Little: Dynamic={little_dynamic_power}, Static={little_static_power}, Total={little_total_power}")  # Added

            # Performance rewards
            big_perf_reward = (
                10 if big_performance >= self.big_performance_threshold else -10
            )
            little_perf_reward = (
                10 if little_performance >= self.little_performance_threshold else -10
            )

            # Power penalties (normalized by a scaling factor for interpretability)
            big_power_penalty = big_total_power / 1e3  # Scale factor to reduce magnitude
            little_power_penalty = little_total_power / 1e3

            # Combine rewards and penalties
            total_reward = (
                    big_perf_reward - big_power_penalty +
                    little_perf_reward - little_power_penalty
            )

            print(f"[DEBUG] Performance -> Big: {big_performance}, Little: {little_performance}")
            print(f"[DEBUG] Power -> Big: {big_total_power}, Little: {little_total_power}")
            print(
                f"[DEBUG] Thresholds -> Big: {self.big_performance_threshold}, Little: {self.little_performance_threshold}")
            print(f"[DEBUG] Total Reward: {total_reward}")

            # Ensure reward is valid
            if np.isnan(total_reward) or np.isinf(total_reward):
                print("[ERROR] Total reward is invalid, setting to 0.")
                total_reward = 0.0
            print(f"[DEBUG] Reward Calculation -> Big: {big_perf_reward} (Penalty: {big_power_penalty}), "
                  f"Little: {little_perf_reward} (Penalty: {little_power_penalty})")
            return total_reward

        except Exception as e:
            print(f"[ERROR] Exception in calculate_reward: {e}")
            return 0.0

    def is_duplicate_chunk(self, current_state):
        """
        Compare all extracted variables to determine if the chunk is a duplicate.
        If all 25 extracted variables are the same, return True (indicating duplicate).
        """
        if self.last_state is None:
            return False  # No previous state, so it's not a duplicate

        # Compare each extracted variable from parse_log_chunk
        for key in self.last_state.keys():
            if key not in current_state or self.last_state[key] != current_state[key]:
                return False  # At least one variable is different → NOT a duplicate

        return True  # All variables are the same → Duplicate chunk

def parse_vf_curve(voltage, clock):
    parsed_voltage = [float(v[:-1]) for v in voltage]
    parsed_clock = [float(c[:-3]) for c in clock]

    # Validate values
    for i, (v, c) in enumerate(zip(parsed_voltage, parsed_clock)):
        if np.isnan(v) or np.isnan(c) or np.isinf(v) or np.isinf(c):
            print(f"[ERROR] Invalid VF curve entry at index {i}: Voltage={v}, Clock={c}. Replacing with defaults.")
            parsed_voltage[i] = 1.0  # Default voltage
            parsed_clock[i] = 1000.0  # Default frequency

    return list(zip(parsed_clock, parsed_voltage))


def watch_stats_file(file_path, interval, chunk_timeout, stats_deque):
    """
    Watch a stats file and append parsed chunks to a shared deque in real-time.

    Parameters:
    - file_path (str): Path to the stats file.
    - interval (float): Interval in seconds to wait between file checks.
    - chunk_timeout (float): Maximum time in seconds to wait for new chunks.
    - stats_deque (deque): Shared deque to store the parsed chunks.
    """
    total_wait_time = 0
    start_time = time.time()  # Start the timeout timer
    file_offset = 0  # Track the file offset for resuming

    while True:
        try:
            with open(file_path, 'r') as file:
                file.seek(file_offset)  # Resume reading from the last position
                chunk = []
                recording = False

                while True:
                    line = file.readline()

                    if not line:  # If no new line is found, wait for a while and check again
                        #print(f"[DEBUG] No new data. Waiting for {interval}s.") hjsolise
                        time.sleep(interval)
                        total_wait_time += interval

                        # Timeout condition if no new data after chunk_timeout
                        if time.time() - start_time >= chunk_timeout:
                            print(f"[WARNING] Timeout: No new chunks after {chunk_timeout}s.")
                            print("[INFO] Guardando el modelo antes de salir...")

                            # Save model before exiting due to timeout
                            if 'model' in globals():
                                model.save(MODEL_PATH)
                                print("[INFO] Modelo guardado antes de salir por timeout.")
                            else:
                                print(
                                    "[WARNING] No se encontró el modelo en el ámbito global antes de salir por timeout.")

                            print(f"[WARNING] Exiting Script.")
                            os._exit(1)  # Force exit immediately
                            return  # Exit if timeout occurs
                        continue

                    # Reset the timeout timer when new data is found
                    start_time = time.time()
                    total_wait_time = 0  # Reset wait time counter
                    line = line.strip()

                    # Start recording when 'Begin' is found
                    if "Begin Simulation Statistics" in line:
                        recording = True
                        chunk = []  # Reset chunk list when new data starts

                    # Stop recording and process chunk when 'End' is found
                    elif "End Simulation Statistics" in line and recording:
                        recording = False
                        try:
                            parsed_chunk = parse_log_chunk(chunk)
                            if parsed_chunk:
                                # Append the parsed chunk to the shared deque
                                stats_deque.append(parsed_chunk)
                                #print(f"[DEBUG] Appended chunk to deque. Deque size: {len(stats_deque)}") hjsolise
                            else:
                                print(f"[ERROR] Failed to parse chunk.")
                        except Exception as e:
                            print(f"[ERROR] Exception during chunk parsing: {e}")

                    # Collect data in the chunk while recording
                    if recording:
                        chunk.append(line)

                # Update the file offset for the next iteration
                file_offset = file.tell()

        except FileNotFoundError:
            print(f"[ERROR] File {file_path} not found. Retrying after {interval}s...")
            time.sleep(interval)
        except Exception as e:
            print(f"[ERROR] Exception in watch_stats_file: {e}")
            time.sleep(interval)

# Función para procesar cada chunk de stats.txt
def parse_log_chunk(chunk):
    # Diccionario de variables inicializado con valores por defecto
    variables = {
        # A15 Variables
        "system.bigCluster.cpus0.numCycles": 0.0,
        "system.bigCluster.cpus1.numCycles": 0.0,
        "simSeconds": 0.0,
        "system.bigCluster.clk_domain.clock": 0.0,
        "system.bigCluster.voltage_domain.voltage": 0.0,
        "system.bigCluster.cpus0.dcache.overallAccesses::total": 0.0,
        "system.bigCluster.cpus1.dcache.overallAccesses::total": 0.0,
        "system.bigCluster.cpus0.iew.instsToCommit": 0.0,
        "system.bigCluster.cpus1.iew.instsToCommit": 0.0,
        "system.bigCluster.cpus0.statFuBusy::IntAlu": 0.0,
        "system.bigCluster.cpus0.statFuBusy::IntMult": 0.0,
        "system.bigCluster.cpus0.statFuBusy::IntDiv": 0.0,
        "system.bigCluster.cpus1.statFuBusy::IntAlu": 0.0,
        "system.bigCluster.cpus1.statFuBusy::IntMult": 0.0,
        "system.bigCluster.cpus1.statFuBusy::IntDiv": 0.0,
        "system.bigCluster.cpus0.dcache.WriteReq.misses::total": 0.0,
        "system.bigCluster.cpus1.dcache.WriteReq.misses::total": 0.0,
        "system.bigCluster.l2.overallAccesses::total": 0.0,
        "system.bigCluster.l2.overallAccesses::bigCluster.cpus0.data": 0.0,
        "system.bigCluster.l2.overallAccesses::bigCluster.cpus1.data": 0.0,
        "system.bigCluster.cpus0.icache.ReadReq.accesses::total": 0.0,
        "system.bigCluster.cpus1.icache.ReadReq.accesses::total": 0.0,

        # A7 Variables
        "system.littleCluster.cpus0.numCycles": 0.0,
        "system.littleCluster.cpus1.numCycles": 0.0,
        "system.littleCluster.clk_domain.clock": 0.0,
        "system.littleCluster.voltage_domain.voltage": 0.0,
        "system.littleCluster.cpus0.commitStats0.numInsts": 0.0,
        "system.littleCluster.cpus1.commitStats0.numInsts": 0.0,
        "system.littleCluster.cpus0.dcache.overallAccesses::total": 0.0,
        "system.littleCluster.cpus1.dcache.overallAccesses::total": 0.0,
        "system.littleCluster.cpus0.dcache.overallMisses::total": 0.0,
        "system.littleCluster.cpus1.dcache.overallMisses::total": 0.0,
        "system.mem_ctrls.bytesRead::total": 0.0,
        "system.mem_ctrls.bytesRead::littleCluster.cpus0.data": 0.0,
        "system.mem_ctrls.bytesRead::littleCluster.cpus1.data": 0.0,

        # Métricas de Instrucciones
        "system.bigCluster.cpus0.numInsts": 0.0,
        "system.littleCluster.cpus0.numInsts": 0.0,
        "system.bigCluster.cpus1.numInsts": 0.0,
        "system.littleCluster.cpus1.numInsts": 0.0,

        # Métricas de Uso y Ocupación
        "system.bigCluster.cpus0.fuBusyRate": 0.0,
        "system.littleCluster.cpus0.fuBusyRate": 0.0,
        "system.bigCluster.cpus1.fuBusyRate": 0.0,
        "system.littleCluster.cpus1.fuBusyRate": 0.0,
        "system.bigCluster.cpus0.branchPred.mispredicted": 0.0,
        "system.littleCluster.cpus0.branchPred.mispredicted": 0.0,
        "system.bigCluster.cpus1.branchPred.mispredicted": 0.0,
        "system.littleCluster.cpus1.branchPred.mispredicted": 0.0,

        # Métricas de Caché y Memoria
        "system.bigCluster.cpus0.dcache.overallMisses::total": 0.0,
        "system.littleCluster.cpus0.dcache.overallMisses::total": 0.0,
        "system.bigCluster.cpus0.icache.overallMisses::total": 0.0,
        "system.littleCluster.cpus0.icache.overallMisses::total": 0.0,
        "system.bigCluster.cpus0.MemDepUnit.insertedLoads": 0.0,
        "system.littleCluster.cpus0.MemDepUnit.insertedLoads": 0.0,
        "system.bigCluster.cpus0.MemDepUnit.conflictingLoads": 0.0,
        "system.littleCluster.cpus0.MemDepUnit.conflictingLoads": 0.0,

        ###High Load Variables
        ###Little Cluster
        "system.littleCluster.cpus0.commitStats0.numOps": 0.0,
        "system.littleCluster.cpus1.commitStats0.numOps": 0.0,
        "system.littleCluster.cpus1.icache.overallMisses::total": 0.0,
        "system.littleCluster.cpus0.dcache.WriteReq.misses::total": 0.0,
        "system.littleCluster.cpus1.dcache.WriteReq.misses::total": 0.0,
        "system.littleCluster.l2.overallAccesses::total": 0.0,
        "system.littleCluster.l2.overallMisses::total": 0.0,
        "system.littleCluster.cpus0.fetch2.intInstructions": 0.0,
        "system.littleCluster.cpus1.fetch2.intInstructions": 0.0,
        "system.littleCluster.cpus0.fetch2.loadInstructions": 0.0,
        "system.littleCluster.cpus1.fetch2.loadInstructions": 0.0,
        "system.littleCluster.cpus0.fetch2.storeInstructions": 0.0,
        "system.littleCluster.cpus1.fetch2.storeInstructions": 0.0,
        ###Big Cluster
        "system.bigCluster.cpus0.decode.blockedCycles": 0.0,
        "system.bigCluster.cpus1.decode.blockedCycles": 0.0,
        "system.bigCluster.cpus0.rename.ROBFullEvents": 0.0,
        "system.bigCluster.cpus1.rename.ROBFullEvents": 0.0,
        "system.bigCluster.cpus0.dcache.demandAvgMissLatency::total": 0.0,
        "system.bigCluster.cpus1.dcache.demandAvgMissLatency::total": 0.0,
        "system.bigCluster.cpus0.icache.demandAvgMissLatency::total": 0.0,
        "system.bigCluster.cpus1.icache.demandAvgMissLatency::total": 0.0,
        "system.bigCluster.cpus0.dcache.demandAvgMshrMissLatency::total": 0.0,
        "system.bigCluster.cpus1.dcache.demandAvgMshrMissLatency::total": 0.0,
        "system.bigCluster.cpus0.mmu.itb_walker.walkServiceTime::mean": 0.0,
        "system.bigCluster.cpus1.mmu.itb_walker.walkServiceTime::mean": 0.0,
        "system.bigCluster.cpus0.fetch.cycles": 0.0,
        "system.bigCluster.cpus1.fetch.cycles": 0.0,
        "system.bigCluster.cpus0.lsq0.loadToUse::mean": 0.0,
        "system.bigCluster.cpus1.lsq0.loadToUse::mean": 0.0,

    }
    #print(f"[DEBUG] Starting to process log chunk with {len(chunk)} lines.")  # Added hjsolise
    for line in chunk:
        parts = line.split()
        if len(parts) < 2:  # Skip lines with fewer than two parts
            continue
        if parts and parts[0] in variables:
            #variables[parts[0]] = float(parts[1])
            try:
                variables[parts[0]] = float(parts[1])
            except ValueError:
                print(f"[ERROR] Failed to parse value for {parts[0]}: {parts[1]}")  # Added
            #print(f"{parts[0]} Value:{float(parts[1])}")  # Debug print
        #print(f"[DEBUG] Processed log chunk. Extracted variables: {list(variables.keys())}")  # Added

    return variables


# Procesar el archivo stats.txt en chunks
def process_stats_file(file_path):
    stats_chunks = []
    with open(file_path, 'r') as file:
        chunk = []
        recording = False

        for line in file:
            line = line.strip()
            if "Begin Simulation Statistics" in line:
                recording = True
                chunk = []  # Start a new chunk
            elif "End Simulation Statistics" in line and recording:
                recording = False
                stats_chunks.append(parse_log_chunk(chunk))
            if recording:
                chunk.append(line)

    return stats_chunks


# Crear un entorno vectorizado con AsyncVectorEnv
def create_env(stats_chunks, stats_generator,big_cores, little_cores, threads, big_vf_curve, little_vf_curve, host, port):
    try:
        print(f"[DEBUG] Initializing Environment with {len(stats_chunks)} chunks.")
        print(f"[DEBUG] Big VF Curve: {big_vf_curve}")
        print(f"[DEBUG] Little VF Curve: {little_vf_curve}")
        assert len(big_vf_curve) > 0, "Big VF Curve is empty!"
        assert len(little_vf_curve) > 0, "Little VF Curve is empty!"
        return Gem5DVFSEnv(stats_chunks, stats_generator,big_cores, little_cores, threads, big_vf_curve, little_vf_curve, host, port)
    except Exception as e:
        print(f"[ERROR] Exception in create_env: {e}")
        raise




# Entrenamiento del modelo de PPO
def train_or_load_model(stats_chunks, stats_generator,big_cores, little_cores, threads, big_vf_curve, little_vf_curve, host, port,
                        mode):
    """
    Train or load the PPO model based on the selected mode.
    - In 'train' mode: process stats_chunks and save the model.
    - In 'live' mode: continue processing and fine-tuning the existing model.
    """
    env = create_env(stats_chunks, stats_generator,big_cores, little_cores, threads, big_vf_curve, little_vf_curve, host, port)

    if mode == 'train':
        print("[INFO] Training mode selected.")
        if os.path.exists(MODEL_PATH):
            model = PPO.load(MODEL_PATH, env=env)
            print("[INFO] Loaded existing model for additional training.")
        else:
            model = PPO("MlpPolicy", env, verbose=1)
            print("[INFO] No existing model found. Training from scratch.")

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./', name_prefix='ppo_dvfs_checkpoint')
        model.learn(total_timesteps=10000, callback=checkpoint_callback)
        model.save(MODEL_PATH)
        print("[INFO] Model training completed and saved.")
        return model

    elif mode == 'live':
        print("[INFO] Live mode selected.")
        if os.path.exists(MODEL_PATH):
            model = PPO.load(MODEL_PATH, env=env)
            print("[INFO] Loaded existing model for live fine-tuning.")
        else:
            raise ValueError("[ERROR] Live mode requires an existing model. Train a model first.")

        return model
# Define background training function
def train_model():
    """
    Function to train the model asynchronously.
    This runs in a separate thread and continuously checks if enough experiences are available.
    """
    global model  # Ensure the model is accessible within the thread
    while True:
        if len(experience_buffer) >= BATCH_SIZE:
            batch = random.sample(experience_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Prepare data for training
            states = np.vstack(states)
            actions = np.vstack(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            next_states = np.vstack(next_states)
            dones = np.array(dones).reshape(-1, 1)

            print("[INFO] Background Training Started")
            model.learn(total_timesteps=BATCH_SIZE, reset_num_timesteps=False)
            print("[INFO] Background Training Completed")

        time.sleep(1800)  # Prevent excessive CPU usage


# Argumentos de línea de comandos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesar archivo de estadísticas y usar DRL para DVFS.')
    parser.add_argument('-file_path', type=str, required=True, help='Ruta al archivo stats.txt')
    parser.add_argument('-big_cores', type=int, required=True, help='Cantidad de big cores')
    parser.add_argument('-little_cores', type=int, required=True, help='Cantidad de little cores')
    parser.add_argument('-threads', type=int, required=True, help='Cantidad de threads del benchmark')

    # VF curve arguments
    parser.add_argument('--big-cpu-voltage', type=str, nargs=8, required=True,
                        help='Voltage levels for the big cores (with units, e.g., "0.981V")')
    parser.add_argument('--little-cpu-voltage', type=str, nargs=8, required=True,
                        help='Voltage levels for the little cores (with units, e.g., "0.981V")')
    parser.add_argument('--big-cpu-clock', type=str, nargs=8, required=True,
                        help='Clock frequencies for the big cores (with units, e.g., "1800MHz")')
    parser.add_argument('--little-cpu-clock', type=str, nargs=8, required=True,
                        help='Clock frequencies for the little cores (with units, e.g., "1900MHz")')
    parser.add_argument('--host', type=str, required=True, help='Host for the Telnet connection.')
    parser.add_argument('--port', type=int, required=True, help='Port for the Telnet connection.')
    parser.add_argument('--process_chunks_num', type=int, required=True,
                        help='Number of chunks to process before sending a decision.')
    parser.add_argument('--chunk_timeout', type=int, default=300,  # 5 minutes default
                        help='Timeout in seconds for waiting for new chunks before exiting.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'live'],
                        help="Select the mode: 'train' for processing an old simulation file, 'live' for real-time data processing.")
    parser.add_argument('--freqwait',type=int, default=300,# 5 minutes default
                        help="Timeout in seconds for waiting for frequency change in gem5")
    parser.add_argument('--dur_at_freq', type=int, default=16,  # 5 minutes default
                        help="High Load duration at freq control")
    parser.add_argument('--user_chunks_var', type=int, default=16,  # 16ms default
                        help="Chunks to wait to let the frequency change to take effect default sweetspot is 16, which equals to 16ms wait")
    parser.add_argument('--changes_2_check', type=int, default=2,  # 5 minutes default
                        help="Changes to check before granting a frequency change")

    args = parser.parse_args()
    chunk_accumulator = []

    # Verify the file path
    if not os.path.isfile(args.file_path):
        raise FileNotFoundError(f"[ERROR] The specified file path does not exist: {args.file_path}")

    # Replace 'stats.txt' in file_path with 'system.terminal'
    simulation_path = args.file_path.replace("stats.txt", "system.terminal")
    print(f"This is the simulation path:{simulation_path}")


    # Ensure `system.terminal` exists
    if not os.path.isfile(simulation_path):
        raise FileNotFoundError(f"[ERROR] system.terminal file not found at {simulation_path}")

    # Parse VF curves
    big_vf_curve = parse_vf_curve(args.big_cpu_voltage, args.big_cpu_clock)
    little_vf_curve = parse_vf_curve(args.little_cpu_voltage, args.little_cpu_clock)

    # Initialize Telnet Governor
    total_cpus = args.big_cores + args.little_cores
    #set_userspace_governor(args.host, args.port, total_cpus)
    if args.mode == "live":
        set_userspace_governor(args.host,args.port, simulation_path, args.big_cores, args.little_cores)
        ##Sending CMD to Run Script
        print(f"[INFO] Sending the telnet commands to run the benchmark script")
        run_script(args.host, args.port, simulation_path) #hjsolise comment
        print(f"[INFO] Done Sending Script")



    # Process initial chunks from stats.txt
    initial_stats_chunks = process_stats_file(args.file_path)
    print(f"[INFO] Using a chunk timeout of {args.chunk_timeout} seconds.")

    # Ensure there is at least one chunk in stats_chunks
    if not initial_stats_chunks:
        raise ValueError("[ERROR] stats.txt does not contain any valid chunks. Ensure it has data.")

    if args.process_chunks_num <= 0:
        raise ValueError("[ERROR] process_chunks_num must be greater than 0.")

    # Initialize stats generator for real-time chunk processing
    #stats_generator = watch_stats_file(args.file_path, 0.5, args.chunk_timeout)

    # After: Initialize a shared deque and generator thread
    stats_generator = deque(maxlen=1)  # Shared deque to hold the latest chunk

    # Start the generator in a separate thread
    generator_thread = threading.Thread(
        target=watch_stats_file,
        args=(args.file_path, 0.5, args.chunk_timeout, stats_generator),  # Pass the deque
        daemon=True  # Allows the thread to exit when the main program exits
    )
    generator_thread.start()

    # Create environment
    #env = create_env([], args.big_cores, args.little_cores, args.threads, big_vf_curve, little_vf_curve)
    env = create_env(initial_stats_chunks, stats_generator,args.big_cores, args.little_cores, args.threads, big_vf_curve, little_vf_curve, args.host, args.port)
    env.run_mode = args.mode
    env.dur_at_freq = args.dur_at_freq
    env.chunk_user_var_to_wait = args.user_chunks_var
    env.user_changes_to_check = args.changes_2_check
    print(f"[DEBUG] Env Gem5 Class created, Duration_At_Freq:{env.dur_at_freq}")
    try:
        model = train_or_load_model(initial_stats_chunks, stats_generator,args.big_cores, args.little_cores, args.threads,
                                    big_vf_curve, little_vf_curve, args.host, args.port, args.mode)

        # Start the training thread in the background
        training_thread = threading.Thread(target=train_model, daemon=True)
        training_thread.start()

        # Simulate and get recommendations
        state, _ = env.reset()
        done = False
        current_chunk_count = 0  # Initialize outside the loop
        max_wait_time = 60  # Maximum wait time in seconds
        total_wait_time = 0
        if not hasattr(env, 'ema_values'):
            env.ema_values = None

        # Simulation loop
        while not done:
            try:
                start_time = time.time()  # Start timeout timer
                # Periodic Progress Monitoring
                print(f"[INFO] Progress Update: Processed {current_chunk_count} chunks so far.")
                print(f"[INFO] Total Accumulated Experiences: {len(experience_buffer)}")
                print(f"[INFO] Time Elapsed: {time.time() - start_time:.1f}s")
                while True:
                    try:
                        # Fetch the next chunk dynamically
                        #new_chunk = next(stats_generator)
                        new_chunk = stats_generator[-1]
                        chunk_accumulator.append(new_chunk)
                        current_chunk_count += 1

                        print(f"[INFO] Processing Chunk #{current_chunk_count}")

                        # Reset the timeout timer
                        start_time = time.time()

                        if len(chunk_accumulator) >= args.process_chunks_num:

                            # Process accumulated chunks
                            if not chunk_accumulator:
                                print("[ERROR] No chunks available for aggregation. Skipping.")
                                continue

                            # Update EMA and state
                            if not hasattr(env, 'ema_values') or env.ema_values is None:
                                env.ema_values = {key: 0.0 for key in chunk_accumulator[0]}

                            for key in chunk_accumulator[0]:
                                for chunk in chunk_accumulator:
                                    env.ema_values[key] = update_ema(env.ema_values[key], chunk.get(key, 0.0),
                                                                     alpha=0.2)

                            state = np.array(list(env.ema_values.values()), dtype=np.float32).reshape(1, -1)

                            # Sanitize the state
                            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                                print("[ERROR] State contains NaN or infinite values. Sanitizing...")
                                print("[ERROR] Sanitizing Current State")
                                state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

                            print(f"[DEBUG] Sanitized State with EMA: {state}")

                            # Get the model's decision
                            action, _ = model.predict(state)
                            # Debugging to check action immediately after prediction
                            print(f"[DEBUG] Action from model.predict: {action}, Length: {len(action)}")
                            if isinstance(action, np.ndarray) and action.shape[0] > 2:
                                print(
                                    f"[WARNING] Model returned a batch of {action.shape[0]}. Selecting only the first action.")
                                action = action[0]  # Take only the first action from the batch
                            ##Exit placed here before
                            #print("After MOdel Predict")
                            # Flatten the action if it is nested
                            if isinstance(action, np.ndarray):
                                action = action.flatten().tolist()

                            # Validate the flattened action
                            if len(action) != 2:
                                raise ValueError(f"Invalid action length: {len(action)}. Expected length: 2.")

                            big_action, little_action = action
                            print(f"[DEBUG] Action successfully processed: Big={big_action}, Little={little_action}")

                            # Decompose the action into big and little parts
                            big_action, little_action = action
                            low_load_state = env.low_load_state
                            low_load_applied = env.low_load_applied
                            print(f'This is the current low_load_state Big:{low_load_state["big"]} Little:{low_load_state["little"]}')

                            # Override actions based on low-load state
                            if low_load_state["big"] and not low_load_applied.get("big", False):
                                #big_action = 7  # Set big core to the lowest frequency
                                print("[INFO] *****Big cluster in LOW LOAD. Overriding to lowest frequency.*****")
                                low_load_applied["big"] = True  # Mark as applied
                                print(f"[INFO] Big Low Load Applied Flag = {low_load_applied['big']}")


                            if low_load_state["little"] and not low_load_applied.get("little", False):
                                #little_action = 7  # Set little core to the lowest frequency
                                print("[INFO] *****Little cluster in LOW LOAD. Overriding to lowest frequency.*****")
                                low_load_applied["little"] = True  # Mark as applied
                                print(f"[INFO] Little Low Load Applied Flag = {low_load_applied['little']}")
                            if low_load_state["big"] or low_load_state["little"]:
                                print(f"[DEBUG] Low-load state detected. Original action: {action}")
                                print(
                                    f"[DEBUG] Overriding actions -> Big: {'Low Load' if low_load_state['big'] else 'Unchanged'}, "
                                    f"Little: {'Low Load' if low_load_state['little'] else 'Unchanged'}")
                            # Recombine the actions if needed later
                            action = [big_action, little_action]
                            if low_load_state["big"] or low_load_state["little"]:
                                print("[DEBUG] Found a low load state")
                                print(f'[DEBUG] This is the current low_load_state Big:{low_load_state["big"]} Little:{low_load_state["little"]}')
                                print(f"[DEBUG] This is the current Action:{action}")



                            print(f"[DEBUG] Action predicted After Low Load Check: {action}")  # Added
                            # Apply the decision and get the next state
                            #next_state, reward, terminated, truncated, action_taken, _ = env.step(action)
                            next_state, reward, terminated, truncated, info = env.step(action)
                            if "action_taken" not in info:
                                print("[ERROR] 'action_taken' missing from step() return. Exiting...")
                                exit()  # Immediately exit if action_taken is missing
                            action_taken = info["action_taken"]

                            print(
                                f"[DEBUG] MainLoop step variables final_state:{next_state} reward:{reward} terminated:{terminated} truncated:{truncated} action_taken:{action_taken}")
                            print(f"[DEBUG] Step -> Next State Shape: {next_state.shape}, Reward: {reward}, Terminated: {terminated}")  # Added

                            # Sanitize the next state
                            if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                                print("[ERROR] Next state contains NaN or infinite values. Sanitizing...")
                                print("[ERROR] Sanitizing Next State")
                                next_state = np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)
                            print(
                                f"[DEBUG] Before appending to experience buffer: Action Taken: {action_taken}, Length: {len(action_taken)}")

                            if isinstance(action_taken, list) and len(action_taken) != 2:
                                print(
                                    f"[ERROR] Action size is incorrect before appending to buffer: {len(action_taken)}")
                                raise ValueError(
                                    f"Invalid action length before appending to buffer: {len(action_taken)}. Expected: 2.")

                            # Store the experience in the buffer
                            experience_buffer.append((state, action_taken, reward, next_state, terminated or truncated))

                            # Periodically save the model
                            if current_chunk_count % 100 == 0:
                                model.save(MODEL_PATH)
                                print(f"Modelo guardado tras {current_chunk_count} chunks.")

                            # Update simulation state
                            state = next_state
                            done = terminated or truncated

                            print(f"Ajuste de DVFS: {action_taken}, Action Modelo: {action} Recompensa: {reward}")

                            # Reset chunk accumulator
                            chunk_accumulator.clear()
                            current_chunk_count = 0
                            break  # Exit inner loop for next chunk

                    except StopIteration:
                        print("[INFO] Stats generator completed. Ending simulation loop.")
                        done = True
                        break

                    # Check for timeout
                    if time.time() - start_time >= args.chunk_timeout:
                        print(f"[INFO] Timeout reached: {args.chunk_timeout} seconds without new chunks.")
                        print(f"[INFO] Terminating simulation. Processed {current_chunk_count} chunks in total.")
                        done = True
                        break

            except Exception as e:
                print(f"[ERROR] Unexpected error during simulation: {e}")

                # Capture and print the full traceback
                traceback_str = traceback.format_exc()
                print(f"[ERROR] Full Traceback:\n{traceback_str}")

                # Optionally, print the last known action and state
                if 'action' in locals():
                    print(f"[DEBUG] Last attempted action: {action}")
                if 'state' in locals():
                    print(
                        f"[DEBUG] Last known state shape: {state.shape if isinstance(state, np.ndarray) else type(state)}")

                done = True

        # Save model before exitingfcalculate_reward
        model.save(MODEL_PATH)
        print("Modelo guardado tras timeout o finalización.")




    finally:
        env.close()  # Ensure the environment is closed properly
        print("[DEBUG] Environment closed.")
