"""
MARL PPO Implementation

This module provides a streamlined version of MARL PPO that works directly with
the existing MARLMilitaryEnvironment.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import csv


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For more detailed GPU info (if available)
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU available, using CPU instead")


# Actor Network
class ActorNetwork(nn.Module):
    """Actor network that processes the revised observation structure"""

    def __init__(self, action_dim):
        super(ActorNetwork, self).__init__()

        # Process agent state
        self.agent_state_encoder = nn.Sequential(
            nn.Linear(5, 32),  # position(2) + health(1) + ammo(1) + suppressed(1)
            nn.ReLU()
        )

        # Process tactical info
        self.tactical_encoder = nn.Sequential(
            nn.Linear(3, 16),  # formation(1) + orientation(1) + unit_type(1)
            nn.ReLU()
        )

        # Process friendly units
        self.friendly_encoder = nn.Sequential(
            nn.Linear(20, 32),  # 10 units x 2 coords
            nn.ReLU()
        )

        # Process enemy units
        self.enemy_encoder = nn.Sequential(
            nn.Linear(20, 32),  # 10 units x 2 coords
            nn.ReLU()
        )

        # Process objective
        self.objective_encoder = nn.Sequential(
            nn.Linear(2, 16),  # x, y coordinates
            nn.ReLU()
        )

        # Process objective info
        self.objective_info_encoder = nn.Sequential(
            nn.Linear(3, 16),  # direction (2) + distance (1)
            nn.ReLU()
        )

        # Combined features layer - adjusted for the revised structure
        combined_input_size = 32 + 16 + 32 + 32 + 16 + 16  # 144

        self.combined_layer = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Action type head (discrete)
        self.action_type_head = nn.Linear(128, action_dim)

        # Movement parameters head
        self.movement_dir_head = nn.Linear(128, 2)  # 2D direction vector
        self.movement_dist_head = nn.Linear(128, 3)  # 3 options: 1, 5, 10

        # Engagement parameters head
        self.target_pos_head = nn.Linear(128, 2)  # 2D target position
        self.max_rounds_head = nn.Linear(128, 3)  # 3 options: 1, 6, 12

        # Binary parameters (suppress_only, adjust_fire_rate)
        self.suppress_only_head = nn.Linear(128, 2)  # Binary choice
        self.adjust_fire_rate_head = nn.Linear(128, 2)  # Binary choice

        # Formation head (discrete)
        self.formation_head = nn.Linear(128, 8)  # 8 formation options

    def forward(self, observation):
        """Process observation with the revised structure for discrete action spaces"""
        # Extract and process agent state
        agent_pos = observation['agent_state']['position']
        agent_health = observation['agent_state']['health']
        agent_ammo = observation['agent_state']['ammo']
        agent_suppressed = observation['agent_state']['suppressed']

        # Combine agent state features
        agent_state = torch.cat([
            agent_pos,
            agent_health,
            agent_ammo,
            agent_suppressed
        ], dim=1)

        agent_features = self.agent_state_encoder(agent_state)

        # Process tactical info
        tactical_info = torch.cat([
            observation['tactical_info']['formation'],
            observation['tactical_info']['orientation'],
            observation['tactical_info']['unit_type']
        ], dim=1)

        tactical_features = self.tactical_encoder(tactical_info)

        # Process friendly units (flatten spatial data)
        friendly_units = observation['friendly_units'].view(observation['friendly_units'].size(0), -1)
        friendly_features = self.friendly_encoder(friendly_units)

        # Process enemy units (flatten spatial data)
        known_enemies = observation['known_enemies'].view(observation['known_enemies'].size(0), -1)
        enemy_features = self.enemy_encoder(known_enemies)

        # Process objective
        objective_features = self.objective_encoder(observation['objective'])

        # Process objective info
        objective_info = torch.cat([
            observation['objective_info']['direction'],
            observation['objective_info']['distance']
        ], dim=1)
        objective_info_features = self.objective_info_encoder(objective_info)

        # Combined features
        combined = torch.cat([
            agent_features,
            tactical_features,
            friendly_features,
            enemy_features,
            objective_features,
            objective_info_features
        ], dim=1)

        combined_features = self.combined_layer(combined)

        # Action type logits (already discrete)
        action_type_logits = self.action_type_head(combined_features)

        # Movement parameters
        # Direction remains continuous
        movement_direction = torch.tanh(self.movement_dir_head(combined_features))  # -1 to 1

        # Distance is now discrete with 3 options [1, 5, 10]
        movement_distance_logits = self.movement_dist_head(combined_features)

        # Engagement parameters
        # Target position remains continuous
        target_pos = torch.sigmoid(self.target_pos_head(combined_features)) * 100  # Scale to map size

        # Max rounds is now discrete with 3 options [1, 6, 12]
        max_rounds_logits = self.max_rounds_head(combined_features)

        # Binary choices (already discrete)
        suppress_only_logits = self.suppress_only_head(combined_features)
        adjust_fire_rate_logits = self.adjust_fire_rate_head(combined_features)

        # Formation parameters (already discrete)
        formation_logits = self.formation_head(combined_features)

        return {
            'action_type': action_type_logits,
            'movement_params': {
                'direction': movement_direction,
                'distance_options': movement_distance_logits  # Renamed to make it clear these are options
            },
            'engagement_params': {
                'target_pos': target_pos,
                'max_rounds_options': max_rounds_logits,  # Renamed to make it clear these are options
                'suppress_only': suppress_only_logits,
                'adjust_fire_rate': adjust_fire_rate_logits
            },
            'formation': formation_logits
        }


# Critic Network
class CriticNetwork(nn.Module):
    """Critic network that processes the revised observation structure"""

    def __init__(self):
        super(CriticNetwork, self).__init__()

        # Process agent state
        self.agent_state_encoder = nn.Sequential(
            nn.Linear(5, 32),  # position(2) + health(1) + ammo(1) + suppressed(1)
            nn.ReLU()
        )

        # Process tactical info
        self.tactical_encoder = nn.Sequential(
            nn.Linear(3, 16),  # formation(1) + orientation(1) + unit_type(1)
            nn.ReLU()
        )

        # Process friendly units
        self.friendly_encoder = nn.Sequential(
            nn.Linear(20, 32),  # 10 units x 2 coords
            nn.ReLU()
        )

        # Process enemy units
        self.enemy_encoder = nn.Sequential(
            nn.Linear(20, 32),  # 10 units x 2 coords
            nn.ReLU()
        )

        # Process objective
        self.objective_encoder = nn.Sequential(
            nn.Linear(2, 16),  # x, y coordinates
            nn.ReLU()
        )

        # Process objective info
        self.objective_info_encoder = nn.Sequential(
            nn.Linear(3, 16),  # direction (2) + distance (1)
            nn.ReLU()
        )

        # Combined features layer - adjusted for the revised structure
        combined_input_size = 32 + 16 + 32 + 32 + 16 + 16  # 144

        self.combined_layer = nn.Sequential(
            nn.Linear(combined_input_size, 256),  # Increase from 128 to 256
            nn.ReLU(),
            nn.Linear(256, 256),  # Increase from 64 to 256
            nn.ReLU(),
            nn.Linear(256, 128),  # Add an extra layer
            nn.ReLU()
        )

        # Value output
        self.value_head = nn.Linear(128, 1)

    def forward(self, observation):
        """Process observation with the revised structure"""
        # Extract and process agent state
        agent_pos = observation['agent_state']['position']
        agent_health = observation['agent_state']['health']
        agent_ammo = observation['agent_state']['ammo']
        agent_suppressed = observation['agent_state']['suppressed']

        # Combine agent state features
        agent_state = torch.cat([
            agent_pos,
            agent_health,
            agent_ammo,
            agent_suppressed
        ], dim=1)

        agent_features = self.agent_state_encoder(agent_state)

        # Process tactical info
        tactical_info = torch.cat([
            observation['tactical_info']['formation'],
            observation['tactical_info']['orientation'],
            observation['tactical_info']['unit_type']
        ], dim=1)

        tactical_features = self.tactical_encoder(tactical_info)

        # Process friendly units (flatten spatial data)
        friendly_units = observation['friendly_units'].view(observation['friendly_units'].size(0), -1)
        friendly_features = self.friendly_encoder(friendly_units)

        # Process enemy units (flatten spatial data)
        known_enemies = observation['known_enemies'].view(observation['known_enemies'].size(0), -1)
        enemy_features = self.enemy_encoder(known_enemies)

        # Process objective
        objective_features = self.objective_encoder(observation['objective'])

        # Process objective info
        objective_info = torch.cat([
            observation['objective_info']['direction'],
            observation['objective_info']['distance']
        ], dim=1)
        objective_info_features = self.objective_info_encoder(objective_info)

        # Combined features
        combined = torch.cat([
            agent_features,
            tactical_features,
            friendly_features,
            enemy_features,
            objective_features,
            objective_info_features
        ], dim=1)

        combined_features = self.combined_layer(combined)

        # Value prediction
        value = self.value_head(combined_features)

        return value


# Memory Buffer for storing experiences
class Memory:
    """Enhanced memory buffer for storing agent experiences with improved tracking."""

    def __init__(self):
        # Core memory containers
        self.states = {}
        self.actions = {}
        self.logprobs = {}
        self.rewards = {}
        self.is_terminals = {}
        self.values = {}

        # Tracking attributes
        self.initialized_agents = set()  # Agents that have been initialized
        self.current_episode_agents = set()  # Agents in current episode
        self.last_updated = {}  # Track last timestep each agent's memory was updated

        # Debugging attributes
        self.debug_enabled = True  # Set to false in production
        self.memory_stats = {}  # Track length of each memory component for debugging

    def clear_memory(self):
        """Clear all memory contents but maintain initialization tracking."""
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        self.last_updated.clear()

        # Keep track of which agents were initialized
        self.current_episode_agents.clear()

        # Reset debugging stats
        self.memory_stats = {}

        # if self.debug_enabled:
        #     print("DEBUG: Memory cleared")

    def initialize_agent(self, agent_id):
        """
        Initialize memory for a specific agent.
        This ensures all necessary lists exist before trying to append to them.
        FIXED: Now initializes rewards and terminals with empty arrays.
        """
        if agent_id not in self.initialized_agents:
            # Create empty lists for this agent in all memory components
            self.states[agent_id] = []
            self.actions[agent_id] = []
            self.logprobs[agent_id] = []
            self.values[agent_id] = []
            self.rewards[agent_id] = []  # Initialize with empty list
            self.is_terminals[agent_id] = []  # Initialize with empty list

            # Mark as initialized and track in current episode
            self.initialized_agents.add(agent_id)
            self.current_episode_agents.add(agent_id)

            # print(f"DEBUG: Initialized memory for agent {agent_id}")
        elif agent_id not in self.current_episode_agents:
            # Agent was initialized in a previous episode but not current one
            self.current_episode_agents.add(agent_id)

            # Re-initialize with empty lists for this episode
            self.states[agent_id] = []
            self.actions[agent_id] = []
            self.logprobs[agent_id] = []
            self.values[agent_id] = []
            self.rewards[agent_id] = []
            self.is_terminals[agent_id] = []

            # print(f"DEBUG: Re-initialized agent {agent_id} for new episode")

    def verify_memory_consistency(self, check_mode="complete"):
        """
        Verify all memory components have consistent lengths for each agent.

        Args:
            check_mode: "complete" to check all arrays or "actions_only" to only check action-related arrays

        Returns:
            dict of agents with inconsistent memory.
        """
        inconsistent_agents = {}

        for agent_id in self.current_episode_agents:
            if check_mode == "complete":
                # Check all memory components
                lengths = {
                    'states': len(self.states.get(agent_id, [])),
                    'actions': len(self.actions.get(agent_id, [])),
                    'logprobs': len(self.logprobs.get(agent_id, [])),
                    'rewards': len(self.rewards.get(agent_id, [])),
                    'is_terminals': len(self.is_terminals.get(agent_id, [])),
                    'values': len(self.values.get(agent_id, []))
                }

                # Check if all lists have the same length
                if len(set(lengths.values())) > 1:
                    inconsistent_agents[agent_id] = lengths
            else:
                # Only check action-related components (states, actions, logprobs, values)
                # This mode is useful during select_actions before rewards are stored
                lengths = {
                    'states': len(self.states.get(agent_id, [])),
                    'actions': len(self.actions.get(agent_id, [])),
                    'logprobs': len(self.logprobs.get(agent_id, [])),
                    'values': len(self.values.get(agent_id, []))
                }

                # Check if these specific lists have the same length
                if len(set(lengths.values())) > 1:
                    inconsistent_agents[agent_id] = lengths

        return inconsistent_agents

    def update_memory_stats(self):
        """Update memory statistics for debugging."""
        if not self.debug_enabled:
            return

        for agent_id in self.current_episode_agents:
            self.memory_stats[agent_id] = {
                'states': len(self.states.get(agent_id, [])),
                'actions': len(self.actions.get(agent_id, [])),
                'logprobs': len(self.logprobs.get(agent_id, [])),
                'rewards': len(self.rewards.get(agent_id, [])),
                'is_terminals': len(self.is_terminals.get(agent_id, [])),
                'values': len(self.values.get(agent_id, []))
            }

    def check_consistency(self, agent_id=None):
        """
        Check consistency of agent memory structures.

        Args:
            agent_id: Optional specific agent to check, or all agents if None

        Returns:
            Dictionary with consistency check results
        """
        results = {}

        # Determine which agents to check
        agents_to_check = [agent_id] if agent_id is not None else list(self.current_episode_agents)

        for aid in agents_to_check:
            # Skip if agent not initialized
            if aid not in self.initialized_agents:
                results[aid] = {"initialized": False, "consistent": False, "error": "Agent not initialized"}
                continue

            # Get lengths of all memory components
            state_len = len(self.states.get(aid, []))
            action_len = len(self.actions.get(aid, []))
            logprob_len = len(self.logprobs.get(aid, []))
            reward_len = len(self.rewards.get(aid, []))
            terminal_len = len(self.is_terminals.get(aid, []))
            value_len = len(self.values.get(aid, []))

            # Check if all lengths match
            lengths = [state_len, action_len, logprob_len, reward_len, terminal_len, value_len]
            base_len = lengths[0]
            consistent = all(length == base_len for length in lengths)

            # Store results
            results[aid] = {
                "initialized": True,
                "consistent": consistent,
                "lengths": {
                    "states": state_len,
                    "actions": action_len,
                    "logprobs": logprob_len,
                    "rewards": reward_len,
                    "terminals": terminal_len,
                    "values": value_len
                },
                "last_updated": self.last_updated.get(aid, -1)
            }

        return results

    def fix_inconsistencies(self):
        """
        Attempt to fix memory inconsistencies by truncating to shortest length.

        Returns:
            Dictionary with fix results
        """
        results = {}

        for agent_id in self.current_episode_agents:
            # Get current consistency status
            status = self.check_consistency(agent_id)
            if not status.get(agent_id, {}).get("consistent", True):
                # Find minimum length
                if "lengths" in status[agent_id]:
                    length_dict = status[agent_id]["lengths"]
                    # Check if lengths is a dictionary before using .values()
                    if isinstance(length_dict, dict):
                        min_length = min(length_dict.values())
                    else:
                        # If not a dictionary, convert to dictionary or use a default value
                        print(f"Warning: lengths for agent {agent_id} is not a dictionary")
                        min_length = 0
                else:
                    # If no lengths field, use 0 as minimum length
                    min_length = 0

                # Truncate all arrays to minimum length
                if min_length > 0:
                    if agent_id in self.states:
                        self.states[agent_id] = self.states[agent_id][:min_length]
                    if agent_id in self.actions:
                        self.actions[agent_id] = self.actions[agent_id][:min_length]
                    if agent_id in self.logprobs:
                        self.logprobs[agent_id] = self.logprobs[agent_id][:min_length]
                    if agent_id in self.rewards:
                        self.rewards[agent_id] = self.rewards[agent_id][:min_length]
                    if agent_id in self.is_terminals:
                        self.is_terminals[agent_id] = self.is_terminals[agent_id][:min_length]
                    if agent_id in self.values:
                        self.values[agent_id] = self.values[agent_id][:min_length]

                    results[agent_id] = {"fixed": True, "new_length": min_length}
                else:
                    # If min_length is 0, remove agent from current episode
                    self.current_episode_agents.remove(agent_id)
                    results[agent_id] = {"fixed": False, "error": "Zero length arrays"}
            else:
                results[agent_id] = {"fixed": False, "error": "No inconsistency detected"}

        return results

    def get_agent_data_for_update(self, agent_id):
        """
        Get all agent data needed for policy update, with consistency checks.

        Args:
            agent_id: Consistent ID of the agent

        Returns:
            Dictionary with all agent data, or None if data is inconsistent
        """
        # Check consistency first
        status = self.check_consistency(agent_id)
        if not status.get(agent_id, {}).get("consistent", False):
            return None

        # Return data dictionary
        return {
            "states": self.states.get(agent_id, []),
            "actions": self.actions.get(agent_id, []),
            "logprobs": self.logprobs.get(agent_id, []),
            "rewards": self.rewards.get(agent_id, []),
            "is_terminals": self.is_terminals.get(agent_id, []),
            "values": self.values.get(agent_id, [])
        }


# Agent Policy
class AgentPolicy:
    """PPO policy for a single agent with enhanced memory integration"""

    def __init__(self, action_dim, lr, agent_id):
        self.agent_id = agent_id
        self.action_dim = action_dim

        # Initialize actor network
        self.actor = ActorNetwork(action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Initialize critic network
        self.critic = CriticNetwork().to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize old actor for PPO updates
        self.old_actor = ActorNetwork(action_dim).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Add policy version tracking for debugging
        self.update_count = 0

    def process_observation(self, observation):
        """Convert numpy observation to PyTorch tensors with error handling"""
        try:
            # Copy observation structure but convert values to tensors
            processed_obs = {
                'agent_state': {
                    'position': torch.FloatTensor(observation['agent_state']['position']).unsqueeze(0).to(device),
                    'health': torch.FloatTensor(observation['agent_state']['health']).unsqueeze(0).to(device),
                    'ammo': torch.FloatTensor(observation['agent_state']['ammo']).unsqueeze(0).to(device),
                    'suppressed': torch.FloatTensor(observation['agent_state']['suppressed']).unsqueeze(0).to(device)
                },
                'tactical_info': {
                    'formation': torch.FloatTensor(observation['tactical_info']['formation']).unsqueeze(0).to(device),
                    'orientation': torch.FloatTensor(observation['tactical_info']['orientation']).unsqueeze(0).to(
                        device),
                    'unit_type': torch.FloatTensor(observation['tactical_info']['unit_type']).unsqueeze(0).to(device)
                },
                'friendly_units': torch.FloatTensor(observation['friendly_units']).unsqueeze(0).to(device),
                'known_enemies': torch.FloatTensor(observation['known_enemies']).unsqueeze(0).to(device),
                'objective': torch.FloatTensor(observation['objective']).unsqueeze(0).to(device),
                'objective_info': {
                    'direction': torch.FloatTensor(observation['objective_info']['direction']).unsqueeze(0).to(device),
                    'distance': torch.FloatTensor(observation['objective_info']['distance']).unsqueeze(0).to(device)
                }
            }
            return processed_obs
        except Exception as e:
            print(f"Error processing observation for agent {self.agent_id}: {e}")
            print(f"Observation keys: {observation.keys()}")
            # Return a default processed observation
            return self._create_default_processed_observation()

    def _create_default_processed_observation(self):
        """Create a default processed observation for error cases"""
        default_obs = {
            'agent_state': {
                'position': torch.zeros(1, 2).to(device),
                'health': torch.ones(1, 1).to(device) * 100.0,
                'ammo': torch.ones(1, 1).to(device) * 100.0,
                'suppressed': torch.zeros(1, 1).to(device)
            },
            'tactical_info': {
                'formation': torch.zeros(1, 1).to(device),
                'orientation': torch.zeros(1, 1).to(device),
                'unit_type': torch.zeros(1, 1).to(device)
            },
            'friendly_units': torch.zeros(1, 10, 2).to(device),
            'known_enemies': torch.zeros(1, 10, 2).to(device),
            'objective': torch.FloatTensor([[15, 15]]).to(device),
            'objective_info': {
                'direction': torch.zeros(1, 2).to(device),
                'distance': torch.ones(1, 1).to(device) * 50.0
            }
        }
        return default_obs

    def process_batch_observations(self, observations):
        """Process a batch of observations for policy update with improved error handling"""
        # Initialize batch_size at the start to ensure it's always defined
        batch_size = 1  # Default fallback value

        try:
            # Now set the actual batch size from observations
            batch_size = len(observations)
            if batch_size == 0:
                raise ValueError("Empty batch of observations")

            # Extract and batch components
            batch_positions = torch.FloatTensor([obs['agent_state']['position'] for obs in observations]).to(device)
            batch_health = torch.FloatTensor([obs['agent_state']['health'] for obs in observations]).to(device)
            batch_ammo = torch.FloatTensor([obs['agent_state']['ammo'] for obs in observations]).to(device)
            batch_suppressed = torch.FloatTensor([obs['agent_state']['suppressed'] for obs in observations]).to(device)

            batch_formation = torch.FloatTensor([obs['tactical_info']['formation'] for obs in observations]).to(device)
            batch_orientation = torch.FloatTensor([obs['tactical_info']['orientation'] for obs in observations]).to(
                device)
            batch_unit_type = torch.FloatTensor([obs['tactical_info']['unit_type'] for obs in observations]).to(device)

            batch_friendly_units = torch.FloatTensor([obs['friendly_units'] for obs in observations]).to(device)
            batch_known_enemies = torch.FloatTensor([obs['known_enemies'] for obs in observations]).to(device)

            batch_objective = torch.FloatTensor([obs['objective'] for obs in observations]).to(device)

            # Process objective info
            batch_objective_direction = torch.stack(
                [torch.FloatTensor(obs['objective_info']['direction']) for obs in observations]).to(device)
            batch_objective_distance = torch.stack(
                [torch.FloatTensor(obs['objective_info']['distance']) for obs in observations]).to(device)

            # Combine into observation structure
            processed_batch = {
                'agent_state': {
                    'position': batch_positions,
                    'health': batch_health,
                    'ammo': batch_ammo,
                    'suppressed': batch_suppressed
                },
                'tactical_info': {
                    'formation': batch_formation,
                    'orientation': batch_orientation,
                    'unit_type': batch_unit_type
                },
                'friendly_units': batch_friendly_units,
                'known_enemies': batch_known_enemies,
                'objective': batch_objective,
                'objective_info': {
                    'direction': batch_objective_direction,
                    'distance': batch_objective_distance
                }
            }

            return processed_batch
        except Exception as e:
            print(f"Error processing batch observations for agent {self.agent_id}: {e}")
            import traceback
            traceback.print_exc()

            # batch_size is already initialized, so it's safe to use
            return self._create_default_batch_observation(max(1, batch_size))

    def _create_default_batch_observation(self, batch_size):
        """Create a default batch observation for error cases"""
        default_obs = {
            'agent_state': {
                'position': torch.zeros(batch_size, 2).to(device),
                'health': torch.ones(batch_size, 1).to(device) * 100.0,
                'ammo': torch.ones(batch_size, 1).to(device) * 100.0,
                'suppressed': torch.zeros(batch_size, 1).to(device)
            },
            'tactical_info': {
                'formation': torch.zeros(batch_size, 1).to(device),
                'orientation': torch.zeros(batch_size, 1).to(device),
                'unit_type': torch.zeros(batch_size, 1).to(device)
            },
            'friendly_units': torch.zeros(batch_size, 10, 2).to(device),
            'known_enemies': torch.zeros(batch_size, 10, 2).to(device),
            'objective': torch.ones(batch_size, 2).to(device) * 15.0,
            'objective_info': {
                'direction': torch.zeros(batch_size, 2).to(device),
                'distance': torch.ones(batch_size, 1).to(device) * 50.0
            }
        }
        return default_obs

    def select_action_with_logprob(self, observation):
        with torch.no_grad():
            # Process observation
            processed_obs = self.process_observation(observation)

            # Forward pass through old_actor
            action_outputs = self.old_actor(processed_obs)

            # Get state value
            state_value = self.critic(processed_obs).item()

            # Sample action type
            action_type_logits = action_outputs['action_type']
            action_type_probs = F.softmax(action_type_logits, dim=1)
            action_type_dist = Categorical(action_type_probs)
            action_type = action_type_dist.sample()
            action_type_logprob = action_type_dist.log_prob(action_type)

            # Sample movement distance (discrete)
            movement_distance_logits = action_outputs['movement_params']['distance_options']
            movement_distance_probs = F.softmax(movement_distance_logits, dim=1)
            movement_distance_dist = Categorical(movement_distance_probs)
            movement_distance_idx = movement_distance_dist.sample()
            movement_distance_logprob = movement_distance_dist.log_prob(movement_distance_idx)

            # Convert index to actual distance value (1, 5, or 10)
            distance_values = [1, 5, 10]
            movement_distance = distance_values[movement_distance_idx.item()]

            # Sample max rounds (discrete)
            max_rounds_logits = action_outputs['engagement_params']['max_rounds_options']
            max_rounds_probs = F.softmax(max_rounds_logits, dim=1)
            max_rounds_dist = Categorical(max_rounds_probs)
            max_rounds_idx = max_rounds_dist.sample()
            max_rounds_logprob = max_rounds_dist.log_prob(max_rounds_idx)

            # Convert index to actual rounds value (1, 6, or 12)
            rounds_values = [1, 6, 12]
            max_rounds = rounds_values[max_rounds_idx.item()]

            # Sample suppress only
            suppress_only_logits = action_outputs['engagement_params']['suppress_only']
            suppress_only_probs = F.softmax(suppress_only_logits, dim=1)
            suppress_only_dist = Categorical(suppress_only_probs)
            suppress_only = suppress_only_dist.sample()
            suppress_only_logprob = suppress_only_dist.log_prob(suppress_only)

            # Sample adjust fire rate
            adjust_fire_rate_logits = action_outputs['engagement_params']['adjust_fire_rate']
            adjust_fire_rate_probs = F.softmax(adjust_fire_rate_logits, dim=1)
            adjust_fire_rate_dist = Categorical(adjust_fire_rate_probs)
            adjust_fire_rate = adjust_fire_rate_dist.sample()
            adjust_fire_rate_logprob = adjust_fire_rate_dist.log_prob(adjust_fire_rate)

            # Sample formation
            formation_logits = action_outputs['formation']
            formation_probs = F.softmax(formation_logits, dim=1)
            formation_dist = Categorical(formation_probs)
            formation = formation_dist.sample()
            formation_logprob = formation_dist.log_prob(formation)

            # Get movement direction (continuous)
            movement_direction = action_outputs['movement_params']['direction'].squeeze(0).cpu().numpy()

            # Get target position (continuous)
            target_pos = action_outputs['engagement_params']['target_pos'].squeeze(0).cpu().numpy()

            # Combine logprobs
            combined_logprob = action_type_logprob + movement_distance_logprob + max_rounds_logprob + \
                               suppress_only_logprob + adjust_fire_rate_logprob + formation_logprob

            # Create action dictionary matching environment expectations
            action = {
                'action_type': action_type.item(),
                'movement_params': {
                    'direction': movement_direction,
                    'distance': [movement_distance]  # Keep as list with single integer value
                },
                'engagement_params': {
                    'target_pos': target_pos,
                    'max_rounds': [max_rounds],  # Keep as list with single integer value
                    'suppress_only': suppress_only.item(),
                    'adjust_for_fire_rate': adjust_fire_rate.item()
                },
                'formation': formation.item()
            }

            return action, combined_logprob.item(), state_value

    def update(self, agent_data, clip_param=0.2, value_coef=0.5, entropy_coef=0.01, ppo_epochs=4, batch_size=64,
               gamma=0.99, gae_lambda=0.95):
        """
        Update policy using PPO algorithm with support for discrete action components.
        Modified to handle both discrete and continuous action components.

        Args:
            agent_data: Dictionary with agent data from memory
            clip_param: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy coefficient for exploration
            ppo_epochs: Number of epochs for PPO update
            batch_size: Batch size
            gamma: Discount factor
            gae_lambda: GAE parameter for advantage estimation

        Returns:
            Tuple of (actor_loss, critic_loss, entropy_loss)
        """
        # Extract data
        old_states = agent_data["states"]
        old_actions = agent_data["actions"]
        old_logprobs = torch.FloatTensor(agent_data["logprobs"]).to(device)
        rewards = torch.FloatTensor(agent_data["rewards"]).to(device)
        old_values = torch.FloatTensor(agent_data["values"]).to(device)
        is_terminals = torch.FloatTensor(agent_data["is_terminals"]).to(device)

        # Verify data consistency
        data_length = min(len(old_states), len(old_actions), len(old_logprobs),
                          len(rewards), len(old_values), len(is_terminals))

        if data_length < 2:  # Need at least 2 data points for meaningful update
            print(f"ERROR: Agent {self.agent_id} has insufficient data ({data_length} points)")
            return 0, 0, 0

        # Ensure all data arrays have the same length
        old_states = old_states[:data_length]
        old_actions = old_actions[:data_length]
        old_logprobs = old_logprobs[:data_length]
        rewards = rewards[:data_length]
        old_values = old_values[:data_length]
        is_terminals = is_terminals[:data_length]

        # Calculate returns and advantages using GAE
        advantages = torch.zeros_like(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)

        # GAE calculation
        gae = 0
        for i in reversed(range(data_length)):
            if i == data_length - 1:
                next_value = 0  # Terminal state has 0 value
            else:
                next_value = old_values[i + 1]

            # If terminal state, reset advantage
            if is_terminals[i]:
                gae = 0
                next_value = 0

            # Calculate TD error and GAE
            delta = rewards[i] + gamma * next_value * (1 - is_terminals[i]) - old_values[i]
            gae = delta + gamma * gae_lambda * (1 - is_terminals[i]) * gae

            # Store returns and advantages
            returns[i] = gae + old_values[i]
            advantages[i] = gae

        # Normalize advantages - critical for stable learning
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize returns for critic training
        normalized_returns = returns.clone()
        if len(returns) > 1:
            normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO update for multiple epochs
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        # Define mappings for discrete actions
        # Map the actual action values to indices for categorical distributions
        distance_map = {1: 0, 5: 1, 10: 2}  # Maps distance value to index
        rounds_map = {1: 0, 6: 1, 12: 2}  # Maps max_rounds value to index

        for _ in range(ppo_epochs):
            # Create random permutation of indices
            indices = np.random.permutation(data_length)

            # Adjust batch size if needed
            actual_batch_size = min(batch_size, data_length)

            # Process batches
            for start_idx in range(0, data_length, actual_batch_size):
                # Get batch indices
                end_idx = min(start_idx + actual_batch_size, data_length)
                batch_idx = indices[start_idx:end_idx]

                if len(batch_idx) == 0:
                    continue

                try:
                    # Get batch data
                    batch_states = [old_states[i] for i in batch_idx]
                    batch_actions = [old_actions[i] for i in batch_idx]
                    batch_logprobs = old_logprobs[batch_idx]
                    batch_returns = normalized_returns[batch_idx]
                    batch_advantages = advantages[batch_idx]

                    # Process batch observations
                    processed_batch = self.process_batch_observations(batch_states)

                    # Forward pass through actor
                    actor_outputs = self.actor(processed_batch)

                    # Forward pass through critic
                    state_values = self.critic(processed_batch).squeeze()

                    # Ensure state values has the right shape
                    if len(batch_returns) == 1 and not isinstance(state_values, torch.Tensor):
                        state_values = torch.tensor([state_values], device=device)
                    elif state_values.shape != batch_returns.shape:
                        if state_values.dim() == 0:  # Scalar tensor
                            state_values = state_values.unsqueeze(0)

                        # Ensure lengths match
                        min_len = min(len(state_values), len(batch_returns))
                        state_values = state_values[:min_len]
                        batch_returns = batch_returns[:min_len]

                    # Extract discrete action components from batch_actions
                    # Convert action values to indices for categorical distributions
                    batch_action_types = torch.LongTensor([a['action_type'] for a in batch_actions]).to(device)

                    # Convert distance values to indices using the mapping
                    batch_distance_indices = torch.LongTensor([
                        distance_map.get(a['movement_params']['distance'][0], 0)
                        for a in batch_actions
                    ]).to(device)

                    # Convert max_rounds values to indices using the mapping
                    batch_max_rounds_indices = torch.LongTensor([
                        rounds_map.get(a['engagement_params']['max_rounds'][0], 0)
                        for a in batch_actions
                    ]).to(device)

                    # Extract other discrete components
                    batch_suppress_only = torch.LongTensor([
                        a['engagement_params']['suppress_only']
                        for a in batch_actions
                    ]).to(device)

                    batch_adjust_fire_rate = torch.LongTensor([
                        a['engagement_params']['adjust_for_fire_rate']
                        for a in batch_actions
                    ]).to(device)

                    batch_formations = torch.LongTensor([
                        a['formation'] for a in batch_actions
                    ]).to(device)

                    # Get distributions for all action components
                    # Action type distribution (no change, already discrete)
                    action_type_probs = F.softmax(actor_outputs['action_type'], dim=1)
                    action_type_dist = Categorical(action_type_probs)

                    # Movement distance distribution (discrete)
                    distance_probs = F.softmax(actor_outputs['movement_params']['distance_options'], dim=1)
                    distance_dist = Categorical(distance_probs)

                    # Max rounds distribution (discrete)
                    max_rounds_probs = F.softmax(actor_outputs['engagement_params']['max_rounds_options'], dim=1)
                    max_rounds_dist = Categorical(max_rounds_probs)

                    # Other discrete distributions
                    suppress_only_probs = F.softmax(actor_outputs['engagement_params']['suppress_only'], dim=1)
                    suppress_only_dist = Categorical(suppress_only_probs)

                    adjust_fire_rate_probs = F.softmax(actor_outputs['engagement_params']['adjust_fire_rate'], dim=1)
                    adjust_fire_rate_dist = Categorical(adjust_fire_rate_probs)

                    formation_probs = F.softmax(actor_outputs['formation'], dim=1)
                    formation_dist = Categorical(formation_probs)

                    # Calculate log probabilities for all action components
                    action_type_logprobs = action_type_dist.log_prob(batch_action_types)
                    distance_logprobs = distance_dist.log_prob(batch_distance_indices)
                    max_rounds_logprobs = max_rounds_dist.log_prob(batch_max_rounds_indices)
                    suppress_only_logprobs = suppress_only_dist.log_prob(batch_suppress_only)
                    adjust_fire_rate_logprobs = adjust_fire_rate_dist.log_prob(batch_adjust_fire_rate)
                    formation_logprobs = formation_dist.log_prob(batch_formations)

                    # Calculate entropy (encourage exploration)
                    entropy = (
                            action_type_dist.entropy() +
                            distance_dist.entropy() +
                            max_rounds_dist.entropy() +
                            suppress_only_dist.entropy() +
                            adjust_fire_rate_dist.entropy() +
                            formation_dist.entropy()
                    ).mean()

                    # Combined log probability for all action components
                    new_logprobs = (
                            action_type_logprobs +
                            distance_logprobs +
                            max_rounds_logprobs +
                            suppress_only_logprobs +
                            adjust_fire_rate_logprobs +
                            formation_logprobs
                    )

                    # Calculate ratios for PPO
                    ratios = torch.exp(new_logprobs - batch_logprobs)

                    # Apply ratio clipping
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * batch_advantages

                    # Calculate losses
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value loss using MSE with normalized returns
                    critic_loss = F.mse_loss(state_values, batch_returns)

                    # Entropy loss
                    entropy_loss = -entropy_coef * entropy

                    # Combined loss
                    loss = actor_loss + value_coef * critic_loss + entropy_loss

                    # Perform backpropagation
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()

                    # Apply gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    # Record losses
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropy_losses.append(-entropy.item())

                except Exception as e:
                    print(f"Error in batch update for agent {self.agent_id}: {e}")
                    import traceback
                    traceback.print_exc()

        # Update old actor to current actor
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Increment update counter
        self.update_count += 1

        # Return mean losses
        actor_loss_mean = np.mean(actor_losses) if actor_losses else 0
        critic_loss_mean = np.mean(critic_losses) if critic_losses else 0
        entropy_loss_mean = np.mean(entropy_losses) if entropy_losses else 0

        return actor_loss_mean, critic_loss_mean, entropy_loss_mean

    def save(self, filepath):
        """Save model checkpoints with version tracking"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'old_actor': self.old_actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'agent_id': self.agent_id
        }, filepath)

    def load(self, filepath):
        """Load model checkpoints with version tracking"""
        try:
            checkpoint = torch.load(filepath)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.old_actor.load_state_dict(checkpoint['old_actor'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

            # Load update count if available
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']

            # Verify agent ID if available
            if 'agent_id' in checkpoint and checkpoint['agent_id'] != self.agent_id:
                print(f"WARNING: Loading model trained for agent {checkpoint['agent_id']} "
                      f"into agent {self.agent_id}")

            return True
        except Exception as e:
            print(f"Error loading model for agent {self.agent_id}: {e}")
            return False


# MARL PPO class
class WarGameMARLPPO:
    """MARL PPO that works directly with the existing environment"""

    def __init__(self, env, action_dim=5, lr=0.0003):
        self.env = env
        self.action_dim = action_dim
        self.lr = lr

        # Initialize agent policies
        self.agent_policies = {}

        # Initialize enhanced memory
        self.memory = Memory()

        # Internal timestep counter for memory tracking
        self.timestep = 0

        # Add reference to self in environment for reward collection
        # Use the generic 'marl_algorithm' name for algorithm-agnostic design
        if hasattr(env, 'marl_algorithm'):
            print("WARNING: Environment already has a marl_algorithm reference. Overwriting.")
        env.marl_algorithm = self

    def add_agent(self, agent_id):
        """Add a new agent policy with proper memory initialization"""
        if agent_id not in self.agent_policies:
            self.agent_policies[agent_id] = AgentPolicy(
                action_dim=self.action_dim,
                lr=self.lr,
                agent_id=agent_id
            )
            print(f"Created new policy for agent {agent_id}")

        # Always initialize memory for the agent
        self.memory.initialize_agent(agent_id)

    def remove_agent(self, agent_id):
        """Remove an agent policy"""
        if agent_id in self.agent_policies:
            del self.agent_policies[agent_id]

    def initialize_episode(self, agent_ids):
        """
        Initialize memory and policies for a new episode.
        Enhanced with better debugging and verification.

        Args:
            agent_ids: List of agent IDs in this episode
        """
        # Reset timestep counter
        self.timestep = 0

        # Clear previous episode memory
        self.memory.clear_memory()

        debug_enabled = True  # Always enable debugging for episode initialization

        if debug_enabled:
            print(f"Initializing episode with {len(agent_ids)} agents: {agent_ids}")

        # Initialize memory for all agents
        for agent_id in agent_ids:
            # Initialize memory
            self.memory.initialize_agent(agent_id)

            # Create policy if needed
            if agent_id not in self.agent_policies:
                self.add_agent(agent_id)
                if debug_enabled:
                    print(f"Created new policy for agent {agent_id}")

        if debug_enabled:
            print(f"Episode initialized with {len(self.memory.current_episode_agents)} agents in memory")
            print(f"Agent IDs in memory: {sorted(list(self.memory.current_episode_agents))}")

    def select_actions(self, observations):
        """
        Select actions for all agents with improved memory initialization.

        Args:
            observations: Dictionary mapping agent IDs to observations

        Returns:
            Dictionary mapping agent IDs to actions
        """
        debug_enabled = self.timestep < 10  # Enable debugging for first few steps

        if debug_enabled:
            print(f"\nSelecting actions at timestep {self.timestep} for {len(observations)} agents")

        # Ensure memory is initialized for all agents in this episode
        for agent_id in observations:
            if agent_id not in self.memory.current_episode_agents:
                self.memory.initialize_agent(agent_id)
                if debug_enabled:
                    print(f"Initialized agent {agent_id} in memory.current_episode_agents")

        actions = {}

        for agent_id, observation in observations.items():
            try:
                # Get or create policy for this agent
                if agent_id not in self.agent_policies:
                    self.add_agent(agent_id)
                    if debug_enabled:
                        print(f"Created new policy for agent {agent_id}")

                # Get action, log probability and state value
                policy = self.agent_policies[agent_id]
                action, logprob, state_value = policy.select_action_with_logprob(observation)

                # Store in memory
                if agent_id not in self.memory.states:
                    self.memory.states[agent_id] = []
                if agent_id not in self.memory.actions:
                    self.memory.actions[agent_id] = []
                if agent_id not in self.memory.logprobs:
                    self.memory.logprobs[agent_id] = []
                if agent_id not in self.memory.values:
                    self.memory.values[agent_id] = []
                if agent_id not in self.memory.rewards:  # ADDED: Initialize rewards if missing
                    self.memory.rewards[agent_id] = []
                if agent_id not in self.memory.is_terminals:  # ADDED: Initialize terminals if missing
                    self.memory.is_terminals[agent_id] = []

                self.memory.states[agent_id].append(observation)
                self.memory.actions[agent_id].append(action)
                self.memory.logprobs[agent_id].append(logprob)
                self.memory.values[agent_id].append(state_value)

                # Update last updated timestamp
                self.memory.last_updated[agent_id] = self.timestep

                # Add action to result dictionary
                actions[agent_id] = action

                if debug_enabled:
                    print(f"Selected action for agent {agent_id}, memory lengths: "
                          f"states={len(self.memory.states[agent_id])}, "
                          f"actions={len(self.memory.actions[agent_id])}, "
                          f"logprobs={len(self.memory.logprobs[agent_id])}, "
                          f"values={len(self.memory.values[agent_id])}, "
                          f"rewards={len(self.memory.rewards[agent_id])}, "  # ADDED: Debug for rewards
                          f"terminals={len(self.memory.is_terminals[agent_id])}")  # ADDED: Debug for terminals

            except Exception as e:
                import traceback
                print(f"Error selecting action for agent {agent_id}: {e}")
                traceback.print_exc()

                # Use a default action
                actions[agent_id] = self._default_action()

        # Increment timestep after all actions are selected
        self.timestep += 1

        # Verify memory consistency
        if debug_enabled:
            # Use "actions_only" mode since rewards haven't been stored yet
            inconsistent = self.memory.verify_memory_consistency(check_mode="actions_only")
            if inconsistent:
                print("WARNING: Memory inconsistency after select_actions:")
                for agent_id, lengths in inconsistent.items():
                    print(f"  Agent {agent_id}: {lengths}")

        return actions

    def store_rewards_and_terminals(self, rewards, dones, truncs):
        """
        Explicitly store rewards and terminal flags for all agents.
        Modified to handle normalized reward values in v11.

        Args:
            rewards: Dictionary mapping agent IDs to rewards
            dones: Dictionary mapping agent IDs to done flags
            truncs: Dictionary mapping agent IDs to truncation flags
        """
        debug_enabled = self.timestep <= 10  # Enable debugging for first few steps

        # if debug_enabled:
        #     print(f"\nDEBUG: Storing rewards at timestep {self.timestep - 1} for {len(self.memory.current_episode_agents)} agents")
        #     print(f"DEBUG: Original dones: {dones}")
        #     print(f"DEBUG: Original truncs: {truncs}")

        # CRITICAL FIX: Override terminal flags for early steps to prevent premature termination
        # Only accept termination after enough steps for meaningful learning
        min_steps_for_termination = 5  # Need at least this many steps for PPO update

        # If we're in early steps, override termination
        if self.timestep < min_steps_for_termination:
            original_dones = dones.copy()
            original_truncs = truncs.copy()

            # Override terminal flags to False for all agents
            dones = {agent_id: False for agent_id in dones}
            truncs = {agent_id: False for agent_id in truncs}

            # if debug_enabled:
            #     print(f"DEBUG: Overriding terminal flags for step {self.timestep - 1} - "
            #           f"need at least {min_steps_for_termination} steps for learning")
            #     print(f"DEBUG: Modified dones: {dones}")
            #     print(f"DEBUG: Modified truncs: {truncs}")

        # Store rewards and terminal flags for each agent
        for agent_id in self.memory.current_episode_agents:
            try:
                # Check if this agent has any actions/states stored
                if agent_id not in self.memory.states or len(self.memory.states[agent_id]) == 0:
                    if debug_enabled:
                        print(f"DEBUG: Agent {agent_id} has no states, skipping reward storage")
                    continue

                # Get the number of actions stored (should match states)
                num_actions = len(self.memory.actions.get(agent_id, []))
                num_rewards = len(self.memory.rewards.get(agent_id, []))

                # If rewards and actions don't match, there's a problem
                if num_rewards > num_actions:
                    if debug_enabled:
                        print(
                            f"DEBUG: Agent {agent_id} has more rewards ({num_rewards}) than actions ({num_actions}), trimming")
                    # Trim excess rewards
                    self.memory.rewards[agent_id] = self.memory.rewards[agent_id][:num_actions]
                    self.memory.is_terminals[agent_id] = self.memory.is_terminals[agent_id][:num_actions]
                    num_rewards = num_actions

                # Get current reward and terminal flag for this agent
                reward = rewards.get(agent_id, 0.0)
                is_terminal = dones.get(agent_id, False) or truncs.get(agent_id, False)

                # Store reward and terminal flag
                if agent_id not in self.memory.rewards:
                    self.memory.rewards[agent_id] = []
                if agent_id not in self.memory.is_terminals:
                    self.memory.is_terminals[agent_id] = []

                # Only add a reward if we need more to match actions
                rewards_to_add = num_actions - num_rewards
                if rewards_to_add > 0:
                    # Add the current reward
                    self.memory.rewards[agent_id].append(reward)
                    self.memory.is_terminals[agent_id].append(is_terminal)

                    # Update last updated timestamp
                    self.memory.last_updated[agent_id] = self.timestep - 1

                    # if debug_enabled:
                    #     print(f"DEBUG: Added reward {reward:.4f} (terminal={is_terminal}) for agent {agent_id}")
                    #     print(f"DEBUG: Agent {agent_id} memory state: actions={num_actions}, rewards={num_rewards + 1}")

            except Exception as e:
                import traceback
                print(f"ERROR: Failed to store rewards for agent {agent_id}: {e}")
                traceback.print_exc()

        # Verify memory consistency after storing rewards
        if debug_enabled:
            inconsistent = {}
            for agent_id in self.memory.current_episode_agents:
                states_len = len(self.memory.states.get(agent_id, []))
                actions_len = len(self.memory.actions.get(agent_id, []))
                rewards_len = len(self.memory.rewards.get(agent_id, []))
                terminals_len = len(self.memory.is_terminals.get(agent_id, []))

                if states_len != actions_len or actions_len != rewards_len or rewards_len != terminals_len:
                    inconsistent[agent_id] = {
                        'states': states_len,
                        'actions': actions_len,
                        'rewards': rewards_len,
                        'is_terminals': terminals_len
                    }

            # if inconsistent:
            #     print("WARNING: Memory still inconsistent after storing rewards:")
            #     for agent_id, lengths in inconsistent.items():
            #         print(f"  Agent {agent_id}: {lengths}")
            # else:
            #     print("DEBUG: Memory consistency verified after storing rewards")

    def check_memory_consistency(self):
        """
        Check consistency of memory structures for all agents.

        Returns:
            Dictionary with consistency check results
        """
        return self.memory.check_consistency()

    def fix_memory_inconsistencies(self):
        """
        Attempt to fix memory inconsistencies by truncating to shortest length.

        Returns:
            Dictionary with fix results
        """
        return self.memory.fix_inconsistencies()

    def _default_action(self):
        """Create a default action when selection fails"""
        return {
            'action_type': 0,  # MOVE
            'movement_params': {
                'direction': np.array([0.0, 0.0]),
                'distance': [1]
            },
            'engagement_params': {
                'target_pos': np.array([50.0, 50.0]),
                'max_rounds': [5],
                'suppress_only': 0,
                'adjust_for_fire_rate': 0
            },
            'formation': 0
        }

    def update(self, clip_param=0.2, value_coef=0.5, entropy_coef=0.01, ppo_epochs=4, batch_size=64, gamma=0.99,
               gae_lambda=0.95):
        """
        Update policies for all agents based on collected experience.
        FIXED: Now handles potential inconsistencies in memory data.

        Args:
            clip_param: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy coefficient for controlling exploration vs exploitation
            ppo_epochs: Number of epochs for PPO update
            batch_size: Mini-batch size for updates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (average actor loss, average critic loss, average entropy loss)
        """
        debug_enabled = True  # Always enable debugging for policy updates

        if debug_enabled:
            print(f"\nUpdating policies for all agents (entropy_coef={entropy_coef:.4f})")

        # Verify memory consistency before update
        inconsistent_agents = self.memory.verify_memory_consistency()
        if inconsistent_agents:
            print("WARNING: Memory inconsistency detected before update:")
            for agent_id, lengths in inconsistent_agents.items():
                print(f"  Agent {agent_id}: {lengths}")

        actor_losses = []
        critic_losses = []
        entropy_losses = []
        updated_agents = 0

        # Update each agent
        for agent_id in self.memory.current_episode_agents:
            try:
                # Skip agents with no data
                if agent_id not in self.memory.states or not self.memory.states[agent_id]:
                    if debug_enabled:
                        print(f"DEBUG: Agent {agent_id} has no data, skipping update")
                    continue

                # Get agent data
                agent_data = self._get_agent_data(agent_id)

                # Verify data lengths
                min_data_length = min(
                    len(agent_data["states"]),
                    len(agent_data["actions"]),
                    len(agent_data["logprobs"]),
                    len(agent_data["rewards"]),
                    len(agent_data["is_terminals"]),
                    len(agent_data["values"])
                )

                if min_data_length < 2:  # Need at least 2 steps for meaningful update
                    print(f"ERROR: Agent {agent_id} has insufficient data ({min_data_length} points)")
                    continue

                # FIXED: Trim data to consistent length
                agent_data["states"] = agent_data["states"][:min_data_length]
                agent_data["actions"] = agent_data["actions"][:min_data_length]
                agent_data["logprobs"] = agent_data["logprobs"][:min_data_length]
                agent_data["rewards"] = agent_data["rewards"][:min_data_length]
                agent_data["is_terminals"] = agent_data["is_terminals"][:min_data_length]
                agent_data["values"] = agent_data["values"][:min_data_length]

                # if debug_enabled:
                #     print(f"DEBUG: Agent {agent_id} has {min_data_length} data points for update")

                # Get agent policy
                policy = self.agent_policies[agent_id]

                # Update policy
                actor_loss, critic_loss, entropy_loss = policy.update(
                    agent_data=agent_data,
                    clip_param=clip_param,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    ppo_epochs=ppo_epochs,
                    batch_size=min(batch_size, min_data_length),
                    gamma=gamma,
                    gae_lambda=gae_lambda
                )

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropy_losses.append(entropy_loss)
                updated_agents += 1

                print(
                    f"Updated policy for agent {agent_id}: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}, entropy_loss={entropy_loss:.4f}")

            except Exception as e:
                import traceback
                print(f"ERROR: Failed to update agent {agent_id}: {e}")
                traceback.print_exc()

        # Calculate average losses
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0
        avg_entropy_loss = sum(entropy_losses) / len(entropy_losses) if entropy_losses else 0

        print(
            f"Average losses for {updated_agents} agents: actor={avg_actor_loss:.4f}, critic={avg_critic_loss:.4f}, entropy={avg_entropy_loss:.4f}")

        return avg_actor_loss, avg_critic_loss, avg_entropy_loss

    def _get_agent_data(self, agent_id):
        """
        Get all collected data for a specific agent from memory.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary containing agent's experience data
        """
        if agent_id not in self.memory.current_episode_agents:
            raise ValueError(f"Agent {agent_id} not in current episode")

        # Get data from memory with safety checks
        states = self.memory.states.get(agent_id, [])
        actions = self.memory.actions.get(agent_id, [])
        logprobs = self.memory.logprobs.get(agent_id, [])
        rewards = self.memory.rewards.get(agent_id, [])
        is_terminals = self.memory.is_terminals.get(agent_id, [])
        values = self.memory.values.get(agent_id, [])

        # Verify data exists
        if not states or not actions or not logprobs or not rewards or not is_terminals or not values:
            print(f"WARNING: Incomplete data for agent {agent_id}")
            # Return empty data set with correct structure
            return {
                "states": [],
                "actions": [],
                "logprobs": [],
                "rewards": [],
                "is_terminals": [],
                "values": []
            }

        # Verify consistent lengths
        min_length = min(len(states), len(actions), len(logprobs), len(rewards), len(is_terminals), len(values))

        # Create consistent data set
        return {
            "states": states[:min_length],
            "actions": actions[:min_length],
            "logprobs": logprobs[:min_length],
            "rewards": rewards[:min_length],
            "is_terminals": is_terminals[:min_length],
            "values": values[:min_length]
        }

    def save_agents(self, directory):
        """Save all agent policies by consistent agent ID"""
        os.makedirs(directory, exist_ok=True)
        for agent_id, policy in self.agent_policies.items():
            filepath = os.path.join(directory, f"agent_{agent_id}.pt")
            try:
                policy.save(filepath)
                print(f"Saved policy for agent {agent_id} to {filepath}")
            except Exception as e:
                print(f"Error saving agent {agent_id}: {e}")

    def load_agents(self, directory):
        """Load all agent policies by consistent agent ID"""
        try:
            # Get list of all saved agents
            agent_files = [f for f in os.listdir(directory) if f.startswith("agent_") and f.endswith(".pt")]

            for agent_file in agent_files:
                # Extract agent ID from filename
                agent_id = int(agent_file.split("_")[1].split(".")[0])

                # Create policy if needed
                if agent_id not in self.agent_policies:
                    self.add_agent(agent_id)

                # Load policy
                filepath = os.path.join(directory, agent_file)
                self.agent_policies[agent_id].load(filepath)
                print(f"Loaded policy for agent {agent_id} from {filepath}")

            print(f"Loaded {len(agent_files)} agent policies from {directory}")
        except Exception as e:
            print(f"Error loading agents: {e}")


# Custom training function
def custom_training(
        num_episodes=50,
        max_steps_per_episode=100,
        map_file="training_map.csv",
        objective_location=(73, 75),
        enemy_positions=None,
        unit_start_positions=None,
        save_interval=5,
        log_interval=1,
        output_dir="./custom_training_output",
        use_wandb=False,
        wandb_project="marl_wargaming",
        wandb_entity=None,
        use_tqdm=True,
        gpu_id=0,

        # Learning rate schedule parameters
        initial_learning_rate=5e-4,
        final_learning_rate=5e-5,
        lr_decay_method='linear',

        # Entropy annealing parameters
        initial_entropy_coef=0.15,
        final_entropy_coef=0.01,
        entropy_annealing_episodes=100,

        # Custom LR callback for warmup
        custom_lr_callback=None,

        # Other PPO parameters
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.1,
        value_coef=0.5,
        entropy_coef=0.01,  # Default value, will be overridden by annealing
        ppo_epochs=4,
        batch_size=64,

        # Early stopping parameters
        patience=40,
        min_delta=0.01
):
    """
    Run MARL PPO training with custom map, objectives and unit placements.
    Enhanced with consistent agent IDs across episodes, early stopping, learning rate scheduling,
    and entropy coefficient annealing.

    Args:
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        map_file: CSV file with terrain information
        objective_location: (x, y) coordinates of the objective
        enemy_positions: List of (x, y) coordinates for enemy placements
        unit_start_positions: Dict mapping unit names to starting positions
        save_interval: Save model every N episodes
        log_interval: Log progress every N episodes
        output_dir: Directory to save models and logs
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name (username or team)
        use_tqdm: Whether to use tqdm progress bars
        gpu_id: GPU ID to use (None for CPU)
        initial_learning_rate: Starting learning rate for PPO optimizer
        final_learning_rate: Final learning rate after decay
        lr_decay_method: Method for learning rate decay ('linear', 'exponential', 'cosine')
        initial_entropy_coef: Starting entropy coefficient (higher = more exploration)
        final_entropy_coef: Final entropy coefficient (lower = more exploitation)
        entropy_annealing_episodes: Number of episodes over which to anneal entropy
        gamma: Discount factor for future rewards
        gae_lambda: GAE parameter for advantage estimation
        clip_param: Clipping parameter for PPO update
        value_coef: Value function loss coefficient
        entropy_coef: Base entropy coefficient (overridden by annealing)
        ppo_epochs: Number of epochs for PPO update
        batch_size: Batch size for PPO update
        patience: Number of episodes to wait for improvement before stopping
        min_delta: Minimum improvement required to reset patience counter
        custom_lr_callback: Optional function(episode, initial_lr, final_lr) that returns a custom lr value
                   for the given episode or None to use standard schedule

    Returns:
        Trained MARL PPO instance and training statistics
    """
    import os
    import time
    import math
    from datetime import datetime, timedelta

    # Set defaults for mutable arguments
    if enemy_positions is None:
        enemy_positions = [(50, 50)]  # [(73, 80), (73, 71), (78, 75), (66, 75)]

    if unit_start_positions is None:
        unit_start_positions = {
            "1SQD": (10, 20),
            "2SQD": (10, 50),
            "3SQD": (10, 80),
            "GTM1": (10, 30),
            "GTM2": (10, 70),
            "JTM1": (10, 40),
            "JTM2": (10, 60)
        }

    # Function to calculate the current entropy coefficient
    def calculate_entropy_coef(current_episode):
        """Calculate the current entropy coefficient using linear annealing."""
        if current_episode >= entropy_annealing_episodes:
            return final_entropy_coef

        # Linear annealing from initial to final
        progress = current_episode / entropy_annealing_episodes
        return initial_entropy_coef + progress * (final_entropy_coef - initial_entropy_coef)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Configure GPU if specified
    if gpu_id is not None:
        import torch
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Initialize wandb to None by default
    wandb = None

    # Initialize W&B if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(project=wandb_project, entity=wandb_entity,
                       config={
                           "num_episodes": num_episodes,
                           "max_steps": max_steps_per_episode,
                           "objective": objective_location,
                           "enemy_count": len(enemy_positions),
                           "map_file": map_file,
                           "initial_learning_rate": initial_learning_rate,
                           "final_learning_rate": final_learning_rate,
                           "lr_decay_method": lr_decay_method,
                           "initial_entropy_coef": initial_entropy_coef,
                           "final_entropy_coef": final_entropy_coef,
                           "entropy_annealing_episodes": entropy_annealing_episodes,
                           "gamma": gamma,
                           "gae_lambda": gae_lambda,
                           "clip_param": clip_param,
                           "value_coef": value_coef,
                           "ppo_epochs": ppo_epochs,
                           "batch_size": batch_size,
                           "patience": patience,
                           "min_delta": min_delta
                       })
            print("Weights & Biases logging initialized")
        except ImportError:
            print("Weights & Biases not installed. Run 'pip install wandb' to use this feature.")
            wandb = None
            use_wandb = False
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            wandb = None
            use_wandb = False

    # Define fallback functions for tqdm/trange before the import attempt
    def simple_tqdm(iterable, *args, **kwargs):
        return iterable

    def simple_trange(n, *args, **kwargs):
        return range(n)

    # Use these as defaults
    tqdm = simple_tqdm
    trange = simple_trange

    # Setup tqdm if requested
    if use_tqdm:
        try:
            from tqdm import tqdm, trange
            print("Using tqdm for progress tracking")
        except ImportError:
            print("tqdm not installed. Run 'pip install tqdm' to use progress bars.")

            # Define tqdm and trange as simple alternatives if not available
            def tqdm(iterable, *args, **kwargs):
                return iterable

            def trange(n, *args, **kwargs):
                return range(n)

            use_tqdm = False

    # Set up logging
    log_file = os.path.join(output_dir, "logs", "training_log.csv")
    with open(log_file, 'w') as f:
        f.write(
            "episode,avg_reward,friendly_casualties,enemy_casualties,steps,actor_loss,critic_loss,entropy_loss,elapsed_time,learning_rate,entropy_coef\n")

    # Initialize environment with custom map size
    from WarGamingEnvironment_v12 import EnvironmentConfig, MARLMilitaryEnvironment, ForceType, UnitType

    # Estimate map size based on objective and unit positions
    all_positions = [objective_location] + list(unit_start_positions.values()) + enemy_positions
    max_x = max(pos[0] for pos in all_positions) + 20  # Add margin
    max_y = max(pos[1] for pos in all_positions) + 20  # Add margin

    # Create environment config
    config = EnvironmentConfig(width=max_x, height=max_y, debug_level=0)
    env = MARLMilitaryEnvironment(config, objective_position=objective_location)

    # Initialize PPO with initial learning rate
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=initial_learning_rate)

    # Connect environment and PPO for reward handling
    setattr(env, 'marl_algorithm', marl_ppo)

    # Initialize agent role mapping to ensure consistent IDs across episodes
    env.agent_manager.initialize_agent_role_mapping()

    # Training statistics
    all_rewards = []
    episode_lengths = []
    friendly_casualties_history = []
    enemy_casualties_history = []
    episode_times = []
    learning_rates = []  # Track learning rates
    entropy_coefs = []  # Track entropy coefficients
    training_start_time = time.time()

    # Early stopping tracking
    best_avg_reward = float('-inf')
    best_enemy_casualties = 0
    episodes_without_improvement = 0
    best_model_saved = False

    # Log start time
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create episode iterator with tqdm if available
    episode_range = trange(num_episodes) if use_tqdm else range(num_episodes)

    # Training loop
    for episode in episode_range:
        episode_start_time = time.time()

        # Calculate learning rate using selected decay method or custom callback
        if custom_lr_callback:
            # Try to use the custom callback for warmup or other adjustments
            custom_lr = custom_lr_callback(episode, initial_learning_rate, final_learning_rate)
            if custom_lr is not None:
                current_lr = custom_lr
            else:
                # Fall back to standard decay if callback returns None
                progress = episode / (num_episodes - 1) if num_episodes > 1 else 1.0
                if lr_decay_method == 'linear':
                    current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)
                elif lr_decay_method == 'exponential':
                    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_episodes)
                    current_lr = initial_learning_rate * (decay_rate ** episode)
                elif lr_decay_method == 'cosine':
                    current_lr = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (
                            1 + math.cos(math.pi * progress))
                else:  # Default to linear if method not recognized
                    current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)
        else:
            # Standard learning rate decay schedule
            progress = episode / (num_episodes - 1) if num_episodes > 1 else 1.0
            if lr_decay_method == 'linear':
                current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)
            elif lr_decay_method == 'exponential':
                decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_episodes)
                current_lr = initial_learning_rate * (decay_rate ** episode)
            elif lr_decay_method == 'cosine':
                current_lr = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (
                        1 + math.cos(math.pi * progress))
            else:  # Default to linear if method not recognized
                current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)

        # Calculate current entropy coefficient
        current_entropy_coef = calculate_entropy_coef(episode)

        # Update learning rate for all agents
        for agent_id, policy in marl_ppo.agent_policies.items():
            for param_group in policy.actor_optimizer.param_groups:
                param_group['lr'] = current_lr
            for param_group in policy.critic_optimizer.param_groups:
                param_group['lr'] = current_lr

        learning_rates.append(current_lr)
        entropy_coefs.append(current_entropy_coef)

        # Update tqdm description if available
        if use_tqdm:
            episode_range.set_description(
                f"Episode {episode + 1}/{num_episodes} (LR: {current_lr:.6f}, Ent: {current_entropy_coef:.4f})")
        else:
            print(
                f"\nStarting Episode {episode + 1}/{num_episodes} (Learning Rate: {current_lr:.6f}, Entropy Coef: {current_entropy_coef:.4f})")

        # Reset environment with retry mechanism
        max_reset_attempts = 3
        observations = None

        for attempt in range(max_reset_attempts):
            try:
                # Reset with objective and platoon
                options = {
                    'unit_init': {
                        'platoon': {
                            'position': (5, 5),
                            'number': 1
                        },
                        'objective': objective_location
                    }
                }

                observations, _ = env.reset(options=options)

                # Verify agents exist
                if not env.agent_ids:
                    print("No agents found after reset, creating platoon manually...")
                    from US_Army_PLT_Composition_v2 import US_IN_create_platoon

                    platoon_id = US_IN_create_platoon(env, plt_num=1, start_position=(5, 5))
                    env.agent_ids = env.agent_manager.identify_agents_from_platoon(platoon_id)

                print(f"Environment reset with {len(env.agent_ids)} agents")
                break
            except Exception as e:
                print(f"Reset attempt {attempt + 1} failed: {e}")
                time.sleep(1)

                if attempt == max_reset_attempts - 1:
                    print("Failed to reset environment after multiple attempts")
                    return marl_ppo, (all_rewards, episode_lengths)

        # Create custom platoon with specific unit positions
        try:
            # Define the custom create_platoon function to place units at specific positions
            def US_IN_create_platoon_custom_positions(env, plt_num=1, unit_positions=None):
                """Create a platoon with units at specified positions"""
                from US_Army_PLT_Composition_v2 import US_IN_create_platoon

                # First create the platoon normally at a default position
                plt_id = US_IN_create_platoon(env, plt_num, start_position=(1, 1))

                # Get all squads and teams in the platoon
                plt_children = env.get_unit_children(plt_id)

                # Now move each unit to its specified position
                for child_id in plt_children:
                    # Get the string identifier
                    string_id = env.get_unit_property(child_id, 'string_id', '')

                    # Find the matching key in unit_positions
                    matching_key = None
                    for key in unit_positions.keys():
                        if key in string_id:
                            matching_key = key
                            break

                    if matching_key:
                        # Move the unit to its position
                        new_pos = unit_positions[matching_key]
                        env.update_unit_position(child_id, new_pos)

                        # Also move all children (soldiers) of this unit
                        for member_id in env.get_unit_children(child_id):
                            env.update_unit_position(member_id, new_pos)

                        print(f"Positioned {string_id} at {new_pos}")

                return plt_id

            # Create the platoon with custom positions
            platoon_id = US_IN_create_platoon_custom_positions(env, plt_num=1, unit_positions=unit_start_positions)

            # Use consistent agent ID mapping
            env.agent_ids = env.agent_manager.map_current_units_to_agent_ids(platoon_id)

            # Get observations for consistent agent IDs
            observations = {}
            for agent_id in env.agent_ids:
                unit_id = env.agent_manager.get_current_unit_id(agent_id)
                if unit_id:
                    observations[agent_id] = env._get_observation_for_agent(unit_id)

            print(f"Created platoon with {len(env.agent_ids)} consistently mapped agents")

        except Exception as e:
            print(f"Error creating custom platoon: {e}")
            print("Falling back to default platoon creation...")

            # Fallback to standard platoon creation
            from US_Army_PLT_Composition_v2 import US_IN_create_platoon
            default_position = list(unit_start_positions.values())[0] if unit_start_positions else (50, 50)
            platoon_id = US_IN_create_platoon(env, plt_num=1, start_position=default_position)

            # Use consistent agent ID mapping
            env.agent_ids = env.agent_manager.map_current_units_to_agent_ids(platoon_id)

            # Get observations for consistent agent IDs
            observations = {}
            for agent_id in env.agent_ids:
                unit_id = env.agent_manager.get_current_unit_id(agent_id)
                if unit_id:
                    observations[agent_id] = env._get_observation_for_agent(unit_id)

        # Add enemies at specified positions
        from US_Army_PLT_Composition_v2 import US_IN_Role
        print(f"Adding {len(enemy_positions)} enemies at specified positions...")

        enemy_ids = []
        for i, pos in enumerate(enemy_positions):
            # Create enemy team
            enemy_id = env.create_unit(
                unit_type=UnitType.INFANTRY_TEAM,
                unit_id_str=f"ENEMY-{i + 1}",
                start_position=pos
            )
            env.update_unit_property(enemy_id, 'force_type', ForceType.ENEMY)
            enemy_ids.append(enemy_id)

            # Add soldiers to the enemy team
            for j in range(3):  # Add 3 soldiers per enemy team
                soldier_role = US_IN_Role.RIFLEMAN
                if j == 0:  # First soldier is team leader
                    soldier_role = US_IN_Role.TEAM_LEADER

                soldier_id = env.create_soldier(
                    role=soldier_role,
                    unit_id_str=f"ENEMY-{i + 1}-{j + 1}",
                    position=pos,
                    is_leader=(j == 0)  # First soldier is leader
                )
                env.update_unit_property(soldier_id, 'force_type', ForceType.ENEMY)
                env.set_unit_hierarchy(soldier_id, enemy_id)

        print(f"Added {len(enemy_ids)} enemy teams")

        # Count initial enemies for tracking casualties
        initial_enemy_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    initial_enemy_count += 1

        # Count initial friendlies
        initial_friendly_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    initial_friendly_count += 1

        # Episode variables
        episode_rewards = {agent_id: 0 for agent_id in env.agent_ids}
        episode_length = 0
        done = False
        step_failures = 0

        # Create step iterator with tqdm if available
        step_range = trange(max_steps_per_episode) if use_tqdm else range(max_steps_per_episode)

        # Episode loop
        for step in step_range:
            if done:
                break

            # Update step progress bar if available
            if use_tqdm:
                step_range.set_description(f"Step {step + 1}/{max_steps_per_episode}")

            # Select actions
            try:
                actions = marl_ppo.select_actions(observations)
                if not use_tqdm and step % 10 == 0:  # Print only every 10 steps if no progress bar
                    print(f"Step {step}: Selected actions for {len(actions)} agents")
            except Exception as e:
                print(f"Error selecting actions: {e}")
                # Use default actions
                actions = {}
                for agent_id in env.agent_ids:
                    unit_id = env.agent_manager.get_current_unit_id(agent_id)
                    if unit_id:
                        unit_pos = env.get_unit_position(unit_id)

                        # Create a default direction toward objective
                        dir_x = objective_location[0] - unit_pos[0]
                        dir_y = objective_location[1] - unit_pos[1]

                        # Normalize direction
                        norm = (dir_x ** 2 + dir_y ** 2) ** 0.5
                        if norm > 0:
                            dir_x /= norm
                            dir_y /= norm

                        actions[agent_id] = {
                            'action_type': 0,  # MOVE
                            'movement_params': {
                                'direction': np.array([dir_x, dir_y]),
                                'distance': [1]
                            },
                            'engagement_params': {
                                'target_pos': np.array([15.0, 15.0]),  # Target objective
                                'max_rounds': [5],
                                'suppress_only': 0,
                                'adjust_for_fire_rate': 0
                            },
                            'formation': 0
                        }

            # Execute step
            try:
                next_observations, rewards, dones, truncs, infos = env.step(actions)
                step_failures = 0  # Reset failure counter on success

                # Store rewards and terminal flags
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward

                # Update observations
                observations = next_observations

                # Check if done
                done = all(dones.values()) or all(truncs.values())
                episode_length += 1

                # Update step progress description with rewards if using tqdm
                if use_tqdm:
                    avg_step_reward = sum(rewards.values()) / len(rewards) if rewards else 0
                    step_range.set_postfix(reward=f"{avg_step_reward:.2f}")

            except Exception as e:
                print(f"Error during step: {e}")
                step_failures += 1

                if step_failures >= 3:
                    print("Too many consecutive step failures, ending episode")
                    break
                else:
                    print("Attempting to continue episode...")
                    time.sleep(0.5)  # Wait before retry

        # Calculate episode statistics
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_times.append(episode_duration)

        # Calculate average reward - this is already normalized in v11
        # No need to scale it up artificially for display since the environment already handles this
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards) if episode_rewards else 0
        all_rewards.append(avg_reward)
        episode_lengths.append(episode_length)

        # Count casualties
        current_enemy_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    current_enemy_count += 1

        current_friendly_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    current_friendly_count += 1

        enemy_casualties = initial_enemy_count - current_enemy_count
        friendly_casualties = initial_friendly_count - current_friendly_count

        enemy_casualties_history.append(enemy_casualties)
        friendly_casualties_history.append(friendly_casualties)

        # Update policies with current entropy coefficient
        try:
            actor_loss, critic_loss, entropy_loss = marl_ppo.update(
                clip_param=clip_param,
                value_coef=value_coef,
                entropy_coef=current_entropy_coef,  # Use the annealed entropy coefficient
                ppo_epochs=ppo_epochs,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda
            )
        except Exception as e:
            print(f"Error during policy update: {e}")
            actor_loss, critic_loss, entropy_loss = 0, 0, 0

        # Calculate elapsed time
        elapsed_time = time.time() - training_start_time
        elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))

        # Log results
        if log_interval > 0 and episode % log_interval == 0:
            print(
                f"Episode {episode + 1}: Steps={episode_length}, Avg Reward={avg_reward:.2f}, LR={current_lr:.6f}, Ent={current_entropy_coef:.4f}")
            print(f"Friendly Casualties: {friendly_casualties}, Enemy Casualties: {enemy_casualties}")
            print(f"Losses: Actor={actor_loss:.4f}, Critic={critic_loss:.4f}, Entropy={entropy_loss:.4f}")
            print(f"Episode Time: {episode_duration:.1f}s, Total Training Time: {elapsed_time_str}")

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(
                    f"{episode},{avg_reward},{friendly_casualties},{enemy_casualties},{episode_length},"
                    f"{actor_loss},{critic_loss},{entropy_loss},{elapsed_time},{current_lr},{current_entropy_coef}\n")

            # Log to W&B if enabled
            if use_wandb and wandb is not None:
                wandb.log({
                    "episode": episode,
                    "reward": avg_reward,
                    "friendly_casualties": friendly_casualties,
                    "enemy_casualties": enemy_casualties,
                    "episode_length": episode_length,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy_loss": entropy_loss,
                    "episode_time": episode_duration,
                    "total_training_time": elapsed_time,
                    "learning_rate": current_lr,
                    "entropy_coefficient": current_entropy_coef
                })

        # Early stopping check - adjusted for normalized rewards
        # Consider both reward and enemy casualties for determining improvement
        # The min_delta parameter has been reduced to match normalized reward scale
        improvement = (avg_reward > best_avg_reward + min_delta) or (enemy_casualties > 0 and friendly_casualties == 0)

        if improvement:
            best_avg_reward = max(best_avg_reward, avg_reward)
            episodes_without_improvement = 0

            # Save best model
            best_dir = os.path.join(output_dir, "models", "best")
            try:
                marl_ppo.save_agents(best_dir)
                print(f"New best model saved with reward: {best_avg_reward:.2f}")
                best_model_saved = True
            except Exception as e:
                print(f"Error saving best model: {e}")
        else:
            episodes_without_improvement += 1
            print(f"Episode {episode + 1}: No improvement for {episodes_without_improvement} episodes")

        # Early stopping check - patience increased for normalized rewards
        if episodes_without_improvement >= patience:
            print(f"Early stopping triggered after {episode + 1} episodes")
            break

        # Save checkpoint
        if save_interval > 0 and episode > 0 and episode % save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, "models", f"checkpoint_{episode}")
            try:
                marl_ppo.save_agents(checkpoint_dir)
                print(f"Saved checkpoint at episode {episode}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    # Save final model if it wasn't saved as the best model
    if not best_model_saved:
        final_dir = os.path.join(output_dir, "models", "final")
        marl_ppo.save_agents(final_dir)
        print("Saved final model")
    else:
        print("Best model already saved, skipping final model save")

    # Calculate final training statistics
    total_training_time = time.time() - training_start_time
    avg_episode_time = sum(episode_times) / len(episode_times) if episode_times else 0

    print(f"\nTraining Complete!")
    print(f"Total Training Time: {str(timedelta(seconds=int(total_training_time)))}")
    print(f"Average Episode Time: {avg_episode_time:.2f} seconds")
    print(f"Average Reward: {sum(all_rewards) / len(all_rewards):.2f}")
    print(f"Average Episode Length: {sum(episode_lengths) / len(episode_lengths):.1f} steps")
    print(f"Total Friendly Casualties: {sum(friendly_casualties_history)}")
    print(f"Total Enemy Casualties: {sum(enemy_casualties_history)}")
    print(f"Best Reward: {best_avg_reward:.2f}")

    # Create plots
    try:
        # Create reward plot
        plt.figure(figsize=(10, 6))
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Training Rewards')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "reward_plot.png"))
        plt.close()

        # Create episode length plot
        plt.figure(figsize=(10, 6))
        plt.plot(episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "episode_length_plot.png"))
        plt.close()

        # Create casualties plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(friendly_casualties_history) + 1), friendly_casualties_history, 'b-',
                 label='Friendly Casualties')
        plt.plot(range(1, len(enemy_casualties_history) + 1), enemy_casualties_history, 'r-',
                 label='Enemy Casualties')
        plt.xlabel('Episode')
        plt.ylabel('Casualties')
        plt.title('Casualties Over Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "casualties_over_time.png"))
        plt.close()

        # Plot training time
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(episode_times) + 1), episode_times, 'g-')
        plt.xlabel('Episode')
        plt.ylabel('Time (seconds)')
        plt.title('Episode Training Time')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "training_time.png"))
        plt.close()

        # Plot learning rate over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(learning_rates) + 1), learning_rates, 'b-')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule ({lr_decay_method})')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "learning_rate.png"))
        plt.close()

        # Plot entropy coefficient over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(entropy_coefs) + 1), entropy_coefs, 'm-')
        plt.xlabel('Episode')
        plt.ylabel('Entropy Coefficient')
        plt.title('Entropy Coefficient Annealing')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "entropy_coefficient.png"))
        plt.close()

        # Plot early stopping metrics
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(all_rewards, 'b-')
        plt.axhline(y=best_avg_reward, color='r', linestyle='--', label=f'Best: {best_avg_reward:.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Reward Progress')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(enemy_casualties_history, 'r-')
        plt.axhline(y=max(enemy_casualties_history), color='g', linestyle='--',
                    label=f'Best: {max(enemy_casualties_history)}')
        plt.xlabel('Episode')
        plt.ylabel('Enemy Casualties')
        plt.title('Enemy Casualties Progress')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logs", "early_stopping_metrics.png"))
        plt.close()

        # Create combined metrics plot showing entropy, learning rate and rewards
        plt.figure(figsize=(12, 10))

        # Three subplots sharing x axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Reward subplot
        ax1.plot(all_rewards, 'b-')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress with Annealing')
        ax1.grid(True)

        # Entropy coefficient subplot
        ax2.plot(entropy_coefs, 'm-')
        ax2.set_ylabel('Entropy Coefficient')
        ax2.grid(True)

        # Learning rate subplot
        ax3.plot(learning_rates, 'g-')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logs", "combined_annealing_metrics.png"))
        plt.close()

    except Exception as e:
        print(f"Error creating plots: {e}")

    # Finish W&B run if enabled
    if use_wandb:
        if wandb is not None:  # Split into separate conditionals
            wandb.finish()

    return marl_ppo, (
        all_rewards, episode_lengths, friendly_casualties_history, enemy_casualties_history, episode_times,
        learning_rates, entropy_coefs, best_avg_reward  # Include entropy_coefs in return value
    )


def resume_custom_training(
        num_episodes=100,
        max_steps_per_episode=200,
        map_file="training_map.csv",
        objective_location=(73, 75),
        enemy_positions=None,
        unit_start_positions=None,

        # Load parameters
        load_from_dir="./custom_training_output/models/final",
        output_dir=None,  # Will create timestamped directory if None

        # Training parameters
        save_interval=20,
        log_interval=1,
        use_wandb=False,
        wandb_project="marl_wargaming",
        wandb_entity=None,
        use_tqdm=True,
        gpu_id=0,

        # Custom LR callback for warmup
        custom_lr_callback=None,

        # Entropy annealing parameters
        initial_entropy_coef=0.12,
        final_entropy_coef=0.01,
        entropy_annealing_episodes=80,

        # PPO hyperparameters with learning rate schedule
        initial_learning_rate=3e-4,
        final_learning_rate=1e-5,
        lr_decay_method='linear',
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_coef=2.0,  # Higher value coefficient based on previous improvements
        entropy_coef=0.01,  # This will be overridden by annealing
        ppo_epochs=4,
        batch_size=64,

        # Early stopping parameters
        patience=20,
        min_delta=0.01,
        stop_on_objective_reached=True  # Stop if objective is secured with minimal casualties
):
    """
    Resume MARL PPO training from a previously saved model with enhanced features.
    Includes learning rate scheduling, entropy coefficient annealing, and early stopping criteria.

    Args:
        num_episodes: Number of additional episodes to train
        max_steps_per_episode: Maximum steps per episode
        map_file: CSV file with terrain information
        objective_location: (x, y) coordinates of the objective
        enemy_positions: List of (x, y) coordinates for enemy placements
        unit_start_positions: Dict mapping unit names to starting positions
        load_from_dir: Directory containing saved agent models to load
        output_dir: Directory to save new models and logs (timestamped if None)
        save_interval: Save model every N episodes
        log_interval: Log progress every N episodes
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name (username or team)
        use_tqdm: Whether to use tqdm progress bars
        gpu_id: GPU ID to use (None for CPU)
        initial_entropy_coef: Starting entropy coefficient (higher = more exploration)
        final_entropy_coef: Final entropy coefficient (lower = more exploitation)
        entropy_annealing_episodes: Number of episodes over which to anneal entropy
        initial_learning_rate: Starting learning rate
        final_learning_rate: Final learning rate after decay
        lr_decay_method: Method for learning rate decay ('linear', 'exponential', 'cosine')
        gamma: Discount factor for future rewards
        gae_lambda: GAE parameter for advantage estimation
        clip_param: Clipping parameter for PPO update
        value_coef: Value function loss coefficient
        entropy_coef: Base entropy coefficient (overridden by annealing)
        ppo_epochs: Number of epochs for PPO update
        batch_size: Batch size for PPO update
        patience: Number of episodes to wait for improvement before stopping
        min_delta: Minimum improvement required to reset patience counter
        stop_on_objective_reached: Whether to stop training when objective is reached with minimal casualties
        custom_lr_callback: Optional function(episode, initial_lr, final_lr) that returns a custom lr value
                   for the given episode or None to use standard schedule

    Returns:
        Trained MARL PPO instance and training statistics
    """
    import os
    import time
    import math
    from datetime import datetime, timedelta

    # Function to calculate the current entropy coefficient
    def calculate_entropy_coef(current_episode):
        """Calculate the current entropy coefficient using linear annealing."""
        if current_episode >= entropy_annealing_episodes:
            return final_entropy_coef

        # Linear annealing from initial to final
        progress = current_episode / entropy_annealing_episodes
        return initial_entropy_coef + progress * (final_entropy_coef - initial_entropy_coef)

    # Set defaults for mutable arguments
    if enemy_positions is None:
        enemy_positions = [(73, 80), (73, 71), (78, 75), (66, 75)]

    if unit_start_positions is None:
        unit_start_positions = {
            "1SQD": (16, 10),
            "2SQD": (9, 20),
            "3SQD": (6, 30),
            "GTM1": (6, 9),
            "GTM2": (4, 24),
            "JTM1": (6, 5),
            "JTM2": (4, 32)
        }

    # Create timestamped output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./resumed_training_output_{timestamp}"

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Configure GPU if specified
    if gpu_id is not None:
        import torch
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Initialize wandb to None by default
    wandb = None

    # Initialize W&B if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(project=wandb_project, entity=wandb_entity,
                       config={
                           "num_episodes": num_episodes,
                           "max_steps": max_steps_per_episode,
                           "objective": objective_location,
                           "enemy_count": len(enemy_positions),
                           "map_file": map_file,
                           "initial_learning_rate": initial_learning_rate,
                           "final_learning_rate": final_learning_rate,
                           "lr_decay_method": lr_decay_method,
                           "initial_entropy_coef": initial_entropy_coef,
                           "final_entropy_coef": final_entropy_coef,
                           "entropy_annealing_episodes": entropy_annealing_episodes,
                           "gamma": gamma,
                           "gae_lambda": gae_lambda,
                           "clip_param": clip_param,
                           "value_coef": value_coef,
                           "ppo_epochs": ppo_epochs,
                           "batch_size": batch_size,
                           "patience": patience,
                           "min_delta": min_delta,
                           "resumed_from": load_from_dir
                       })
            print("Weights & Biases logging initialized")
        except ImportError:
            print("Weights & Biases not installed. Run 'pip install wandb' to use this feature.")
            wandb = None
            use_wandb = False
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            wandb = None
            use_wandb = False

    # Define fallback functions for tqdm/trange before the import attempt
    def simple_tqdm(iterable, *args, **kwargs):
        return iterable

    def simple_trange(n, *args, **kwargs):
        return range(n)

    # Use these as defaults
    tqdm = simple_tqdm
    trange = simple_trange

    # Setup tqdm if requested
    if use_tqdm:
        try:
            from tqdm import tqdm, trange
            print("Using tqdm for progress tracking")
        except ImportError:
            print("tqdm not installed. Run 'pip install tqdm' to use progress bars.")
            use_tqdm = False

    # Set up logging
    log_file = os.path.join(output_dir, "logs", "training_log.csv")
    with open(log_file, 'w') as f:
        f.write(
            "episode,avg_reward,friendly_casualties,enemy_casualties,steps,actor_loss,critic_loss,entropy_loss,"
            "elapsed_time,learning_rate,entropy_coef,objective_status\n")

    # Initialize environment with custom map size
    from WarGamingEnvironment_v12 import EnvironmentConfig, MARLMilitaryEnvironment, ForceType, UnitType

    # Estimate map size based on objective and unit positions
    all_positions = [objective_location] + list(unit_start_positions.values()) + enemy_positions
    max_x = max(pos[0] for pos in all_positions) + 20  # Add margin
    max_y = max(pos[1] for pos in all_positions) + 20  # Add margin

    # Create environment config
    config = EnvironmentConfig(width=max_x, height=max_y, debug_level=0)
    env = MARLMilitaryEnvironment(config, objective_position=objective_location)

    # Initialize PPO with specified parameters
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=initial_learning_rate)

    # Connect environment and PPO for reward handling
    setattr(env, 'marl_algorithm', marl_ppo)

    # Initialize agent role mapping to ensure consistent IDs across episodes
    env.agent_manager.initialize_agent_role_mapping()

    # Load previously trained agents
    print(f"Loading pre-trained agents from {load_from_dir}...")
    try:
        agents_loaded = marl_ppo.load_agents(load_from_dir)

        # Count how many agents were loaded
        agent_count = len(marl_ppo.agent_policies)
        print(f"Loaded {agent_count} pre-trained agents")

        # Print info about loaded agents
        for agent_id, policy in marl_ppo.agent_policies.items():
            print(f"Agent {agent_id}: Update count = {policy.update_count}")
    except Exception as e:
        print(f"Error loading agents: {e}")
        print("Will train with newly initialized agents")

    # Training statistics
    all_rewards = []
    episode_lengths = []
    friendly_casualties_history = []
    enemy_casualties_history = []
    episode_times = []
    learning_rates = []
    entropy_coefs = []
    training_start_time = time.time()
    objective_secured_count = 0
    consecutive_objective_secured = 0

    # Early stopping tracking
    best_avg_reward = float('-inf')
    best_enemy_casualties = 0
    episodes_without_improvement = 0
    objective_reached = False

    # Log start time
    print(f"Resumed training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create episode iterator with tqdm if available
    episode_range = trange(num_episodes) if use_tqdm else range(num_episodes)

    # Training loop
    for episode in episode_range:
        episode_start_time = time.time()

        # Calculate learning rate using selected decay method or custom callback
        if custom_lr_callback:
            # Try to use the custom callback for warmup or other adjustments
            custom_lr = custom_lr_callback(episode, initial_learning_rate, final_learning_rate)
            if custom_lr is not None:
                current_lr = custom_lr
            else:
                # Fall back to standard decay if callback returns None
                progress = episode / (num_episodes - 1) if num_episodes > 1 else 1.0
                if lr_decay_method == 'linear':
                    current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)
                elif lr_decay_method == 'exponential':
                    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_episodes)
                    current_lr = initial_learning_rate * (decay_rate ** episode)
                elif lr_decay_method == 'cosine':
                    current_lr = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (
                            1 + math.cos(math.pi * progress))
                else:  # Default to linear if method not recognized
                    current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)
        else:
            # Standard learning rate decay schedule
            progress = episode / (num_episodes - 1) if num_episodes > 1 else 1.0
            if lr_decay_method == 'linear':
                current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)
            elif lr_decay_method == 'exponential':
                decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_episodes)
                current_lr = initial_learning_rate * (decay_rate ** episode)
            elif lr_decay_method == 'cosine':
                current_lr = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (
                        1 + math.cos(math.pi * progress))
            else:  # Default to linear if method not recognized
                current_lr = initial_learning_rate + progress * (final_learning_rate - initial_learning_rate)

        # Calculate current entropy coefficient
        current_entropy_coef = calculate_entropy_coef(episode)

        # Update learning rate for all agents
        for agent_id, policy in marl_ppo.agent_policies.items():
            for param_group in policy.actor_optimizer.param_groups:
                param_group['lr'] = current_lr
            for param_group in policy.critic_optimizer.param_groups:
                param_group['lr'] = current_lr

        learning_rates.append(current_lr)
        entropy_coefs.append(current_entropy_coef)

        # Update tqdm description if available
        if use_tqdm:
            episode_range.set_description(
                f"Episode {episode + 1}/{num_episodes} (LR: {current_lr:.6f}, Ent: {current_entropy_coef:.4f})")
        else:
            print(
                f"\nStarting Episode {episode + 1}/{num_episodes} (Learning Rate: {current_lr:.6f}, Entropy Coef: {current_entropy_coef:.4f})")

        # Reset environment
        options = {
            'unit_init': {
                'objective': objective_location
            }
        }

        observations, _ = env.reset(options=options)

        # Load terrain from CSV if provided
        if map_file and os.path.exists(map_file):
            try:
                env.terrain_manager.load_from_csv(map_file)
                print(f"Loaded terrain from {map_file}")
            except Exception as e:
                print(f"Error loading terrain: {e}")

        # Create platoon with custom unit positions
        try:
            # Define the custom create_platoon function to place units at specific positions
            def US_IN_create_platoon_custom_positions(env, plt_num=1, unit_positions=None):
                """Create a platoon with units at specified positions"""
                from US_Army_PLT_Composition_v2 import US_IN_create_platoon

                # First create the platoon normally at a default position
                plt_id = US_IN_create_platoon(env, plt_num, start_position=(1, 1))

                # Get all squads and teams in the platoon
                plt_children = env.get_unit_children(plt_id)

                # Now move each unit to its specified position
                for child_id in plt_children:
                    # Get the string identifier
                    string_id = env.get_unit_property(child_id, 'string_id', '')

                    # Find the matching key in unit_positions
                    matching_key = None
                    for key in unit_positions.keys():
                        if key in string_id:
                            matching_key = key
                            break

                    if matching_key:
                        # Move the unit to its position
                        new_pos = unit_positions[matching_key]
                        env.update_unit_position(child_id, new_pos)

                        # Also move all children (soldiers) of this unit
                        for member_id in env.get_unit_children(child_id):
                            env.update_unit_position(member_id, new_pos)

                        if not use_tqdm or episode == 0:
                            print(f"Positioned {string_id} at {new_pos}")

                return plt_id

            # Create the platoon with custom positions
            platoon_id = US_IN_create_platoon_custom_positions(env, plt_num=1, unit_positions=unit_start_positions)

            # Use consistent agent ID mapping - this is critical for resumed training
            env.agent_ids = env.agent_manager.map_current_units_to_agent_ids(platoon_id)

            # Get observations for consistent agent IDs
            observations = {}
            for agent_id in env.agent_ids:
                unit_id = env.agent_manager.get_current_unit_id(agent_id)
                if unit_id:
                    observations[agent_id] = env._get_observation_for_agent(unit_id)

            if not use_tqdm or episode == 0:
                print(f"Created platoon with {len(env.agent_ids)} consistently mapped agents")

        except Exception as e:
            print(f"Error creating custom platoon: {e}")
            print("Falling back to default platoon creation...")

            # Fallback to standard platoon creation
            from US_Army_PLT_Composition_v2 import US_IN_create_platoon
            default_position = list(unit_start_positions.values())[0] if unit_start_positions else (50, 50)
            platoon_id = US_IN_create_platoon(env, plt_num=1, start_position=default_position)

            # Use consistent agent ID mapping
            env.agent_ids = env.agent_manager.map_current_units_to_agent_ids(platoon_id)

            # Get observations for consistent agent IDs
            observations = {}
            for agent_id in env.agent_ids:
                unit_id = env.agent_manager.get_current_unit_id(agent_id)
                if unit_id:
                    observations[agent_id] = env._get_observation_for_agent(unit_id)

        # Add enemies at specified positions
        from US_Army_PLT_Composition_v2 import US_IN_Role
        if not use_tqdm or episode == 0:
            print(f"Adding {len(enemy_positions)} enemies at specified positions...")

        enemy_ids = []
        for i, pos in enumerate(enemy_positions):
            # Create enemy team
            enemy_id = env.create_unit(
                unit_type=UnitType.INFANTRY_TEAM,
                unit_id_str=f"ENEMY-{i + 1}",
                start_position=pos
            )
            env.update_unit_property(enemy_id, 'force_type', ForceType.ENEMY)
            enemy_ids.append(enemy_id)

            # Add soldiers to the enemy team
            for j in range(3):  # Add 3 soldiers per enemy team
                soldier_role = US_IN_Role.RIFLEMAN
                if j == 0:  # First soldier is team leader
                    soldier_role = US_IN_Role.TEAM_LEADER

                soldier_id = env.create_soldier(
                    role=soldier_role,
                    unit_id_str=f"ENEMY-{i + 1}-{j + 1}",
                    position=pos,
                    is_leader=(j == 0)  # First soldier is leader
                )
                env.update_unit_property(soldier_id, 'force_type', ForceType.ENEMY)
                env.set_unit_hierarchy(soldier_id, enemy_id)

        if not use_tqdm or episode == 0:
            print(f"Added {len(enemy_ids)} enemy teams")

        # Count initial enemies for tracking casualties
        initial_enemy_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    initial_enemy_count += 1

        # Count initial friendlies
        initial_friendly_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    initial_friendly_count += 1

        # Episode variables
        episode_rewards = {agent_id: 0 for agent_id in env.agent_ids}
        episode_length = 0
        done = False
        step_failures = 0

        # Create step iterator with tqdm if available
        step_range = trange(max_steps_per_episode) if use_tqdm else range(max_steps_per_episode)

        # Episode loop
        for step in step_range:
            if done:
                break

            # Update step progress bar if available
            if use_tqdm:
                step_range.set_description(f"Step {step + 1}/{max_steps_per_episode}")

            # Select actions
            try:
                actions = marl_ppo.select_actions(observations)
                if not use_tqdm and step % 10 == 0:  # Print only every 10 steps if no progress bar
                    print(f"Step {step}: Selected actions for {len(actions)} agents")
            except Exception as e:
                print(f"Error selecting actions: {e}")
                # Use default actions as fallback
                actions = {}
                for agent_id in env.agent_ids:
                    unit_id = env.agent_manager.get_current_unit_id(agent_id)
                    if unit_id:
                        unit_pos = env.get_unit_position(unit_id)

                        # Create a default direction toward objective
                        dir_x = objective_location[0] - unit_pos[0]
                        dir_y = objective_location[1] - unit_pos[1]

                        # Normalize direction
                        norm = (dir_x ** 2 + dir_y ** 2) ** 0.5
                        if norm > 0:
                            dir_x /= norm
                            dir_y /= norm

                        actions[agent_id] = {
                            'action_type': 0,  # MOVE
                            'movement_params': {
                                'direction': np.array([dir_x, dir_y]),
                                'distance': [3]
                            },
                            'engagement_params': {
                                'target_pos': np.array(objective_location),
                                'max_rounds': [5],
                                'suppress_only': 0,
                                'adjust_for_fire_rate': 1
                            },
                            'formation': 0
                        }

            # Execute actions
            try:
                next_observations, rewards, dones, truncs, infos = env.step(actions)
                step_failures = 0  # Reset failure counter on success

                # Update observations and accumulate rewards
                observations = next_observations
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward

                # Check termination
                done = all(dones.values()) or all(truncs.values())
                episode_length += 1

                # Update step progress description with rewards if using tqdm
                if use_tqdm:
                    avg_step_reward = sum(rewards.values()) / len(rewards) if rewards else 0
                    step_range.set_postfix(reward=f"{avg_step_reward:.2f}")

            except Exception as e:
                print(f"Error during step execution: {e}")
                step_failures += 1

                if step_failures >= 3:
                    print("Too many consecutive failures, ending episode")
                    break

        # Episode is complete - calculate statistics
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_times.append(episode_duration)

        avg_reward = sum(episode_rewards.values()) / len(episode_rewards) if episode_rewards else 0
        all_rewards.append(avg_reward)
        episode_lengths.append(episode_length)

        # Count casualties
        current_enemy_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    current_enemy_count += 1

        current_friendly_count = 0
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    current_friendly_count += 1

        enemy_casualties = initial_enemy_count - current_enemy_count
        friendly_casualties = initial_friendly_count - current_friendly_count

        enemy_casualties_history.append(enemy_casualties)
        friendly_casualties_history.append(friendly_casualties)

        # Check if objective is secured
        objective_secured = False
        if hasattr(env, '_check_objective_secured'):
            objective_secured = env._check_objective_secured()
            if objective_secured:
                objective_secured_count += 1
                consecutive_objective_secured += 1
                if not objective_reached:
                    print(f"🎯 OBJECTIVE SECURED on episode {episode + 1}! Enemy casualties: {enemy_casualties}, Friendly casualties: {friendly_casualties}")
                objective_reached = True
            else:
                # Reset consecutive counter if objective not secured
                consecutive_objective_secured = 0

        # Update policies with current entropy coefficient
        try:
            actor_loss, critic_loss, entropy_loss = marl_ppo.update(
                clip_param=clip_param,
                value_coef=value_coef,
                entropy_coef=current_entropy_coef,  # Use the annealed entropy coefficient
                ppo_epochs=ppo_epochs,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda
            )
        except Exception as e:
            print(f"Error updating policies: {e}")
            actor_loss, critic_loss, entropy_loss = 0, 0, 0

        # Calculate elapsed time
        elapsed_time = time.time() - training_start_time
        elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))

        # Log results
        if log_interval > 0 and episode % log_interval == 0:
            obj_status = "SECURED" if objective_secured else "NOT SECURED"

            print(
                f"Episode {episode + 1}: Steps={episode_length}, Avg Reward={avg_reward:.2f}, LR={current_lr:.6f}, Ent={current_entropy_coef:.4f}")
            print(
                f"Friendly Casualties: {friendly_casualties}, Enemy Casualties: {enemy_casualties}, Objective: {obj_status}")
            print(f"Losses: Actor={actor_loss:.4f}, Critic={critic_loss:.4f}, Entropy={entropy_loss:.4f}")
            print(f"Episode Time: {episode_duration:.1f}s, Total Training Time: {elapsed_time_str}")

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(
                    f"{episode},{avg_reward},{friendly_casualties},{enemy_casualties},{episode_length},"
                    f"{actor_loss},{critic_loss},{entropy_loss},{elapsed_time},{current_lr},{current_entropy_coef},{obj_status}\n")

            # Log to W&B if enabled
            if use_wandb and wandb is not None:
                wandb.log({
                    "episode": episode,
                    "reward": avg_reward,
                    "friendly_casualties": friendly_casualties,
                    "enemy_casualties": enemy_casualties,
                    "episode_length": episode_length,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy_loss": entropy_loss,
                    "episode_time": episode_duration,
                    "total_training_time": elapsed_time,
                    "learning_rate": current_lr,
                    "entropy_coefficient": current_entropy_coef,
                    "objective_secured": 1 if objective_secured else 0
                })

        # Early stopping check - consider multiple criteria
        # An improvement is defined as:
        # 1. Increase in average reward beyond min_delta, OR
        # 2. More enemy casualties without losing more friendly units, OR
        # 3. Objective secured with minimal friendly casualties
        improvement = (avg_reward > best_avg_reward + min_delta) or \
                      (enemy_casualties > best_enemy_casualties and friendly_casualties <= 1) or \
                      (objective_secured and friendly_casualties <= 1)

        # Special condition: if objective is secured with minimal casualties, consider it a major improvement
        major_improvement = objective_secured and friendly_casualties <= 1 and enemy_casualties >= initial_enemy_count * 0.75

        if improvement:
            if avg_reward > best_avg_reward + min_delta:
                print(f"Episode {episode + 1}: Reward improved: {best_avg_reward:.2f} -> {avg_reward:.2f}")
                best_avg_reward = avg_reward

            if enemy_casualties > best_enemy_casualties:
                print(
                    f"Episode {episode + 1}: Enemy casualties improved: {best_enemy_casualties} -> {enemy_casualties}")
                best_enemy_casualties = enemy_casualties

            episodes_without_improvement = 0

            # Save best model
            best_dir = os.path.join(output_dir, "models", "best")
            try:
                marl_ppo.save_agents(best_dir)
                print(f"New best model saved!")
            except Exception as e:
                print(f"Error saving best model: {e}")

            # For major improvements (objective secured), create a special checkpoint
            if major_improvement:
                objective_dir = os.path.join(output_dir, "models", f"objective_secured_ep{episode}")
                try:
                    marl_ppo.save_agents(objective_dir)
                    print(f"🏆 Objective secured model saved to {objective_dir}!")
                except Exception as e:
                    print(f"Error saving objective secured model: {e}")
        else:
            episodes_without_improvement += 1
            if episodes_without_improvement % 5 == 0 or episodes_without_improvement >= patience // 2:
                print(f"Episode {episode + 1}: No improvement for {episodes_without_improvement} episodes")

        # Save checkpoint at regular intervals regardless of improvement
        if save_interval > 0 and episode % save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, "models", f"checkpoint_{episode}")
            try:
                marl_ppo.save_agents(checkpoint_dir)
                print(f"Saved checkpoint at episode {episode}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        # Early stopping checks
        stop_training = False

        # Check 1: No improvement for patience episodes
        if episodes_without_improvement >= patience:
            print(f"Early stopping triggered: No improvement for {patience} episodes")
            stop_training = True

        # Check 2: Objective consistently reached (if enabled)
        if stop_on_objective_reached and consecutive_objective_secured >= required_objective_secured:
            print(f"Early stopping triggered: Objective consistently secured for {consecutive_objective_secured} consecutive episodes (required: {required_objective_secured})")
            stop_training = True

        if stop_training:
            print(f"Training stopped at episode {episode + 1}")
            print(f"Best reward: {best_avg_reward:.2f}, Best enemy casualties: {best_enemy_casualties}")

            # Log early stopping to wandb
            if use_wandb and wandb is not None:
                wandb.log({
                    "early_stopping": True,
                    "best_reward": best_avg_reward,
                    "best_enemy_casualties": best_enemy_casualties,
                    "stopped_at_episode": episode,
                    "objective_secured_count": objective_secured_count
                })

            break

    # Save final model
    final_dir = os.path.join(output_dir, "models", "final")
    marl_ppo.save_agents(final_dir)
    print("Saved final model")

    # Calculate final training statistics
    total_training_time = time.time() - training_start_time
    avg_episode_time = sum(episode_times) / len(episode_times) if episode_times else 0

    print(f"\nTraining Complete!")
    print(f"Total Training Time: {str(timedelta(seconds=int(total_training_time)))}")
    print(f"Average Episode Time: {avg_episode_time:.2f} seconds")
    print(f"Average Reward: {sum(all_rewards) / len(all_rewards):.2f}")
    print(f"Average Episode Length: {sum(episode_lengths) / len(episode_lengths):.1f} steps")
    print(f"Total Friendly Casualties: {sum(friendly_casualties_history)}")
    print(f"Total Enemy Casualties: {sum(enemy_casualties_history)}")
    print(f"Objective Secured Count: {objective_secured_count}")
    print(f"Best Reward: {best_avg_reward:.2f}")

    # Create plots
    try:
        # Create reward plot
        plt.figure(figsize=(10, 6))
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Training Rewards')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "reward_plot.png"))
        plt.close()

        # Create episode length plot
        plt.figure(figsize=(10, 6))
        plt.plot(episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "episode_length_plot.png"))
        plt.close()

        # Create casualties plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(friendly_casualties_history) + 1), friendly_casualties_history, 'b-',
                 label='Friendly Casualties')
        plt.plot(range(1, len(enemy_casualties_history) + 1), enemy_casualties_history, 'r-', label='Enemy Casualties')
        plt.xlabel('Episode')
        plt.ylabel('Casualties')
        plt.title('Casualties Over Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "casualties_plot.png"))
        plt.close()

        # Create learning rate plot
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates)
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule ({lr_decay_method})')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "learning_rate_plot.png"))
        plt.close()

        # Create entropy coefficient plot
        plt.figure(figsize=(10, 6))
        plt.plot(entropy_coefs)
        plt.xlabel('Episode')
        plt.ylabel('Entropy Coefficient')
        plt.title('Entropy Coefficient Annealing')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "entropy_coefficient_plot.png"))
        plt.close()

        # Create additional plot comparing rewards with casualties and learning rate
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

        # Reward subplot
        ax1.plot(all_rewards, 'b-')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.grid(True)

        # Casualties subplot
        ax2.plot(friendly_casualties_history, 'r-', label='Friendly')
        ax2.plot(enemy_casualties_history, 'g-', label='Enemy')
        ax2.set_ylabel('Casualties')
        ax2.legend()
        ax2.grid(True)

        # Entropy coefficient subplot
        ax3.plot(entropy_coefs, 'm-')
        ax3.set_ylabel('Entropy Coefficient')
        ax3.grid(True)

        # Learning rate subplot
        ax4.plot(learning_rates, 'c-')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Learning Rate')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logs", "combined_training_metrics.png"))
        plt.close()

        # Create plots from log file if enough data
        try:
            data = []
            with open(log_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Skip header
                for row in reader:
                    data.append(
                        [float(x) if i > 0 else int(x) for i, x in enumerate(row) if
                         i < 11])  # First 11 cols are numeric

            if data and len(data) > 5:
                data = np.array(data)
                episodes = data[:, 0]
                rewards = data[:, 1]
                friendly_casualties = data[:, 2]
                enemy_casualties = data[:, 3]
                steps = data[:, 4]
                actor_losses = data[:, 5]
                critic_losses = data[:, 6]
                entropy_losses = data[:, 7]
                entropy_coefs_log = data[:, 10] if data.shape[1] > 10 else None

                # Create losses plot
                plt.figure(figsize=(10, 6))
                plt.plot(episodes, actor_losses, 'g-', label='Actor Loss')
                plt.plot(episodes, critic_losses, 'm-', label='Critic Loss')
                plt.plot(episodes, entropy_losses, 'c-', label='Entropy Loss')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.title('Training Losses')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "logs", "losses_plot.png"))
                plt.close()

                # Create entropy vs entropy loss plot if both available
                if entropy_coefs_log is not None:
                    plt.figure(figsize=(10, 6))
                    fig, ax1 = plt.subplots(figsize=(10, 6))

                    color = 'tab:blue'
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Entropy Loss', color=color)
                    ax1.plot(episodes, entropy_losses, color=color)
                    ax1.tick_params(axis='y', labelcolor=color)

                    ax2 = ax1.twinx()
                    color = 'tab:red'
                    ax2.set_ylabel('Entropy Coefficient', color=color)
                    ax2.plot(episodes, entropy_coefs_log, color=color)
                    ax2.tick_params(axis='y', labelcolor=color)

                    plt.title('Entropy Loss vs Entropy Coefficient')
                    fig.tight_layout()
                    plt.savefig(os.path.join(output_dir, "logs", "entropy_analysis.png"))
                    plt.close()
        except Exception as e:
            print(f"Error creating advanced plots from log file: {e}")

    except Exception as e:
        print(f"Error creating plots: {e}")

    # Finish W&B run if enabled
    if use_wandb and wandb is not None:
        try:
            # Log final summary metrics
            wandb.run.summary["best_reward"] = best_avg_reward
            wandb.run.summary["best_enemy_casualties"] = best_enemy_casualties
            wandb.run.summary["total_training_time"] = total_training_time
            wandb.run.summary["objective_secured_count"] = objective_secured_count
            wandb.run.summary["avg_episode_length"] = sum(episode_lengths) / len(
                episode_lengths) if episode_lengths else 0

            # Log best model if available
            best_model_path = os.path.join(output_dir, "models", "best")
            if os.path.exists(best_model_path):
                wandb.save(best_model_path + "/*", base_path=output_dir)

            wandb.finish()
        except Exception as e:
            print(f"Error finalizing W&B logging: {e}")
            wandb.finish()

    return marl_ppo, (
        all_rewards, episode_lengths, friendly_casualties_history, enemy_casualties_history,
        episode_times, learning_rates, entropy_coefs, best_avg_reward, best_enemy_casualties, objective_secured_count
    )


# Utility functions
def create_default_observation(objective=(15, 15)):
    """Create a default observation when actual observation fails"""
    return {
        'agent_state': {
            'position': np.array([5, 5], dtype=np.int32),
            'health': np.array([100], dtype=np.float32),
            'ammo': np.array([100], dtype=np.int32),
            'suppressed': np.array([0], dtype=np.float32)
        },
        'tactical_info': {
            'formation': np.array([0], dtype=np.int32),
            'orientation': np.array([0], dtype=np.int32),
            'unit_type': np.array([0], dtype=np.int32)
        },
        'friendly_units': np.zeros((10, 2), dtype=np.int32),
        'known_enemies': np.zeros((10, 2), dtype=np.int32),
        'objective': np.array(objective, dtype=np.int32),
        'objective_info': {
            'direction': np.array([0, 0], dtype=np.float32),
            'distance': np.array([50.0], dtype=np.float32)
        }
    }


def count_enemies(env):
    """Count active enemy units"""
    from WarGamingEnvironment_v12 import ForceType

    enemy_count = 0
    for unit_id in env.state_manager.active_units:
        try:
            force_type = env.get_unit_property(unit_id, 'force_type')
            health = env.get_unit_property(unit_id, 'health', 0)

            if force_type == ForceType.ENEMY and health > 0:
                enemy_count += 1
        except:
            pass

    return enemy_count


def count_casualties(env, force_type):
    """Count casualties for a specific force type"""
    casualties = 0
    for unit_id in env.state_manager.active_units:
        try:
            unit_force = env.get_unit_property(unit_id, 'force_type')
            health = env.get_unit_property(unit_id, 'health', 100)

            if unit_force == force_type and health <= 0:
                casualties += 1
        except:
            pass

    return casualties


def create_training_plots(rewards, episode_lengths, log_file, output_dir):
    """Create plots for visualizing training progress"""
    # Create reward plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "logs", "reward_plot.png"))
    plt.close()

    # Create episode length plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "logs", "episode_length_plot.png"))
    plt.close()

    # Create combined plots from log file
    try:
        data = []
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header
            for row in reader:
                data.append([float(x) if i > 0 else int(x) for i, x in enumerate(row)])

        if data:
            data = np.array(data)
            episodes = data[:, 0]
            rewards = data[:, 1]
            friendly_casualties = data[:, 2]
            enemy_casualties = data[:, 3]
            steps = data[:, 4]
            actor_losses = data[:, 5]
            critic_losses = data[:, 6]
            entropy_losses = data[:, 7]

            # Create casualties plot
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, friendly_casualties, 'b-', label='Friendly Casualties')
            plt.plot(episodes, enemy_casualties, 'r-', label='Enemy Casualties')
            plt.xlabel('Episode')
            plt.ylabel('Casualties')
            plt.title('Casualties Over Training')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "logs", "casualties_plot.png"))
            plt.close()

            # Create losses plot
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, actor_losses, 'g-', label='Actor Loss')
            plt.plot(episodes, critic_losses, 'm-', label='Critic Loss')
            plt.plot(episodes, entropy_losses, 'c-', label='Entropy Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "logs", "losses_plot.png"))
            plt.close()

            # Create combined metrics plot
            plt.figure(figsize=(12, 8))
            fig, ax1 = plt.subplots(figsize=(12, 8))

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward', color='tab:blue')
            ax1.plot(episodes, rewards, color='tab:blue', label='Reward')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Casualties', color='tab:red')
            ax2.plot(episodes, friendly_casualties, color='tab:red', linestyle='--', label='Friendly Casualties')
            ax2.plot(episodes, enemy_casualties, color='tab:green', linestyle='--', label='Enemy Casualties')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            fig.tight_layout()
            plt.title('Training Progress - Combined Metrics')
            plt.grid(True)

            # Create combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

            plt.savefig(os.path.join(output_dir, "logs", "combined_metrics_plot.png"))
            plt.close()

    except Exception as e:
        print(f"Error creating plots from log file: {e}")


def visualize_reward_components(env, episode_rewards, output_dir=None):
    """
    Create visualizations of reward components for analysis.

    Args:
        env: The environment instance with reward component tracking
        episode_rewards: Dictionary of total rewards by agent
        output_dir: Directory to save visualization files
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime

    # Create output directory if needed
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./reward_analysis_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Extract reward components if available
    team_components = getattr(env, '_team_reward_components', {})
    agent_components = getattr(env, '_agent_reward_components', {})

    # Get list of all component types
    all_team_components = set()
    for step_data in team_components.values():
        all_team_components.update(step_data.keys())

    all_agent_components = set()
    for agent_data in agent_components.values():
        for component_data in agent_data.values():
            all_agent_components.update(component_data.keys())

    # 1. Create team reward component stacked bar chart
    if team_components:
        plt.figure(figsize=(12, 6))

        # Prepare data for stacking
        steps = sorted(team_components.keys())
        components_by_type = {comp: [team_components.get(step, {}).get(comp, 0) for step in steps]
                              for comp in all_team_components}

        # Create stacked bar chart
        bottom = np.zeros(len(steps))
        for component, values in components_by_type.items():
            plt.bar(steps, values, bottom=bottom, label=component)
            # Only stack positive values on positive and negative on negative
            pos_values = np.maximum(values, 0)
            neg_values = np.minimum(values, 0)
            bottom = bottom + pos_values

        plt.xlabel('Step')
        plt.ylabel('Reward Component Value')
        plt.title('Team Reward Components by Step')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'team_reward_components.png'))
        plt.close()

    # 2. Create agent reward component comparison
    if agent_components and episode_rewards:
        # Get most common agent IDs between components and rewards
        agent_ids = set(episode_rewards.keys())
        for agent_dict in agent_components.values():
            agent_ids.update(agent_dict.keys())

        # Create a subplot for each agent
        num_agents = len(agent_ids)
        fig, axs = plt.subplots(num_agents, 1, figsize=(12, 4 * num_agents), sharex=True)

        # Handle the case with only one agent
        if num_agents == 1:
            axs = [axs]

        for i, agent_id in enumerate(sorted(agent_ids)):
            agent_steps = []
            agent_component_values = {}

            # Collect all steps where this agent has data
            for step, agent_dict in agent_components.items():
                if str(agent_id) in agent_dict:
                    agent_steps.append(step)
                    for comp, value in agent_dict[str(agent_id)].items():
                        if comp not in agent_component_values:
                            agent_component_values[comp] = []
                        agent_component_values[comp].append(value)

            # Create stacked bar chart for this agent
            if agent_steps:
                bottom = np.zeros(len(agent_steps))
                for component, values in agent_component_values.items():
                    axs[i].bar(agent_steps, values, bottom=bottom, label=component, alpha=0.7)
                    # Only stack positive values on positive and negative on negative
                    pos_values = np.maximum(values, 0)
                    bottom = bottom + pos_values

                axs[i].set_ylabel(f'Agent {agent_id}')
                axs[i].grid(True, linestyle='--', alpha=0.5)
                # Add total reward as a text box
                total_reward = episode_rewards.get(agent_id, 'N/A')
                axs[i].text(0.02, 0.95, f'Total Reward: {total_reward:.2f}',
                            transform=axs[i].transAxes, fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.5))

                # Only show legend for the first subplot
                if i == 0:
                    axs[i].legend(loc='upper right')

        plt.xlabel('Step')
        fig.suptitle('Individual Agent Reward Components by Step')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'agent_reward_components.png'))
        plt.close()

    # 3. Create reward distribution pie chart
    if episode_rewards:
        plt.figure(figsize=(10, 10))

        # Sort agents by total reward
        sorted_agents = sorted(episode_rewards.items(), key=lambda x: x[1], reverse=True)
        agent_ids = [f'Agent {a}' for a, _ in sorted_agents]
        reward_values = [r for _, r in sorted_agents]

        # Handle negative rewards in pie chart
        total_positive = sum(max(0, r) for r in reward_values)
        normalized_values = [max(0, r) / total_positive if total_positive > 0 else 0 for r in reward_values]

        # Create pie chart
        plt.pie(normalized_values, labels=agent_ids, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Distribution of Positive Rewards Across Agents')
        plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
        plt.close()

    # 4. Create a summary text file
    with open(os.path.join(output_dir, 'reward_analysis_summary.txt'), 'w') as f:
        f.write("Reward Component Analysis Summary\n")
        f.write("===============================\n\n")

        f.write("Team Components:\n")
        for comp in sorted(all_team_components):
            f.write(f"  - {comp}\n")

        f.write("\nAgent Components:\n")
        for comp in sorted(all_agent_components):
            f.write(f"  - {comp}\n")

        f.write("\nAgent Rewards:\n")
        for agent_id, reward in sorted(episode_rewards.items()):
            f.write(f"  - Agent {agent_id}: {reward:.4f}\n")

        # Calculate team-individual reward ratio
        if team_components and agent_components:
            total_team = sum(sum(comps.values()) for comps in team_components.values())
            total_individual = 0
            for step_data in agent_components.values():
                for agent_data in step_data.values():
                    total_individual += sum(agent_data.values())

            if total_individual != 0:  # Avoid division by zero
                ratio = total_team / total_individual
                f.write(f"\nTeam-to-Individual Reward Ratio: {ratio:.2f}\n")

        f.write("\nGenerated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return output_dir


if __name__ == "__main__":
    import os
    from datetime import datetime

    # Create a main output directory with timestamp
    main_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    main_output_dir = f"./curriculum_training_{main_timestamp}"
    os.makedirs(main_output_dir, exist_ok=True)

    # Define the curriculum stages
    curriculum = [
        # Close Distance Category (30-40 grid units from objective)
        {
            "name": "Stage1_Close_TopRight",
            "objective_location": (90, 25),
            "unit_start_positions": {
                "1SQD": (60, 20),
                "2SQD": (60, 25),
                "3SQD": (60, 30),
                "GTM1": (55, 22),
                "GTM2": (55, 28),
                "JTM1": (50, 22),
                "JTM2": (50, 28)
            },
            "params": {
                "max_steps_per_episode": 1000,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.15,  # Start with high exploration
                "final_entropy_coef": 0.01,  # End with more exploitation
                "entropy_annealing_episodes": 80,  # Gradually reduce over 80 episodes
                "batch_size": 128,
                "initial_learning_rate": 0.0012,  # Slightly higher initial LR for first stage
                "final_learning_rate": 0.0003,
                "clip_param": 0.2,
                "gamma": 0.99,
                "ppo_epochs": 6,
                "patience": 50  # Slightly higher patience for first stage
            }
        },
        {
            "name": "Stage2_Close_MiddleRight",
            "objective_location": (90, 50),
            "unit_start_positions": {
                "1SQD": (60, 45),
                "2SQD": (60, 50),
                "3SQD": (60, 55),
                "GTM1": (55, 47),
                "GTM2": (55, 53),
                "JTM1": (50, 47),
                "JTM2": (50, 53)
            },
            "params": {
                "max_steps_per_episode": 1000,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.12,  # Slightly less exploration in stage 2
                "final_entropy_coef": 0.01,
                "entropy_annealing_episodes": 80,
                "batch_size": 128,
                "initial_learning_rate": 0.001,
                "final_learning_rate": 0.0003,
                "clip_param": 0.2,
                "gamma": 0.99,
                "ppo_epochs": 6,
                "patience": 40
            }
        },
        {
            "name": "Stage3_Close_BottomRight",
            "objective_location": (90, 75),
            "unit_start_positions": {
                "1SQD": (60, 70),
                "2SQD": (60, 75),
                "3SQD": (60, 80),
                "GTM1": (55, 72),
                "GTM2": (55, 78),
                "JTM1": (50, 72),
                "JTM2": (50, 78)
            },
            "params": {
                "max_steps_per_episode": 1000,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.12,
                "final_entropy_coef": 0.01,
                "entropy_annealing_episodes": 80,
                "batch_size": 128,
                "initial_learning_rate": 0.001,
                "final_learning_rate": 0.0003,
                "clip_param": 0.2,
                "gamma": 0.99,
                "ppo_epochs": 6,
                "patience": 40
            }
        },
        # Medium Distance Category (50-60 grid units from objective)
        {
            "name": "Stage4_Medium_TopRight",
            "objective_location": (90, 25),
            "unit_start_positions": {
                "1SQD": (40, 20),
                "2SQD": (40, 25),
                "3SQD": (40, 30),
                "GTM1": (35, 22),
                "GTM2": (35, 28),
                "JTM1": (30, 22),
                "JTM2": (30, 28)
            },
            "params": {
                "max_steps_per_episode": 1500,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.12,
                "final_entropy_coef": 0.008,  # Slightly lower final entropy for more refined policies
                "entropy_annealing_episodes": 100,  # Longer annealing for more complex tasks
                "batch_size": 128,
                "initial_learning_rate": 0.0008,
                "final_learning_rate": 0.0002,
                "clip_param": 0.15,
                "gamma": 0.995,
                "ppo_epochs": 6,
                "patience": 50
            }
        },
        {
            "name": "Stage5_Medium_MiddleRight",
            "objective_location": (90, 50),
            "unit_start_positions": {
                "1SQD": (40, 45),
                "2SQD": (40, 50),
                "3SQD": (40, 55),
                "GTM1": (35, 47),
                "GTM2": (35, 53),
                "JTM1": (30, 47),
                "JTM2": (30, 53)
            },
            "params": {
                "max_steps_per_episode": 1500,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.12,
                "final_entropy_coef": 0.008,
                "entropy_annealing_episodes": 100,
                "batch_size": 128,
                "initial_learning_rate": 0.0008,
                "final_learning_rate": 0.0002,
                "clip_param": 0.15,
                "gamma": 0.995,
                "ppo_epochs": 6,
                "patience": 50
            }
        },
        {
            "name": "Stage6_Medium_BottomRight",
            "objective_location": (90, 75),
            "unit_start_positions": {
                "1SQD": (40, 70),
                "2SQD": (40, 75),
                "3SQD": (40, 80),
                "GTM1": (35, 72),
                "GTM2": (35, 78),
                "JTM1": (30, 72),
                "JTM2": (30, 78)
            },
            "params": {
                "max_steps_per_episode": 1500,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.12,
                "final_entropy_coef": 0.008,
                "entropy_annealing_episodes": 100,
                "batch_size": 128,
                "initial_learning_rate": 0.0008,
                "final_learning_rate": 0.0002,
                "clip_param": 0.15,
                "gamma": 0.995,
                "ppo_epochs": 6,
                "patience": 50
            }
        },
        # Far Distance Category (70-80 grid units from objective)
        {
            "name": "Stage7_Far_TopRight",
            "objective_location": (90, 25),
            "unit_start_positions": {
                "1SQD": (20, 20),
                "2SQD": (20, 25),
                "3SQD": (20, 30),
                "GTM1": (15, 22),
                "GTM2": (15, 28),
                "JTM1": (10, 22),
                "JTM2": (10, 28)
            },
            "params": {
                "max_steps_per_episode": 2000,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.1,  # Less initial exploration as policy is more refined
                "final_entropy_coef": 0.005,  # Lower final value for more exploitation
                "entropy_annealing_episodes": 120,  # Longer annealing for complex tasks
                "batch_size": 128,
                "initial_learning_rate": 0.0007,
                "final_learning_rate": 0.0001,
                "clip_param": 0.1,
                "gamma": 0.997,
                "ppo_epochs": 8,
                "patience": 60,
                "gae_lambda": 0.97
            }
        },
        {
            "name": "Stage8_Far_MiddleRight",
            "objective_location": (90, 50),
            "unit_start_positions": {
                "1SQD": (20, 45),
                "2SQD": (20, 50),
                "3SQD": (20, 55),
                "GTM1": (15, 47),
                "GTM2": (15, 53),
                "JTM1": (10, 47),
                "JTM2": (10, 53)
            },
            "params": {
                "max_steps_per_episode": 2000,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.1,
                "final_entropy_coef": 0.005,
                "entropy_annealing_episodes": 120,
                "batch_size": 128,
                "initial_learning_rate": 0.0007,
                "final_learning_rate": 0.0001,
                "clip_param": 0.1,
                "gamma": 0.997,
                "ppo_epochs": 8,
                "patience": 60,
                "gae_lambda": 0.97
            }
        },
        {
            "name": "Stage9_Far_BottomRight",
            "objective_location": (90, 75),
            "unit_start_positions": {
                "1SQD": (20, 70),
                "2SQD": (20, 75),
                "3SQD": (20, 80),
                "GTM1": (15, 72),
                "GTM2": (15, 78),
                "JTM1": (10, 72),
                "JTM2": (10, 78)
            },
            "params": {
                "max_steps_per_episode": 2000,
                "value_coef": 2.0,
                "initial_entropy_coef": 0.1,
                "final_entropy_coef": 0.005,
                "entropy_annealing_episodes": 120,
                "batch_size": 128,
                "initial_learning_rate": 0.0007,
                "final_learning_rate": 0.0001,
                "clip_param": 0.1,
                "gamma": 0.997,
                "ppo_epochs": 8,
                "patience": 60,
                "gae_lambda": 0.97
            }
        },
    ]

    print(f"=== STARTING CURRICULUM TRAINING ===")
    print(f"Main output directory: {main_output_dir}")
    print(f"Total stages: {len(curriculum)}")

    # Track the best model path for loading in subsequent stages
    previous_best_model_path = None

    # Required progress to advance to next stage (number of times objective must be secured)
    required_objective_secured = 10

    # Run each curriculum stage
    for stage_idx, stage in enumerate(curriculum):
        print(f"\n\n{'=' * 80}")
        print(f"STARTING STAGE {stage_idx + 1}/{len(curriculum)}: {stage['name']}")
        print(f"Objective: {stage['objective_location']}")
        print(f"{'=' * 80}\n")

        # Create stage-specific output directory
        stage_output_dir = os.path.join(main_output_dir, stage['name'])
        os.makedirs(stage_output_dir, exist_ok=True)

        # Extract training parameters
        max_steps = stage['params'].get('max_steps_per_episode', 700)
        value_coef = stage['params'].get('value_coef', 2.0)
        initial_entropy_coef = stage['params'].get('initial_entropy_coef', 0.15)
        final_entropy_coef = stage['params'].get('final_entropy_coef', 0.01)
        entropy_annealing_episodes = stage['params'].get('entropy_annealing_episodes', 100)
        batch_size = stage['params'].get('batch_size', 128)
        initial_lr = stage['params'].get('initial_learning_rate', 0.001)
        final_lr = stage['params'].get('final_learning_rate', 0.0003)
        clip_param = stage['params'].get('clip_param', 0.2)
        gamma = stage['params'].get('gamma', 0.99)
        ppo_epochs = stage['params'].get('ppo_epochs', 6)
        patience = stage['params'].get('patience', 40)
        gae_lambda = stage['params'].get('gae_lambda', 0.95)

        # Common parameters used across all stages
        common_params = {
            "num_episodes": 250,  # Increased max episodes to allow for sufficient learning
            "map_file": "training_map_lvl_1.csv",
            "enemy_positions": [],
            # "enemy_positions": [
            #     # Add a few enemies near the objective
            #     (stage['objective_location'][0] - 5, stage['objective_location'][1]),
            #     (stage['objective_location'][0], stage['objective_location'][1] + 5)
            # ],
            "save_interval": 20,
            "log_interval": 1,
            "use_wandb": False,
            "use_tqdm": True,
            "min_delta": 0.5,
            "lr_decay_method": 'linear'
        }

        # Add stage-specific parameters
        stage_params = {
            "max_steps_per_episode": max_steps,
            "value_coef": value_coef,
            "initial_entropy_coef": initial_entropy_coef,
            "final_entropy_coef": final_entropy_coef,
            "entropy_annealing_episodes": entropy_annealing_episodes,
            "batch_size": batch_size,
            "initial_learning_rate": initial_lr,
            "final_learning_rate": final_lr,
            "clip_param": clip_param,
            "gamma": gamma,
            "ppo_epochs": ppo_epochs,
            "patience": patience,
            "gae_lambda": gae_lambda
        }

        # Combine parameters for the current stage
        training_params = {
            **common_params,
            **stage_params,
        }

        # Special handling for first stage vs. subsequent stages
        if stage_idx == 0:
            # First stage - initialize new models and use specified early stopping
            first_stage_params = training_params.copy()
            # We're now using the patience value from the parameters (50 for first stage)

            marl_ppo, stats = custom_training(
                output_dir=stage_output_dir,
                objective_location=stage['objective_location'],
                unit_start_positions=stage['unit_start_positions'],
                **first_stage_params,
                # Custom callback to implement learning rate warmup
                custom_lr_callback=lambda episode, initial_lr, final_lr:
                initial_lr * 0.5 + (initial_lr * 0.5) * min(1.0, episode / 5.0) if episode < 5 else None
            )
        else:
            # Subsequent stages - load from previous best model
            print(f"Loading models from: {previous_best_model_path}")

            marl_ppo, stats = resume_custom_training(
                output_dir=stage_output_dir,
                objective_location=stage['objective_location'],
                unit_start_positions=stage['unit_start_positions'],
                load_from_dir=previous_best_model_path,
                **training_params,
                # Custom callback to implement learning rate warmup
                custom_lr_callback=lambda episode, initial_lr, final_lr:
                initial_lr * 0.5 + (initial_lr * 0.5) * min(1.0, episode / 5.0) if episode < 5 else None
            )

        # Extract useful statistics
        try:
            all_rewards, episode_lengths, friendly_casualties, enemy_casualties, *rest = stats
            best_reward = max(all_rewards) if all_rewards else 0

            # Get objective secured count if available in stats
            objective_secured_count = 0
            if len(rest) >= 5:  # Check if objective_secured_count is in the returned stats
                objective_secured_count = rest[4]

            print(f"\nStage {stage_idx + 1} ({stage['name']}) completed!")
            print(f"Best reward: {best_reward:.2f}")
            print(
                f"Average episode length: {sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0:.1f}")
            print(f"Total friendly casualties: {sum(friendly_casualties)}")
            print(f"Objective secured count: {objective_secured_count}")

            # Save stage summary
            with open(os.path.join(stage_output_dir, "stage_summary.txt"), "w") as f:
                f.write(f"Stage: {stage['name']}\n")
                f.write(f"Objective: {stage['objective_location']}\n")
                f.write(f"Best reward: {best_reward:.2f}\n")
                f.write(f"Episodes trained: {len(all_rewards)}\n")
                f.write(
                    f"Average episode length: {sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0:.1f}\n")
                f.write(f"Total friendly casualties: {sum(friendly_casualties)}\n")
                f.write(f"Total enemy casualties: {sum(enemy_casualties)}\n")
                f.write(f"Objective secured count: {objective_secured_count}\n")

                # Add detailed parameters used
                f.write("\nParameters:\n")
                for key, value in stage_params.items():
                    f.write(f"  {key}: {value}\n")

            # Check if stage completion criteria met
            stage_completed = objective_secured_count >= required_objective_secured

            if stage_completed:
                print(
                    f"🎯 STAGE COMPLETED! Secured objective {objective_secured_count} times (required: {required_objective_secured})")
            else:
                print(
                    f"⚠️ Stage not completed. Objective secured {objective_secured_count}/{required_objective_secured} times.")

                # If we're halfway through the curriculum and still not completing stages,
                # continue to next stage anyway to prevent getting stuck
                if stage_idx >= len(curriculum) // 2:
                    print(
                        "Proceeding to next stage despite not meeting completion criteria (we're past the halfway point).")
                else:
                    # Option to repeat the stage (could be implemented with user input)
                    repeat_stage = False
                    if repeat_stage:
                        stage_idx -= 1  # Repeat the current stage
                        print("Repeating this stage...")
                        continue

        except Exception as e:
            print(f"Error extracting stage statistics: {e}")
            import traceback

            traceback.print_exc()

        # Update best model path for next stage
        previous_best_model_path = os.path.join(stage_output_dir, "models", "best")

    print("\n\n=== CURRICULUM TRAINING COMPLETE ===")
    print(f"All training data saved to: {main_output_dir}")

    # Create final summary file
    try:
        with open(os.path.join(main_output_dir, "curriculum_summary.txt"), "w") as f:
            f.write("CURRICULUM TRAINING SUMMARY\n")
            f.write("===========================\n\n")
            f.write(f"Completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total stages: {len(curriculum)}\n\n")

            for stage_idx, stage in enumerate(curriculum):
                stage_dir = os.path.join(main_output_dir, stage['name'])
                summary_file = os.path.join(stage_dir, "stage_summary.txt")

                f.write(f"Stage {stage_idx + 1}: {stage['name']}\n")
                f.write(f"  Objective: {stage['objective_location']}\n")

                # Try to read the stage summary if it exists
                if os.path.exists(summary_file):
                    with open(summary_file, "r") as stage_f:
                        lines = stage_f.readlines()
                        # Extract and include key metrics
                        for line in lines:
                            if any(key in line for key in
                                   ["Best reward:", "Objective secured count:", "Total friendly casualties:"]):
                                f.write(f"  {line.strip()}\n")

                f.write("\n")
    except Exception as e:
        print(f"Error creating curriculum summary: {e}")
