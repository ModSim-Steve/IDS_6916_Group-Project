"""
MARL PPO Implementation

This module provides a streamlined version of MARL PPO that works directly with
the existing MARLMilitaryEnvironment.
"""

import os
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

        # Movement parameters head (continuous)
        self.movement_dir_head = nn.Linear(128, 2)  # 2D direction vector
        self.movement_dist_head = nn.Linear(128, 1)  # Scalar distance

        # Engagement parameters head (continuous)
        self.target_pos_head = nn.Linear(128, 2)  # 2D target position
        self.max_rounds_head = nn.Linear(128, 1)  # Scalar rounds

        # Binary parameters (suppress_only, adjust_fire_rate)
        self.suppress_only_head = nn.Linear(128, 2)  # Binary choice
        self.adjust_fire_rate_head = nn.Linear(128, 2)  # Binary choice

        # Formation head (discrete)
        self.formation_head = nn.Linear(128, 8)  # 8 formation options

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

        # Action type logits
        action_type_logits = self.action_type_head(combined_features)

        # Movement parameters
        movement_direction = torch.tanh(self.movement_dir_head(combined_features))  # -1 to 1
        movement_distance = F.softplus(self.movement_dist_head(combined_features))  # Positive value

        # Engagement parameters
        target_pos = torch.sigmoid(self.target_pos_head(combined_features)) * 100  # Scale to map size
        max_rounds = F.softplus(self.max_rounds_head(combined_features))

        # Binary choices
        suppress_only_logits = self.suppress_only_head(combined_features)
        adjust_fire_rate_logits = self.adjust_fire_rate_head(combined_features)

        # Formation parameters
        formation_logits = self.formation_head(combined_features)

        return {
            'action_type': action_type_logits,
            'movement_params': {
                'direction': movement_direction,
                'distance': movement_distance
            },
            'engagement_params': {
                'target_pos': target_pos,
                'max_rounds': max_rounds,
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
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Value output
        self.value_head = nn.Linear(64, 1)

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
        # Main storage dictionaries
        self.states = {}
        self.actions = {}
        self.logprobs = {}
        self.rewards = {}
        self.is_terminals = {}
        self.values = {}

        # Tracking attributes
        self.initialized_agents = set()
        self.current_episode_agents = set()
        self.last_updated = {}  # Track last timestep each agent's memory was updated

    def clear_memory(self):
        """Clear all memory contents but maintain initialization tracking."""
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        self.last_updated.clear()
        self.current_episode_agents.clear()
        # Note: We intentionally keep initialized_agents to track which agents
        # have been seen before, for debugging purposes

    def initialize_agent(self, agent_id):
        """
        Safely initialize memory structures for an agent with validation.

        Args:
            agent_id: Consistent ID of the agent to initialize
        """
        # Initialize states if not already present
        if agent_id not in self.states:
            self.states[agent_id] = []

        # Initialize actions if not already present
        if agent_id not in self.actions:
            self.actions[agent_id] = []

        # Initialize logprobs if not already present
        if agent_id not in self.logprobs:
            self.logprobs[agent_id] = []

        # Initialize rewards if not already present - critical for update
        if agent_id not in self.rewards:
            self.rewards[agent_id] = []

        # Initialize is_terminals if not already present
        if agent_id not in self.is_terminals:
            self.is_terminals[agent_id] = []

        # Initialize values if not already present
        if agent_id not in self.values:
            self.values[agent_id] = []

        # Mark agent as initialized and active for current episode
        self.initialized_agents.add(agent_id)
        self.current_episode_agents.add(agent_id)
        self.last_updated[agent_id] = 0  # Initialize timestep counter

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
        """
        Select action and return action, logprob and state value.
        This version separates action selection from memory storage,
        allowing the MARL controller to handle memory consistently.

        Args:
            observation: Agent observation

        Returns:
            Tuple of (action_dict, combined_logprob, state_value)
        """
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

            # Sample suppress only (for engagement)
            suppress_only_logits = action_outputs['engagement_params']['suppress_only']
            suppress_only_probs = F.softmax(suppress_only_logits, dim=1)
            suppress_only_dist = Categorical(suppress_only_probs)
            suppress_only = suppress_only_dist.sample()
            suppress_only_logprob = suppress_only_dist.log_prob(suppress_only)

            # Sample adjust fire rate (for engagement)
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

            # Combine logprobs
            combined_logprob = action_type_logprob + suppress_only_logprob + adjust_fire_rate_logprob + formation_logprob

            # Get continuous action parameters
            movement_direction = action_outputs['movement_params']['direction'].squeeze(0).cpu().numpy()
            movement_distance = action_outputs['movement_params']['distance'].item()
            target_pos = action_outputs['engagement_params']['target_pos'].squeeze(0).cpu().numpy()
            max_rounds = action_outputs['engagement_params']['max_rounds'].item()

            # Create action dictionary matching environment expectations
            action = {
                'action_type': action_type.item(),
                'movement_params': {
                    'direction': movement_direction,
                    'distance': [max(1, int(movement_distance))]
                },
                'engagement_params': {
                    'target_pos': target_pos,
                    'max_rounds': [max(1, int(max_rounds))],
                    'suppress_only': suppress_only.item(),
                    'adjust_for_fire_rate': adjust_fire_rate.item()
                },
                'formation': formation.item()
            }

            return action, combined_logprob.item(), state_value

    def update(self, agent_data, clip_param, value_coef, entropy_coef, ppo_epochs, batch_size=64, gamma=0.99,
               gae_lambda=0.95):
        """
        Update policy using PPO algorithm with enhanced agent data handling.
        This version takes pre-processed agent data from the memory system.

        Args:
            agent_data: Dictionary with agent data from memory
            clip_param: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy coefficient
            ppo_epochs: Number of PPO epochs
            batch_size: Batch size
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

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

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        actor_losses = []
        critic_losses = []
        entropy_losses = []

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
                    batch_returns = returns[batch_idx]
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

                    # Calculate log probabilities for action components
                    batch_action_types = torch.LongTensor([a['action_type'] for a in batch_actions]).to(device)
                    batch_suppress_only = torch.LongTensor([a['engagement_params']['suppress_only']
                                                            for a in batch_actions]).to(device)
                    batch_adjust_fire_rate = torch.LongTensor([a['engagement_params']['adjust_for_fire_rate']
                                                               for a in batch_actions]).to(device)
                    batch_formations = torch.LongTensor([a['formation'] for a in batch_actions]).to(device)

                    # Get distributions
                    action_type_probs = F.softmax(actor_outputs['action_type'], dim=1)
                    action_type_dist = Categorical(action_type_probs)

                    suppress_only_probs = F.softmax(actor_outputs['engagement_params']['suppress_only'], dim=1)
                    suppress_only_dist = Categorical(suppress_only_probs)

                    adjust_fire_rate_probs = F.softmax(actor_outputs['engagement_params']['adjust_fire_rate'], dim=1)
                    adjust_fire_rate_dist = Categorical(adjust_fire_rate_probs)

                    formation_probs = F.softmax(actor_outputs['formation'], dim=1)
                    formation_dist = Categorical(formation_probs)

                    # Calculate log probabilities
                    action_type_logprobs = action_type_dist.log_prob(batch_action_types)
                    suppress_only_logprobs = suppress_only_dist.log_prob(batch_suppress_only)
                    adjust_fire_rate_logprobs = adjust_fire_rate_dist.log_prob(batch_adjust_fire_rate)
                    formation_logprobs = formation_dist.log_prob(batch_formations)

                    # Combined log probability
                    new_logprobs = action_type_logprobs + suppress_only_logprobs + adjust_fire_rate_logprobs + formation_logprobs

                    # Calculate entropy (encourage exploration)
                    entropy = (action_type_dist.entropy() + suppress_only_dist.entropy() +
                               adjust_fire_rate_dist.entropy() + formation_dist.entropy()).mean()

                    # Calculate ratios for PPO
                    ratios = torch.exp(new_logprobs - batch_logprobs)

                    # Calculate surrogate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * batch_advantages

                    # Calculate losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(state_values, batch_returns)
                    entropy_loss = -entropy  # Negative because we want to maximize entropy

                    # Combined loss
                    loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

                    # Perform backpropagation
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()

                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    # Record losses
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropy_losses.append(entropy_loss.item())

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
        Initialize memory and policies for all agents at episode start.
        This is a critical improvement that ensures all agents have
        properly initialized memory structures before any actions.

        Args:
            agent_ids: List of agent IDs for the current episode
        """
        # Reset timestep counter
        self.timestep = 0

        # Initialize memory for all agents
        for agent_id in agent_ids:
            self.memory.initialize_agent(agent_id)

        # Create policies for any new agents
        for agent_id in agent_ids:
            if agent_id not in self.agent_policies:
                self.add_agent(agent_id)

        return len(agent_ids)

    def select_actions(self, observations):
        """
        Select actions for all agents with improved memory tracking.
        Updated to use select_action_with_logprob instead of select_action.

        Args:
            observations: Dictionary mapping agent IDs to observations

        Returns:
            Dictionary mapping agent IDs to actions
        """
        # Initialize memory at episode start if not already done
        # This handles cases where initialize_episode wasn't explicitly called
        if self.timestep == 0:
            self.initialize_episode(list(observations.keys()))

        actions = {}

        for agent_id, observation in observations.items():
            try:
                # Get agent policy or create if needed
                if agent_id not in self.agent_policies:
                    self.add_agent(agent_id)

                # Updated to use select_action_with_logprob instead of select_action
                action, logprob, state_value = self.agent_policies[agent_id].select_action_with_logprob(observation)

                # Store in memory
                if agent_id not in self.memory.states:
                    self.memory.initialize_agent(agent_id)

                self.memory.states[agent_id].append(observation)
                self.memory.actions[agent_id].append(action)
                self.memory.logprobs[agent_id].append(logprob)
                self.memory.values[agent_id].append(state_value)

                # Update last updated timestamp
                self.memory.last_updated[agent_id] = self.timestep

                # Add action to result dictionary
                actions[agent_id] = action
            except Exception as e:
                print(f"Error selecting action for agent {agent_id}: {e}")
                import traceback
                traceback.print_exc()

                # Use default action as fallback
                actions[agent_id] = self._default_action()

        # Increment timestep
        self.timestep += 1

        return actions

    def store_rewards_and_terminals(self, rewards, dones, truncs):
        """
        Explicitly store rewards and terminal flags for all agents.
        This should be called from the environment's step function.

        Args:
            rewards: Dictionary mapping agent IDs to rewards
            dones: Dictionary mapping agent IDs to done flags
            truncs: Dictionary mapping agent IDs to truncation flags
        """
        current_timestep = self.timestep - 1  # -1 because timestep was already incremented

        # Store rewards and terminal flags
        for agent_id in self.memory.current_episode_agents:
            # Store reward (0 if not provided)
            reward = rewards.get(agent_id, 0.0)
            if agent_id in self.memory.rewards:
                self.memory.rewards[agent_id].append(reward)
                self.memory.last_updated[agent_id] = current_timestep

            # Store terminal flag
            is_terminal = dones.get(agent_id, False) or truncs.get(agent_id, False)
            if agent_id in self.memory.is_terminals:
                self.memory.is_terminals[agent_id].append(is_terminal)
                self.memory.last_updated[agent_id] = current_timestep

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

    def update(self, clip_param=0.2, value_coef=0.5, entropy_coef=0.01, ppo_epochs=5, batch_size=64, gamma=0.99,
               gae_lambda=0.95):
        """
        Update policies for all agents with improved memory handling.
        Fixed to use get_agent_data_for_update instead of passing memory directly.

        Args:
            clip_param: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy loss coefficient
            ppo_epochs: Number of PPO update epochs
            batch_size: Batch size for updates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (avg_actor_loss, avg_critic_loss, avg_entropy_loss)
        """
        # Check memory consistency before update
        consistency = self.check_memory_consistency()
        inconsistent_agents = [agent_id for agent_id, status in consistency.items()
                               if not status.get("consistent", True)]

        if inconsistent_agents:
            print(f"WARNING: Found {len(inconsistent_agents)} agents with inconsistent memory. Attempting to fix...")
            fix_results = self.fix_memory_inconsistencies()
            print(f"Fix results: {fix_results}")

            # Check consistency again
            consistency = self.check_memory_consistency()
            inconsistent_agents = [agent_id for agent_id, status in consistency.items()
                                   if not status.get("consistent", True)]

            if inconsistent_agents:
                print(f"WARNING: {len(inconsistent_agents)} agents still have inconsistent memory after fix attempt.")
                for agent_id in inconsistent_agents:
                    print(f"  Agent {agent_id}: {consistency[agent_id]}")

        avg_actor_loss = 0
        avg_critic_loss = 0
        avg_entropy_loss = 0
        active_agents = 0

        for agent_id in self.memory.current_episode_agents:
            # Skip agents with inconsistent memory
            if agent_id in inconsistent_agents:
                print(f"Skipping update for agent {agent_id} due to inconsistent memory.")
                continue

            # Get agent policy
            if agent_id not in self.agent_policies:
                print(f"No policy found for agent {agent_id}, skipping update.")
                continue

            policy = self.agent_policies[agent_id]

            # Get agent data from memory
            agent_data = self.memory.get_agent_data_for_update(agent_id)
            if agent_data is None:
                print(f"Agent {agent_id} has no valid data in memory, skipping update.")
                continue

            try:
                # Update policy with agent data instead of passing memory
                actor_loss, critic_loss, entropy_loss = policy.update(
                    agent_data=agent_data,  # Pass prepared agent data instead of memory
                    clip_param=clip_param,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    ppo_epochs=ppo_epochs,
                    batch_size=min(batch_size, len(agent_data["states"])),
                    gamma=gamma,
                    gae_lambda=gae_lambda
                )

                # Accumulate losses
                avg_actor_loss += actor_loss
                avg_critic_loss += critic_loss
                avg_entropy_loss += entropy_loss
                active_agents += 1

                print(f"Updated policy for agent {agent_id}: actor_loss={actor_loss:.4f}, "
                      f"critic_loss={critic_loss:.4f}, entropy_loss={entropy_loss:.4f}")

            except Exception as e:
                print(f"Error updating policy for agent {agent_id}: {e}")
                import traceback
                traceback.print_exc()

        # Clear memory after update
        self.memory.clear_memory()
        self.timestep = 0

        # Calculate average losses
        if active_agents > 0:
            avg_actor_loss /= active_agents
            avg_critic_loss /= active_agents
            avg_entropy_loss /= active_agents

            print(f"Average losses for {active_agents} agents: "
                  f"actor={avg_actor_loss:.4f}, critic={avg_critic_loss:.4f}, entropy={avg_entropy_loss:.4f}")

        return avg_actor_loss, avg_critic_loss, avg_entropy_loss

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


# Training function
def train_ppo(env, num_episodes=100, max_steps_per_episode=100,
              save_interval=10, log_interval=1, output_dir="./training_output",
              patience=20, min_delta=0.01):  # Add early stopping parameters
    """
    Train agents using the simplified PPO implementation directly with the complex environment.

    Args:
        env: The military environment
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        save_interval: Save model every N episodes
        log_interval: Log progress every N episodes
        output_dir: Directory to save models and logs
        patience: Number of episodes to wait for improvement before stopping
        min_delta: Minimum improvement required to reset patience counter

    Returns:
        The trained MARL PPO instance and training statistics
    """

    from WarGamingEnvironment_v10 import ForceType

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, "logs", "training_log.csv")
    with open(log_file, 'w') as f:
        f.write("episode,avg_reward,friendly_casualties,enemy_casualties,steps,actor_loss,critic_loss,entropy_loss\n")

    # Initialize PPO
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=3e-4)

    # Training statistics
    all_rewards = []
    episode_lengths = []

    # Early stopping tracking
    best_avg_reward = float('-inf')
    episodes_without_improvement = 0

    # Training loop
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")

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
                        'objective': (15, 15)
                    }
                }

                observations, _ = env.reset(options=options)

                # Verify agents exist
                if not env.agent_ids:
                    print("No agents found after reset, creating platoon manually...")
                    from US_Army_PLT_Composition_vTest import US_IN_create_platoon

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

        # Add enemies if needed
        try:
            # Check for existing enemies
            from WarGamingEnvironment_v10 import ForceType

            enemy_count = len([uid for uid in env.state_manager.active_units
                               if env.get_unit_property(uid, 'force_type') == ForceType.ENEMY])

            if enemy_count == 0:
                print("No enemies found, adding enemies...")
                from WarGamingEnvironment_v10 import UnitType

                # Add enemies at objective
                for i in range(2):
                    pos = (15 + i, 15 + i)
                    enemy_id = env.create_unit(
                        unit_type=UnitType.INFANTRY_TEAM,
                        unit_id_str=f"ENEMY-{i + 1}",
                        start_position=pos
                    )
                    env.update_unit_property(enemy_id, 'force_type', ForceType.ENEMY)

                    # Add soldier to enemy team
                    from US_Army_PLT_Composition_vTest import US_IN_Role
                    soldier_id = env.create_soldier(
                        role=US_IN_Role.RIFLEMAN,
                        unit_id_str=f"ENEMY-{i + 1}-RFLM",
                        position=pos,
                        is_leader=False
                    )
                    env.update_unit_property(soldier_id, 'force_type', ForceType.ENEMY)
                    env.set_unit_hierarchy(soldier_id, enemy_id)

                print(f"Added {2} enemies")
        except Exception as e:
            print(f"Error adding enemies: {e}")

        # Ensure observations are valid
        if not observations:
            observations = {}
            for agent_id in env.agent_ids:
                try:
                    observations[agent_id] = env._get_observation_for_agent(agent_id)
                except Exception as e:
                    print(f"Error getting observation for agent {agent_id}: {e}")
                    # Create a default observation
                    observations[agent_id] = create_default_observation(env.objective)

        # Episode variables
        episode_rewards = {agent_id: 0 for agent_id in env.agent_ids}
        episode_length = 0
        done = False
        step_failures = 0

        # Initial enemy count for tracking casualties
        initial_enemy_count = count_enemies(env)

        # Episode loop
        while not done and episode_length < max_steps_per_episode:
            # Select actions
            try:
                actions = marl_ppo.select_actions(observations)
            except Exception as e:
                print(f"Error selecting actions: {e}")
                # Use default actions
                actions = {}
                for agent_id in env.agent_ids:
                    actions[agent_id] = {
                        'action_type': 0,  # MOVE
                        'movement_params': {
                            'direction': np.array([1.0, 0.0]),  # Move east
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

                    # Store terminal flags and rewards in memory if agent exists in memory
                    if agent_id in marl_ppo.memory.states:
                        is_terminal = dones.get(agent_id, False) or truncs.get(agent_id, False)
                        marl_ppo.memory.is_terminals[agent_id].append(is_terminal)
                        marl_ppo.memory.rewards[agent_id].append(reward)

                # Update observations
                observations = next_observations

                # Check if done
                done = all(dones.values()) or all(truncs.values())
                episode_length += 1

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
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards) if episode_rewards else 0
        all_rewards.append(avg_reward)
        episode_lengths.append(episode_length)

        # Count casualties
        friendly_casualties = count_casualties(env, ForceType.FRIENDLY)
        current_enemy_count = count_enemies(env)
        enemy_casualties = initial_enemy_count - current_enemy_count

        # Update policies
        try:
            actor_loss, critic_loss, entropy_loss = marl_ppo.update(
                clip_param=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                ppo_epochs=5,  # Fewer epochs for stability
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95
            )
        except Exception as e:
            print(f"Error during policy update: {e}")
            actor_loss, critic_loss, entropy_loss = 0, 0, 0

        # Log results
        if log_interval > 0 and episode % log_interval == 0:
            print(f"Episode {episode + 1}: Steps={episode_length}, Avg Reward={avg_reward:.2f}")
            print(f"Friendly Casualties: {friendly_casualties}, Enemy Casualties: {enemy_casualties}")
            print(f"Losses: Actor={actor_loss:.4f}, Critic={critic_loss:.4f}, Entropy={entropy_loss:.4f}")

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(
                    f"{episode},{avg_reward},{friendly_casualties},{enemy_casualties},{episode_length},{actor_loss},{critic_loss},{entropy_loss}\n")

        # Save checkpoint
        if save_interval > 0 and episode > 0 and episode % save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, "models", f"checkpoint_{episode}")
            try:
                marl_ppo.save_agents(checkpoint_dir)
                print(f"Saved checkpoint at episode {episode}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        # Early stopping check
        # Consider both reward and enemy casualties for determining improvement
        improvement = (avg_reward > best_avg_reward + min_delta) or (enemy_casualties > 0 and friendly_casualties == 0)

        if improvement:
            best_avg_reward = max(best_avg_reward, avg_reward)
            episodes_without_improvement = 0

            # Save best model
            best_dir = os.path.join(output_dir, "models", "best")
            try:
                marl_ppo.save_agents(best_dir)
                print(f"New best model saved with reward: {best_avg_reward:.2f}")
            except Exception as e:
                print(f"Error saving best model: {e}")
        else:
            episodes_without_improvement += 1
            print(f"Episode {episode + 1}: No improvement for {episodes_without_improvement} episodes")

        if episodes_without_improvement >= patience:
            print(f"Early stopping triggered after {episode + 1} episodes")
            break

    # Save final model
    try:
        final_dir = os.path.join(output_dir, "models", "final")
        marl_ppo.save_agents(final_dir)
        print("Saved final model")
    except Exception as e:
        print(f"Error saving final model: {e}")

    # Create reward plot
    try:
        create_training_plots(all_rewards, episode_lengths, log_file, output_dir)
    except Exception as e:
        print(f"Error creating plots: {e}")

    return marl_ppo, (all_rewards, episode_lengths)


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
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=64,
        # Early stopping parameters
        patience=40,
        min_delta=0.008):
    """
    Run MARL PPO training with custom map, objectives and unit placements.
    Enhanced with consistent agent IDs across episodes and early stopping.

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
        learning_rate: Learning rate for PPO optimizer
        gamma: Discount factor for future rewards
        gae_lambda: GAE parameter for advantage estimation
        clip_param: Clipping parameter for PPO update
        value_coef: Value function loss coefficient
        entropy_coef: Entropy coefficient for policy exploration
        ppo_epochs: Number of epochs for PPO update
        batch_size: Batch size for PPO update
        patience: Number of episodes to wait for improvement before stopping
        min_delta: Minimum improvement required to reset patience counter

    Returns:
        Trained MARL PPO instance and training statistics
    """
    import time
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
                           "learning_rate": learning_rate,
                           "gamma": gamma,
                           "gae_lambda": gae_lambda,
                           "clip_param": clip_param,
                           "value_coef": value_coef,
                           "entropy_coef": entropy_coef,
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
            "episode,avg_reward,friendly_casualties,enemy_casualties,steps,actor_loss,critic_loss,entropy_loss,elapsed_time\n")

    # Initialize environment with custom map size
    from WarGamingEnvironment_v10 import EnvironmentConfig, MARLMilitaryEnvironment, ForceType, UnitType

    # Estimate map size based on objective and unit positions
    all_positions = [objective_location] + list(unit_start_positions.values()) + enemy_positions
    max_x = max(pos[0] for pos in all_positions) + 20  # Add margin
    max_y = max(pos[1] for pos in all_positions) + 20  # Add margin

    # Create environment config
    config = EnvironmentConfig(width=max_x, height=max_y, debug_level=0)
    env = MARLMilitaryEnvironment(config, objective_position=objective_location)

    # Initialize PPO with specified parameters
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=learning_rate)

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

        # Update tqdm description if available
        if use_tqdm:
            episode_range.set_description(f"Episode {episode + 1}/{num_episodes}")
        else:
            print(f"\nStarting Episode {episode + 1}/{num_episodes}")

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
                from US_Army_PLT_Composition_vTest import US_IN_create_platoon

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
            from US_Army_PLT_Composition_vTest import US_IN_create_platoon
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
        from US_Army_PLT_Composition_vTest import US_IN_Role
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

        # Update policies
        try:
            actor_loss, critic_loss, entropy_loss = marl_ppo.update(
                clip_param=clip_param,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
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
            print(f"Episode {episode + 1}: Steps={episode_length}, Avg Reward={avg_reward:.2f}")
            print(f"Friendly Casualties: {friendly_casualties}, Enemy Casualties: {enemy_casualties}")
            print(f"Losses: Actor={actor_loss:.4f}, Critic={critic_loss:.4f}, Entropy={entropy_loss:.4f}")
            print(f"Episode Time: {episode_duration:.1f}s, Total Training Time: {elapsed_time_str}")

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(
                    f"{episode},{avg_reward},{friendly_casualties},{enemy_casualties},{episode_length},{actor_loss},{critic_loss},{entropy_loss},{elapsed_time}\n")

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
                    "total_training_time": elapsed_time
                })

        # Early stopping check - consider both reward and performance metrics
        # An improvement is defined as either:
        # 1. Increase in average reward beyond min_delta, OR
        # 2. More enemy casualties without losing more friendly units
        improvement = (avg_reward > best_avg_reward + min_delta) or \
                      (enemy_casualties > best_enemy_casualties and friendly_casualties <= 1)

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
                best_model_saved = True
            except Exception as e:
                print(f"Error saving best model: {e}")
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

        # Early stopping check
        if episodes_without_improvement >= patience:
            print(f"Early stopping triggered after {episode + 1} episodes")
            print(f"Best reward: {best_avg_reward:.2f}, Best enemy casualties: {best_enemy_casualties}")

            # Log early stopping to wandb
            if use_wandb and wandb is not None:
                wandb.log({
                    "early_stopping": True,
                    "best_reward": best_avg_reward,
                    "best_enemy_casualties": best_enemy_casualties,
                    "stopped_at_episode": episode
                })

            break

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
        create_training_plots(all_rewards, episode_lengths, log_file, output_dir)

        # Create additional plots
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
        plt.axhline(y=best_enemy_casualties, color='g', linestyle='--',
                    label=f'Best: {best_enemy_casualties}')
        plt.xlabel('Episode')
        plt.ylabel('Enemy Casualties')
        plt.title('Enemy Casualties Progress')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logs", "early_stopping_metrics.png"))
        plt.close()

    except Exception as e:
        print(f"Error creating plots: {e}")

    # Finish W&B run if enabled
    if use_wandb:
        if wandb is not None:  # Split into separate conditionals
            wandb.finish()

    return marl_ppo, (
        all_rewards, episode_lengths, friendly_casualties_history, enemy_casualties_history, episode_times,
        best_avg_reward, best_enemy_casualties  # Include best metrics in return value
    )


def resume_custom_training(
        num_episodes=1000,
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

        # PPO hyperparameters
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=64
):
    """
    Resume MARL PPO training from a previously saved model.

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
        learning_rate: Learning rate for PPO optimizer
        gamma: Discount factor for future rewards
        gae_lambda: GAE parameter for advantage estimation
        clip_param: Clipping parameter for PPO update
        value_coef: Value function loss coefficient
        entropy_coef: Entropy coefficient for policy exploration
        ppo_epochs: Number of epochs for PPO update
        batch_size: Batch size for PPO update

    Returns:
        Trained MARL PPO instance and training statistics
    """
    from datetime import datetime, timedelta

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
        output_dir = f"./custom_training_output_{timestamp}"

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
                           "learning_rate": learning_rate,
                           "gamma": gamma,
                           "gae_lambda": gae_lambda,
                           "clip_param": clip_param,
                           "value_coef": value_coef,
                           "entropy_coef": entropy_coef,
                           "ppo_epochs": ppo_epochs,
                           "batch_size": batch_size,
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
            "episode,avg_reward,friendly_casualties,enemy_casualties,steps,actor_loss,critic_loss,entropy_loss,elapsed_time\n")

    # Initialize environment with custom map size
    from WarGamingEnvironment_v10 import EnvironmentConfig, MARLMilitaryEnvironment, ForceType, UnitType

    # Estimate map size based on objective and unit positions
    all_positions = [objective_location] + list(unit_start_positions.values()) + enemy_positions
    max_x = max(pos[0] for pos in all_positions) + 20  # Add margin
    max_y = max(pos[1] for pos in all_positions) + 20  # Add margin

    # Create environment config
    config = EnvironmentConfig(width=max_x, height=max_y, debug_level=0)
    env = MARLMilitaryEnvironment(config, objective_position=objective_location)

    # Initialize PPO with specified parameters
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=learning_rate)

    # Connect environment and PPO for reward handling
    setattr(env, 'marl_algorithm', marl_ppo)

    # Initialize agent role mapping to ensure consistent IDs across episodes
    env.agent_manager.initialize_agent_role_mapping()

    # Load previously trained agents
    print(f"Loading pre-trained agents from {load_from_dir}...")
    agents_loaded = marl_ppo.load_agents(load_from_dir)

    # Count how many agents were loaded
    agent_count = len(marl_ppo.agent_policies)
    print(f"Loaded {agent_count} pre-trained agents")

    # Print info about loaded agents
    for agent_id, policy in marl_ppo.agent_policies.items():
        print(f"Agent {agent_id}: Update count = {policy.update_count}")

    # Training statistics
    all_rewards = []
    episode_lengths = []
    friendly_casualties_history = []
    enemy_casualties_history = []
    episode_times = []
    training_start_time = time.time()

    # Log start time
    print(f"Resumed training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create episode iterator with tqdm if available
    episode_range = trange(num_episodes) if use_tqdm else range(num_episodes)

    # Training loop
    for episode in episode_range:
        episode_start_time = time.time()

        # Update tqdm description if available
        if use_tqdm:
            episode_range.set_description(f"Episode {episode + 1}/{num_episodes}")
        else:
            print(f"\nStarting Episode {episode + 1}/{num_episodes}")

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
                from US_Army_PLT_Composition_vTest import US_IN_create_platoon

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
            from US_Army_PLT_Composition_vTest import US_IN_create_platoon
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
        from US_Army_PLT_Composition_vTest import US_IN_Role
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

        # Update policies
        try:
            actor_loss, critic_loss, entropy_loss = marl_ppo.update(
                clip_param=clip_param,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
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
            print(f"Episode {episode + 1}: Steps={episode_length}, Avg Reward={avg_reward:.2f}")
            print(f"Friendly Casualties: {friendly_casualties}, Enemy Casualties: {enemy_casualties}")
            print(f"Losses: Actor={actor_loss:.4f}, Critic={critic_loss:.4f}, Entropy={entropy_loss:.4f}")
            print(f"Episode Time: {episode_duration:.1f}s, Total Training Time: {elapsed_time_str}")

            # Write to log file
            with open(log_file, 'a') as f:
                f.write(
                    f"{episode},{avg_reward},{friendly_casualties},{enemy_casualties},{episode_length},{actor_loss},{critic_loss},{entropy_loss},{elapsed_time}\n")

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
                    "total_training_time": elapsed_time
                })

        # Save checkpoint
        if save_interval > 0 and (episode % save_interval == 0 or episode == num_episodes - 1):
            checkpoint_dir = os.path.join(output_dir, "models", f"checkpoint_{episode}")
            try:
                marl_ppo.save_agents(checkpoint_dir)
                print(f"Saved checkpoint at episode {episode}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

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

    # Create plots
    try:
        create_training_plots(all_rewards, episode_lengths, log_file, output_dir)

        # Create additional plots
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_episodes + 1), friendly_casualties_history, 'b-', label='Friendly Casualties')
        plt.plot(range(1, num_episodes + 1), enemy_casualties_history, 'r-', label='Enemy Casualties')
        plt.xlabel('Episode')
        plt.ylabel('Casualties')
        plt.title('Casualties Over Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "casualties_over_time.png"))
        plt.close()

        # Plot training time
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_episodes + 1), episode_times, 'g-')
        plt.xlabel('Episode')
        plt.ylabel('Time (seconds)')
        plt.title('Episode Training Time')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "logs", "training_time.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plots: {e}")

    # Finish W&B run if enabled
    if use_wandb:
        if wandb is not None:  # Split into separate conditionals
            wandb.finish()

    return marl_ppo, (
        all_rewards, episode_lengths, friendly_casualties_history, enemy_casualties_history, episode_times)


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
    from WarGamingEnvironment_v10 import ForceType

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


if __name__ == "__main__":
    # Example usage: First run an initial training session
    initial_training_dir = "./initial_training_output"

    print("=== STARTING INITIAL TRAINING ===")
    marl_ppo, stats = custom_training(
        # Environment parameters
        num_episodes=500,
        max_steps_per_episode=3000,
        map_file="training_map_lvl_1.csv",
        objective_location=(88, 50),
        enemy_positions=None,  # Will use default
        unit_start_positions=None,  # Will use default

        # PPO parameters
        learning_rate=1e-3,  # 5e-4 # 3e-4
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.1,  # 0.3 # 0.2
        value_coef=0.5,  # 1 # 0.5
        entropy_coef=0.3,  # 0.1 # 0.01
        ppo_epochs=3,  # 4
        batch_size=128,  # 64

        # Training control parameters
        save_interval=10,
        log_interval=1,
        output_dir=initial_training_dir,
        use_wandb=False,  # Set to True to enable W&B logging
        wandb_project="wargaming_training",
        use_tqdm=True,  # Set to True to enable progress bars

        # Early stopping parameters
        patience=50,
        min_delta=0.5
    )

    print("Initial training complete!")
    best_reward = stats[5]  # Access the best reward from returned stats
    best_enemy_casualties = stats[6]  # Access the best enemy casualties
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Best enemy casualties achieved: {best_enemy_casualties}")

    # # Now resume training from the saved model
    # print("\n=== STARTING RESUMED TRAINING ===")
    # resumed_marl_ppo, resumed_stats = resume_custom_training(
    #     # Environment parameters
    #     num_episodes=1000,  # Continue with more episodes
    #     max_steps_per_episode=300,
    #     map_file="training_map.csv",
    #     objective_location=(73, 75),
    #     enemy_positions=None,
    #     unit_start_positions=None,
    #
    #     # Load from the final model of the initial training
    #     # load_from_dir=f"{initial_training_dir}/models/final",
    #     load_from_dir="./custom_training_output/models/final",
    #
    #     # PPO parameters - can adjust these for continued training
    #     learning_rate=1e-4,  # Lower learning rate for fine-tuning
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_param=0.2,
    #     value_coef=0.5,
    #     entropy_coef=0.01,
    #     ppo_epochs=4,
    #     batch_size=64,
    #
    #     save_interval=20,
    #     log_interval=1,
    #     output_dir=None,  # Will create a timestamped directory
    #     use_wandb=False,
    #     wandb_project="wargaming_training",
    #     use_tqdm=True
    # )
    #
    # print("Resumed training complete!")
