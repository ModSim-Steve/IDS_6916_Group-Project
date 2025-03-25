"""
Model Verification MARL - Multi-Agent RL Testing Framework

This script provides comprehensive tests for the MARL wrapper of the WarGaming Environment.
It includes tests for:
1. Environment initialization and agent creation
2. Observation space verification with tactical features
3. Agent identification with consistent role mapping

Uses the standard coordinate system:
- 0° = East (right)
- 90° = South (down)
- 180° = West (left)
- 270° = North (up)
"""

import os
import math
import argparse
import time
import traceback
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import lines

from WarGamingEnvironment_v9 import (
    MilitaryEnvironment,
    MARLMilitaryEnvironment,
    EnvironmentConfig,
    TerrainType,
    ElevationType,
    ForceType,
    UnitType
)

from US_Army_PLT_Composition_vTest import (
    US_IN_create_platoon,
    US_IN_Role
)


# Create output directories
def create_output_directories():
    """Create directories for test outputs."""
    base_dir = "Model_Verification_MARL"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create subdirectories
    subdirs = [
        "initialization",
        "observation_space",
        "agent_management"
    ]

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

    return base_dir


# Visualization Utilities
class MARLVisualizer:
    """Visualization utility for the MARL test."""

    def __init__(self, env):
        self.env = env

        # Color schemes
        self.terrain_colors = {
            TerrainType.BARE: '#F5F5DC',  # Beige
            TerrainType.SPARSE_VEG: '#90EE90',  # Light green
            TerrainType.DENSE_VEG: '#228B22',  # Forest green
            TerrainType.WOODS: '#006400',  # Dark green
            TerrainType.STRUCTURE: '#808080'  # Gray
        }

        self.elevation_colors = {
            ElevationType.GROUND_LEVEL: '#F5F5DC',  # Beige
            ElevationType.ELEVATED_LEVEL: '#DEB887',  # Burlywood
            ElevationType.LOWER_LEVEL: '#8B4513'  # Saddle brown
        }

        # Agent colors (for up to 10 agents)
        self.agent_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

    def _hex_to_rgb(self, hex_color: str):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def _plot_terrain(self, ax):
        """Plot the terrain and elevation."""
        # Create terrain map
        terrain_img = np.zeros((self.env.height, self.env.width, 3))

        for y in range(self.env.height):
            for x in range(self.env.width):
                terrain_type = self.env.terrain_manager.get_terrain_type((x, y))
                elevation_type = self.env.terrain_manager.get_elevation_type((x, y))

                # Get base colors
                terrain_color = self._hex_to_rgb(self.terrain_colors[terrain_type])
                elevation_color = self._hex_to_rgb(self.elevation_colors[elevation_type])

                # Blend colors (70% terrain, 30% elevation)
                blended_color = 0.7 * np.array(terrain_color) + 0.3 * np.array(elevation_color)
                terrain_img[y, x] = blended_color

        # Plot the terrain
        ax.imshow(terrain_img, extent=(0, self.env.width, 0, self.env.height), origin='lower')

    def plot_environment_state(self, agent_ids=None, enemy_ids=None, title="Environment State",
                               output_path=None, highlight_agent=None, agent_observations=None):
        """
        Create a visualization of the environment state with agents and their observations.

        Args:
            agent_ids: List of agent IDs to plot
            enemy_ids: List of enemy IDs to plot
            title: Title for the plot
            output_path: Path to save the visualization
            highlight_agent: ID of agent to highlight (show observation area)
            agent_observations: Observations for the highlighted agent

        Returns:
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot terrain
        self._plot_terrain(ax)

        # If no agent IDs provided, use all active units that are friendly
        if agent_ids is None:
            agent_ids = [uid for uid in self.env.state_manager.active_units if
                         self.env.get_unit_property(uid, 'force_type') == ForceType.FRIENDLY]

        # If no enemy IDs provided, use all active units that are enemy
        if enemy_ids is None:
            enemy_ids = [uid for uid in self.env.state_manager.active_units if
                         self.env.get_unit_property(uid, 'force_type') == ForceType.ENEMY]

        # Plot agents
        legend_elements = []
        for i, agent_id in enumerate(agent_ids):
            color = self.agent_colors[i % len(self.agent_colors)]

            # Plot agent
            agent_pos = self.env.get_unit_position(agent_id)
            ax.plot(agent_pos[0], agent_pos[1], 'o', color=color, markersize=10)

            # Get actual unit ID if using consistent agent mapping
            unit_id = agent_id
            if hasattr(self.env, 'agent_manager') and hasattr(self.env.agent_manager, 'get_current_unit_id'):
                current_unit_id = self.env.agent_manager.get_current_unit_id(agent_id)
                if current_unit_id:
                    unit_id = current_unit_id
                    # Draw line connecting agent ID to current unit ID if they differ
                    if unit_id != agent_id:
                        unit_pos = self.env.get_unit_position(unit_id)
                        ax.plot([agent_pos[0], unit_pos[0]], [agent_pos[1], unit_pos[1]],
                                '--', color=color, alpha=0.5)

            # Add label with both consistent agent ID and current unit ID if different
            if unit_id != agent_id:
                label_text = f"Agent {agent_id} → Unit {unit_id}"
            else:
                label_text = f"Agent {agent_id}"

            ax.annotate(label_text, xy=agent_pos, xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, color=color)

            # Add to legend
            legend_elements.append(lines.Line2D([0], [0], color=color, marker='o',
                                                linestyle='None', markersize=8,
                                                label=label_text))

            # Plot agent children (team members)
            children = self.env.get_unit_children(unit_id)
            for child_id in children:
                child_pos = self.env.get_unit_position(child_id)
                is_leader = self.env.get_unit_property(child_id, 'is_leader', False)
                marker = '^' if is_leader else 'o'

                # Plot with smaller marker
                ax.plot(child_pos[0], child_pos[1], marker, color=color, markersize=6, alpha=0.7)

                # Add role label
                role = self.env.get_unit_property(child_id, 'role')
                if role is not None:
                    role_name = US_IN_Role(role).name if isinstance(role, int) else str(role)
                    ax.annotate(role_name, xy=child_pos, xytext=(2, 2),
                                textcoords='offset points', fontsize=7, color=color)

        # Plot enemies
        for enemy_id in enemy_ids:
            enemy_pos = self.env.get_unit_position(enemy_id)
            health = self.env.get_unit_property(enemy_id, 'health', 100)

            # Use darker shade of red for low health
            color = 'darkred' if health < 50 else 'red'
            alpha = 0.5 if health <= 0 else 1.0  # Fade out dead enemies

            ax.plot(enemy_pos[0], enemy_pos[1], 'X', color=color, markersize=8, alpha=alpha)
            ax.annotate(f"Enemy {enemy_id}", xy=enemy_pos, xytext=(0, -10),
                        textcoords='offset points', ha='center', va='top',
                        fontsize=8, color=color)

            # Draw health bar
            if health > 0:
                # Background (red) bar
                ax.add_patch(patches.Rectangle(
                    (enemy_pos[0] - 1.5, enemy_pos[1] + 1.5),
                    3.0, 0.3,
                    facecolor='red', alpha=0.7
                ))

                # Health (green) bar
                ax.add_patch(patches.Rectangle(
                    (enemy_pos[0] - 1.5, enemy_pos[1] + 1.5),
                    3.0 * health / 100.0, 0.3,
                    facecolor='green', alpha=0.7
                ))

        # Add enemies to legend
        legend_elements.append(lines.Line2D([0], [0], color='red', marker='X',
                                            linestyle='None', markersize=8,
                                            label="Enemy"))

        # Highlight specific agent's observation
        if highlight_agent is not None and highlight_agent in agent_ids:
            # Get current unit ID if using consistent agent mapping
            unit_id = highlight_agent
            if hasattr(self.env, 'agent_manager') and hasattr(self.env.agent_manager, 'get_current_unit_id'):
                current_unit_id = self.env.agent_manager.get_current_unit_id(highlight_agent)
                if current_unit_id:
                    unit_id = current_unit_id

            agent_pos = self.env.get_unit_position(unit_id)

            # Draw observation radius
            obs_range = self.env.get_unit_property(unit_id, 'observation_range', 50)
            obs_circle = plt.Circle(agent_pos, obs_range, color='blue', fill=False,
                                    linestyle='--', alpha=0.5)
            ax.add_patch(obs_circle)

            # If we have observation data, visualize known enemies
            if agent_observations and highlight_agent in agent_observations:
                obs = agent_observations[highlight_agent]

                # Add known enemies from observation
                if 'known_enemies' in obs:
                    known_enemies = obs['known_enemies']
                    for i, enemy_pos in enumerate(known_enemies):
                        # Skip zero positions (padding)
                        if enemy_pos[0] == 0 and enemy_pos[1] == 0:
                            continue

                        # Draw line to known enemy
                        ax.plot([agent_pos[0], enemy_pos[0]], [agent_pos[1], enemy_pos[1]],
                                ':', color='blue', alpha=0.5)

                        # Mark known enemy
                        ax.plot(enemy_pos[0], enemy_pos[1], 'o', color='blue', markersize=5, alpha=0.5)

                # Add observation info to legend
                legend_elements.append(lines.Line2D([0], [0], color='blue', linestyle='--',
                                                    label=f"Agent {highlight_agent} Observation Range"))

                # Visualize objective direction if available
                if 'objective_info' in obs and 'direction' in obs['objective_info']:
                    # Draw direction arrow to objective
                    direction = obs['objective_info']['direction']
                    if isinstance(direction, np.ndarray) and direction.size >= 2:
                        # Scale arrow to be visible
                        arrow_length = 10  # Fixed visible length
                        dx, dy = direction[0] * arrow_length, direction[1] * arrow_length

                        # Add direction arrow - convert numpy array to float
                        ax.arrow(float(agent_pos[0]), float(agent_pos[1]), float(dx), float(dy),
                                 head_width=1.0, head_length=1.5, fc='gold', ec='black', alpha=0.7)

                        # Add to legend
                        legend_elements.append(lines.Line2D([0], [0], color='gold', linestyle='-',
                                                            label="Direction to Objective"))

                # Add tactical features visualization in a corner box
                if 'tactical_features' in obs:
                    # Create inset for tactical features
                    width_ratio = 0.25
                    height_ratio = 0.2
                    # Convert list to tuple for fig.add_axes to fix type error
                    inset_ax = fig.add_axes((0.02, 0.02, width_ratio, height_ratio))

                    # Extract tactical features
                    features = obs['tactical_features']

                    # List feature names for clarity
                    feature_names = [
                        'Enemy Count', 'Friendly Count',
                        'Forest Count', 'Building Count',
                        'Mean Elevation', 'Has Cover',
                        'Enemy Center X', 'Enemy Center Y'
                    ]

                    # Create horizontal bar chart of features
                    y_pos = np.arange(len(features))
                    inset_ax.barh(y_pos, features, align='center')
                    inset_ax.set_yticks(y_pos)
                    inset_ax.set_yticklabels(feature_names, fontsize=8)
                    inset_ax.set_title(f"Agent {highlight_agent} Tactical Features", fontsize=9)

                    # Add values next to bars
                    for i, v in enumerate(features):
                        inset_ax.text(v + 0.1, i, f"{v:.2f}", va='center', fontsize=7)

        # Plot objective position if available
        if hasattr(self.env, 'objective') and self.env.objective:
            obj_pos = self.env.objective
            ax.plot(obj_pos[0], obj_pos[1], '*', color='gold', markersize=15,
                    label='Objective')
            ax.annotate('OBJECTIVE', xy=obj_pos, xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=10,
                        weight='bold', color='darkgoldenrod')

            # Add objective to legend
            legend_elements.append(lines.Line2D([0], [0], color='gold', marker='*',
                                                linestyle='None', markersize=10,
                                                label="Objective"))

        # Set title, legend, and grid
        ax.set_title(title, fontsize=16)
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)

        # Set limits
        ax.set_xlim(-1, self.env.width + 1)
        ax.set_ylim(-1, self.env.height + 1)

        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def visualize_observation_space(self, observation: Dict, title: str = "Agent Observation",
                                    output_path: Optional[str] = None):
        """
        Create visualization of agent's observation space components with tactical features.

        Args:
            observation: Observation dictionary for an agent
            title: Title for the plot
            output_path: Path to save the visualization

        Returns:
            Figure and axis objects
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        grid = plt.GridSpec(3, 3, figure=fig)

        # Main title
        fig.suptitle(title, fontsize=16)

        # 1. Tactical Features visualization (replaces local view)
        if 'tactical_features' in observation:
            tactical_features = observation['tactical_features']
            ax_features = fig.add_subplot(grid[0:2, 0:2])

            # Define feature names based on expected structure
            feature_names = [
                'Enemy Count', 'Friendly Count',
                'Forest Count', 'Building Count',
                'Mean Elevation', 'Has Cover',
                'Enemy Center X', 'Enemy Center Y'
            ]

            # Truncate names if we have more features than expected
            if len(tactical_features) > len(feature_names):
                feature_names.extend([f'Feature {i + len(feature_names)}'
                                      for i in range(len(tactical_features) - len(feature_names))])

            # Or truncate features if we have more names
            features_to_plot = tactical_features[:len(feature_names)]

            # Create horizontal bar chart
            y_pos = np.arange(len(features_to_plot))
            bars = ax_features.barh(y_pos, features_to_plot, align='center',
                                    color='skyblue', alpha=0.7)
            ax_features.set_yticks(y_pos)
            ax_features.set_yticklabels(feature_names[:len(features_to_plot)])

            # Add values as text
            for i, v in enumerate(features_to_plot):
                ax_features.text(v + 0.1, i, f"{v:.2f}", va='center')

            ax_features.set_title("Tactical Features", fontsize=12)
            ax_features.set_xlabel("Value", fontsize=10)

            # Add explanation of tactical features
            feature_explanation = (
                "Tactical Features:\n"
                "- Enemy/Friendly Count: Detected units in view\n"
                "- Forest/Building Count: Terrain features in view\n"
                "- Mean Elevation: Average elevation in area\n"
                "- Has Cover: Binary indicator of available cover\n"
                "- Enemy Center: Center of mass of enemy positions"
            )

            ax_features.text(0.5, -0.15, feature_explanation, transform=ax_features.transAxes,
                             fontsize=9, ha='center', va='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        elif 'local_view' in observation:
            # For backward compatibility with old observation format
            local_view = observation['local_view']
            ax_local = fig.add_subplot(grid[0:2, 0:2])

            # Create RGB visualization of local view
            local_rgb = np.zeros((*local_view.shape[:2], 3))

            for y in range(local_view.shape[0]):
                for x in range(local_view.shape[1]):
                    # Use terrain channel for color
                    terrain_val = local_view[y, x, 0]
                    terrain_type = TerrainType(terrain_val) if 0 <= terrain_val < len(
                        TerrainType) else TerrainType.BARE
                    terrain_color = self._hex_to_rgb(self.terrain_colors[terrain_type])

                    # Mark units (add blue for friendly, red for enemy)
                    unit_id = local_view[y, x, 2]
                    if unit_id > 0:
                        # Check if unit exists in environment
                        if unit_id in self.env.state_manager.active_units:
                            force_type = self.env.get_unit_property(int(unit_id), 'force_type', None)
                            if force_type == ForceType.FRIENDLY:
                                # Add blue tint for friendly
                                local_rgb[y, x] = np.array([0.3, 0.3, 0.8])
                            elif force_type == ForceType.ENEMY:
                                # Add red tint for enemy
                                local_rgb[y, x] = np.array([0.8, 0.3, 0.3])
                            else:
                                local_rgb[y, x] = terrain_color
                        else:
                            local_rgb[y, x] = terrain_color
                    else:
                        local_rgb[y, x] = terrain_color

            # Display local view
            ax_local.imshow(local_rgb, origin='lower')
            ax_local.set_title("Local View (Legacy Format)", fontsize=12)
            ax_local.set_xticks([])
            ax_local.set_yticks([])

        # 2. Agent state visualization
        if 'agent_state' in observation:
            agent_state = observation['agent_state']
            ax_state = fig.add_subplot(grid[0, 2])

            # Extract state components
            state_labels = []
            state_values = []

            for key, value in agent_state.items():
                if isinstance(value, np.ndarray):
                    # Handle different sized arrays
                    if value.size == 1:
                        state_labels.append(key)
                        state_values.append(float(value[0]))
                    elif value.size == 2 and key == 'position':
                        state_labels.extend([f"{key}_x", f"{key}_y"])
                        state_values.extend([float(value[0]), float(value[1])])
                else:
                    state_labels.append(key)
                    state_values.append(float(value))

            # Plot agent state as horizontal bar chart
            y_pos = np.arange(len(state_labels))
            ax_state.barh(y_pos, state_values, align='center', alpha=0.7)
            ax_state.set_yticks(y_pos)
            ax_state.set_yticklabels(state_labels)
            ax_state.invert_yaxis()  # Labels read top-to-bottom
            ax_state.set_title("Agent State", fontsize=12)

            # Add values as text labels
            for i, v in enumerate(state_values):
                ax_state.text(v + 0.1, i, f"{v:.1f}", va='center')

        # 3. Known enemies visualization
        if 'known_enemies' in observation:
            known_enemies = observation['known_enemies']
            ax_enemies = fig.add_subplot(grid[1, 2])

            # Calculate distances to each known enemy
            if 'agent_state' in observation and 'position' in observation['agent_state']:
                agent_pos = observation['agent_state']['position']
                distances = []

                for enemy_pos in known_enemies:
                    # Skip zeros (padding)
                    if enemy_pos[0] == 0 and enemy_pos[1] == 0:
                        continue

                    # Calculate distance
                    dx = enemy_pos[0] - agent_pos[0]
                    dy = enemy_pos[1] - agent_pos[1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    distances.append(distance)

                # Plot distances to enemies
                if distances:
                    enemy_indices = np.arange(len(distances))
                    ax_enemies.bar(enemy_indices, distances, alpha=0.7, color='red')
                    ax_enemies.set_xticks(enemy_indices)
                    ax_enemies.set_xticklabels([f"E{i + 1}" for i in range(len(distances))])
                    ax_enemies.set_title(f"Distances to {len(distances)} Known Enemies", fontsize=12)
                    ax_enemies.set_ylabel("Distance", fontsize=10)
                else:
                    ax_enemies.text(0.5, 0.5, "No known enemies", ha='center', va='center',
                                    transform=ax_enemies.transAxes, fontsize=10)
                    ax_enemies.set_title("Known Enemies", fontsize=12)
            else:
                # Just show enemy count
                non_zero_enemies = sum(1 for e in known_enemies if e[0] != 0 or e[1] != 0)
                ax_enemies.text(0.5, 0.5, f"{non_zero_enemies} known enemies",
                                ha='center', va='center',
                                transform=ax_enemies.transAxes, fontsize=10)
                ax_enemies.set_title("Known Enemies", fontsize=12)

            ax_enemies.set_ylim(bottom=0)

        # 4. Friendly units visualization
        if 'friendly_units' in observation:
            friendly_units = observation['friendly_units']
            ax_friendly = fig.add_subplot(grid[2, 0])

            # Count non-zero friendly units
            non_zero_friendly = sum(1 for f in friendly_units if f[0] != 0 or f[1] != 0)

            if non_zero_friendly > 0:
                # Create friendly unit positions plot
                for i, pos in enumerate(friendly_units):
                    # Skip zeros (padding)
                    if pos[0] == 0 and pos[1] == 0:
                        continue

                    ax_friendly.plot(pos[0], pos[1], 'o', color='blue', markersize=8)
                    ax_friendly.annotate(f"F{i + 1}", xy=pos, xytext=(3, 3),
                                         textcoords='offset points', fontsize=8)

                # Add agent position for reference if available
                if 'agent_state' in observation and 'position' in observation['agent_state']:
                    agent_pos = observation['agent_state']['position']
                    ax_friendly.plot(agent_pos[0], agent_pos[1], 'D', color='green', markersize=10)
                    ax_friendly.annotate("Agent", xy=agent_pos, xytext=(3, 3),
                                         textcoords='offset points', fontsize=8)

                ax_friendly.set_title(f"{non_zero_friendly} Friendly Units", fontsize=12)

                # Set reasonable limits based on environment size
                ax_friendly.set_xlim(0, self.env.width)
                ax_friendly.set_ylim(0, self.env.height)
                ax_friendly.grid(True, linestyle='--', alpha=0.3)
            else:
                ax_friendly.text(0.5, 0.5, "No friendly units in radio range",
                                 ha='center', va='center',
                                 transform=ax_friendly.transAxes, fontsize=10)
                ax_friendly.set_title("Friendly Units", fontsize=12)

        # 5. Objective visualization with direction and distance
        if 'objective' in observation or 'objective_info' in observation:
            ax_objective = fig.add_subplot(grid[2, 1])

            # Get objective position
            objective_pos = None
            if 'objective' in observation:
                objective_pos = observation['objective']

            # Only show if objective is not at origin (which is often the default)
            if objective_pos is not None and (objective_pos[0] != 0 or objective_pos[1] != 0):
                # Plot objective
                ax_objective.plot(objective_pos[0], objective_pos[1], '*',
                                  color='gold', markersize=15)

                # Add agent position for reference if available
                if 'agent_state' in observation and 'position' in observation['agent_state']:
                    agent_pos = observation['agent_state']['position']
                    ax_objective.plot(agent_pos[0], agent_pos[1], 'D', color='green', markersize=10)

                    # Draw line from agent to objective
                    ax_objective.plot([agent_pos[0], objective_pos[0]],
                                      [agent_pos[1], objective_pos[1]],
                                      '--', color='gray', alpha=0.7)

                    # Get distance if available, otherwise calculate it
                    distance = None
                    if ('objective_info' in observation and
                            'distance' in observation['objective_info']):
                        distance = float(observation['objective_info']['distance'][0])
                    else:
                        # Calculate distance
                        dx = objective_pos[0] - agent_pos[0]
                        dy = objective_pos[1] - agent_pos[1]
                        distance = math.sqrt(dx * dx + dy * dy)

                    # Add distance label
                    midpoint = ((agent_pos[0] + objective_pos[0]) / 2,
                                (agent_pos[1] + objective_pos[1]) / 2)
                    ax_objective.annotate(f"Dist: {distance:.1f}", xy=midpoint,
                                          xytext=(0, 5), textcoords='offset points',
                                          ha='center', fontsize=9)

                    # If we have direction information, add a direction arrow
                    if ('objective_info' in observation and
                            'direction' in observation['objective_info']):
                        direction = observation['objective_info']['direction']
                        if isinstance(direction, np.ndarray) and direction.size >= 2:
                            # Draw direction arrow
                            arrow_length = 5  # Fixed visible length for arrow
                            dx, dy = direction[0] * arrow_length, direction[1] * arrow_length

                            # Add direction arrow from agent position
                            ax_objective.arrow(float(agent_pos[0]), float(agent_pos[1]), float(dx), float(dy),
                                               head_width=0.8, head_length=1.0,
                                               fc='gold', ec='black', alpha=0.7)

                            # Add direction label
                            arrow_midpoint = (agent_pos[0] + dx / 2, agent_pos[1] + dy / 2)
                            ax_objective.annotate("Direction", xy=arrow_midpoint,
                                                  xytext=(0, -5), textcoords='offset points',
                                                  ha='center', fontsize=8, color='darkgoldenrod')

                ax_objective.set_title("Objective Location", fontsize=12)

                # Set reasonable limits based on environment size
                ax_objective.set_xlim(0, self.env.width)
                ax_objective.set_ylim(0, self.env.height)
                ax_objective.grid(True, linestyle='--', alpha=0.3)
            else:
                ax_objective.text(0.5, 0.5, "No objective defined",
                                  ha='center', va='center',
                                  transform=ax_objective.transAxes, fontsize=10)
                ax_objective.set_title("Objective", fontsize=12)

        # 6. Observation summary/metadata
        ax_summary = fig.add_subplot(grid[2, 2])
        summary_text = "Observation Summary:\n\n"

        # Count observation components
        component_counts = {}
        total_elements = 0

        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                shape_str = 'x'.join(str(dim) for dim in value.shape)
                component_counts[key] = f"Array ({shape_str})"
                total_elements += value.size
            elif isinstance(value, dict):
                component_counts[key] = f"Dict with {len(value)} keys"
                total_elements += len(value)
            else:
                component_counts[key] = str(type(value).__name__)
                total_elements += 1

        # Add component info to summary
        for key, info in component_counts.items():
            summary_text += f"• {key}: {info}\n"

        # Highlight new tactical features if present
        if 'tactical_features' in observation:
            summary_text += "\n* Using new tactical features condensed format *\n"

        if 'objective_info' in observation:
            summary_text += "* Enhanced objective information available *\n"

        summary_text += f"\nTotal Components: {len(component_counts)}\n"
        summary_text += f"Total Elements: {total_elements}"

        # Display summary
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                        transform=ax_summary.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax_summary.set_title("Observation Structure", fontsize=12)
        ax_summary.axis('off')

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make room for the main title

        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def visualize_agent_mapping(self, env, output_path=None):
        """
        Visualize the consistent agent mapping system showing the relationship
        between agent IDs and unit IDs.

        Args:
            env: MARLMilitaryEnvironment instance
            output_path: Path to save the visualization

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define consistent agent roles
        roles = {
            1: "1SQD_Leader",
            2: "2SQD_Leader",
            3: "3SQD_Leader",
            4: "GTM1_Leader",
            5: "GTM2_Leader",
            6: "JTM1_Leader",
            7: "JTM2_Leader"
        }

        # Get current mapping
        mapping = {}
        if hasattr(env, 'agent_manager') and hasattr(env.agent_manager, 'unit_id_to_agent_id'):
            mapping = env.agent_manager.unit_id_to_agent_id

        # Reverse mapping
        agent_to_unit = {}
        for unit_id, agent_id in mapping.items():
            agent_to_unit[agent_id] = unit_id

        # Prepare node positions
        role_positions = {}
        unit_positions = {}
        agent_positions = {}

        # Set up positions
        num_agents = len(roles)

        # Role nodes (left column)
        for i, (agent_id, role) in enumerate(roles.items()):
            role_positions[role] = (1, 9 - i)

        # Agent ID nodes (middle column)
        for i, agent_id in enumerate(sorted(roles.keys())):
            agent_positions[agent_id] = (3, 9 - i)

        # Unit ID nodes (right column)
        for agent_id, unit_id in agent_to_unit.items():
            if agent_id in agent_positions:
                # Same y-position as the agent ID
                unit_positions[unit_id] = (5, agent_positions[agent_id][1])

        # Plot nodes
        # Role nodes
        for role, pos in role_positions.items():
            ax.plot(pos[0], pos[1], 'o', markersize=10, color='lightblue')
            ax.annotate(role, xy=pos, xytext=(0, -10),
                        textcoords='offset points', ha='center', fontsize=9)

        # Agent ID nodes
        for agent_id, pos in agent_positions.items():
            ax.plot(pos[0], pos[1], 'o', markersize=10, color='lightgreen')
            ax.annotate(f"Agent {agent_id}", xy=pos, xytext=(0, -10),
                        textcoords='offset points', ha='center', fontsize=9)

        # Unit ID nodes
        for unit_id, pos in unit_positions.items():
            ax.plot(pos[0], pos[1], 'o', markersize=10, color='salmon')

            # Get additional unit info
            role_value = env.get_unit_property(unit_id, 'role', None)
            role_name = "Unknown"
            if isinstance(role_value, int):
                try:
                    role_name = US_IN_Role(role_value).name
                except:
                    role_name = str(role_value)

            # Include unit role in the label
            ax.annotate(f"Unit {unit_id}\n({role_name})", xy=pos, xytext=(0, -20),
                        textcoords='offset points', ha='center', fontsize=9)

        # Plot connections
        # Role to Agent connections
        for agent_id, role in roles.items():
            if role in role_positions and agent_id in agent_positions:
                role_pos = role_positions[role]
                agent_pos = agent_positions[agent_id]
                ax.plot([role_pos[0], agent_pos[0]], [role_pos[1], agent_pos[1]],
                        '-', color='gray', alpha=0.7)

        # Agent to Unit connections
        for agent_id, unit_id in agent_to_unit.items():
            if agent_id in agent_positions and unit_id in unit_positions:
                agent_pos = agent_positions[agent_id]
                unit_pos = unit_positions[unit_id]
                ax.plot([agent_pos[0], unit_pos[0]], [agent_pos[1], unit_pos[1]],
                        '-', color='gray', alpha=0.7)

        # Add labels
        ax.text(1, 10.5, "Tactical Role", ha='center', fontsize=12, weight='bold')
        ax.text(3, 10.5, "Consistent Agent ID", ha='center', fontsize=12, weight='bold')
        ax.text(5, 10.5, "Current Unit ID", ha='center', fontsize=12, weight='bold')

        # Add title and description
        ax.set_title("Consistent Agent Mapping System", fontsize=14)
        description = (
            "The agent mapping system maintains a consistent mapping between:\n"
            "1. Tactical roles (e.g., Squad Leader)\n"
            "2. Consistent agent IDs (1-7 for standard platoon)\n"
            "3. Current unit IDs (which may change during succession)"
        )
        ax.text(3, 0.5, description, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set limits
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 11)

        # Add border
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig


# Environment Setup Functions
def setup_terrain(env, width=70, height=25):
    """Set up the terrain according to the requirements."""
    # Default terrain - BARE
    for y in range(height):
        for x in range(width):
            env.state_manager.state_tensor[y, x, 0] = TerrainType.BARE.value
            env.state_manager.state_tensor[y, x, 1] = ElevationType.GROUND_LEVEL.value

    # Set up terrain for center part of map
    mid_y = height // 2
    mid_y_range = range(mid_y - 3, mid_y + 3)

    # Create terrain bands from left to right
    # - sparse vegetation from x=5 to x=15
    for y in mid_y_range:
        for x in range(5, 15):
            env.state_manager.state_tensor[y, x, 0] = TerrainType.SPARSE_VEG.value

    # - dense vegetation from x=15 to x=35
    for y in mid_y_range:
        for x in range(15, 35):
            env.state_manager.state_tensor[y, x, 0] = TerrainType.DENSE_VEG.value

    # - woods terrain from x=35 to x=55
    for y in mid_y_range:
        for x in range(35, 55):
            env.state_manager.state_tensor[y, x, 0] = TerrainType.WOODS.value

    # - elevated terrain with structures from x=55 to x=65
    for y in mid_y_range:
        for x in range(55, 65):
            env.state_manager.state_tensor[y, x, 0] = TerrainType.STRUCTURE.value
            env.state_manager.state_tensor[y, x, 1] = ElevationType.ELEVATED_LEVEL.value

    # Create some scattered vegetation in other areas
    for _ in range(50):  # 50 random spots
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        # Skip if already in the center band
        if y in mid_y_range and 5 <= x < 65:
            continue
        # Random vegetation
        veg_type = np.random.choice([TerrainType.SPARSE_VEG.value, TerrainType.DENSE_VEG.value])
        env.state_manager.state_tensor[y, x, 0] = veg_type


def create_enemy_targets(env, positions=None):
    """
    Create enemy targets at the specified positions.

    Args:
        env: Military environment
        positions: Optional list of (x,y) positions. If None, default positions are used.

    Returns:
        List of enemy unit IDs
    """
    enemy_ids = []

    # Use default positions if none provided
    if positions is None:
        # Create enemies at various distances
        positions = [
            (25, 10), (25, 15),
            (35, 10), (35, 15),
            (45, 10), (45, 15)
        ]

    for i, pos in enumerate(positions):
        # Create enemy rifleman - use UnitType directly from the import
        enemy_id = env.create_unit(
            unit_type=UnitType.INFANTRY_TEAM,
            unit_id_str=f"ENEMY-{i + 1}",
            start_position=pos
        )

        # Set as enemy force
        env.update_unit_property(enemy_id, 'force_type', ForceType.ENEMY)

        # Create an enemy soldier
        soldier_id = env.create_soldier(
            role=US_IN_Role.RIFLEMAN,
            unit_id_str=f"ENEMY-{i + 1}-RFLM",
            position=pos,
            is_leader=False
        )

        # Set as enemy force
        env.update_unit_property(soldier_id, 'force_type', ForceType.ENEMY)

        # Set parent-child relationship
        env.set_unit_hierarchy(soldier_id, enemy_id)

        enemy_ids.append(enemy_id)

    return enemy_ids


def setup_environment(width=70, height=25, debug_level=1, marl_wrapper=True, objective_pos=None):
    """
    Set up the environment with the specified dimensions.

    Args:
        width: Environment width
        height: Environment height
        debug_level: Debug output level
        marl_wrapper: If True, use the MARL wrapper
        objective_pos: Optional position of objective

    Returns:
        Environment instance
    """
    # Create basic config
    config = EnvironmentConfig(
        width=width,
        height=height,
        debug_level=debug_level
    )

    # Create appropriate environment type
    if marl_wrapper:
        env = MARLMilitaryEnvironment(config, objective_position=objective_pos)
    else:
        env = MilitaryEnvironment(config)

    # Initialize terrain
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)

    # Set up custom terrain
    setup_terrain(env, width, height)

    return env


def print_detailed_platoon_state(env, platoon_id):
    """
    Print detailed state information for a platoon, including all positions and soldiers.
    Updated to show both position IDs and soldier IDs in the position-based system.

    Args:
        env: MARLMilitaryEnvironment instance
        platoon_id: ID of platoon
    """
    print("\n" + "=" * 80)
    print("DETAILED PLATOON STATE")
    print("=" * 80)

    # Get platoon information
    plt_string = env.get_unit_property(platoon_id, 'string_id', str(platoon_id))
    plt_pos = env.get_unit_position(platoon_id)
    plt_orientation = env.get_unit_property(platoon_id, 'orientation', 0)
    plt_formation = env.get_unit_property(platoon_id, 'formation', None)

    print(f"Platoon: {plt_string} (Position ID: {platoon_id})")
    print(f"Position: {plt_pos}, Orientation: {plt_orientation}°, Formation: {plt_formation}")

    # Get platoon leader
    plt_leader = None
    for child_id in env.get_unit_children(platoon_id):
        is_leader = env.get_unit_property(child_id, 'is_leader', False)
        role = env.get_unit_property(child_id, 'role', None)

        if is_leader and role == US_IN_Role.PLATOON_LEADER.value:
            plt_leader = child_id
            break

    if plt_leader:
        plt_leader_pos = env.get_unit_position(plt_leader)
        plt_leader_health = env.get_unit_property(plt_leader, 'health', 0)
        plt_leader_soldier = env.get_unit_property(plt_leader, 'soldier_id', 'Unknown')
        print(f"Platoon Leader: Position {plt_leader}, Soldier {plt_leader_soldier}")
        print(f"  Position: {plt_leader_pos}, Health: {plt_leader_health}/100")

    # Print squads information
    print("\n----- SQUADS -----")

    # Get squads from platoon
    squads = []
    for unit_id in env.get_unit_children(platoon_id):
        unit_type = env.get_unit_property(unit_id, 'type')
        string_id = env.get_unit_property(unit_id, 'string_id', '')
        if unit_type == UnitType.INFANTRY_SQUAD or ("1SQD" in string_id or "2SQD" in string_id or "3SQD" in string_id):
            squads.append(unit_id)

    for squad_id in squads:
        print("\n" + "-" * 50)
        squad_string = env.get_unit_property(squad_id, 'string_id', str(squad_id))
        squad_pos = env.get_unit_position(squad_id)
        squad_orientation = env.get_unit_property(squad_id, 'orientation', 0)
        squad_formation = env.get_unit_property(squad_id, 'formation', None)

        print(f"Squad: {squad_string} (Position ID: {squad_id})")
        print(f"Position: {squad_pos}, Orientation: {squad_orientation}°, Formation: {squad_formation}")

        # Find squad leader
        squad_leader = None
        for child_id in env.get_unit_children(squad_id):
            is_leader = env.get_unit_property(child_id, 'is_leader', False)
            role = env.get_unit_property(child_id, 'role', None)

            if is_leader and role == US_IN_Role.SQUAD_LEADER.value:
                squad_leader = child_id
                break

        if squad_leader:
            sl_pos = env.get_unit_position(squad_leader)
            sl_health = env.get_unit_property(squad_leader, 'health', 0)
            sl_soldier = env.get_unit_property(squad_leader, 'soldier_id', 'Unknown')
            is_leader = env.get_unit_property(squad_leader, 'is_leader', False)
            leader_str = "LEADER" if is_leader else "MEMBER"

            # Check if SL is an agent
            is_agent = env.get_unit_property(squad_leader, 'is_agent', False)

            # Also check consistent agent mapping
            agent_id = None
            if hasattr(env, 'agent_manager') and hasattr(env.agent_manager, 'get_agent_id'):
                agent_id = env.agent_manager.get_agent_id(squad_leader)

            agent_str = f"AGENT {agent_id}" if is_agent else "NON-AGENT"

            print(f"Squad Leader: Position {squad_leader}, Soldier {sl_soldier}")
            print(f"  Position: {sl_pos}, Health: {sl_health}/100, [{leader_str}] [{agent_str}]")

            # Get weapon and ammo info if available
            weapon = env.get_unit_property(squad_leader, 'primary_weapon')
            ammo = 0
            if hasattr(env, 'combat_manager'):
                ammo = env.combat_manager._get_unit_ammo(squad_leader, 'primary')

            if weapon:
                weapon_name = weapon.name if hasattr(weapon, 'name') else str(weapon)
                print(f"  Weapon: {weapon_name}, Ammo: {ammo}")

        # Get teams in squad
        teams = []
        for unit_id in env.get_unit_children(squad_id):
            unit_type = env.get_unit_property(unit_id, 'type')
            string_id = env.get_unit_property(unit_id, 'string_id', '')
            if unit_type == UnitType.INFANTRY_TEAM or (
                    "ATM" in string_id or "BTM" in string_id):
                teams.append(unit_id)

        # Print team info
        for team_id in teams:
            team_string = env.get_unit_property(team_id, 'string_id', str(team_id))
            team_pos = env.get_unit_position(team_id)
            team_orientation = env.get_unit_property(team_id, 'orientation', 0)
            team_formation = env.get_unit_property(team_id, 'formation', None)

            print(f"\n  Team: {team_string} (Position ID: {team_id})")
            print(f"  Position: {team_pos}, Orientation: {team_orientation}°, Formation: {team_formation}")

            # Print team members
            members = env.get_unit_children(team_id)
            print(f"  Members ({len(members)}):")

            for member_id in members:
                member_pos = env.get_unit_position(member_id)
                role_value = env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
                health = env.get_unit_property(member_id, 'health', 0)
                soldier_id = env.get_unit_property(member_id, 'soldier_id', 'Unknown')
                is_leader = env.get_unit_property(member_id, 'is_leader', False)
                leader_str = "LEADER" if is_leader else "MEMBER"

                # Check if member is an agent
                is_agent = env.get_unit_property(member_id, 'is_agent', False)

                # Also check consistent agent mapping
                agent_id = None
                if hasattr(env, 'agent_manager') and hasattr(env.agent_manager, 'get_agent_id'):
                    agent_id = env.agent_manager.get_agent_id(member_id)

                agent_str = f"AGENT {agent_id}" if is_agent else "NON-AGENT"

                print(f"    {role_name} - Position {member_id}, Soldier {soldier_id}")
                print(f"      Pos {member_pos}, Health {health}/100, [{leader_str}] [{agent_str}]")

                # Get weapon and ammo info
                weapon = env.get_unit_property(member_id, 'primary_weapon')
                ammo = 0
                if hasattr(env, 'combat_manager'):
                    ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')

                weapon_name = weapon.name if hasattr(weapon, 'name') else str(weapon)

                print(f"      Weapon: {weapon_name}, Ammo: {ammo}")

    # Print weapons teams information
    print("\n----- WEAPONS TEAMS -----")

    # Get weapons teams from platoon
    weapon_teams = []
    for unit_id in env.get_unit_children(platoon_id):
        unit_type = env.get_unit_property(unit_id, 'type')
        string_id = env.get_unit_property(unit_id, 'string_id', '')

        if unit_type == UnitType.WEAPONS_TEAM or 'GTM' in string_id or 'JTM' in string_id:
            weapon_teams.append(unit_id)

    for team_id in weapon_teams:
        print("\n" + "-" * 50)
        team_string = env.get_unit_property(team_id, 'string_id', str(team_id))
        team_pos = env.get_unit_position(team_id)
        team_orientation = env.get_unit_property(team_id, 'orientation', 0)
        team_formation = env.get_unit_property(team_id, 'formation', None)

        print(f"Weapons Team: {team_string} (Position ID: {team_id})")
        print(f"Position: {team_pos}, Orientation: {team_orientation}°, Formation: {team_formation}")

        # Print team members
        members = env.get_unit_children(team_id)
        print(f"Members ({len(members)}):")

        for member_id in members:
            member_pos = env.get_unit_position(member_id)
            role_value = env.get_unit_property(member_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
            health = env.get_unit_property(member_id, 'health', 0)
            soldier_id = env.get_unit_property(member_id, 'soldier_id', 'Unknown')
            is_leader = env.get_unit_property(member_id, 'is_leader', False)
            leader_str = "LEADER" if is_leader else "MEMBER"

            # Check if this member is an agent
            is_agent = env.get_unit_property(member_id, 'is_agent', False)

            # Also check consistent agent mapping
            agent_id = None
            if hasattr(env, 'agent_manager') and hasattr(env.agent_manager, 'get_agent_id'):
                agent_id = env.agent_manager.get_agent_id(member_id)

            agent_str = f"AGENT {agent_id}" if is_agent else "NON-AGENT"

            print(f"    {role_name} - Position {member_id}, Soldier {soldier_id}")
            print(f"      Pos {member_pos}, Health {health}/100, [{leader_str}] [{agent_str}]")

            # Get weapon and ammo info
            weapon = env.get_unit_property(member_id, 'primary_weapon')
            ammo = 0
            if hasattr(env, 'combat_manager'):
                ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')

            weapon_name = weapon.name if hasattr(weapon, 'name') else str(weapon)

            print(f"      Weapon: {weapon_name}, Ammo: {ammo}")

    # Print agent summary
    if hasattr(env, 'agent_ids') and env.agent_ids:
        print("\n----- AGENT SUMMARY -----")
        print(f"Total Agents: {len(env.agent_ids)}")

        for i, agent_id in enumerate(env.agent_ids):
            agent_type = "Unknown"
            if hasattr(env, 'agent_manager') and hasattr(env.agent_manager, 'agent_types'):
                agent_type = env.agent_manager.agent_types.get(agent_id, "Unknown")

            # Get current unit ID
            current_unit_id = None
            if hasattr(env, 'agent_manager') and hasattr(env.agent_manager, 'get_current_unit_id'):
                current_unit_id = env.agent_manager.get_current_unit_id(agent_id)

            if current_unit_id:
                role_value = env.get_unit_property(current_unit_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
                soldier_id = env.get_unit_property(current_unit_id, 'soldier_id', 'Unknown')
                parent_id = env.get_unit_property(current_unit_id, 'parent_id')
                parent_string = "None"
                if parent_id:
                    parent_string = env.get_unit_property(parent_id, 'string_id', str(parent_id))

                print(f"Agent {agent_id}: Mapped to Position {current_unit_id}, Soldier {soldier_id}")
                print(f"  Type {agent_type}, Role {role_name}, Unit {parent_string}")
            else:
                print(f"Agent {agent_id}: No current unit mapping")

    # Print consistent mapping summary if available
    if hasattr(env, 'agent_manager') and hasattr(env.agent_manager, 'unit_id_to_agent_id'):
        print("\n----- CONSISTENT AGENT MAPPING -----")
        mapping = env.agent_manager.unit_id_to_agent_id
        print(f"Total mappings: {len(mapping)}")

        # Define roles for reference
        roles = {
            1: "1SQD_Leader",
            2: "2SQD_Leader",
            3: "3SQD_Leader",
            4: "GTM1_Leader",
            5: "GTM2_Leader",
            6: "JTM1_Leader",
            7: "JTM2_Leader"
        }

        # Print each mapping
        for unit_id, agent_id in mapping.items():
            role_name = roles.get(agent_id, "Unknown")
            unit_string = env.get_unit_property(unit_id, 'string_id', str(unit_id))
            print(f"  Unit {unit_id} ({unit_string}) → Agent {agent_id} ({role_name})")


# Test Functions
def test_environment_initialization(output_dir="Model_Verification_MARL/initialization"):
    """
    Test the initialization of the MARL environment with updated agent mapping system.
    Validates:
    1. Environment creation with MARL wrapper
    2. Consistent agent creation with role-based mapping
    3. Tactical features observation format
    4. Objective direction and distance information

    Args:
        output_dir: Directory to save output files

    Returns:
        Tuple of (environment, agent_ids, observations)
    """
    print("\n" + "=" * 80)
    print("MARL ENVIRONMENT INITIALIZATION TEST")
    print("=" * 80)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Initialize environment
    print("\nStep 1: Creating MARL Environment")
    print("-" * 50)

    # Set objective position
    objective_pos = (60, 12)

    # Create environment with higher debug level
    env = setup_environment(width=70, height=60, debug_level=1,
                            marl_wrapper=True, objective_pos=objective_pos)

    print(f"Environment created: {type(env).__name__}")
    print(f"Dimensions: {env.width} x {env.height}")
    print(f"Objective Position: {objective_pos}")

    # Step 2: Create platoon and enemies
    print("\nStep 2: Creating Units (Platoon and Enemies)")
    print("-" * 50)

    # Create platoon
    platoon_id = US_IN_create_platoon(env, plt_num=1, start_position=(2, 30))
    print(f"Platoon created with ID: {platoon_id}")

    # Create enemy targets
    enemy_ids = create_enemy_targets(env)
    print(f"Created {len(enemy_ids)} enemy targets")

    # Step 3: Initialize consistent agent mapping
    print("\nStep 3: Initializing Consistent Agent Mapping")
    print("-" * 50)

    # Ensure the mapping is initialized
    if hasattr(env.agent_manager, 'initialize_agent_role_mapping'):
        env.agent_manager.initialize_agent_role_mapping()
        print("Initialized agent role mapping system")

        # Print the role-to-agent mapping
        if hasattr(env.agent_manager, 'role_to_agent_id'):
            print("\nRole-to-Agent ID Mapping:")
            for role, agent_id in env.agent_manager.role_to_agent_id.items():
                print(f"  {role} → Agent {agent_id}")
    else:
        print("Warning: agent_manager doesn't have initialize_agent_role_mapping method")

    # Step 4: Map current units to consistent agent IDs
    print("\nStep 4: Mapping Current Units to Consistent Agent IDs")
    print("-" * 50)

    # Map the current unit IDs to consistent agent IDs
    if hasattr(env.agent_manager, 'map_current_units_to_agent_ids'):
        agent_ids = env.agent_manager.map_current_units_to_agent_ids(platoon_id)
        print(f"Mapped {len(agent_ids)} agents using consistent ID system")

        # Print the unit-to-agent mapping
        if hasattr(env.agent_manager, 'unit_id_to_agent_id'):
            print("\nUnit-to-Agent ID Mapping:")
            for unit_id, agent_id in env.agent_manager.unit_id_to_agent_id.items():
                unit_string = env.get_unit_property(unit_id, 'string_id', str(unit_id))
                print(f"  Unit {unit_id} ({unit_string}) → Agent {agent_id}")
    else:
        print("Warning: agent_manager doesn't have map_current_units_to_agent_ids method")
        # Fallback to original agent identification
        agent_ids = env.agent_manager.identify_agents_from_platoon(platoon_id)
        print(f"Identified {len(agent_ids)} agents using original method")

    # Print agent details
    if agent_ids:
        print("\nAgent details:")
        for i, agent_id in enumerate(agent_ids[:5]):  # Show first 5 agents
            # Get current unit ID for this agent
            current_unit_id = None
            if hasattr(env.agent_manager, 'get_current_unit_id'):
                current_unit_id = env.agent_manager.get_current_unit_id(agent_id)

            if current_unit_id:
                agent_type = env.agent_manager.agent_types.get(agent_id, "Unknown")
                string_id = env.get_unit_property(current_unit_id, 'string_id')
                role_value = env.get_unit_property(current_unit_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
                print(
                    f"  Agent {agent_id}: Unit ID={current_unit_id}, Type={agent_type}, Role={role_name}, String ID={string_id}")
            else:
                print(f"  Agent {agent_id}: No current unit mapping")

        if len(agent_ids) > 5:
            print(f"  ... and {len(agent_ids) - 5} more agents")

    # Print detailed platoon state for verification
    print("\nPrinting detailed platoon state for verification:")
    print_detailed_platoon_state(env, platoon_id)

    # Step 5: Generate observations for agents - with tactical features
    print("\nStep 5: Generating Enhanced Observations for Agents")
    print("-" * 50)

    observations = {}
    for agent_id in agent_ids:
        if hasattr(env, '_get_observation_for_agent'):
            # Get the current unit ID for the agent
            current_unit_id = agent_id
            if hasattr(env.agent_manager, 'get_current_unit_id'):
                current_unit_id = env.agent_manager.get_current_unit_id(agent_id) or agent_id

            # Get observation
            observations[agent_id] = env._get_observation_for_agent(current_unit_id)
        else:
            # Create minimal observation if method not available
            observations[agent_id] = {
                'agent_state': {
                    'position': np.array(env.get_unit_position(agent_id), dtype=np.int32),
                    'health': np.array([env.get_unit_property(agent_id, 'health', 100)], dtype=np.float32)
                }
            }

    print(f"Generated observations for {len(observations)} agents")

    # Initialize first_agent variable
    first_agent = None

    # Print observation example if available
    if observations and agent_ids:
        first_agent = agent_ids[0]
        print("\nSample observation structure for first agent:")

        # Check specifically for tactical features
        if 'tactical_features' in observations[first_agent]:
            print("  * Using new tactical features format *")
            features = observations[first_agent]['tactical_features']
            # Get length instead of shape to avoid type error
            if isinstance(features, np.ndarray):
                print(f"  tactical_features: ndarray with shape {features.shape}")
            else:
                print(f"  tactical_features: {type(features).__name__} with length {len(features)}")

            # Define expected feature names based on implementation
            feature_names = [
                'Enemy Count', 'Friendly Count',
                'Forest Count', 'Building Count',
                'Mean Elevation', 'Has Cover',
                'Enemy Center X', 'Enemy Center Y'
            ]

            # Print feature values
            for i, name in enumerate(feature_names):
                if i < len(features):
                    print(f"    - {name}: {features[i]}")

        # Check for objective info
        if 'objective_info' in observations[first_agent]:
            print("  * Enhanced objective information available *")
            obj_info = observations[first_agent]['objective_info']
            for key, value in obj_info.items():
                if isinstance(value, np.ndarray):
                    print(f"    - {key}: ndarray with shape {value.shape}")
                else:
                    print(f"    - {key}: {type(value).__name__}")

        # Print remaining observation structure
        for key, value in observations[first_agent].items():
            if key not in ['tactical_features', 'objective_info']:
                if isinstance(value, np.ndarray):
                    print(f"  {key}: ndarray with shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} keys")
                else:
                    print(f"  {key}: {type(value).__name__}")

    # Step 6: Create visualizations
    print("\nStep 6: Creating Visualizations")
    print("-" * 50)

    # Create visualizer
    visualizer = MARLVisualizer(env)

    # Visualize environment state
    print("Creating environment state visualization...")
    fig, ax = visualizer.plot_environment_state(
        agent_ids=agent_ids,
        enemy_ids=enemy_ids,
        title="MARL Environment Initial State with Consistent Agent Mapping",
        output_path=os.path.join(output_dir, "environment_state.png")
    )
    plt.close(fig)

    # Visualize agent mapping system
    print("Creating agent mapping visualization...")
    fig = visualizer.visualize_agent_mapping(
        env,
        output_path=os.path.join(output_dir, "agent_mapping.png")
    )
    plt.close(fig)

    # Check that we have valid observations before visualization
    if agent_ids and first_agent is not None and first_agent in observations:
        # Visualize agent observation
        print("Creating agent observation visualization...")
        fig = visualizer.visualize_observation_space(
            observations[first_agent],
            title=f"Agent {first_agent} Enhanced Observation Space",
            output_path=os.path.join(output_dir, "agent_observation.png")
        )
        plt.close(fig)

        # Visualize specific agent with observation overlay
        print("Creating agent-specific visualization with observation overlay...")
        fig, ax = visualizer.plot_environment_state(
            agent_ids=agent_ids,
            enemy_ids=enemy_ids,
            title=f"Agent {first_agent} with Tactical Features Overlay",
            output_path=os.path.join(output_dir, "agent_with_observation.png"),
            highlight_agent=first_agent,
            agent_observations={first_agent: observations[first_agent]}
        )
        plt.close(fig)

    print("\nInitialization test complete!")
    print(f"Visualizations saved to {output_dir}\n")

    return env, agent_ids, observations


def test_agent_identification_and_management(fresh_env=False):
    """
    Enhanced test for agent identification and position-based casualty management with succession.
    Tests the position-based succession system with consistent agent mapping.
    Also tests automatic squad consolidation when exceeding the casualty threshold.

    Args:
        fresh_env: If True, create a new environment instead of using existing one

    Returns:
        Tuple of (environment, agent_ids)
    """
    print("\n" + "=" * 80)
    print("AGENT IDENTIFICATION AND MANAGEMENT TEST")
    print("=" * 80)

    output_dir = "Model_Verification_MARL/agent_management"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a fresh environment if requested
    if fresh_env:
        # Set up a new environment with MARL wrapper
        env = setup_environment(width=70, height=60, debug_level=2,
                                marl_wrapper=True, objective_pos=(60, 12))
    else:
        # Use a pre-existing environment
        env = globals().get('env', None)
        if env is None:
            # No environment exists, create one
            env = setup_environment(width=70, height=60, debug_level=2,
                                    marl_wrapper=True, objective_pos=(60, 12))

    # Create a platoon for testing
    plt_id = US_IN_create_platoon(env, 1, (2, 30))

    # Optional: Set higher debug level to see more information
    env.debug_level = 2

    # Step 1: Initialize agent role mapping
    print("\nStep 1: Initializing Agent Role Mapping")
    print("-" * 50)

    # Initialize the consistent agent mapping
    if hasattr(env.agent_manager, 'initialize_agent_role_mapping'):
        env.agent_manager.initialize_agent_role_mapping()
        print("Initialized agent role mapping system")

        # Print the role-to-agent mapping
        if hasattr(env.agent_manager, 'role_to_agent_id'):
            print("\nRole-to-Agent ID Mapping:")
            for role, agent_id in env.agent_manager.role_to_agent_id.items():
                print(f"  {role} → Agent {agent_id}")
    else:
        print("Warning: agent_manager doesn't have initialize_agent_role_mapping method")

    # Step 2: Map current units to consistent agent IDs
    print("\nStep 2: Mapping Current Units to Consistent Agent IDs")
    print("-" * 50)

    # Map the current unit IDs to consistent agent IDs
    if hasattr(env.agent_manager, 'map_current_units_to_agent_ids'):
        agents = env.agent_manager.map_current_units_to_agent_ids(plt_id)
        print(f"Mapped {len(agents)} agents using consistent ID system")

        # Print the unit-to-agent mapping
        if hasattr(env.agent_manager, 'unit_id_to_agent_id'):
            print("\nUnit-to-Agent ID Mapping:")
            for unit_id, agent_id in env.agent_manager.unit_id_to_agent_id.items():
                unit_string = env.get_unit_property(unit_id, 'string_id', str(unit_id))
                print(f"  Unit {unit_id} ({unit_string}) → Agent {agent_id}")
    else:
        print("Warning: agent_manager doesn't have map_current_units_to_agent_ids method")
        # Fallback to original agent identification
        agents = env.agent_manager.identify_agents_from_platoon(plt_id)
        print(f"Identified {len(agents)} agents using original method")

    # Verify we have expected number of agents (typically 7 in standard platoon)
    print(f"Identified {len(agents)} agents")
    assert len(agents) >= 6, f"Expected at least 6 agents, got {len(agents)}"

    # Print detailed platoon state
    print("\n================================================================================")
    print("DETAILED PLATOON STATE (BEFORE CASUALTIES)")
    print("================================================================================")
    print_detailed_platoon_state(env, plt_id)

    # Create visualizer for images
    visualizer = MARLVisualizer(env)

    # Create initial visualization
    print("\nStep 3: Creating Initial Agent Mapping Visualization")
    print("-" * 50)

    # Get enemy IDs for visualization
    enemy_ids = [unit_id for unit_id in env.state_manager.active_units
                 if env.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY]

    # Visualize initial state
    fig, ax = visualizer.plot_environment_state(
        agent_ids=agents,
        enemy_ids=enemy_ids,
        title="Agent Mapping System - Initial State",
        output_path=os.path.join(output_dir, "agent_mapping_initial.png")
    )
    plt.close(fig)

    # Visualize agent mapping
    fig = visualizer.visualize_agent_mapping(
        env,
        output_path=os.path.join(output_dir, "agent_mapping_after_succession_diagram.png")
    )
    plt.close(fig)

    # Find a squad leader agent for testing
    print("\nStep 4: Simulating Squad Leader Casualty with Succession")
    print("-" * 50)

    squad_leader_id = None
    squad_id = None
    agent_id = None

    # Look for a squad leader with consistent agent ID 1 (1SQD Leader)
    if hasattr(env.agent_manager, 'get_current_unit_id'):
        agent_id = 1  # Consistent ID for 1SQD Leader
        squad_leader_id = env.agent_manager.get_current_unit_id(agent_id)
        if squad_leader_id:
            squad_id = env.get_unit_property(squad_leader_id, 'parent_id')
            print(f"Found squad leader position {squad_leader_id} mapped to agent ID {agent_id}")
            print(f"Squad ID: {squad_id}")

    # Fallback if consistent mapping didn't work
    if not squad_leader_id:
        for agent_id in agents:
            role = env.get_unit_property(agent_id, 'role')
            if role == US_IN_Role.SQUAD_LEADER.value:
                squad_leader_id = agent_id
                squad_id = env.get_unit_property(agent_id, 'parent_id')
                break

    # Verify we found a squad leader
    assert squad_leader_id is not None, "No squad leader found"

    # Store original soldier ID for comparison
    original_soldier_id = env.get_unit_property(squad_leader_id, 'soldier_id')
    initial_health = env.get_unit_property(squad_leader_id, 'health')

    print(f"\nSimulating casualty for Squad Leader position {squad_leader_id} (Soldier ID: {original_soldier_id})")
    print(f"Initial health: {initial_health}")
    print(f"Squad ID: {squad_id}")
    print(f"Agent ID: {agent_id}")

    # Find potential successors for informational purposes
    print("\nPotential successors:")
    for team_id in env.get_unit_children(squad_id):
        if env.get_unit_property(team_id, 'type') == UnitType.INFANTRY_TEAM:
            for member_id in env.get_unit_children(team_id):
                is_leader = env.get_unit_property(member_id, 'is_leader', False)
                if is_leader:
                    team_leader_health = env.get_unit_property(member_id, 'health', 0)
                    soldier_id = env.get_unit_property(member_id, 'soldier_id')
                    position_status = env.get_unit_property(member_id, 'position_status', 'occupied')
                    print(f"  Team Leader Position: {member_id}, Soldier ID: {soldier_id}")
                    print(f"  Health: {team_leader_health}, Status: {position_status}")

    # Set health to 0 to trigger casualty handling
    print("\nSetting health to 0...")
    env.update_unit_property(squad_leader_id, 'health', 0)
    print(f"After setting: Health = {env.get_unit_property(squad_leader_id, 'health')}")

    # Give a moment for succession to complete if needed
    print("\nWaiting for succession processing to complete...")
    time.sleep(1)  # Small pause to allow for any asynchronous processing

    # Check if succession was successful
    new_soldier_id = env.get_unit_property(squad_leader_id, 'soldier_id')
    current_health = env.get_unit_property(squad_leader_id, 'health')
    position_status = env.get_unit_property(squad_leader_id, 'position_status')

    print("\nAfter succession attempt:")
    print(f"  Position ID: {squad_leader_id} (unchanged)")
    print(f"  Original soldier ID: {original_soldier_id}")
    print(f"  New soldier ID: {new_soldier_id}")
    print(f"  Current health: {current_health}")
    print(f"  Position status: {position_status}")

    # Check if succession succeeded - new soldier ID and restored health
    success = (new_soldier_id != original_soldier_id and current_health > 0
               and position_status == 'occupied')

    if success:
        print(
            f"✓ Position-based succession successful: Position {squad_leader_id} now occupied by soldier {new_soldier_id}")
    else:
        print(f"✗ Position-based succession failed or not implemented correctly")
        print(f"  Current soldier: {new_soldier_id}, Health: {current_health}, Status: {position_status}")

    # Check agent mapping - the agent ID should still map to the same position ID
    if agent_id and hasattr(env.agent_manager, 'get_current_unit_id'):
        current_unit_id = env.agent_manager.get_current_unit_id(agent_id)
        if current_unit_id == squad_leader_id:
            print(f"✓ Agent ID {agent_id} still maps to same position ID {squad_leader_id}")
            print(f"  Position now has new soldier {new_soldier_id} (succession successful)")
        else:
            print(f"✗ Agent mapping changed: Agent ID {agent_id} now maps to {current_unit_id}")

    # Visualize after succession
    print("\nStep 5: Creating Visualization After Succession")
    print("-" * 50)

    fig, ax = visualizer.plot_environment_state(
        agent_ids=agents,
        enemy_ids=enemy_ids,
        title="Agent Mapping System - After Leader Succession",
        output_path=os.path.join(output_dir, "agent_mapping_after_succession.png")
    )
    plt.close(fig)

    # Visualize updated agent mapping
    fig = visualizer.visualize_agent_mapping(
        env,
        output_path=os.path.join(output_dir, "agent_mapping_after_succession_diagram.png")
    )
    plt.close(fig)

    # Print updated platoon state
    print("\nPlatoon state after Squad Leader casualty:")
    print("\n================================================================================")
    print("DETAILED PLATOON STATE (AFTER SQUAD LEADER CASUALTY)")
    print("================================================================================")
    print_detailed_platoon_state(env, plt_id)

    # Test heavy casualty scenario with squad consolidation
    print("\nStep 6: Testing Squad Consolidation with Heavy Casualties")
    print("-" * 50)

    # Use 3rd squad (not the one used in previous test)
    consolidation_squad_id = None
    for unit_id in env.get_unit_children(plt_id):
        string_id = env.get_unit_property(unit_id, 'string_id', '')
        if '3SQD' in string_id:
            consolidation_squad_id = unit_id
            break

    if not consolidation_squad_id:
        print("Could not find 3SQD for testing, falling back to 2SQD")
        # Fall back to second squad if third not found
        for unit_id in env.get_unit_children(plt_id):
            string_id = env.get_unit_property(unit_id, 'string_id', '')
            if '2SQD' in string_id:
                consolidation_squad_id = unit_id
                break

    # Find the agent ID for this squad's leader
    squad_agent_id = None
    for squad_num, agent_id in [(1, 1), (2, 2), (3, 3)]:
        squad_string = f"{squad_num}SQD"
        sqd_string_id = env.get_unit_property(consolidation_squad_id, 'string_id', '')
        if squad_string in sqd_string_id and hasattr(env.agent_manager, 'get_current_unit_id'):
            squad_agent_id = agent_id
            print(f"Found squad {squad_string} with agent ID {squad_agent_id}")
            break

    # Find the squad leader to use for consolidation
    squad_leader_id = None
    for child_id in env.get_unit_children(consolidation_squad_id):
        is_leader = env.get_unit_property(child_id, 'is_leader', False)
        role = env.get_unit_property(child_id, 'role', None)
        if is_leader and role == US_IN_Role.SQUAD_LEADER.value:
            squad_leader_id = child_id
            break

    assert squad_leader_id is not None, f"No squad leader found for squad {consolidation_squad_id}"
    squad_string = env.get_unit_property(consolidation_squad_id, 'string_id', str(consolidation_squad_id))

    print(f"\nSimulating heavy casualties for Squad {squad_string} (ID: {consolidation_squad_id})")
    print(f"Squad leader position ID: {squad_leader_id}")
    if squad_agent_id:
        print(f"Squad leader agent ID: {squad_agent_id}")

    # Find all positions in the squad to set as casualties
    casualty_positions = []
    total_positions = []

    # Include squad leader
    casualty_positions.append(squad_leader_id)
    total_positions.append(squad_leader_id)

    # Find and collect ALL positions in the squad - not just the squad leader!
    for team_id in env.get_unit_children(consolidation_squad_id):
        # Only process infantry teams
        if hasattr(env.get_unit_property(team_id, 'type'), 'value') and env.get_unit_property(team_id,
                                                                                              'type').value == UnitType.INFANTRY_TEAM.value:
            team_string = env.get_unit_property(team_id, 'string_id', '')
            print(f"  Found team: {team_string} (ID: {team_id})")

            # Add all positions from this team to the total
            for pos_id in env.get_unit_children(team_id):
                role_value = env.get_unit_property(pos_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
                total_positions.append(pos_id)
                print(f"    Added position: {pos_id} ({role_name})")

                # For Alpha team, keep the team leader alive
                # For Bravo team, kill everyone except one rifleman
                # This ensures we have enough casualties but still have someone to consolidate around
                if (team_string == "1PLT-3SQD-ATM" and not env.get_unit_property(pos_id, 'is_leader', False)) or \
                        (team_string == "1PLT-3SQD-BTM" and (env.get_unit_property(pos_id, 'is_leader', False) or
                                                             'AUTO_RIFLEMAN' in role_name or 'GRENADIER' in role_name)):
                    casualty_positions.append(pos_id)
                    print(f"    Adding position {pos_id} to casualties")

    # Ensure we have enough casualties to trigger consolidation (need at least 5)
    casualty_threshold = 5
    if len(casualty_positions) < casualty_threshold:
        # Add more casualties if needed
        for pos_id in total_positions:
            if pos_id not in casualty_positions and len(casualty_positions) < casualty_threshold:
                casualty_positions.append(pos_id)
                print(f"    Adding additional position {pos_id} to casualties to meet threshold")

    print(f"Setting casualties for {len(casualty_positions)} positions out of {len(total_positions)} total positions")
    print(f"This should exceed the casualty threshold of {casualty_threshold}")

    if len(casualty_positions) >= casualty_threshold:
        print(
            f"✓ Found {len(casualty_positions)} positions to set as casualties, which meets the threshold of {casualty_threshold}")
    else:
        print(
            f"✗ Only found {len(casualty_positions)} positions to set as casualties, which does NOT meet the threshold of {casualty_threshold}")

    # Set health to 0 for multiple positions to trigger consolidation
    print("\nSetting health to 0 for multiple positions to trigger consolidation...")
    for position_id in casualty_positions:
        print(f"  Setting position {position_id} health to 0")
        env.update_unit_property(position_id, 'health', 0)

        # Verify position is now marked as a casualty
        new_status = env.get_unit_property(position_id, 'position_status', 'unknown')
        if new_status == 'casualty':
            print(f"  ✓ Position {position_id} correctly marked as 'casualty'")
        else:
            print(f"  ✗ Position {position_id} has unexpected status '{new_status}'")

    # Check if squad needs consolidation and let the environment handle it automatically
    print("\nTriggering agent manager update after casualties...")
    if hasattr(env.agent_manager, 'update_after_casualties'):
        env.agent_manager.update_after_casualties([consolidation_squad_id])
    else:
        print("Warning: agent_manager doesn't have update_after_casualties method")

    # Give some time for consolidation to complete
    time.sleep(1)

    # Visualize after consolidation
    print("\nStep 7: Creating Visualization After Consolidation")
    print("-" * 50)

    fig, ax = visualizer.plot_environment_state(
        agent_ids=agents,
        enemy_ids=enemy_ids,
        title="Agent Mapping System - After Squad Consolidation",
        output_path=os.path.join(output_dir, "agent_mapping_after_consolidation.png")
    )
    plt.close(fig)

    # Visualize updated agent mapping
    fig = visualizer.visualize_agent_mapping(
        env,
        output_path=os.path.join(output_dir, "agent_mapping_after_consolidation_diagram.png")
    )
    plt.close(fig)

    # Check agent mapping after consolidation
    if squad_agent_id and hasattr(env.agent_manager, 'get_current_unit_id'):
        print("\nChecking agent mapping after consolidation:")
        current_unit_id = env.agent_manager.get_current_unit_id(squad_agent_id)
        if current_unit_id:
            print(f"Agent ID {squad_agent_id} now maps to unit ID {current_unit_id}")

            # Check if it's a team leader now
            unit_string = env.get_unit_property(current_unit_id, 'string_id', '')
            role_value = env.get_unit_property(current_unit_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
            parent_id = env.get_unit_property(current_unit_id, 'parent_id')
            parent_string = env.get_unit_property(parent_id, 'string_id', '') if parent_id else "None"

            print(f"  Unit string: {unit_string}")
            print(f"  Role: {role_name}")
            print(f"  Parent: {parent_string} (ID: {parent_id})")

            # Check if parent is now a team instead of a squad
            if parent_id:
                parent_type = env.get_unit_property(parent_id, 'type')
                parent_type_str = str(parent_type)
                if hasattr(parent_type, 'name'):
                    parent_type_str = parent_type.name

                if "INFANTRY_TEAM" in parent_type_str:
                    print(f"✓ Parent is now a team (consolidation successful)")
                else:
                    print(f"✗ Parent is not a team (type: {parent_type_str})")
        else:
            print(f"✗ Agent ID {squad_agent_id} has no current unit mapping")

    # Print final platoon state
    print("\nPlatoon state after Squad consolidation:")
    print("\n================================================================================")
    print("DETAILED PLATOON STATE (AFTER SQUAD CONSOLIDATION)")
    print("================================================================================")
    print_detailed_platoon_state(env, plt_id)

    # Debug agent structure using our improved function
    if hasattr(env.agent_manager, 'debug_agent_structure'):
        print("\nAgent structure after consolidation:")
        env.agent_manager.debug_agent_structure()

    # Final assertions for test success
    print("\nAgent identification and position-based succession test completed!")

    # Return the environment for potential reuse
    return env, agents


def main():
    # Create output directories
    base_dir = create_output_directories()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MARL verification tests')
    parser.add_argument('--test', choices=['init', 'agents', 'all'],
                        default='all', help='Test type to run')
    args = parser.parse_args()

    # Define which tests to run
    tests_to_run = []
    if args.test == 'all':
        tests_to_run = ['init', 'agents']
    else:
        tests_to_run = [args.test]

    # Track test results and environment/agents for reuse
    results = {}
    env = None
    agent_ids = None

    try:
        # Run each test
        for test_name in tests_to_run:
            print(f"\nRunning {test_name.capitalize()} Test...")

            if test_name == 'init':
                # Always create a fresh environment for initialization test
                env, agent_ids, observations = test_environment_initialization()
                results[test_name] = (env, agent_ids, observations)

            elif test_name == 'agents':
                # Create fresh environment for agent test
                env, agent_ids = test_agent_identification_and_management(fresh_env=True)
                results[test_name] = (env, agent_ids)

            print(f"{test_name.capitalize()} test completed successfully!")

        # Print summary
        print("\n" + "=" * 40)
        print("TEST EXECUTION SUMMARY")
        print("=" * 40)

        for test_name in ['init', 'agents']:
            status = "✓ PASSED" if test_name in results else "✗ SKIPPED"
            print(f"{test_name.capitalize()} Test: {status}")

        return 0

    except Exception as e:
        print(f"\nError during test execution: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
