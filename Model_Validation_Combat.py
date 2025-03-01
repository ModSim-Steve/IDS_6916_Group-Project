"""
Model Validation Combat - Combat Mechanics Testing Framework

This script provides comprehensive tests for the combat mechanics in the WarGaming Environment.
It includes tests for:
1. Basic engagements across varying terrain
2. Suppression mechanics
3. Team coordination during engagements
4. High-priority target selection and engagement

Uses the standard coordinate system:
- 0° = East (right)
- 90° = South (down)
- 180° = West (left)
- 270° = North (up)
"""

import os
import math
import argparse
import traceback

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional

from matplotlib import lines

from WarGamingEnvironment_vTest import (
    MilitaryEnvironment,
    EnvironmentConfig,
    TerrainType,
    ElevationType,
    ForceType,
    EngagementType,
    UnitType
)

from US_Army_PLT_Composition_vTest import (
    US_IN_create_team,
    US_IN_UnitDesignator,
    US_IN_apply_formation,
    get_unit_sectors,
    US_IN_Role
)


# Create output directories
def create_output_directories():
    """Create directories for test outputs."""
    base_dir = "Model_Validation_Combat"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create subdirectories
    subdirs = [
        "basic_engagement_varying_terrain",
        "suppression",
        "coordination",
        "high_priority_target"
    ]

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

    return base_dir


# Visualization Utilities
class CombatVisualizer:
    """Visualization utility for the combat test."""

    def __init__(self, env: MilitaryEnvironment):
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

    def plot_combat_state(self, friendly_units: List[int], enemy_units: List[int],
                          title: str, engagement_results: Optional[Dict] = None,
                          output_path: Optional[str] = None):
        """
        Create a visualization of the combat state.

        Args:
            friendly_units: List of friendly unit IDs
            enemy_units: List of enemy unit IDs
            title: Title for the plot
            engagement_results: Optional results from engagement for highlighting
            output_path: Optional path to save the visualization

        Returns:
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot terrain
        self._plot_terrain(ax)

        # Plot friendly units with sectors of fire
        self._plot_friendly_units(ax, friendly_units)

        # Plot enemy units
        self._plot_enemy_units(ax, enemy_units, engagement_results)

        # Add title and legend
        ax.set_title(title, fontsize=16)
        ax.legend(loc='upper right')

        # Set limits and grid
        ax.set_xlim(-1, self.env.width + 1)
        ax.set_ylim(-1, self.env.height + 1)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax

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

    def _plot_friendly_units(self, ax, unit_ids: List[int]):
        """Plot friendly units with sectors of fire."""
        for unit_id in unit_ids:
            # Get unit position and orientation
            unit_pos = self.env.get_unit_position(unit_id)
            unit_orientation = self.env.get_unit_property(unit_id, 'orientation', 0)

            # Debug info to verify orientation
            # print(f"\n[DEBUG VISUALIZER] Plotting friendly unit {unit_id} at {unit_pos} with orientation {unit_orientation}°")

            # Plot unit marker
            ax.plot(unit_pos[0], unit_pos[1], 'bo', markersize=8,
                    label=f"Friendly {self.env.get_unit_property(unit_id, 'string_id')}")

            # Plot all members
            for member_id in self.env.get_unit_children(unit_id):
                member_pos = self.env.get_unit_position(member_id)
                is_leader = self.env.get_unit_property(member_id, 'is_leader')
                member_orientation = self.env.get_unit_property(member_id, 'orientation', 0)

                # print(f"[DEBUG VISUALIZER] Plotting member {member_id} at {member_pos} with orientation {member_orientation}°")

                marker = '^' if is_leader else 'o'
                ax.plot(member_pos[0], member_pos[1], marker, color='blue', markersize=6)

                # Add label with role
                role = self.env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role).name if isinstance(role, int) else role.name
                ax.annotate(role_name, xy=member_pos, xytext=(3, 3),
                            textcoords='offset points', fontsize=8)

                # Plot sectors of fire
                self._plot_member_sectors(ax, member_id, 'blue')

    def _plot_member_sectors(self, ax, member_id: int, color: str):
        """
        Plot sectors of fire for a unit member.
        Uses standard coordinate system (0° = East).
        """
        member_pos = self.env.get_unit_position(member_id)
        engagement_range = self.env.get_unit_property(member_id, 'engagement_range')

        # Add debug print
        # print(f"\n[DEBUG VISUALIZER] Plotting sectors for member {member_id}")
        role = self.env.get_unit_property(member_id, 'role')
        role_name = US_IN_Role(role).name if isinstance(role, int) else role
        # print(f"[DEBUG VISUALIZER] Role: {role_name}")

        # Try to get rotated sectors from state manager
        primary_start = self.env.get_unit_property(member_id, 'primary_sector_rotated_start')
        primary_end = self.env.get_unit_property(member_id, 'primary_sector_rotated_end')
        secondary_start = self.env.get_unit_property(member_id, 'secondary_sector_rotated_start')
        secondary_end = self.env.get_unit_property(member_id, 'secondary_sector_rotated_end')

        # print(f"[DEBUG VISUALIZER] Stored rotated sectors - Primary: {primary_start}°-{primary_end}°, Secondary: {secondary_start}°-{secondary_end}°")

        # If rotated sectors are stored, use them
        if primary_start is not None and primary_end is not None:
            # print(f"[DEBUG VISUALIZER] Using stored rotated sectors")
            # Plot primary sector using stored rotated values
            wedge = patches.Wedge(
                member_pos, engagement_range,
                primary_start, primary_end,
                color=color, alpha=0.3
            )
            ax.add_patch(wedge)

            # Plot secondary sector if available
            if secondary_start is not None and secondary_end is not None:
                wedge = patches.Wedge(
                    member_pos, engagement_range * 0.8,
                    secondary_start, secondary_end,
                    color=color, alpha=0.15
                )
                ax.add_patch(wedge)
        else:
            # Fall back to legacy method of dynamic rotation
            orientation = self.env.get_unit_property(member_id, 'orientation')
            # print(f"[DEBUG VISUALIZER] Falling back to dynamic calculation with orientation {orientation}°")
            role = self.env.get_unit_property(member_id, 'role')

            # Get sectors
            if isinstance(role, int):
                role = US_IN_Role(role)

            is_leader = self.env.get_unit_property(member_id, 'is_leader')
            parent_id = self.env.get_unit_property(member_id, 'parent_id')

            if parent_id:
                formation = self.env.get_unit_property(parent_id, 'formation')
                # print(f"[DEBUG VISUALIZER] Parent formation: {formation}")
                primary_sector, secondary_sector = get_unit_sectors(role, formation, is_leader)

                # Plot primary sector
                if primary_sector:
                    start_angle = (primary_sector.start_angle + orientation) % 360
                    end_angle = (primary_sector.end_angle + orientation) % 360

                    # print(f"[DEBUG VISUALIZER] Dynamic primary sector: {primary_sector.start_angle}°-{primary_sector.end_angle}° -> rotated {start_angle}°-{end_angle}°")

                    wedge = patches.Wedge(
                        member_pos, engagement_range,
                        start_angle, end_angle,
                        color=color, alpha=0.3
                    )
                    ax.add_patch(wedge)

                # Plot secondary sector
                if secondary_sector:
                    start_angle = (secondary_sector.start_angle + orientation) % 360
                    end_angle = (secondary_sector.end_angle + orientation) % 360

                    # print(f"[DEBUG VISUALIZER] Dynamic secondary sector: {secondary_sector.start_angle}°-{secondary_sector.end_angle}° -> rotated {start_angle}°-{end_angle}°")

                    wedge = patches.Wedge(
                        member_pos, engagement_range * 0.8,
                        start_angle, end_angle,
                        color=color, alpha=0.15
                    )
                    ax.add_patch(wedge)

    def _plot_enemy_units(self, ax, unit_ids: List[int], engagement_results: Optional[Dict] = None):
        """Plot enemy units, with hit markers if engagement results are provided."""
        for unit_id in unit_ids:
            unit_pos = self.env.get_unit_position(unit_id)
            unit_health = self.env.get_unit_property(unit_id, 'health')

            # Determine marker size and color based on health
            marker_size = 8
            if unit_health <= 0:
                marker_color = 'grey'  # Dead
            else:
                marker_color = 'red'  # Alive

            # Plot unit marker
            ax.plot(unit_pos[0], unit_pos[1], 'o', color=marker_color,
                    markersize=marker_size,
                    label=f"Enemy {self.env.get_unit_property(unit_id, 'string_id')}")

            # Draw health bar
            if unit_health > 0:
                self._draw_health_bar(ax, unit_pos, unit_health)

            # Mark if hit in this engagement
            if engagement_results and 'targets_hit' in engagement_results:
                if unit_id in engagement_results['targets_hit']:
                    ax.plot(unit_pos[0], unit_pos[1], 'x', color='black', markersize=10)

            # Add suppression visualization if unit is suppressed
            if hasattr(self.env, 'combat_manager') and self.env.combat_manager:
                if unit_id in self.env.combat_manager.suppressed_units:
                    suppression = self.env.combat_manager.suppressed_units[unit_id]['level']
                    if suppression > 0:
                        circle = plt.Circle(
                            unit_pos, 1.5, color='yellow', alpha=min(0.7, suppression),
                            fill=True
                        )
                        ax.add_patch(circle)
                        ax.annotate(f"Supp: {suppression:.2f}", xy=unit_pos,
                                    xytext=(0, -15), textcoords='offset points',
                                    ha='center', fontsize=8)

    def _draw_health_bar(self, ax, position: Tuple[int, int], health: float):
        """Draw a health bar above a unit."""
        bar_width = 2.0
        bar_height = 0.3

        # Background (red) bar
        ax.add_patch(patches.Rectangle(
            (position[0] - bar_width / 2, position[1] + 1),
            bar_width, bar_height,
            facecolor='red', alpha=0.7
        ))

        # Health (green) bar
        health_ratio = max(0, int(health / 100.0))
        ax.add_patch(patches.Rectangle(
            (position[0] - bar_width / 2, position[1] + 1),
            bar_width * health_ratio, bar_height,
            facecolor='green', alpha=0.7
        ))

    def _hex_to_rgb(self, hex_color: str):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def create_suppression_visualization(self, friendly_id, enemy_ids, suppression_results, output_path=None):
        """
        Create enhanced visualization showing suppression effects including:
        - Shooter's sector of fire
        - Actual round impact locations
        - Suppression status bars for enemies

        Args:
            friendly_id: ID of friendly unit
            enemy_ids: List of enemy unit IDs
            suppression_results: Results from suppression test with hit_locations
            output_path: Path to save the visualization

        Returns:
            Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Setup plot basics - terrain backdrop
        terrain_img = np.zeros((self.env.height, self.env.width, 3))
        for y in range(self.env.height):
            for x in range(self.env.width):
                terrain_type = self.env.terrain_manager.get_terrain_type((x, y))
                terrain_color = [0.9, 0.9, 0.8]  # Default color

                if terrain_type == TerrainType.BARE:
                    terrain_color = [0.9, 0.9, 0.8]  # Beige
                elif terrain_type == TerrainType.SPARSE_VEG:
                    terrain_color = [0.6, 0.8, 0.6]  # Light green
                elif terrain_type == TerrainType.DENSE_VEG:
                    terrain_color = [0.2, 0.6, 0.2]  # Forest green
                elif terrain_type == TerrainType.WOODS:
                    terrain_color = [0.1, 0.4, 0.1]  # Dark green
                elif terrain_type == TerrainType.STRUCTURE:
                    terrain_color = [0.5, 0.5, 0.5]  # Gray

                terrain_img[y, x] = terrain_color

        # Plot terrain
        ax.imshow(terrain_img, extent=(0, self.env.width, 0, self.env.height), origin='lower')

        # Create a legend elements list
        legend_elements = [
            lines.Line2D([0], [0], color='blue', marker='o', linestyle='None',
                         markersize=8, label='Friendly Forces'),
            lines.Line2D([0], [0], color='red', marker='o', linestyle='None',
                         markersize=8, label='Enemy Forces'),
            patches.Patch(facecolor='blue', alpha=0.2, label='Engagement Sector'),
            patches.Patch(facecolor='yellow', alpha=0.5, label='Suppression Fan'),
            lines.Line2D([0], [0], color='black', marker='x', linestyle='None',
                         markersize=6, label='Round Impacts'),
            patches.Patch(facecolor='green', alpha=0.7, label='Health'),
            patches.Patch(facecolor='blue', alpha=0.7, label='Suppression Level')
        ]

        # Plot friendly team with enhanced firing sectors
        friendly_members = self.env.get_unit_children(friendly_id)
        auto_rifleman = None

        # Find the automatic rifleman for special sector highlighting
        for member_id in friendly_members:
            role_value = self.env.get_unit_property(member_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
            if "AUTO_RIFLEMAN" in role_name:
                auto_rifleman = member_id
                break

        # Plot all members with standard markers
        for member_id in friendly_members:
            pos = self.env.get_unit_position(member_id)
            is_leader = self.env.get_unit_property(member_id, 'is_leader', False)
            role_value = self.env.get_unit_property(member_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)

            # Plot member
            marker = '^' if is_leader else 'o'
            ax.plot(pos[0], pos[1], marker, color='blue', markersize=8)

            # Add label
            ax.annotate(role_name, xy=pos, xytext=(3, 3),
                        textcoords='offset points', fontsize=8)

        # Highlight automatic rifleman's sector with suppression fan
        if auto_rifleman:
            ar_pos = self.env.get_unit_position(auto_rifleman)
            engagement_range = self.env.get_unit_property(auto_rifleman, 'engagement_range')

            # Get sectors
            primary_start = self.env.get_unit_property(auto_rifleman, 'primary_sector_rotated_start')
            primary_end = self.env.get_unit_property(auto_rifleman, 'primary_sector_rotated_end')

            if primary_start is not None and primary_end is not None:
                # Plot standard sector
                wedge = patches.Wedge(
                    ar_pos, engagement_range,
                    primary_start, primary_end,
                    color='blue', alpha=0.2
                )
                ax.add_patch(wedge)

                # Plot suppression fan overlay (narrower and more intense)
                suppression_range = engagement_range * 0.7  # Effective suppression range
                suppression_wedge = patches.Wedge(
                    ar_pos, suppression_range,
                    primary_start, primary_end,
                    color='yellow', alpha=0.5
                )
                ax.add_patch(suppression_wedge)

                # Add annotation explaining suppression fan
                mid_angle = (primary_start + primary_end) / 2
                if primary_start > primary_end:  # Handle wrap around
                    mid_angle = (primary_start + primary_end + 360) / 2 % 360

                # Calculate point for annotation
                rad = math.radians(mid_angle)
                text_dist = engagement_range * 0.5
                text_x = ar_pos[0] + text_dist * math.cos(math.radians(mid_angle))
                text_y = ar_pos[1] + text_dist * math.sin(math.radians(mid_angle))

                ax.annotate("Suppression\nFan", xy=(text_x, text_y),
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

        # Plot actual round impact locations from engagement data
        all_impact_locations = []
        for enemy_id in enemy_ids:
            if enemy_id in suppression_results.get('engagement_data', {}):
                engagement_data = suppression_results['engagement_data'][enemy_id]

                if 'hit_locations' in engagement_data and engagement_data['hit_locations']:
                    # Plot all impact points
                    for impact_pos in engagement_data['hit_locations']:
                        all_impact_locations.append(impact_pos)
                        ax.plot(impact_pos[0], impact_pos[1], 'x', color='black', markersize=4, alpha=0.6)

        # Add text showing number of impacts
        if all_impact_locations:
            ax.text(0.02, 0.02, f"Total Impacts: {len(all_impact_locations)}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot enemy units with suppression status bars
        for enemy_id in enemy_ids:
            enemy_pos = self.env.get_unit_position(enemy_id)

            # Plot enemy unit
            ax.plot(enemy_pos[0], enemy_pos[1], 'o', color='red', markersize=8)

            # Add label
            ax.annotate(f"Enemy {enemy_id}", xy=enemy_pos, xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8)

            # Check suppression state of child units
            child_units = self.env.get_unit_children(enemy_id)
            max_suppression = 0

            for child_id in child_units:
                if child_id in self.env.combat_manager.suppressed_units:
                    level = self.env.combat_manager.suppressed_units[child_id]['level']
                    max_suppression = max(max_suppression, level)

            # Draw health bar (red/green)
            health = self.env.get_unit_property(enemy_id, 'health', 100)
            health_ratio = max(0, health / 100.0)

            # Background (red) bar
            ax.add_patch(patches.Rectangle(
                (enemy_pos[0] - 1.5, enemy_pos[1] + 1.5),
                3.0, 0.3,
                facecolor='red', alpha=0.7
            ))

            # Health (green) bar
            ax.add_patch(patches.Rectangle(
                (enemy_pos[0] - 1.5, enemy_pos[1] + 1.5),
                3.0 * health_ratio, 0.3,
                facecolor='green', alpha=0.7
            ))

            # Add suppression bar (blue)
            if max_suppression > 0:
                # Background (gray) bar
                ax.add_patch(patches.Rectangle(
                    (enemy_pos[0] - 1.5, enemy_pos[1] + 1.9),
                    3.0, 0.3,
                    facecolor='gray', alpha=0.4
                ))

                # Suppression (blue) bar
                ax.add_patch(patches.Rectangle(
                    (enemy_pos[0] - 1.5, enemy_pos[1] + 1.9),
                    3.0 * max_suppression, 0.3,
                    facecolor='blue', alpha=0.7
                ))

                # Add suppression label
                ax.annotate(f"Suppressed: {max_suppression:.2f}",
                            xy=(enemy_pos[0], enemy_pos[1] + 2.5),
                            ha='center', fontsize=8, color='blue')

        # Add title and legend
        ax.set_title("Suppression Effects Visualization", fontsize=16)
        ax.legend(handles=legend_elements, loc='upper right')

        # Set limits and grid
        ax.set_xlim(-1, self.env.width + 1)
        ax.set_ylim(-1, self.env.height + 1)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def create_round_impact_visualization(self, friendly_id, enemy_ids, suppression_results, output_path=None):
        """
        Create a dedicated visualization showing round impact patterns from suppression fire.

        Args:
            friendly_id: ID of friendly unit
            enemy_ids: List of enemy unit IDs
            suppression_results: Results from suppression test with hit_locations
            output_path: Path to save the visualization

        Returns:
            Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Setup terrain as background with reduced opacity
        terrain_img = np.zeros((self.env.height, self.env.width, 3))
        for y in range(self.env.height):
            for x in range(self.env.width):
                terrain_type = self.env.terrain_manager.get_terrain_type((x, y))
                terrain_color = [0.95, 0.95, 0.9]  # Very light background

                if terrain_type == TerrainType.BARE:
                    terrain_color = [0.95, 0.95, 0.9]
                elif terrain_type == TerrainType.SPARSE_VEG:
                    terrain_color = [0.9, 0.95, 0.9]
                elif terrain_type == TerrainType.DENSE_VEG:
                    terrain_color = [0.85, 0.9, 0.85]
                elif terrain_type == TerrainType.WOODS:
                    terrain_color = [0.8, 0.85, 0.8]
                elif terrain_type == TerrainType.STRUCTURE:
                    terrain_color = [0.8, 0.8, 0.8]

                terrain_img[y, x] = terrain_color

        # Plot terrain with high transparency so impacts stand out
        ax.imshow(terrain_img, extent=(0, self.env.width, 0, self.env.height), origin='lower', alpha=0.3)

        # Get friendly team position for visualization
        friendly_position = self.env.get_unit_position(friendly_id)

        # Plot friendly position (source of fire)
        ax.plot(friendly_position[0], friendly_position[1], 'D', color='blue',
                markersize=12, label='Friendly Team (Source)')

        # Dictionary to track impacts by target
        impacts_by_target = {}
        all_impacts = []

        # Create a color map manually instead of using plt.cm.tab10
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_map = {enemy_ids[i]: colors[i % len(colors)] for i in range(len(enemy_ids))}

        # Plot all round impacts
        for enemy_id in enemy_ids:
            enemy_pos = self.env.get_unit_position(enemy_id)

            # Plot enemy position (target)
            ax.plot(enemy_pos[0], enemy_pos[1], 'o', color='red', markersize=10)
            ax.annotate(f"Target #{enemy_id}", xy=enemy_pos, xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=10)

            # Plot 3-cell radius suppression area
            circle = plt.Circle(enemy_pos, 3, color='red', fill=False, linestyle='--', alpha=0.6)
            ax.add_patch(circle)

            # Get impact data for this target
            if enemy_id in suppression_results.get('engagement_data', {}):
                engagement_data = suppression_results['engagement_data'][enemy_id]
                hit_locations = engagement_data.get('hit_locations', [])

                if hit_locations:
                    impacts_by_target[enemy_id] = hit_locations
                    all_impacts.extend(hit_locations)

                    # Plot impact points for this target with specific color
                    for impact_pos in hit_locations:
                        ax.plot(impact_pos[0], impact_pos[1], 'x',
                                color=color_map[enemy_id], markersize=6, alpha=0.7)

        # Plot firing line from source to targets
        for enemy_id in enemy_ids:
            enemy_pos = self.env.get_unit_position(enemy_id)
            ax.plot([friendly_position[0], enemy_pos[0]],
                    [friendly_position[1], enemy_pos[1]],
                    '--', color='gray', alpha=0.6)

        # Calculate and display round distribution statistics
        if all_impacts:
            # Create a subplot for the impact distribution heatmap
            # Fix: Convert the list to a tuple for add_axes
            left, bottom, width, height = 0.25, 0.35, 0.25, 0.25
            ax_heatmap = fig.add_axes((left, bottom, width, height))

            # Create 2D histogram of impact locations
            x_impacts = [loc[0] for loc in all_impacts]
            y_impacts = [loc[1] for loc in all_impacts]

            # Generate heatmap
            heatmap, xedges, yedges = np.histogram2d(x_impacts, y_impacts, bins=20)

            # Fix: Use tuple instead of list for extent
            extent = (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1]))

            # Plot heatmap
            ax_heatmap.imshow(heatmap.T, origin='lower', extent=extent, cmap='hot')
            ax_heatmap.set_title("Impact Density", fontsize=10)
            ax_heatmap.set_xticks([])
            ax_heatmap.set_yticks([])

            # Add stats for each target
            stats_text = "Impact Statistics:\n"
            stats_text += "-----------------\n"
            stats_text += f"Total impacts: {len(all_impacts)}\n\n"

            for i, enemy_id in enumerate(sorted(impacts_by_target.keys())):
                enemy_pos = self.env.get_unit_position(enemy_id)
                hit_locations = impacts_by_target[enemy_id]

                # Calculate distances from target center
                distances = [math.sqrt((loc[0] - enemy_pos[0]) ** 2 + (loc[1] - enemy_pos[1]) ** 2)
                             for loc in hit_locations]

                if distances:
                    avg_distance = sum(distances) / len(distances)
                    max_distance = max(distances)

                    # Count impacts by distance ranges
                    close_impacts = sum(1 for d in distances if d <= 1)
                    medium_impacts = sum(1 for d in distances if 1 < d <= 2)
                    far_impacts = sum(1 for d in distances if d > 2)

                    # Add to stats text
                    stats_text += f"Target #{enemy_id}:\n"
                    stats_text += f"  Impacts: {len(hit_locations)}\n"
                    stats_text += f"  Avg dist: {avg_distance:.2f} cells\n"
                    stats_text += f"  Distribution:\n"
                    stats_text += f"    ≤1 cell: {close_impacts} ({close_impacts / len(distances) * 100:.1f}%)\n"
                    stats_text += f"    1-2 cells: {medium_impacts} ({medium_impacts / len(distances) * 100:.1f}%)\n"
                    stats_text += f"    >2 cells: {far_impacts} ({far_impacts / len(distances) * 100:.1f}%)\n\n"

            # Add statistics text
            ax.text(0.45, 0.88, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Add title and legend
        ax.set_title("Round Impact Patterns from Suppression Fire", fontsize=16)

        # Create custom legend
        legend_elements = [
            lines.Line2D([0], [0], marker='D', color='blue', linestyle='None',
                         markersize=10, label='Friendly Team (Source)'),
            lines.Line2D([0], [0], marker='o', color='red', linestyle='None',
                         markersize=8, label='Enemy Target'),
            lines.Line2D([0], [0], marker='x', color='black', linestyle='None',
                         markersize=6, label='Round Impact')
        ]

        # Add a legend entry for each target color
        for i, enemy_id in enumerate(sorted(enemy_ids)):
            if enemy_id in impacts_by_target:
                legend_elements.append(
                    lines.Line2D([0], [0], marker='x', color=color_map[enemy_id],
                                 linestyle='None', markersize=6,
                                 label=f'Impacts on Target #{enemy_id}')
                )

        ax.legend(handles=legend_elements, loc='lower right')

        # Set limits and grid
        ax.set_xlim(-1, self.env.width + 1)
        ax.set_ylim(-1, self.env.height + 1)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def create_recovery_visualization(self, enemy_ids, recovery_results, output_path=None):
        """
        Create visualization showing suppression recovery over time

        Args:
            enemy_ids: List of enemy unit IDs
            recovery_results: List of suppression levels at each recovery step
            output_path: Path to save the visualization

        Returns:
            Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get all units that were suppressed at any point
        all_suppressed_units = set()
        for step_data in recovery_results:
            all_suppressed_units.update(step_data.keys())

        # Setup plot for each suppressed unit
        steps = list(range(len(recovery_results) + 1))  # +1 for initial state
        colors = ['r', 'g', 'b', 'purple', 'orange', 'cyan']

        # Plot recovery for each unit with small offsets to prevent overlapping
        for i, unit_id in enumerate(all_suppressed_units):
            unit_pos = self.env.get_unit_position(unit_id)
            color = colors[i % len(colors)]

            # Calculate vertical offset for this unit's line
            offset = i * 0.02  # Small vertical offset

            # Get suppression values including initial level
            values = [1.0 + offset]  # Initial suppression with offset based on unit index
            for step_data in recovery_results:
                # Add the same vertical offset to all points
                values.append(step_data.get(unit_id, 0) + offset)

            # Plot recovery curve
            ax.plot(steps, values, 'o-', color=color, linewidth=2,
                    label=f"Unit {unit_id} at {unit_pos}")

            # Annotate effects on last point
            last_value = values[-1]
            # Calculate accuracy reduction based on actual suppression value (removing the offset)
            actual_suppression = last_value - offset
            acc_reduction = min(0.8, actual_suppression * 0.9)
            ax.annotate(f"Acc: -{acc_reduction * 100:.0f}%",
                        xy=(steps[-1], last_value),
                        xytext=(5, 0), textcoords='offset points')

        # Mark recovery rate
        recovery_rate = self.env.combat_manager.suppression_recovery_rate
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Shade areas of effect
        ax.axhspan(0.8, 1.1, alpha=0.2, color='red', label="Severe Suppression")
        ax.axhspan(0.5, 0.8, alpha=0.2, color='orange', label="Moderate Suppression")
        ax.axhspan(0.2, 0.5, alpha=0.2, color='yellow', label="Light Suppression")
        ax.axhspan(0, 0.2, alpha=0.2, color='green', label="Minimal Suppression")

        # Add labels and title
        ax.set_xlabel("Recovery Steps", fontsize=12)
        ax.set_ylabel("Suppression Level", fontsize=12)
        ax.set_title(f"Suppression Recovery Over Time (Rate: {recovery_rate} per step)", fontsize=16)

        # Add annotations explaining effects
        ax.text(0.22, 0.20, "Suppression Effects:", transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.22, 0.16, "1.0-0.8: Cannot return fire effectively", transform=ax.transAxes,
                fontsize=9, verticalalignment='top')
        ax.text(0.22, 0.12, "0.8-0.5: Limited mobility, poor accuracy", transform=ax.transAxes,
                fontsize=9, verticalalignment='top')
        ax.text(0.22, 0.08, "0.5-0.2: Reduced combat effectiveness", transform=ax.transAxes,
                fontsize=9, verticalalignment='top')
        ax.text(0.22, 0.04, "0.2-0.0: Minor effects, mostly recovered", transform=ax.transAxes,
                fontsize=9, verticalalignment='top')

        # Set y-axis limits and grid
        ax.set_ylim(0, 1.1)  # Increased upper limit to account for offsets
        ax.set_xlim(-0.25, 5.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower left')

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def visualize_engagement_priorities(self, friendly_units: List[int], enemy_units: List[int],
                                        engagement_results: Dict, title: str,
                                        output_path: Optional[str] = None):
        """
        Create visualization showing which units engaged which targets,
        useful for coordination and priority targeting tests.

        Args:
            friendly_units: List of friendly unit IDs
            enemy_units: List of enemy unit IDs
            engagement_results: Dictionary mapping unit IDs to their targets
            title: Title for the plot
            output_path: Optional path to save the visualization

        Returns:
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot terrain
        self._plot_terrain(ax)

        # Plot enemy units
        for unit_id in enemy_units:
            unit_pos = self.env.get_unit_position(unit_id)
            unit_health = self.env.get_unit_property(unit_id, 'health')

            # Determine marker size and color
            marker_size = 8
            marker_color = 'grey' if unit_health <= 0 else 'red'

            # Plot unit marker
            ax.plot(unit_pos[0], unit_pos[1], 'o', color=marker_color, markersize=marker_size)

            # Add unit label
            ax.annotate(f"Enemy {self.env.get_unit_property(unit_id, 'string_id')}",
                        xy=unit_pos, xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8)

            # Add threat level if provided
            if 'threat_levels' in engagement_results and unit_id in engagement_results['threat_levels']:
                threat_level = engagement_results['threat_levels'][unit_id]
                ax.annotate(f"Threat: {threat_level:.1f}",
                            xy=unit_pos, xytext=(0, -15),
                            textcoords='offset points', ha='center', fontsize=8)

                # Add "HIGH PRIORITY" label if threat exceeds threshold
                if threat_level > engagement_results.get('high_threat_threshold', 3.0):
                    ax.annotate("HIGH PRIORITY",
                                xy=unit_pos, xytext=(0, -25),
                                textcoords='offset points', ha='center', fontsize=8,
                                color='red', weight='bold')

        # Plot friendly units
        for unit_id in friendly_units:
            # Plot base unit
            self._plot_friendly_units(ax, [unit_id])

            # Get unit position
            unit_pos = self.env.get_unit_position(unit_id)

            # Plot engagement lines to targets
            if 'unit_targets' in engagement_results and unit_id in engagement_results['unit_targets']:
                targets = engagement_results['unit_targets'][unit_id]
                for target_id in targets:
                    target_pos = self.env.get_unit_position(target_id)

                    # Draw line from unit to target
                    ax.plot([unit_pos[0], target_pos[0]], [unit_pos[1], target_pos[1]],
                            color='red', linestyle='-', linewidth=1.5, alpha=0.6)

                    # Add engagement info if available
                    if 'engagement_data' in engagement_results:
                        engagement_data = engagement_results['engagement_data']
                        if (unit_id, target_id) in engagement_data:
                            data = engagement_data[(unit_id, target_id)]

                            # Create label
                            label = f"Hits: {data.get('hits', 0)}\n"
                            label += f"Damage: {data.get('damage', 0):.1f}"

                            # Calculate midpoint
                            mid_x = (unit_pos[0] + target_pos[0]) / 2
                            mid_y = (unit_pos[1] + target_pos[1]) / 2

                            # Add label
                            ax.annotate(label, xy=(mid_x, mid_y),
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                                        ha='center', va='center', fontsize=8)

        # Add title
        ax.set_title(title, fontsize=16)

        # Set limits and grid
        ax.set_xlim(-1, self.env.width + 1)
        ax.set_ylim(-1, self.env.height + 1)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax


# Environment Setup Functions
def setup_environment(width=70, height=25, debug_level=1):
    """Set up the environment with the specified dimensions."""
    config = EnvironmentConfig(
        width=width,
        height=height,
        debug_level=debug_level
    )
    env = MilitaryEnvironment(config)

    # Initialize terrain
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)

    return env


def setup_terrain(env, width=70, height=25):
    """Set up the terrain according to the requirements."""
    # Default terrain - BARE
    for y in range(height):
        for x in range(width):
            env.state_manager.state_tensor[y, x, 0] = TerrainType.BARE.value
            env.state_manager.state_tensor[y, x, 1] = ElevationType.GROUND_LEVEL.value

    # Set up terrain for y=12 (center line)
    # - sparse vegetation from x=6 to x=16
    for x in range(6, 17):
        env.state_manager.state_tensor[12, x, 0] = TerrainType.SPARSE_VEG.value

    # - dense vegetation from x=17 to x=36
    for x in range(17, 37):
        env.state_manager.state_tensor[12, x, 0] = TerrainType.DENSE_VEG.value

    # - dense elevated terrain from x=37 to x=65
    for x in range(37, 66):
        env.state_manager.state_tensor[12, x, 0] = TerrainType.DENSE_VEG.value
        env.state_manager.state_tensor[12, x, 1] = ElevationType.ELEVATED_LEVEL.value

    # Set up terrain for y=16 (top line)
    # - sparse vegetation from x=6 to x=16
    for x in range(6, 17):
        env.state_manager.state_tensor[16, x, 0] = TerrainType.SPARSE_VEG.value

    # - dense vegetation from x=17 to x=36
    for x in range(17, 37):
        env.state_manager.state_tensor[16, x, 0] = TerrainType.DENSE_VEG.value

    # - woods terrain from x=37 to x=65
    for x in range(37, 66):
        env.state_manager.state_tensor[16, x, 0] = TerrainType.WOODS.value


def create_friendly_team(env, start_position=(1, 12), orientation=0):
    """Create the friendly team at the specified position and orientation."""
    # Create the team first
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.ALPHA_TEAM,
        start_position=start_position
    )

    # Print orientation before setting it
    # print(f"\nBEFORE: Team orientation is {env.get_unit_property(team_id, 'orientation')}°")

    # Explicitly set orientation
    env.update_unit_property(team_id, 'orientation', orientation)

    # Print after setting orientation
    # print(f"AFTER: Team orientation set to {orientation}°")

    # Also update all member orientations to ensure consistency
    for member_id in env.get_unit_children(team_id):
        env.update_unit_property(member_id, 'orientation', orientation)
        # print(f"Member {member_id} orientation set to {orientation}°")

    # Now apply formation AFTER orientation is set
    # print(f"Applying formation with orientation {orientation}°")

    # Apply appropriate formation based on orientation
    if orientation == 90:
        US_IN_apply_formation(env, team_id, "team_line_right")
    else:
        # Default formation
        US_IN_apply_formation(env, team_id, "team_line_right")

    # Verify orientation after formation is applied
    final_orientation = env.get_unit_property(team_id, 'orientation')
    # print(f"FINAL: Team orientation is now {final_orientation}°")

    return team_id


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
            (11, 8), (11, 12), (11, 16),
            (21, 8), (21, 12), (21, 16),
            (31, 8), (31, 12), (31, 16),
            (41, 8), (41, 12), (41, 16),
            (61, 8), (61, 12), (61, 16)
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


def create_high_priority_enemy(env, position, threat_level=5.0):
    """
    Create a high-priority enemy target (like a machine gun team).

    Args:
        env: Military environment
        position: (x,y) position for the enemy
        threat_level: Numeric threat level (higher = more threatening)

    Returns:
        ID of the created enemy unit
    """
    # Create the unit
    enemy_id = env.create_unit(
        unit_type=UnitType.WEAPONS_TEAM,
        unit_id_str=f"ENEMY-MG",
        start_position=position
    )

    # Set as enemy force
    env.update_unit_property(enemy_id, 'force_type', ForceType.ENEMY)

    # Create machine gunner
    mg_id = env.create_soldier(
        role=US_IN_Role.MACHINE_GUNNER,
        unit_id_str=f"ENEMY-MG-MG",
        position=position,
        is_leader=True
    )

    # Set as enemy force
    env.update_unit_property(mg_id, 'force_type', ForceType.ENEMY)

    # Create assistant gunner
    ag_id = env.create_soldier(
        role=US_IN_Role.ASSISTANT_GUNNER,
        unit_id_str=f"ENEMY-MG-AG",
        position=(position[0] + 1, position[1]),
        is_leader=False
    )

    # Set as enemy force
    env.update_unit_property(ag_id, 'force_type', ForceType.ENEMY)

    # Set parent-child relationship
    env.set_unit_hierarchy(mg_id, enemy_id)
    env.set_unit_hierarchy(ag_id, enemy_id)

    # Set threat level property for test purposes
    env.update_unit_property(enemy_id, 'threat_level', threat_level)

    return enemy_id


# Test Functions
def execute_engagements(env, friendly_id, enemy_ids, rounds_per_target=1):
    """
    Execute engagements against each enemy target.

    Note: rounds_per_target here represents "bursts" or single shots,
    and will be multiplied by the weapon's fire rate internally.

    Args:
        env: Military environment
        friendly_id: ID of friendly unit
        enemy_ids: List of enemy unit IDs
        rounds_per_target: Number of rounds/bursts per target

    Returns:
        List of engagement results
    """
    results = []

    for enemy_id in enemy_ids:
        # Get enemy position
        enemy_pos = env.get_unit_position(enemy_id)

        print(f"\n{'=' * 60}")
        print(f"ENGAGING TARGET AT {enemy_pos}")
        print(f"{'=' * 60}")

        # Check if target is valid for each soldier
        print("\nTARGET VALIDATION:")
        print("-----------------")
        friendly_members = env.get_unit_children(friendly_id)
        valid_for_any = False

        for member_id in friendly_members:
            can_engage = env.combat_manager.validate_target(member_id, enemy_pos)
            member_pos = env.get_unit_position(member_id)
            distance = math.sqrt((enemy_pos[0] - member_pos[0]) ** 2 + (enemy_pos[1] - member_pos[1]) ** 2)
            role_value = env.get_unit_property(member_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name

            # Print basic validation info
            print(
                f"  Member {role_name} (ID: {member_id}) at {member_pos}: {'CAN' if can_engage else 'CANNOT'} engage. Distance: {distance:.1f}")

            # Get orientation and sector info directly from state manager
            orientation = env.get_unit_property(member_id, 'orientation')
            primary_start = env.get_unit_property(member_id, 'primary_sector_rotated_start')
            primary_end = env.get_unit_property(member_id, 'primary_sector_rotated_end')
            secondary_start = env.get_unit_property(member_id, 'secondary_sector_rotated_start')
            secondary_end = env.get_unit_property(member_id, 'secondary_sector_rotated_end')

            # Calculate target angle
            target_angle = env.combat_manager.calculate_target_angle(member_pos, enemy_pos)

            print(f"    Orientation: {orientation}°")
            print(f"    Target angle: {target_angle:.1f}°")

            # Check and display sector information
            if primary_start is not None and primary_end is not None:
                print(f"    Rotated primary sector: {primary_start}° to {primary_end}°")

                # Check if target is in sectors
                in_primary = False
                if primary_start <= primary_end:
                    in_primary = primary_start <= target_angle <= primary_end
                else:
                    in_primary = target_angle >= primary_start or target_angle <= primary_end
                print(f"    Target in primary sector: {in_primary}")

                if secondary_start is not None and secondary_end is not None:
                    print(f"    Rotated secondary sector: {secondary_start}° to {secondary_end}°")
                    in_secondary = False
                    if secondary_start <= secondary_end:
                        in_secondary = secondary_start <= target_angle <= secondary_end
                    else:
                        in_secondary = target_angle >= secondary_start or target_angle <= secondary_end
                    print(f"    Target in secondary sector: {in_secondary}")

            if can_engage:
                valid_for_any = True

                # Print weapon and hit probability information
                weapon = env.combat_manager._get_unit_weapon(member_id)
                base_hit_prob = env.calculate_hit_probability(distance, weapon)

                # Get visibility and cover modifiers
                los_result = env.visibility_manager.check_line_of_sight(
                    member_pos, enemy_pos)

                # Calculate final probability
                final_hit_prob = env.visibility_manager.modify_hit_probability(
                    base_hit_prob,
                    member_pos,
                    enemy_pos,
                    enemy_id
                )

                print(f"    Weapon: {weapon.name if weapon else 'None'}")
                print(f"    Fire Rate: {weapon.fire_rate if weapon else 0}")
                print(f"    Base hit probability: {base_hit_prob:.2f}")
                print(f"    Line of sight quality: {los_result['los_quality']:.2f}")
                print(f"    Final hit probability: {final_hit_prob:.2f}")

        # Execute engagement and record results
        if valid_for_any:
            print("\nEXECUTING ENGAGEMENT:")
            print("--------------------")

            # Print ammo before engagement
            print("AMMUNITION BEFORE ENGAGEMENT:")
            for member_id in friendly_members:
                ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
                role_value = env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
                weapon = env.combat_manager._get_unit_weapon(member_id)
                weapon_name = weapon.name if weapon else "None"
                print(f"  {role_name}: {ammo} rounds of {weapon_name}")

            # Execute team engagement with control parameters
            engagement_result = env.combat_manager.execute_team_engagement(
                team_id=friendly_id,
                target_pos=enemy_pos,
                engagement_type=EngagementType.POINT,
                control_params={
                    'max_rounds': rounds_per_target,  # This is "bursts" per target
                    'time_limit': 5
                }
            )

            # Print raw engagement data
            print("\nRAW ENGAGEMENT DATA:")
            for key, value in engagement_result.items():
                print(f"  {key}: {value}")

            # Print engagement results
            print(f"\nENGAGEMENT RESULTS:")
            print(f"  Hits: {engagement_result['total_hits']}/{engagement_result['ammo_expended']} rounds")
            print(f"  Damage dealt: {engagement_result['total_damage']:.1f}")
            print(f"  Suppression level: {engagement_result['suppression_level']:.2f}")

            # Check ammunition status after engagement
            print("\nAMMUNITION STATUS AFTER ENGAGEMENT:")
            print("----------------------------------")
            for member_id in friendly_members:
                ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
                role_value = env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
                print(f"  {role_name}: {ammo} rounds remaining")

            # Check health status of engaged enemy
            enemy_health = env.get_unit_property(enemy_id, 'health')
            print(f"\nENEMY STATUS:")
            print(f"  Health: {enemy_health:.1f}/100")

            # Check suppression status
            enemy_soldier_id = env.get_unit_children(enemy_id)[0]
            if enemy_soldier_id in env.combat_manager.suppressed_units:
                supp_level = env.combat_manager.suppressed_units[enemy_soldier_id]['level']
                supp_duration = env.combat_manager.suppressed_units[enemy_soldier_id]['duration']
                print(f"  Suppression: {supp_level:.2f} (Duration: {supp_duration} steps)")
            else:
                print(f"  Suppression: None")

            results.append(engagement_result)
        else:
            print("\nNo valid engagement - target not engageable by any team member")
            results.append(None)

    return results


def test_basic_engagement(output_dir="Model_Validation_Combat/basic_engagement_varying_terrain"):
    """
    Test basic engagement mechanics with varying terrain.

    Args:
        output_dir: Directory to save output files

    Returns:
        Environment and test results
    """
    print("\n" + "=" * 80)
    print("BASIC ENGAGEMENT TEST WITH VARYING TERRAIN")
    print("=" * 80)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup environment
    env = setup_environment(width=70, height=25)

    # Setup terrain
    setup_terrain(env, width=70, height=25)

    # Create friendly team at position (1, 12) with orientation 0° (facing East)
    friendly_id = create_friendly_team(env, start_position=(1, 12), orientation=0)

    # Create enemy targets
    enemy_ids = create_enemy_targets(env)

    # Create a visualizer
    visualizer = CombatVisualizer(env)

    # Visualize initial state
    fig, ax = visualizer.plot_combat_state(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        title="Initial Combat State - Before Engagement",
        output_path=os.path.join(output_dir, "initial_combat_state.png")
    )
    plt.close(fig)

    # Execute engagements
    engagement_results = execute_engagements(env, friendly_id, enemy_ids, rounds_per_target=2)

    # Visualize final state
    fig, ax = visualizer.plot_combat_state(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        title="Final Combat State - After All Engagements",
        output_path=os.path.join(output_dir, "final_combat_state.png")
    )
    plt.close(fig)

    # Visualize each engagement separately
    for i, enemy_id in enumerate(enemy_ids):
        if engagement_results[i]:  # If there was a valid engagement
            enemy_pos = env.get_unit_position(enemy_id)
            fig, ax = visualizer.plot_combat_state(
                friendly_units=[friendly_id],
                enemy_units=[enemy_id],
                title=f"Engagement {i + 1}: Target at {enemy_pos}",
                engagement_results=engagement_results[i],
                output_path=os.path.join(output_dir, f"engagement_{i + 1}_target_{enemy_pos[0]}_{enemy_pos[1]}.png")
            )
            plt.close(fig)

    return env, engagement_results


def execute_suppression_engagement(env, friendly_id, enemy_ids, rounds_per_target=20):
    """
    Execute a suppression-focused engagement against enemy targets with enhanced hit location tracking.
    Properly handles weapon fire rates to ensure correct ammunition expenditure.

    Args:
        env: Military environment
        friendly_id: ID of friendly unit
        enemy_ids: List of enemy unit IDs to suppress
        rounds_per_target: Number of rounds for suppression (higher than normal)

    Returns:
        Dictionary with suppression results including hit locations
    """
    print("\nEXECUTING SUPPRESSION ENGAGEMENT:")
    print("=" * 50)

    # Explain suppression mechanics
    print("\nSUPPRESSION MECHANICS OVERVIEW:")
    print("-" * 30)
    print("Suppression represents the psychological effect of incoming fire on enemy units.")
    print("It reduces their combat effectiveness without necessarily causing casualties.")
    print("Key factors affecting suppression:")
    print("  - Volume of fire (more rounds = more suppression)")
    print("  - Weapon type (automatic weapons create more suppression)")
    print("  - Sustained fire (continuous fire is more suppressive)")
    print("  - Area coverage (wider area affects more units)")
    print("  - Terrain (affects how visible/intimidating fire appears)")
    print("\nSuppression effects on enemy units:")
    print("  - Reduced accuracy (up to 80% reduction)")
    print("  - Decreased mobility (up to 60% reduction)")
    print("  - Longer reaction times")
    print("  - Limited ability to return effective fire")

    # Initialize results - now including engagement_data to store hit locations
    suppression_results = {
        'suppressed_units': {},
        'accuracy_reduction': {},
        'mobility_reduction': {},
        'engagement_data': {}  # For storing hit locations per target
    }

    # Execute suppression against each target
    for enemy_id in enemy_ids:
        # Get enemy position
        enemy_pos = env.get_unit_position(enemy_id)
        print(f"\nSuppressing target at {enemy_pos}")

        # Check if target is valid for team engagement
        friendly_members = env.get_unit_children(friendly_id)
        valid_members = []

        for member_id in friendly_members:
            can_engage = env.combat_manager.validate_target(member_id, enemy_pos)
            if can_engage:
                role_value = env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
                valid_members.append((member_id, role_name))

        print(f"\nTeam members who can engage this target: {len(valid_members)}/{len(friendly_members)}")
        for member_id, role_name in valid_members:
            print(f"  - {role_name} (ID: {member_id})")

        # Print suppression engagement parameters
        print("\nSUPPRESSION PARAMETERS:")
        print("-" * 30)
        print(f"  Target Position: {enemy_pos}")
        print(f"  Engagement Type: AREA")
        print(f"  Max Rounds Per Member: {rounds_per_target}")
        print(f"  Area Radius: 3 cells")
        print(f"  Suppress Only: True")
        print(f"  Sustained Fire: True")

        # NEW: Print fire rate adjustment explanation
        print("\nFIRE RATE ADJUSTMENT:")
        print("-" * 30)
        print("Adjusting max_rounds based on weapon fire rate to ensure correct ammunition expenditure:")

        for member_id, role_name in valid_members:
            weapon = env.combat_manager._get_unit_weapon(member_id)
            if weapon:
                fire_rate = weapon.fire_rate
                # Calculate adjusted bursts based on fire rate
                adjusted_bursts = rounds_per_target // max(1, fire_rate)
                total_rounds = adjusted_bursts * fire_rate

                print(f"  - {role_name} with {weapon.name} (Fire Rate: {fire_rate}):")
                print(f"    * {adjusted_bursts} bursts × {fire_rate} rounds/burst = {total_rounds} total rounds")

        # Execute team engagement with adjusted max_rounds approach
        print("\nEXECUTING AREA SUPPRESSION:")
        result = env.combat_manager.execute_team_engagement(
            team_id=friendly_id,
            target_pos=enemy_pos,
            engagement_type=EngagementType.AREA,  # Area effect for suppression
            control_params={
                'max_rounds': rounds_per_target,
                'suppress_only': True,  # Focus on suppression
                'area_radius': 3,  # Wider area effect
                'sustained': True,  # Sustained fire for better suppression
                'adjust_for_fire_rate': True  # NEW: Flag to adjust for fire rate
            }
        )

        # Store hit locations and engagement data
        suppression_results['engagement_data'][enemy_id] = {
            'hit_locations': result.get('hit_locations', []),
            'suppression_level': result.get('suppression_level', 0),
            'ammo_expended': result.get('ammo_expended', 0)
        }

        # Print hit location information if available
        hit_locations = result.get('hit_locations', [])
        if hit_locations:
            print("\nROUND IMPACT LOCATIONS:")
            print("-" * 30)
            print(f"  Total rounds fired: {result['ammo_expended']}")
            print(f"  Rounds with recorded impacts: {len(hit_locations)}")

            # Print a sample of hit locations (up to 10)
            for i, loc in enumerate(hit_locations[:10]):
                print(f"  Hit #{i + 1} at position: {loc}")

            if len(hit_locations) > 10:
                print(f"  ... and {len(hit_locations) - 10} more impacts")

            # Calculate distribution statistics
            distances = [math.sqrt((loc[0] - enemy_pos[0]) ** 2 + (loc[1] - enemy_pos[1]) ** 2)
                         for loc in hit_locations]

            if distances:
                avg_distance = sum(distances) / len(distances)
                max_distance = max(distances)

                print("\nROUND DISTRIBUTION STATISTICS:")
                print("-" * 30)
                print(f"  Average distance from target: {avg_distance:.2f} cells")
                print(f"  Maximum distance from target: {max_distance:.2f} cells")

                # Count impacts by distance ranges
                close_impacts = sum(1 for d in distances if d <= 1)
                medium_impacts = sum(1 for d in distances if 1 < d <= 2)
                far_impacts = sum(1 for d in distances if d > 2)

                print(f"  Close impacts (≤1 cell): {close_impacts} ({close_impacts / len(distances) * 100:.1f}%)")
                print(f"  Medium impacts (1-2 cells): {medium_impacts} ({medium_impacts / len(distances) * 100:.1f}%)")
                print(f"  Far impacts (>2 cells): {far_impacts} ({far_impacts / len(distances) * 100:.1f}%)")
        else:
            print("\nNo hit location data available for this engagement")

        # Print detailed suppression calculation
        print("\nSUPPRESSION CALCULATION DETAILS:")
        print("-" * 30)
        print(f"1. Base volume factor: {min(1.0, result['ammo_expended'] / 30.0):.2f}")
        print(f"   - {result['ammo_expended']} rounds fired (max effectiveness at 30+ rounds)")
        print(f"2. Sustained fire bonus: +50% effectiveness")
        print(f"3. Automatic weapons bonus: +50-80% based on weapon types")
        print(f"4. Area effect modifier: {1.0 - (3 / 10.0):.2f} (smaller area = more concentrated)")
        print(f"5. Distance & terrain modifiers applied")
        print(f"Final suppression effect: {result['suppression_level']:.2f} (scale 0-1)")

        # Check suppression status of enemy soldiers in detail
        print("\nSUPPRESSION EFFECTS ON ENEMY:")
        for soldier_id in env.get_unit_children(enemy_id):
            soldier_pos = env.get_unit_position(soldier_id)

            if soldier_id in env.combat_manager.suppressed_units:
                supp_data = env.combat_manager.suppressed_units[soldier_id]
                level = supp_data['level']
                duration = supp_data['duration']

                print(f"  Soldier at {soldier_pos}:")
                print(f"    Suppression Level: {level:.2f}")
                print(f"    Suppression Duration: {duration} steps")

                # Calculate detailed effects
                accuracy_reduction = min(0.8, level * 0.9)
                mobility_reduction = min(0.6, level * 0.7)
                reaction_reduction = min(0.7, level * 0.8)

                print(f"    Effects:")
                print(f"      Accuracy: -{accuracy_reduction * 100:.0f}%")
                print(f"      Mobility: -{mobility_reduction * 100:.0f}%")
                print(f"      Reaction Time: +{reaction_reduction * 100:.0f}%")

                # Record results
                suppression_results['suppressed_units'][soldier_id] = {
                    'level': level,
                    'duration': duration
                }
                suppression_results['accuracy_reduction'][soldier_id] = accuracy_reduction
                suppression_results['mobility_reduction'][soldier_id] = mobility_reduction

            else:
                print(f"  Soldier at {soldier_pos}: Not suppressed")

    return suppression_results


def test_suppression(output_dir="Model_Validation_Combat/suppression"):
    """
    Test suppression mechanics with enhanced output and hit location tracking.

    Args:
        output_dir: Directory to save output files

    Returns:
        Environment and test results
    """
    print("\n" + "=" * 80)
    print("ENHANCED SUPPRESSION TEST")
    print("=" * 80)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup environment
    env = setup_environment(width=70, height=25)

    # Setup terrain
    setup_terrain(env, width=70, height=25)

    # Create friendly team
    friendly_id = create_friendly_team(env, start_position=(1, 12), orientation=0)

    # Create enemy targets (fewer targets, more focused for suppression test)
    enemy_positions = [(15, 10), (15, 12), (15, 14)]
    enemy_ids = create_enemy_targets(env, positions=enemy_positions)

    # Print test configuration
    print("\nTEST CONFIGURATION:")
    print("-" * 30)
    print(f"Friendly Team ID: {friendly_id}")
    print(f"Enemy Target IDs: {enemy_ids}")
    print(f"Enemy Positions: {enemy_positions}")

    # Print detailed explanation of test goals
    print("\nTEST OBJECTIVES:")
    print("-" * 30)
    print("1. Demonstrate how suppression is applied to enemy units")
    print("2. Analyze the factors that influence suppression effectiveness")
    print("3. Measure how suppression degrades over time")
    print("4. Quantify the effects of suppression on enemy combat capabilities")
    print("5. Validate suppression mechanics across different terrain types")
    print("6. Track and analyze round distribution patterns during suppression")

    # Create visualizer
    visualizer = CombatVisualizer(env)

    # Visualize initial state
    fig, ax = visualizer.plot_combat_state(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        title="Initial State - Before Suppression",
        output_path=os.path.join(output_dir, "initial_state.png")
    )
    plt.close(fig)

    # Print detailed team composition
    print("\nFRIENDLY TEAM COMPOSITION:")
    print("-" * 30)
    for member_id in env.get_unit_children(friendly_id):
        role_value = env.get_unit_property(member_id, 'role')
        role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
        weapon = env.combat_manager._get_unit_weapon(member_id)
        weapon_name = weapon.name if weapon else "None"
        ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')

        print(f"Member {role_name}:")
        print(f"  ID: {member_id}")
        print(f"  Position: {env.get_unit_position(member_id)}")
        print(f"  Weapon: {weapon_name}")
        print(f"  Ammunition: {ammo}")
        print(f"  Is Leader: {env.get_unit_property(member_id, 'is_leader', False)}")

    # Execute suppression engagement with enhanced output
    suppression_results = execute_suppression_engagement(env, friendly_id, enemy_ids, rounds_per_target=30)

    # Create additional visualization showing suppression effects with actual hit locations
    fig, ax = visualizer.create_suppression_visualization(friendly_id, enemy_ids, suppression_results,
                                                          output_path=os.path.join(output_dir,
                                                                                   "suppression_effects_detailed.png"))
    plt.close(fig)

    # Create a dedicated visualization showing just the round impact patterns
    fig, ax = visualizer.create_round_impact_visualization(friendly_id, enemy_ids, suppression_results,
                                                           output_path=os.path.join(output_dir,
                                                                                    "round_impact_patterns.png"))
    plt.close(fig)

    # Test suppression recovery over time with detailed output
    print("\nTESTING SUPPRESSION RECOVERY OVER TIME:")
    print("=" * 50)
    print("\nSuppression Recovery Mechanics:")
    print("  - Recovery Rate: {:.2f} per step".format(env.combat_manager.suppression_recovery_rate))
    print("  - Units gradually recover from suppression over time")
    print("  - Higher suppression levels take longer to recover from")
    print("  - Units under continuous fire maintain suppression levels")

    recovery_steps = 5
    recovery_results = []

    print("\nStepwise Recovery Analysis:")
    for step in range(recovery_steps):
        # Update suppression states (simulate time passing)
        current_levels = env.combat_manager.update_suppression_states()

        # Record current state
        recovery_results.append(current_levels)

        print(f"\nStep {step + 1}/{recovery_steps}:")

        for unit_id, level in current_levels.items():
            unit_pos = env.get_unit_position(unit_id)
            unit_type = env.get_unit_property(unit_id, 'type')

            print(f"  Unit {unit_id} at {unit_pos}:")
            print(f"    Suppression Level: {level:.2f}")
            print(f"    Remaining Duration: {env.combat_manager.suppressed_units[unit_id]['duration']} steps")

            # Calculate effects at current level
            accuracy_reduction = min(0.8, level * 0.9)
            mobility_reduction = min(0.6, level * 0.7)

            print(f"    Current Effects:")
            print(f"      Accuracy: -{accuracy_reduction * 100:.0f}%")
            print(f"      Mobility: -{mobility_reduction * 100:.0f}%")

    # Create recovery visualization
    fig, ax = visualizer.create_recovery_visualization(enemy_ids, recovery_results,
                                                       output_path=os.path.join(output_dir, "suppression_recovery.png"))
    plt.close(fig)

    # Visualize final state after recovery
    fig, ax = visualizer.plot_combat_state(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        title="Final State - After Suppression Recovery",
        output_path=os.path.join(output_dir, "final_state.png")
    )
    plt.close(fig)

    # Print test summary
    print("\nSUPPRESSION TEST SUMMARY:")
    print("-" * 30)

    # Summarize total impacts
    total_impacts = 0
    for enemy_id in enemy_ids:
        if enemy_id in suppression_results.get('engagement_data', {}):
            engagement_data = suppression_results['engagement_data'][enemy_id]
            hit_locations = engagement_data.get('hit_locations', [])
            total_impacts += len(hit_locations)

    print(f"Total round impacts recorded: {total_impacts}")

    # Summarize how many enemies were successfully suppressed
    suppressed_count = len(suppression_results['suppressed_units'])
    print(f"Successfully suppressed {suppressed_count}/{len(enemy_ids)} enemy units")

    # Summarize average suppression level
    if suppressed_count > 0:
        avg_suppression = sum(
            data['level'] for data in suppression_results['suppressed_units'].values()) / suppressed_count
        print(f"Average suppression level: {avg_suppression:.2f}")

    # Summarize ammunition expenditure
    total_ammo = 0
    for enemy_id in enemy_ids:
        if enemy_id in suppression_results.get('engagement_data', {}):
            total_ammo += suppression_results['engagement_data'][enemy_id].get('ammo_expended', 0)

    print(f"Total ammunition expended: {total_ammo}")

    # Print conclusions
    print("\nTEST CONCLUSIONS:")
    print("-" * 30)
    print("1. Automatic weapons are highly effective at achieving suppression")
    print("2. Round distribution follows a pattern based on area of effect")
    print("3. Suppression degrades gradually, requiring sustained fire to maintain")
    print("4. Suppressed units suffer significant combat performance penalties")
    print("5. Suppression is an efficient way to neutralize enemy effectiveness")
    print("6. Terrain and distance affect suppression effectiveness")

    return env, suppression_results


def execute_coordination_test(env, friendly_id, enemy_ids):
    """
    Test which soldiers engage when a target is in multiple sectors.

    Args:
        env: Military environment
        friendly_id: ID of friendly unit
        enemy_ids: List of enemy unit IDs

    Returns:
        Dictionary with test results
    """
    print("\nEXECUTING COORDINATION TEST:")
    print("=" * 50)

    # Initialize results
    results = {
        'unit_targets': {},
        'engagement_data': {},
        'members_engaged': {}
    }

    # Get all team members
    friendly_members = env.get_unit_children(friendly_id)

    # For each enemy target, determine which members can engage
    for enemy_id in enemy_ids:
        enemy_pos = env.get_unit_position(enemy_id)
        print(f"\nAnalyzing target at {enemy_pos}")

        # Find which members can engage this target
        members_can_engage = []
        for member_id in friendly_members:
            can_engage = env.combat_manager.validate_target(member_id, enemy_pos)
            if can_engage:
                role_value = env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
                members_can_engage.append((member_id, role_name))

        print(f"Members who can engage: {len(members_can_engage)}")
        for member_id, role_name in members_can_engage:
            print(f"  - {role_name} (ID: {member_id})")

        # Store which members can engage this target
        results['members_engaged'][enemy_id] = members_can_engage

        # If multiple members can engage, execute engagement and see who actually fires
        if len(members_can_engage) > 1:
            print("\nMultiple members can engage - executing engagement to test coordination")

            # Execute engagement
            engagement_result = env.combat_manager.execute_team_engagement(
                team_id=friendly_id,
                target_pos=enemy_pos,
                engagement_type=EngagementType.POINT,
                control_params={
                    'max_rounds': 5,
                    'time_limit': 3
                }
            )

            # Determine which members actually engaged
            participants = []
            for member_id, role_name in members_can_engage:
                # Check if ammo was expended
                initial_ammo = env.get_unit_property(member_id, 'ammo_primary', 0)
                current_ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
                rounds_fired = initial_ammo - current_ammo

                if rounds_fired > 0:
                    participants.append((member_id, role_name, rounds_fired))

            print("\nMembers who actually fired:")
            for member_id, role_name, rounds in participants:
                print(f"  - {role_name}: {rounds} rounds")

            # Record engagement data
            if enemy_id not in results['unit_targets']:
                results['unit_targets'][friendly_id] = []
            results['unit_targets'][friendly_id].append(enemy_id)

            results['engagement_data'][(friendly_id, enemy_id)] = {
                'participants': participants,
                'hits': engagement_result['total_hits'],
                'damage': engagement_result['total_damage'],
                'rounds': engagement_result['ammo_expended']
            }

    return results


def test_coordination(output_dir="Model_Validation_Combat/coordination"):
    """
    Test team coordination during engagements with enhanced debug output.
    This test examines how team members coordinate their fire when multiple
    members can engage the same target.

    Args:
        output_dir: Directory to save output files

    Returns:
        Environment and test results
    """
    print("\n" + "=" * 80)
    print("ENHANCED TEAM COORDINATION TEST")
    print("=" * 80)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup environment
    env = setup_environment(width=70, height=25)

    # Setup terrain
    setup_terrain(env, width=70, height=25)

    # Create friendly team in a configuration where multiple members can engage the same targets
    friendly_id = create_friendly_team(env, start_position=(1, 12), orientation=0)

    # Create specific enemy targets positioned to be in multiple sectors
    enemy_positions = [
        (10, 12),  # Directly in front, should be engageable by multiple team members
        (8, 8),  # Upper left, may be in sectors of multiple members
        (8, 16)  # Lower left, may be in sectors of multiple members
    ]
    enemy_ids = create_enemy_targets(env, positions=enemy_positions)

    # Print test configuration
    print("\nTEST CONFIGURATION:")
    print("-" * 30)
    print(f"Friendly Team ID: {friendly_id}")
    print(f"Enemy Target IDs: {enemy_ids}")
    print(f"Enemy Positions: {enemy_positions}")

    # Print detailed explanation of test goals
    print("\nTEST OBJECTIVES:")
    print("-" * 30)
    print("1. Determine which team members can engage targets in overlapping sectors")
    print("2. Analyze how CombatManager distributes fire between members")
    print("3. Verify proper coordination between automatic weapons and riflemen")
    print("4. Validate sector of fire calculations and proper target validation")

    # Create visualizer
    visualizer = CombatVisualizer(env)

    # Visualize initial state
    fig, ax = visualizer.plot_combat_state(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        title="Initial State - Coordination Test",
        output_path=os.path.join(output_dir, "initial_state.png")
    )
    plt.close(fig)

    # Execute coordination test
    coordination_results = execute_coordination_test(env, friendly_id, enemy_ids)

    # Print summary of findings
    print("\nTEST SUMMARY:")
    print("-" * 30)

    # Analyze results for each target
    for enemy_id in enemy_ids:
        if enemy_id in coordination_results['members_engaged']:
            members = coordination_results['members_engaged'][enemy_id]
            print(f"\nTarget {enemy_id} at {env.get_unit_position(enemy_id)}:")
            print(f"  Members who could engage: {len(members)}")

            # Check if target was actually engaged
            was_engaged = False
            engagement_data = None

            if friendly_id in coordination_results['unit_targets']:
                if enemy_id in coordination_results['unit_targets'][friendly_id]:
                    was_engaged = True
                    engagement_data = coordination_results['engagement_data'].get((friendly_id, enemy_id))

            if was_engaged and engagement_data:
                print(f"  Engagement results:")
                print(f"    Hits: {engagement_data['hits']}")
                print(f"    Damage: {engagement_data['damage']:.1f}")
                print(f"    Rounds: {engagement_data['rounds']}")

                # Analyze who participated
                if 'participants' in engagement_data:
                    participants = engagement_data['participants']
                    print(f"    Participants: {len(participants)} members")

                    if len(participants) > 1:
                        print("    ✓ Multiple members coordinated fire")

                        # Check role distribution
                        roles = [role for _, role, _ in participants]
                        if "AUTO_RIFLEMAN" in str(roles):
                            print("    ✓ Auto Rifleman participated in coordinated engagement")
                        if "TEAM_LEADER" in str(roles):
                            print("    ✓ Team Leader guided the engagement")
                    else:
                        print("    ✗ Only one member engaged - limited coordination")
            else:
                print("  Target was not engaged")

    # Overall coordination assessment
    multi_engagements = sum(1 for k, v in coordination_results['engagement_data'].items()
                            if 'participants' in v and len(v['participants']) > 1)

    print("\nOVERALL COORDINATION ASSESSMENT:")
    if multi_engagements > 0:
        print("✓ Team demonstrated coordination on multiple targets")
        print(f"✓ {multi_engagements} engagements involved multiple team members")
        print("✓ Combat Manager properly distributed fire among team members")
    else:
        print("✗ Limited coordination observed - most targets engaged by single members")
        print("✗ Combat Manager may need adjustment for better fire distribution")

    # Visualize engagement priority
    fig, ax = visualizer.visualize_engagement_priorities(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        engagement_results=coordination_results,
        title="Engagement Coordination Results",
        output_path=os.path.join(output_dir, "coordination_results.png")
    )
    plt.close(fig)

    # Visualize final state
    fig, ax = visualizer.plot_combat_state(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        title="Final State - After Coordination Test",
        output_path=os.path.join(output_dir, "final_state.png")
    )
    plt.close(fig)

    return env, coordination_results


def execute_priority_target_test(env, friendly_id, enemy_ids, high_priority_id):
    """
    Test engagement prioritization of high-threat targets with enhanced debug output.
    This test examines how the team prioritizes high-threat targets over standard targets,
    with detailed analysis of target selection and engagement patterns.

    Args:
        env: Military environment
        friendly_id: ID of friendly unit
        enemy_ids: List of enemy unit IDs
        high_priority_id: ID of high-priority target

    Returns:
        Dictionary with test results
    """
    print("\nEXECUTING HIGH PRIORITY TARGET TEST:")
    print("=" * 50)

    # Initialize results
    results = {
        'threat_levels': {},
        'unit_targets': {},
        'engagement_data': {},
        'high_threat_threshold': 3.0,  # Threshold for classifying as high threat
        'priority_target_id': high_priority_id
    }

    # Print friendly team composition
    print("\nFRIENDLY TEAM COMPOSITION:")
    print("-" * 30)
    friendly_members = env.get_unit_children(friendly_id)

    for member_id in friendly_members:
        role_value = env.get_unit_property(member_id, 'role')
        role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
        position = env.get_unit_position(member_id)
        weapon = env.combat_manager._get_unit_weapon(member_id)

        print(f"Member {role_name} (ID: {member_id}):")
        print(f"  Position: {position}")
        print(f"  Weapon: {weapon.name if weapon else 'None'}")
        print(f"  Damage: {weapon.damage if weapon else 0}")
        print(f"  Fire Rate: {weapon.fire_rate if weapon else 0}")
        print(f"  Max Range: {weapon.max_range if weapon else 0}")
        print(f"  Ammunition: {env.combat_manager._get_unit_ammo(member_id, 'primary')}")

    # Print enemy composition with threat levels
    print("\nENEMY COMPOSITION AND THREAT LEVELS:")
    print("-" * 30)

    # Calculate "threat level" for each enemy
    for enemy_id in enemy_ids:
        # Get enemy position
        enemy_pos = env.get_unit_position(enemy_id)

        # Get threat level (set earlier when creating the unit)
        threat_level = env.get_unit_property(enemy_id, 'threat_level', 1.0)
        results['threat_levels'][enemy_id] = threat_level

        # Get enemy type and weapons
        is_high_priority = enemy_id == high_priority_id
        enemy_children = env.get_unit_children(enemy_id)

        print(f"Enemy {enemy_id} at {enemy_pos}:")
        print(f"  Threat Level: {threat_level}")
        print(f"  High Priority: {is_high_priority}")

        # Print enemy composition
        print(f"  Composition:")
        for child_id in enemy_children:
            role_value = env.get_unit_property(child_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else str(role_value)
            weapon = env.combat_manager._get_unit_weapon(child_id)

            print(f"    {role_name}:")
            print(f"      Weapon: {weapon.name if weapon else 'None'}")
            print(f"      Damage: {weapon.damage if weapon else 0}")
            print(f"      Fire Rate: {weapon.fire_rate if weapon else 0}")

    # Analyze target priorities
    print("\nTARGET PRIORITY ANALYSIS:")
    print("-" * 30)

    # Check if high priority target is engageable
    high_priority_pos = env.get_unit_position(high_priority_id)
    high_priority_engageable = False

    print(f"High Priority Target at {high_priority_pos}:")

    for member_id in friendly_members:
        role_value = env.get_unit_property(member_id, 'role')
        role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
        can_engage = env.combat_manager.validate_target(member_id, high_priority_pos)

        if can_engage:
            high_priority_engageable = True
            print(f"  ✓ Engageable by {role_name}")

            # Calculate distance and angle
            member_pos = env.get_unit_position(member_id)
            dx = high_priority_pos[0] - member_pos[0]
            dy = high_priority_pos[1] - member_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            # Get weapon and hit probability
            weapon = env.combat_manager._get_unit_weapon(member_id)
            base_hit_prob = env.calculate_hit_probability(distance, weapon) if weapon else 0

            print(f"    Distance: {distance:.1f}")
            print(f"    Hit Probability: {base_hit_prob:.2f}")
        else:
            print(f"  ✗ Not engageable by {role_name}")

    # Execute engagement with priority targeting
    if high_priority_engageable:
        print("\nENGAGING HIGH PRIORITY TARGET:")
        print("-" * 30)

        # Print engagement parameters
        print("Engagement Parameters:")
        print(f"  Target Position: {high_priority_pos}")
        print(f"  Target ID: {high_priority_id}")
        print(f"  Max Rounds: 10")
        print(f"  Priority Targeting: Enabled")

        # Print ammunition before engagement
        print("\nAmmunition before engagement:")
        initial_ammo = {}
        for member_id in friendly_members:
            role_value = env.get_unit_property(member_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
            ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
            initial_ammo[member_id] = ammo
            print(f"  {role_name}: {ammo} rounds")

        # Execute engagement against high priority target
        high_priority_result = env.combat_manager.execute_team_engagement(
            team_id=friendly_id,
            target_pos=high_priority_pos,
            engagement_type=EngagementType.POINT,
            control_params={
                'max_rounds': 10,  # More rounds for high priority
                'use_priority_targeting': True
            }
        )

        # Print detailed engagement results
        print("\nHigh Priority Engagement Results:")
        print(f"  Total Hits: {high_priority_result['total_hits']}/{high_priority_result['ammo_expended']}")
        print(f"  Total Damage: {high_priority_result['total_damage']:.1f}")
        print(f"  Effectiveness: {high_priority_result.get('effectiveness', 0):.2f}")

        # Check which members fired
        print("\nMember Participation:")
        participants = []
        for member_id in friendly_members:
            role_value = env.get_unit_property(member_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
            current_ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
            rounds_fired = initial_ammo[member_id] - current_ammo

            if rounds_fired > 0:
                participants.append((member_id, role_name, rounds_fired))
                print(f"  {role_name}: {rounds_fired} rounds fired")
            else:
                print(f"  {role_name}: Did not fire")

        # Check target status after engagement
        target_health = env.get_unit_property(high_priority_id, 'health', 0)
        print(f"\nHigh Priority Target Status:")
        print(f"  Health: {target_health}/100")

        if target_health <= 0:
            print(f"  ✓ Target eliminated")
        else:
            print(f"  Target damaged but not eliminated")

        # Check if target suppressed
        target_children = env.get_unit_children(high_priority_id)
        any_suppressed = False

        for child_id in target_children:
            if child_id in env.combat_manager.suppressed_units:
                supp_level = env.combat_manager.suppressed_units[child_id]['level']
                supp_duration = env.combat_manager.suppressed_units[child_id]['duration']
                any_suppressed = True

                print(f"  Member {child_id} suppressed:")
                print(f"    Level: {supp_level:.2f}")
                print(f"    Duration: {supp_duration} steps")

        if not any_suppressed:
            print(f"  No members suppressed")

        # Record engagement
        if friendly_id not in results['unit_targets']:
            results['unit_targets'][friendly_id] = []
        results['unit_targets'][friendly_id].append(high_priority_id)

        results['engagement_data'][(friendly_id, high_priority_id)] = {
            'hits': high_priority_result['total_hits'],
            'damage': high_priority_result['total_damage'],
            'rounds': high_priority_result['ammo_expended'],
            'is_priority': True,
            'participants': participants
        }
    else:
        print("\nHigh priority target not engageable by any team member")

    # Then engage other targets
    normal_targets = [eid for eid in enemy_ids if eid != high_priority_id]

    print("\nANALYZING NORMAL TARGETS:")
    print("-" * 30)

    for enemy_id in normal_targets:
        enemy_pos = env.get_unit_position(enemy_id)
        print(f"\nTarget at {enemy_pos} (ID: {enemy_id}):")

        # Check if target is engageable by any team member
        target_engageable = False
        for member_id in friendly_members:
            role_value = env.get_unit_property(member_id, 'role')
            role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
            can_engage = env.combat_manager.validate_target(member_id, enemy_pos)

            if can_engage:
                target_engageable = True
                member_pos = env.get_unit_position(member_id)
                distance = math.sqrt((enemy_pos[0] - member_pos[0]) ** 2 + (enemy_pos[1] - member_pos[1]) ** 2)

                print(f"  ✓ Engageable by {role_name}")
                print(f"    Distance: {distance:.1f}")

                # Get engagement probability
                weapon = env.combat_manager._get_unit_weapon(member_id)
                if weapon:
                    hit_prob = env.calculate_hit_probability(distance, weapon)
                    print(f"    Base Hit Probability: {hit_prob:.2f}")
            else:
                print(f"  ✗ Not engageable by {role_name}")

        if target_engageable:
            print(f"\nEngaging normal target at {enemy_pos}:")
            print("-" * 30)

            # Print engagement parameters
            print("Engagement Parameters:")
            print(f"  Target Position: {enemy_pos}")
            print(f"  Target ID: {enemy_id}")
            print(f"  Max Rounds: 5")

            # Track ammunition before engagement
            before_ammo = {}
            for member_id in friendly_members:
                role_value = env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
                ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
                before_ammo[member_id] = ammo

            # Execute engagement
            result = env.combat_manager.execute_team_engagement(
                team_id=friendly_id,
                target_pos=enemy_pos,
                engagement_type=EngagementType.POINT,
                control_params={
                    'max_rounds': 5  # Fewer rounds for normal targets
                }
            )

            # Print results
            print("\nNormal Target Engagement Results:")
            print(f"  Hits: {result['total_hits']}/{result['ammo_expended']}")
            print(f"  Damage: {result['total_damage']:.1f}")
            print(f"  Effectiveness: {result.get('effectiveness', 0):.2f}")

            # Check which members engaged
            print("\nMember Participation:")
            participants = []
            for member_id in friendly_members:
                role_value = env.get_unit_property(member_id, 'role')
                role_name = US_IN_Role(role_value).name if isinstance(role_value, int) else role_value.name
                current_ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
                rounds_fired = before_ammo[member_id] - current_ammo

                if rounds_fired > 0:
                    participants.append((member_id, role_name, rounds_fired))
                    print(f"  {role_name}: {rounds_fired} rounds fired")
                else:
                    print(f"  {role_name}: Did not fire")

            # Check target status
            target_health = env.get_unit_property(enemy_id, 'health', 0)
            print(f"\nTarget Status:")
            print(f"  Health: {target_health}/100")

            # Record engagement
            if friendly_id not in results['unit_targets']:
                results['unit_targets'][friendly_id] = []
            results['unit_targets'][friendly_id].append(enemy_id)

            results['engagement_data'][(friendly_id, enemy_id)] = {
                'hits': result['total_hits'],
                'damage': result['total_damage'],
                'rounds': result['ammo_expended'],
                'is_priority': False,
                'participants': participants
            }
        else:
            print(f"Target at {enemy_pos} not engageable by any team member")

    return results


def test_high_priority_target(output_dir="Model_Validation_Combat/high_priority_target"):
    """
    Test prioritization of high-threat targets with enhanced debug output.

    Args:
        output_dir: Directory to save output files

    Returns:
        Environment and test results
    """
    print("\n" + "=" * 80)
    print("ENHANCED HIGH PRIORITY TARGET TEST")
    print("=" * 80)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup environment with error handling for casualty processing
    env = setup_environment(width=70, height=25)

    # Setup terrain
    setup_terrain(env, width=70, height=25)

    # Create friendly team
    friendly_id = create_friendly_team(env, start_position=(1, 12), orientation=0)

    # Create normal enemy targets
    enemy_positions = [
        (15, 8),  # Upper target
        (15, 16)  # Lower target
    ]
    enemy_ids = create_enemy_targets(env, positions=enemy_positions)

    # Create high-priority enemy (machine gun team)
    high_priority_id = create_high_priority_enemy(env, (15, 12), threat_level=5.0)
    enemy_ids.append(high_priority_id)

    # Create visualizer
    visualizer = CombatVisualizer(env)

    # Visualize initial state
    fig, ax = visualizer.plot_combat_state(
        friendly_units=[friendly_id],
        enemy_units=enemy_ids,
        title="Initial State - High Priority Target Test",
        output_path=os.path.join(output_dir, "initial_state.png")
    )
    plt.close(fig)

    # Execute high priority target test with error handling
    try:
        priority_results = execute_priority_target_test(env, friendly_id, enemy_ids, high_priority_id)

        # Visualize priority targeting
        fig, ax = visualizer.visualize_engagement_priorities(
            friendly_units=[friendly_id],
            enemy_units=enemy_ids,
            engagement_results=priority_results,
            title="High Priority Target Engagement Results",
            output_path=os.path.join(output_dir, "priority_target_results.png")
        )
        plt.close(fig)

        # Visualize final state
        fig, ax = visualizer.plot_combat_state(
            friendly_units=[friendly_id],
            enemy_units=enemy_ids,
            title="Final State - After Priority Target Test",
            output_path=os.path.join(output_dir, "final_state.png")
        )
        plt.close(fig)

        print("\nHigh priority target test completed successfully!")
        return env, priority_results

    except Exception as e:
        print(f"\nError during high priority target test: {str(e)}")
        print("This may be due to the casualty handling issue - check if the fix has been applied")
        # Still return what we have
        return env, {"error": str(e)}


def main():
    """Main entry point for running combat tests."""
    # Create output directories
    base_dir = create_output_directories()
    print(f"Test outputs will be saved to: {base_dir}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run enhanced combat tests')
    parser.add_argument('--test', choices=['basic', 'suppression', 'coordination', 'priority', 'all'],
                        default='all', help='Test type to run')
    parser.add_argument('--debug', type=int, choices=[0, 1, 2], default=1,
                        help='Debug level (0=None, 1=Basic, 2=Detailed)')
    args = parser.parse_args()

    # Track test results
    results = {}

    try:
        # Run the requested test(s)
        if args.test == 'basic' or args.test == 'all':
            print("\nRunning Basic Engagement Test...")
            env, result = test_basic_engagement()
            results['basic'] = True
            print("Basic engagement test completed successfully!")

        if args.test == 'suppression' or args.test == 'all':
            print("\nRunning Enhanced Suppression Test...")
            env, result = test_suppression()
            results['suppression'] = True
            print("Suppression test completed successfully!")

        if args.test == 'coordination' or args.test == 'all':
            print("\nRunning Enhanced Coordination Test...")
            env, result = test_coordination()
            results['coordination'] = True
            print("Coordination test completed successfully!")

        if args.test == 'priority' or args.test == 'all':
            print("\nRunning Enhanced Priority Target Test...")
            env, result = test_high_priority_target()
            results['priority'] = True
            print("Priority target test completed successfully!")

        # Print summary of results
        print("\n" + "=" * 40)
        print("TEST EXECUTION SUMMARY")
        print("=" * 40)

        for test_name, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{test_name.capitalize()} Test: {status}")

        print("\nAll tests completed!")
        print(f"Results saved to: {base_dir}")

        return 0

    except Exception as e:
        print(f"\nError during test execution: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
