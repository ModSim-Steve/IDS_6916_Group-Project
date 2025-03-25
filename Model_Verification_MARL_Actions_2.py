"""
Model Verification MARL - Actions Testing Framework

This script provides comprehensive tests for the action space implementation in the MARL
wrapper of the WarGaming Environment. It includes tests for:
1. Movement actions with different directions and distances
2. Engagement actions with targets at different ranges
3. Formation change actions with different formations and orientations
4. Tests for different unit types (infantry teams, weapons teams, squads)

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
from typing import Dict, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt

# Import from main environment file
from WarGamingEnvironment_v9 import UnitType

# Import functions from composition file
from US_Army_PLT_Composition_vTest import US_IN_create_squad

# Import test utilities from existing test file
from Model_Verification_MARL_int_Env_Agents_2 import (
    MARLVisualizer,
    setup_environment,
    create_enemy_targets,
    create_output_directories
)


# Test Configuration Parameters
CONFIG = {
    'width': 100,
    'height': 100,
    'objective_pos': (60, 12),
    'debug_level': 1,
    'platoon_start': (2, 30),
    'enemy_positions': [
        (20, 10), (20, 15),  # Close range
        (35, 10), (35, 15),  # Medium range
        (50, 10), (50, 15)   # Long range
    ],
    'movement_directions': [
        (1, 0),    # East
        (0, 1),    # South
        (-1, 0),   # West
        (0, -1),   # North
        (1, 1),    # Southeast
        (-1, -1)   # Northwest
    ],
    'movement_distances': [1, 5],
    'formations': [
        "team_wedge_right",
        "team_wedge_left",
        "team_line_right",
        "team_line_left",
        "team_column",
        "squad_column_team_wedge",
        "squad_column_team_column",
        "squad_line_team_wedge"
    ],
    'output_dir': 'Model_Verification_MARL/action_space'
}


# Action Generation Functions
def generate_movement_action(
        direction: Tuple[float, float],
        distance: int = 3,
        action_type: int = 0  # 0 for MOVE, 3 for BOUND
) -> Dict:
    """
    Generate a movement action.

    Args:
        direction: (dx, dy) direction vector
        distance: Movement distance
        action_type: 0 for MOVE, 3 for BOUND

    Returns:
        Action dictionary
    """
    # Normalize direction vector
    dx, dy = direction
    magnitude = math.sqrt(dx * dx + dy * dy)
    if magnitude > 0:
        dx = dx / magnitude
        dy = dy / magnitude

    return {
        'action_type': action_type,
        'movement_params': {
            'direction': (dx, dy),
            'distance': np.array([distance], dtype=np.int32)
        },
        'engagement_params': {
            'target_pos': np.array([0, 0], dtype=np.int32),
            'max_rounds': np.array([0], dtype=np.int32),
            'suppress_only': 0,
            'adjust_for_fire_rate': 0
        },
        'formation': 0
    }


def generate_engagement_action(
        target_pos: Tuple[int, int],
        max_rounds: int = 10,
        action_type: int = 1,  # 1 for ENGAGE, 2 for SUPPRESS
        suppress_only: bool = False,
        adjust_for_fire_rate: bool = True
) -> Dict:
    """
    Generate an engagement action.

    Args:
        target_pos: Position to engage
        max_rounds: Maximum rounds to expend
        action_type: 1 for ENGAGE, 2 for SUPPRESS
        suppress_only: Whether to use suppressive fire
        adjust_for_fire_rate: Whether to adjust rounds based on weapon fire rate

    Returns:
        Action dictionary
    """
    if action_type == 2:  # SUPPRESS always has suppress_only=True
        suppress_only = True

    return {
        'action_type': action_type,
        'movement_params': {
            'direction': (0, 0),
            'distance': np.array([0], dtype=np.int32)
        },
        'engagement_params': {
            'target_pos': target_pos,
            'max_rounds': np.array([max_rounds], dtype=np.int32),
            'suppress_only': 1 if suppress_only else 0,
            'adjust_for_fire_rate': 1 if adjust_for_fire_rate else 0
        },
        'formation': 0
    }


def generate_formation_action(formation_index: int) -> Dict:
    """
    Generate a formation change action.

    Args:
        formation_index: Index of formation to apply (0-7)

    Returns:
        Action dictionary
    """
    return {
        'action_type': 4,  # Use formation change action type
        'movement_params': {
            'direction': (0, 0),
            'distance': np.array([0], dtype=np.int32)
        },
        'engagement_params': {
            'target_pos': np.array([0, 0], dtype=np.int32),
            'max_rounds': np.array([0], dtype=np.int32),
            'suppress_only': 0,
            'adjust_for_fire_rate': 0
        },
        'formation': formation_index
    }


def generate_halt_action() -> Dict:
    """
    Generate a halt action.

    Returns:
        Action dictionary
    """
    return {
        'action_type': 4,  # HALT (using same enum as formation change, with formation=0)
        'movement_params': {
            'direction': (0, 0),
            'distance': np.array([0], dtype=np.int32)
        },
        'engagement_params': {
            'target_pos': np.array([0, 0], dtype=np.int32),
            'max_rounds': np.array([0], dtype=np.int32),
            'suppress_only': 0,
            'adjust_for_fire_rate': 0
        },
        'formation': 0
    }


# Action Test Functions
def test_box_movement_pattern(env, output_dir=None):
    """
    Test a squad agent's movement in a box pattern (right, up, left, down).
    Creates visualizations of each step in the movement pattern.

    Args:
        env: MARL environment
        output_dir: Directory to save visualizations

    Returns:
        Dictionary with test results
    """
    print("\n" + "=" * 80)
    print("BOX MOVEMENT PATTERN TEST")
    print("=" * 80)

    # Create visualizer
    visualizer = MARLVisualizer(env)

    # Helper function to create a cleaner visualization focused on the squad
    def visualize_squad(title, squad_id):
        """Create a visualization focused just on the squad."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Get squad position
        squad_pos = env.get_unit_position(squad_id)

        # Get squad components
        squad_leader = None
        alpha_team = None
        bravo_team = None

        for child_id in env.get_unit_children(squad_id):
            if env.get_unit_property(child_id, 'is_leader', False):
                squad_leader = child_id
            else:
                string_id = env.get_unit_property(child_id, 'string_id', '')
                if 'ATM' in string_id or ('TEAM' in string_id and 'A' in string_id):
                    alpha_team = child_id
                elif 'BTM' in string_id or ('TEAM' in string_id and 'B' in string_id):
                    bravo_team = child_id

        # Plot squad position
        ax.plot(squad_pos[0], squad_pos[1], 'bs', markersize=12, label='Squad')

        # Plot squad leader if found
        if squad_leader:
            leader_pos = env.get_unit_position(squad_leader)
            ax.plot(leader_pos[0], leader_pos[1], 'r^', markersize=10, label='Squad Leader')

        # Plot alpha team if found
        if alpha_team:
            alpha_pos = env.get_unit_position(alpha_team)
            ax.plot(alpha_pos[0], alpha_pos[1], 'gs', markersize=8, label='Alpha Team')

            # Plot alpha team members
            for member_id in env.get_unit_children(alpha_team):
                member_pos = env.get_unit_position(member_id)
                ax.plot(member_pos[0], member_pos[1], 'g.', markersize=6)

        # Plot bravo team if found
        if bravo_team:
            bravo_pos = env.get_unit_position(bravo_team)
            ax.plot(bravo_pos[0], bravo_pos[1], 'ms', markersize=8, label='Bravo Team')

            # Plot bravo team members
            for member_id in env.get_unit_children(bravo_team):
                member_pos = env.get_unit_position(member_id)
                ax.plot(member_pos[0], member_pos[1], 'm.', markersize=6)

        # Set title and grid
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc='upper left')

        # Set axis limits to focus on squad
        ax.set_xlim(squad_pos[0] - 20, squad_pos[0] + 20)
        ax.set_ylim(squad_pos[1] - 20, squad_pos[1] + 20)

        return fig, ax

    # Create a single squad directly rather than a full platoon
    print("Creating a single squad for box movement test...")
    squad_id = US_IN_create_squad(env, plt_num=1, squad_num=1, start_position=(40, 40))

    # Register the squad with the agent manager
    env.agent_manager.agent_ids = [squad_id]

    # Since we're using a standalone squad, manually set it as an agent
    env.update_unit_property(squad_id, 'is_agent', True)

    # Check if squad creation was successful
    if not squad_id:
        print("Error: Failed to create squad")
        return {"success": False, "error": "Failed to create squad"}
    squad_string = env.get_unit_property(squad_id, 'string_id', str(squad_id))
    print(f"\nTesting box movement pattern for squad: {squad_string} (ID: {squad_id})")

    # Get initial position
    initial_position = env.get_unit_position(squad_id)
    print(f"Initial position: {initial_position}")

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define box pattern moves
    box_moves = [
        {"direction": (1, 0), "distance": 10, "description": "Right"},  # Right
        {"direction": (0, -1), "distance": 10, "description": "Up"},  # Up
        {"direction": (-1, 0), "distance": 10, "description": "Left"},  # Left
        {"direction": (0, 1), "distance": 10, "description": "Down"}  # Down
    ]

    # Track movement results
    results = {
        "initial_position": initial_position,
        "moves": [],
        "positions": [initial_position],
        "success": True
    }

    # Save initial state using our focused visualization
    if output_dir:
        initial_state = visualize_squad(
            title=f"Initial State: {squad_string} at {initial_position}",
            squad_id=squad_id
        )
        plt.savefig(os.path.join(output_dir, "box_movement_00_initial.png"))
        plt.close(initial_state[0])

    # Plot positions for animation
    all_positions = []

    # Initialize variables for tracking
    squad_leader = None
    alpha_team = None
    bravo_team = None
    template = None

    # Helper function to calculate template position from reference point and template offset
    def calculate_template_position(ref_pos, offset, orientation):
        """Calculate expected position from template offset and orientation"""
        angle_rad = math.radians(orientation - 90)  # Adjust for standard coordinate system
        rot_x = int(offset[0] * math.cos(angle_rad) - offset[1] * math.sin(angle_rad))
        rot_y = int(offset[0] * math.sin(angle_rad) + offset[1] * math.cos(angle_rad))
        return (ref_pos[0] + rot_x, ref_pos[1] + rot_y)

    # Define log file for movement tracking
    log_file = os.path.join(output_dir, "movement_log.txt") if output_dir else None

    # Open log file if specified
    log_handle = None
    if log_file:
        # Use utf-8 encoding to handle special characters
        log_handle = open(log_file, "w", encoding="utf-8")

        # Write header
        log_handle.write("=" * 80 + "\n")
        log_handle.write("SQUAD MOVEMENT LOG - BOX PATTERN TEST\n")
        log_handle.write("=" * 80 + "\n\n")
        log_handle.write(f"Squad ID: {squad_id} ({squad_string})\n")
        log_handle.write(f"Initial Position: {initial_position}\n\n")

        # Log the formation and template
        formation_type = env.get_unit_property(squad_id, 'formation')
        log_handle.write(f"Formation: {formation_type}\n")

        # Get the formation template
        template = env.get_unit_property(squad_id, 'formation_template')
        if template:
            log_handle.write("Template Offsets:\n")
            for key, offset in template.items():
                log_handle.write(f"  {key}: {offset}\n")

        # Log the squad member structure
        log_handle.write("\nSQUAD COMPONENT STRUCTURE:\n")
        log_handle.write("-" * 50 + "\n")

        # Find squad components
        squad_leader = None
        alpha_team = None
        bravo_team = None

        for child_id in env.get_unit_children(squad_id):
            string_id = env.get_unit_property(child_id, 'string_id', '')
            is_leader = env.get_unit_property(child_id, 'is_leader', False)

            if is_leader:
                squad_leader = child_id
                log_handle.write(f"Squad Leader (ID: {squad_leader}): {env.get_unit_position(squad_leader)}\n")
            elif 'ATM' in string_id or ('TEAM' in string_id and 'A' in string_id):
                alpha_team = child_id
                log_handle.write(f"Alpha Team (ID: {alpha_team}): {env.get_unit_position(alpha_team)}\n")

                # Log Alpha team members
                log_handle.write("  Alpha Team Members:\n")
                for member_id in env.get_unit_children(alpha_team):
                    role = env.get_unit_property(member_id, 'role')
                    role_name = str(role)
                    if hasattr(role, 'name'):
                        role_name = role.name
                    log_handle.write(f"    {role_name} (ID: {member_id}): {env.get_unit_position(member_id)}\n")

            elif 'BTM' in string_id or ('TEAM' in string_id and 'B' in string_id):
                bravo_team = child_id
                log_handle.write(f"Bravo Team (ID: {bravo_team}): {env.get_unit_position(bravo_team)}\n")

                # Log Bravo team members
                log_handle.write("  Bravo Team Members:\n")
                for member_id in env.get_unit_children(bravo_team):
                    role = env.get_unit_property(member_id, 'role')
                    role_name = str(role)
                    if hasattr(role, 'name'):
                        role_name = role.name
                    log_handle.write(f"    {role_name} (ID: {member_id}): {env.get_unit_position(member_id)}\n")

        # Log the template interpretation with actual positions
        log_handle.write("\nTEMPLATE VERIFICATION - INITIAL POSITIONS:\n")
        log_handle.write("-" * 50 + "\n")

        # Function to calculate template position from reference point and template offset
        def calculate_template_position(ref_pos, offset, orientation):
            """Calculate expected position from template offset and orientation"""
            angle_rad = math.radians(orientation - 90)  # Adjust for standard coordinate system
            rot_x = int(offset[0] * math.cos(angle_rad) - offset[1] * math.sin(angle_rad))
            rot_y = int(offset[0] * math.sin(angle_rad) + offset[1] * math.cos(angle_rad))
            return (ref_pos[0] + rot_x, ref_pos[1] + rot_y)

        # Get squad orientation for template calculations
        squad_orientation = env.get_unit_property(squad_id, 'orientation')
        log_handle.write(f"Squad Orientation: {squad_orientation}°\n\n")

        # Verify Squad Leader position (should be at squad position)
        if squad_leader:
            squad_pos = env.get_unit_position(squad_id)
            leader_pos = env.get_unit_position(squad_leader)
            log_handle.write(f"Squad Leader:\n")
            log_handle.write(f"  Expected position (squad position): {squad_pos}\n")
            log_handle.write(f"  Actual position: {leader_pos}\n")
            log_handle.write(f"  Offset: {leader_pos[0] - squad_pos[0], leader_pos[1] - squad_pos[1]}\n\n")

        # Verify Alpha Team position
        if alpha_team and 'ATM' in template:
            alpha_offset = template['ATM']
            expected_alpha_pos = calculate_template_position(squad_pos, alpha_offset, squad_orientation)
            actual_alpha_pos = env.get_unit_position(alpha_team)
            log_handle.write(f"Alpha Team:\n")
            log_handle.write(f"  Template offset: {alpha_offset}\n")
            log_handle.write(f"  Expected position: {expected_alpha_pos}\n")
            log_handle.write(f"  Actual position: {actual_alpha_pos}\n")
            log_handle.write(
                f"  Difference: {actual_alpha_pos[0] - expected_alpha_pos[0], actual_alpha_pos[1] - expected_alpha_pos[1]}\n\n")

            # Verify Alpha Team members
            log_handle.write("  Alpha Team Members:\n")
            alpha_orientation = env.get_unit_property(alpha_team, 'orientation')
            alpha_template = env.get_unit_property(alpha_team, 'formation_template')

            for member_id in env.get_unit_children(alpha_team):
                role = env.get_unit_property(member_id, 'role')
                actual_pos = env.get_unit_position(member_id)

                # Convert role to proper format for template lookup
                if isinstance(role, int):
                    try:
                        from US_Army_PLT_Composition_vTest import US_IN_Role
                        role_enum = US_IN_Role(role)
                    except:
                        role_enum = None
                else:
                    role_enum = role

                # Get expected position from template if available
                expected_pos = None
                if role_enum in alpha_template:
                    offset = alpha_template[role_enum]
                    expected_pos = calculate_template_position(actual_alpha_pos, offset, alpha_orientation)

                role_name = str(role)
                if hasattr(role, 'name'):
                    role_name = role.name

                log_handle.write(f"    {role_name}:\n")
                if expected_pos:
                    log_handle.write(f"      Template offset: {alpha_template[role_enum]}\n")
                    log_handle.write(f"      Expected position: {expected_pos}\n")
                    log_handle.write(f"      Actual position: {actual_pos}\n")
                    log_handle.write(
                        f"      Difference: {actual_pos[0] - expected_pos[0], actual_pos[1] - expected_pos[1]}\n")
                else:
                    log_handle.write(f"      Actual position: {actual_pos}\n")
                    log_handle.write(f"      No template offset found for this role\n")

        # Verify Bravo Team position
        if bravo_team and 'BTM' in template:
            bravo_offset = template['BTM']
            expected_bravo_pos = calculate_template_position(squad_pos, bravo_offset, squad_orientation)
            actual_bravo_pos = env.get_unit_position(bravo_team)
            log_handle.write(f"Bravo Team:\n")
            log_handle.write(f"  Template offset: {bravo_offset}\n")
            log_handle.write(f"  Expected position: {expected_bravo_pos}\n")
            log_handle.write(f"  Actual position: {actual_bravo_pos}\n")
            log_handle.write(
                f"  Difference: {actual_bravo_pos[0] - expected_bravo_pos[0], actual_bravo_pos[1] - expected_bravo_pos[1]}\n\n")

            # Verify Bravo Team members
            log_handle.write("  Bravo Team Members:\n")
            bravo_orientation = env.get_unit_property(bravo_team, 'orientation')
            bravo_template = env.get_unit_property(bravo_team, 'formation_template')

            for member_id in env.get_unit_children(bravo_team):
                role = env.get_unit_property(member_id, 'role')
                actual_pos = env.get_unit_position(member_id)

                # Convert role to proper format for template lookup
                if isinstance(role, int):
                    try:
                        from US_Army_PLT_Composition_vTest import US_IN_Role
                        role_enum = US_IN_Role(role)
                    except:
                        role_enum = None
                else:
                    role_enum = role

                # Get expected position from template if available
                expected_pos = None
                if role_enum in bravo_template:
                    offset = bravo_template[role_enum]
                    expected_pos = calculate_template_position(actual_bravo_pos, offset, bravo_orientation)

                role_name = str(role)
                if hasattr(role, 'name'):
                    role_name = role.name

                log_handle.write(f"    {role_name}:\n")
                if expected_pos:
                    log_handle.write(f"      Template offset: {bravo_template[role_enum]}\n")
                    log_handle.write(f"      Expected position: {expected_pos}\n")
                    log_handle.write(f"      Actual position: {actual_pos}\n")
                    log_handle.write(
                        f"      Difference: {actual_pos[0] - expected_pos[0], actual_pos[1] - expected_pos[1]}\n")
                else:
                    log_handle.write(f"      Actual position: {actual_pos}\n")
                    log_handle.write(f"      No template offset found for this role\n")

        log_handle.write("\n" + "=" * 80 + "\n")
        log_handle.write("MOVEMENT EXECUTION LOG\n")
        log_handle.write("=" * 80 + "\n\n")

    # Execute each move in the box pattern
    for i, move in enumerate(box_moves):
        direction = move["direction"]
        distance = move["distance"]
        description = move["description"]

        print(f"\nExecuting move {i + 1}: {description} - Direction {direction}, Distance {distance}")

        # Log movement details
        if log_handle:
            log_handle.write(f"\nMOVE {i + 1}: {description}\n")
            log_handle.write("-" * 50 + "\n")
            log_handle.write(f"Direction: {direction}, Distance: {distance}\n")

            # Calculate expected end position
            start_pos = env.get_unit_position(squad_id)
            expected_end_x = start_pos[0] + direction[0] * distance
            expected_end_y = start_pos[1] + direction[1] * distance
            expected_end_pos = (expected_end_x, expected_end_y)

            log_handle.write(f"Start Position: {start_pos}\n")
            log_handle.write(f"Expected End Position: {expected_end_pos}\n\n")

            # Log the movement steps that will be taken
            log_handle.write("Expected Movement Steps:\n")
            for step in range(1, distance + 1):
                step_x = start_pos[0] + direction[0] * step
                step_y = start_pos[1] + direction[1] * step
                log_handle.write(f"  Step {step}: ({step_x}, {step_y})\n")

            log_handle.write("\nCOMPONENT POSITIONS BEFORE MOVEMENT:\n")
            # Log Squad position
            log_handle.write(f"Squad: {env.get_unit_position(squad_id)}\n")

            # Log Squad Leader position
            if squad_leader:
                log_handle.write(f"Squad Leader: {env.get_unit_position(squad_leader)}\n")

            # Log Alpha Team positions
            if alpha_team:
                log_handle.write(f"Alpha Team: {env.get_unit_position(alpha_team)}\n")
                log_handle.write("  Members:\n")
                for member_id in env.get_unit_children(alpha_team):
                    role = env.get_unit_property(member_id, 'role')
                    role_name = str(role)
                    if hasattr(role, 'name'):
                        role_name = role.name
                    log_handle.write(f"    {role_name}: {env.get_unit_position(member_id)}\n")

            # Log Bravo Team positions
            if bravo_team:
                log_handle.write(f"Bravo Team: {env.get_unit_position(bravo_team)}\n")
                log_handle.write("  Members:\n")
                for member_id in env.get_unit_children(bravo_team):
                    role = env.get_unit_property(member_id, 'role')
                    role_name = str(role)
                    if hasattr(role, 'name'):
                        role_name = role.name
                    log_handle.write(f"    {role_name}: {env.get_unit_position(member_id)}\n")

        # Create movement action
        action = generate_movement_action(direction, distance)

        # Save pre-move state using our focused visualization
        if output_dir:
            pre_move_state = visualize_squad(
                title=f"Pre-Move {i + 1}: {description}",
                squad_id=squad_id
            )
            plt.savefig(os.path.join(output_dir, f"box_movement_{i + 1:02d}_pre_{description.lower()}.png"))
            plt.close(pre_move_state[0])

        try:
            # Execute movement
            print(f"Executing movement action: {action}")

            # First, get all positions before movement for detailed tracking
            squad_positions_before = {}
            squad_positions_before["squad"] = env.get_unit_position(squad_id)

            # Get team IDs
            alpha_team = None
            bravo_team = None
            squad_leader = None

            # Find squad components
            for child_id in env.get_unit_children(squad_id):
                string_id = env.get_unit_property(child_id, 'string_id', '')
                if 'ATM' in string_id or ('TEAM' in string_id and 'A' in string_id):
                    alpha_team = child_id
                elif 'BTM' in string_id or ('TEAM' in string_id and 'B' in string_id):
                    bravo_team = child_id

                if env.get_unit_property(child_id, 'is_leader', False):
                    squad_leader = child_id

            # Track team positions
            if alpha_team:
                squad_positions_before["alpha_team"] = env.get_unit_position(alpha_team)
                # Track alpha team members
                squad_positions_before["alpha_members"] = {}
                for member_id in env.get_unit_children(alpha_team):
                    role = env.get_unit_property(member_id, 'role')
                    role_name = str(role)
                    if hasattr(role, 'name'):
                        role_name = role.name
                    squad_positions_before["alpha_members"][member_id] = {
                        "position": env.get_unit_position(member_id),
                        "role": role_name
                    }

            if bravo_team:
                squad_positions_before["bravo_team"] = env.get_unit_position(bravo_team)
                # Track bravo team members
                squad_positions_before["bravo_members"] = {}
                for member_id in env.get_unit_children(bravo_team):
                    role = env.get_unit_property(member_id, 'role')
                    role_name = str(role)
                    if hasattr(role, 'name'):
                        role_name = role.name
                    squad_positions_before["bravo_members"][member_id] = {
                        "position": env.get_unit_position(member_id),
                        "role": role_name
                    }

            if squad_leader:
                squad_positions_before["squad_leader"] = env.get_unit_position(squad_leader)

            # Use the MARL action space (action 0 for movement)
            # Create proper MARL action format
            marl_action = {
                'action_type': 0,  # MOVE action
                'movement_params': {
                    'direction': direction,
                    'distance': [distance]  # Needs to be a list/array for MARL format
                }
            }

            print(f"Executing MARL action: {marl_action}")

            # Execute the action through the agent manager
            movement_result = env.agent_manager.execute_agent_action(squad_id, marl_action)

            # Get all positions after movement
            squad_positions_after = {}
            squad_positions_after["squad"] = env.get_unit_position(squad_id)

            if alpha_team:
                squad_positions_after["alpha_team"] = env.get_unit_position(alpha_team)
                # Track alpha team members
                squad_positions_after["alpha_members"] = {}
                for member_id in env.get_unit_children(alpha_team):
                    squad_positions_after["alpha_members"][member_id] = {
                        "position": env.get_unit_position(member_id)
                    }

            if bravo_team:
                squad_positions_after["bravo_team"] = env.get_unit_position(bravo_team)
                # Track bravo team members
                squad_positions_after["bravo_members"] = {}
                for member_id in env.get_unit_children(bravo_team):
                    squad_positions_after["bravo_members"][member_id] = {
                        "position": env.get_unit_position(member_id)
                    }

            if squad_leader:
                squad_positions_after["squad_leader"] = env.get_unit_position(squad_leader)

            # Check if movement was successful
            new_position = env.get_unit_position(squad_id)
            print(f"New position: {new_position}")

            # Calculate movement distance
            dx = new_position[0] - results["positions"][-1][0]
            dy = new_position[1] - results["positions"][-1][1]
            actual_distance = math.sqrt(dx * dx + dy * dy)

            # Determine success
            expected_dx = direction[0] * distance
            expected_dy = direction[1] * distance
            expected_distance = math.sqrt(expected_dx * expected_dx + expected_dy * expected_dy)

            move_success = actual_distance > 0

            # Store move result
            move_result = {
                "direction": direction,
                "distance": distance,
                "description": description,
                "expected_position": (
                results["positions"][-1][0] + expected_dx, results["positions"][-1][1] + expected_dy),
                "actual_position": new_position,
                "expected_distance": expected_distance,
                "actual_distance": actual_distance,
                "success": move_success,
                "positions_before": squad_positions_before,
                "positions_after": squad_positions_after
            }

            results["moves"].append(move_result)
            results["positions"].append(new_position)

            if not move_success:
                results["success"] = False

            # Report success/failure
            if move_success:
                print(f"[SUCCESS] Movement successful: Expected {expected_distance:.1f}, actual {actual_distance:.1f}")
            else:
                print(f"[FAILED] Movement failed: Expected {expected_distance:.1f}, actual {actual_distance:.1f}")

            # Log the positions after movement
            if log_handle:
                log_handle.write("\nCOMPONENT POSITIONS AFTER MOVEMENT:\n")
                # Log Squad position
                log_handle.write(f"Squad: {env.get_unit_position(squad_id)}\n")

                # Log Squad Leader position
                if squad_leader:
                    log_handle.write(f"Squad Leader: {env.get_unit_position(squad_leader)}\n")

                # Log Alpha Team positions
                if alpha_team:
                    log_handle.write(f"Alpha Team: {env.get_unit_position(alpha_team)}\n")
                    log_handle.write("  Members:\n")
                    for member_id in env.get_unit_children(alpha_team):
                        role = env.get_unit_property(member_id, 'role')
                        role_name = str(role)
                        if hasattr(role, 'name'):
                            role_name = role.name
                        log_handle.write(f"    {role_name}: {env.get_unit_position(member_id)}\n")

                # Log Bravo Team positions
                if bravo_team:
                    log_handle.write(f"Bravo Team: {env.get_unit_position(bravo_team)}\n")
                    log_handle.write("  Members:\n")
                    for member_id in env.get_unit_children(bravo_team):
                        role = env.get_unit_property(member_id, 'role')
                        role_name = str(role)
                        if hasattr(role, 'name'):
                            role_name = role.name
                        log_handle.write(f"    {role_name}: {env.get_unit_position(member_id)}\n")

                # Calculate and log template verification
                log_handle.write("\nTEMPLATE VERIFICATION AFTER MOVEMENT:\n")

                # Get current squad position and orientation
                squad_pos = env.get_unit_position(squad_id)
                squad_orientation = env.get_unit_property(squad_id, 'orientation')

                # Verify template positions using local calculate_template_position function
                def calculate_template_position(ref_pos, offset, orientation):
                    """Calculate expected position from template offset and orientation"""
                    angle_rad = math.radians(orientation - 90)  # Adjust for standard coordinate system
                    rot_x = int(offset[0] * math.cos(angle_rad) - offset[1] * math.sin(angle_rad))
                    rot_y = int(offset[0] * math.sin(angle_rad) + offset[1] * math.cos(angle_rad))
                    return (ref_pos[0] + rot_x, ref_pos[1] + rot_y)

                if alpha_team and 'ATM' in template:
                    alpha_offset = template['ATM']
                    expected_alpha_pos = calculate_template_position(squad_pos, alpha_offset, squad_orientation)
                    actual_alpha_pos = env.get_unit_position(alpha_team)
                    log_handle.write(f"Alpha Team:\n")
                    log_handle.write(f"  Expected position: {expected_alpha_pos}\n")
                    log_handle.write(f"  Actual position: {actual_alpha_pos}\n")
                    log_handle.write(
                        f"  Difference: {actual_alpha_pos[0] - expected_alpha_pos[0], actual_alpha_pos[1] - expected_alpha_pos[1]}\n")

                if bravo_team and 'BTM' in template:
                    bravo_offset = template['BTM']
                    expected_bravo_pos = calculate_template_position(squad_pos, bravo_offset, squad_orientation)
                    actual_bravo_pos = env.get_unit_position(bravo_team)
                    log_handle.write(f"Bravo Team:\n")
                    log_handle.write(f"  Expected position: {expected_bravo_pos}\n")
                    log_handle.write(f"  Actual position: {actual_bravo_pos}\n")
                    log_handle.write(
                        f"  Difference: {actual_bravo_pos[0] - expected_bravo_pos[0], actual_bravo_pos[1] - expected_bravo_pos[1]}\n")

                # Log movement results
                log_handle.write(f"\nMovement results:\n")
                log_handle.write(f"  Success: {move_success}\n")
                log_handle.write(f"  Expected distance: {expected_distance:.1f}\n")
                log_handle.write(f"  Actual distance: {actual_distance:.1f}\n")
                log_handle.write(f"  Difference: {abs(expected_distance - actual_distance):.1f}\n\n")
                log_handle.write("-" * 80 + "\n")

            # Record positions for animation
            if squad_leader:
                all_positions.append({
                    "step": i + 1,
                    "squad_position": new_position,
                    "squad_leader_position": squad_positions_after.get("squad_leader"),
                    "alpha_team_position": squad_positions_after.get("alpha_team"),
                    "bravo_team_position": squad_positions_after.get("bravo_team")
                })

            # Save post-move state using our focused visualization
            if output_dir:
                post_move_state = visualize_squad(
                    title=f"Post-Move {i + 1}: {description}",
                    squad_id=squad_id
                )
                plt.savefig(os.path.join(output_dir, f"box_movement_{i + 1:02d}_post_{description.lower()}.png"))
                plt.close(post_move_state[0])

        except Exception as e:
            print(f"Error executing movement: {e}")
            traceback.print_exc()

            move_result = {
                "direction": direction,
                "distance": distance,
                "description": description,
                "error": str(e),
                "success": False
            }

            results["moves"].append(move_result)
            results["success"] = False

    # Save final state using our focused visualization
    if output_dir:
        final_state = visualize_squad(
            title=f"Final State: {squad_string} after box movement",
            squad_id=squad_id
        )
        plt.savefig(os.path.join(output_dir, "box_movement_final.png"))
        plt.close(final_state[0])

    # Create combined visualization showing the entire box pattern
    if output_dir and len(results["positions"]) > 1:
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot the box pattern
        x_positions = [pos[0] for pos in results["positions"]]
        y_positions = [pos[1] for pos in results["positions"]]

        # Plot path
        ax.plot(x_positions, y_positions, 'b-', linewidth=2, alpha=0.7, label='Path')

        # Plot start point
        ax.plot(results["positions"][0][0], results["positions"][0][1], 'go', markersize=12, label='Start')

        # Plot end point
        ax.plot(results["positions"][-1][0], results["positions"][-1][1], 'ro', markersize=12, label='End')

        # Plot intermediate points
        for i in range(1, len(results["positions"]) - 1):
            ax.plot(results["positions"][i][0], results["positions"][i][1], 'bo', markersize=8)
            ax.annotate(f"Move {i}",
                        (results["positions"][i][0], results["positions"][i][1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10)

        # Add arrows to show movement direction
        for i in range(len(results["positions"]) - 1):
            start = results["positions"][i]
            end = results["positions"][i + 1]

            # Calculate midpoint for arrow
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2

            # Calculate direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]

            # Add arrow at midpoint
            ax.arrow(mid_x - dx / 4, mid_y - dy / 4, dx / 2, dy / 2,
                     head_width=1.0, head_length=1.5, fc='b', ec='b', alpha=0.7)

        # Add details about each move
        move_descriptions = []
        for i, move in enumerate(results["moves"]):
            success_marker = "[SUCCESS]" if move.get("success", False) else "[FAILED]"
            move_descriptions.append(f"Move {i + 1} ({move['description']}): {success_marker}")

        info_text = "\n".join(move_descriptions)
        overall_result = "[SUCCESS] All moves successful" if results["success"] else "[FAILED] Some moves failed"
        info_text += f"\n\n{overall_result}"

        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Set title and grid
        ax.set_title(f"Box Movement Pattern Test: {squad_string}")
        ax.grid(True)
        ax.legend()

        # Equal aspect ratio
        ax.set_aspect('equal')

        # Save figure
        plt.savefig(os.path.join(output_dir, "box_movement_pattern.png"))
        plt.close(fig)

        # Create animation frames showing movement of squad components
        if all_positions:
            fig, ax = plt.subplots(figsize=(12, 12))

            # Plot the full path for reference
            ax.plot(x_positions, y_positions, 'b--', linewidth=1, alpha=0.3, label='Full Path')

            # Plot start and end points
            ax.plot(results["positions"][0][0], results["positions"][0][1], 'go', markersize=8, label='Start')
            ax.plot(results["positions"][-1][0], results["positions"][-1][1], 'ro', markersize=8, label='End')

            # Create frames directory
            frames_dir = os.path.join(output_dir, "frames")
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)

            # Create animation frames
            for i, pos_data in enumerate([{"squad_position": results["positions"][0]}] + all_positions):
                # Clear previous step's data
                # Remove text elements individually instead of reassigning ax.texts
                for text in ax.texts[:]:
                    if text.get_text().startswith('Step'):
                        text.remove()

                # Remove previous movement lines/points
                for artist in ax.lines + ax.collections:
                    if artist.get_label() in ['Squad', 'Squad Leader', 'Alpha Team', 'Bravo Team', 'Current Path']:
                        artist.remove()

                # Plot squad position
                squad_pos = pos_data.get("squad_position")
                ax.plot(squad_pos[0], squad_pos[1], 'bs', markersize=10, label='Squad')

                # Plot squad leader if available
                if "squad_leader_position" in pos_data:
                    leader_pos = pos_data.get("squad_leader_position")
                    if leader_pos:
                        ax.plot(leader_pos[0], leader_pos[1], 'r^', markersize=8, label='Squad Leader')

                # Plot alpha team if available
                if "alpha_team_position" in pos_data:
                    alpha_pos = pos_data.get("alpha_team_position")
                    if alpha_pos:
                        ax.plot(alpha_pos[0], alpha_pos[1], 'gs', markersize=7, label='Alpha Team')

                # Plot bravo team if available
                if "bravo_team_position" in pos_data:
                    bravo_pos = pos_data.get("bravo_team_position")
                    if bravo_pos:
                        ax.plot(bravo_pos[0], bravo_pos[1], 'ms', markersize=7, label='Bravo Team')

                # Plot path up to current step
                current_path_x = x_positions[:i + 1]
                current_path_y = y_positions[:i + 1]
                ax.plot(current_path_x, current_path_y, 'b-', linewidth=2, alpha=0.7, label='Current Path')

                # Add step indicator
                step_text = f"Step {i}" if i == 0 else f"Step {i}: {box_moves[i - 1]['description']}"
                ax.text(0.5, 0.02, step_text, transform=ax.transAxes,
                        fontsize=12, ha='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Set title and grid
                ax.set_title(f"Box Movement Pattern Animation: {squad_string}")
                ax.grid(True)

                # Create custom legend with unique entries
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper left')

                # Equal aspect ratio
                ax.set_aspect('equal')

                # Set consistent view limits
                min_x = min(x_positions) - 5
                max_x = max(x_positions) + 5
                min_y = min(y_positions) - 5
                max_y = max(y_positions) + 5
                ax.set_xlim(min_x, max_x)
                ax.set_ylim(min_y, max_y)

                # Save frame
                plt.savefig(os.path.join(frames_dir, f"frame_{i:03d}.png"))

            plt.close(fig)

            print(f"\nAnimation frames saved to {frames_dir}")
            print("You can convert these to a GIF using a tool like ImageMagick:")
            print(f"  convert -delay 100 {frames_dir}/frame_*.png {output_dir}/box_movement_animation.gif")

    # Print summary
    print("\nBox Movement Pattern Test Summary:")
    print("-" * 50)
    print(f"Initial Position: {results['initial_position']}")
    print(f"Final Position: {results['positions'][-1]}")
    print(f"Success: {results['success']}")

    for i, move in enumerate(results["moves"]):
        success_str = "[SUCCESS]" if move.get("success", False) else "[FAILED]"
        print(f"Move {i + 1} ({move['description']}): {success_str}")
        if "actual_distance" in move:
            print(f"  Expected distance: {move['expected_distance']:.1f}, Actual: {move['actual_distance']:.1f}")
        if "error" in move:
            print(f"  Error: {move['error']}")

    # Add final summary to log file
    if log_handle:
        log_handle.write("\n\n" + "=" * 80 + "\n")
        log_handle.write("FINAL SUMMARY\n")
        log_handle.write("=" * 80 + "\n\n")
        log_handle.write(f"Initial Position: {results['initial_position']}\n")
        log_handle.write(f"Final Position: {results['positions'][-1]}\n")
        log_handle.write(f"Expected Final Position: Should return to start position\n")
        log_handle.write(
            f"Difference: {results['positions'][-1][0] - results['initial_position'][0]}, {results['positions'][-1][1] - results['initial_position'][1]}\n\n")

        log_handle.write("Movement Results:\n")
        for i, move in enumerate(results["moves"]):
            success_str = "[SUCCESS]" if move.get("success", False) else "[FAILED]"
            log_handle.write(f"Move {i + 1} ({move['description']}): {success_str}\n")
            if "expected_position" in move and "actual_position" in move:
                log_handle.write(f"  Expected position: {move['expected_position']}\n")
                log_handle.write(f"  Actual position: {move['actual_position']}\n")
                log_handle.write(
                    f"  Difference: {move['actual_position'][0] - move['expected_position'][0]}, {move['actual_position'][1] - move['expected_position'][1]}\n")
            if "expected_distance" in move and "actual_distance" in move:
                log_handle.write(f"  Expected distance: {move['expected_distance']:.1f}\n")
                log_handle.write(f"  Actual distance: {move['actual_distance']:.1f}\n")
                log_handle.write(f"  Difference: {abs(move['expected_distance'] - move['actual_distance']):.1f}\n")
            if "error" in move:
                log_handle.write(f"  Error: {move['error']}\n")
            log_handle.write("\n")

        log_handle.write(f"Overall Success: {results['success']}\n")

        # Close the log file
        log_handle.close()
        print(f"\nDetailed movement log written to: {log_file}")

    return results


def test_engagement_actions(env, output_dir=None, visualize=True):
    """
    Test engagement actions using the MARL action space.
    Uses a single squad agent for cleaner visualization and logging.

    Args:
        env: MARL environment
        output_dir: Directory to save visualizations
        visualize: Whether to create visualizations

    Returns:
        Dictionary with test results
    """
    print("\n" + "=" * 80)
    print("ENGAGEMENT ACTIONS TEST (MARL ACTION SPACE)")
    print("=" * 80)

    # Create visualizer if needed
    visualizer = MARLVisualizer(env) if visualize else None

    # Create a single squad directly for cleaner testing
    print("Creating a single squad for engagement test...")
    squad_id = US_IN_create_squad(env, plt_num=1, squad_num=1, start_position=(40, 40))

    # Register the squad with the agent manager
    env.agent_manager.agent_ids = [squad_id]

    # Since we're using a standalone squad, manually set it as an agent
    env.update_unit_property(squad_id, 'is_agent', True)

    # Check if squad creation was successful
    if not squad_id:
        print("Error: Failed to create squad")
        return {"success": False, "error": "Failed to create squad"}

    squad_string = env.get_unit_property(squad_id, 'string_id', str(squad_id))
    print(f"Testing engagement actions for squad: {squad_string} (ID: {squad_id})")

    # Create enemy targets
    enemy_positions = CONFIG['enemy_positions']
    enemy_ids = create_enemy_targets(env, positions=enemy_positions)

    # Track results
    results = {
        'successful_engagements': 0,
        'successful_validations': 0,
        'unexpected_failures': 0,
        'total_tests': 0,
        'total_validations': 0,
        'details': []
    }

    # Print enemy positions for reference
    print("\nEnemy targets for engagement tests:")
    for i, enemy_id in enumerate(enemy_ids):
        enemy_pos = env.get_unit_position(enemy_id)
        print(f"Enemy {enemy_id}: position {enemy_pos}")

    # Test each enemy target with different engagement parameters
    engagement_types = [1, 2]  # 1 for ENGAGE, 2 for SUPPRESS
    round_counts = [6, 12, 24]

    # Get squad position for distance calculations
    squad_pos = env.get_unit_position(squad_id)

    # Print detailed squad component information
    print("\nSQUAD ENGAGEMENT DIAGNOSTIC:")

    # Identify squad components
    alpha_team = None
    bravo_team = None
    squad_leader = None

    for child_id in env.get_unit_children(squad_id):
        string_id = env.get_unit_property(child_id, 'string_id', '')
        is_leader = env.get_unit_property(child_id, 'is_leader', False)

        if 'ATM' in string_id or ('TEAM' in string_id and 'A' in string_id):
            alpha_team = child_id
            print(f"Alpha Team: {child_id} ({string_id})")
        elif 'BTM' in string_id or ('TEAM' in string_id and 'B' in string_id):
            bravo_team = child_id
            print(f"Bravo Team: {child_id} ({string_id})")

        if is_leader:
            squad_leader = child_id
            print(f"Squad Leader: {child_id}")

    # Print squad position
    print(f"Squad position: {squad_pos}")

    # Alpha team details
    if alpha_team:
        alpha_pos = env.get_unit_position(alpha_team)
        print(f"\nAlpha Team position: {alpha_pos}")
        print("Alpha Team members:")
        for member_id in env.get_unit_children(alpha_team):
            role = env.get_unit_property(member_id, 'role')
            role_name = role.name if hasattr(role, 'name') else str(role)
            pos = env.get_unit_position(member_id)
            print(f"  Member {member_id} ({role_name}): {pos}")

    # Bravo team details
    if bravo_team:
        bravo_pos = env.get_unit_position(bravo_team)
        print(f"\nBravo Team position: {bravo_pos}")
        print("Bravo Team members:")
        for member_id in env.get_unit_children(bravo_team):
            role = env.get_unit_property(member_id, 'role')
            role_name = role.name if hasattr(role, 'name') else str(role)
            pos = env.get_unit_position(member_id)
            print(f"  Member {member_id} ({role_name}): {pos}")

    # Squad leader details
    if squad_leader:
        sl_pos = env.get_unit_position(squad_leader)
        role = env.get_unit_property(squad_leader, 'role')
        role_name = role.name if hasattr(role, 'name') else str(role)
        print(f"\nSquad Leader position: {sl_pos}")
        print(f"Squad Leader role: {role_name}")

    # Test each enemy target
    for i, enemy_id in enumerate(enemy_ids):
        enemy_pos = env.get_unit_position(enemy_id)

        # Calculate distance to enemy
        dx = enemy_pos[0] - squad_pos[0]
        dy = enemy_pos[1] - squad_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        print(f"\nTargeting enemy {enemy_id} at position {enemy_pos} (distance: {distance:.1f})")

        # Reset enemy health to ensure consistent testing
        env.update_unit_property(enemy_id, 'health', 100)

        # Reset health for all enemy components if it's a team/squad
        enemy_members = env.get_unit_children(enemy_id)
        if enemy_members:
            for member_id in enemy_members:
                env.update_unit_property(member_id, 'health', 100)

        # Perform target validation analysis
        can_engage_unit = False
        engagement_diagnostic = []

        # Check each member's ability to engage
        print("\nMEMBER ENGAGEMENT VALIDATION:")
        for member_id in env.get_unit_children(squad_id):
            role = env.get_unit_property(member_id, 'role')
            role_name = str(role)
            if hasattr(role, 'name'):
                role_name = role.name
            member_pos = env.get_unit_position(member_id)

            # Calculate angle and distance for diagnostic
            member_dx = enemy_pos[0] - member_pos[0]
            member_dy = enemy_pos[1] - member_pos[1]
            member_distance = math.sqrt(member_dx * member_dx + member_dy * member_dy)
            member_angle = (math.degrees(math.atan2(member_dy, member_dx)) + 360) % 360

            # Get engagement parameters
            engagement_range = env.get_unit_property(member_id, 'engagement_range', 40)

            # Get sector of fire information
            primary_start = env.get_unit_property(member_id, 'primary_sector_rotated_start')
            primary_end = env.get_unit_property(member_id, 'primary_sector_rotated_end')

            # Check individual engagement factors
            in_range = member_distance <= engagement_range

            in_sector = False
            if primary_start is not None and primary_end is not None:
                if primary_start <= primary_end:
                    in_sector = primary_start <= member_angle <= primary_end
                else:
                    in_sector = member_angle >= primary_start or member_angle <= primary_end

            # Check line of sight
            los_result = env.visibility_manager.check_line_of_sight(member_pos, enemy_pos)
            has_los = los_result['has_los']

            # Determine if this member can engage
            member_can_engage = in_range and in_sector and has_los
            if member_can_engage:
                can_engage_unit = True

            # Print diagnostic for this member
            print(f"  Member {member_id} ({role_name}):")
            print(f"    Position: {member_pos}, Angle to target: {member_angle:.1f}°, Distance: {member_distance:.1f}")
            print(f"    Engagement range: {engagement_range}, In range: {in_range}")
            if primary_start is not None and primary_end is not None:
                print(f"    Sector of fire: {primary_start}°-{primary_end}°, In sector: {in_sector}")
            print(f"    Line of sight: {has_los}")
            print(f"    Can engage: {member_can_engage}")

            # Save member diagnostic
            engagement_diagnostic.append({
                'member_id': member_id,
                'role': role_name,
                'position': member_pos,
                'angle_to_target': member_angle,
                'distance_to_target': member_distance,
                'engagement_range': engagement_range,
                'in_range': in_range,
                'in_sector': in_sector,
                'has_los': has_los,
                'can_engage': member_can_engage
            })

        print(f"\nSquad {squad_id} can engage enemy {enemy_id}: {can_engage_unit}")

        # Test different engagement types (ENGAGE and SUPPRESS)
        for engagement_type in engagement_types:
            eng_type_name = "ENGAGE" if engagement_type == 1 else "SUPPRESS"

            # Test different round counts
            for max_rounds in round_counts:
                # Reset enemy health for consistent testing
                env.update_unit_property(enemy_id, 'health', 100)

                # Reset health for enemy components
                if enemy_members:
                    for member_id in enemy_members:
                        env.update_unit_property(member_id, 'health', 100)

                # Track test case
                test_case = {
                    'unit_id': squad_id,
                    'enemy_id': enemy_id,
                    'engagement_type': eng_type_name,
                    'distance': distance,
                    'max_rounds': max_rounds,
                    'success': False,
                    'rounds_expended': 0,
                    'hits': 0,
                    'should_engage': can_engage_unit,
                    'member_diagnostics': engagement_diagnostic,
                    'description': f"{eng_type_name} enemy {enemy_id} at distance {distance:.1f} with {max_rounds} rounds"
                }

                print(f"\n{eng_type_name} with {max_rounds} max rounds...")

                # Track initial ammunition
                initial_ammo = {}
                if hasattr(env, 'combat_manager'):
                    for member_id in env.get_unit_children(squad_id):
                        initial_ammo[member_id] = env.combat_manager._get_unit_ammo(member_id, 'primary')

                # Save pre-engagement state for visualization
                if visualize:
                    pre_action_state = visualizer.plot_environment_state(
                        title=f"Pre-{eng_type_name}: {squad_string} targeting enemy {enemy_id}",
                        highlight_agent=squad_id
                    )
                    plt.close(pre_action_state[0])

                try:
                    # Create a proper MARL action for engagement
                    marl_action = {
                        'action_type': engagement_type,  # 1 for ENGAGE, 2 for SUPPRESS
                        'engagement_params': {
                            'target_pos': enemy_pos,  # Target position
                            'max_rounds': [max_rounds],  # Rounds as array/list for MARL format
                            'suppress_only': 1 if engagement_type == 2 else 0,  # True for SUPPRESS
                            'adjust_for_fire_rate': 1  # Adjust for weapon fire rate
                        }
                    }

                    # Execute the action through the agent manager
                    result = env.agent_manager.execute_agent_action(squad_id, marl_action)

                    # Extract result data
                    result_ammo_expended = 0
                    result_hits = 0
                    result_damage = 0.0
                    result_suppression = 0.0
                    result_participating_teams = []

                    if result:
                        if isinstance(result, dict):
                            result_ammo_expended = result.get('ammo_expended', 0)
                            result_hits = result.get('total_hits', 0)
                            result_damage = result.get('total_damage', 0.0)
                            result_suppression = result.get('suppression_level', 0.0)
                            result_participating_teams = result.get('participating_teams', [])
                        else:
                            result_ammo_expended = getattr(result, 'rounds_expended', 0)
                            result_hits = getattr(result, 'hits', 0)
                            result_damage = getattr(result, 'damage_dealt', 0.0)
                            result_suppression = getattr(result, 'suppression_effect', 0.0)

                    # Determine if engagement occurred
                    engagement_occurred = result_ammo_expended > 0

                    # Print ammunition usage
                    if hasattr(env, 'combat_manager'):
                        print("\nAmmunition usage:")
                        for member_id in env.get_unit_children(squad_id):
                            role = env.get_unit_property(member_id, 'role')
                            role_name = str(role)
                            if hasattr(role, 'name'):
                                role_name = role.name

                            post_ammo = env.combat_manager._get_unit_ammo(member_id, 'primary')
                            used = max(0, initial_ammo.get(member_id, 0) - post_ammo)

                            print(f"  Member {member_id} ({role_name}): Used {used} rounds")

                    # Update test case with result data
                    test_case['rounds_expended'] = result_ammo_expended
                    test_case['hits'] = result_hits
                    test_case['damage'] = result_damage
                    test_case['suppression'] = result_suppression
                    test_case['participating_teams'] = result_participating_teams

                    # Check participating teams
                    if isinstance(result, dict):
                        participating_teams = result.get('participating_teams', [])
                        print(f"\nParticipating teams in squad engagement: {participating_teams}")

                        if alpha_team and alpha_team in participating_teams:
                            print(f"✓ Alpha Team participated in engagement")
                        elif alpha_team:
                            print(f"✗ Alpha Team did NOT participate in engagement")

                        if bravo_team and bravo_team in participating_teams:
                            print(f"✓ Bravo Team participated in engagement")
                        elif bravo_team:
                            print(f"✗ Bravo Team did NOT participate in engagement")

                    # Update result tracking
                    if can_engage_unit:
                        # Valid target case
                        results['total_tests'] += 1

                        if engagement_occurred:
                            # Success - engagement occurred when it should
                            results['successful_engagements'] += 1

                            if engagement_type == 1:  # ENGAGE
                                print(
                                    f"✓ Engagement successful: {result_hits} hits, {result_damage:.1f} damage with {result_ammo_expended} rounds")
                            else:  # SUPPRESS
                                print(
                                    f"✓ Suppression successful: {result_suppression:.2f} suppression effect with {result_ammo_expended} rounds")
                        else:
                            # Failure - no engagement when it should have occurred
                            results['unexpected_failures'] += 1

                            print(f"✗ Engagement failed: No rounds expended (but should have worked)")
                    else:
                        # Invalid target case
                        results['total_validations'] += 1

                        if not engagement_occurred:
                            # Success - no engagement with invalid target
                            results['successful_validations'] += 1
                            print(f"✓ Validation successful: No engagement occurred as expected")
                        else:
                            # Unexpected engagement with invalid target
                            print(f"✗ Validation failed: Engagement succeeded when it should have failed")

                    # Set success flag
                    test_case['success'] = (
                            (can_engage_unit and engagement_occurred) or
                            (not can_engage_unit and not engagement_occurred)
                    )

                    # Save post-engagement state for visualization
                    if visualize:
                        post_action_state = visualizer.plot_environment_state(
                            title=f"Post-{eng_type_name}: {squad_string} targeted enemy {enemy_id}",
                            highlight_agent=squad_id
                        )
                        plt.close(post_action_state[0])

                        # Create engagement visualization
                        if output_dir:
                            fig, ax = plt.subplots(figsize=(10, 8))

                            # Plot squad and enemy positions
                            ax.plot(squad_pos[0], squad_pos[1], 'bo', label=f'{squad_string}')
                            ax.plot(enemy_pos[0], enemy_pos[1], 'ro', label=f'Enemy {enemy_id}')

                            # Draw line from squad to enemy
                            ax.plot([squad_pos[0], enemy_pos[0]], [squad_pos[1], enemy_pos[1]], 'r--', alpha=0.6)

                            # Add engagement info
                            engagement_info = []
                            engagement_info.append(f"Engagement: {eng_type_name}")
                            engagement_info.append(f"Max Rounds: {max_rounds}")
                            engagement_info.append(f"Should Engage: {can_engage_unit}")

                            if result:
                                engagement_info.append(f"Rounds Used: {result_ammo_expended}")
                                engagement_info.append(f"Hits: {result_hits}")

                            info_text = "\n".join(engagement_info)

                            ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                                    fontsize=10, va='top',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                            # Set title and grid
                            ax.set_title(f"Engagement Test: {squad_string} -> Enemy {enemy_id}")
                            ax.grid(True)
                            ax.legend()

                            # Save figure
                            plt.savefig(os.path.join(output_dir,
                                                     f"engagement_squad_{squad_id}_{eng_type_name}_enemy_{enemy_id}_rounds_{max_rounds}.png"))
                            plt.close(fig)

                except Exception as e:
                    print(f"Error executing engagement action: {e}")
                    import traceback
                    traceback.print_exc()
                    test_case['error'] = str(e)

                # Append test case details
                results['details'].append(test_case)

    # Calculate overall success rates
    total_success = results['successful_engagements'] + results['successful_validations']
    total_tests = results['total_tests'] + results['total_validations']

    # Add for backward compatibility
    results['success'] = total_success
    results['total'] = total_tests

    # Print summary
    print("\nEngagement Actions Test Summary:")
    print("-" * 50)

    # Engagement success rate
    if results['total_tests'] > 0:
        engagement_rate = (results['successful_engagements'] / results['total_tests'] * 100)
        print(
            f"  Engagement tests: {results['successful_engagements']}/{results['total_tests']} successful ({engagement_rate:.1f}%)")
    else:
        print(f"  Engagement tests: 0/0 successful (0.0%)")

    # Validation success rate
    if results['total_validations'] > 0:
        validation_rate = (results['successful_validations'] / results['total_validations'] * 100)
        print(
            f"  Validation tests: {results['successful_validations']}/{results['total_validations']} successful ({validation_rate:.1f}%)")
    else:
        print(f"  Validation tests: 0/0 successful (0.0%)")

    # Unexpected failures
    if results['unexpected_failures'] > 0:
        print(f"  Unexpected failures: {results['unexpected_failures']}")

    # Overall success rate
    if total_tests > 0:
        overall_rate = (total_success / total_tests * 100)
        print(f"  Overall success: {total_success}/{total_tests} ({overall_rate:.1f}%)")

    return results


def test_formation_actions(env, output_dir=None, visualize=True):
    """
    Test formation change actions using the MARL action space.
    Uses a single squad agent for cleaner visualization and logging.

    Args:
        env: MARL environment
        output_dir: Directory to save visualizations
        visualize: Whether to create visualizations

    Returns:
        Dictionary with test results
    """
    print("\n" + "=" * 80)
    print("FORMATION ACTIONS TEST (MARL ACTION SPACE)")
    print("=" * 80)

    # Create visualizer if needed
    visualizer = MARLVisualizer(env) if visualize else None

    # Create a single squad directly for cleaner testing
    print("Creating a single squad for formation test...")
    squad_id = US_IN_create_squad(env, plt_num=1, squad_num=1, start_position=(40, 40))

    # Register the squad with the agent manager
    env.agent_manager.agent_ids = [squad_id]

    # Since we're using a standalone squad, manually set it as an agent
    env.update_unit_property(squad_id, 'is_agent', True)

    # Check if squad creation was successful
    if not squad_id:
        print("Error: Failed to create squad")
        return {"success": False, "error": "Failed to create squad"}

    squad_string = env.get_unit_property(squad_id, 'string_id', str(squad_id))
    print(f"Testing formation actions for squad: {squad_string} (ID: {squad_id})")

    # Track results
    results = {
        'success': 0,
        'total': 0,
        'details': []
    }

    # Get initial formation
    initial_formation = env.get_unit_property(squad_id, 'formation')
    print(f"Initial formation: {initial_formation}")

    # Define valid squad formations
    valid_formations = [
        "squad_column_team_wedge",
        "squad_column_team_column",
        "squad_line_team_wedge"
    ]

    # Get formation indices from CONFIG
    formation_mapping = {}
    for i, formation in enumerate(CONFIG['formations']):
        formation_mapping[formation] = i

    # Test each valid formation
    for formation in valid_formations:
        # Get the formation index from mapping
        if formation in formation_mapping:
            formation_index = formation_mapping[formation]
        else:
            # If not in CONFIG, assign a default index
            formation_index = len(formation_mapping)
            formation_mapping[formation] = formation_index

        print(f"\nChanging formation to {formation} (index: {formation_index})...")

        # Track test case
        test_case = {
            'unit_id': squad_id,
            'formation': formation,
            'formation_index': formation_index,
            'initial_formation': initial_formation,
            'success': False,
            'description': f"Change squad formation to {formation}"
        }

        # Save pre-formation state for visualization
        if visualize:
            pre_action_state = visualizer.plot_environment_state(
                title=f"Pre-Formation Change: {squad_string} to {formation}",
                highlight_agent=squad_id
            )
            plt.close(pre_action_state[0])

        try:
            # Create a proper MARL action for formation change
            marl_action = {
                'action_type': 4,  # Formation change action type
                'formation': formation_index
            }

            # Execute the action through the agent manager
            result = env.agent_manager.execute_agent_action(squad_id, marl_action)

            # Check if formation was successfully applied
            new_formation = env.get_unit_property(squad_id, 'formation')
            success = new_formation == formation

            # Update test case
            test_case['success'] = success
            test_case['new_formation'] = new_formation
            test_case['result'] = str(result)

            # Update results tracking
            results['total'] += 1
            if success:
                results['success'] += 1
                print(f"✓ Formation change successful: {initial_formation} -> {new_formation}")
            else:
                print(f"✗ Formation change failed: Still {new_formation} instead of {formation}")

            # Save post-formation state for visualization
            if visualize:
                post_action_state = visualizer.plot_environment_state(
                    title=f"Post-Formation Change: {squad_string} to {formation}",
                    highlight_agent=squad_id
                )
                plt.close(post_action_state[0])

                # Create formation visualization
                if output_dir:
                    fig, ax = plt.subplots(figsize=(12, 10))

                    # Plot unit and its members
                    unit_pos = env.get_unit_position(squad_id)
                    ax.plot(unit_pos[0], unit_pos[1], 'bs', markersize=10, label=f'{squad_string}')

                    # Identify squad components
                    alpha_team = None
                    bravo_team = None
                    squad_leader = None

                    for child_id in env.get_unit_children(squad_id):
                        string_id = env.get_unit_property(child_id, 'string_id', '')
                        if 'ATM' in string_id or ('TEAM' in string_id and 'A' in string_id):
                            alpha_team = child_id
                        elif 'BTM' in string_id or ('TEAM' in string_id and 'B' in string_id):
                            bravo_team = child_id
                        if env.get_unit_property(child_id, 'is_leader', False):
                            squad_leader = child_id

                    # Plot squad leader
                    if squad_leader:
                        sl_pos = env.get_unit_position(squad_leader)
                        ax.plot(sl_pos[0], sl_pos[1], 'ro', markersize=8, label='Squad Leader')
                        ax.plot([unit_pos[0], sl_pos[0]], [unit_pos[1], sl_pos[1]], 'k--', alpha=0.4)

                    # Plot alpha team
                    if alpha_team:
                        alpha_pos = env.get_unit_position(alpha_team)
                        ax.plot(alpha_pos[0], alpha_pos[1], 'go', markersize=8, label='Alpha Team')
                        ax.plot([unit_pos[0], alpha_pos[0]], [unit_pos[1], alpha_pos[1]], 'k--', alpha=0.4)

                        # Plot alpha team members
                        for member_id in env.get_unit_children(alpha_team):
                            member_pos = env.get_unit_position(member_id)
                            is_leader = env.get_unit_property(member_id, 'is_leader', False)
                            marker = 'r^' if is_leader else 'g.'
                            label = 'Alpha TL' if is_leader else None
                            ax.plot(member_pos[0], member_pos[1], marker, markersize=6, label=label)
                            ax.plot([alpha_pos[0], member_pos[0]], [alpha_pos[1], member_pos[1]], 'g--', alpha=0.2)

                    # Plot bravo team
                    if bravo_team:
                        bravo_pos = env.get_unit_position(bravo_team)
                        ax.plot(bravo_pos[0], bravo_pos[1], 'mo', markersize=8, label='Bravo Team')
                        ax.plot([unit_pos[0], bravo_pos[0]], [unit_pos[1], bravo_pos[1]], 'k--', alpha=0.4)

                        # Plot bravo team members
                        for member_id in env.get_unit_children(bravo_team):
                            member_pos = env.get_unit_position(member_id)
                            is_leader = env.get_unit_property(member_id, 'is_leader', False)
                            marker = 'r^' if is_leader else 'm.'
                            label = 'Bravo TL' if is_leader else None
                            ax.plot(member_pos[0], member_pos[1], marker, markersize=6, label=label)
                            ax.plot([bravo_pos[0], member_pos[0]], [bravo_pos[1], member_pos[1]], 'm--', alpha=0.2)

                    # Add formation info
                    info_text = f"Formation: {formation}\nSuccess: {success}"
                    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                            fontsize=10, va='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    # Set title and grid
                    ax.set_title(f"Formation Test: {squad_string} to {formation}")
                    ax.grid(True)

                    # Create custom legend with unique entries
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())

                    # Save figure
                    plt.savefig(os.path.join(output_dir, f"formation_squad_{squad_id}_{formation}.png"))
                    plt.close(fig)

        except Exception as e:
            print(f"Error executing formation action: {e}")
            import traceback
            traceback.print_exc()
            test_case['error'] = str(e)

        # Append test case details
        results['details'].append(test_case)

    # Print summary
    print("\nFormation Actions Test Summary:")
    print("-" * 50)
    success_rate = (results['success'] / results['total'] * 100) if results['total'] > 0 else 0
    print(f"Squad formation changes: {results['success']}/{results['total']} successful ({success_rate:.1f}%)")

    # Restore original formation
    try:
        if initial_formation:
            print(f"\nRestoring original formation: {initial_formation}")

            # Use MARL action to restore initial formation
            if initial_formation in formation_mapping:
                restore_formation_index = formation_mapping[initial_formation]
            else:
                # If not in mapping, use a default formation
                restore_formation_index = 0

            restore_action = {
                'action_type': 4,  # Formation change
                'formation': restore_formation_index
            }

            env.agent_manager.execute_agent_action(squad_id, restore_action)

            # Verify restoration
            final_formation = env.get_unit_property(squad_id, 'formation')
            print(f"Final formation: {final_formation}")
    except Exception as e:
        print(f"Error restoring original formation: {e}")

    return results


def test_action_combination(env, output_dir=None, visualize=True):
    """
    Test combinations of actions using the MARL action space.
    Uses a single squad agent for cleaner visualization and logging.

    Args:
        env: MARL environment
        output_dir: Directory to save visualizations
        visualize: Whether to create visualizations

    Returns:
        Dictionary with test results
    """
    print("\n" + "=" * 80)
    print("ACTION COMBINATION TEST (MARL ACTION SPACE)")
    print("=" * 80)

    # Create visualizer if needed
    visualizer = MARLVisualizer(env) if visualize else None

    # Create a single squad directly for cleaner testing
    print("Creating a single squad for action combination test...")
    squad_id = US_IN_create_squad(env, plt_num=1, squad_num=1, start_position=(40, 40))

    # Register the squad with the agent manager
    env.agent_manager.agent_ids = [squad_id]

    # Since we're using a standalone squad, manually set it as an agent
    env.update_unit_property(squad_id, 'is_agent', True)

    # Check if squad creation was successful
    if not squad_id:
        print("Error: Failed to create squad")
        return {"success": False, "error": "Failed to create squad"}

    squad_string = env.get_unit_property(squad_id, 'string_id', str(squad_id))
    print(f"Testing action combinations for squad: {squad_string} (ID: {squad_id})")

    # Create enemy targets
    enemy_positions = CONFIG['enemy_positions']
    enemy_ids = create_enemy_targets(env, positions=enemy_positions)

    # Track results
    results = {
        'success': 0,
        'total': 0,
        'details': []
    }

    # Get initial state
    initial_position = env.get_unit_position(squad_id)
    initial_formation = env.get_unit_property(squad_id, 'formation')

    print(f"Initial position: {initial_position}, Formation: {initial_formation}")

    # Define formation mapping
    formation_mapping = {}
    for i, formation in enumerate(CONFIG['formations']):
        formation_mapping[formation] = i

    # Define action sequences using MARL action format
    action_sequences = [
        # Sequence 1: Move, Engage, Move
        [
            {
                'name': 'move',
                'action': {
                    'action_type': 0,  # MOVE
                    'movement_params': {
                        'direction': (1, 0),
                        'distance': [5]
                    }
                }
            },
            {
                'name': 'engage',
                'action': {
                    'action_type': 1,  # ENGAGE
                    'engagement_params': {
                        'target_pos': enemy_positions[0],
                        'max_rounds': [10],
                        'suppress_only': 0,
                        'adjust_for_fire_rate': 1
                    }
                }
            },
            {
                'name': 'move',
                'action': {
                    'action_type': 0,  # MOVE
                    'movement_params': {
                        'direction': (0, 1),
                        'distance': [3]
                    }
                }
            }
        ],
        # Sequence 2: Change formation, Move, Suppress
        [
            {
                'name': 'formation',
                'action': {
                    'action_type': 4,  # FORMATION
                    'formation': formation_mapping.get("squad_line_team_wedge", 7)
                }
            },
            {
                'name': 'move',
                'action': {
                    'action_type': 0,  # MOVE
                    'movement_params': {
                        'direction': (1, 1),
                        'distance': [4]
                    }
                }
            },
            {
                'name': 'suppress',
                'action': {
                    'action_type': 2,  # SUPPRESS
                    'engagement_params': {
                        'target_pos': enemy_positions[1],
                        'max_rounds': [15],
                        'suppress_only': 1,
                        'adjust_for_fire_rate': 1
                    }
                }
            }
        ],
        # Sequence 3: Move, Bound, Engage
        [
            {
                'name': 'move',
                'action': {
                    'action_type': 0,  # MOVE
                    'movement_params': {
                        'direction': (1, 0),
                        'distance': [3]
                    }
                }
            },
            {
                'name': 'bound',
                'action': {
                    'action_type': 3,  # BOUND
                    'movement_params': {
                        'direction': (1, 0),
                        'distance': [3]
                    }
                }
            },
            {
                'name': 'engage',
                'action': {
                    'action_type': 1,  # ENGAGE
                    'engagement_params': {
                        'target_pos': enemy_positions[2],
                        'max_rounds': [8],
                        'suppress_only': 0,
                        'adjust_for_fire_rate': 1
                    }
                }
            }
        ]
    ]

    # Test each sequence
    for seq_idx, sequence in enumerate(action_sequences):
        sequence_name = f"Sequence {seq_idx + 1}"
        action_names = [action['name'] for action in sequence]
        print(f"\nExecuting {sequence_name}: {action_names}")

        # Track sequence results
        sequence_results = {
            'unit_id': squad_id,
            'sequence': sequence_name,
            'actions': [],
            'success': True,  # Will be set to False if any action fails
            'description': f"Execute {action_names} sequence on squad"
        }

        # Save initial state for visualization
        if visualize:
            initial_state = visualizer.plot_environment_state(
                title=f"Initial State: {squad_string} before {sequence_name}",
                highlight_agent=squad_id
            )
            plt.close(initial_state[0])

        # Execute each action in sequence
        action_frames = []  # For animation

        for action_idx, action_info in enumerate(sequence):
            action_name = action_info['name']
            action = action_info['action']

            print(f"\nStep {action_idx + 1}: {action_name.upper()}")
            print(f"Action details: {action}")

            # Track action result
            action_result = {
                'action_name': action_name,
                'action_details': action,
                'success': False,
                'error': None
            }

            # Make sure enemy targets have full health for each engagement action
            if action_name in ['engage', 'suppress']:
                target_pos = action['engagement_params']['target_pos']
                # Find enemy at this position
                for enemy_id in enemy_ids:
                    enemy_pos = env.get_unit_position(enemy_id)
                    if enemy_pos == tuple(target_pos):
                        print(f"Resetting health for enemy {enemy_id} to 100")
                        env.update_unit_property(enemy_id, 'health', 100)
                        # Reset health for enemy components if it's a team/squad
                        enemy_members = env.get_unit_children(enemy_id)
                        if enemy_members:
                            for member_id in enemy_members:
                                env.update_unit_property(member_id, 'health', 100)

            # Get position before action
            position_before = env.get_unit_position(squad_id)

            try:
                # Execute the action through agent manager
                result = env.agent_manager.execute_agent_action(squad_id, action)

                # Get position after action
                position_after = env.get_unit_position(squad_id)

                # Determine success based on action type
                if action_name in ['move', 'bound']:
                    # Movement is successful if position changed
                    movement_vector = (
                        position_after[0] - position_before[0],
                        position_after[1] - position_before[1]
                    )
                    expected_direction = action['movement_params']['direction']

                    # Calculate dot product to check direction alignment
                    dot_product = (movement_vector[0] * expected_direction[0] +
                                   movement_vector[1] * expected_direction[1])

                    success = (position_after != position_before) and (dot_product > 0)
                    action_result['success'] = success
                    action_result['position_before'] = position_before
                    action_result['position_after'] = position_after
                    action_result['movement_vector'] = movement_vector
                    action_result['dot_product'] = dot_product

                elif action_name in ['engage', 'suppress']:
                    # Engagement/suppression is successful if result contains hits or rounds expended
                    if isinstance(result, dict):
                        rounds_expended = result.get('ammo_expended', 0)
                        hits = result.get('total_hits', 0)
                        damage = result.get('total_damage', 0)
                    else:
                        rounds_expended = getattr(result, 'rounds_expended', 0)
                        hits = getattr(result, 'hits', 0)
                        damage = getattr(result, 'damage_dealt', 0)

                    success = rounds_expended > 0
                    action_result['success'] = success
                    action_result['rounds_expended'] = rounds_expended
                    action_result['hits'] = hits
                    action_result['damage'] = damage

                elif action_name == 'formation':
                    # Formation change is successful if new formation matches expected
                    formation_index = action['formation']
                    expected_formation = None
                    for name, idx in formation_mapping.items():
                        if idx == formation_index:
                            expected_formation = name
                            break

                    current_formation = env.get_unit_property(squad_id, 'formation')
                    success = current_formation == expected_formation
                    action_result['success'] = success
                    action_result['expected_formation'] = expected_formation
                    action_result['actual_formation'] = current_formation

                # Report success/failure
                if action_result['success']:
                    print(f"✓ {action_name.upper()} successful")
                else:
                    print(f"✗ {action_name.upper()} failed")
                    sequence_results['success'] = False

                # Capture frame for visualization
                if visualize:
                    current_pos = env.get_unit_position(squad_id)
                    action_frames.append({
                        'step': action_idx,
                        'action_name': action_name,
                        'position': current_pos,
                        'success': action_result['success']
                    })

            except Exception as e:
                print(f"Error executing {action_name}: {e}")
                import traceback
                traceback.print_exc()
                action_result['error'] = str(e)
                sequence_results['success'] = False

            # Add action result to sequence
            sequence_results['actions'].append(action_result)

        # Save final state for visualization
        if visualize:
            final_state = visualizer.plot_environment_state(
                title=f"Final State: {squad_string} after {sequence_name}",
                highlight_agent=squad_id
            )
            plt.close(final_state[0])

            # Create action sequence visualization
            if output_dir and action_frames:
                fig, ax = plt.subplots(figsize=(12, 10))

                # Plot unit's path through the sequence
                positions = [initial_position] + [frame['position'] for frame in action_frames]

                # Plot the path
                xs = [pos[0] for pos in positions]
                ys = [pos[1] for pos in positions]
                ax.plot(xs, ys, 'b-', alpha=0.7, label='Path')

                # Plot start and end points
                ax.plot(positions[0][0], positions[0][1], 'go', markersize=10, label='Start')
                ax.plot(positions[-1][0], positions[-1][1], 'ro', markersize=10, label='End')

                # Plot each intermediate point with action label
                for i, frame in enumerate(action_frames):
                    pos = frame['position']
                    color = 'g' if frame['success'] else 'r'
                    ax.plot(pos[0], pos[1], f'{color}o', markersize=8)
                    ax.annotate(f"{i + 1}: {frame['action_name'].upper()}",
                                (pos[0], pos[1]),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=9)

                # Plot enemies for reference
                for i, enemy_pos in enumerate(enemy_positions):
                    ax.plot(enemy_pos[0], enemy_pos[1], 'rx', markersize=8)
                    ax.annotate(f"Enemy {i}",
                                (enemy_pos[0], enemy_pos[1]),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=9)

                # Add info about the sequence
                info_text = f"Sequence {seq_idx + 1}:\n" + "\n".join(action_names)
                success_text = "✓ All actions successful" if sequence_results['success'] else "✗ Some actions failed"
                info_text += f"\n\n{success_text}"

                ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                        fontsize=10, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Set title and grid
                ax.set_title(f"Action Sequence: {squad_string} - {sequence_name}")
                ax.grid(True)

                # Create custom legend with unique entries
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                # Save figure
                plt.savefig(os.path.join(output_dir, f"sequence_squad_{squad_id}_seq_{seq_idx + 1}.png"))
                plt.close(fig)

        # Update results tracking
        results['total'] += 1
        if sequence_results['success']:
            results['success'] += 1

        # Add sequence results to details
        results['details'].append(sequence_results)

        # Reset squad to initial state for next sequence
        env.update_unit_position(squad_id, initial_position)
        if initial_formation:
            try:
                # Use MARL action to restore formation
                if initial_formation in formation_mapping:
                    restore_index = formation_mapping[initial_formation]
                else:
                    restore_index = 0

                restore_action = {
                    'action_type': 4,  # FORMATION
                    'formation': restore_index
                }

                env.agent_manager.execute_agent_action(squad_id, restore_action)
            except Exception as e:
                print(f"Error restoring original formation: {e}")

    # Print summary
    print("\nAction Sequence Test Summary:")
    print("-" * 50)
    success_rate = (results['success'] / results['total'] * 100) if results['total'] > 0 else 0
    print(f"Sequences: {results['success']}/{results['total']} successful ({success_rate:.1f}%)")

    return results


# Helper functions for unit selection
def get_unit_by_type(env, platoon_id, unit_type):
    """
    Get units of the specified type from a platoon.

    Args:
        env: MARL environment
        platoon_id: ID of platoon
        unit_type: Type of unit to find (infantry_team, weapons_team, squad)

    Returns:
        List of unit IDs
    """
    result = []

    if unit_type == 'squad':
        # Find squads in platoon
        for unit_id in env.get_unit_children(platoon_id):
            unit_type_val = env.get_unit_property(unit_id, 'type')
            string_id = env.get_unit_property(unit_id, 'string_id', '')
            if unit_type_val == UnitType.INFANTRY_SQUAD or ("SQD" in string_id):
                result.append(unit_id)

    elif unit_type == 'infantry_team':
        # Find infantry teams in platoon (via squads)
        for squad_id in get_unit_by_type(env, platoon_id, 'squad'):
            for unit_id in env.get_unit_children(squad_id):
                unit_type_val = env.get_unit_property(unit_id, 'type')
                string_id = env.get_unit_property(unit_id, 'string_id', '')
                if (unit_type_val == UnitType.INFANTRY_TEAM or
                        ("TM" in string_id and "GTM" not in string_id and "JTM" not in string_id)):
                    result.append(unit_id)

    elif unit_type == 'weapons_team':
        # Find weapons teams in platoon
        for unit_id in env.get_unit_children(platoon_id):
            unit_type_val = env.get_unit_property(unit_id, 'type')
            string_id = env.get_unit_property(unit_id, 'string_id', '')
            if unit_type_val == UnitType.WEAPONS_TEAM or 'GTM' in string_id or 'JTM' in string_id:
                result.append(unit_id)

    return result


def run_marl_action_tests(output_base_dir=None):
    """
    Run all MARL action space tests with a single squad agent.

    Args:
        output_base_dir: Base directory for output

    Returns:
        Dictionary with all test results
    """
    # Set output directory
    if output_base_dir is None:
        output_base_dir = CONFIG['output_dir']

    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Create environment
    env = setup_environment(
        width=CONFIG['width'],
        height=CONFIG['height'],
        debug_level=CONFIG['debug_level'],
        marl_wrapper=True,
        objective_pos=CONFIG['objective_pos']
    )

    # Create subdirectories for each test
    box_dir = os.path.join(output_base_dir, 'box_movement')
    engagement_dir = os.path.join(output_base_dir, 'engagement')
    formation_dir = os.path.join(output_base_dir, 'formation')
    combination_dir = os.path.join(output_base_dir, 'combination')

    for directory in [box_dir, engagement_dir, formation_dir, combination_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Run all tests
    results = {}

    # Test box movement pattern
    print("\nRunning box movement pattern test...")
    results['box'] = test_box_movement_pattern(env, output_dir=box_dir)

    # Create a new environment for each test to ensure clean state
    env = setup_environment(
        width=CONFIG['width'],
        height=CONFIG['height'],
        debug_level=CONFIG['debug_level'],
        marl_wrapper=True,
        objective_pos=CONFIG['objective_pos']
    )

    # Test engagement actions
    print("\nRunning engagement actions test...")
    results['engagement'] = test_engagement_actions(env, output_dir=engagement_dir)

    # Create a new environment for formation test
    env = setup_environment(
        width=CONFIG['width'],
        height=CONFIG['height'],
        debug_level=CONFIG['debug_level'],
        marl_wrapper=True,
        objective_pos=CONFIG['objective_pos']
    )

    # Test formation actions
    print("\nRunning formation actions test...")
    results['formation'] = test_formation_actions(env, output_dir=formation_dir)

    # Create a new environment for combination test
    env = setup_environment(
        width=CONFIG['width'],
        height=CONFIG['height'],
        debug_level=CONFIG['debug_level'],
        marl_wrapper=True,
        objective_pos=CONFIG['objective_pos']
    )

    # Test action combinations
    print("\nRunning action combination test...")
    results['combination'] = test_action_combination(env, output_dir=combination_dir)

    # Save overall results
    result_file = os.path.join(output_base_dir, 'marl_action_test_results.json')

    # Convert results to JSON-serializable format
    serializable_results = {}
    for test_name, test_results in results.items():
        if test_name == 'box':
            # Special handling for box results
            serializable_results[test_name] = {
                'success': test_results.get('success', False)
            }
        else:
            serializable_results[test_name] = {
                'success': test_results.get('success', 0),
                'total': test_results.get('total', 0)
            }

    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nAll tests completed. Results saved to {result_file}")

    return results


def main():
    """Main function to parse arguments and run tests."""
    # Create output directories
    base_dir = create_output_directories()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MARL action space verification tests')
    parser.add_argument('--test', choices=['box', 'engagement', 'formation', 'combination', 'all'],
                        default='all', help='Test type to run')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    # Set output directory
    output_dir = args.output if args.output else os.path.join(base_dir, 'action_space')

    try:
        # Create environment
        env = setup_environment(
            width=CONFIG['width'],
            height=CONFIG['height'],
            debug_level=CONFIG['debug_level'],
            marl_wrapper=True,
            objective_pos=CONFIG['objective_pos']
        )

        # Create subdirectories for test outputs
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        box_dir = os.path.join(output_dir, 'box_movement')
        engagement_dir = os.path.join(output_dir, 'engagement')
        formation_dir = os.path.join(output_dir, 'formation')
        combination_dir = os.path.join(output_dir, 'combination')

        for directory in [box_dir, engagement_dir, formation_dir, combination_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Run specified tests
        results = {}

        if args.test == 'all' or args.test == 'box':
            print("\nRunning box movement pattern test...")
            results['box'] = test_box_movement_pattern(
                env, output_dir=box_dir
            )

            # Create new environment for next test
            env = setup_environment(
                width=CONFIG['width'],
                height=CONFIG['height'],
                debug_level=CONFIG['debug_level'],
                marl_wrapper=True,
                objective_pos=CONFIG['objective_pos']
            )

        if args.test == 'all' or args.test == 'engagement':
            print("\nRunning engagement actions test...")
            results['engagement'] = test_engagement_actions(
                env, output_dir=engagement_dir, visualize=args.visualize
            )

            # Create new environment for next test
            env = setup_environment(
                width=CONFIG['width'],
                height=CONFIG['height'],
                debug_level=CONFIG['debug_level'],
                marl_wrapper=True,
                objective_pos=CONFIG['objective_pos']
            )

        if args.test == 'all' or args.test == 'formation':
            print("\nRunning formation actions test...")
            results['formation'] = test_formation_actions(
                env, output_dir=formation_dir, visualize=args.visualize
            )

            # Create new environment for next test
            env = setup_environment(
                width=CONFIG['width'],
                height=CONFIG['height'],
                debug_level=CONFIG['debug_level'],
                marl_wrapper=True,
                objective_pos=CONFIG['objective_pos']
            )

        if args.test == 'all' or args.test == 'combination':
            print("\nRunning action combination test...")
            results['combination'] = test_action_combination(
                env, output_dir=combination_dir, visualize=args.visualize
            )

        # Print summary
        print("\n" + "=" * 40)
        print("TEST EXECUTION SUMMARY")
        print("=" * 40)

        for test_name in ['box', 'engagement', 'formation', 'combination']:
            if test_name in results:
                print(f"\n{test_name.capitalize()} Test Results:")
                print("-" * 30)

                if test_name == 'box':
                    # Special handling for box movement results
                    box_results = results[test_name]
                    success_str = "✓ Success" if box_results.get("success", False) else "✗ Failed"
                    print(f"Box Movement: {success_str}")
                    print(f"Initial Position: {box_results.get('initial_position', 'Unknown')}")
                    print(
                        f"Final Position: {box_results.get('positions', ['Unknown'])[-1] if box_results.get('positions') else 'Unknown'}")
                else:
                    # Standard test results
                    test_results = results[test_name]
                    success_rate = (test_results['success'] / test_results['total'] * 100) if test_results[
                                                                                                  'total'] > 0 else 0
                    print(
                        f"{test_name.capitalize()}: {test_results['success']}/{test_results['total']} successful ({success_rate:.1f}%)")

        return 0

    except Exception as e:
        print(f"\nError during test execution: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
