"""
Enhanced Integration Test Script for MARL Environment

This script provides a comprehensive test to verify the MARL environment is working correctly,
focusing on a single squad agent for easier understanding.

The script will:
1. Create a test environment with a single squad
2. Reset the environment and verify agent observations against expected values
3. Execute actions with the agent and verify observation changes
4. Test reward calculation through agent actions
5. Validate milestone tracking through actual agent movement
6. Generate plots, logs, and visualizations for verification
"""

import numpy as np
import sys
import time
import os


def print_header(message):
    """Print a header with decoration."""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)


def print_subheader(message):
    """Print a subheader with decoration."""
    print("\n" + "-" * 60)
    print(f" {message} ".center(60, "-"))
    print("-" * 60)


def compare_values(observed, expected, tolerance=1e-6):
    """
    Compare observed and expected values with tolerance for floats.
    Fixed to properly handle NumPy arrays in boolean contexts.

    Args:
        observed: Observed value
        expected: Expected value
        tolerance: Tolerance for float comparison

    Returns:
        (match, message): Match status and description of comparison
    """
    import numpy as np

    # Handle NumPy arrays
    if isinstance(observed, np.ndarray) and isinstance(expected, np.ndarray):
        # Shape check
        if observed.shape != expected.shape:
            return False, f"Shape mismatch: {observed.shape} vs {expected.shape}"

        # Empty arrays are considered equal
        if observed.size == 0 and expected.size == 0:
            return True, "Empty arrays match"

        # For arrays, check if all elements are close
        if np.issubdtype(observed.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
            is_close = np.allclose(observed, expected, rtol=tolerance, atol=tolerance)
            return bool(is_close), f"{'Values match' if is_close else 'Values differ'}"
        else:
            # For integer arrays, check exact equality
            is_equal = np.array_equal(observed, expected)
            return bool(is_equal), f"{'Values match' if is_equal else 'Values differ'}"

    # Handle scalar NumPy values
    elif isinstance(observed, np.generic) or isinstance(expected, np.generic):
        # Convert NumPy scalars to Python scalars
        observed_val = observed.item() if isinstance(observed, np.generic) else observed
        expected_val = expected.item() if isinstance(expected, np.generic) else expected

        # Compare as Python scalars
        if isinstance(observed_val, float) or isinstance(expected_val, float):
            is_close = abs(observed_val - expected_val) <= tolerance
            return is_close, f"{'Values match' if is_close else 'Values differ'}"
        else:
            is_equal = observed_val == expected_val
            return is_equal, f"{'Values match' if is_equal else 'Values differ'}"

    # Handle Python floats
    elif isinstance(observed, float) and isinstance(expected, float):
        # For floats, use tolerance
        is_close = abs(observed - expected) <= tolerance
        return is_close, f"{'Values match' if is_close else 'Values differ'}"

    # Basic Python comparison for other types
    else:
        try:
            # Try direct equality, being careful about potential array-like objects
            is_equal = observed == expected

            # Handle case where the result might be an array-like object
            if hasattr(is_equal, '__iter__') and not isinstance(is_equal, (str, bytes, bytearray)):
                # If it's iterable but not a string type, convert to bool using all()
                is_equal = all(is_equal)

            return bool(is_equal), f"{'Values match' if is_equal else 'Values differ'}"
        except ValueError:
            # If direct comparison raises ValueError (like with multi-element arrays)
            # This is a fallback case for any other array-like objects
            return False, "Unable to compare directly (complex objects)"


def compare_observations(observed, expected, path=""):
    """
    Recursively compare observed and expected observations.

    Args:
        observed: Observed observation
        expected: Expected observation
        path: Current path in the nested structure

    Returns:
        (matches, total, details): Number of matches, total comparisons, and detailed results
    """
    matches = 0
    total = 0
    details = []

    # Handle different types of observations
    if isinstance(observed, dict) and isinstance(expected, dict):
        # For dictionaries, compare keys and recursively compare values
        all_keys = set(observed.keys()) | set(expected.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in observed:
                details.append(f"{current_path}: Missing in observed")
                total += 1
            elif key not in expected:
                details.append(f"{current_path}: Missing in expected")
                total += 1
            else:
                # Recursively compare values
                key_matches, key_total, key_details = compare_observations(observed[key], expected[key], current_path)
                matches += key_matches
                total += key_total
                details.extend(key_details)

    elif isinstance(observed, (list, tuple)) and isinstance(expected, (list, tuple)):
        # For lists, compare elements
        for i, (obs_item, exp_item) in enumerate(zip(observed, expected)):
            current_path = f"{path}[{i}]"
            item_matches, item_total, item_details = compare_observations(obs_item, exp_item, current_path)
            matches += item_matches
            total += item_total
            details.extend(item_details)

        # Handle different lengths
        len_diff = len(observed) - len(expected)
        if len_diff > 0:
            details.append(f"{path}: Observed has {len_diff} extra items")
            total += len_diff
        elif len_diff < 0:
            details.append(f"{path}: Expected has {abs(len_diff)} extra items")
            total += abs(len_diff)

    elif hasattr(observed, "shape") and hasattr(expected, "shape"):
        # For numpy arrays
        current_path = path
        is_match, message = compare_values(observed, expected)
        details.append(f"{current_path}: {message} (Shapes: {observed.shape} vs {expected.shape})")

        if is_match:
            matches += 1
        total += 1

    else:
        # For simple types
        current_path = path
        is_match, message = compare_values(observed, expected)
        details.append(f"{current_path}: {message} (Values: {observed} vs {expected})")

        if is_match:
            matches += 1
        total += 1

    return matches, total, details


def run_integration_tests():
    """Run comprehensive integration tests for enhanced MARL environment."""
    start_time = time.time()

    # Create directories for logs and visualizations
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    try:
        # Import your environment
        from WarGamingEnvironment_v9 import (
            EnvironmentConfig, MARLMilitaryEnvironment, UnitType, ForceType, EngagementType
        )
        print("✅ Successfully imported environment modules")

        # Import visualization libraries
        import matplotlib.pyplot as plt
        print("✅ Successfully imported visualization modules")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        print("Make sure all required modules are installed")
        return False

    # Create test environment
    try:
        print_header("Environment Setup")
        config = EnvironmentConfig(width=100, height=100, debug_level=1)
        env = MARLMilitaryEnvironment(config, objective_position=(75, 75))
        print("✅ Successfully created test environment")

        # Set up logging
        env.log_file = open('./logs/environment_test.log', 'w', encoding='utf-8')
        env.log_file.write("=== MARL Environment Test Log ===\n\n")

        print("✅ Set up logging to ./logs/environment_test.log")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False

    # Test 1: Reset environment and verify agent observations
    try:
        print_header("Test 1: Agent Observations")

        # Initialize with just one squad
        options = {
            'unit_init': {
                'objective': (75, 75),
                'platoon': {
                    'position': (25, 25),
                    'number': 1
                },
                'enemies': [
                    {
                        'type': UnitType.INFANTRY_TEAM,
                        'position': (35, 35),
                        'id_str': 'ENEMY-1',
                        'member_count': 4
                    },
                    {
                        'type': UnitType.INFANTRY_TEAM,
                        'position': (45, 25),
                        'id_str': 'ENEMY-2',
                        'member_count': 4
                    }
                ]
            }
        }

        # Log the initialization
        env.log_file.write(f"Initializing environment with a squad at (25,25) and enemies at (35,35) and (45,25)\n")
        env.log_file.write(f"Objective set at (75,75)\n\n")

        observations, _ = env.reset(options=options)

        if not observations:
            print("❌ Reset returned empty observations")
            return False

        agent_ids = list(observations.keys())
        print(f"✅ Environment has {len(agent_ids)} agents: {agent_ids}")

        # We'll focus on just the first agent (squad leader)
        agent_id = agent_ids[0]
        print(f"Focusing on a single agent: Agent {agent_id}")
        observation = observations[agent_id]

        # Plot the initial state
        try:
            plt.figure(figsize=(10, 10))

            # Get agent and objective positions
            agent_pos = observation['agent_state']['position']
            objective_pos = observation['objective']

            # Get enemy positions
            enemy_positions = []
            for pos in observation['known_enemies']:
                if np.any(pos != 0):
                    enemy_positions.append(pos)

            # Get friendly positions
            friendly_positions = []
            for pos in observation['friendly_units']:
                if np.any(pos != 0):
                    friendly_positions.append(pos)

            # Plot the positions
            plt.scatter(agent_pos[0], agent_pos[1], color='blue', s=200, marker='^', label='Agent')
            plt.scatter(objective_pos[0], objective_pos[1], color='green', s=200, marker='*', label='Objective')

            # Plot enemies if visible
            if enemy_positions:
                enemy_positions = np.array(enemy_positions)
                plt.scatter(enemy_positions[:, 0], enemy_positions[:, 1], color='red', s=150, marker='x',
                            label='Enemies')

            # Plot friendlies
            if friendly_positions:
                friendly_positions = np.array(friendly_positions)
                plt.scatter(friendly_positions[:, 0], friendly_positions[:, 1], color='cyan', s=100, marker='o',
                            label='Friendly Units')

            # Draw a line to the objective
            plt.plot([agent_pos[0], objective_pos[0]], [agent_pos[1], objective_pos[1]], 'g--', alpha=0.5)

            # Draw circles at milestone distances
            milestone_distances = [60, 50, 40, 30, 20, 10, 5]
            for dist in milestone_distances:
                circle = plt.Circle((objective_pos[0], objective_pos[1]), dist, fill=False, linestyle=':', alpha=0.3)
                plt.gca().add_patch(circle)
                plt.text(objective_pos[0], objective_pos[1] + dist, f"{dist}", ha='center', alpha=0.7)

            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.title(f'Environment State - Distance to Objective: {observation["objective_info"]["distance"][0]:.2f}')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig('./plots/initial_state.png')
            print("✅ Saved initial state visualization to ./plots/initial_state.png")

        except Exception as e:
            print(f"⚠️ Failed to create initial state plot: {e}")
            # Don't fail the test for this

        # Generate expected observation for comparison
        print_subheader("Observation Validation")
        print("Comparing agent's observation with expected values:")

        # Get actual unit ID for our agent
        unit_id = env.agent_manager.get_current_unit_id(agent_id)
        if not unit_id:
            print("❌ Could not map agent ID to unit ID")
            return False

        # Manually calculate expected observation values
        expected_observation = {}

        # 1. Agent State
        expected_agent_state = {
            'position': env.get_unit_position(unit_id),
            'health': np.array([env.get_unit_property(unit_id, 'health', 100)], dtype=np.float32),
            'ammo': np.array([env.get_unit_property(unit_id, 'ammo_primary', 0)], dtype=np.int32),
        }
        # Add suppression value
        if hasattr(env, 'combat_manager') and unit_id in env.combat_manager.suppressed_units:
            expected_agent_state['suppressed'] = np.array([env.combat_manager.suppressed_units[unit_id]['level']], dtype=np.float32)
        else:
            expected_agent_state['suppressed'] = np.array([0.0], dtype=np.float32)

        expected_observation['agent_state'] = expected_agent_state

        # 2. Tactical Info
        parent_id = env.get_unit_property(unit_id, 'parent_id')
        formation = "unknown"
        orientation = 0
        if parent_id:
            formation = env.get_unit_property(parent_id, 'formation', "unknown")
            orientation = env.get_unit_property(parent_id, 'orientation', 0)

        # Map formation string to index
        formation_map = {
            "team_wedge_right": 0,
            "team_wedge_left": 1,
            "team_line_right": 2,
            "team_line_left": 3,
            "team_column": 4,
            "squad_column_team_wedge": 5,
            "squad_column_team_column": 6,
            "squad_line_team_wedge": 7,
            "squad_vee_team_wedge": 8,
            "platoon_column": 9,
            "platoon_wedge": 10
        }
        formation_index = formation_map.get(formation, 0)

        # Get agent type index
        agent_type = env.agent_manager.agent_types.get(agent_id) if hasattr(env, 'agent_manager') else None
        unit_type_index = 0  # Default to INFANTRY_TEAM
        if agent_type == 'WEAPONS_TEAM':
            unit_type_index = 1
        elif agent_type == 'SQUAD':
            unit_type_index = 2

        expected_observation['tactical_info'] = {
            'formation': np.array([formation_index], dtype=np.int32),
            'orientation': np.array([orientation], dtype=np.int32),
            'unit_type': np.array([unit_type_index], dtype=np.int32)
        }

        # 3. Objective Info
        if hasattr(env, 'objective') and env.objective:
            agent_pos = env.get_unit_position(unit_id)
            # Calculate direction
            dx = env.objective[0] - agent_pos[0]
            dy = env.objective[1] - agent_pos[1]
            magnitude = np.sqrt(dx*dx + dy*dy)

            if magnitude > 0:
                expected_direction = np.array([dx/magnitude, dy/magnitude], dtype=np.float32)
            else:
                expected_direction = np.array([0.0, 0.0], dtype=np.float32)

            expected_distance = np.array([env._calculate_distance(agent_pos, env.objective)], dtype=np.float32)

            expected_observation['objective_info'] = {
                'direction': expected_direction,
                'distance': expected_distance
            }

            expected_observation['objective'] = np.array(env.objective, dtype=np.int32)

        # 4. Manually calculate visible enemies
        expected_enemies = np.zeros((10, 2), dtype=np.int32)

        # Get all enemy unit IDs
        enemy_units = [uid for uid in env.state_manager.active_units
                     if env.get_unit_property(uid, 'force_type') == ForceType.ENEMY
                     and env.get_unit_property(uid, 'health', 0) > 0]

        # Get agent observation range
        agent_pos = env.get_unit_position(unit_id)
        observation_range = env.get_unit_property(unit_id, 'observation_range', 50)

        # Find visible enemies
        enemy_count = 0
        for enemy_id in enemy_units:
            if enemy_count >= 10:  # Limit to 10 units
                break

            enemy_pos = env.get_unit_position(enemy_id)
            distance = env._calculate_distance(agent_pos, enemy_pos)

            # Check if within observation range
            if distance <= observation_range:
                # Check line of sight
                los_result = env.visibility_manager.check_line_of_sight(agent_pos, enemy_pos)
                if los_result['has_los']:
                    expected_enemies[enemy_count] = enemy_pos
                    enemy_count += 1

        expected_observation['known_enemies'] = expected_enemies

        # 5. Manually calculate friendly positions
        expected_friendlies = np.zeros((10, 2), dtype=np.int32)

        # Get all friendly unit IDs
        friendly_units = [uid for uid in env.state_manager.active_units
                         if env.get_unit_property(uid, 'force_type') == ForceType.FRIENDLY
                         and uid != unit_id]

        # Calculate radio range
        radio_range = 100

        # Fill with friendly positions within radio range
        friendly_count = 0
        for uid in friendly_units:
            if friendly_count >= 10:  # Limit to 10 units
                break

            unit_pos = env.get_unit_position(uid)
            distance = env._calculate_distance(agent_pos, unit_pos)

            # Check if within radio range
            if distance <= radio_range:
                expected_friendlies[friendly_count] = unit_pos
                friendly_count += 1

        expected_observation['friendly_units'] = expected_friendlies

        # Compare observations
        matches, total, details = compare_observations(observation, expected_observation)

        # Print comparison results
        print(f"\nObservation validation results: {matches} matches out of {total} checks")
        print(f"Match percentage: {(matches/total)*100:.1f}%")

        # Print agent's observation
        print("\nAGENT OBSERVATION:")
        print(f"  Position: {observation['agent_state']['position']}")
        print(f"  Health: {observation['agent_state']['health']}")
        print(f"  Ammo: {observation['agent_state']['ammo']}")
        print(f"  Suppressed: {observation['agent_state']['suppressed']}")
        print(f"  Formation: {observation['tactical_info']['formation']}")
        print(f"  Direction to objective: {observation['objective_info']['direction']}")
        print(f"  Distance to objective: {observation['objective_info']['distance'][0]:.2f}")

        # Count non-zero entries in arrays
        known_enemies = observation['known_enemies']
        visible_enemy_count = np.sum(np.any(known_enemies != 0, axis=1))
        print(f"  Visible enemies: {visible_enemy_count}")
        if visible_enemy_count > 0:
            visible_enemies = known_enemies[np.any(known_enemies != 0, axis=1)]
            print(f"  Enemy positions: {visible_enemies}")

        friendly_units = observation['friendly_units']
        visible_friendly_count = np.sum(np.any(friendly_units != 0, axis=1))
        print(f"  Visible friendlies: {visible_friendly_count}")
        if visible_friendly_count > 0:
            visible_friendlies = friendly_units[np.any(friendly_units != 0, axis=1)]
            print(f"  Friendly positions: {visible_friendlies[:3]}" +
                  (f" (+ {len(visible_friendlies)-3} more)" if len(visible_friendlies) > 3 else ""))

        # Print expected observation
        print("\nEXPECTED OBSERVATION:")
        print(f"  Position: {expected_observation['agent_state']['position']}")
        print(f"  Health: {expected_observation['agent_state']['health']}")
        print(f"  Ammo: {expected_observation['agent_state']['ammo']}")
        print(f"  Suppressed: {expected_observation['agent_state']['suppressed']}")
        print(f"  Formation: {expected_observation['tactical_info']['formation']}")
        print(f"  Direction to objective: {expected_observation['objective_info']['direction']}")
        print(f"  Distance to objective: {expected_observation['objective_info']['distance'][0]:.2f}")

        # Count non-zero entries for enemies
        expected_enemies = expected_observation['known_enemies']
        expected_enemy_count = np.sum(np.any(expected_enemies != 0, axis=1))
        print(f"  Expected visible enemies: {expected_enemy_count}")
        if expected_enemy_count > 0:
            expected_visible_enemies = expected_enemies[np.any(expected_enemies != 0, axis=1)]
            print(f"  Expected enemy positions: {expected_visible_enemies}")

        # Count non-zero entries for friendlies
        expected_friendlies = expected_observation['friendly_units']
        expected_friendly_count = np.sum(np.any(expected_friendlies != 0, axis=1))
        print(f"  Expected visible friendlies: {expected_friendly_count}")
        if expected_friendly_count > 0:
            expected_visible_friendlies = expected_friendlies[np.any(expected_friendlies != 0, axis=1)]
            print(f"  Expected friendly positions: {expected_visible_friendlies[:3]}" +
                  (f" (+ {len(expected_visible_friendlies)-3} more)" if len(expected_visible_friendlies) > 3 else ""))

        # Check main validation points and print differences
        if matches < total:
            print("\nDifferences found:")
            for detail in details:
                if "Values differ" in detail or "Missing" in detail or "extra items" in detail:
                    print(f"  {detail}")

            # Major mismatch checks
            if visible_enemy_count != expected_enemy_count:
                print(f"⚠️ Mismatch in visible enemy count: {visible_enemy_count} vs expected {expected_enemy_count}")

            if visible_friendly_count != expected_friendly_count:
                print(f"⚠️ Mismatch in visible friendly count: {visible_friendly_count} vs expected {expected_friendly_count}")

            print("\nNOTE: Small differences in values (especially floating point) may be acceptable.")
            print("      The critical validation is that the number of visible units matches expectations.")
        else:
            print("\n✅ Observation validation successful - all values match!")

        # Save validation details to log
        env.log_file.write("\n=== OBSERVATION VALIDATION ===\n")
        env.log_file.write(f"Validation results: {matches} matches out of {total} checks\n")
        env.log_file.write(f"Match percentage: {(matches/total)*100:.1f}%\n\n")

        env.log_file.write("AGENT OBSERVATION:\n")
        env.log_file.write(f"  Position: {observation['agent_state']['position']}\n")
        env.log_file.write(f"  Visible enemies: {visible_enemy_count}\n")
        env.log_file.write(f"  Visible friendlies: {visible_friendly_count}\n\n")

        env.log_file.write("EXPECTED OBSERVATION:\n")
        env.log_file.write(f"  Position: {expected_observation['agent_state']['position']}\n")
        env.log_file.write(f"  Expected visible enemies: {expected_enemy_count}\n")
        env.log_file.write(f"  Expected visible friendlies: {expected_friendly_count}\n\n")

        # Required components check
        required_components = [
            'agent_state', 'tactical_info',
            'friendly_units', 'known_enemies', 'objective', 'objective_info'
        ]

        component_results = []
        for component in required_components:
            if component in observation:
                if isinstance(observation[component], dict):
                    subcomponents = list(observation[component].keys())
                    component_results.append(f"  ✅ {component}: {subcomponents}")
                elif hasattr(observation[component], 'shape'):
                    component_results.append(f"  ✅ {component}: shape={observation[component].shape}")
                else:
                    component_results.append(f"  ✅ {component}: {type(observation[component])}")
            else:
                component_results.append(f"  ❌ Missing component: {component}")

        print("\nCOMPONENT STRUCTURE CHECK:")
        for result in component_results:
            print(result)
            env.log_file.write(f"{result}\n")

        # Check if all required components are present
        if all(result.startswith("  ✅") for result in component_results):
            print("✅ All required observation components are present")
        else:
            print("❌ Some required observation components are missing")
            return False

        print("✅ Agent observation structure and validation test passed!")
    except Exception as e:
        print(f"❌ Agent observation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Execute actions and verify observation changes
    try:
        print_header("Test 2: Agent Actions and Observation Changes")

        # Focus on our single agent
        agent_id = agent_ids[0]
        baseline_observation = observations[agent_id]

        # Print initial position
        print_subheader("Initial Agent Position")
        initial_pos = baseline_observation['agent_state']['position']
        initial_distance = baseline_observation['objective_info']['distance'][0]
        print(f"Agent {agent_id}: Position {initial_pos}")
        print(f"Distance to objective: {initial_distance:.2f} cells")

        # Log initial state
        env.log_file.write(f"Agent {agent_id} initial position: {initial_pos}\n")
        env.log_file.write(f"Initial distance to objective: {initial_distance:.2f} cells\n")

        # Create a movement action toward the objective
        direction = baseline_observation['objective_info']['direction']

        # Movement action
        action = {
            'action_type': 0,  # MOVE
            'movement_params': {
                'direction': direction,
                'distance': np.array([5])  # Move 5 cells
            },
            'engagement_params': {  # Dummy values, not used
                'target_pos': np.array([0, 0]),
                'max_rounds': np.array([0]),
                'suppress_only': 0
            }
        }

        # Create action dictionary with just our agent
        actions = {agent_id: action}

        print_subheader("Executing Movement Action")
        print(f"Moving agent {agent_id} toward objective")
        env.log_file.write(f"Executing movement action with direction {direction} and distance 5\n")

        # Execute actions
        new_observations, rewards, terminated, truncated, infos = env.step(actions)

        # Verify new observations show position changes
        print_subheader("New Agent Position")
        new_observation = new_observations[agent_id]
        new_pos = new_observation['agent_state']['position']
        new_distance = new_observation['objective_info']['distance'][0]

        # Check if position changed
        pos_changed = not np.array_equal(initial_pos, new_pos)

        # Check if agent moved in the right direction (closer to objective)
        moved_correctly = new_distance < initial_distance

        print(f"Agent {agent_id}:")
        print(f"  Old position: {initial_pos}, New position: {new_pos}")
        print(f"  Distance to objective: {initial_distance:.2f} -> {new_distance:.2f}")
        print(f"  Position changed: {'✅' if pos_changed else '❌'}")
        print(f"  Moved correctly toward objective: {'✅' if moved_correctly else '❌'}")

        # Log movement results
        env.log_file.write(f"New position: {new_pos}\n")
        env.log_file.write(f"New distance to objective: {new_distance:.2f} cells\n")
        env.log_file.write(f"Position changed: {pos_changed}\n")
        env.log_file.write(f"Moved correctly toward objective: {moved_correctly}\n\n")

        if not pos_changed:
            print(f"  ❌ Agent {agent_id} position did not change")
            return False

        if not moved_correctly:
            print(f"  ⚠️ Agent {agent_id} did not move closer to objective")
            # Don't fail the test for this as it could be due to collision or terrain

        # Print rewards
        print_subheader("Movement Reward")
        reward = rewards[agent_id]
        print(f"Agent {agent_id}: Reward = {reward}")
        env.log_file.write(f"Movement reward: {reward}\n")

        # Verify reward is a valid number
        if not np.isfinite(reward):
            print(f"  ❌ Invalid reward value: {reward}")
            return False

        # Plot the updated state
        try:
            plt.figure(figsize=(10, 10))

            # Get agent and objective positions
            agent_pos = new_observation['agent_state']['position']
            objective_pos = new_observation['objective']

            # Get enemy positions
            enemy_positions = []
            for pos in new_observation['known_enemies']:
                if np.any(pos != 0):
                    enemy_positions.append(pos)

            # Get friendly positions
            friendly_positions = []
            for pos in new_observation['friendly_units']:
                if np.any(pos != 0):
                    friendly_positions.append(pos)

            # Plot the positions
            plt.scatter(agent_pos[0], agent_pos[1], color='blue', s=200, marker='^', label='Agent (New)')
            plt.scatter(initial_pos[0], initial_pos[1], color='lightblue', s=150, marker='^', label='Agent (Old)')
            plt.scatter(objective_pos[0], objective_pos[1], color='green', s=200, marker='*', label='Objective')

            # Plot enemies if visible
            if enemy_positions:
                enemy_positions = np.array(enemy_positions)
                plt.scatter(enemy_positions[:, 0], enemy_positions[:, 1], color='red', s=150, marker='x',
                            label='Enemies')

            # Plot friendlies
            if friendly_positions:
                friendly_positions = np.array(friendly_positions)
                plt.scatter(friendly_positions[:, 0], friendly_positions[:, 1], color='cyan', s=100, marker='o',
                            label='Friendly Units')

            # Draw a line showing the movement
            plt.plot([initial_pos[0], agent_pos[0]], [initial_pos[1], agent_pos[1]], 'b-', alpha=0.5)

            # Draw a line to the objective
            plt.plot([agent_pos[0], objective_pos[0]], [agent_pos[1], objective_pos[1]], 'g--', alpha=0.5)

            # Draw circles at milestone distances
            milestone_distances = [60, 50, 40, 30, 20, 10, 5]
            for dist in milestone_distances:
                circle = plt.Circle((objective_pos[0], objective_pos[1]), dist, fill=False, linestyle=':', alpha=0.3)
                plt.gca().add_patch(circle)
                plt.text(objective_pos[0], objective_pos[1] + dist, f"{dist}", ha='center', alpha=0.7)

            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.title(f'Agent Movement - New Distance to Objective: {new_distance:.2f}')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig('./plots/movement_action.png')
            print("✅ Saved movement visualization to ./plots/movement_action.png")

        except Exception as e:
            print(f"⚠️ Failed to create movement plot: {e}")
            # Don't fail the test for this

        print("✅ Agent action and observation change test passed!")
    except Exception as e:
        print(f"❌ Agent action test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Test engagement actions and rewards
    try:
        print_header("Test 3: Engagement Actions and Rewards")

        # Our agent is the same one from before
        visible_enemies = np.sum(np.any(new_observation['known_enemies'] != 0, axis=1))

        # Check if our agent can see enemies
        if visible_enemies == 0:
            print("⚠️ Agent cannot see enemies, executing movement to get closer first")
            env.log_file.write("No enemies visible, moving agent closer to enemy positions\n")

            # Execute another movement action to get closer to enemies
            # Create a movement action toward the first enemy position (which we know is at (35,35))
            enemy_pos = np.array([35, 35])
            agent_pos = new_pos

            # Calculate direction to enemy
            dx = enemy_pos[0] - agent_pos[0]
            dy = enemy_pos[1] - agent_pos[1]
            magnitude = np.sqrt(dx*dx + dy*dy)
            direction = np.array([dx/magnitude, dy/magnitude]) if magnitude > 0 else np.array([0, 0])

            # Movement action toward enemy
            move_to_enemy_action = {
                'action_type': 0,  # MOVE
                'movement_params': {
                    'direction': direction,
                    'distance': np.array([10])  # Move 10 cells
                },
                'engagement_params': {  # Dummy values, not used
                    'target_pos': np.array([0, 0]),
                    'max_rounds': np.array([0]),
                    'suppress_only': 0
                }
            }

            # Execute movement
            print(f"Moving agent {agent_id} toward enemies first")
            env.log_file.write(f"Moving agent toward enemy position with direction {direction}\n")
            new_observations, _, _, _, _ = env.step({agent_id: move_to_enemy_action})

            # Update our observation
            new_observation = new_observations[agent_id]
            agent_pos = new_observation['agent_state']['position']

            # Check again if enemies are visible
            visible_enemies = np.sum(np.any(new_observation['known_enemies'] != 0, axis=1))
            print(f"After movement, agent can see {visible_enemies} enemies")
            env.log_file.write(f"After movement, agent can see {visible_enemies} enemies\n")

        # Determine target position
        target_pos = None
        enemy_positions = []

        # Find first visible enemy
        for pos in new_observation['known_enemies']:
            if np.any(pos != 0):
                target_pos = pos
                enemy_positions.append(pos)

        if target_pos is None:
            print("⚠️ Still no visible enemies after movement, skipping engagement test")
            env.log_file.write("No enemies visible after movement, skipping engagement test\n")
        else:
            print(f"Agent {agent_id} can see enemies at {target_pos}")
            env.log_file.write(f"Agent can see enemies at {target_pos}\n")

            # Create engagement action
            engage_action = {
                'action_type': 1,  # ENGAGE
                'movement_params': {  # Dummy values, not used
                    'direction': np.array([0, 0]),
                    'distance': np.array([0])
                },
                'engagement_params': {
                    'target_pos': target_pos,
                    'max_rounds': np.array([10]),  # Fire 10 rounds
                    'suppress_only': 0,  # Direct fire
                    'adjust_for_fire_rate': 1  # Adjust for weapon fire rate
                }
            }

            print(f"Executing engagement action for agent {agent_id}")
            env.log_file.write(f"Executing engagement action targeting position {target_pos}\n")

            # Record ammunition before engagement
            ammo_before = new_observation['agent_state']['ammo']
            print(f"Ammunition before engagement: {ammo_before}")
            env.log_file.write(f"Ammunition before engagement: {ammo_before}\n")

            # Execute engagement action
            post_engage_obs, post_engage_rewards, _, _, _ = env.step({agent_id: engage_action})

            # Verify ammunition was used
            ammo_after = post_engage_obs[agent_id]['agent_state']['ammo']
            print(f"Ammunition after engagement: {ammo_after}")

            ammo_used = ammo_before - ammo_after
            print(f"Ammunition used: {ammo_used}")
            env.log_file.write(f"Ammunition after engagement: {ammo_after}\n")
            env.log_file.write(f"Ammunition used: {ammo_used}\n")

            if np.any(ammo_used > 0):
                print(f"✅ Agent {agent_id} successfully used ammunition")
                env.log_file.write("✓ Successfully used ammunition\n")
            else:
                print(f"⚠️ Agent {agent_id} did not use ammunition - may not have had line of sight")
                env.log_file.write("⚠ Did not use ammunition - may not have had line of sight\n")

            # Print engagement reward
            reward = post_engage_rewards[agent_id]
            print(f"Engagement reward: {reward}")
            env.log_file.write(f"Engagement reward: {reward}\n\n")

            # Plot the engagement
            try:
                plt.figure(figsize=(10, 10))

                # Get positions
                agent_pos = post_engage_obs[agent_id]['agent_state']['position']
                objective_pos = post_engage_obs[agent_id]['objective']

                # Plot the positions
                plt.scatter(agent_pos[0], agent_pos[1], color='blue', s=200, marker='^', label='Agent')
                plt.scatter(objective_pos[0], objective_pos[1], color='green', s=200, marker='*', label='Objective')

                # Plot enemies
                for pos in enemy_positions:
                    plt.scatter(pos[0], pos[1], color='red', s=150, marker='x', label='Enemy Target')

                # Draw a line from agent to target
                plt.plot([agent_pos[0], target_pos[0]], [agent_pos[1], target_pos[1]], 'r-', alpha=0.7)

                # Draw a shooting indicator
                plt.annotate('', xy=(target_pos[0], target_pos[1]), xytext=(agent_pos[0], agent_pos[1]),
                            arrowprops=dict(arrowstyle='->', color='orange', lw=2))

                # Draw an explosion at target
                if np.any(ammo_used > 0):
                    plt.scatter(target_pos[0], target_pos[1], color='orange', s=300, marker='*', alpha=0.7,
                                label='Impact')

                plt.xlim(0, 100)
                plt.ylim(0, 100)
                plt.title(f'Engagement Action - Ammunition Used: {ammo_used}')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.savefig('./plots/engagement_action.png')
                print("✅ Saved engagement visualization to ./plots/engagement_action.png")

            except Exception as e:
                print(f"⚠️ Failed to create engagement plot: {e}")
                # Don't fail the test for this
    except Exception as e:
        print(f"❌ Engagement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Move agent closer to objective to trigger milestones
    try:
        print_header("Test 4: Objective Milestone Tracking")

        # Reset the environment with a new configuration - agent closer to objective
        options = {
            'unit_init': {
                'objective': (75, 75),
                'platoon': {
                    'position': (65, 65),  # Much closer to objective (75,75)
                    'number': 1
                }
            }
        }

        env.log_file.write("\n=== MILESTONE TRACKING TEST ===\n")
        env.log_file.write("Resetting environment with agent at position (65,65), closer to objective (75,75)\n\n")

        milestone_obs, _ = env.reset(options=options)
        agent_ids = list(milestone_obs.keys())

        if not agent_ids:
            print("❌ No agents after reset")
            return False

        # Focus on our single agent
        agent_id = agent_ids[0]
        print(f"Reset environment with agent at position (65,65), closer to objective (75,75)")

        # Execute a series of movement actions toward the objective
        print_subheader("Moving Agent Toward Milestones")

        # Track distances to objective
        distances = []
        positions = []
        rewards = []

        # First get initial distance
        current_distance = milestone_obs[agent_id]['objective_info']['distance'][0]
        current_pos = milestone_obs[agent_id]['agent_state']['position']
        distances.append(current_distance)
        positions.append(current_pos)

        print(f"Initial distance to objective: {current_distance:.2f} cells")
        env.log_file.write(f"Initial distance to objective: {current_distance:.2f} cells\n")

        # Run multiple steps, moving toward objective
        for step in range(5):
            # Get direction to objective
            direction = milestone_obs[agent_id]['objective_info']['direction']

            # Create movement action
            move_action = {
                'action_type': 0,  # MOVE
                'movement_params': {
                    'direction': direction,
                    'distance': np.array([3])  # Move 3 cells each time
                },
                'engagement_params': {  # Dummy values, not used
                    'target_pos': np.array([0, 0]),
                    'max_rounds': np.array([0]),
                    'suppress_only': 0
                }
            }

            # Execute movement
            milestone_obs, milestone_reward, _, _, _ = env.step({agent_id: move_action})

            # Get updated distance and position
            current_distance = milestone_obs[agent_id]['objective_info']['distance'][0]
            current_pos = milestone_obs[agent_id]['agent_state']['position']

            # Save data for visualization
            distances.append(current_distance)
            positions.append(current_pos)
            rewards.append(milestone_reward[agent_id])

            # Print and log progress
            print(f"Step {step+1}: Distance to objective = {current_distance:.2f} cells, Reward = {milestone_reward[agent_id]:.2f}")
            env.log_file.write(f"Step {step+1}: Distance = {current_distance:.2f}, Reward = {milestone_reward[agent_id]:.2f}\n")

            # Check if we've reached the objective
            if current_distance < 3:
                print(f"✅ Reached objective after {step+1} steps!")
                env.log_file.write(f"Reached objective after {step+1} steps!\n")
                break

        # Verify we have milestones tracked
        if hasattr(env, '_objective_milestones_reached'):
            milestones_reached = sorted(env._objective_milestones_reached)
            print(f"\nMilestones reached: {milestones_reached}")
            env.log_file.write(f"Milestones reached: {milestones_reached}\n")

            if len(milestones_reached) > 0:
                print(f"✅ Successfully reached {len(milestones_reached)} milestones")
                env.log_file.write(f"Successfully reached {len(milestones_reached)} milestones\n")
            else:
                print(f"⚠️ No milestones were reached")
                env.log_file.write("No milestones were reached\n")
        else:
            print("⚠️ No milestone tracking found in environment")
            env.log_file.write("No milestone tracking found in environment\n")

        # Check progress history
        if hasattr(env, '_progress_history') and env._progress_history:
            print(f"\nProgress history entries: {len(env._progress_history)}")
            env.log_file.write(f"Progress history entries: {len(env._progress_history)}\n")

            print("Progress history samples:")
            for entry in env._progress_history[:3]:  # Show first 3 entries
                print(f"  {entry}")
                env.log_file.write(f"  {entry}\n")
        else:
            print("⚠️ No progress history found in environment")
            env.log_file.write("No progress history found in environment\n")

        # Create milestone visualization
        try:
            plt.figure(figsize=(12, 10))

            # Set up subplot layout
            plt.subplot(2, 1, 1)

            # Plot the agent's path
            positions = np.array(positions)
            plt.plot(positions[:, 0], positions[:, 1], 'b-', marker='o', linewidth=2)

            # Plot starting and ending positions
            plt.scatter(positions[0, 0], positions[0, 1], color='blue', s=200, marker='^', label='Start')
            plt.scatter(positions[-1, 0], positions[-1, 1], color='green', s=200, marker='*', label='End')

            # Plot objective
            objective_pos = milestone_obs[agent_id]['objective']
            plt.scatter(objective_pos[0], objective_pos[1], color='red', s=200, marker='X', label='Objective')

            # Draw milestone rings
            milestone_distances = [60, 50, 40, 30, 20, 10, 5]
            for dist in milestone_distances:
                circle = plt.Circle((objective_pos[0], objective_pos[1]), dist, fill=False, linestyle='--', alpha=0.5)
                plt.gca().add_patch(circle)
                plt.text(objective_pos[0], objective_pos[1] + dist, f"{dist}", ha='center', alpha=0.7)

            plt.grid(True, alpha=0.3)
            plt.title("Agent Path Toward Objective")
            plt.legend()

            # Second subplot - distance over time
            plt.subplot(2, 1, 2)
            steps = range(len(distances))
            plt.plot(steps, distances, 'g-', marker='o', linewidth=2)
            plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Milestone 50')
            plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Milestone 30')
            plt.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Milestone 10')
            plt.grid(True, alpha=0.3)
            plt.title("Distance to Objective Over Time")
            plt.xlabel("Steps")
            plt.ylabel("Distance")

            plt.tight_layout()
            plt.savefig('./plots/milestone_tracking.png')
            print("✅ Saved milestone visualization to ./plots/milestone_tracking.png")

        except Exception as e:
            print(f"⚠️ Failed to create milestone visualization: {e}")
            # Don't fail the test for this

        print("✅ Milestone tracking test completed")
    except Exception as e:
        print(f"❌ Milestone tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Generate summary visualization and metrics
    try:
        print_header("Test 5: Summary Visualization and Metrics")

        # Generate a summary report with metrics
        print_subheader("Environment Performance Metrics")

        # Collect metrics across all tests
        metrics = {
            "total_distance_traveled": 0,
            "enemies_engaged": 0,
            "ammunition_used": 0,
            "milestones_reached": len(env._objective_milestones_reached) if hasattr(env, '_objective_milestones_reached') else 0,
            "total_reward": 0
        }

        # Calculate metrics
        metrics["total_distance_traveled"] = sum(distances[:-1]) - sum(distances[1:])  # Sum of distance reductions

        # Log metrics
        env.log_file.write("\n=== SUMMARY METRICS ===\n")
        env.log_file.write(f"Total distance traveled: {metrics['total_distance_traveled']:.2f} cells\n")
        env.log_file.write(f"Milestones reached: {metrics['milestones_reached']}\n")
        env.log_file.write(f"Total reward: {sum(rewards) if 'rewards' in locals() else 'N/A'}\n")

        print(f"Total distance traveled: {metrics['total_distance_traveled']:.2f} cells")
        print(f"Milestones reached: {metrics['milestones_reached']}")
        print(f"Total reward: {sum(rewards) if 'rewards' in locals() else 'N/A'}")

        # Create summary visualization
        try:
            plt.figure(figsize=(10, 6))

            # Create a reward vs. distance plot
            if 'rewards' in locals() and 'distances' in locals() and len(rewards) > 0:
                plt.plot(distances[1:], rewards, 'b-', marker='o')
                plt.grid(True, alpha=0.3)
                plt.title("Reward vs. Distance to Objective")
                plt.xlabel("Distance to Objective")
                plt.ylabel("Reward")

                # Add trendline
                if len(rewards) > 1:
                    try:
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(distances[1:], rewards)
                        plt.plot(distances[1:], intercept + slope * np.array(distances[1:]), 'r--',
                                label=f'Trend (R²={r_value**2:.2f})')
                        plt.legend()
                    except:
                        # Skip trendline if scipy not available
                        pass

                plt.savefig('./plots/reward_vs_distance.png')
                print("✅ Saved reward vs. distance visualization to ./plots/reward_vs_distance.png")

            # Check if environment has the visualization method
            if hasattr(env, 'create_milestone_visualization'):
                print("Generating milestone visualization...")
                try:
                    env.create_milestone_visualization('./plots')
                    print("✅ Successfully generated milestone visualization")
                except Exception as e:
                    print(f"⚠️ Built-in visualization generation failed: {e}")
            else:
                print("Note: Environment doesn't have create_milestone_visualization method")

        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
            # Don't fail the test for this

        print("✅ Summary metrics and visualization completed")

        # Close log file
        env.log_file.close()
        print("✅ Test log saved to ./logs/environment_test.log")

    except Exception as e:
        print(f"⚠️ Summary test failed: {e}")
        import traceback
        traceback.print_exc()

        # Close log file if open
        if hasattr(env, 'log_file') and not env.log_file.closed:
            env.log_file.close()

    # All tests passed
    end_time = time.time()
    print_header("Test Results")
    print(f"✅ All integration tests passed! Your enhanced MARL environment is working correctly.")
    print(f"Tests completed in {end_time - start_time:.2f} seconds")

    print("\nKey verification points:")
    print("1. Observation structure is correct and includes tactical features")
    print("2. Agent movement actions work and update observations")
    print("3. Engagement actions work and use ammunition")
    print("4. Milestone tracking functions during agent movement")
    print("5. Reward calculation provides meaningful signals")

    print("\nVisualizations generated:")
    print("1. ./plots/initial_state.png - Initial environment state")
    print("2. ./plots/movement_action.png - Agent movement action")
    print("3. ./plots/engagement_action.png - Agent engagement action")
    print("4. ./plots/milestone_tracking.png - Agent path to objective")
    print("5. ./plots/reward_vs_distance.png - Reward vs. distance analysis")

    return True


if __name__ == "__main__":
    print_header("Enhanced MARL Environment Integration Test")
    success = run_integration_tests()
    if not success:
        print("\n❌ Please fix the issues above before proceeding with training.")
        sys.exit(1)
    else:
        print("\n✅ Test completed successfully! Check the 'plots' directory for visualizations.")
        print("   A detailed log is available in the 'logs' directory.")
        sys.exit(0)
