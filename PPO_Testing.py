import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def evaluate_trained_agents(
        model_dir,
        num_episodes=100,
        max_steps_per_episode=200,
        map_file="training_map_lvl_1.csv",
        objective_location=(73, 75),
        enemy_positions=None,
        unit_start_positions=None,
        output_dir=None,
        use_tqdm=True,
        seed=None,
        verbose=True
):
    """
    Evaluate pre-trained agents and collect comprehensive statistical data.

    Args:
        model_dir: Directory containing saved agent models to load
        num_episodes: Number of test episodes to run
        max_steps_per_episode: Maximum steps per episode
        map_file: CSV file with terrain information
        objective_location: (x, y) coordinates of the objective
        enemy_positions: List of (x, y) coordinates for enemy placements
        unit_start_positions: Dict mapping unit names to starting positions
        output_dir: Directory to save test results and logs
        use_tqdm: Whether to use tqdm progress bars
        seed: Random seed for reproducibility
        verbose: Whether to print detailed progress information

    Returns:
        Dictionary with aggregated test results and statistics
    """
    import os
    import time
    import numpy as np
    import json
    from datetime import datetime
    import matplotlib.pyplot as plt
    import pandas as pd

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        import random
        random.seed(seed)

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

    # Create output directory with timestamp if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./evaluation_results_{timestamp}"

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Setup tqdm if requested
    tqdm = lambda x, **kwargs: x  # Default no-op tqdm
    trange = range  # Default range function

    if use_tqdm:
        try:
            from tqdm import tqdm, trange
            if verbose:
                print("Using tqdm for progress tracking")
        except ImportError:
            if verbose:
                print("tqdm not installed. Run 'pip install tqdm' to use progress bars.")
            use_tqdm = False

    # Initialize environment
    from WarGamingEnvironment_v13 import EnvironmentConfig, MARLMilitaryEnvironment, ForceType, UnitType

    # Estimate map size based on objective and unit positions
    all_positions = [objective_location] + list(unit_start_positions.values()) + enemy_positions
    max_x = max(pos[0] for pos in all_positions) + 20  # Add margin
    max_y = max(pos[1] for pos in all_positions) + 20  # Add margin

    # Save test configuration
    config = {
        "model_dir": model_dir,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "map_file": map_file,
        "objective_location": objective_location,
        "enemy_positions": enemy_positions,
        "unit_start_positions": unit_start_positions,
        "seed": seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(output_dir, "test_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Create environment config
    env_config = EnvironmentConfig(width=max_x, height=max_y, debug_level=0)
    env = MARLMilitaryEnvironment(env_config, objective_position=objective_location)

    # Initialize PPO with test mode settings (no learning)
    from PPO_Training_v4 import WarGameMARLPPO
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=0.0)  # Zero learning rate for testing

    # Connect environment and PPO for observation handling
    setattr(env, 'marl_algorithm', marl_ppo)

    # Initialize agent role mapping to ensure consistent IDs
    env.agent_manager.initialize_agent_role_mapping()

    # Load trained agents
    if verbose:
        print(f"Loading trained agents from {model_dir}...")

    marl_ppo.load_agents(model_dir)

    # Count how many agents were loaded
    agent_count = len(marl_ppo.agent_policies)
    if verbose:
        print(f"Loaded {agent_count} trained agents")

        # Print info about loaded agents
        for agent_id, policy in marl_ppo.agent_policies.items():
            print(f"Agent {agent_id}: Update count = {policy.update_count}")

    # Prepare data storage for statistics
    episode_stats = []
    unit_stats = {}  # Dictionary to track per-unit statistics
    team_stats = []  # Track overall team performance
    episode_durations = []
    win_rate = 0

    # Define tracking data structures for detailed statistics
    engagement_log = []
    movement_log = []
    casualty_log = []
    objective_progress_log = []

    # Define criteria for mission success/failure
    def evaluate_mission_outcome(env, stats):
        """Determine if mission was successful based on criteria"""
        # Default is failure
        outcome = {
            "success": False,
            "reason": "Mission incomplete",
            "objective_reached": False,
            "team_survival": 0.0
        }

        # Check if any friendly unit reached objective
        objective_reached = False
        objective_pos = objective_location
        objective_radius = 5  # Units within this distance of objective are considered to have reached it

        friendly_units = []
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    friendly_units.append(unit_id)
                    unit_pos = env.get_unit_position(unit_id)
                    distance_to_objective = ((unit_pos[0] - objective_pos[0]) ** 2 +
                                             (unit_pos[1] - objective_pos[1]) ** 2) ** 0.5
                    if distance_to_objective <= objective_radius:
                        objective_reached = True

        # Calculate team survival rate
        team_survival = len(friendly_units) / stats["initial_friendly_count"] if stats[
                                                                                     "initial_friendly_count"] > 0 else 0

        # Check if all enemies eliminated
        enemies_eliminated = stats["enemy_casualties"] == stats["initial_enemy_count"]

        # Define success criteria
        if objective_reached and team_survival >= 0.5:
            outcome["success"] = True
            outcome["reason"] = "Objective reached with sufficient team survival"
        elif enemies_eliminated and team_survival >= 0.4:
            outcome["success"] = True
            outcome["reason"] = "All enemies eliminated with sufficient team survival"

        outcome["objective_reached"] = objective_reached
        outcome["team_survival"] = team_survival
        outcome["enemies_eliminated"] = enemies_eliminated

        return outcome

    # Test loop
    episode_range = trange(num_episodes) if use_tqdm else range(num_episodes)
    if verbose:
        print(f"Starting evaluation with {num_episodes} episodes...")

    for episode in episode_range:
        episode_start_time = time.time()

        # Update tqdm description if available
        if use_tqdm:
            episode_range.set_description(f"Episode {episode + 1}/{num_episodes}")
        elif verbose and episode % 10 == 0:
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
            except Exception as e:
                if verbose:
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

                # Initialize unit stats tracking for this episode
                for child_id in plt_children:
                    unit_string_id = env.get_unit_property(child_id, 'string_id', '')
                    # Initialize or reset unit stats for this episode
                    if unit_string_id not in unit_stats:
                        unit_stats[unit_string_id] = {
                            "total_rounds_fired": 0,
                            "hits": 0,
                            "missions_participated": 0,
                            "enemy_casualties_caused": 0,
                            "distance_moved": 0,
                            "survival_rate": 0,
                            "successful_missions": 0
                        }

                    # Increment missions participated
                    unit_stats[unit_string_id]["missions_participated"] += 1

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

        except Exception as e:
            if verbose:
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

        # Track initial units for casualty tracking
        initial_enemy_count = 0
        initial_enemy_units = {}
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    initial_enemy_count += 1
                    unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
                    if unit_string_id:
                        initial_enemy_units[unit_string_id] = {
                            "id": unit_id,
                            "health": env.get_unit_property(unit_id, 'health', 0),
                            "position": env.get_unit_position(unit_id)
                        }

        initial_friendly_count = 0
        initial_friendly_units = {}
        for unit_id in env.state_manager.active_units:
            if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                if env.get_unit_property(unit_id, 'health', 0) > 0:
                    initial_friendly_count += 1
                    unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
                    if unit_string_id:
                        initial_friendly_units[unit_string_id] = {
                            "id": unit_id,
                            "health": env.get_unit_property(unit_id, 'health', 0),
                            "position": env.get_unit_position(unit_id),
                            "ammo": env.get_unit_property(unit_id, 'ammo', 0)
                        }

        # Episode variables
        episode_rewards = {agent_id: 0 for agent_id in env.agent_ids}
        episode_length = 0
        done = False
        rounds_fired = 0
        enemy_casualties = 0
        friendly_casualties = 0

        # Track unit positions for distance calculation
        unit_positions = {}
        for unit_id in env.state_manager.active_units:
            unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
            if unit_string_id:
                unit_positions[unit_string_id] = env.get_unit_position(unit_id)

        # Track distance to objective for progress tracking
        objective_distances = {}
        for unit_id, data in initial_friendly_units.items():
            pos = data["position"]
            distance_to_objective = ((pos[0] - objective_location[0]) ** 2 +
                                     (pos[1] - objective_location[1]) ** 2) ** 0.5
            objective_distances[unit_id] = distance_to_objective

        # Create step iterator
        step_range = trange(max_steps_per_episode) if use_tqdm else range(max_steps_per_episode)

        # Episode loop
        for step in step_range:
            if done:
                break

            # Update step progress bar if available
            if use_tqdm:
                step_range.set_description(f"Step {step + 1}/{max_steps_per_episode}")

            # Select actions with trained policies (no exploration)
            try:
                # Use eval mode for deterministic actions
                for agent_id, policy in marl_ppo.agent_policies.items():
                    policy.actor.eval()
                    policy.critic.eval()

                actions = {}
                for agent_id in env.agent_ids:
                    if agent_id in observations:
                        # Get the policy for this agent
                        if agent_id in marl_ppo.agent_policies:
                            # Get deterministic action
                            action, _, _ = marl_ppo.agent_policies[agent_id].select_action_with_logprob(
                                observations[agent_id])
                            actions[agent_id] = action

            except Exception as e:
                if verbose:
                    print(f"Error selecting actions: {e}")
                continue

            # Execute actions
            try:
                next_observations, rewards, dones, truncs, infos = env.step(actions)

                # Update observations and accumulate rewards
                observations = next_observations
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward

                # Track action outcomes
                for agent_id, action in actions.items():
                    unit_id = env.agent_manager.get_current_unit_id(agent_id)
                    if unit_id:
                        unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
                        action_type = action.get('action_type', -1)

                        # Track movement
                        if action_type == 0:  # MOVE
                            old_pos = unit_positions.get(unit_string_id, (0, 0))
                            new_pos = env.get_unit_position(unit_id)

                            # Calculate distance moved
                            distance = ((new_pos[0] - old_pos[0]) ** 2 + (new_pos[1] - old_pos[1]) ** 2) ** 0.5

                            # Update position tracking
                            unit_positions[unit_string_id] = new_pos

                            # Update unit stats
                            if unit_string_id in unit_stats:
                                unit_stats[unit_string_id]["distance_moved"] += distance

                            # Record movement event
                            movement_log.append({
                                "episode": episode,
                                "step": step,
                                "unit": unit_string_id,
                                "from": old_pos,
                                "to": new_pos,
                                "distance": distance
                            })

                            # Update distance to objective
                            distance_to_objective = ((new_pos[0] - objective_location[0]) ** 2 +
                                                     (new_pos[1] - objective_location[1]) ** 2) ** 0.5
                            objective_distances[unit_string_id] = distance_to_objective

                            # Log objective progress
                            objective_progress_log.append({
                                "episode": episode,
                                "step": step,
                                "unit": unit_string_id,
                                "distance_to_objective": distance_to_objective
                            })

                        # Track engagement
                        elif action_type == 1:  # ENGAGE
                            rounds = action.get('engagement_params', {}).get('max_rounds', [1])[0]
                            rounds_fired += rounds

                            # Update unit stats
                            if unit_string_id in unit_stats:
                                unit_stats[unit_string_id]["total_rounds_fired"] += rounds

                            # Track engagement event
                            engagement_log.append({
                                "episode": episode,
                                "step": step,
                                "unit": unit_string_id,
                                "rounds": rounds,
                                "target_pos": action.get('engagement_params', {}).get('target_pos', [0, 0]),
                                "suppress_only": action.get('engagement_params', {}).get('suppress_only', 0)
                            })

                # Check for casualties
                current_enemy_count = 0
                for unit_id in env.state_manager.active_units:
                    if env.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY:
                        unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
                        health = env.get_unit_property(unit_id, 'health', 0)

                        if health > 0:
                            current_enemy_count += 1
                        elif unit_string_id in initial_enemy_units and initial_enemy_units[unit_string_id][
                            "health"] > 0:
                            # This unit was just killed
                            casualty_log.append({
                                "episode": episode,
                                "step": step,
                                "unit": unit_string_id,
                                "type": "enemy",
                                "position": env.get_unit_position(unit_id)
                            })
                            initial_enemy_units[unit_string_id]["health"] = 0

                # Update enemy casualties
                new_enemy_casualties = initial_enemy_count - current_enemy_count
                enemy_casualties = new_enemy_casualties

                # Check friendly casualties
                current_friendly_count = 0
                friendly_unit_status = {}
                for unit_id in env.state_manager.active_units:
                    if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                        unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
                        health = env.get_unit_property(unit_id, 'health', 0)

                        if health > 0:
                            current_friendly_count += 1
                        elif unit_string_id in initial_friendly_units and initial_friendly_units[unit_string_id][
                            "health"] > 0:
                            # This unit was just killed
                            casualty_log.append({
                                "episode": episode,
                                "step": step,
                                "unit": unit_string_id,
                                "type": "friendly",
                                "position": env.get_unit_position(unit_id)
                            })
                            initial_friendly_units[unit_string_id]["health"] = 0

                        if unit_string_id:
                            friendly_unit_status[unit_string_id] = {
                                "id": unit_id,
                                "health": health,
                                "position": env.get_unit_position(unit_id),
                                "ammo": env.get_unit_property(unit_id, 'ammo', 0)
                            }

                # Update friendly casualties
                friendly_casualties = initial_friendly_count - current_friendly_count

                # Check termination
                done = all(dones.values()) or all(truncs.values())
                episode_length += 1

                # Update step progress description if using tqdm
                if use_tqdm:
                    avg_step_reward = sum(rewards.values()) / len(rewards) if rewards else 0
                    friendly_ratio = current_friendly_count / initial_friendly_count if initial_friendly_count > 0 else 0
                    enemy_ratio = current_enemy_count / initial_enemy_count if initial_enemy_count > 0 else 0
                    step_range.set_postfix(
                        reward=f"{avg_step_reward:.2f}",
                        friendly=f"{friendly_ratio:.2f}",
                        enemy=f"{enemy_ratio:.2f}"
                    )

            except Exception as e:
                if verbose:
                    print(f"Error during step execution: {e}")
                continue

        # Episode complete - calculate statistics
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_durations.append(episode_duration)

        # Calculate average reward
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards) if episode_rewards else 0

        # Calculate remaining ammo and total used
        ammo_remaining = 0
        ammo_used = 0
        unit_survival = {}

        for unit_string_id, initial_data in initial_friendly_units.items():
            initial_ammo = initial_data.get("ammo", 0)

            # Initialize friendly_unit_status dictionary
            friendly_unit_status = {}
            for unit_id in env.state_manager.active_units:
                if env.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY:
                    health = env.get_unit_property(unit_id, 'health', 0)
                    unit_string_id = env.get_unit_property(unit_id, 'string_id', '')

                    if unit_string_id:
                        friendly_unit_status[unit_string_id] = {
                            "id": unit_id,
                            "health": health,
                            "position": env.get_unit_position(unit_id),
                            "ammo": env.get_unit_property(unit_id, 'ammo', 0)
                        }

            # Check if unit is still alive
            if unit_string_id in friendly_unit_status:
                current_ammo = friendly_unit_status[unit_string_id].get("ammo", 0)
                current_health = friendly_unit_status[unit_string_id].get("health", 0)

                ammo_remaining += current_ammo
                ammo_used += max(0, initial_ammo - current_ammo)

                # Track unit survival
                survived = current_health > 0
                unit_survival[unit_string_id] = survived

                # Update unit statistics
                if unit_string_id in unit_stats:
                    if survived:
                        unit_stats[unit_string_id]["survival_rate"] += 1
            else:
                # Unit not found in current status, assume casualty
                ammo_used += initial_ammo
                unit_survival[unit_string_id] = False

        # Evaluate mission outcome
        mission_stats = {
            "episode": episode,
            "length": episode_length,
            "reward": avg_reward,
            "friendly_casualties": friendly_casualties,
            "initial_friendly_count": initial_friendly_count,
            "enemy_casualties": enemy_casualties,
            "initial_enemy_count": initial_enemy_count,
            "rounds_fired": rounds_fired,
            "ammo_used": ammo_used,
            "ammo_remaining": ammo_remaining,
            "duration": episode_duration
        }

        mission_outcome = evaluate_mission_outcome(env, mission_stats)
        mission_stats.update(mission_outcome)

        # Record mission success in unit stats
        if mission_outcome["success"]:
            win_rate += 1
            for unit_id, survived in unit_survival.items():
                if survived and unit_id in unit_stats:
                    unit_stats[unit_id]["successful_missions"] += 1

        # Add to episode stats
        episode_stats.append(mission_stats)

        # Add to team stats
        team_stats.append({
            "episode": episode,
            "mission_success": mission_outcome["success"],
            "objective_reached": mission_outcome["objective_reached"],
            "team_survival_rate": mission_outcome["team_survival"],
            "enemy_elimination_rate": enemy_casualties / initial_enemy_count if initial_enemy_count > 0 else 0,
            "avg_reward": avg_reward
        })

        # Log episode results
        if verbose and (not use_tqdm or episode % 10 == 0):
            print(f"Episode {episode}: {'SUCCESS' if mission_outcome['success'] else 'FAILURE'}")
            print(f"  Reason: {mission_outcome['reason']}")
            print(f"  Friendly Survival: {mission_outcome['team_survival']:.2f}")
            print(f"  Enemy Casualties: {enemy_casualties}/{initial_enemy_count}")
            print(f"  Steps: {episode_length}, Reward: {avg_reward:.2f}")

    # Calculate aggregate statistics
    win_rate = win_rate / num_episodes
    avg_episode_length = sum(stat["length"] for stat in episode_stats) / len(episode_stats)
    avg_enemy_casualties = sum(stat["enemy_casualties"] for stat in episode_stats) / len(episode_stats)
    avg_friendly_casualties = sum(stat["friendly_casualties"] for stat in episode_stats) / len(episode_stats)
    avg_rounds_fired = sum(stat["rounds_fired"] for stat in episode_stats) / len(episode_stats)

    # Calculate objective distance metrics
    closest_approach = float('inf')
    for progress in objective_progress_log:
        closest_approach = min(closest_approach, progress["distance_to_objective"])

    # Finalize unit stats
    for unit_id, stats in unit_stats.items():
        if stats["missions_participated"] > 0:
            stats["survival_rate"] = stats["survival_rate"] / stats["missions_participated"]
            stats["success_rate"] = stats["successful_missions"] / stats["missions_participated"]

    # Create summary report
    summary = {
        "test_config": config,
        "aggregate_stats": {
            "win_rate": win_rate,
            "avg_episode_length": avg_episode_length,
            "avg_enemy_casualties": avg_enemy_casualties,
            "avg_friendly_casualties": avg_friendly_casualties,
            "avg_rounds_fired": avg_rounds_fired,
            "avg_episode_duration": sum(episode_durations) / len(episode_durations),
            "closest_approach_to_objective": closest_approach
        },
        "unit_stats": unit_stats
    }

    # Save summary report
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Create CSV logs for easier analysis
    episode_df = pd.DataFrame(episode_stats)
    episode_df.to_csv(os.path.join(output_dir, "logs", "episode_stats.csv"), index=False)

    team_df = pd.DataFrame(team_stats)
    team_df.to_csv(os.path.join(output_dir, "logs", "team_stats.csv"), index=False)

    unit_df = pd.DataFrame.from_dict(unit_stats, orient='index').reset_index().rename(columns={'index': 'unit_id'})
    unit_df.to_csv(os.path.join(output_dir, "logs", "unit_stats.csv"), index=False)

    # Save detailed event logs
    engagement_df = pd.DataFrame(engagement_log)
    movement_df = pd.DataFrame(movement_log)
    casualty_df = pd.DataFrame(casualty_log)
    objective_df = pd.DataFrame(objective_progress_log)

    if not engagement_df.empty:
        engagement_df.to_csv(os.path.join(output_dir, "logs", "engagement_log.csv"), index=False)
    if not movement_df.empty:
        movement_df.to_csv(os.path.join(output_dir, "logs", "movement_log.csv"), index=False)
    if not casualty_df.empty:
        casualty_df.to_csv(os.path.join(output_dir, "logs", "casualty_log.csv"), index=False)
    if not objective_df.empty:
        objective_df.to_csv(os.path.join(output_dir, "logs", "objective_progress_log.csv"), index=False)

    # Generate visualization plots
    try:
        # Win rate over episodes (rolling average)
        plt.figure(figsize=(10, 6))
        team_df['mission_success_rolling'] = team_df['mission_success'].rolling(window=min(10, len(team_df)),
                                                                                min_periods=1).mean()
        plt.plot(team_df['episode'], team_df['mission_success_rolling'], 'b-')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (rolling avg)')
        plt.title('Mission Success Rate')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "plots", "success_rate.png"))
        plt.close()

        # Casualties comparison
        plt.figure(figsize=(10, 6))
        episode_df['friendly_casualty_rate'] = episode_df['friendly_casualties'] / episode_df['initial_friendly_count']
        episode_df['enemy_casualty_rate'] = episode_df['enemy_casualties'] / episode_df['initial_enemy_count']

        plt.plot(episode_df['episode'], episode_df['friendly_casualty_rate'], 'r-', label='Friendly Casualties')
        plt.plot(episode_df['episode'], episode_df['enemy_casualty_rate'], 'g-', label='Enemy Casualties')
        plt.xlabel('Episode')
        plt.ylabel('Casualty Rate')
        plt.title('Casualty Rates by Episode')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "plots", "casualty_rates.png"))
        plt.close()

        # Unit performance comparison
        if len(unit_df) > 0:
            plt.figure(figsize=(12, 8))
            unit_df = unit_df.sort_values('survival_rate', ascending=False)
            plt.bar(unit_df['unit_id'], unit_df['survival_rate'], color='blue', alpha=0.6, label='Survival Rate')
            plt.bar(unit_df['unit_id'], unit_df['success_rate'], color='green', alpha=0.6, label='Mission Success Rate')
            plt.xlabel('Unit')
            plt.ylabel('Rate')
            plt.title('Unit Performance')
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "unit_performance.png"))
            plt.close()

            # Unit movement distance
            plt.figure(figsize=(12, 6))
            plt.bar(unit_df['unit_id'], unit_df['distance_moved'], color='orange')
            plt.xlabel('Unit')
            plt.ylabel('Total Distance Moved')
            plt.title('Movement by Unit')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "unit_movement.png"))
            plt.close()

        # Create heatmap of casualties if data available
        if not casualty_df.empty:
            try:
                # Calculate map boundaries
                max_x = max([pos[0] for pos in enemy_positions] + [pos[0] for pos in unit_start_positions.values()])
                max_y = max([pos[1] for pos in enemy_positions] + [pos[1] for pos in unit_start_positions.values()])

                # Create a heatmap for casualty locations
                plt.figure(figsize=(10, 8))
                friendly_casualties = casualty_df[casualty_df['type'] == 'friendly']
                enemy_casualties = casualty_df[casualty_df['type'] == 'enemy']

                if not friendly_casualties.empty:
                    friendly_x = [pos[0] for pos in friendly_casualties['position']]
                    friendly_y = [pos[1] for pos in friendly_casualties['position']]
                    plt.scatter(friendly_x, friendly_y, color='blue', alpha=0.6, label='Friendly Casualties')

                if not enemy_casualties.empty:
                    enemy_x = [pos[0] for pos in enemy_casualties['position']]
                    enemy_y = [pos[1] for pos in enemy_casualties['position']]
                    plt.scatter(enemy_x, enemy_y, color='red', alpha=0.6, label='Enemy Casualties')

                # Plot objective and starting positions
                plt.scatter([objective_location[0]], [objective_location[1]], color='green', marker='*', s=200,
                            label='Objective')

                # Plot initial unit positions
                for unit, pos in unit_start_positions.items():
                    plt.scatter([pos[0]], [pos[1]], color='blue', marker='o', s=50)

                # Plot enemy positions
                for pos in enemy_positions:
                    plt.scatter([pos[0]], [pos[1]], color='red', marker='o', s=50)

                plt.xlim(0, max_x + 10)
                plt.ylim(0, max_y + 10)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Casualty Locations')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "plots", "casualty_map.png"))
                plt.close()
            except Exception as e:
                if verbose:
                    print(f"Error creating casualty map: {e}")

    except Exception as e:
        if verbose:
            print(f"Error generating plots: {e}")

    # Print summary results
    if verbose:
        print("\n===== EVALUATION SUMMARY =====")
        print(f"Model tested: {model_dir}")
        print(f"Episodes: {num_episodes}")
        print(f"Win rate: {win_rate:.2f}")
        print(f"Average episode length: {avg_episode_length:.1f} steps")
        print(f"Average episode duration: {summary['aggregate_stats']['avg_episode_duration']:.1f} seconds")
        print(f"Average friendly casualties: {avg_friendly_casualties:.2f}")
        print(f"Average enemy casualties: {avg_enemy_casualties:.2f}")
        print(f"Summary saved to: {summary_file}")

    return summary


def visualize_trained_agents(
        model_dir,
        map_file="training_map_lvl_1.csv",
        objective_location=(73, 75),
        enemy_positions=None,
        unit_start_positions=None,
        max_steps=200,
        output_dir=None,
        save_video=True,
        video_fps=5,
        render_scale=10,
        show_realtime=False,
        step_delay=0.2
):
    """
    Run a visual simulation of trained agents on a specific scenario and create a video recording.

    Args:
        model_dir: Directory containing saved agent models to load
        map_file: CSV file with terrain information
        objective_location: (x, y) coordinates of the objective
        enemy_positions: List of (x, y) coordinates for enemy placements
        unit_start_positions: Dict mapping unit names to starting positions
        max_steps: Maximum steps to simulate
        output_dir: Directory to save visualization outputs
        save_video: Whether to save a video of the simulation
        video_fps: Frames per second for saved video
        render_scale: Scale factor for rendering (larger = bigger visualization)
        show_realtime: Whether to display the visualization in real-time
        step_delay: Delay between steps when showing real-time (seconds)

    Returns:
        Path to saved video file and episode data
    """
    import os
    import numpy as np
    import json
    from datetime import datetime
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

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

    # Create output directory with timestamp if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./visualization_{timestamp}"

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    # Initialize environment with rendering support
    from WarGamingEnvironment_v13 import EnvironmentConfig, MARLMilitaryEnvironment, ForceType, UnitType

    # Estimate map size based on objective and unit positions
    all_positions = [objective_location] + list(unit_start_positions.values()) + enemy_positions
    max_x = max(pos[0] for pos in all_positions) + 20  # Add margin
    max_y = max(pos[1] for pos in all_positions) + 20  # Add margin

    # Create environment config - note render_mode is set to None because we'll handle rendering ourselves
    env_config = EnvironmentConfig(width=max_x, height=max_y, debug_level=0)
    env = MARLMilitaryEnvironment(env_config, objective_position=objective_location)

    # Initialize PPO with test mode settings (no learning)
    from PPO_Training_v4 import WarGameMARLPPO
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=0.0)  # Zero learning rate for visualization

    # Connect environment and PPO for observation handling
    setattr(env, 'marl_algorithm', marl_ppo)

    # Initialize agent role mapping to ensure consistent IDs
    env.agent_manager.initialize_agent_role_mapping()

    # Load trained agents
    print(f"Loading trained agents from {model_dir}...")
    marl_ppo.load_agents(model_dir)

    # Count how many agents were loaded
    agent_count = len(marl_ppo.agent_policies)
    print(f"Loaded {agent_count} trained agents")

    # Print info about loaded agents
    for agent_id, policy in marl_ppo.agent_policies.items():
        print(f"Agent {agent_id}: Update count = {policy.update_count}")

    # Helper function to convert numpy arrays to serializable objects
    def serialize_numpy(obj):
        """Convert numpy arrays and other non-serializable objects to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [serialize_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: serialize_numpy(value) for key, value in obj.items()}
        else:
            return obj

    # Data structure for recording episode
    episode_data = {
        "metadata": {
            "model_dir": model_dir,
            "objective": objective_location,
            "enemy_positions": enemy_positions,
            "unit_start_positions": unit_start_positions,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "frames": []
    }

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

    # Set up visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # Function to capture state for each frame
    def capture_frame(step, env, actions=None):
        """Capture the current state of the environment for visualization"""
        frame_data = {
            "step": step,
            "friendly_units": {},
            "enemy_units": {},
            "actions": serialize_numpy(actions.copy() if actions else {}),
            "engagement_lines": []
        }

        # Capture unit states
        for unit_id in env.state_manager.active_units:
            force_type = env.get_unit_property(unit_id, 'force_type')
            unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
            unit_type = env.get_unit_property(unit_id, 'unit_type', None)
            health = env.get_unit_property(unit_id, 'health', 0)
            position = env.get_unit_position(unit_id)
            ammo = env.get_unit_property(unit_id, 'ammo', 0)
            suppressed = env.get_unit_property(unit_id, 'suppressed', 0)

            unit_data = {
                "id": unit_id,
                "string_id": unit_string_id,
                "type": str(unit_type) if unit_type else "UNKNOWN",
                "health": health,
                "position": serialize_numpy(position),
                "ammo": ammo,
                "suppressed": suppressed
            }

            if force_type == ForceType.FRIENDLY:
                frame_data["friendly_units"][unit_string_id] = unit_data
            elif force_type == ForceType.ENEMY:
                frame_data["enemy_units"][unit_string_id] = unit_data

        # Track engagement lines
        if actions:
            for agent_id, action in actions.items():
                unit_id = env.agent_manager.get_current_unit_id(agent_id)
                if unit_id and action.get('action_type', -1) == 1:  # ENGAGE
                    unit_pos = env.get_unit_position(unit_id)
                    target_pos = action.get('engagement_params', {}).get('target_pos', [0, 0])
                    suppress_only = action.get('engagement_params', {}).get('suppress_only', 0) == 1

                    frame_data["engagement_lines"].append({
                        "from": serialize_numpy(unit_pos),
                        "to": serialize_numpy(target_pos),
                        "suppress_only": suppress_only
                    })

        return frame_data

    # Capture initial state
    initial_frame = capture_frame(0, env)
    episode_data["frames"].append(initial_frame)

    # Function to render a single frame
    def render_frame(frame_data, ax, render_scale=10):
        """Render a frame of the simulation"""
        ax.clear()

        # Set plot limits
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)

        # Plot terrain if available - FIXED: Use state_tensor instead of terrain_data
        try:
            # Access terrain from the state tensor instead
            if hasattr(env, 'state_manager') and hasattr(env.state_manager, 'state_tensor'):
                # Create a terrain heatmap from state tensor
                # The first channel (index 0) of the state tensor contains terrain type
                terrain_img = np.zeros((max_y, max_x))
                state_tensor = env.state_manager.state_tensor

                # Only copy terrain data within bounds
                for y in range(min(max_y, state_tensor.shape[0])):
                    for x in range(min(max_x, state_tensor.shape[1])):
                        # Get terrain value from state tensor
                        terrain_img[y, x] = state_tensor[y, x, 0]  # First channel is terrain

                # Plot terrain heatmap with a terrain-appropriate colormap
                ax.imshow(terrain_img, cmap='terrain', alpha=0.5, origin='lower', extent=(0, max_x, 0, max_y))
        except Exception as e:
            print(f"Error rendering terrain: {e}")

        # Draw objective
        objective_circle = Circle(objective_location, 2, color='green', alpha=0.7)
        ax.add_patch(objective_circle)
        ax.text(objective_location[0], objective_location[1], "OBJECTIVE",
                horizontalalignment='center', verticalalignment='center',
                color='black', fontweight='bold')

        # Draw friendly units
        for unit_id, unit_data in frame_data["friendly_units"].items():
            pos = unit_data["position"]
            health = unit_data["health"]

            # Set color based on health
            if health <= 0:
                color = 'black'  # Dead
                alpha = 0.5
            else:
                health_ratio = health / 100.0
                import matplotlib as mpl
                cmap = mpl.colormaps['RdYlGn']
                color = cmap(health_ratio)  # Red to Yellow to Green based on health
                alpha = 0.8

            # Draw unit with size based on unit type
            if "SQD" in unit_id:
                size = 3
            else:
                size = 2

            # Add suppression indicator
            if unit_data["suppressed"] > 0:
                # Draw a yellow ring around suppressed units
                suppression_ring = Circle(pos, size + 1, color='yellow', alpha=0.4)
                ax.add_patch(suppression_ring)

            unit_circle = Circle(pos, size, color=color, alpha=alpha)
            ax.add_patch(unit_circle)

            # Label the unit
            shortened_id = unit_id.split('-')[-1] if '-' in unit_id else unit_id
            ax.text(pos[0], pos[1], shortened_id,
                    horizontalalignment='center', verticalalignment='center',
                    color='white', fontweight='bold', fontsize=8)

        # Draw enemy units
        for unit_id, unit_data in frame_data["enemy_units"].items():
            pos = unit_data["position"]
            health = unit_data["health"]

            # Set color based on health
            if health <= 0:
                color = 'black'  # Dead
                alpha = 0.5
            else:
                color = 'red'  # Enemy
                alpha = 0.8

            enemy_circle = Circle(pos, 2, color=color, alpha=alpha)
            ax.add_patch(enemy_circle)

            # Label the unit
            shortened_id = unit_id.split('-')[-1] if '-' in unit_id else unit_id
            ax.text(pos[0], pos[1], shortened_id,
                    horizontalalignment='center', verticalalignment='center',
                    color='white', fontsize=8)

        # Draw engagement lines
        for engagement in frame_data.get("engagement_lines", []):
            from_pos = engagement["from"]
            to_pos = engagement["to"]
            suppress_only = engagement.get("suppress_only", False)

            # Draw the line with different style based on engagement type
            if suppress_only:
                # Suppression fire is dotted yellow
                ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                        color='yellow', linestyle=':', linewidth=1, alpha=0.7)
            else:
                # Direct fire is solid red
                ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                        color='red', linestyle='-', linewidth=1, alpha=0.7)

        # Add step counter and other information
        step = frame_data["step"]
        friendly_count = sum(1 for unit in frame_data["friendly_units"].values() if unit["health"] > 0)
        enemy_count = sum(1 for unit in frame_data["enemy_units"].values() if unit["health"] > 0)

        ax.set_title(f"Step: {step} | Friendly: {friendly_count} | Enemy: {enemy_count}")
        ax.grid(True, linestyle='--', alpha=0.7)

        return ax

    # Initialize termination flags
    dones = {}
    truncs = {}
    for agent_id in env.agent_ids:
        dones[agent_id] = False
        truncs[agent_id] = False

    # Main simulation loop
    done = False
    frames = []
    step = 0

    # Create initial render
    render_frame(initial_frame, ax, render_scale)
    plt.tight_layout()

    # Save initial frame
    plt.savefig(os.path.join(output_dir, "frames", f"frame_{step:04d}.png"))
    frames.append(os.path.join(output_dir, "frames", f"frame_{step:04d}.png"))

    print(f"Starting visualization simulation with {max_steps} max steps...")

    while not done and step < max_steps:
        step += 1

        # Update step counter
        print(f"Step {step}/{max_steps}", end='\r')

        # Select actions with trained policies
        try:
            # Use eval mode for deterministic actions
            for agent_id, policy in marl_ppo.agent_policies.items():
                policy.actor.eval()
                policy.critic.eval()

            actions = {}
            for agent_id in env.agent_ids:
                if agent_id in observations:
                    # Get the policy for this agent
                    if agent_id in marl_ppo.agent_policies:
                        # Get deterministic action
                        action, _, _ = marl_ppo.agent_policies[agent_id].select_action_with_logprob(
                            observations[agent_id])
                        actions[agent_id] = action

        except Exception as e:
            print(f"Error selecting actions: {e}")
            continue

        # Capture pre-step state
        pre_step_frame = capture_frame(step, env, actions)
        episode_data["frames"].append(pre_step_frame)

        # Execute actions
        try:
            next_observations, rewards, dones, truncs, infos = env.step(actions)
            observations = next_observations
        except Exception as e:
            print(f"Error during step execution: {e}")
            continue

        # Render current frame
        render_frame(pre_step_frame, ax, render_scale)
        plt.tight_layout()

        # Save frame
        frame_path = os.path.join(output_dir, "frames", f"frame_{step:04d}.png")
        plt.savefig(frame_path)
        frames.append(frame_path)

        # Show real-time display if requested
        if show_realtime:
            plt.pause(step_delay)

        # Check termination
        done = all(dones.values()) or all(truncs.values())

    # Save episode data - FIXED with serialize_numpy
    episode_data["metadata"]["steps"] = step
    if done:
        episode_data["metadata"]["success"] = not all(dones.values())
    else:
        episode_data["metadata"]["success"] = True  # Didn't terminate

    # Check final state
    final_frame = capture_frame(step, env)
    friendly_alive = sum(1 for unit in final_frame["friendly_units"].values() if unit["health"] > 0)
    enemy_alive = sum(1 for unit in final_frame["enemy_units"].values() if unit["health"] > 0)

    episode_data["metadata"]["friendly_survivors"] = friendly_alive
    episode_data["metadata"]["enemy_survivors"] = enemy_alive

    # Serialize all data to make it JSON serializable
    serialized_episode_data = serialize_numpy(episode_data)

    with open(os.path.join(output_dir, "episode_data.json"), 'w') as f:
        json.dump(serialized_episode_data, f, indent=2)

    video_path = None
    if save_video:
        video_path = os.path.join(output_dir, "simulation.mp4")

        # First try to use imageio
        imageio_available = False
        try:
            import imageio
            imageio_available = True
        except ImportError:
            print("imageio not found. Will try matplotlib animation.")

        if imageio_available:
            try:
                print("Creating video from frames using imageio...")
                # Load frames as images
                images = []
                for frame_path in frames:
                    images.append(imageio.imread(frame_path))

                # Create video
                imageio.mimsave(video_path, images, fps=video_fps)
                print(f"Video saved to {video_path}")
            except Exception as e:
                print(f"Error creating video with imageio: {e}")
                imageio_available = False

        # Fall back to matplotlib animation if imageio fails or is not available
        if not imageio_available:
            try:
                print("Creating video using matplotlib animation...")
                from matplotlib import animation

                # Create a new figure for the animation
                ani_fig = plt.figure(figsize=(12, 10))
                ani_ax = ani_fig.add_subplot(111)

                def init_animation():
                    ani_ax.clear()
                    return []

                def animate(i):
                    ani_ax.clear()
                    img = plt.imread(frames[i])
                    im = ani_ax.imshow(img)
                    return [im]

                # Create animation
                ani = animation.FuncAnimation(ani_fig, animate, frames=len(frames),
                                              init_func=init_animation, blit=True)

                # Save animation
                ani.save(video_path, fps=video_fps)
                print(f"Video saved to {video_path}")
                plt.close(ani_fig)
            except Exception as e:
                print(f"Error creating video with matplotlib animation: {e}")
                print("For video creation, install either:")
                print("  - imageio: pip install imageio imageio-ffmpeg")
                print("  - or ffmpeg: pip install matplotlib ffmpeg")

    plt.close()

    # Return paths to generated files and episode data
    result = {
        "output_dir": output_dir,
        "video_path": video_path,
        "episode_data_path": os.path.join(output_dir, "episode_data.json"),
        "frame_count": len(frames),
        "metadata": episode_data["metadata"]
    }

    print(f"\nVisualization complete! {len(frames)} frames generated.")
    print(f"Output saved to {output_dir}")

    return result


def visualize_trained_agents_with_logging(
        model_dir,
        map_file="training_map_lvl_1.csv",
        objective_location=(73, 75),
        enemy_positions=None,
        unit_start_positions=None,
        max_steps=200,
        output_dir=None,
        save_video=True,
        video_fps=5,
        render_scale=10,
        show_realtime=False,
        step_delay=0.2,
        detailed_logging=True,
        log_first_n_steps=100
):
    """
    Run a visual simulation of trained agents with detailed action logging.
    This is an enhanced version of visualize_trained_agents with comprehensive
    logging of agent actions, state changes, and rewards.

    Args:
        model_dir: Directory containing saved agent models to load
        map_file: CSV file with terrain information
        objective_location: (x, y) coordinates of the objective
        enemy_positions: List of (x, y) coordinates for enemy placements
        unit_start_positions: Dict mapping unit names to starting positions
        max_steps: Maximum steps to simulate
        output_dir: Directory to save visualization outputs
        save_video: Whether to save a video of the simulation
        video_fps: Frames per second for saved video
        render_scale: Scale factor for rendering (larger = bigger visualization)
        show_realtime: Whether to display the visualization in real-time
        step_delay: Delay between steps when showing real-time (seconds)
        detailed_logging: Whether to enable detailed action logging
        log_first_n_steps: How many steps to log in detail (to avoid huge log files)

    Returns:
        Path to saved video file, episode data, and log file
    """
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

    # Create output directory with timestamp if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./visualization_{timestamp}"

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    # Create log directory if detailed logging is enabled
    log_file = None
    if detailed_logging:
        logs_dir = os.path.join(output_dir, "action_logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "action_details.log")

        # Initialize log file
        with open(log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DETAILED ACTION LOG - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    # Log message helper function
    def log_message(message):
        print(message)
        if detailed_logging and log_file:
            with open(log_file, 'a') as f:
                f.write(message + "\n")

    # Initialize environment
    from WarGamingEnvironment_v13 import EnvironmentConfig, MARLMilitaryEnvironment, ForceType, UnitType

    # Estimate map size based on objective and unit positions
    all_positions = [objective_location] + list(unit_start_positions.values()) + enemy_positions
    max_x = max(pos[0] for pos in all_positions) + 20  # Add margin
    max_y = max(pos[1] for pos in all_positions) + 20  # Add margin

    # Create environment config
    env_config = EnvironmentConfig(width=max_x, height=max_y, debug_level=0)
    env = MARLMilitaryEnvironment(env_config, objective_position=objective_location)

    # Initialize PPO with test mode settings (no learning)
    from PPO_Training_v4 import WarGameMARLPPO
    marl_ppo = WarGameMARLPPO(env=env, action_dim=5, lr=0.0)  # Zero learning rate for visualization

    # Connect environment and PPO for observation handling
    setattr(env, 'marl_algorithm', marl_ppo)

    # Initialize agent role mapping to ensure consistent IDs
    env.agent_manager.initialize_agent_role_mapping()

    # Load trained agents
    log_message(f"Loading trained agents from {model_dir}...")
    marl_ppo.load_agents(model_dir)

    # Count how many agents were loaded
    agent_count = len(marl_ppo.agent_policies)
    log_message(f"Loaded {agent_count} trained agents")

    # Print info about loaded agents
    for agent_id, policy in marl_ppo.agent_policies.items():
        log_message(f"Agent {agent_id}: Update count = {policy.update_count}")

    # Helper function to convert numpy arrays to serializable objects
    def serialize_numpy(obj):
        """Convert numpy arrays and other non-serializable objects to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [serialize_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: serialize_numpy(value) for key, value in obj.items()}
        else:
            return obj

    # Data structure for recording episode
    episode_data = {
        "metadata": {
            "model_dir": model_dir,
            "objective": objective_location,
            "enemy_positions": enemy_positions,
            "unit_start_positions": unit_start_positions,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "frames": []
    }

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
            log_message(f"Loaded terrain from {map_file}")
        except Exception as e:
            log_message(f"Error loading terrain: {e}")

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

                    log_message(f"Positioned {string_id} at {new_pos}")

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

        log_message(f"Created platoon with {len(env.agent_ids)} consistently mapped agents")

    except Exception as e:
        log_message(f"Error creating custom platoon: {e}")
        log_message("Falling back to default platoon creation...")

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

    log_message(f"Added {len(enemy_ids)} enemy teams")

    # Set up visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # Function to capture state for each frame
    def capture_frame(step, env, actions=None):
        """Capture the current state of the environment for visualization"""
        frame_data = {
            "step": step,
            "friendly_units": {},
            "enemy_units": {},
            "actions": serialize_numpy(actions.copy() if actions else {}),
            "engagement_lines": []
        }

        # Capture unit states
        for unit_id in env.state_manager.active_units:
            force_type = env.get_unit_property(unit_id, 'force_type')
            unit_string_id = env.get_unit_property(unit_id, 'string_id', '')
            unit_type = env.get_unit_property(unit_id, 'unit_type', None)
            health = env.get_unit_property(unit_id, 'health', 0)
            position = env.get_unit_position(unit_id)
            ammo = env.get_unit_property(unit_id, 'ammo', 0)
            suppressed = env.get_unit_property(unit_id, 'suppressed', 0)

            unit_data = {
                "id": unit_id,
                "string_id": unit_string_id,
                "type": str(unit_type) if unit_type else "UNKNOWN",
                "health": health,
                "position": serialize_numpy(position),
                "ammo": ammo,
                "suppressed": suppressed
            }

            if force_type == ForceType.FRIENDLY:
                frame_data["friendly_units"][unit_string_id] = unit_data
            elif force_type == ForceType.ENEMY:
                frame_data["enemy_units"][unit_string_id] = unit_data

        # Track engagement lines
        if actions:
            for agent_id, action in actions.items():
                unit_id = env.agent_manager.get_current_unit_id(agent_id)
                if unit_id and action.get('action_type', -1) == 1:  # ENGAGE
                    unit_pos = env.get_unit_position(unit_id)
                    target_pos = action.get('engagement_params', {}).get('target_pos', [0, 0])
                    suppress_only = action.get('engagement_params', {}).get('suppress_only', 0) == 1

                    frame_data["engagement_lines"].append({
                        "from": serialize_numpy(unit_pos),
                        "to": serialize_numpy(target_pos),
                        "suppress_only": suppress_only
                    })

        return frame_data

    # Track agent states and actions for logging
    def log_agent_actions_and_states(step, actions, previous_states=None):
        """
        Log detailed information about agent actions and state changes.
        Only logs if detailed_logging is enabled and step is within log_first_n_steps.
        """
        if not detailed_logging or step > log_first_n_steps:
            return

        # Separator for new step
        log_message("\n" + "-" * 80)
        log_message(f"STEP {step}")
        log_message("-" * 80)

        # Track current states for comparison
        current_states = {}

        # Log each agent's action
        for agent_id, action in actions.items():
            unit_id = env.agent_manager.get_current_unit_id(agent_id)
            if unit_id:
                # Get string ID for easier identification
                string_id = env.get_unit_property(unit_id, 'string_id', str(unit_id))
                log_message(f"\nAGENT {agent_id} (Unit: {string_id}, ID: {unit_id})")

                # Get action type
                action_type = action.get('action_type', -1)
                action_types = ["MOVE", "ENGAGE", "SUPPRESS", "BOUND", "HALT", "CHANGE_FORMATION"]
                action_name = action_types[action_type] if 0 <= action_type < len(
                    action_types) else f"UNKNOWN({action_type})"

                log_message(f"Action: {action_name}")

                # Track current state
                current_states[agent_id] = {
                    'unit_id': unit_id,
                    'position': env.get_unit_position(unit_id),
                    'string_id': string_id,
                    'health': env.get_unit_property(unit_id, 'health', 0),
                    'formation': env.get_unit_property(unit_id, 'formation', 'unknown'),
                    'orientation': env.get_unit_property(unit_id, 'orientation', 0)
                }

                # For squad/team, track member positions
                unit_type = env.get_unit_property(unit_id, 'type')
                if (hasattr(unit_type, 'name') and ('TEAM' in str(unit_type.name) or 'SQUAD' in str(unit_type.name))) or \
                        ('TEAM' in str(unit_type) or 'SQUAD' in str(unit_type)):
                    current_states[agent_id]['members'] = {}
                    for member_id in env.get_unit_children(unit_id):
                        current_states[agent_id]['members'][member_id] = {
                            'position': env.get_unit_position(member_id),
                            'string_id': env.get_unit_property(member_id, 'string_id', str(member_id)),
                            'health': env.get_unit_property(member_id, 'health', 0)
                        }

                # Log action details based on type
                if action_type == 0:  # MOVE
                    direction = action.get('movement_params', {}).get('direction', (0, 0))
                    distance = action.get('movement_params', {}).get('distance', [1])
                    if isinstance(distance, (list, np.ndarray)):
                        distance = distance[0] if len(distance) > 0 else 1

                    log_message(f"  Direction: {direction}")
                    log_message(f"  Distance: {distance}")
                    log_message(f"  Current position: {current_states[agent_id]['position']}")

                    # Calculate expected new position
                    current_pos = current_states[agent_id]['position']
                    dx, dy = direction
                    magnitude = (dx ** 2 + dy ** 2) ** 0.5
                    if magnitude > 0:
                        expected_dx = int((dx * distance) / magnitude)
                        expected_dy = int((dy * distance) / magnitude)
                        expected_new_pos = (
                            max(0, min(env.width - 1, current_pos[0] + expected_dx)),
                            max(0, min(env.height - 1, current_pos[1] + expected_dy))
                        )
                        log_message(f"  Expected new position: {expected_new_pos}")

                elif action_type in [1, 2]:  # ENGAGE or SUPPRESS
                    target_pos = action.get('engagement_params', {}).get('target_pos', [0, 0])
                    max_rounds = action.get('engagement_params', {}).get('max_rounds', [10])
                    suppress_only = action.get('engagement_params', {}).get('suppress_only', 0)

                    if isinstance(max_rounds, (list, np.ndarray)):
                        max_rounds = max_rounds[0] if len(max_rounds) > 0 else 10

                    engagement_type = "Suppression" if action_type == 2 or suppress_only else "Direct"
                    log_message(f"  Engagement type: {engagement_type}")
                    log_message(f"  Target position: {target_pos}")
                    log_message(f"  Max rounds: {max_rounds}")

                    # Check for enemies at target position
                    enemies_at_target = []
                    for enemy_id in env.state_manager.active_units:
                        if env.get_unit_property(enemy_id, 'force_type') == ForceType.ENEMY:
                            enemy_pos = env.get_unit_position(enemy_id)
                            dist_to_target = ((enemy_pos[0] - target_pos[0]) ** 2 +
                                              (enemy_pos[1] - target_pos[1]) ** 2) ** 0.5
                            if dist_to_target <= 3:  # Within 3 cells of target
                                enemy_string = env.get_unit_property(enemy_id, 'string_id', str(enemy_id))
                                enemy_health = env.get_unit_property(enemy_id, 'health', 0)
                                enemies_at_target.append((enemy_id, enemy_string, enemy_health))

                    if enemies_at_target:
                        log_message(f"  Enemies near target:")
                        for enemy_id, enemy_string, enemy_health in enemies_at_target:
                            log_message(f"    {enemy_string} (ID: {enemy_id}), Health: {enemy_health}")
                    else:
                        log_message(f"  No enemies found near target position")

                elif action_type == 3:  # BOUND
                    direction = action.get('movement_params', {}).get('direction', (0, 0))
                    distance = action.get('movement_params', {}).get('distance', [1])
                    if isinstance(distance, (list, np.ndarray)):
                        distance = distance[0] if len(distance) > 0 else 1

                    log_message(f"  Direction: {direction}")
                    log_message(f"  Distance: {distance}")

                    # For squad bounding, show team positions
                    if 'members' in current_states[agent_id]:
                        log_message(f"  Current team positions:")
                        for member_id, member_data in current_states[agent_id]['members'].items():
                            log_message(f"    {member_data['string_id']}: {member_data['position']}")

                elif action_type == 4:  # HALT
                    log_message(f"  Halting movement")

                elif action_type == 5:  # CHANGE_FORMATION
                    formation_index = action.get('formation', 0)

                    # Map formation index to name
                    formation_map = {
                        0: "team_wedge_right",
                        1: "team_wedge_left",
                        2: "team_line_right",
                        3: "team_line_left",
                        4: "team_column",
                        5: "squad_column_team_wedge",
                        6: "squad_column_team_column",
                        7: "squad_line_team_wedge"
                    }

                    formation = formation_map.get(formation_index, f"unknown({formation_index})")
                    log_message(f"  Current formation: {current_states[agent_id]['formation']}")
                    log_message(f"  Target formation: {formation}")

                    # For squad/team, show member positions
                    if 'members' in current_states[agent_id]:
                        log_message(f"  Current member positions:")
                        for member_id, member_data in current_states[agent_id]['members'].items():
                            log_message(f"    {member_data['string_id']}: {member_data['position']}")

        return current_states

    # Function to log state changes after a step
    def log_state_changes(states_before, states_after, rewards):
        """Log changes in unit states after action execution"""
        if not detailed_logging or not states_before:
            return

        log_message("\nRESULTS:")

        # Track casualties in this step
        enemy_casualties = []
        friendly_casualties = []

        # Compare states before and after for each agent
        for agent_id, state_before in states_before.items():
            if agent_id not in states_after:
                continue

            state_after = states_after[agent_id]
            unit_id = state_after['unit_id']
            string_id = state_after['string_id']

            log_message(f"\nUnit: {string_id} (Agent {agent_id})")
            log_message(f"  Reward: {rewards.get(agent_id, 'N/A')}")

            # Position change
            pos_before = state_before['position']
            pos_after = state_after['position']
            if pos_after != pos_before:
                dx = pos_after[0] - pos_before[0]
                dy = pos_after[1] - pos_before[1]
                dist = ((dx ** 2 + dy ** 2) ** 0.5)
                log_message(f"  Position changed: {pos_before} -> {pos_after} (moved {dist:.2f} units)")
            else:
                log_message(f"  Position unchanged: {pos_after}")

            # Health change
            health_before = state_before['health']
            health_after = state_after['health']
            if health_after != health_before:
                log_message(f"  Health changed: {health_before} -> {health_after}")

            # Formation change
            formation_before = state_before['formation']
            formation_after = state_after['formation']
            if formation_after != formation_before:
                log_message(f"  Formation changed: {formation_before} -> {formation_after}")

            # Orientation change
            orientation_before = state_before['orientation']
            orientation_after = state_after['orientation']
            if orientation_after != orientation_before:
                log_message(f"  Orientation changed: {orientation_before}° -> {orientation_after}°")

            # For teams/squads, track member positions
            if 'members' in state_before and 'members' in state_after:
                log_message(f"  Member position changes:")

                # Track member changes
                for member_id, member_before in state_before['members'].items():
                    if member_id in state_after['members']:
                        member_after = state_after['members'][member_id]
                        member_string = member_after['string_id']

                        # Position change
                        member_pos_before = member_before['position']
                        member_pos_after = member_after['position']
                        if member_pos_after != member_pos_before:
                            member_dx = member_pos_after[0] - member_pos_before[0]
                            member_dy = member_pos_after[1] - member_pos_before[1]
                            member_dist = ((member_dx ** 2 + member_dy ** 2) ** 0.5)
                            log_message(
                                f"    {member_string}: {member_pos_before} -> {member_pos_after} (moved {member_dist:.2f} units)")

                        # Health change
                        member_health_before = member_before['health']
                        member_health_after = member_after['health']
                        if member_health_after != member_health_before:
                            log_message(f"    {member_string}: Health {member_health_before} -> {member_health_after}")

                            # Check for new casualty
                            if member_health_before > 0 and member_health_after <= 0:
                                force_type = env.get_unit_property(member_id, 'force_type')
                                if force_type == ForceType.ENEMY:
                                    enemy_casualties.append((member_id, member_string))
                                else:
                                    friendly_casualties.append((member_id, member_string))
                    else:
                        log_message(f"    {member_before['string_id']}: No longer present in unit")

        # Log casualties
        if enemy_casualties:
            log_message("\nEnemy casualties this step:")
            for unit_id, string_id in enemy_casualties:
                log_message(f"  {string_id} (ID: {unit_id})")

        if friendly_casualties:
            log_message("\nFriendly casualties this step:")
            for unit_id, string_id in friendly_casualties:
                log_message(f"  {string_id} (ID: {unit_id})")

    # Function to render a single frame
    def render_frame(frame_data, ax, render_scale=10):
        """Render a frame of the simulation"""
        ax.clear()

        # Set plot limits
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)

        # Plot terrain if available - FIXED: Use state_tensor instead of terrain_data
        try:
            # Access terrain from the state tensor instead
            if hasattr(env, 'state_manager') and hasattr(env.state_manager, 'state_tensor'):
                # Create a terrain heatmap from state tensor
                # The first channel (index 0) of the state tensor contains terrain type
                terrain_img = np.zeros((max_y, max_x))
                state_tensor = env.state_manager.state_tensor

                # Only copy terrain data within bounds
                for y in range(min(max_y, state_tensor.shape[0])):
                    for x in range(min(max_x, state_tensor.shape[1])):
                        # Get terrain value from state tensor
                        terrain_img[y, x] = state_tensor[y, x, 0]  # First channel is terrain

                # Plot terrain heatmap with a terrain-appropriate colormap
                ax.imshow(terrain_img, cmap='terrain', alpha=0.5, origin='lower', extent=(0, max_x, 0, max_y))
        except Exception as e:
            log_message(f"Error rendering terrain: {e}")

        # Draw objective
        objective_circle = Circle(objective_location, 2, color='green', alpha=0.7)
        ax.add_patch(objective_circle)
        ax.text(objective_location[0], objective_location[1], "OBJECTIVE",
                horizontalalignment='center', verticalalignment='center',
                color='black', fontweight='bold')

        # Draw friendly units
        for unit_id, unit_data in frame_data["friendly_units"].items():
            pos = unit_data["position"]
            health = unit_data["health"]

            # Set color based on health
            if health <= 0:
                color = 'black'  # Dead
                alpha = 0.5
            else:
                health_ratio = health / 100.0
                import matplotlib as mpl
                cmap = mpl.colormaps['RdYlGn']
                color = cmap(health_ratio)  # Red to Yellow to Green based on health
                alpha = 0.8

            # Draw unit with size based on unit type
            if "SQD" in unit_id:
                size = 3
            else:
                size = 2

            # Add suppression indicator
            if unit_data["suppressed"] > 0:
                # Draw a yellow ring around suppressed units
                suppression_ring = Circle(pos, size + 1, color='yellow', alpha=0.4)
                ax.add_patch(suppression_ring)

            unit_circle = Circle(pos, size, color=color, alpha=alpha)
            ax.add_patch(unit_circle)

            # Label the unit
            shortened_id = unit_id.split('-')[-1] if '-' in unit_id else unit_id
            ax.text(pos[0], pos[1], shortened_id,
                    horizontalalignment='center', verticalalignment='center',
                    color='white', fontweight='bold', fontsize=8)

            # Draw enemy units

        for unit_id, unit_data in frame_data["enemy_units"].items():
            pos = unit_data["position"]
            health = unit_data["health"]

            # Set color based on health
            if health <= 0:
                color = 'black'  # Dead
                alpha = 0.5
            else:
                color = 'red'  # Enemy
                alpha = 0.8

            enemy_circle = Circle(pos, 2, color=color, alpha=alpha)
            ax.add_patch(enemy_circle)

            # Label the unit
            shortened_id = unit_id.split('-')[-1] if '-' in unit_id else unit_id
            ax.text(pos[0], pos[1], shortened_id,
                    horizontalalignment='center', verticalalignment='center',
                    color='white', fontsize=8)

            # Draw engagement lines
        for engagement in frame_data.get("engagement_lines", []):
            from_pos = engagement["from"]
            to_pos = engagement["to"]
            suppress_only = engagement.get("suppress_only", False)

            # Draw the line with different style based on engagement type
            if suppress_only:
                # Suppression fire is dotted yellow
                ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                        color='yellow', linestyle=':', linewidth=1, alpha=0.7)
            else:
                # Direct fire is solid red
                ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                        color='red', linestyle='-', linewidth=1, alpha=0.7)

            # Add step counter and other information
        step = frame_data["step"]
        friendly_count = sum(1 for unit in frame_data["friendly_units"].values() if unit["health"] > 0)
        enemy_count = sum(1 for unit in frame_data["enemy_units"].values() if unit["health"] > 0)

        ax.set_title(f"Step: {step} | Friendly: {friendly_count} | Enemy: {enemy_count}")
        ax.grid(True, linestyle='--', alpha=0.7)

        return ax

        # Initialize termination flags

    dones = {}
    truncs = {}
    for agent_id in env.agent_ids:
        dones[agent_id] = False
        truncs[agent_id] = False

    # Main simulation loop
    done = False
    frames = []
    step = 0

    # Capture initial state
    initial_frame = capture_frame(0, env)
    episode_data["frames"].append(initial_frame)

    # Create initial render
    render_frame(initial_frame, ax, render_scale)
    plt.tight_layout()

    # Save initial frame
    plt.savefig(os.path.join(output_dir, "frames", f"frame_{step:04d}.png"))
    frames.append(os.path.join(output_dir, "frames", f"frame_{step:04d}.png"))

    log_message(f"Starting visualization simulation with {max_steps} max steps...")

    while not done and step < max_steps:
        step += 1

        # Update step counter
        print(f"Step {step}/{max_steps}", end='\r')

        # Select actions with trained policies
        try:
            # Use eval mode for deterministic actions
            for agent_id, policy in marl_ppo.agent_policies.items():
                policy.actor.eval()
                policy.critic.eval()

            actions = {}
            for agent_id in env.agent_ids:
                if agent_id in observations:
                    # Get the policy for this agent
                    if agent_id in marl_ppo.agent_policies:
                        # Get deterministic action
                        action, _, _ = marl_ppo.agent_policies[agent_id].select_action_with_logprob(
                            observations[agent_id])
                        actions[agent_id] = action

        except Exception as e:
            log_message(f"Error selecting actions: {e}")
            continue

        # Capture pre-step state
        pre_step_frame = capture_frame(step, env, actions)
        episode_data["frames"].append(pre_step_frame)

        # Log agent actions and states before step execution
        states_before = log_agent_actions_and_states(step, actions)

        # Execute actions
        try:
            next_observations, rewards, dones, truncs, infos = env.step(actions)
            observations = next_observations
        except Exception as e:
            log_message(f"Error during step execution: {e}")
            continue

        # Log agent states after step execution
        if states_before:
            states_after = {}
            for agent_id in states_before:
                unit_id = env.agent_manager.get_current_unit_id(agent_id)
                if unit_id:
                    states_after[agent_id] = {
                        'unit_id': unit_id,
                        'position': env.get_unit_position(unit_id),
                        'string_id': env.get_unit_property(unit_id, 'string_id', str(unit_id)),
                        'health': env.get_unit_property(unit_id, 'health', 0),
                        'formation': env.get_unit_property(unit_id, 'formation', 'unknown'),
                        'orientation': env.get_unit_property(unit_id, 'orientation', 0)
                    }

                    # For squad/team, track member positions
                    unit_type = env.get_unit_property(unit_id, 'type')
                    if (hasattr(unit_type, 'name') and (
                            'TEAM' in str(unit_type.name) or 'SQUAD' in str(unit_type.name))) or \
                            ('TEAM' in str(unit_type) or 'SQUAD' in str(unit_type)):
                        states_after[agent_id]['members'] = {}
                        for member_id in env.get_unit_children(unit_id):
                            states_after[agent_id]['members'][member_id] = {
                                'position': env.get_unit_position(member_id),
                                'string_id': env.get_unit_property(member_id, 'string_id', str(member_id)),
                                'health': env.get_unit_property(member_id, 'health', 0)
                            }

            # Log state changes
            log_state_changes(states_before, states_after, rewards)

        # Render current frame
        render_frame(pre_step_frame, ax, render_scale)
        plt.tight_layout()

        # Save frame
        frame_path = os.path.join(output_dir, "frames", f"frame_{step:04d}.png")
        plt.savefig(frame_path)
        frames.append(frame_path)

        # Show real-time display if requested
        if show_realtime:
            plt.pause(step_delay)

        # Check termination
        done = all(dones.values()) or all(truncs.values())

    # Save episode data - FIXED with serialize_numpy
    episode_data["metadata"]["steps"] = step
    if done:
        episode_data["metadata"]["success"] = not all(dones.values())
    else:
        episode_data["metadata"]["success"] = True  # Didn't terminate

    # Check final state
    final_frame = capture_frame(step, env)
    friendly_alive = sum(1 for unit in final_frame["friendly_units"].values() if unit["health"] > 0)
    enemy_alive = sum(1 for unit in final_frame["enemy_units"].values() if unit["health"] > 0)

    episode_data["metadata"]["friendly_survivors"] = friendly_alive
    episode_data["metadata"]["enemy_survivors"] = enemy_alive

    # Serialize all data to make it JSON serializable
    serialized_episode_data = serialize_numpy(episode_data)

    with open(os.path.join(output_dir, "episode_data.json"), 'w') as f:
        json.dump(serialized_episode_data, f, indent=2)

    video_path = None
    if save_video:
        video_path = os.path.join(output_dir, "simulation.mp4")

        # First try to use imageio
        imageio_available = False
        try:
            import imageio

            imageio_available = True
        except ImportError:
            log_message("imageio not found. Will try matplotlib animation.")

        if imageio_available:
            try:
                log_message("Creating video from frames using imageio...")
                # Load frames as images
                images = []
                for frame_path in frames:
                    images.append(imageio.imread(frame_path))

                # Create video
                imageio.mimsave(video_path, images, fps=video_fps)
                log_message(f"Video saved to {video_path}")
            except Exception as e:
                log_message(f"Error creating video with imageio: {e}")
                imageio_available = False

        # Fall back to matplotlib animation if imageio fails or is not available
        if not imageio_available:
            try:
                log_message("Creating video using matplotlib animation...")
                from matplotlib import animation

                # Create a new figure for the animation
                ani_fig = plt.figure(figsize=(12, 10))
                ani_ax = ani_fig.add_subplot(111)

                def init_animation():
                    ani_ax.clear()
                    return []

                def animate(i):
                    ani_ax.clear()
                    img = plt.imread(frames[i])
                    im = ani_ax.imshow(img)
                    return [im]

                # Create animation
                ani = animation.FuncAnimation(ani_fig, animate, frames=len(frames),
                                              init_func=init_animation, blit=True)

                # Save animation
                ani.save(video_path, fps=video_fps)
                log_message(f"Video saved to {video_path}")
                plt.close(ani_fig)
            except Exception as e:
                log_message(f"Error creating video with matplotlib animation: {e}")
                log_message("For video creation, install either:")
                log_message("  - imageio: pip install imageio imageio-ffmpeg")
                log_message("  - or ffmpeg: pip install matplotlib ffmpeg")

    plt.close()

    # Add completion message to log
    if detailed_logging and log_file:
        with open(log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ACTION LOGGING COMPLETE - Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

    # Return paths to generated files and episode data
    result = {
        "output_dir": output_dir,
        "video_path": video_path,
        "episode_data_path": os.path.join(output_dir, "episode_data.json"),
        "frame_count": len(frames),
        "metadata": episode_data["metadata"],
        "action_log_file": log_file
    }

    log_message(f"\nVisualization complete! {len(frames)} frames generated.")
    log_message(f"Output saved to {output_dir}")
    if log_file:
        log_message(f"Detailed action logs saved to {log_file}")

    return result


if __name__ == "__main__":
    # Path to your trained model
    model_dir = "./curriculum_training_20250330_1354/Stage7_Far_TopRight/models/best"

    # First run a statistical evaluation
    print("=== RUNNING STATISTICAL EVALUATION ===")
    eval_results = evaluate_trained_agents(
        model_dir=model_dir,
        num_episodes=5,  # Run 50 evaluation episodes for good statistics
        max_steps_per_episode=1000,

        # Scenario configuration
        map_file="training_map_lvl_1.csv",  # "training_map_lvl_1.csv"
        objective_location=(88, 50),  # (88, 50)
        enemy_positions=[],  # (50, 50), (73, 71), (78, 75), (66, 75)
        unit_start_positions={
            "1SQD": (10, 20),
            "2SQD": (10, 50),
            "3SQD": (10, 80),
            "GTM1": (10, 30),
            "GTM2": (10, 70),
            "JTM1": (10, 40),
            "JTM2": (10, 60)
        },

        # Output configuration
        output_dir="./curriculum_training_results",
        use_tqdm=True,
        verbose=True
    )

    # Print key evaluation results
    print(f"Win Rate: {eval_results['aggregate_stats']['win_rate']:.2f}")
    print(f"Average Episode Length: {eval_results['aggregate_stats']['avg_episode_length']:.1f} steps")
    print(f"Average Friendly Casualties: {eval_results['aggregate_stats']['avg_friendly_casualties']:.2f}")
    print(f"Average Enemy Casualties: {eval_results['aggregate_stats']['avg_enemy_casualties']:.2f}")

    # Then create a visualization with detailed action logging
    print("\n=== CREATING VISUALIZATION WITH DETAILED LOGGING ===")
    vis_results = visualize_trained_agents_with_logging(
        # Use the same model and scenario configuration
        model_dir=model_dir,
        map_file="training_map_lvl_1.csv",  # "training_map_lvl_1.csv"
        objective_location=(88, 50),  # (88, 50)
        enemy_positions=[],  # (50, 50), (73, 71), (78, 75), (66, 75)
        unit_start_positions={
            "1SQD": (10, 20),
            "2SQD": (10, 50),
            "3SQD": (10, 80),
            "GTM1": (10, 30),
            "GTM2": (10, 70),
            "JTM1": (10, 40),
            "JTM2": (10, 60)
        },

        # Visualization configuration
        max_steps=300,
        output_dir="./curriculum_training_results/visualization_with_logging",
        save_video=True,
        video_fps=10,  # 10 frames per second in video
        render_scale=10,
        show_realtime=True,  # Set to True to see the visualization in real-time
        step_delay=0.1,  # 100ms delay between steps when showing real-time

        # Logging configuration
        detailed_logging=True,
        log_first_n_steps=100  # Log only the first 100 steps
    )

    print(f"Visualization saved to: {vis_results['video_path']}")
    print(f"Detailed action log saved to: {vis_results['action_log_file']}")

# if __name__ == "__main__":
#     # Example of using the evaluation and visualization functions
#
#     # Path to your trained model
#     model_dir = "./initial_training_output/models/final"
#
#     # First run a statistical evaluation
#     print("=== RUNNING STATISTICAL EVALUATION ===")
#     eval_results = evaluate_trained_agents(
#         model_dir=model_dir,
#         num_episodes=2,  # Run 50 evaluation episodes for good statistics
#         max_steps_per_episode=300,
#
#         # Scenario configuration
#         map_file="training_map_lvl_1.csv",
#         objective_location=(73, 75),
#         enemy_positions=[
#             (73, 80), (73, 71), (78, 75), (66, 75)
#         ],
#         unit_start_positions={
#             "1SQD": (16, 10),
#             "2SQD": (9, 20),
#             "3SQD": (6, 30),
#             "GTM1": (6, 9),
#             "GTM2": (4, 24),
#             "JTM1": (6, 5),
#             "JTM2": (4, 32)
#         },
#
#         # Output configuration
#         output_dir="./evaluation_results",
#         use_tqdm=True,
#         verbose=True
#     )
#
#     # Print key evaluation results
#     print(f"Win Rate: {eval_results['aggregate_stats']['win_rate']:.2f}")
#     print(f"Average Episode Length: {eval_results['aggregate_stats']['avg_episode_length']:.1f} steps")
#     print(f"Average Friendly Casualties: {eval_results['aggregate_stats']['avg_friendly_casualties']:.2f}")
#     print(f"Average Enemy Casualties: {eval_results['aggregate_stats']['avg_enemy_casualties']:.2f}")
#
#     # Then create a visualization of the trained agents in action
#     print("\n=== CREATING VISUALIZATION ===")
#     vis_results = visualize_trained_agents(
#         # Use the same model and scenario configuration
#         model_dir=model_dir,
#         map_file="test_map.csv",
#         objective_location=(73, 75),
#         enemy_positions=[
#             (73, 80), (73, 71), (78, 75), (66, 75)
#         ],
#         unit_start_positions={
#             "1SQD": (16, 10),
#             "2SQD": (9, 20),
#             "3SQD": (6, 30),
#             "GTM1": (6, 9),
#             "GTM2": (4, 24),
#             "JTM1": (6, 5),
#             "JTM2": (4, 32)
#         },
#
#         # Visualization configuration
#         max_steps=300,
#         output_dir="./visualization_results",
#         save_video=True,
#         video_fps=10,  # 10 frames per second in video
#         render_scale=10,
#         show_realtime=True,  # Set to True to see the visualization in real-time
#         step_delay=0.1  # 100ms delay between steps when showing real-time
#     )
#
#     print(f"Visualization saved to: {vis_results['video_path']}")
#
#     # Example of testing with multiple scenario variations
#     def test_scenarios(model_dir, scenarios):
#         """Run evaluations with multiple scenario variations"""
#         scenario_results = {}
#
#         for scenario_name, scenario_config in scenarios.items():
#             print(f"\n=== EVALUATING SCENARIO: {scenario_name} ===")
#
#             # Create scenario-specific output directory
#             output_dir = f"./scenario_evaluations/{scenario_name}"
#
#             # Run evaluation with this scenario configuration
#             results = evaluate_trained_agents(
#                 model_dir=model_dir,
#                 num_episodes=30,  # Fewer episodes (30) per scenario for efficiency
#                 output_dir=output_dir,
#                 **scenario_config
#             )
#
#             scenario_results[scenario_name] = {
#                 "win_rate": results["aggregate_stats"]["win_rate"],
#                 "avg_friendly_casualties": results["aggregate_stats"]["avg_friendly_casualties"],
#                 "avg_enemy_casualties": results["aggregate_stats"]["avg_enemy_casualties"]
#             }
#
#             # Create visualization for this scenario
#             vis_output_dir = f"./scenario_visualizations/{scenario_name}"
#             visualize_trained_agents(
#                 model_dir=model_dir,
#                 output_dir=vis_output_dir,
#                 max_steps=200,
#                 **scenario_config
#             )
#
#         return scenario_results
#
#
#     # Define test scenarios
#     test_scenario_configs = {
#         "standard": {
#             "enemy_positions": [(73, 80), (73, 71), (78, 75), (66, 75)]
#         },
#         "enemy_reinforced": {
#             "enemy_positions": [(73, 80), (73, 71), (78, 75), (66, 75), (70, 70), (80, 80)]
#         },
#         "flanking": {
#             "enemy_positions": [(73, 80), (60, 60), (90, 60), (60, 90), (90, 90)]
#         },
#         "objective_defended": {
#             "enemy_positions": [(73, 76), (73, 74), (74, 75), (72, 75)]  # Enemies right at objective
#         }
#     }
#
#     # Uncomment to run scenario tests
#     # print("\n=== RUNNING SCENARIO TESTS ===")
#     # scenario_results = test_scenarios(model_dir, test_scenario_configs)
#     #
#     # # Print scenario comparison
#     # print("\n=== SCENARIO COMPARISON ===")
#     # for scenario, results in scenario_results.items():
#     #     print(f"{scenario}: Win Rate = {results['win_rate']:.2f}, " +
#     #           f"Friendly Casualties = {results['avg_friendly_casualties']:.2f}, " +
#     #           f"Enemy Casualties = {results['avg_enemy_casualties']:.2f}")
