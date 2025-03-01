import math
import os
import time
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Dict, List, Tuple

from matplotlib.patches import Circle

from Excel_to_CSV_Map_Converter import excel_to_csv
from tactical_position_analyzer import TacticalPosition, PositionPurpose, TacticalFilter, Threat

from tactical_route_analyzer_NEW import (
    TacticalRouteAnalyzer,
    RouteStrategy,
    EnemyThreat,
    SquadElement, TacticalRoute
)

from Planned_Route_to_MARL_Mvmt_Actions import (
    plan_and_execute_unit_route,
    plan_coordinated_unit_routes
)

from US_Army_PLT_Composition_vTest import (
    US_IN_Role,
    US_IN_UnitDesignator,
    US_IN_create_team,
    US_IN_create_squad,
    US_IN_create_platoon,
    US_IN_apply_formation,
    US_IN_handle_leader_casualty,
    US_IN_execute_movement, _get_unit_leader, _get_unit_members,
    US_IN_create_route, US_IN_execute_route_movement,
    RouteWaypoint, MovementRoute, MovementTechnique, execute_squad_movement, create_movement_order_from_routes
)

from WarGamingEnvironment_vTest import (
    MilitaryEnvironment,
    EnvironmentConfig,
    TerrainType,
    ElevationType,
    UnitType, VisibilityManager, BaseWeapon, CoordinationType
)

"""
US Army Infantry Unit Tests

This file contains tests to validate the integration between the US Army Infantry
composition file and the WarGaming Environment. It tests:

1. Team level operations
   - Team creation
   - Formation changes
   - Leader succession

2. Squad level operations
   - Squad creation with teams
   - Formation changes
   - Leader succession across teams
   - Agent identification/tracking

3. Platoon level operations
   - Full platoon creation
   - Formation changes
   - Leader succession across squads
   - Agent coordination structure
"""


def print_soldier_details(env: MilitaryEnvironment, soldier_id: int, indent: str = "") -> None:
    """Print detailed information about a soldier."""
    role = US_IN_Role(env.get_unit_property(soldier_id, 'role'))
    pos = env.get_unit_position(soldier_id)
    is_leader = env.get_unit_property(soldier_id, 'is_leader', False)

    # Print basic info
    print(f"{indent}{role.name}: {pos}{' (Leader)' if is_leader else ''}")

    # Print weapon details
    primary_weapon = env.get_unit_property(soldier_id, 'primary_weapon')
    if primary_weapon:
        print(f"{indent}    Primary: {primary_weapon.name} ({primary_weapon.ammo_capacity} rounds)")

    secondary_weapon = env.get_unit_property(soldier_id, 'secondary_weapon')
    if secondary_weapon:
        print(f"{indent}    Secondary: {secondary_weapon.name} ({secondary_weapon.ammo_capacity} rounds)")

    # Print health and ranges
    health = env.get_unit_property(soldier_id, 'health')
    obs_range = env.get_unit_property(soldier_id, 'observation_range')
    eng_range = env.get_unit_property(soldier_id, 'engagement_range')
    print(f"{indent}    Health: {health}%")
    print(f"{indent}    Observation Range: {obs_range}m")
    print(f"{indent}    Engagement Range: {eng_range}m")


def print_team_details(env: MilitaryEnvironment, team_id: int, indent: str = "") -> None:
    """Print detailed information about a team."""
    team_string = env.get_unit_property(team_id, 'string_id')
    team_pos = env.get_unit_position(team_id)
    team_formation = env.get_unit_property(team_id, 'formation_type', 'No formation')

    print(f"\n{indent}{team_string}:")
    print(f"{indent}Position: {team_pos}")
    print(f"{indent}Formation: {team_formation}")

    # Print team members
    team_members = env.get_unit_children(team_id)
    leader_id = next((mid for mid in team_members
                      if env.get_unit_property(mid, 'is_leader')), None)

    # Print leader first
    if leader_id:
        print_soldier_details(env, leader_id, indent + "    ")

    # Print other members
    for member_id in team_members:
        if member_id != leader_id:
            print_soldier_details(env, member_id, indent + "    ")


def print_squad_details(env: MilitaryEnvironment, squad_id: int, indent: str = "") -> None:
    """Print detailed information about a squad."""
    squad_string = env.get_unit_property(squad_id, 'string_id')
    squad_pos = env.get_unit_position(squad_id)
    squad_formation = env.get_unit_property(squad_id, 'formation_type', 'No formation')

    print(f"\n{indent}{squad_string}:")
    print(f"{indent}Position: {squad_pos}")
    print(f"{indent}Formation: {squad_formation}")

    # Print squad leader first
    squad_members = env.get_unit_children(squad_id)
    sl_id = next((mid for mid in squad_members
                  if env.get_unit_property(mid, 'role') == US_IN_Role.SQUAD_LEADER.value), None)
    if sl_id:
        print_soldier_details(env, sl_id, indent + "    ")

    # Print teams
    teams = [mid for mid in squad_members
             if env.get_unit_property(mid, 'type') == UnitType.INFANTRY_TEAM]
    for team_id in teams:
        print_team_details(env, team_id, indent + "    ")


def print_platoon_details(env: MilitaryEnvironment, plt_id: int) -> None:
    """Print detailed information about a platoon."""
    plt_string = env.get_unit_property(plt_id, 'string_id')
    plt_formation = env.get_unit_property(plt_id, 'formation_type', 'No formation')
    print(f"\nPlatoon Composition ({plt_string}):")
    print(f"Formation: {plt_formation}")

    # Print platoon leader
    plt_members = env.get_unit_children(plt_id)
    pl_id = next((mid for mid in plt_members
                  if env.get_unit_property(mid, 'role') == US_IN_Role.PLATOON_LEADER.value), None)
    if pl_id:
        print_soldier_details(env, pl_id, "    ")

    # Print squads
    squads = [mid for mid in plt_members
              if env.get_unit_property(mid, 'type') == UnitType.INFANTRY_SQUAD]
    print(f"\nSquads ({len(squads)}):")
    for squad_id in squads:
        print_squad_details(env, squad_id, "    ")

    # Print weapons teams
    wpn_teams = [mid for mid in plt_members
                 if env.get_unit_property(mid, 'type') == UnitType.WEAPONS_TEAM]
    print(f"\nWeapons Teams ({len(wpn_teams)}):")
    for team_id in wpn_teams:
        print_team_details(env, team_id, "    ")


def test_team_operations():
    """Test team level operations with enhanced information display."""
    print("\n=== Testing Team Operations ===")

    # Initialize environment
    config = EnvironmentConfig(width=20, height=20, debug_level=1)
    env = MilitaryEnvironment(config)

    # Create fire team
    print("\nCreating Fire Team...")
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.ALPHA_TEAM,
        start_position=(5, 5)
    )

    # Print initial configuration
    print("\nInitial Team Configuration:")
    print_team_details(env, team_id)

    # Test formation change
    print("\nChanging to Wedge Right Formation...")
    US_IN_apply_formation(env, team_id, "team_wedge_right")
    print("\nNew Team Configuration:")
    print_team_details(env, team_id)

    # Test leader succession
    print("\nTesting Leader Succession...")
    print("Team Leader becomes casualty")

    # Find and "kill" team leader
    members = env.get_unit_children(team_id)
    tl_id = next(mid for mid in members
                 if env.get_unit_property(mid, 'role') == US_IN_Role.TEAM_LEADER.value)
    env.update_unit_property(tl_id, 'health', 0)

    # Handle succession
    success = US_IN_handle_leader_casualty(env, team_id)
    print(f"Succession successful: {success}")

    # Print final configuration
    print("\nTeam Configuration After Succession:")
    print_team_details(env, team_id)


def test_squad_operations():
    """Test squad level operations with enhanced information display."""
    print("\n=== Testing Squad Operations ===")

    # Initialize environment
    config = EnvironmentConfig(width=30, height=30, debug_level=1)
    env = MilitaryEnvironment(config)

    # Create squad
    print("\nCreating Infantry Squad...")
    squad_id = US_IN_create_squad(
        env=env,
        plt_num=1,
        squad_num=1,
        start_position=(10, 10)
    )

    # Print initial configuration
    print("\nInitial Squad Configuration:")
    print_squad_details(env, squad_id)

    # Test formation change
    print("\nChanging to Squad Line Formation...")
    US_IN_apply_formation(env, squad_id, "squad_line_team_wedge")
    print("\nNew Squad Configuration:")
    print_squad_details(env, squad_id)

    # Test leader succession
    print("\nTesting Squad Leader Succession...")
    print("Squad Leader becomes casualty")

    # Find and "kill" squad leader
    members = env.get_unit_children(squad_id)
    sl_id = next(mid for mid in members
                 if env.get_unit_property(mid, 'role') == US_IN_Role.SQUAD_LEADER.value)
    env.update_unit_property(sl_id, 'health', 0)

    # Handle succession
    success = US_IN_handle_leader_casualty(env, squad_id)
    print(f"Succession successful: {success}")

    # Print final configuration
    print("\nSquad Configuration After Succession:")
    print_squad_details(env, squad_id)


def test_platoon_operations():
    """Test platoon level operations with enhanced information display."""
    print("\n=== Testing Platoon Operations ===")

    # Initialize environment
    config = EnvironmentConfig(width=50, height=50, debug_level=1)
    env = MilitaryEnvironment(config)

    # Create platoon
    print("\nCreating Infantry Platoon...")
    plt_id = US_IN_create_platoon(
        env=env,
        plt_num=1,
        start_position=(25, 25)
    )

    # Print initial configuration
    print("\nInitial Platoon Configuration:")
    print_platoon_details(env, plt_id)

    # Test formation change
    print("\nChanging to Platoon Line Formation...")
    US_IN_apply_formation(env, plt_id, "platoon_line")
    print("\nNew Platoon Configuration:")
    print_platoon_details(env, plt_id)

    # Test leader succession
    print("\nTesting Platoon Leader Succession...")
    print("Platoon Leader becomes casualty")

    # Find and "kill" platoon leader
    members = env.get_unit_children(plt_id)
    pl_id = next(mid for mid in members
                 if env.get_unit_property(mid, 'role') == US_IN_Role.PLATOON_LEADER.value)
    env.update_unit_property(pl_id, 'health', 0)

    # Handle succession
    success = US_IN_handle_leader_casualty(env, plt_id)
    print(f"Succession successful: {success}")

    # Print final configuration
    print("\nPlatoon Configuration After Succession:")
    print_platoon_details(env, plt_id)


"""Test team movement with advanced scenarios including complex routes and leadership changes."""


def test_team_movement():
    """Test team movements with increasingly complex scenarios."""
    print("\n=== Testing Team Movement ===")

    # Initialize environment
    config = EnvironmentConfig(width=100, height=100, debug_level=0)
    env = MilitaryEnvironment(config)

    # Stage 1: Basic Movement Tests
    print("\n=== Stage 1: Basic Movement Tests ===")
    test_basic_movements(env)

    # Stage 2: Movement with Leadership Changes
    print("\n=== Stage 3: Movement with Leadership Changes ===")
    test_movement_with_succession(env)


def test_basic_movements(env: MilitaryEnvironment):
    """Test basic wagon wheel and follow leader movements."""
    # Test 1: Wagon Wheel Movement (Wedge Formation)
    print("\nTest 1: Wagon Wheel Movement (Wedge Formation)")
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.ALPHA_TEAM,
        start_position=(50, 50)
    )

    # Apply wedge formation
    print("Applying initial wedge formation...")
    US_IN_apply_formation(env, team_id, "team_wedge_right")

    # Basic movement sequence
    movements = [
        ((0, 1), 10, "Forward"),  # Move north
        ((1, 1), 10, "Diagonal right"),  # Move northeast
        ((-1, 1), 10, "Diagonal left")  # Move northwest
    ]

    execute_movement_sequence(env, team_id, movements,
                              "basic_wagon_wheel", (30, 70, 30, 70))

    # Test 2: Follow Leader Movement (Column Formation)
    print("\nTest 2: Follow Leader Movement (Column Formation)")
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.BRAVO_TEAM,
        start_position=(50, 50)
    )

    # Apply column formation
    print("Applying initial column formation...")
    US_IN_apply_formation(env, team_id, "team_column")

    # Basic movement sequence
    movements = [
        ((0, 1), 10, "Forward"),  # Move north
        ((1, 0), 10, "Right"),  # Move east
        ((1, 1), 10, "Diagonal")  # Move northeast
    ]

    execute_movement_sequence(env, team_id, movements,
                              "basic_follow_leader", (30, 70, 30, 70))


def test_movement_with_succession(env: MilitaryEnvironment):
    """Test movement with leadership changes."""

    # Test 1: Wagon Wheel with Leadership Change
    print("\nTest 1: Wagon Wheel Formation with Leadership Change")
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.ALPHA_TEAM,
        start_position=(50, 50)
    )
    US_IN_apply_formation(env, team_id, "team_wedge_right")

    # First leg of movement
    initial_movements = [
        ((0, 1), 10, "Initial Forward"),
        ((1, 0), 10, "Initial Right")
    ]

    frames = execute_movement_sequence(env, team_id, initial_movements,
                                       "succession_wagon_wheel_p1", (20, 80, 20, 80))

    # Change leadership
    print("\nExecuting leadership change...")
    leader_id = next(mid for mid in env.get_unit_children(team_id)
                     if env.get_unit_property(mid, 'is_leader'))
    env.update_unit_property(leader_id, 'health', 0)
    success = US_IN_handle_leader_casualty(env, team_id)
    print(f"Leadership succession successful: {success}")

    # Continue movement with new leader
    remaining_movements = [
        ((1, 1), 10, "New Leader Northeast"),
        ((0, -1), 10, "New Leader South"),
        ((-1, 0), 10, "New Leader West")
    ]

    continue_frames = execute_movement_sequence(env, team_id, remaining_movements,
                                                "succession_wagon_wheel_p2", (20, 80, 20, 80))
    frames.extend(continue_frames)

    # Create combined animation
    create_movement_animation(frames, "Wagon Wheel Movement with Leadership Change",
                              "wagon_wheel_succession_complete.gif")

    # Test 2: Follow Leader with Leadership Change
    print("\nTest 2: Column Formation with Leadership Change")
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.BRAVO_TEAM,
        start_position=(50, 50)
    )
    US_IN_apply_formation(env, team_id, "team_column")

    # First leg of movement
    initial_movements = [
        ((0, 1), 10, "Initial Forward"),
        ((1, 0), 10, "Initial Right")
    ]

    frames = execute_movement_sequence(env, team_id, initial_movements,
                                       "succession_follow_leader_p1", (20, 80, 20, 80))

    # Change leadership
    print("\nExecuting leadership change...")
    leader_id = next(mid for mid in env.get_unit_children(team_id)
                     if env.get_unit_property(mid, 'is_leader'))
    env.update_unit_property(leader_id, 'health', 0)
    success = US_IN_handle_leader_casualty(env, team_id)
    print(f"Leadership succession successful: {success}")

    # Continue movement with new leader
    remaining_movements = [
        ((1, 1), 10, "New Leader Northeast"),
        ((0, -1), 10, "New Leader South"),
        ((-1, -1), 10, "New Leader Southwest")
    ]

    continue_frames = execute_movement_sequence(env, team_id, remaining_movements,
                                                "succession_follow_leader_p2", (20, 80, 20, 80))
    frames.extend(continue_frames)

    # Create combined animation
    create_movement_animation(frames, "Follow Leader Movement with Leadership Change",
                              "follow_leader_succession_complete.gif")


def execute_movement_sequence(env: MilitaryEnvironment, team_id: int,
                              movements: List[Tuple], animation_name: str,
                              view_bounds: Tuple[int, int, int, int]) -> List[Dict]:
    """Execute a sequence of movements and create visualizations."""
    all_frames = []

    # Get initial positions for recording
    initial_frame = capture_team_positions(env, team_id)
    all_frames.append(initial_frame)

    # Execute each movement
    for direction, distance, description in movements:
        print(f"\nExecuting {description} movement:")
        print(f"Direction: {direction}, Distance: {distance}")

        # Get team's current orientation before move
        orientation = env.get_unit_property(team_id, 'orientation')
        print(f"Starting orientation: {orientation}°")

        # Execute movement
        frames = US_IN_execute_movement(env, team_id, direction, distance)
        all_frames.extend(frames)

        # Get final orientation
        new_orientation = env.get_unit_property(team_id, 'orientation')
        print(f"Ending orientation: {new_orientation}°")

    # Create visualization
    create_movement_animation(all_frames, f"Team Movement - {animation_name}",
                              f"{animation_name}.gif", view_bounds)

    return all_frames


def capture_team_positions(env: MilitaryEnvironment, team_id: int) -> Dict:
    """Capture current positions of all team members."""
    positions = []

    # Get all team members
    members = env.get_unit_children(team_id)

    for member_id in members:
        positions.append({
            'role': env.get_unit_property(member_id, 'role'),
            'position': env.get_unit_position(member_id),
            'is_leader': env.get_unit_property(member_id, 'is_leader', False)
        })

    return {
        'unit_type': 'Team',
        'team_id': team_id,
        'positions': positions
    }


def _capture_positions(env: MilitaryEnvironment, unit_id: int) -> Dict:
    """Capture positions of all unit elements."""
    unit_type = env.get_unit_property(unit_id, 'type')
    unit_string = env.get_unit_property(unit_id, 'string_id')

    positions = []

    # Add leader
    leader_id = _get_unit_leader(env, unit_id)
    if leader_id:
        leader_pos = env.get_unit_position(leader_id)
        print(f"Capturing leader position: {leader_pos}")  # Debug output
        positions.append({
            'role': env.get_unit_property(leader_id, 'role'),
            'position': leader_pos,
            'is_leader': True
        })

    # Add members
    for member_id in _get_unit_members(env, unit_id):
        member_pos = env.get_unit_position(member_id)
        print(f"Capturing member {member_id} position: {member_pos}")  # Debug output
        positions.append({
            'role': env.get_unit_property(member_id, 'role'),
            'position': member_pos,
            'is_leader': False
        })

    frame = {
        'unit_type': unit_type,
        'unit_id': unit_string,
        'positions': positions
    }
    return frame


def create_movement_animation(frames: List[Dict], title: str, filename: str,
                              view_bounds: Tuple[int, int, int, int] = (0, 100, 0, 100)):
    """Create animation showing team movement with role labels."""
    print(f"\nCreating animation: {title}")
    min_x, max_x, min_y, max_y = view_bounds

    # Verify frames are unique and in sequence
    unique_frames = []
    seen_positions = set()

    for frame in frames:
        # Create position tuple for this frame
        frame_positions = tuple(
            (pos['role'], pos['position'][0], pos['position'][1])
            for pos in frame['positions']
        )

        # Only add if we haven't seen these positions before
        if frame_positions not in seen_positions:
            unique_frames.append(frame)
            seen_positions.add(frame_positions)

    print(f"Processing {len(unique_frames)} unique frames")

    # Create figure with higher DPI for better quality
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()
        ax.grid(True)

        # Plot each member
        for member in frame['positions']:
            pos = member['position']
            is_leader = member['is_leader']
            role = member['role']

            # Different marker for leader
            marker = 'o' if is_leader else '^'
            color = 'red' if is_leader else 'blue'
            size = 100 if is_leader else 80

            ax.scatter(pos[0], pos[1], c=color, marker=marker, s=size)

            # Add role label
            role_name = US_IN_Role(role).name if isinstance(role, int) else str(role)
            ax.annotate(role_name, (pos[0], pos[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_title(f"{title}\nFrame {unique_frames.index(frame) + 1}/{len(unique_frames)}")
        ax.set_aspect('equal')

        return []

    # Create animation with smoother frame rate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=unique_frames,
                                   interval=50,  # Decreased interval for smoother animation
                                   blit=True,
                                   repeat=False)  # Don't repeat the animation

    # Save animation with higher quality settings
    writer = animation.PillowWriter(fps=5)  # Increased FPS for smoother playback
    print(f"Saving animation to: {filename}")
    anim.save(filename, writer=writer)
    plt.close()


def test_route_movement():
    """Test route-based movement with varying complexity."""
    print("\n=== Testing Route Movement ===")

    # Initialize environment
    config = EnvironmentConfig(width=100, height=100, debug_level=1)
    env = MilitaryEnvironment(config)

    # Test 1: Simple Route
    print("\n=== Test 1: Simple A-to-B Route ===")
    test_simple_route(env)

    # Test 2: Multi-point Route
    print("\n=== Test 2: Multi-point Route with Formation Changes ===")
    test_multi_point_route(env)

    # Test 3: Complex Route
    # print("\n=== Test 3: Complex Route with Holds and Mixed Formations ===")
    # test_complex_route(env)


def test_simple_route(env: MilitaryEnvironment):
    """Test simple A-to-B route movement."""
    print("\nTesting simple A-to-B route movement...")

    # Create team at start position
    start_pos = (20, 20)
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.ALPHA_TEAM,
        start_position=start_pos
    )

    # Verify initial position
    actual_pos = env.get_unit_position(team_id)
    print(f"Initial position: {actual_pos}")

    # Create simple route
    # Note: First waypoint should be different from start position
    waypoints = [
        (20, 30),  # First waypoint - move north
        (40, 40)  # Second waypoint - move northeast
    ]

    # Print waypoint sequence
    print("\nWaypoint sequence:")
    for i, pos in enumerate(waypoints):
        print(f"Waypoint {i + 1}: {pos}")

    # Create route with formations for each waypoint
    route = US_IN_create_route(
        waypoints=waypoints,
        technique=MovementTechnique.TRAVELING,
        formations=["team_wedge_right", "team_wedge_right"]  # Formation at each waypoint
    )

    # Execute route movement
    print("\nExecuting simple route movement...")
    frames = US_IN_execute_route_movement(env, team_id, route, debug_level=1)

    # Create visualization
    create_movement_animation(
        frames,
        "Simple Route Movement",
        "simple_route.gif",
        (10, 50, 10, 50)
    )


def test_multi_point_route(env: MilitaryEnvironment):
    """Test route with multiple waypoints and formation changes."""
    # Create team
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.BRAVO_TEAM,
        start_position=(30, 30)
    )

    # Create multi-point route
    waypoints = [
        (30, 30),  # Start
        (50, 30),  # East
        (50, 50),  # North
        (30, 50),  # West
        (30, 30)  # Return to start
    ]

    # Different formation at each waypoint
    formations = [
        "team_wedge_right",
        "team_line_right",
        "team_wedge_left",
        "team_line_left",
        "team_wedge_right"
    ]

    route = US_IN_create_route(
        waypoints=waypoints,
        technique=MovementTechnique.TRAVELING,
        formations=formations
    )

    # Execute route movement
    print("\nExecuting multi-point route movement...")
    frames = US_IN_execute_route_movement(env, team_id, route)

    # Create visualization
    create_movement_animation(
        frames,
        "Multi-point Route with Formation Changes",
        "multi_point_route.gif",
        (20, 60, 20, 60)
    )


"""Test squad movement with advanced scenarios including complex routes and movement techniques."""


def test_squad_movement_system():
    """Test suite for squad-level movement mechanics."""
    print("\n=== Testing Squad Movement System ===")

    # Initialize environment with larger space for movement
    config = EnvironmentConfig(width=100, height=100, debug_level=1)
    env = MilitaryEnvironment(config)

    # Test 1: Formation Changes During Movement
    # print("\n=== Test 1: Formation Changes During Movement ===")
    # test_squad_formations(env)

    # Test 2: Complex Route Execution
    # print("\n=== Test 2: Complex Route Execution ===")
    # test_squad_route(env)

    # Test 3: Mixed Movement Techniques
    print("\n=== Test 3: Mixed Movement Techniques ===")
    test_mixed_movement(env)

    # Test 4: Bounding Movement Coordination
    print("\n=== Test 4: Bounding Movement Coordination ===")
    test_bounding_coordination(env)


def test_squad_formations(env: MilitaryEnvironment):
    """Test squad movement with formation changes."""
    print("\nTesting squad movement with formation changes...")

    # Create squad at start position
    squad_id = US_IN_create_squad(
        env=env,
        plt_num=1,
        squad_num=1,
        start_position=(20, 20)
    )

    # Define waypoints with formations and movements
    waypoints = [
        RouteWaypoint(
            position=(40, 40),  # Diagonal movement target: 20,20 -> 40,40
            formation='squad_column_team_wedge',
            hold_time=0
        ),
        RouteWaypoint(
            position=(60, 40),  # Horizontal movement target: 40,40 -> 60,40
            formation='squad_line_team_wedge',
            hold_time=0
        )
    ]

    # Create route
    route = MovementRoute(
        waypoints=waypoints,
        technique=MovementTechnique.TRAVELING
    )

    # Execute route movement
    frames = execute_squad_movement(
        env=env,
        squad_id=squad_id,
        direction=(0, 0),  # Not used for route movement
        distance=0,  # Not used for route movement
        technique=MovementTechnique.TRAVELING,
        debug_level=1,
        route=route
    )

    # Create visualization
    create_movement_animation(
        frames,
        "Squad Formation Changes",
        "squad_formations.gif",
        (10, 70, 10, 70)  # Adjusted bounds to see full movement
    )


def test_squad_route(env: MilitaryEnvironment):
    """Test squad movement along a complex route."""
    print("\nTesting squad route movement...")

    # Create squad at start position
    squad_id = US_IN_create_squad(
        env=env,
        plt_num=1,
        squad_num=2,
        start_position=(30, 30)
    )

    # Define complex route with formation changes
    waypoints = [
        RouteWaypoint(
            position=(50, 30),
            formation='squad_column_team_wedge',
            hold_time=2
        ),
        RouteWaypoint(
            position=(50, 50),
            formation='squad_line_team_wedge',
            hold_time=2
        ),
        RouteWaypoint(
            position=(30, 50),
            formation='squad_column_team_wedge',
            hold_time=2
        ),
        RouteWaypoint(
            position=(30, 30),
            formation='squad_line_team_wedge',
            hold_time=0
        )
    ]

    # Create route
    route = MovementRoute(
        waypoints=waypoints,
        technique=MovementTechnique.TRAVELING
    )

    # Execute route movement using enhanced execute_squad_movement
    frames = execute_squad_movement(
        env=env,
        squad_id=squad_id,
        direction=(0, 0),  # Not used for route movement
        distance=0,  # Not used for route movement
        technique=MovementTechnique.TRAVELING,
        debug_level=1,
        route=route
    )

    # Create visualization
    create_movement_animation(
        frames,
        "Squad Route Movement",
        "squad_route.gif",
        (20, 60, 20, 60)
    )


def test_mixed_movement(env: MilitaryEnvironment):
    """Test squad movement with mixed techniques."""
    print("\nTesting mixed movement techniques...")

    # Create squad at start position
    squad_id = US_IN_create_squad(
        env=env,
        plt_num=1,
        squad_num=3,
        start_position=(40, 40)
    )

    # Define waypoints with mixed techniques
    waypoints = [
        RouteWaypoint(
            position=(60, 40),  # Move east using traveling
            formation='squad_column_team_wedge',
            hold_time=2
        ),
        RouteWaypoint(
            position=(60, 60),  # Move north using bounding
            formation='squad_line_team_wedge',
            hold_time=2
        ),
        RouteWaypoint(
            position=(40, 60),  # Move west using traveling
            formation='squad_column_team_wedge',
            hold_time=2
        ),
        RouteWaypoint(
            position=(40, 40),  # Return to start using bounding
            formation='squad_line_team_wedge',
            hold_time=0
        )
    ]

    # Create routes with alternating techniques
    traveling_route = MovementRoute(
        waypoints=[waypoints[0]],
        technique=MovementTechnique.TRAVELING
    )

    bounding_route = MovementRoute(
        waypoints=[waypoints[1]],
        technique=MovementTechnique.BOUNDING
    )

    traveling_return = MovementRoute(
        waypoints=[waypoints[2]],
        technique=MovementTechnique.TRAVELING
    )

    bounding_return = MovementRoute(
        waypoints=[waypoints[3]],
        technique=MovementTechnique.BOUNDING
    )

    # Execute movement sequence
    all_frames = []

    # Move east using traveling
    print("\nExecuting traveling movement east...")
    frames = execute_squad_movement(
        env=env,
        squad_id=squad_id,
        direction=(0, 0),
        distance=0,
        technique=MovementTechnique.TRAVELING,
        debug_level=1,
        route=traveling_route
    )
    all_frames.extend(frames)

    # Move north using bounding
    print("\nExecuting bounding movement north...")
    frames = execute_squad_movement(
        env=env,
        squad_id=squad_id,
        direction=(0, 0),
        distance=0,
        technique=MovementTechnique.BOUNDING,
        debug_level=1,
        route=bounding_route
    )
    all_frames.extend(frames)

    # Move west using traveling
    print("\nExecuting traveling movement west...")
    frames = execute_squad_movement(
        env=env,
        squad_id=squad_id,
        direction=(0, 0),
        distance=0,
        technique=MovementTechnique.TRAVELING,
        debug_level=1,
        route=traveling_return
    )
    all_frames.extend(frames)

    # Return to start using bounding
    print("\nExecuting bounding movement south...")
    frames = execute_squad_movement(
        env=env,
        squad_id=squad_id,
        direction=(0, 0),
        distance=0,
        technique=MovementTechnique.BOUNDING,
        debug_level=1,
        route=bounding_return
    )
    all_frames.extend(frames)

    # Create visualization
    create_movement_animation(
        all_frames,
        "Mixed Movement Techniques",
        "mixed_movement.gif",
        (30, 70, 30, 70)
    )


def test_bounding_coordination(env: MilitaryEnvironment):
    """Test coordination between teams during bounding movement."""
    print("\nTesting bounding movement coordination...")

    # Create squad
    squad_id = US_IN_create_squad(
        env=env,
        plt_num=1,
        squad_num=4,
        start_position=(20, 80)
    )

    # Apply initial formation
    US_IN_apply_formation(env, squad_id, 'squad_line_team_wedge')

    # Execute long-distance bounding movement
    frames = execute_squad_movement(
        env=env,
        squad_id=squad_id,
        direction=(1, 0),  # Move east
        distance=200,
        technique=MovementTechnique.BOUNDING,
        debug_level=1
    )

    # Create visualization using existing animation system
    create_movement_animation(
        frames,
        "Bounding Movement Coordination",
        "bounding_coordination.gif",
        (10, 90, 70, 90)
    )


# Map Conversion - Environment Tests

def test_map_conversion():
    """Test map conversion and display results."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(project_dir, "map_design.xlsx")
    csv_file = os.path.join(project_dir, "generated_map.csv")

    print("Converting Excel to CSV...")
    excel_to_csv(excel_file, csv_file)

    df = pd.read_csv(csv_file)
    print("\nFirst 5 rows of generated CSV:")
    print(df.head())

    config = EnvironmentConfig(width=100, height=100, debug_level=1)
    env = MilitaryEnvironment(config)

    # Initialize terrain first
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)

    # Then load from CSV
    env.terrain_manager.load_from_csv(csv_file)

    # Create visualization
    plt.figure(figsize=(15, 10))

    terrain_colors = {
        TerrainType.BARE.value: [0.9, 0.9, 0.9],
        TerrainType.SPARSE_VEG.value: [0.8, 0.9, 0.8],
        TerrainType.DENSE_VEG.value: [0.4, 0.7, 0.4],
        TerrainType.WOODS.value: [0.2, 0.5, 0.2],
        TerrainType.STRUCTURE.value: [0.6, 0.6, 0.7]
    }

    # Convert terrain data to RGB
    terrain_data = env.state_manager.state_tensor[:, :, 0]
    elevation_data = env.state_manager.state_tensor[:, :, 1]
    rgb_map = np.zeros((*terrain_data.shape, 3))

    for terrain_value, color in terrain_colors.items():
        mask = terrain_data == terrain_value
        for i in range(3):
            rgb_map[:, :, i][mask] = color[i]

    # Adjust for elevation
    for i in range(3):
        rgb_map[:, :, i][elevation_data == ElevationType.ELEVATED_LEVEL.value] *= 1.2
        rgb_map[:, :, i][elevation_data == ElevationType.LOWER_LEVEL.value] *= 0.8

    # Plot maps
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_map)
    plt.title('Terrain Map')
    plt.colorbar(label='Terrain Type')

    plt.subplot(1, 2, 2)
    elevation_plot = plt.imshow(elevation_data, cmap='terrain')
    plt.title('Elevation Map')
    plt.colorbar(elevation_plot, label='Elevation Level')

    plt.suptitle('Generated Environment Map')
    plt.tight_layout()
    plt.savefig('environment_map.png')
    plt.close()

    print("\nMap visualization saved as 'environment_map.png'")

    # Print statistics
    unique_terrain, terrain_counts = np.unique(terrain_data, return_counts=True)
    print("\nTerrain Distribution:")
    for t_value, count in zip(unique_terrain, terrain_counts):
        terrain_type = TerrainType(t_value).name
        percentage = (count / terrain_data.size) * 100
        print(f"{terrain_type}: {percentage:.1f}%")


def test_planned_movements():
    """Test route planning and execution for different unit sizes."""
    # Setup environment
    project_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(project_dir, "map_design.xlsx")
    csv_file = os.path.join(project_dir, "generated_map.csv")

    print("\nConverting map data...")
    excel_to_csv(excel_file, csv_file)

    config = EnvironmentConfig(width=400, height=100, debug_level=1)
    env = MilitaryEnvironment(config)

    # Initialize terrain
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)
    env.terrain_manager.load_from_csv(csv_file)

    # Test parameters
    objective = (350, 50)
    enemy_positions = [
        (346, 50),  # Main position
        (340, 48),  # Support
        (342, 52)  # Security
    ]

    # Initialize analyzer
    analyzer = TacticalRouteAnalyzer(
        terrain=env.terrain_manager.state_tensor[:, :, 0:2],  # Get terrain and elevation channels
        env_width=env.width,
        env_height=env.height,
        objective=objective
    )

    # Add enemy threats
    print("\nSetting up enemy threats...")
    for pos in enemy_positions:
        threat = EnemyThreat(
            position=pos,
            unit=None,  # Simplified for test
            observation_range=48,  # 480m
            engagement_range=30,  # 300m
            suspected_accuracy=0.8
        )
        analyzer.enemy_threats.append(threat)

    print("\n=== Testing Team Movement ===")
    test_team_planned_movement(env, analyzer, objective)

    print("\n=== Testing Squad Movement ===")
    test_squad_planned_movement(env, analyzer, objective)

    print("\n=== Testing Platoon Movement ===")
    test_platoon_planned_movement(env, analyzer, objective)


def test_team_planned_movement(env: MilitaryEnvironment,
                               analyzer: TacticalRouteAnalyzer,
                               objective: Tuple[int, int]):
    """Test team route planning and execution."""
    print("Creating fire team...")
    team_id = US_IN_create_team(
        env=env,
        plt_num=1,
        squad_num=1,
        designator=US_IN_UnitDesignator.ALPHA_TEAM,
        start_position=(50, 50)
    )

    print("Planning and executing team movement...")
    movements = plan_and_execute_unit_route(
        env=env,
        unit_id=team_id,
        objective=objective,
        terrain_analyzer=analyzer,
        strategy=RouteStrategy.BALANCED
    )

    print(f"Generated {len(movements)} movement frames")
    print(f"Start position: {env.get_unit_position(team_id)}")

    if movements:
        child_units = env.get_unit_children(team_id)
        print(f"Team member positions:")
        for child_id in child_units:
            print(f"- {env.get_unit_property(child_id, 'role')}: {env.get_unit_position(child_id)}")


def test_squad_planned_movement(env: MilitaryEnvironment,
                                analyzer: TacticalRouteAnalyzer,
                                objective: Tuple[int, int]):
    """Test squad route planning and execution."""
    print("Creating squad...")
    squad_id = US_IN_create_squad(
        env=env,
        plt_num=1,
        squad_num=1,
        start_position=(50, 50)
    )

    print("Planning and executing squad movement...")
    movements = plan_and_execute_unit_route(
        env=env,
        unit_id=squad_id,
        objective=objective,
        terrain_analyzer=analyzer,
        strategy=RouteStrategy.BALANCED
    )

    print(f"Generated {len(movements)} movement frames")
    print(f"Squad position: {env.get_unit_position(squad_id)}")

    # Print team positions
    teams = [unit for unit in env.get_unit_children(squad_id)
             if env.get_unit_property(unit, 'type') == UnitType.INFANTRY_TEAM]
    print("Team positions:")
    for team_id in teams:
        print(f"- {env.get_unit_property(team_id, 'string_id')}: {env.get_unit_position(team_id)}")


def test_platoon_planned_movement(env: MilitaryEnvironment,
                                  analyzer: TacticalRouteAnalyzer,
                                  objective: Tuple[int, int]):
    """Test platoon route planning and execution."""
    print("Creating platoon...")
    plt_id = US_IN_create_platoon(
        env=env,
        plt_num=1,
        start_position=(50, 50)
    )

    # Get subordinate units
    squads = []
    weapons_teams = []
    for unit_id in env.get_unit_children(plt_id):
        unit_type = env.get_unit_property(unit_id, 'type')
        if unit_type == UnitType.INFANTRY_SQUAD:
            squads.append(unit_id)
        elif unit_type == UnitType.WEAPONS_TEAM:
            weapons_teams.append(unit_id)

    print(f"Planning movement for {len(squads)} squads and {len(weapons_teams)} weapons teams...")
    movements = plan_coordinated_unit_routes(
        env=env,
        unit_ids=squads + weapons_teams,
        objective=objective
    )

    total_frames = sum(len(m) for m in movements.values())
    print(f"Generated {total_frames} total movement frames")

    # Print unit positions
    print("\nFinal positions:")
    for unit_id in squads:
        print(f"Squad {env.get_unit_property(unit_id, 'string_id')}: {env.get_unit_position(unit_id)}")
    for unit_id in weapons_teams:
        print(f"Team {env.get_unit_property(unit_id, 'string_id')}: {env.get_unit_position(unit_id)}")


# Testing visibility characteristics (cover - concealment - fields of view)
def test_visibility_manager():
    """
    Test VisibilityManager calculations with detailed output.
    Tests terrain effects on visibility, concealment, cover, and weapon effectiveness.
    """
    print("\n=== Testing VisibilityManager ===")

    # Initialize environment
    config = EnvironmentConfig(width=100, height=100, debug_level=1)
    env = MilitaryEnvironment(config)
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)

    # Define test weapons
    test_weapons = {
        'M4': BaseWeapon("M4 Carbine", 50, 210, 1, 40),  # 500m range, single shot
        'M249': BaseWeapon("M249 SAW", 80, 600, 6, 45, True),  # 800m range, burst fire
        'M240': BaseWeapon("M240B", 100, 1000, 9, 50, True),  # 1000m range, burst fire
        'Javelin': BaseWeapon("Javelin", 200, 3, 1, 1000, True)  # 2000m range, high damage
    }

    # Set up specific terrain for testing
    terrain_setups = {
        'bare': (10, 10, TerrainType.BARE),
        'sparse': (20, 20, TerrainType.SPARSE_VEG),
        'dense': (30, 30, TerrainType.DENSE_VEG),
        'woods': (40, 40, TerrainType.WOODS),
        'structure': (50, 50, TerrainType.STRUCTURE)
    }

    # Set terrain in environment
    print("\nSetting up test terrain...")
    for name, (x, y, terrain_type) in terrain_setups.items():
        # Set terrain type in state tensor
        env.state_manager.state_tensor[y, x, 0] = terrain_type.value
        # Verify terrain was set
        actual_type = env.terrain_manager.get_terrain_type((x, y))
        print(f"Position ({x}, {y}): Set {name.upper()} terrain -> Got {actual_type.name}")

    # Create visibility manager
    visibility_mgr = VisibilityManager(env)

    # Test positions at different ranges and elevations
    test_positions = [
        {
            'name': 'Short Distance',
            'pos': (15, 15),
            'elevation': ElevationType.GROUND_LEVEL
        },
        {
            'name': 'Medium Distance',
            'pos': (35, 35),
            'elevation': ElevationType.ELEVATED_LEVEL
        },
        {
            'name': 'Long Distance',
            'pos': (55, 55),
            'elevation': ElevationType.LOWER_LEVEL
        }
    ]

    # Set test position elevations
    for pos_info in test_positions:
        x, y = pos_info['pos']
        env.state_manager.state_tensor[y, x, 1] = pos_info['elevation'].value

    print("\nTesting weapon effectiveness across terrain and ranges...")

    # Test each weapon type
    for weapon_name, weapon in test_weapons.items():
        print(f"\n{'-' * 20}")
        print(f"Testing {weapon_name}:")
        print(f"Max Range: {weapon.max_range * 10}m")
        print(f"Base Damage: {weapon.damage}")
        print(f"Fire Rate: {weapon.fire_rate}")
        print(f"Area Weapon: {weapon.is_area_weapon}")

        # Test against each position
        for pos_info in test_positions:
            print(f"\nTesting against {pos_info['name']} at {pos_info['pos']}:")
            target_pos = pos_info['pos']

            # Test from each terrain type
            for terrain_name, (x, y, _) in terrain_setups.items():
                shooter_pos = (x, y)
                print(f"\nFiring from {terrain_name.upper()} terrain:")

                # Calculate actual distance
                distance = math.sqrt(
                    (target_pos[0] - shooter_pos[0]) ** 2 +
                    (target_pos[1] - shooter_pos[1]) ** 2
                )
                print(f"Distance: {distance * 10:.0f}m")

                # Get terrain effects
                los_result = visibility_mgr.check_line_of_sight(shooter_pos, target_pos)
                # Create visualization
                visualize_line_of_sight(
                    env=env,
                    shooter_pos=shooter_pos,
                    target_pos=target_pos,
                    los_result=los_result,
                    title=f"LOS Analysis - {weapon_name} from {terrain_name}"
                )

                print("\nLine of Sight Check:")
                print(f"- Has LOS: {los_result['has_los']}")
                print(f"- LOS Quality: {los_result['los_quality']:.2f}")
                if los_result['degradation']:
                    print("- Degradation Sources:")
                    for deg in los_result['degradation']:
                        print(f"  * {deg}")
                    if 'degradation_details' in los_result:
                        details = los_result['degradation_details']
                        print(f"- Total Path Length: {details['path_length']} cells")
                        print(f"- Total Degradation: {details['total']:.2f}")
                        print("- Degradation by Terrain Type:")
                        for terrain_type, cells in details['by_terrain'].items():
                            print(f"  * {terrain_type.name}: {cells} cells")
                        print(f"- FOV Penalty: {details['fov_penalty']:.2f}")

                if los_result['has_los']:
                    # Calculate hit probability
                    base_hit_prob = env._calculate_hit_probability(distance, weapon)

                    # Create test unit for hit probability modification
                    test_unit_id = 1
                    visibility_mgr.record_unit_fired(test_unit_id)  # Simulate unit has fired

                    # Get modified hit probability
                    modified_hit_prob = visibility_mgr.modify_hit_probability(
                        base_hit_prob,
                        shooter_pos,
                        target_pos,
                        test_unit_id
                    )

                    print("\nHit Probability Analysis:")
                    print(f"- Base Hit Probability: {base_hit_prob:.2f}")
                    print(f"- Modified Hit Probability: {modified_hit_prob:.2f}")

                    # Calculate damage
                    base_damage = env._calculate_damage(distance, weapon)
                    modified_damage = visibility_mgr.modify_damage(
                        base_damage,
                        target_pos,
                        shooter_pos
                    )

                    print("\nDamage Analysis:")
                    print(f"- Base Damage: {base_damage:.1f}")
                    print(f"- Modified Damage: {modified_damage:.1f}")

                    # Show major factors
                    print("\nMajor Factors:")
                    range_percentage = (distance / weapon.max_range) * 100
                    if range_percentage <= 50.0:
                        range_desc = f"Close Range ({range_percentage:.0f}% of max)"
                    elif range_percentage <= 89.0:
                        range_desc = f"Medium Range ({range_percentage:.0f}% of max)"
                    else:
                        range_desc = f"Long Range ({range_percentage:.0f}% of max)"
                    print(f"- Range: {range_desc}")
                    print(f"- Distance: {distance * 10:.0f}m of {weapon.max_range * 10:.0f}m max range")
                    print(f"- LOS Quality: {los_result['los_quality']:.2f}")
                    if los_result['elevation_advantage']:
                        print(f"- Has Elevation Advantage")

                    # Calculate effectiveness score
                    effectiveness = modified_hit_prob * modified_damage
                    print(f"\nOverall Effectiveness Score: {effectiveness:.1f}")
                else:
                    print("\nNo line of sight - cannot engage target")

    visibility_mgr.reset_fired_units()
    print("\nVisibility Manager test complete.")


def visualize_line_of_sight(env: MilitaryEnvironment,
                            shooter_pos: Tuple[int, int],
                            target_pos: Tuple[int, int],
                            los_result: Dict,
                            title: str = "Line of Sight Analysis") -> None:
    """
    Create visualization showing terrain, shooter, target, and line of sight path.

    Args:
        env: Military environment instance
        shooter_pos: Position of shooter (x,y)
        target_pos: Position of target (x,y)
        los_result: Result from line of sight check
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Create terrain visualization
    terrain_colors = {
        TerrainType.BARE: '#E8E8E8',  # Light gray
        TerrainType.SPARSE_VEG: '#C8E6C9',  # Light green
        TerrainType.DENSE_VEG: '#81C784',  # Medium green
        TerrainType.WOODS: '#2E7D32',  # Dark green
        TerrainType.STRUCTURE: '#616161'  # Dark gray
    }

    # Create terrain grid
    terrain_grid = np.zeros((env.height, env.width, 3))
    for y in range(env.height):
        for x in range(env.width):
            terrain_type = env.terrain_manager.get_terrain_type((x, y))
            color = terrain_colors[terrain_type]
            # Convert hex to RGB
            rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            terrain_grid[y, x] = rgb

    # Get line of sight points using environment's method
    los_points = env._get_line_points(
        shooter_pos[0], shooter_pos[1],
        target_pos[0], target_pos[1]
    )

    # Plot base terrain
    ax.imshow(terrain_grid, origin='lower')

    # Plot line of sight path with color based on success
    path_x = [p[0] for p in los_points]
    path_y = [p[1] for p in los_points]
    if los_result['has_los']:
        ax.plot(path_x, path_y, 'y--', linewidth=2, label='Line of Sight', alpha=0.7)
    else:
        ax.plot(path_x, path_y, 'r--', linewidth=2, label='Blocked Line of Sight', alpha=0.7)

    # Plot degradation points if available
    if 'degradation' in los_result:
        for deg in los_result['degradation']:
            if 'at (' in deg:
                # Extract coordinates from string like "DENSE_VEG at (30,30)"
                coords = deg[deg.find("(") + 1:deg.find(")")].split(',')
                x, y = int(coords[0]), int(coords[1])
                ax.plot(x, y, 'rx', markersize=10, label=f'Degradation: {deg}')

    # Plot shooter and target
    ax.plot(shooter_pos[0], shooter_pos[1], 'b^', markersize=15, label='Shooter')
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')

    # Calculate view bounds to show relevant area
    min_x = min(shooter_pos[0], target_pos[0]) - 5
    max_x = max(shooter_pos[0], target_pos[0]) + 5
    min_y = min(shooter_pos[1], target_pos[1]) - 5
    max_y = max(shooter_pos[1], target_pos[1]) + 5

    # Ensure bounds are within environment
    min_x = max(0, min_x)
    max_x = min(env.width - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(env.height - 1, max_y)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Add grid
    ax.grid(True, color='gray', linestyle='-', alpha=0.3)

    # Add terrain type legend
    terrain_patches = [plt.Rectangle((0, 0), 1, 1, fc=color)
                       for terrain_type, color in terrain_colors.items()]
    terrain_labels = [terrain_type.name for terrain_type in terrain_colors.keys()]

    # Combine both legends
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(terrain_patches)
    labels.extend(terrain_labels)
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    # Add title with distance and LOS quality
    distance = math.sqrt((target_pos[0] - shooter_pos[0]) ** 2 +
                         (target_pos[1] - shooter_pos[1]) ** 2)
    los_quality = los_result['los_quality'] if los_result['has_los'] else 0.0
    title = f"{title}\nDistance: {distance * 10:.0f}m, LOS Quality: {los_quality:.2f}"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
    plt.close()


# Route Planning
def test_tactical_route_analyzer(env: MilitaryEnvironment,
                                 start_pos: Tuple[int, int],
                                 objective: Tuple[int, int],
                                 enemy_threats: List[Dict],
                                 element_type: SquadElement,
                                 debug_level: int = 0) -> Dict:
    """
    Test tactical route planning with controlled debug output.

    Debug Levels:
    0 - Silent (errors only)
    1 - Basic progress and results
    2 - Detailed analysis and intermediate steps

    Args:
        env: Military environment instance
        start_pos: Starting position
        objective: Objective position
        enemy_threats: List of enemy threat dictionaries
        element_type: Type of element (ASSAULT/SUPPORT/RESERVE)
        debug_level: Level of debug output
    """
    results = {
        'success': False,
        'route': None,
        'error': None,
        'metrics': {},
        'phase_timings': {},
        'visualization_data': {}
    }

    try:
        # Phase 1: Initialize Analyzers
        if debug_level >= 1:
            print("\nPhase 1: Initializing Analyzers...")
        start_time = time.time()

        analyzer = TacticalRouteAnalyzer(env, objective)

        # Add enemy threats
        for threat in enemy_threats:
            analyzer.threat_analyzer.add_threat(Threat(
                position=threat['position'],
                unit=None,
                observation_range=threat['observation_range'],
                engagement_range=threat['engagement_range'],
                suspected_accuracy=threat['accuracy']
            ))

        results['phase_timings']['initialization'] = time.time() - start_time
        if debug_level >= 2:
            print(f"Added {len(enemy_threats)} enemy threats")

        # Phase 2: Position Analysis
        if debug_level >= 1:
            print("\nPhase 2: Analyzing Tactical Positions...")
        start_time = time.time()

        tactical_filter = TacticalFilter(env)
        position_purpose = SquadElement.to_position_purpose(element_type)

        positions_result = tactical_filter.find_positions(
            objective=objective,
            unit_type=UnitType.INFANTRY_TEAM,
            position_purpose=position_purpose,
            start_positions=[start_pos],
            enemy_threats=enemy_threats,
            max_positions=5
        )

        tactical_positions = positions_result['tactical_positions']
        if not tactical_positions:
            raise ValueError("No valid tactical positions found")

        results['phase_timings']['position_analysis'] = time.time() - start_time
        results['metrics']['num_positions'] = len(tactical_positions)

        if debug_level >= 1:
            print(f"Found {len(tactical_positions)} tactical positions")

        if debug_level >= 2:
            for i, pos in enumerate(tactical_positions):
                print(f"\nPosition {i + 1}:")
                print(f"Location: {pos.position}")
                print(f"Quality Score: {pos.quality_score:.2f}")
                print(f"Cover: {pos.cover_score:.2f}")
                print(f"Concealment: {pos.concealment_score:.2f}")

        # Phase 3: Route Planning
        if debug_level >= 1:
            print("\nPhase 3: Planning Tactical Route...")
        start_time = time.time()

        route = analyzer.find_routes_with_coordination(
            start_pos=start_pos,
            tactical_positions=tactical_positions,
            element_type=element_type,
            strategy=RouteStrategy.BALANCED,
            max_threat_exposure=0.7,
            # debug_level=debug_level  # Pass through debug level
        )

        if not route:
            raise ValueError("No valid route found")

        results['phase_timings']['route_planning'] = time.time() - start_time
        results['route'] = route

        # Calculate metrics
        results['metrics'].update({
            'total_distance': route.total_distance,
            'avg_cover': route.avg_cover,
            'avg_concealment': route.avg_concealment,
            'threat_exposure': route.total_threat_exposure,
            'movement_time': route.movement_time_estimate,
            'quality_score': route.quality_score,
            'num_segments': len(route.segments),
            'num_coordination_points': len(route.coordination_points)
        })

        if debug_level >= 1:
            print("\nRoute Planning Complete:")
            print(f"Segments: {len(route.segments)}")
            print(f"Distance: {route.total_distance:.0f}m")
            print(f"Quality Score: {route.quality_score:.2f}")

        if debug_level >= 2:
            print("\nDetailed Route Analysis:")
            print(f"Average Cover: {route.avg_cover:.2f}")
            print(f"Average Concealment: {route.avg_concealment:.2f}")
            print(f"Threat Exposure: {route.total_threat_exposure:.2f}")
            print(f"Estimated Movement Time: {route.movement_time_estimate:.1f} minutes")
            print(f"Coordination Points: {len(route.coordination_points)}")

        # Phase 4: Visualization
        if debug_level >= 1:
            print("\nPhase 4: Creating Visualization...")
        start_time = time.time()

        visualization_data = create_route_visualization(
            env=env,
            route=route,
            enemy_threats=enemy_threats,
            tactical_positions=tactical_positions,
            start_pos=start_pos,
            objective=objective,
            debug_level=debug_level
        )

        results['phase_timings']['visualization'] = time.time() - start_time
        results['visualization_data'] = visualization_data
        results['success'] = True

        if debug_level >= 1:
            print("\nTest Complete - Route Found Successfully")

    except Exception as e:
        results['error'] = str(e)
        if debug_level >= 0:  # Always show errors
            print(f"\nTest Failed: {str(e)}")
            if debug_level >= 2:
                traceback.print_exc()

    return results


def create_route_visualization(env: MilitaryEnvironment,
                               route: TacticalRoute,
                               enemy_threats: List[Dict],
                               tactical_positions: List[TacticalPosition],
                               start_pos: Tuple[int, int],
                               objective: Tuple[int, int],
                               debug_level: int = 0) -> Dict:
    """Create visualization of tactical route with controlled debug output."""
    if debug_level >= 2:
        print("\nCreating route visualization...")

    # Set up figure
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot terrain base
    terrain_colors = {
        TerrainType.BARE: [0.9, 0.9, 0.9],
        TerrainType.SPARSE_VEG: [0.8, 0.9, 0.8],
        TerrainType.DENSE_VEG: [0.4, 0.7, 0.4],
        TerrainType.WOODS: [0.2, 0.5, 0.2],
        TerrainType.STRUCTURE: [0.6, 0.6, 0.7]
    }

    if debug_level >= 2:
        print("Creating terrain grid...")

    terrain_grid = np.zeros((env.height, env.width, 3))
    for y in range(env.height):
        for x in range(env.width):
            terrain_type = env.terrain_manager.get_terrain_type((x, y))
            terrain_grid[y, x] = terrain_colors[terrain_type]

    ax.imshow(terrain_grid, origin='lower')

    if debug_level >= 2:
        print("Adding enemy positions and ranges...")

    # Plot enemy positions and ranges
    for threat in enemy_threats:
        pos = threat['position']
        obs_circle = Circle(pos, threat['observation_range'],
                            fill=False, linestyle='--', color='red', alpha=0.3)
        eng_circle = Circle(pos, threat['engagement_range'],
                            fill=False, color='red', alpha=0.5)
        ax.add_patch(obs_circle)
        ax.add_patch(eng_circle)
        ax.plot(pos[0], pos[1], 'r^', markersize=10)

    if debug_level >= 2:
        print("Adding tactical positions...")

    # Plot tactical positions
    for pos in tactical_positions:
        ax.plot(pos.position[0], pos.position[1], 'go', markersize=8, alpha=0.5)
        ax.annotate(f'{pos.quality_score:.2f}',
                    (pos.position[0], pos.position[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)

    if debug_level >= 2:
        print("Adding route segments...")

    # Plot route segments
    colors = {
        'APPROACH': 'blue',
        'TACTICAL': 'green',
        'ASSAULT': 'red'
    }

    for segment in route.segments:
        x_coords = [p[0] for p in segment.path]
        y_coords = [p[1] for p in segment.path]
        line_style = '--' if segment.movement_technique == 'bounding' else '-'
        color = colors.get(segment.segment_type.name, 'gray')

        ax.plot(x_coords, y_coords,
                color=color,
                linestyle=line_style,
                alpha=0.8,
                linewidth=2)

        if segment.coordination_point:
            coord = segment.coordination_point
            ax.plot(coord.position[0], coord.position[1],
                    'wo', markersize=10, markeredgecolor=color)
            circle = Circle(coord.position, 5,
                            fill=False, color=color, alpha=0.3)
            ax.add_patch(circle)

    # Plot start and objective
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
    ax.plot(objective[0], objective[1], 'r*', markersize=15, label='Objective')

    ax.legend()
    ax.set_title(f'Tactical Route - {route.element_type.name}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)

    filename = f'tactical_route_{route.element_type.name.lower()}.png'
    if debug_level >= 1:
        print(f"Saving visualization to {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'filename': filename,
        'terrain_grid': terrain_grid,
        'route_segments': [(segment.path, segment.segment_type.name)
                           for segment in route.segments],
        'coordination_points': [(coord.position, coord.coord_type.name)
                                for coord in route.coordination_points]
    }


def test_route_analyzer_scenarios(debug_level: int = 0):
    """Run test scenarios for tactical route analyzer with controlled debug output."""
    # Initialize environment
    config = EnvironmentConfig(width=100, height=100, debug_level=debug_level)
    env = MilitaryEnvironment(config)

    # Load terrain
    try:
        env.terrain_manager.load_from_csv('generated_map.csv')
        if debug_level >= 1:
            print("Loaded terrain from CSV")
    except Exception as e:
        if debug_level >= 1:
            print(f"Using default terrain: {str(e)}")
        env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)

    # Define test scenarios
    scenarios = [
        {
            'name': 'Support Element Route',
            'start_pos': (20, 50),
            'objective': (80, 50),
            'element_type': SquadElement.SUPPORT,
            'enemy_threats': [
                {
                    'position': (75, 50),
                    'observation_range': 48,
                    'engagement_range': 30,
                    'accuracy': 0.8
                },
                {
                    'position': (70, 48),
                    'observation_range': 40,
                    'engagement_range': 25,
                    'accuracy': 0.7
                }
            ]
        },
        {
            'name': 'Assault Element Route',
            'start_pos': (20, 45),
            'objective': (80, 50),
            'element_type': SquadElement.ASSAULT,
            'enemy_threats': [
                {
                    'position': (75, 50),
                    'observation_range': 48,
                    'engagement_range': 30,
                    'accuracy': 0.8
                },
                {
                    'position': (70, 48),
                    'observation_range': 40,
                    'engagement_range': 25,
                    'accuracy': 0.7
                }
            ]
        }
    ]

    all_results = {}

    for scenario in scenarios:
        if debug_level >= 1:
            print(f"\n{'=' * 80}")
            print(f"Testing Scenario: {scenario['name']}")
            print(f"{'=' * 80}")

        results = test_tactical_route_analyzer(
            env=env,
            start_pos=scenario['start_pos'],
            objective=scenario['objective'],
            enemy_threats=scenario['enemy_threats'],
            element_type=scenario['element_type'],
            debug_level=debug_level
        )

        all_results[scenario['name']] = results

        # Print scenario summary based on debug level
        if debug_level >= 1:
            print(f"\nScenario Results - {scenario['name']}:")
            print(f"Success: {results['success']}")

            if results['success']:
                if debug_level >= 2:
                    print("\nDetailed Metrics:")
                    for metric, value in results['metrics'].items():
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.2f}")
                        else:
                            print(f"  {metric}: {value}")

                    print("\nPhase Timings:")
                    for phase, time in results['phase_timings'].items():
                        print(f"  {phase}: {time:.2f}s")
                else:
                    # Basic metrics for debug_level 1
                    print(f"Route Quality: {results['metrics']['quality_score']:.2f}")
                    print(f"Total Distance: {results['metrics']['total_distance']:.0f}m")
                    print(f"Number of Segments: {results['metrics']['num_segments']}")
            else:
                if debug_level >= 0:  # Always show errors
                    print(f"Error: {results['error']}")

    return all_results


if __name__ == "__main__":
    # Run tests in order
    test_team_operations()
    # test_squad_operations()
    # test_platoon_operations()
    # test_team_movement()
    # test_route_movement()
    # test_squad_movement_system()
    # test_map_conversion()
    # test_planned_movements()
    # test_visibility_manager()
    # test_tactical_route_analyzer
    # results = test_route_analyzer_scenarios(debug_level=2)
    #
    # # Print overall summary
    # print("\nOverall Test Summary:")
    # for scenario, result in results.items():
    #     print(f"\n{scenario}:")
    #     print(f"Success: {result['success']}")
    #     if result['success']:
    #         print(f"Route Quality: {result['metrics']['quality_score']:.2f}")
    #         print(f"Total Distance: {result['metrics']['total_distance']:.0f}m")
    #         print(f"Segments: {result['metrics']['num_segments']}")
    #         print(f"Coordination Points: {result['metrics']['num_coordination_points']}")

