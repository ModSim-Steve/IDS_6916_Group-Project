"""
Test suite for TacticalRouteAnalyzer with saved positions.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle
from typing import Dict, List, Tuple, Optional

from US_Army_PLT_Composition_vTest import (
    MovementRoute,
    RouteWaypoint,
    MovementTechnique,
    US_IN_execute_route_movement,
    US_IN_UnitDesignator,
    US_IN_create_team,
    US_IN_create_squad
)

from WarGamingEnvironment_vTest import (
    MilitaryEnvironment,
    EnvironmentConfig,
    TerrainType,
    ElevationType,
    UnitType
)

from Excel_to_CSV_Map_Converter import excel_to_csv

from tactical_position_analyzer import (
    PositionPurpose,
    TacticalPosition,
    TacticalFilter
)

from TacticalRouteAnalyzer import (
    TacticalRouteAnalyzer,
    RouteStrategy,
    EnemyThreat,
    SquadElement,
    TacticalRoute
)


class TestConfiguration:
    """Test configuration with saved positions and scenario data."""

    def __init__(self):
        # Standard test scenario
        self.objective = (350, 50)
        self.squad_positions = {
            "Squad1": (50, 50),  # Support By Fire
            "Squad2": (50, 60),  # Assault
            "Squad3": (50, 40)  # Reserve
        }
        self.enemy_threats = [
            {
                'position': (346, 50),
                'observation_range': 48,  # 480m
                'engagement_range': 30,  # 300m
                'accuracy': 0.8
            },
            {
                'position': (340, 48),
                'observation_range': 40,  # 400m
                'engagement_range': 25,  # 250m
                'accuracy': 0.7
            },
            {
                'position': (342, 52),
                'observation_range': 35,  # 350m
                'engagement_range': 20,  # 200m
                'accuracy': 0.6
            }
        ]

        # Create project directories
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.validation_dir = os.path.join(self.project_dir, "Model_Validation_Route")

        # Create unit type directories
        self.unit_dirs = {
            UnitType.INFANTRY_TEAM: os.path.join(self.validation_dir, "route_analysis_team"),
            UnitType.WEAPONS_TEAM: os.path.join(self.validation_dir, "route_analysis_weapons_team"),
            UnitType.INFANTRY_SQUAD: os.path.join(self.validation_dir, "route_analysis_squad")
        }

        # Create directories
        self._create_directories()

        # Will store saved positions
        self.saved_positions = {}

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.validation_dir, exist_ok=True)
        for unit_dir in self.unit_dirs.values():
            os.makedirs(unit_dir, exist_ok=True)

    def get_unit_dir(self, unit_type: UnitType) -> str:
        """Get directory for specific unit type."""
        return self.unit_dirs[unit_type]

    def save_positions(self,
                       unit_type: UnitType,
                       purpose: PositionPurpose,
                       positions: List[TacticalPosition]) -> None:
        """Save tactical positions for specific unit type and purpose."""
        unit_dir = self.get_unit_dir(unit_type)
        filename = f"positions_{unit_type.name.lower()}_{purpose.name.lower()}.json"
        filepath = os.path.join(unit_dir, filename)

        # Convert positions to serializable format
        serializable_positions = [
            {
                'position': pos.position,
                'quality_score': pos.quality_score,
                'cover_score': pos.cover_score,
                'concealment_score': pos.concealment_score,
                'threat_exposure': pos.threat_exposure
            }
            for pos in positions
        ]

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_positions, f, indent=2)

        print(f"Saved {len(positions)} positions to {filepath}")

    def save_route(self,
                   unit_type: UnitType,
                   purpose: PositionPurpose,
                   position_index: int,
                   route: TacticalRoute) -> None:
        """Save individual route data."""
        unit_dir = self.get_unit_dir(unit_type)
        filename = f"route_{unit_type.name.lower()}_{purpose.name.lower()}_pos_{position_index}.json"
        filepath = os.path.join(unit_dir, filename)

        # Convert route to serializable format
        route_data = {
            'segments': [
                {
                    'start_pos': segment.start_pos,
                    'end_pos': segment.end_pos,
                    'path': segment.path,
                    'movement_technique': segment.movement_technique,
                    'threat_exposure': segment.threat_exposure,
                    'cover_score': segment.cover_score,
                    'concealment_score': segment.concealment_score
                }
                for segment in route.segments
            ],
            'total_distance': route.total_distance,
            'avg_threat_exposure': route.avg_threat_exposure,
            'avg_cover_score': route.avg_cover_score,
            'avg_concealment_score': route.avg_concealment_score,
            'quality_score': route.quality_score
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(route_data, f, indent=2)

        print(f"Saved route to {filepath}")

    def load_positions(self,
                       unit_type: UnitType,
                       purpose: PositionPurpose) -> Optional[List[Dict]]:
        """Load tactical positions for specific unit type and purpose."""
        unit_dir = self.get_unit_dir(unit_type)
        filename = f"positions_{unit_type.name.lower()}_{purpose.name.lower()}.json"
        filepath = os.path.join(unit_dir, filename)

        try:
            with open(filepath, 'r') as f:
                positions = json.load(f)
            print(f"Loaded {len(positions)} positions from {filepath}")
            return positions
        except FileNotFoundError:
            print(f"No saved positions found at {filepath}")
            return None


def setup_test_environment() -> Tuple[MilitaryEnvironment, TestConfiguration]:
    """Set up test environment and configuration."""
    print("\n=== Setting Up Test Environment ===")

    # Initialize configuration
    config = TestConfiguration()

    # Setup environment
    env_config = EnvironmentConfig(width=400, height=100, debug_level=1)
    env = MilitaryEnvironment(env_config)

    # Setup map files
    excel_file = os.path.join(config.project_dir, "map_design.xlsx")
    csv_file = os.path.join(config.validation_dir, "generated_map.csv")

    if not os.path.exists(csv_file):
        print("Converting map data...")
        excel_to_csv(excel_file, csv_file)

    # Initialize terrain
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)
    env.terrain_manager.load_from_csv(csv_file)

    # Check if positions exist and run analysis if needed
    any_positions_exist = False
    for unit_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM, UnitType.INFANTRY_SQUAD]:
        for purpose in [PositionPurpose.SUPPORT, PositionPurpose.ASSAULT, PositionPurpose.RESERVE]:
            if unit_type == UnitType.WEAPONS_TEAM and purpose == PositionPurpose.ASSAULT:
                continue
            if config.load_positions(unit_type, purpose) is not None:
                any_positions_exist = True
                break

    if not any_positions_exist:
        print("\nNo saved positions found - running tactical position analysis...")
        tactical_filter = TacticalFilter(env)

        # Find positions for each unit type and purpose
        for unit_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM, UnitType.INFANTRY_SQUAD]:
            for purpose in [PositionPurpose.SUPPORT, PositionPurpose.ASSAULT, PositionPurpose.RESERVE]:
                # Skip assault for weapons teams
                if unit_type == UnitType.WEAPONS_TEAM and purpose == PositionPurpose.ASSAULT:
                    continue

                print(f"\nFinding positions for {unit_type.name} {purpose.name}...")
                results = tactical_filter.find_positions(
                    objective=config.objective,
                    unit_type=unit_type,
                    position_purpose=purpose,
                    start_positions=[config.squad_positions[f"Squad{i + 1}"] for i in range(3)],
                    enemy_threats=config.enemy_threats,
                    max_positions=10
                )

                # Save positions
                config.save_positions(
                    unit_type=unit_type,
                    purpose=purpose,
                    positions=results['tactical_positions']
                )
                print(f"Saved {len(results['tactical_positions'])} positions")

        print("\nPosition analysis complete")

        # Reload all positions
        for unit_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM, UnitType.INFANTRY_SQUAD]:
            for purpose in [PositionPurpose.SUPPORT, PositionPurpose.ASSAULT, PositionPurpose.RESERVE]:
                if unit_type == UnitType.WEAPONS_TEAM and purpose == PositionPurpose.ASSAULT:
                    continue
                config.load_positions(unit_type, purpose)

    return env, config


def test_basic_route_planning(env: MilitaryEnvironment,
                              config: TestConfiguration,
                              visualize: bool = True) -> None:
    """Test basic route planning for Support By Fire positions."""
    print("\n=== Testing Basic Route Planning ===")

    # Initialize route analyzer
    analyzer = TacticalRouteAnalyzer(env, config.objective)

    # Add enemy threats
    print("\nSetting up enemy threats...")
    for threat_data in config.enemy_threats:
        threat = EnemyThreat(
            position=threat_data['position'],
            unit=None,
            observation_range=threat_data['observation_range'],
            engagement_range=threat_data['engagement_range'],
            suspected_accuracy=threat_data['accuracy']
        )
        analyzer.enemy_threats.append(threat)
        print(f"Added threat at {threat.position}")

    # Plan routes for Squad 1 (Support By Fire)
    print("\nPlanning Support By Fire routes...")

    # Look for Support positions for different unit types
    support_keys = [key for key in config.saved_positions.keys()
                    if 'SUPPORT' in key]

    all_routes = {}

    for position_key in support_keys:
        print(f"\nPlanning routes using {position_key} positions...")
        support_positions = config.saved_positions[position_key]

        routes = analyzer.find_tactical_routes(
            start_pos=config.squad_positions["Squad1"],
            tactical_positions=support_positions,
            strategy=RouteStrategy.BALANCED,  # COVER / CONCEALMENT
            squad_id=f"Squad1_{position_key}"
        )

        print(f"Found {len(routes)} possible routes for {position_key}")

        # Analyze route quality
        for route_id, route in routes.items():
            print(f"\nRoute {route_id}:")
            print(f"Total Distance: {route.total_distance:.0f}m")
            print(f"Average Threat Exposure: {route.avg_threat_exposure:.2f}")
            print(f"Average Cover Score: {route.avg_cover_score:.2f}")
            print(f"Quality Score: {route.quality_score:.2f}")

        # Add routes to collection
        all_routes.update(routes)

    # Visualize all routes if requested
    if visualize and all_routes:
        visualize_routes(
            env=env,
            routes=all_routes,
            config=config,
            title="Support By Fire Routes"
        )
    elif not all_routes:
        print("\nNo valid routes found to visualize")


def visualize_routes(env: MilitaryEnvironment,
                     routes: Dict[str, TacticalRoute],
                     config: TestConfiguration,
                     title: str = "Route Analysis") -> None:
    """Create visualization of planned routes."""
    print("\nCreating route visualization...")

    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot terrain base
    terrain_colors = {
        TerrainType.BARE: [0.9, 0.9, 0.9],
        TerrainType.SPARSE_VEG: [0.8, 0.9, 0.8],
        TerrainType.DENSE_VEG: [0.4, 0.7, 0.4],
        TerrainType.WOODS: [0.2, 0.5, 0.2],
        TerrainType.STRUCTURE: [0.6, 0.6, 0.7]
    }

    # Create terrain grid
    terrain_grid = np.zeros((env.height, env.width, 3))
    for y in range(env.height):
        for x in range(env.width):
            terrain_type = TerrainType(env.terrain_manager.state_tensor[y, x, 0])
            terrain_grid[y, x] = terrain_colors[terrain_type]

    # Plot terrain
    ax.imshow(terrain_grid, origin='lower')

    # Plot enemy positions and ranges
    for threat in config.enemy_threats:
        pos = threat['position']
        # Plot observation range
        obs_circle = Circle(
            pos,
            threat['observation_range'],
            fill=False,
            linestyle='--',
            color='red',
            alpha=0.3
        )
        ax.add_patch(obs_circle)

        # Plot engagement range
        eng_circle = Circle(
            pos,
            threat['engagement_range'],
            fill=False,
            color='red',
            alpha=0.5
        )
        ax.add_patch(eng_circle)

        # Plot position
        ax.plot(pos[0], pos[1], 'r^', markersize=10)

    # Plot objective
    ax.plot(config.objective[0], config.objective[1], 'r*',
            markersize=15, label='Objective')

    # Plot routes with different colors by unit type
    unit_colors = {
        'INFANTRY_TEAM': 'blue',
        'WEAPONS_TEAM': 'green',
        'SQUAD': 'yellow'
    }

    for route_id, route in routes.items():
        # Determine unit type from route_id
        for unit_type in unit_colors.keys():
            if unit_type in route_id:
                color = unit_colors[unit_type]
                break
        else:
            color = 'cyan'  # Default color if no match

        # Plot each segment
        for segment in route.segments:
            # Get path points
            x_coords = [p[0] for p in segment.path]
            y_coords = [p[1] for p in segment.path]

            # Plot route
            ax.plot(x_coords, y_coords,
                    color=color,
                    linestyle='--' if segment.movement_technique == 'bounding' else '-',
                    alpha=0.8,
                    linewidth=2,
                    label=f'Route {route_id}' if segment == route.segments[0] else "")

            # Add quality score label at end point
            ax.annotate(
                f"{route.quality_score:.2f}",
                (x_coords[-1], y_coords[-1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='center left', bbox_to_anchor=(1, 0.5))

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)

    # Save visualization
    output_path = os.path.join(config.validation_dir, 'route_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Route visualization saved as {output_path}")


def test_unit_route_planning(env: MilitaryEnvironment,
                             config: TestConfiguration,
                             unit_type: UnitType,
                             purpose: PositionPurpose,
                             squad_id: str,
                             start_pos: Tuple[int, int]) -> None:
    """Test route planning for specific unit type and purpose."""
    print(f"\n=== Testing {unit_type.name} {purpose.name} Route Planning ===")

    # Load saved positions
    positions = config.load_positions(unit_type, purpose)
    if not positions:
        print("No saved positions found - skipping route planning")
        return

    # Initialize route analyzer
    analyzer = TacticalRouteAnalyzer(
        env=env,
        objective=config.objective
    )

    # Add enemy threats
    for threat_data in config.enemy_threats:
        threat = EnemyThreat(
            position=threat_data['position'],
            unit=None,
            observation_range=threat_data['observation_range'],
            engagement_range=threat_data['engagement_range'],
            suspected_accuracy=threat_data['accuracy']
        )
        analyzer.enemy_threats.append(threat)

    # Determine route strategy based on purpose
    if purpose == PositionPurpose.ASSAULT:
        strategy = RouteStrategy.CONCEALMENT
    elif purpose == PositionPurpose.SUPPORT:
        strategy = RouteStrategy.COVER
    else:  # RESERVE
        strategy = RouteStrategy.BALANCED

    # Plan routes
    routes = analyzer.find_tactical_routes(
        start_pos=start_pos,
        tactical_positions=positions,
        strategy=strategy,
        squad_id=squad_id
    )

    print(f"Found {len(routes)} possible routes")

    # Save individual routes
    for i, (route_id, route) in enumerate(routes.items()):
        config.save_route(unit_type, purpose, i, route)

        print(f"\nRoute {i} Quality Metrics:")
        print(f"Total Distance: {route.total_distance:.0f}m")
        print(f"Average Threat Exposure: {route.avg_threat_exposure:.2f}")
        print(f"Average Cover Score: {route.avg_cover_score:.2f}")
        print(f"Quality Score: {route.quality_score:.2f}")

    # Create visualization
    create_route_visualization(env, routes, config, unit_type, purpose)


def create_route_visualization(env: MilitaryEnvironment,
                               routes: Dict[str, TacticalRoute],
                               config: TestConfiguration,
                               unit_type: UnitType,
                               purpose: PositionPurpose) -> None:
    """Create visualization of routes for specific unit type and purpose."""
    print(f"\nCreating visualization for {unit_type.name} {purpose.name} routes...")

    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot terrain base
    terrain_colors = {
        TerrainType.BARE: [0.9, 0.9, 0.9],
        TerrainType.SPARSE_VEG: [0.8, 0.9, 0.8],
        TerrainType.DENSE_VEG: [0.4, 0.7, 0.4],
        TerrainType.WOODS: [0.2, 0.5, 0.2],
        TerrainType.STRUCTURE: [0.6, 0.6, 0.7]
    }

    # Create terrain grid
    terrain_grid = np.zeros((env.height, env.width, 3))
    for y in range(env.height):
        for x in range(env.width):
            terrain_type = TerrainType(env.terrain_manager.state_tensor[y, x, 0])
            terrain_grid[y, x] = terrain_colors[terrain_type]

    # Plot terrain
    ax.imshow(terrain_grid, origin='lower')

    # Plot enemy positions and ranges
    for threat in config.enemy_threats:
        pos = threat['position']
        # Plot observation range
        obs_circle = Circle(
            pos,
            threat['observation_range'],
            fill=False,
            linestyle='--',
            color='red',
            alpha=0.3
        )
        ax.add_patch(obs_circle)

        # Plot engagement range
        eng_circle = Circle(
            pos,
            threat['engagement_range'],
            fill=False,
            color='red',
            alpha=0.5
        )
        ax.add_patch(eng_circle)

        # Plot position
        ax.plot(pos[0], pos[1], 'r^', markersize=10)

    # Plot objective
    ax.plot(config.objective[0], config.objective[1], 'r*',
            markersize=15, label='Objective')

    # Plot routes
    colors = ['b', 'g', 'y', 'c', 'm']  # Different colors for routes
    for i, (route_id, route) in enumerate(routes.items()):
        color = colors[i % len(colors)]

        # Plot each segment
        for segment in route.segments:
            # Get path points
            x_coords = [p[0] for p in segment.path]
            y_coords = [p[1] for p in segment.path]

            # Plot route
            ax.plot(x_coords, y_coords,
                    color=color,
                    linestyle='--' if segment.movement_technique == 'bounding' else '-',
                    alpha=0.8,
                    linewidth=2,
                    label=f'Route {route_id}')

            # Add quality score label at end point
            ax.annotate(
                f"{route.quality_score:.2f}",
                (x_coords[-1], y_coords[-1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set title and labels
    ax.set_title(f"{unit_type.name} {purpose.name} Routes")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)

    # Save visualization
    filename = f"route_analysis_{unit_type.name.lower()}_{purpose.name.lower()}.png"
    filepath = os.path.join(config.get_unit_dir(unit_type), filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Route visualization saved as {filepath}")


def test_execute_saved_route(env: MilitaryEnvironment,
                             config: TestConfiguration,
                             unit_type: UnitType,
                             purpose: PositionPurpose,
                             pos_index: int = 0):
    """Test execution of a saved route using US Army movement functions."""
    print(f"\n=== Testing Route Execution for {unit_type.name} {purpose.name} Position {pos_index} ===")

    # Load route file
    unit_dir = config.get_unit_dir(unit_type)
    route_file = f"route_{unit_type.name.lower()}_{purpose.name.lower()}_pos_{pos_index}.json"
    route_path = os.path.join(unit_dir, route_file)

    try:
        with open(route_path, 'r') as f:
            route_data = json.load(f)
            print(f"Loaded route from {route_path}")
            print("\nRoute characteristics:")
            print(f"Total distance: {route_data['total_distance']:.1f}")
            print(f"Average threat exposure: {route_data['avg_threat_exposure']:.2f}")
            print(f"Average cover score: {route_data['avg_cover_score']:.2f}")
    except FileNotFoundError:
        print(f"No route file found at {route_path}")
        return

    # Create unit at start position
    start_pos = route_data['segments'][0]['start_pos']
    if unit_type == UnitType.INFANTRY_TEAM:
        unit_id = US_IN_create_team(
            env=env,
            plt_num=1,
            squad_num=1,
            designator=US_IN_UnitDesignator.ALPHA_TEAM,
            start_position=start_pos
        )
    elif unit_type == UnitType.WEAPONS_TEAM:
        unit_id = US_IN_create_team(
            env=env,
            plt_num=1,
            squad_num=1,
            designator=US_IN_UnitDesignator.GUN_TEAM_1,
            start_position=start_pos
        )
    else:  # INFANTRY_SQUAD
        unit_id = US_IN_create_squad(
            env=env,
            plt_num=1,
            squad_num=1,
            start_position=start_pos
        )

    # Determine movement technique based on threat exposure
    avg_threat = route_data['avg_threat_exposure']
    technique = MovementTechnique.BOUNDING if avg_threat > 0.5 else MovementTechnique.TRAVELING

    # Determine formation based on unit type and purpose
    if unit_type == UnitType.INFANTRY_TEAM:
        formation = "team_wedge_right" if purpose == PositionPurpose.ASSAULT else "team_line_right"
    elif unit_type == UnitType.WEAPONS_TEAM:
        formation = "gun_team" if "GUN" in env.get_unit_property(unit_id, 'string_id') else "javelin_team"
    else:
        formation = "squad_column_team_wedge"

    # Convert route data to waypoints
    waypoints = []
    for segment in route_data['segments']:
        waypoint = RouteWaypoint(
            position=segment['end_pos'],
            formation=formation,
            hold_time=2 if segment['threat_exposure'] > 0.6 else 0
        )
        waypoints.append(waypoint)

    # Create movement route
    movement_route = MovementRoute(
        waypoints=waypoints,
        technique=technique
    )

    # Execute route movement
    print(f"\nExecuting route movement using {technique.name} technique...")
    print(f"Formation: {formation}")
    frames = US_IN_execute_route_movement(
        env=env,
        unit_id=unit_id,
        route=movement_route,
        debug_level=1
    )

    print(f"\nMovement complete - generated {len(frames)} frames")

    # Create animation
    create_movement_animation(frames, env, config, unit_type,
                              f"{unit_type.name}_{purpose.name}_Movement")

    return frames


def create_movement_animation(frames: List[Dict],
                              env: MilitaryEnvironment,
                              config: TestConfiguration,
                              unit_type: UnitType,
                              title: str):
    """Create animation of unit movement."""
    print("\nCreating movement animation...")

    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot terrain base
    terrain_colors = {
        TerrainType.BARE: [0.9, 0.9, 0.9],
        TerrainType.SPARSE_VEG: [0.8, 0.9, 0.8],
        TerrainType.DENSE_VEG: [0.4, 0.7, 0.4],
        TerrainType.WOODS: [0.2, 0.5, 0.2],
        TerrainType.STRUCTURE: [0.6, 0.6, 0.7]
    }

    # Create terrain grid
    terrain_grid = np.zeros((env.height, env.width, 3))
    for y in range(env.height):
        for x in range(env.width):
            terrain_type = TerrainType(env.terrain_manager.state_tensor[y, x, 0])
            terrain_grid[y, x] = terrain_colors[terrain_type]

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()

        # Plot terrain
        ax.imshow(terrain_grid, origin='lower')

        # Plot enemy positions and ranges
        for threat in config.enemy_threats:
            pos = threat['position']
            # Plot observation range
            obs_circle = Circle(
                pos,
                threat['observation_range'],
                fill=False,
                linestyle='--',
                color='red',
                alpha=0.3
            )
            ax.add_patch(obs_circle)

            # Plot engagement range
            eng_circle = Circle(
                pos,
                threat['engagement_range'],
                fill=False,
                color='red',
                alpha=0.5
            )
            ax.add_patch(eng_circle)

            # Plot position
            ax.plot(pos[0], pos[1], 'r^', markersize=10)

        # Plot objective
        ax.plot(config.objective[0], config.objective[1], 'r*',
                markersize=15, label='Objective')

        # Plot unit positions
        for position in frame['positions']:
            pos = position['position']
            is_leader = position['is_leader']

            # Different marker for leader
            marker = 'o' if is_leader else '^'
            color = 'red' if is_leader else 'blue'
            size = 100 if is_leader else 80

            ax.scatter(pos[0], pos[1], c=color, marker=marker, s=size)

        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_title(f"{title}\nFrame {frames.index(frame) + 1}/{len(frames)}")
        ax.grid(True)

        return []

    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=100,
                                   blit=True)

    # Save animation
    writer = animation.PillowWriter(fps=5)
    filename = f"movement_animation_{title}.gif"
    filepath = os.path.join(config.get_unit_dir(unit_type), filename)
    anim.save(filepath, writer=writer)
    plt.close()

    print(f"Animation saved as {filename}")


if __name__ == "__main__":
    # Setup environment
    env, config = setup_test_environment()

    # Run basic test suite
    # test_basic_route_planning(env, config)

    # Test each unit type and purpose
    unit_configs = [
        (UnitType.INFANTRY_TEAM, "Squad1"),
        (UnitType.WEAPONS_TEAM, "Squad1"),
        (UnitType.INFANTRY_SQUAD, "Squad1")
    ]

    purposes = [
        PositionPurpose.SUPPORT,
        PositionPurpose.ASSAULT,
        PositionPurpose.RESERVE
    ]

    for unit_type, squad_id in unit_configs:
        for purpose in purposes:
            # Skip assault for weapons teams
            if unit_type == UnitType.WEAPONS_TEAM and purpose == PositionPurpose.ASSAULT:
                continue

            test_unit_route_planning(
                env=env,
                config=config,
                unit_type=unit_type,
                purpose=purpose,
                squad_id=squad_id,
                start_pos=config.squad_positions[squad_id]
            )

    # # Test execution of a specific route
    # test_frames = test_execute_saved_route(
    #     env=env,
    #     config=config,
    #     unit_type=UnitType.INFANTRY_SQUAD,
    #     purpose=PositionPurpose.SUPPORT,
    #     pos_index=0  # First route
    # )
