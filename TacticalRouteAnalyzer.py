"""
Tactical Route Analyzer

Initial implementation focused on route planning for Support By Fire positions
using A* pathfinding with tactical considerations.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Set
import math
import heapq
import numpy as np


from WarGamingEnvironment_vTest import (
    TerrainType,
    ElevationType,
    VisibilityManager, MilitaryEnvironment
)


class RouteStrategy(Enum):
    """Different strategies for route planning."""
    CONCEALMENT = auto()  # Prioritize staying hidden
    COVER = auto()  # Prioritize protection from fire
    BALANCED = auto()  # Balance between concealment and cover


@dataclass
class EnemyThreat:
    """Represents an enemy unit and its threat characteristics."""
    position: Tuple[int, int]
    unit: Optional[object]  # For future integration with specific unit types
    observation_range: int  # In cells (10m per cell)
    engagement_range: int  # In cells
    suspected_accuracy: float  # 0.0 to 1.0


@dataclass
class RouteSegment:
    """Segment of a tactical route."""
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    path: List[Tuple[int, int]]  # List of positions forming the path
    movement_technique: str  # 'traveling' or 'bounding'
    threat_exposure: float  # 0.0 to 1.0
    cover_score: float  # 0.0 to 1.0
    concealment_score: float  # 0.0 to 1.0


@dataclass
class TacticalRoute:
    """Complete tactical route with multiple segments."""
    segments: List[RouteSegment]
    total_distance: float
    avg_threat_exposure: float
    avg_cover_score: float
    avg_concealment_score: float
    quality_score: float


class SquadElement(Enum):
    """Types of squad elements for route planning."""
    SUPPORT = auto()  # Support by fire
    ASSAULT = auto()  # Assault element
    RESERVE = auto()  # Reserve element


class TacticalRouteAnalyzer:
    """
    Plans tactical routes considering terrain, threats, and unit roles.
    Initial implementation focused on Support By Fire positions.
    """

    def __init__(self, env: MilitaryEnvironment, objective: Tuple[int, int]):
        """Initialize analyzer with environment reference and objective."""
        self.env = env
        self.width = env.width
        self.height = env.height
        self.objective = objective
        self.enemy_threats: List[EnemyThreat] = []

        # Get terrain data from environment
        self.terrain = env.terrain_manager.get_terrain_data() if hasattr(env.terrain_manager, 'get_terrain_data') \
            else env.terrain_manager.state_tensor[:, :, 0:2]

        # Movement costs by terrain type
        self.terrain_costs = {
            TerrainType.BARE.value: 1.0,
            TerrainType.SPARSE_VEG.value: 1.2,
            TerrainType.DENSE_VEG.value: 1.5,
            TerrainType.WOODS.value: 2.0,
            TerrainType.STRUCTURE.value: float('inf')  # Impassable
        }

    def find_tactical_routes(self,
                             start_pos: Tuple[int, int],
                             tactical_positions: List[Dict],
                             strategy: RouteStrategy = RouteStrategy.BALANCED,
                             squad_id: str = "") -> Dict[str, TacticalRoute]:
        """
        Find tactical routes from start position to tactical positions.

        Args:
            start_pos: Starting position (x, y)
            tactical_positions: List of possible tactical positions to route to
            strategy: Route planning strategy to use
            squad_id: Optional squad identifier for logging

        Returns:
            Dictionary mapping position IDs to TacticalRoute objects
        """
        routes = {}

        for pos_idx, pos_data in enumerate(tactical_positions):
            # Extract position from saved data format
            if isinstance(pos_data, dict) and 'position' in pos_data:
                pos = pos_data['position']
                # Ensure we have a tuple of integers
                end_pos = (int(pos[0]), int(pos[1]))
            else:
                print(f"Warning: Invalid position data format at index {pos_idx}")
                continue

            route = self._plan_route(
                start_pos=start_pos,
                end_pos=end_pos,
                strategy=strategy
            )

            if route:
                pos_id = f"{squad_id}_pos_{pos_idx}"
                routes[pos_id] = route

        return routes

    def _plan_route(self,
                    start_pos: Tuple[int, int],
                    end_pos: Tuple[int, int],
                    strategy: RouteStrategy) -> Optional[TacticalRoute]:
        """
        Plan a single tactical route using A* pathfinding.

        Args:
            start_pos: Starting position
            end_pos: Ending position
            strategy: Route planning strategy

        Returns:
            TacticalRoute if path found, None otherwise
        """
        # A* initialization
        frontier = []
        heapq.heappush(frontier, (0, start_pos))
        came_from = {start_pos: None}
        cost_so_far = {start_pos: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == end_pos:
                # Reconstruct path
                path = self._reconstruct_path(came_from, start_pos, end_pos)

                # Create route segment
                segment = RouteSegment(
                    start_pos=start_pos,
                    end_pos=end_pos,
                    path=path,
                    movement_technique='traveling',  # Default to traveling
                    threat_exposure=self._calculate_threat_exposure(path),
                    cover_score=self._calculate_cover_score(path),
                    concealment_score=self._calculate_concealment_score(path)
                )

                # Create complete route
                route = TacticalRoute(
                    segments=[segment],
                    total_distance=len(path),
                    avg_threat_exposure=segment.threat_exposure,
                    avg_cover_score=segment.cover_score,
                    avg_concealment_score=segment.concealment_score,
                    quality_score=self._calculate_route_quality(
                        segment, strategy)
                )

                return route

            # Check neighbors
            for next_pos in self._get_neighbors(current):
                # Calculate new cost
                new_cost = cost_so_far[current] + self._movement_cost(
                    current, next_pos, strategy)

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(next_pos, end_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        return None  # No path found

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = []

        # Check 8 adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                new_x = x + dx
                new_y = y + dy

                # Check bounds
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    # Get terrain type from first channel
                    terrain_type = int(self.terrain[new_y][new_x][0])

                    # Check if terrain is passable
                    if self.terrain_costs[terrain_type] < float('inf'):
                        neighbors.append((new_x, new_y))

        return neighbors

    def _movement_cost(self,
                       current: Tuple[int, int],
                       next_pos: Tuple[int, int],
                       strategy: RouteStrategy) -> float:
        """
        Calculate movement cost between positions based on strategy.

        Returns:
            Combined cost value (higher = more costly movement)
        """
        # Get base terrain cost
        x, y = next_pos
        terrain_type = int(self.terrain[y][x][0])  # Get terrain type from first channel
        terrain_cost = self.terrain_costs[terrain_type]

        # Calculate threat exposure
        threat_cost = self._calculate_threat_exposure([next_pos])

        # Calculate concealment
        concealment = self._calculate_concealment_score([next_pos])
        concealment_cost = 1.0 - concealment

        # Calculate cover
        cover = self._calculate_cover_score([next_pos])
        cover_cost = 1.0 - cover

        # Combine costs based on strategy
        if strategy == RouteStrategy.CONCEALMENT:
            return terrain_cost * (1.0 + concealment_cost * 2.0 + threat_cost)
        elif strategy == RouteStrategy.COVER:
            return terrain_cost * (1.0 + cover_cost * 2.0 + threat_cost)
        else:  # BALANCED
            return terrain_cost * (1.0 + concealment_cost + cover_cost + threat_cost)

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate heuristic distance to goal."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def _reconstruct_path(self,
                          came_from: Dict[Tuple[int, int], Tuple[int, int]],
                          start: Tuple[int, int],
                          goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from A* came_from dictionary."""
        current = goal
        path = []

        while current != start:
            path.append(current)
            current = came_from[current]

        path.append(start)
        path.reverse()
        return path

    def _calculate_threat_exposure(self, path: List[Tuple[int, int]]) -> float:
        """Calculate average threat exposure along path."""
        if not path or not self.enemy_threats:
            return 0.0

        total_exposure = 0.0

        for pos in path:
            pos_exposure = 0.0
            for threat in self.enemy_threats:
                # Calculate distance to threat
                dist = math.sqrt(
                    (pos[0] - threat.position[0]) ** 2 +
                    (pos[1] - threat.position[1]) ** 2
                )

                # Check if in observation range
                if dist <= threat.observation_range:
                    obs_exposure = (threat.observation_range - dist) / threat.observation_range

                    # Higher exposure if in engagement range
                    if dist <= threat.engagement_range:
                        eng_exposure = (threat.engagement_range - dist) / threat.engagement_range
                        pos_exposure = max(pos_exposure,
                                           obs_exposure * 0.5 + eng_exposure * 0.5)
                    else:
                        pos_exposure = max(pos_exposure, obs_exposure * 0.3)

            total_exposure += pos_exposure

        return total_exposure / len(path)

    def _calculate_cover_score(self, path: List[Tuple[int, int]]) -> float:
        """Calculate average cover score along path."""
        if not path:
            return 0.0

        total_cover = 0.0

        for pos in path:
            x, y = pos
            terrain_type = int(self.terrain[y][x][0])  # Get terrain type from first channel

            # Assign cover values based on terrain
            if terrain_type == TerrainType.STRUCTURE.value:
                cover = 1.0
            elif terrain_type == TerrainType.WOODS.value:
                cover = 0.8
            elif terrain_type == TerrainType.DENSE_VEG.value:
                cover = 0.6
            elif terrain_type == TerrainType.SPARSE_VEG.value:
                cover = 0.3
            else:  # BARE
                cover = 0.0

            total_cover += cover

        return total_cover / len(path)

    def _calculate_concealment_score(self, path: List[Tuple[int, int]]) -> float:
        """Calculate average concealment score along path."""
        if not path:
            return 0.0

        total_concealment = 0.0

        for pos in path:
            x, y = pos
            terrain_type = int(self.terrain[y][x][0])  # Get terrain type from first channel

            # Assign concealment values based on terrain
            if terrain_type == TerrainType.WOODS.value:
                concealment = 1.0
            elif terrain_type == TerrainType.DENSE_VEG.value:
                concealment = 0.8
            elif terrain_type == TerrainType.SPARSE_VEG.value:
                concealment = 0.5
            elif terrain_type == TerrainType.STRUCTURE.value:
                concealment = 0.3  # Less concealment than woods/vegetation
            else:  # BARE
                concealment = 0.0

            total_concealment += concealment

        return total_concealment / len(path)

    def _calculate_route_quality(self,
                                 segment: RouteSegment,
                                 strategy: RouteStrategy) -> float:
        """Calculate overall route quality score based on strategy."""
        if strategy == RouteStrategy.CONCEALMENT:
            return (
                    segment.concealment_score * 0.5 +
                    (1.0 - segment.threat_exposure) * 0.3 +
                    segment.cover_score * 0.2
            )
        elif strategy == RouteStrategy.COVER:
            return (
                    segment.cover_score * 0.5 +
                    (1.0 - segment.threat_exposure) * 0.3 +
                    segment.concealment_score * 0.2
            )
        else:  # BALANCED
            return (
                    (1.0 - segment.threat_exposure) * 0.4 +
                    segment.cover_score * 0.3 +
                    segment.concealment_score * 0.3
            )
