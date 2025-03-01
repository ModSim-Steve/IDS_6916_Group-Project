"""
---------------------------------------------------------
---------------------------------------------------------
           PLT-OPS-WarGaming-Tool Environment
---------------------------------------------------------
---------------------------------------------------------

This file defines the core environment for a military platoon operations wargaming simulation.
The environment serves as the source of truth for all unit states and handles all state changes.

Features:
- Scalable grid-based environment (default 20x20, expandable to 400x100)
- Terrain and elevation tracking per cell
- Single source of truth for unit states
- Support for unit hierarchy and formations
- Integrated order system (future)

---------------------------------------------------------
Coordinate System Documentation for WarGaming Environment
---------------------------------------------------------

This file documents the coordinate system convention used throughout the WarGaming Environment.

STANDARD COORDINATE SYSTEM
-------------------------
- Origin: Top-left corner (0,0)
- X-axis: Increases to the right
- Y-axis: Increases downward
- Angle orientation: Measured clockwise from East

ANGLE CONVENTIONS
-------------------------
- 0°   = East (right)
- 90°  = South (down)
- 180° = West (left)
- 270° = North (up)

This convention aligns with Pygame's coordinate system and is used consistently
throughout the codebase for positioning, orientation, sector calculations, and visualization.

USAGE IN KEY COMPONENTS
-------------------------
1. Unit Orientation
   - When a unit has orientation = 0°, it faces East
   - Orientation increases clockwise (e.g., 90° = facing South)

2. Sectors of Fire
   - Defined as angular ranges using the standard system
   - Example: A sector from 315° to 45° covers the Northeast quadrant

3. Movement
   - Direction vectors use the same coordinate system
   - Example: Direction (1,0) = East, (0,1) = South

4. Target Validation
   - Target angles are calculated using this system
   - A target is in a unit's sector if its angle falls within the sector's range

5. Visualization
   - All visualizations (Matplotlib, Pygame) use this coordinate system
   - Sector wedges are drawn with the same angle convention

IMPLEMENTATION NOTES
-------------------------
- Orientation values are stored in degrees (0-359)
- All angle calculations normalize results to the 0-359 range
- When applying rotations, we add the unit's orientation to the base sector angles
"""
import csv
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any
from enum import Enum, auto
from datetime import datetime


class TerrainType(Enum):
    """Types of terrain in the environment"""
    BARE = 0
    SPARSE_VEG = 1
    DENSE_VEG = 2
    WOODS = 3
    STRUCTURE = 4


class ElevationType(Enum):
    """Elevation levels in the environment"""
    GROUND_LEVEL = 0
    ELEVATED_LEVEL = 1
    LOWER_LEVEL = 2


class ForceType(Enum):
    """Types of forces in the environment"""
    FRIENDLY = "FRIENDLY"
    ENEMY = "ENEMY"


class UnitType(Enum):
    """Types of units that can be created"""
    INFANTRY_TEAM = 0
    INFANTRY_SQUAD = 1
    INFANTRY_PLATOON = 2
    WEAPONS_TEAM = 3  # For both gun and javelin teams


class WeaponsStatus(Enum):
    """Weapons control status"""
    HOLD = 0
    TIGHT = 1
    FREE = 2


class BaseWeapon:
    """Base class for weapon definitions"""

    def __init__(self, name: str, max_range: int, ammo_capacity: int,
                 fire_rate: int, damage: int, is_area_weapon: bool = False):
        self.name = name
        self.max_range = max_range
        self.ammo_capacity = ammo_capacity
        self.fire_rate = fire_rate
        self.damage = damage
        self.is_area_weapon = is_area_weapon


class EngagementType(Enum):
    """Types of fire engagement."""
    POINT = auto()  # Direct fire at specific target
    AREA = auto()   # Area fire/suppression


@dataclass
class FireControl:
    """Fire control measures for engagement."""
    target_area: Tuple[int, int]  # Center of target area
    area_radius: int = 0  # Radius for area fire (0 for point)
    max_rounds: int = 10  # Maximum rounds to expend
    time_limit: int = 5  # Time limit in steps
    suppress_only: bool = False  # True for suppression only
    priority_targets: List[int] = field(default_factory=list)  # Priority target IDs
    sustained: bool = False  # Whether this is sustained fire
    adjust_for_fire_rate: bool = False  # NEW: Flag to adjust rounds based on weapon fire rate


@dataclass
class EngagementResults:
    """Results of an engagement."""
    hits: int = 0  # Number of hits achieved
    targets_hit: List[int] = field(default_factory=list)  # IDs of hit targets
    damage_dealt: float = 0.0  # Total damage dealt
    suppression_effect: float = 0.0  # Suppression effect (0-1)
    rounds_expended: int = 0  # Ammunition used
    time_steps: int = 0  # Steps taken
    los_quality: float = 0.0  # Average line of sight quality
    hit_locations: List[Tuple[int, int]] = field(default_factory=list)  # Where rounds landed


@dataclass
class SectorOfFire:
    """Defines a sector of fire for a unit position."""
    start_angle: float  # Starting angle in degrees (0-360)
    end_angle: float  # Ending angle in degrees (0-360)
    is_primary: bool  # True if primary sector, False if secondary

    def contains_angle(self, angle: float, unit_orientation: float = 0) -> bool:
        """
        Check if an angle falls within this sector, accounting for unit orientation.
        Uses standard coordinate system (0° = East).

        Args:
            angle: Angle to check (in degrees)
            unit_orientation: Current orientation of unit (in degrees)

        Returns:
            True if angle is within rotated sector
        """
        # Normalize angles to 0-360
        angle = angle % 360
        unit_orientation = unit_orientation % 360

        # Rotate sector bounds by unit orientation
        rotated_start = (self.start_angle + unit_orientation) % 360
        rotated_end = (self.end_angle + unit_orientation) % 360

        # Handle sector wrap-around
        if rotated_start <= rotated_end:
            return rotated_start <= angle <= rotated_end
        else:
            return angle >= rotated_start or angle <= rotated_end

    def get_center(self, unit_orientation: float = 0) -> float:
        """
        Get center angle of sector accounting for unit orientation.
        Uses standard coordinate system (0° = East).
        """
        # Calculate center of unrotated sector
        if self.start_angle <= self.end_angle:
            center = (self.start_angle + self.end_angle) / 2
        else:
            # Handle wrap-around by adding 360 to end angle
            center = (self.start_angle + (self.end_angle + 360)) / 2
            center = center % 360  # Normalize to 0-360

        # Apply rotation and normalize
        return (center + unit_orientation) % 360


class CombatManager:
    """
    Combat Manager for Military Environment
    """

    def __init__(self, env):
        """
        Initialize with environment reference.

        Args:
            env: Reference to MilitaryEnvironment
        """
        self.env = env
        self.visibility_mgr = env.visibility_manager
        self.terrain_mgr = env.terrain_manager
        self.state_mgr = env.state_manager

        # Track engagement history
        self.engagement_history: Dict[int, List[Dict]] = {}

        # Track suppressed units and duration
        self.suppressed_units: Dict[int, Dict] = {}

        # Track ammunition by unit
        self.ammo_tracking: Dict[int, Dict] = {}

        # Suppression recovery rate (per step)
        self.suppression_recovery_rate = 0.05

    def calculate_target_angle(self, unit_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> float:
        """
        Calculate angle to target using standard coordinate system (0° = East).

        Coordinate System Standard:
        - 0° = East (right)
        - 90° = South (down)
        - 180° = West (left)
        - 270° = North (up)

        This matches Pygame's coordinate system where (0,0) is top-left.

        Args:
            unit_pos: (x, y) position of unit
            target_pos: (x, y) position of target

        Returns:
            Angle in degrees (0-359) where 0° is East, 90° is South, etc.
        """
        dx = target_pos[0] - unit_pos[0]
        dy = target_pos[1] - unit_pos[1]

        # Calculate angle in radians (-π to π)
        angle = math.atan2(dy, dx)

        # Convert to degrees and normalize to 0-360 range
        degrees = math.degrees(angle)
        return (degrees + 360) % 360  # Normalize to 0-360 range

    def is_target_in_sector(self, observer_pos: Tuple[int, int],
                            unit_orientation: float,
                            target_pos: Tuple[int, int],
                            sector: SectorOfFire) -> bool:
        """
        Check if target is within a sector of fire.
        Uses standard coordinate system (0° = East).

        Args:
            observer_pos: Position of firing unit
            unit_orientation: Unit's current orientation in degrees
            target_pos: Position of target
            sector: SectorOfFire to check against

        Returns:
            True if target is in sector
        """
        # Calculate angle to target
        target_angle = self.calculate_target_angle(observer_pos, target_pos)

        # Check if angle falls within sector
        return sector.contains_angle(target_angle, unit_orientation)

    def execute_engagement(self, unit_id: int, fire_control: FireControl) -> EngagementResults:
        """
        Execute engagement considering unit hierarchy. For teams/squads, this means
        executing engagements through their members.

        Args:
            unit_id: ID of firing unit (can be team/squad or individual)
            fire_control: Fire control parameters

        Returns:
            EngagementResults with combined effects and resource usage
        """
        # First check if any valid targets exist
        valid_targets = self._get_valid_targets(unit_id, fire_control.target_area, fire_control.area_radius)

        if self.env.debug_level > 0:
            print(f"\nDEBUG: Found {len(valid_targets)} valid targets for unit {unit_id}")

        # Initialize combined results
        combined_results = EngagementResults()

        # Get unit type
        unit_type = self.env.get_unit_property(unit_id, 'type')

        if unit_type in [UnitType.INFANTRY_TEAM, UnitType.INFANTRY_SQUAD, UnitType.WEAPONS_TEAM]:
            if self.env.debug_level > 0:
                print(f"DEBUG: Executing team/squad engagement for unit {unit_id}")

            # Get all members
            members = self.env.get_unit_children(unit_id)
            engaging_members = []

            # Find members who can engage
            for member_id in members:
                if self.validate_target(member_id, fire_control.target_area):
                    engaging_members.append(member_id)

            if self.env.debug_level > 0:
                print(f"DEBUG: {len(engaging_members)} members can engage target")

            # Execute engagement for each capable member
            for member_id in engaging_members:
                # Create member-specific fire control with valid targets
                member_fire_control = FireControl(
                    target_area=fire_control.target_area,
                    area_radius=fire_control.area_radius,
                    max_rounds=fire_control.max_rounds,
                    time_limit=fire_control.time_limit,
                    suppress_only=fire_control.suppress_only,
                    priority_targets=valid_targets,  # Use validated targets
                    sustained=fire_control.sustained
                )

                # Execute individual engagement
                results = self._execute_member_engagement(member_id, member_fire_control)

                # Combine results
                combined_results.hits += results.hits
                combined_results.damage_dealt += results.damage_dealt
                combined_results.suppression_effect = max(
                    combined_results.suppression_effect,
                    results.suppression_effect
                )
                combined_results.rounds_expended += results.rounds_expended
                combined_results.time_steps = max(
                    combined_results.time_steps,
                    results.time_steps
                )
                combined_results.los_quality = max(
                    combined_results.los_quality,
                    results.los_quality
                )
                combined_results.targets_hit.extend(
                    [t for t in results.targets_hit
                     if t not in combined_results.targets_hit]
                )
                combined_results.hit_locations.extend(results.hit_locations)

        else:
            # For individual soldiers, execute directly
            # Still use validated targets
            fire_control.priority_targets = valid_targets
            combined_results = self._execute_member_engagement(unit_id, fire_control)

        return combined_results

    def _execute_member_engagement(self, member_id: int, fire_control: FireControl) -> EngagementResults:
        """
        Execute engagement for an individual soldier with improved round distribution.

        Args:
            member_id: ID of soldier executing engagement
            fire_control: Fire control parameters

        Returns:
            EngagementResults for this member's engagement
        """
        results = EngagementResults()

        # Skip if member can't engage
        if not self.validate_target(member_id, fire_control.target_area):
            if self.env.debug_level > 0:
                print(f"DEBUG: Member {member_id} cannot engage target")
            return results

        # Get member's weapon and position
        weapon = self._get_unit_weapon(member_id)
        member_pos = self.env.get_unit_position(member_id)

        # Check line of sight
        los_result = self.visibility_mgr.check_line_of_sight(
            member_pos, fire_control.target_area)
        if not los_result['has_los']:
            if self.env.debug_level > 0:
                print(f"DEBUG: Member {member_id} has no line of sight")
            return results

        results.los_quality = los_result['los_quality']

        # Calculate distance
        distance = math.sqrt(
            (fire_control.target_area[0] - member_pos[0]) ** 2 +
            (fire_control.target_area[1] - member_pos[1]) ** 2
        )

        # Use weapon's fire rate directly
        fire_rate = max(1, int(weapon.fire_rate))

        # Adjust max_rounds based on fire rate if the flag is set
        total_bursts = fire_control.max_rounds
        if hasattr(fire_control, 'adjust_for_fire_rate') and fire_control.adjust_for_fire_rate:
            total_bursts = fire_control.max_rounds // fire_rate
            if self.env.debug_level > 0:
                role_value = self.env.get_unit_property(member_id, 'role')
                role_name = str(role_value)
                if hasattr(role_value, 'name'):
                    role_name = role_value.name
                print(f"DEBUG: Adjusting {role_name}'s rounds for fire rate {fire_rate}")
                print(f"DEBUG: Original max_rounds: {fire_control.max_rounds}, Adjusted bursts: {total_bursts}")

        # Calculate rounds per target based on weapon fire rate
        # Each burst represents one "trigger pull"
        rounds_per_trigger = fire_rate

        # Calculate total rounds needed for engagement
        total_rounds_needed = rounds_per_trigger * total_bursts

        # Calculate total rounds available for engagement
        available_ammo = self._get_unit_ammo(member_id, 'primary')
        max_rounds = min(total_rounds_needed, available_ammo)

        # if self.env.debug_level > 0:
        #    print(f"[DEBUG ENGAGEMENT]: Weapon {weapon.name} with fire rate {fire_rate}")
        #    print(f"[DEBUG ENGAGEMENT]: Rounds per trigger pull: {rounds_per_trigger}")
        #    print(f"[DEBUG ENGAGEMENT]: Total bursts: {total_bursts}")
        #    print(f"[DEBUG ENGAGEMENT]: Total rounds needed: {total_rounds_needed}")
        #    print(f"[DEBUG ENGAGEMENT]: Available ammo: {available_ammo}")
        #    print(f"[DEBUG ENGAGEMENT]: Max rounds to use: {max_rounds}")

        # Use valid targets from fire control
        targets = fire_control.priority_targets if fire_control.priority_targets else []

        # Handle suppression-only fire
        if fire_control.suppress_only or not targets:
            # Calculate suppression effects
            results.suppression_effect = self._calculate_suppression(
                member_id, weapon, fire_control, max_rounds, distance, los_result)

            # Apply suppression to units in area - track where rounds land
            if results.suppression_effect > 0:
                # Distribute rounds across the area
                hit_locations = self._distribute_rounds(
                    fire_control.target_area,
                    fire_control.area_radius,
                    max_rounds,
                    EngagementType.AREA
                )
                results.hit_locations = hit_locations

                # Apply suppression to units based on proximity to hits
                self._apply_area_suppression(
                    hit_locations,
                    results.suppression_effect,
                    fire_control.time_limit * 2  # Duration in steps
                )

            # Update ammunition and results
            results.rounds_expended = max_rounds
            self._reduce_unit_ammo(member_id, 'primary', max_rounds)
            results.time_steps = max(1, max_rounds // fire_rate)  # Time steps based on bursts, not total rounds

            # if self.env.debug_level > 0:
            #    print(f"[DEBUG ENGAGEMENT]: Member {member_id} executed suppression")
            #    print(f"[DEBUG ENGAGEMENT]: Rounds expended: {max_rounds}")
            #    print(f"[DEBUG ENGAGEMENT]: Suppression effect: {results.suppression_effect}")

            # Record firing for visibility
            self.visibility_mgr.record_unit_fired(member_id)
            return results

        # Execute direct fire at targets
        rounds_remaining = max_rounds

        # Engage each target with appropriate distribution
        for target_id in targets:
            target_pos = self.env.get_unit_position(target_id)
            if target_pos is None:
                continue

            # Allocate rounds for this target (equal distribution among targets)
            # Ensure we allocate in multiples of fire_rate
            rounds_per_target = min(rounds_remaining, (max_rounds // len(targets) // fire_rate) * fire_rate)
            if rounds_per_target == 0:
                break

            target_hits = 0
            target_damage = 0.0
            hit_positions = []

            # Calculate bursts for this target (rounds divided by fire rate)
            bursts_for_target = rounds_per_target // fire_rate

            # Fire allocated bursts at target
            for _ in range(bursts_for_target):
                # For each burst, fire fire_rate rounds
                for _ in range(fire_rate):
                    # Check if target is suppressed for hit probability modifier
                    hit_prob_modifier = 1.0
                    if target_id in self.suppressed_units:
                        hit_prob_modifier = 1.5  # 50% bonus against suppressed targets

                    # Execute fire with distributed round location
                    hit, damage, hit_pos = self._execute_fire_with_distributed_rounds(
                        member_id,
                        target_pos,
                        distance,
                        hit_prob_modifier,
                        EngagementType.POINT if fire_control.area_radius == 0 else EngagementType.AREA,
                        fire_control.area_radius
                    )

                    # Record hit location
                    hit_positions.append(hit_pos)
                    results.hit_locations.append(hit_pos)

                    if hit:
                        # Check which unit was hit (might be a different one if round landed in adjacent cell)
                        hit_unit_id = self._get_unit_at_position(hit_pos)

                        if hit_unit_id:
                            target_hits += 1
                            target_damage += damage
                            results.hits += 1
                            results.damage_dealt += damage

                            # Apply damage to the hit unit
                            self.env.apply_damage(hit_unit_id, damage)

                            if hit_unit_id not in results.targets_hit:
                                results.targets_hit.append(hit_unit_id)

                            # Calculate and apply suppression from hits
                            hit_ratio = target_hits / rounds_per_target
                            suppression = min(0.8, hit_ratio * 0.8)  # Cap at 80% suppression
                            self._apply_suppression(
                                hit_unit_id,
                                suppression,
                                results.time_steps * 3  # Longer suppression from direct hits
                            )

                    # Apply area suppression effects from missed shots or area fire
                    # Ensure we pass a single position, not a list of positions
                    self._apply_area_suppression(
                        [hit_pos],  # Pass a list containing the single hit position
                        0.05,  # Light suppression from nearby misses
                        results.time_steps * 2
                    )

            # Update rounds remaining
            rounds_remaining -= rounds_per_target

            # if self.env.debug_level > 0:
            #    print(f"[DEBUG ENGAGEMENT]: Target {target_id} hit {target_hits} times with {rounds_per_target} rounds")
            #    print(f"[DEBUG ENGAGEMENT]: Damage dealt: {target_damage}")

        # Update final results
        results.rounds_expended = max_rounds - rounds_remaining
        self._reduce_unit_ammo(member_id, 'primary', results.rounds_expended)

        # Time steps is based on number of bursts (rounds divided by fire rate)
        results.time_steps = max(1, results.rounds_expended // fire_rate)

        # if self.env.debug_level > 0:
        #    print(f"[DEBUG ENGAGEMENT]: Member {member_id} engagement complete")
        #    print(f"[DEBUG ENGAGEMENT]: Total hits: {results.hits}/{results.rounds_expended}")
        #    print(f"[DEBUG ENGAGEMENT]: Total damage: {results.damage_dealt}")
        #    print(f"[DEBUG ENGAGEMENT]: Targets hit: {results.targets_hit}")

        # Record that the member fired
        self.visibility_mgr.record_unit_fired(member_id)

        return results

    def execute_team_engagement(self, team_id: int,
                                target_pos: Tuple[int, int],
                                engagement_type: EngagementType = EngagementType.POINT,
                                control_params: Dict = None) -> Dict:
        """
        Execute coordinated team engagement with tactical target distribution.
        Now properly adjusts rounds based on weapon fire rate.

        Args:
            team_id: ID of firing team
            target_pos: Position to engage
            engagement_type: Type of engagement to conduct
            control_params: Additional control parameters (optional)

        Returns:
            Dictionary with aggregated engagement results
        """
        # Ensure control_params is initialized
        if control_params is None:
            control_params = {}

        # Prepare tracking of aggregated results
        team_results = {
            'total_hits': 0,
            'targets_hit': set(),
            'total_damage': 0.0,
            'suppression_level': 0.0,
            'ammo_expended': 0,
            'time_steps': 0,
            'team_id': team_id,
            'hit_locations': []
        }

        # Get team members
        members = self.env.get_unit_children(team_id)
        if not members:
            return team_results

        # Sort members from left to right by x-coordinate
        members_by_position = sorted(
            members,
            key=lambda m: self.env.get_unit_position(m)[0]
        )

        # Get all potential targets in the area
        targets = self._get_targets_in_area(target_pos, max(1, control_params.get('area_radius', 3)))

        # Filter targets to those that can be engaged by at least one team member
        valid_targets = []
        for target_id in targets:
            target_pos = self.env.get_unit_position(target_id)
            if any(self.validate_target(member_id, target_pos) for member_id in members):
                valid_targets.append(target_id)

        if not valid_targets:
            if self.env.debug_level > 0:
                print(f"DEBUG: No valid targets for team {team_id}")
            return team_results

        # Check for priority targets (high threat targets)
        priority_targets = self._identify_priority_targets(valid_targets)

        # If priority targets exist, all members focus on them
        if priority_targets and control_params.get('use_priority_targeting', True):
            # Sort priority targets by threat level (highest first)
            sorted_priority = sorted(
                priority_targets,
                key=lambda t: self._calculate_threat_level(t),
                reverse=True
            )

            # All members focus on highest threat target first
            target_assignments = {member_id: sorted_priority for member_id in members}
        else:
            # Create base fire control
            area_radius = 0 if engagement_type == EngagementType.POINT else 3
            suppress_only = control_params.get('suppress_only', False)
            sustained = control_params.get('sustained', False)

            # Distribute targets tactically across team members
            if engagement_type == EngagementType.POINT and len(valid_targets) > 1:
                # Sort targets from left to right
                targets_by_position = sorted(
                    valid_targets,
                    key=lambda t: self.env.get_unit_position(t)[0]
                )

                # Assign targets to members (leftmost member -> leftmost target, etc.)
                target_assignments = {}

                # If we have more members than targets, assign multiple members to same targets
                if len(members_by_position) >= len(targets_by_position):
                    members_per_target = len(members_by_position) // len(targets_by_position)
                    for i, member_id in enumerate(members_by_position):
                        target_idx = min(i // members_per_target, len(targets_by_position) - 1)
                        target_assignments[member_id] = [targets_by_position[target_idx]]
                # If we have more targets than members, assign multiple targets to same members
                else:
                    targets_per_member = len(targets_by_position) // len(members_by_position)
                    for i, member_id in enumerate(members_by_position):
                        start_idx = i * targets_per_member
                        end_idx = start_idx + targets_per_member
                        if i == len(members_by_position) - 1:  # Last member gets all remaining targets
                            end_idx = len(targets_by_position)
                        target_assignments[member_id] = targets_by_position[start_idx:end_idx]
            else:
                # For area fire or single target, all members engage the same targets
                target_assignments = {member_id: valid_targets for member_id in members}

        # Get basic parameters
        max_rounds = control_params.get('max_rounds', 1)  # Default 1 round per target
        adjust_for_fire_rate = control_params.get('adjust_for_fire_rate', False)  # NEW: Flag to adjust for fire rate

        # Execute engagements based on assignments
        for member_id, assigned_targets in target_assignments.items():
            # Skip casualties or heavily suppressed units
            if self.env.get_unit_property(member_id, 'health', 0) <= 0:
                continue

            if member_id in self.suppressed_units and self.suppressed_units[member_id]['level'] >= 0.7:
                continue

            # Get weapon to determine rate of fire
            weapon = self._get_unit_weapon(member_id)
            if weapon:
                # Create fire control with proper rate of fire consideration
                member_control = FireControl(
                    target_area=target_pos,
                    area_radius=0 if engagement_type == EngagementType.POINT else 3,
                    max_rounds=max_rounds,  # This represents "bursts" or single shots
                    time_limit=control_params.get('time_limit', 5),
                    suppress_only=control_params.get('suppress_only', False),
                    priority_targets=assigned_targets,
                    sustained=control_params.get('sustained', False),
                    adjust_for_fire_rate=adjust_for_fire_rate  # Pass the fire rate adjustment flag
                )

                # Execute engagement
                results = self.execute_engagement(member_id, member_control)

                # Aggregate results
                team_results['total_hits'] += results.hits
                team_results['targets_hit'].update(results.targets_hit)
                team_results['total_damage'] += results.damage_dealt
                team_results['suppression_level'] = max(
                    team_results['suppression_level'],
                    results.suppression_effect
                )
                team_results['ammo_expended'] += results.rounds_expended
                team_results['time_steps'] = max(
                    team_results['time_steps'],
                    results.time_steps
                )
                team_results['hit_locations'].extend(results.hit_locations)

        # Convert set to list for serialization
        team_results['targets_hit'] = list(team_results['targets_hit'])

        # Calculate team effectiveness using shared function
        team_results['effectiveness'] = self._calculate_effectiveness({
            'total_hits': team_results['total_hits'],
            'ammo_expended': team_results['ammo_expended'],
            'total_damage': team_results['total_damage'],
            'suppression_level': team_results['suppression_level']
        })

        # Record team engagement
        self._record_engagement(team_id, {
            'type': engagement_type.name.lower(),
            'target_area': target_pos,
            'results': team_results,
            'time': datetime.now()
        })

        return team_results

    def execute_squad_engagement(self, squad_id: int,
                                 target_area: Tuple[int, int],
                                 teams_to_engage: List[int] = None,
                                 engagement_type: EngagementType = EngagementType.POINT) -> Dict:
        """
        Execute coordinated squad-level engagement with mixed team tactics.

        Args:
            squad_id: ID of squad
            target_area: Central target area
            teams_to_engage: Specific teams to use (default: all teams)
            engagement_type: Base type of engagement

        Returns:
            Dictionary with squad engagement results
        """
        # Initialize results
        squad_results = {
            'total_hits': 0,
            'targets_hit': set(),
            'total_damage': 0.0,
            'suppression_level': 0.0,
            'ammo_expended': 0,
            'time_steps': 0,
            'participating_teams': [],
            'squad_id': squad_id,
            'hit_locations': []
        }

        # Get teams in squad
        team_ids = [unit_id for unit_id in self.env.get_unit_children(squad_id)
                    if self.env.get_unit_property(unit_id, 'type') in
                    [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM]]

        # Filter to specified teams if provided
        if teams_to_engage:
            team_ids = [team_id for team_id in team_ids if team_id in teams_to_engage]

        if not team_ids:
            return squad_results

        # Record participating teams
        squad_results['participating_teams'] = team_ids

        # Analyze squad composition and determine tactical roles
        # Typically: weapons teams provide suppression, fire teams maneuver/engage point targets
        weapons_teams = []
        fire_teams = []

        for team_id in team_ids:
            team_type = self.env.get_unit_property(team_id, 'team_type')
            if team_type in ['weapons', 'gun', 'javelin']:
                weapons_teams.append(team_id)
            else:
                fire_teams.append(team_id)

        # Determine tactical employment for each team
        team_engagements = {}

        # If we have a mix of team types, use complementary tactics
        if weapons_teams and fire_teams:
            # Weapons teams provide suppression with area fire
            for team_id in weapons_teams:
                team_engagements[team_id] = (EngagementType.AREA, {
                    'suppress_only': True,
                    'sustained': True,
                    'area_radius': 4,  # Wider suppression area
                    'max_rounds': 50
                })

            # Fire teams engage with point fire
            for team_id in fire_teams:
                team_engagements[team_id] = (EngagementType.POINT, {
                    'suppress_only': False,
                    'use_priority_targeting': True,
                    'max_rounds': 15
                })

        # If all the same type, use the specified engagement type
        else:
            for team_id in team_ids:
                # Get team capability profile
                auto_weapons_count = 0
                high_damage_weapons_count = 0

                for member_id in self.env.get_unit_children(team_id):
                    weapon = self._get_unit_weapon(member_id)
                    if weapon:
                        if weapon.fire_rate > 1:
                            auto_weapons_count += 1
                        if weapon.damage > 40:
                            high_damage_weapons_count += 1

                # Configure based on capability
                if auto_weapons_count >= 2:  # Team with multiple automatic weapons
                    team_engagements[team_id] = (
                        EngagementType.AREA if engagement_type != EngagementType.POINT else EngagementType.POINT,
                        {'sustained': True}
                    )
                elif high_damage_weapons_count >= 1:  # Team with high damage capability
                    team_engagements[team_id] = (EngagementType.POINT, {'use_priority_targeting': True})
                else:  # Standard team
                    team_engagements[team_id] = (engagement_type, {})

        # Execute team engagements with appropriate configurations
        for team_id in team_ids:
            # Skip if team is combat ineffective
            if self.get_team_combat_effectiveness(team_id) < 0.5:  # Team is ineffective if below 50%
                continue

            # Get engagement configuration for this team
            engagement_type, control_params = team_engagements[team_id]

            # Execute team engagement
            team_result = self.execute_team_engagement(
                team_id,
                target_area,  # Each team engages the main target area
                engagement_type,
                control_params
            )

            # Aggregate results
            squad_results['total_hits'] += team_result['total_hits']
            squad_results['targets_hit'].update(team_result['targets_hit'])
            squad_results['total_damage'] += team_result['total_damage']
            squad_results['suppression_level'] = max(
                squad_results['suppression_level'],
                team_result['suppression_level']
            )
            squad_results['ammo_expended'] += team_result['ammo_expended']
            squad_results['time_steps'] = max(
                squad_results['time_steps'],
                team_result['time_steps']
            )
            squad_results['hit_locations'].extend(team_result.get('hit_locations', []))

        # Convert set to list for serialization
        squad_results['targets_hit'] = list(squad_results['targets_hit'])

        # Calculate squad effectiveness using shared function
        squad_results['effectiveness'] = self._calculate_effectiveness({
            'total_hits': squad_results['total_hits'],
            'ammo_expended': squad_results['ammo_expended'],
            'total_damage': squad_results['total_damage'],
            'suppression_level': squad_results['suppression_level']
        })

        # Record squad engagement
        self._record_engagement(squad_id, {
            'type': f'squad_{engagement_type.name.lower()}',
            'target_area': target_area,
            'results': squad_results,
            'time': datetime.now()
        })

        if self.env.debug_level > 0:
            print(f"\nSquad {squad_id} engagement complete:")
            print(f"- Total hits: {squad_results['total_hits']}")
            print(f"- Total damage: {squad_results['total_damage']:.1f}")
            print(f"- Suppression level: {squad_results['suppression_level']:.2f}")
            print(f"- Ammunition expended: {squad_results['ammo_expended']}")
            print(f"- Effectiveness: {squad_results['effectiveness']:.2f}")

        return squad_results

    def update_suppression_states(self) -> Dict[int, float]:
        """
        Update suppression states for all units, reducing effects over time.
        Should be called every environment step.

        Returns:
            Dictionary mapping unit IDs to current suppression levels
        """
        # Track units that recovered
        recovered_units = []
        current_levels = {}

        # Update each suppressed unit
        for unit_id, supp_data in self.suppressed_units.items():
            # Reduce duration
            supp_data['duration'] -= 1

            # Unit recovered if duration expired
            if supp_data['duration'] <= 0:
                recovered_units.append(unit_id)
                continue

            # Gradually reduce suppression level
            new_level = max(0.0, supp_data['level'] - self.suppression_recovery_rate)
            supp_data['level'] = new_level

            # Record current level
            current_levels[unit_id] = new_level

            # Remove if below threshold
            if new_level < 0.05:
                recovered_units.append(unit_id)

        # Remove recovered units
        for unit_id in recovered_units:
            self.suppressed_units.pop(unit_id, None)

        return current_levels

    def get_unit_combat_state(self, unit_id: int) -> Dict:
        """
        Get combat state considering unit hierarchy.

        Args:
            unit_id: Unit to check (can be team/squad or individual)

        Returns:
            Dictionary with combat state information
        """
        # Get unit type
        unit_type = self.env.get_unit_property(unit_id, 'type')

        if unit_type in [UnitType.INFANTRY_TEAM, UnitType.INFANTRY_SQUAD, UnitType.WEAPONS_TEAM]:
            # For teams/squads, combine member states
            members = self.env.get_unit_children(unit_id)

            # Initialize aggregate values
            total_health = 0
            total_ammo_primary = 0
            total_ammo_secondary = 0
            max_suppression = 0.0
            member_states = []
            can_engage = False

            for member_id in members:
                member_state = self._get_member_combat_state(member_id)
                member_states.append(member_state)

                # Aggregate values
                total_health += member_state['health']
                total_ammo_primary += member_state['ammo_primary']
                total_ammo_secondary += member_state['ammo_secondary']
                max_suppression = max(max_suppression,
                                      member_state.get('suppression_level', 0))
                can_engage = can_engage or member_state['can_engage']

            # Calculate averages
            num_members = len(members)
            avg_health = total_health / num_members if num_members > 0 else 0

            return {
                'unit_id': unit_id,
                'unit_type': unit_type,
                'avg_health': avg_health,
                'total_ammo_primary': total_ammo_primary,
                'total_ammo_secondary': total_ammo_secondary,
                'suppressed': max_suppression > 0.1,
                'suppression_level': max_suppression,
                'can_engage': can_engage,
                'member_states': member_states,
                'recent_engagements': self.engagement_history.get(unit_id, [])[-5:]
            }
        else:
            # For individual soldiers, get direct state
            return self._get_member_combat_state(unit_id)

    def get_team_combat_effectiveness(self, team_id: int) -> float:
        """
        Calculate team's current combat effectiveness (0-1).
        This measures the team's current ability to engage in combat,
        not the effectiveness of a specific engagement.

        Factors:
        - Health status of members
        - Suppression status
        - Ammunition status
        - Leader status
        - Percentage of team still active

        Args:
            team_id: Team to evaluate

        Returns:
            Combat effectiveness score (0-1)
        """
        members = self.env.get_unit_children(team_id)
        if not members:
            return 0.0

        total_members = len(members)
        if total_members == 0:
            return 0.0

        # Track component scores
        total_health = 0.0
        total_ammo = 0.0
        total_suppression = 0.0
        active_members = 0
        has_leader = False

        for member_id in members:
            # Skip if member is a casualty
            health = self.env.get_unit_property(member_id, 'health', 0)
            if health <= 0:
                continue

            active_members += 1

            # Health component (0-1)
            total_health += health / 100.0

            # Ammo component (0-1)
            ammo = self._get_unit_ammo(member_id, 'primary')
            weapon = self._get_unit_weapon(member_id)
            if weapon and weapon.ammo_capacity > 0:
                total_ammo += min(1.0, ammo / weapon.ammo_capacity)

            # Suppression component (0-1, inverted since higher suppression is worse)
            if member_id in self.suppressed_units:
                total_suppression += (1.0 - self.suppressed_units[member_id]['level'])
            else:
                total_suppression += 1.0

            # Check if this is the leader
            if self.env.get_unit_property(member_id, 'is_leader', False):
                has_leader = True

        if active_members == 0:
            return 0.0  # No active members means team is ineffective

        # Calculate component averages
        health_score = total_health / total_members
        ammo_score = total_ammo / total_members
        suppression_score = total_suppression / total_members
        strength_score = active_members / total_members
        leadership_score = 1.0 if has_leader else 0.5

        # Weight the components
        effectiveness = (
                health_score * 0.25 +  # Health status (25%)
                ammo_score * 0.20 +  # Ammunition status (20%)
                suppression_score * 0.20 +  # Suppression status (20%)
                strength_score * 0.20 +  # Team strength (20%)
                leadership_score * 0.15  # Leadership status (15%)
        )

        if self.env.debug_level > 1:
            print(f"\nTeam {team_id} Combat Effectiveness Analysis:")
            print(f"Active Members: {active_members}/{total_members}")
            print(f"Health Score: {health_score:.2f}")
            print(f"Ammo Score: {ammo_score:.2f}")
            print(f"Suppression Score: {suppression_score:.2f}")
            print(f"Strength Score: {strength_score:.2f}")
            print(f"Leadership Score: {leadership_score:.2f}")
            print(f"Overall Effectiveness: {effectiveness:.2f}")

        return effectiveness

    def reset(self):
        """Reset combat manager state for new episode."""
        self.engagement_history.clear()
        self.suppressed_units.clear()
        self.ammo_tracking.clear()

    # ===== Core Target Validation Methods =====

    def validate_target(self, unit_id: int, target_pos: Tuple[int, int]) -> bool:
        """
        Unified method to validate if a target is engageable by a unit.
        Consolidates checks previously split across multiple methods.

        Args:
            unit_id: ID of unit checking target
            target_pos: Position to engage

        Returns:
            Boolean indicating whether engagement is possible
        """
        # Add debug print
        # print(f"\n[DEBUG VALIDATE] Starting validation for unit {unit_id} -> target {target_pos}")

        # Check if unit exists and is active
        if unit_id not in self.env.state_manager.active_units:
            # print(f"[DEBUG VALIDATE] Unit {unit_id} not in active units")
            return False

        # Check if unit is alive
        health = self.env.get_unit_property(unit_id, 'health', 0)
        if health <= 0:
            # print(f"[DEBUG VALIDATE] Unit {unit_id} health is {health}")
            return False

        # Check if unit is heavily suppressed
        if unit_id in self.suppressed_units:
            if self.suppressed_units[unit_id]['level'] >= 0.8:
                # print(f"[DEBUG VALIDATE] Unit {unit_id} heavily suppressed")
                return False

        # Get unit position and engagement properties
        unit_pos = self.env.get_unit_position(unit_id)
        engagement_range = self.env.get_unit_property(unit_id, 'engagement_range', 40)
        observation_range = self.env.get_unit_property(unit_id, 'observation_range', 50)

        # Check ranges
        distance = math.sqrt(
            (target_pos[0] - unit_pos[0]) ** 2 +
            (target_pos[1] - unit_pos[1]) ** 2
        )

        # print(f"[DEBUG VALIDATE] Distance: {distance}, Engagement range: {engagement_range}, Observation range: {
        # observation_range}")

        if distance > engagement_range or distance > observation_range:
            # print(f"[DEBUG VALIDATE] Target beyond range")
            return False

        # Check if unit has weapon and ammunition
        weapon = self._get_unit_weapon(unit_id)
        if not weapon:
            # print(f"[DEBUG VALIDATE] No weapon")
            return False

        ammo = self._get_unit_ammo(unit_id, 'primary')
        if ammo <= 0:
            # print(f"[DEBUG VALIDATE] No ammo")
            return False

        # Check if weapons operational
        weapons_operational = self.env.get_unit_property(unit_id, 'weapons_operational', True)
        if not weapons_operational:
            # print(f"[DEBUG VALIDATE] Weapons not operational")
            return False

        # Check line of sight
        los_result = self.visibility_mgr.check_line_of_sight(unit_pos, target_pos)
        if not los_result['has_los']:
            # print(f"[DEBUG VALIDATE] No line of sight")
            return False

        # Calculate angle to target
        target_angle = self.calculate_target_angle(unit_pos, target_pos)
        # print(f"[DEBUG VALIDATE] Target angle: {target_angle}")

        # Check if target is in either sector using the stored rotated sector values
        primary_start = self.env.get_unit_property(unit_id, 'primary_sector_rotated_start')
        primary_end = self.env.get_unit_property(unit_id, 'primary_sector_rotated_end')

        secondary_start = self.env.get_unit_property(unit_id, 'secondary_sector_rotated_start')
        secondary_end = self.env.get_unit_property(unit_id, 'secondary_sector_rotated_end')

        # print(f"[DEBUG VALIDATE] Primary sector: {primary_start}°-{primary_end}°")
        # print(f"[DEBUG VALIDATE] Secondary sector: {secondary_start}°-{secondary_end}°")

        # If rotated sectors are stored in properties, use them
        if primary_start is not None and primary_end is not None:
            # print(f"[DEBUG VALIDATE] Using stored rotated sectors")

            # Check if angle is in primary sector
            in_primary = False
            if primary_start <= primary_end:
                in_primary = primary_start <= target_angle <= primary_end
            else:
                in_primary = target_angle >= primary_start or target_angle <= primary_end

            # Check if angle is in secondary sector
            in_secondary = False
            if secondary_start is not None and secondary_end is not None:
                if secondary_start <= secondary_end:
                    in_secondary = secondary_start <= target_angle <= secondary_end
                else:
                    in_secondary = target_angle >= secondary_start or target_angle <= secondary_end

            # If not in either sector, target is not valid
            if not (in_primary or in_secondary):
                # print(f"[DEBUG VALIDATE] Target not in stored rotated sectors")
                return False

            # print(f"[DEBUG VALIDATE] Target in sectors: Primary={in_primary}, Secondary={in_secondary}")
            # print(f"[DEBUG VALIDATE] Target validation SUCCESSFUL")
            return True

        else:
            # Fall back to legacy method of dynamic rotation
            # print(f"[DEBUG VALIDATE] No stored rotated sectors, falling back to dynamic calculation")
            # Legacy sector check
            orientation = self.env.get_unit_property(unit_id, 'orientation', 0)
            role = self.env.get_unit_property(unit_id, 'role')
            if role is not None:
                # Get parent unit's formation
                parent_id = self.env.get_unit_property(unit_id, 'parent_id')
                if parent_id:
                    formation = self.env.get_unit_property(parent_id, 'formation', 'team_wedge_right')
                    is_leader = self.env.get_unit_property(unit_id, 'is_leader', False)

                    from US_Army_PLT_Composition_vTest import get_unit_sectors, US_IN_Role
                    if isinstance(role, int):
                        role = US_IN_Role(role)
                    primary_sector, secondary_sector = get_unit_sectors(role, formation, is_leader)

                    # Check if target is in either sector
                    if primary_sector and not self.is_target_in_sector(unit_pos, orientation, target_pos,
                                                                       primary_sector):
                        if not secondary_sector or not self.is_target_in_sector(unit_pos, orientation, target_pos,
                                                                                secondary_sector):
                            # print(f"[DEBUG VALIDATE] Target not in dynamically calculated sectors")
                            return False
                    # print(f"[DEBUG VALIDATE] Target in dynamically calculated sectors")

            return True

    def _get_valid_targets(self, unit_id: int, target_area: Tuple[int, int],
                           area_radius: int = 0) -> List[int]:
        """
        Get list of valid targets considering unit hierarchy and combat mechanics.

        Args:
            unit_id: ID of firing unit/team/squad
            target_area: Center of target area
            area_radius: Radius for area fire (0 for point target)

        Returns:
            List of valid target unit IDs
        """
        valid_targets = []

        # Get unit type
        unit_type = self.env.get_unit_property(unit_id, 'type')

        if unit_type in [UnitType.INFANTRY_TEAM, UnitType.INFANTRY_SQUAD, UnitType.WEAPONS_TEAM]:
            # For teams/squads, consider member capabilities
            members = self.env.get_unit_children(unit_id)
            for member_id in members:
                member_targets = self._get_member_valid_targets(
                    member_id, target_area, area_radius)
                valid_targets.extend([t for t in member_targets if t not in valid_targets])
        else:
            # For individual soldiers
            valid_targets = self._get_member_valid_targets(unit_id, target_area, area_radius)

        # if self.env.debug_level > 0:
        #    print(f"DEBUG: Found {len(valid_targets)} valid targets for unit {unit_id}")

        return valid_targets

    def _get_member_valid_targets(self, member_id: int, target_area: Tuple[int, int],
                                  area_radius: int = 0) -> List[int]:
        """Get valid targets for an individual soldier."""
        # Get all potential targets in area
        potential_targets = self._get_targets_in_area(target_area, max(1, area_radius))
        valid_targets = []

        # Check each target
        for target_id in potential_targets:
            target_pos = self.env.get_unit_position(target_id)
            if target_pos is None:
                continue

            # Skip if target is dead
            if self.env.get_unit_property(target_id, 'health', 0) <= 0:
                continue

            # Check if member can engage this specific target
            if self.validate_target(member_id, target_pos):
                valid_targets.append(target_id)

        return valid_targets

    def _get_targets_in_area(self, center: Tuple[int, int], radius: int) -> List[int]:
        """Get unit IDs of potential targets in area."""
        targets = []
        center_x, center_y = center

        # Check all active units
        for unit_id in self.env.state_manager.active_units:
            # Skip units with zero health
            if self.env.get_unit_property(unit_id, 'health', 0) <= 0:
                continue

            pos = self.env.get_unit_position(unit_id)
            if pos is None:
                continue

            distance = math.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2)

            if distance <= radius:
                # Only add team/squad units, not their individual members
                parent_id = self.env.get_unit_property(unit_id, 'parent_id')
                if parent_id is None:  # This is a team/squad
                    targets.append(unit_id)
                else:
                    # This is an individual - only add if its parent isn't already in targets
                    if parent_id not in targets:
                        targets.append(unit_id)

        return targets

    # ===== Enhanced Firing Methods =====
    def _distribute_rounds(self, target_pos: Tuple[int, int], area_radius: int,
                           num_rounds: int, engagement_type: EngagementType) -> List[Tuple[int, int]]:
        """
        Distribute rounds across target area based on engagement type.

        Args:
            target_pos: Center of target area
            area_radius: Radius for area fire
            num_rounds: Number of rounds to distribute
            engagement_type: POINT or AREA

        Returns:
            List of (x, y) positions where rounds land
        """
        hit_positions = []
        target_x, target_y = target_pos

        # For point engagement, all rounds are aimed at target with small variance
        if engagement_type == EngagementType.POINT:
            for _ in range(num_rounds):
                # Small random deviation from point of aim
                deviation_x = random.randint(-1, 1)
                deviation_y = random.randint(-1, 1)

                # Ensure position is within environment bounds
                x = max(0, min(target_x + deviation_x, self.env.width - 1))
                y = max(0, min(target_y + deviation_y, self.env.height - 1))

                hit_positions.append((x, y))

        # For area engagement, distribute rounds across the area
        else:
            effective_radius = max(1, area_radius)
            for _ in range(num_rounds):
                # Random position within radius
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, effective_radius)

                # Calculate position
                x = int(target_x + distance * math.cos(angle))
                y = int(target_y + distance * math.sin(angle))

                # Ensure position is within environment bounds
                x = max(0, min(x, self.env.width - 1))
                y = max(0, min(y, self.env.height - 1))

                hit_positions.append((x, y))

        return hit_positions

    def _execute_fire_with_distributed_rounds(self, unit_id: int, target_pos: Tuple[int, int],
                                              distance: float, hit_prob_modifier: float = 1.0,
                                              engagement_type: EngagementType = EngagementType.POINT,
                                              area_radius: int = 0) -> Tuple[bool, float, Tuple[int, int]]:
        """
        Execute fire with distributed round location, accounting for fire rate.

        Args:
            unit_id: ID of firing unit
            target_pos: Position of target
            distance: Distance to target
            hit_prob_modifier: Modifier for hit probability
            engagement_type: POINT or AREA
            area_radius: Radius for area fire

        Returns:
            Tuple of (hit_successful, damage_dealt, hit_position)
        """
        # Get base hit probability
        unit_pos = self.env.get_unit_position(unit_id)
        weapon = self._get_unit_weapon(unit_id)

        base_hit_prob = self.env.calculate_hit_probability(distance, weapon)

        # Apply suppression effects
        if unit_id in self.suppressed_units:
            supp_level = self.suppressed_units[unit_id]['level']
            if supp_level > 0:
                # Reduce accuracy based on suppression
                base_hit_prob *= (1.0 - (supp_level * 0.8))

        # Apply visibility modifiers
        target_id = self._get_unit_at_position(target_pos)
        modified_hit_prob = self.visibility_mgr.modify_hit_probability(
            base_hit_prob * hit_prob_modifier,
            unit_pos,
            target_pos,
            target_id if target_id else 1  # Use 1 as default ID
        )

        # Debug hit probability for each round
        # if self.env.debug_level > 0:
        #    print(f"[DEBUG FIRE]: Round hit probability: {modified_hit_prob:.2f}")

        # Determine if shot hits
        hit = random.random() < modified_hit_prob

        # Determine where round lands
        if hit:
            # Direct hit at target position
            hit_pos = target_pos
        else:
            # Miss - calculate miss location within 1-2 cells of target
            # The further the distance, the greater the spread
            spread_factor = min(2.0, distance / 20.0)  # Max spread of 2 cells

            # Random direction
            miss_angle = random.uniform(0, 2 * math.pi)
            miss_distance = random.uniform(1.0, spread_factor)

            # Calculate miss position
            miss_x = int(target_pos[0] + miss_distance * math.cos(miss_angle))
            miss_y = int(target_pos[1] + miss_distance * math.sin(miss_angle))

            # Ensure position is within environment bounds
            miss_x = max(0, min(miss_x, self.env.width - 1))
            miss_y = max(0, min(miss_y, self.env.height - 1))

            hit_pos = (miss_x, miss_y)

            # Check if miss actually hit another unit
            miss_target_id = self._get_unit_at_position(hit_pos)
            if miss_target_id:
                hit = True  # Counts as a hit, just on a different target
                target_id = miss_target_id

        # Check if unit is already dead before applying damage
        if hit and target_id:
            unit_health = self.env.get_unit_property(target_id, 'health', 0)
            if unit_health <= 0:
                # Skip damage calculation for already dead units
                # if self.env.debug_level > 0:
                    # print(f"[DEBUG FIRE]: Target {target_id} is already eliminated, no damage applied")
                return hit, 0.0, hit_pos

        # Calculate damage if hit
        damage = 0.0
        if hit and target_id:
            base_damage = self.env.calculate_damage(distance, weapon)
            damage = self.visibility_mgr.modify_damage(
                base_damage,
                hit_pos,
                unit_pos
            )

            # Debug damage for this round
            # if self.env.debug_level > 0:
            #    print(f"[DEBUG FIRE]: Round damage on hit: {damage:.1f}")

        return hit, damage, hit_pos

    def _apply_area_suppression(self, hit_locations: List[Tuple[int, int]],
                                suppression_level: float, duration: int) -> None:
        """
        Apply suppression to units based on proximity to hit locations.

        Args:
            hit_locations: List of positions where rounds landed
            suppression_level: Base suppression level to apply
            duration: Duration in steps
        """
        # Track which units have been suppressed to apply strongest effect
        suppressed_units = {}

        # Check each hit location and nearby cells
        for hit_pos in hit_locations:
            x, y = hit_pos

            # Check surrounding cells (3x3 grid)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    # Calculate cell position
                    cell_x = x + dx
                    cell_y = y + dy

                    # Skip if out of bounds
                    if not (0 <= cell_x < self.env.width and 0 <= cell_y < self.env.height):
                        continue

                    # Check for unit at this position
                    unit_id = self._get_unit_at_position((cell_x, cell_y))
                    if not unit_id:
                        continue

                    # Skip if unit is already eliminated
                    if self.env.get_unit_property(unit_id, 'health', 0) <= 0:
                        continue

                    # Calculate distance-based suppression modifier
                    distance = math.sqrt(dx ** 2 + dy ** 2)
                    dist_modifier = 1.0 if distance == 0 else (1.0 / distance)

                    # Calculate cell-specific suppression level
                    cell_suppression = suppression_level * dist_modifier

                    # Track highest suppression for this unit
                    if unit_id not in suppressed_units or cell_suppression > suppressed_units[unit_id]:
                        suppressed_units[unit_id] = cell_suppression

        # Apply suppression to affected units
        for unit_id, level in suppressed_units.items():
            self._apply_suppression(unit_id, level, duration)

    def _apply_suppression(self, unit_id: int, level: float, duration: int) -> None:
        """Apply suppression effect to unit."""
        # Skip if unit doesn't exist or is already eliminated
        if unit_id not in self.env.state_manager.active_units:
            return

        if self.env.get_unit_property(unit_id, 'health', 0) <= 0:
            return

        # Check if already suppressed
        current_level = 0.0
        current_duration = 0
        if unit_id in self.suppressed_units:
            current_level = self.suppressed_units[unit_id]['level']
            current_duration = self.suppressed_units[unit_id]['duration']

        # Apply the highest suppression level
        new_level = max(current_level, level)

        # Add duration (doesn't completely reset when already suppressed)
        new_duration = current_duration
        if level >= current_level:
            # Higher intensity extends duration
            new_duration = max(current_duration, duration)
        else:
            # Lower intensity adds some duration
            new_duration = current_duration + (duration // 2)

        # Update suppression
        self.suppressed_units[unit_id] = {
            'level': new_level,
            'duration': new_duration,
            'applied_time': datetime.now()
        }

        # Add visual indicator via status flag if possible
        try:
            unit_pos = self.env.get_unit_position(unit_id)
            self.env.state_manager.state_tensor[unit_pos[1], unit_pos[0], 3] |= 1  # Set suppression bit
        except:
            pass  # Cannot update visualization

    def _calculate_suppression(self, unit_id: int, weapon: Any,
                               fire_control: FireControl, rounds: int,
                               distance: float, los_result: Dict) -> float:
        """Calculate suppression effect of fire."""
        # No suppression if no line of sight
        if not los_result['has_los']:
            return 0.0

        # Base suppression from volume of fire
        volume_factor = min(1.0, rounds / 30.0)  # Scale with rounds

        # Add sustained fire bonus
        if fire_control.sustained:
            volume_factor = min(1.0, volume_factor * 1.5)  # 50% bonus for sustained fire

        # Weapon type modifier
        weapon_multiplier = 1.0
        if weapon:
            if weapon.is_area_weapon:
                weapon_multiplier = 2.0  # Area weapons are more suppressive
            elif 'SAW' in weapon.name or 'LMG' in weapon.name:
                weapon_multiplier = 1.5  # Light machine guns
            elif 'M240' in weapon.name:
                weapon_multiplier = 1.8  # Medium machine guns

        # Distance falloff
        distance_factor = max(0.2, 1.0 - (distance / weapon.max_range))

        # Terrain effect on suppression
        terrain_factor = los_result['los_quality']

        # Area effect
        area_factor = 1.0
        if fire_control.area_radius > 0:
            # Wider area means less concentrated suppression
            area_factor = max(0.5, 1.0 - (fire_control.area_radius / 10.0))

        # Calculate final effect
        suppression = (
                volume_factor *
                weapon_multiplier *
                distance_factor *
                terrain_factor *
                area_factor
        )

        return min(1.0, suppression)

    def _identify_priority_targets(self, targets: List[int]) -> List[int]:
        """
        Identify high-threat priority targets from a list of targets.
        Priority targets are those with high fire rate or high damage weapons.

        Args:
            targets: List of target unit IDs

        Returns:
            List of priority target IDs
        """
        priority_targets = []

        for target_id in targets:
            # Check if target is a high threat
            if self._is_high_threat_target(target_id):
                priority_targets.append(target_id)

        return priority_targets

    def _is_high_threat_target(self, target_id: int) -> bool:
        """
        Check if a target is a high threat based on weapons.

        Args:
            target_id: Target unit ID

        Returns:
            True if target is a high threat
        """
        # For groups (teams/squads), check if any member has high threat weapon
        unit_type = self.env.get_unit_property(target_id, 'type')

        if unit_type in [UnitType.INFANTRY_TEAM, UnitType.INFANTRY_SQUAD, UnitType.WEAPONS_TEAM]:
            members = self.env.get_unit_children(target_id)
            return any(self._has_high_threat_weapon(member_id) for member_id in members)
        else:
            # For individuals
            return self._has_high_threat_weapon(target_id)

    def _has_high_threat_weapon(self, unit_id: int) -> bool:
        """
        Check if a unit has a high threat weapon (high fire rate or high damage).

        Args:
            unit_id: Unit ID to check

        Returns:
            True if unit has high threat weapon
        """
        weapon = self._get_unit_weapon(unit_id)
        if not weapon:
            return False

    def _calculate_threat_level(self, target_id: int) -> float:
        """
        Calculate the threat level of a target based on weapons and status.

        Args:
            target_id: Target unit ID

        Returns:
            Float representing threat level (higher is more threatening)
        """
        # Base threat level
        threat_level = 0.0

        # For groups, calculate max threat from members
        unit_type = self.env.get_unit_property(target_id, 'type')

        if unit_type in [UnitType.INFANTRY_TEAM, UnitType.INFANTRY_SQUAD, UnitType.WEAPONS_TEAM]:
            members = self.env.get_unit_children(target_id)
            member_threats = [self._calculate_member_threat(m) for m in members]

            # Use highest member threat, plus a bonus for coordinated teams
            if member_threats:
                threat_level = max(member_threats) * (1.0 + (len(member_threats) * 0.1))
        else:
            # For individuals
            threat_level = self._calculate_member_threat(target_id)

        return threat_level

    def _calculate_member_threat(self, member_id: int) -> float:
        """
        Calculate threat level of an individual unit based on weapons and status.

        Args:
            member_id: ID of unit

        Returns:
            Float representing threat level
        """
        # Check if unit is alive
        health = self.env.get_unit_property(member_id, 'health', 0)
        if health <= 0:
            return 0.0

        # Check if unit is suppressed
        suppression = 0.0
        if member_id in self.suppressed_units:
            suppression = self.suppressed_units[member_id]['level']

        # Get weapon info
        weapon = self._get_unit_weapon(member_id)
        if not weapon:
            return 0.0

        # Calculate weapon threat value
        weapon_threat = 0.0

        # High fire rate is a significant threat
        if weapon.fire_rate > 5:  # Very high fire rate
            weapon_threat += 5.0
        elif weapon.fire_rate > 1:  # Automatic weapon
            weapon_threat += 3.0
        else:  # Single shot
            weapon_threat += 1.0

        # High damage is a significant threat
        if weapon.damage > 80:  # Very high damage
            weapon_threat += 5.0
        elif weapon.damage > 40:  # High damage
            weapon_threat += 3.0
        else:  # Standard damage
            weapon_threat += 1.0

        # Area weapons are more threatening
        if weapon.is_area_weapon:
            weapon_threat *= 1.5

        # Apply health and suppression modifiers
        health_factor = health / 100.0
        suppression_factor = 1.0 - suppression

        # Final threat calculation
        threat = weapon_threat * health_factor * suppression_factor

        return threat

    # ===== Helper Methods =====
    def _get_member_combat_state(self, member_id: int) -> Dict:
        """Get combat state for an individual soldier."""
        # Get basic member info
        health = self.env.get_unit_property(member_id, 'health', 0)

        # Get ammunition status
        ammo_primary = self._get_unit_ammo(member_id, 'primary')
        ammo_secondary = self._get_unit_ammo(member_id, 'secondary')

        # Get weapon info
        weapon = self._get_unit_weapon(member_id)
        weapon_name = weapon.name if weapon else "None"

        # Check suppression status
        suppression_level = 0.0
        suppression_duration = 0
        if member_id in self.suppressed_units:
            suppression_level = self.suppressed_units[member_id]['level']
            suppression_duration = self.suppressed_units[member_id]['duration']

        # Get recent engagements
        recent_engagements = self.engagement_history.get(member_id, [])[-5:]

        # Get member position for engagement check
        member_pos = self.env.get_unit_position(member_id)
        # Use current position as reference point for engagement check
        reference_pos = (member_pos[0] + 1, member_pos[1])  # Check 1 unit ahead

        return {
            'member_id': member_id,
            'health': health,
            'ammo_primary': ammo_primary,
            'ammo_secondary': ammo_secondary,
            'weapon': weapon_name,
            'suppressed': suppression_level > 0.1,
            'suppression_level': suppression_level,
            'suppression_duration': suppression_duration,
            'can_engage': self.validate_target(member_id, reference_pos),
            'recent_engagements': recent_engagements
        }

    def _get_unit_weapon(self, unit_id: int) -> Optional[Any]:
        """Get unit's primary weapon."""
        return self.env.get_unit_property(unit_id, 'primary_weapon')

    def _get_unit_ammo(self, unit_id: int, weapon_type: str) -> int:
        """Get unit's current ammunition."""
        # Check ammo tracking first
        if unit_id in self.ammo_tracking and weapon_type in self.ammo_tracking[unit_id]:
            return self.ammo_tracking[unit_id][weapon_type]

        # Get from unit properties with fallbacks
        weapon = self.env.get_unit_property(unit_id, f'{weapon_type}_weapon')
        if not weapon:
            return 0

        # Initialize tracking with weapon capacity
        if unit_id not in self.ammo_tracking:
            self.ammo_tracking[unit_id] = {}

        self.ammo_tracking[unit_id][weapon_type] = weapon.ammo_capacity
        return weapon.ammo_capacity

    def _reduce_unit_ammo(self, unit_id: int, weapon_type: str, amount: int) -> int:
        """
        Reduce unit's ammunition and return remaining amount.
        """
        current_ammo = self._get_unit_ammo(unit_id, weapon_type)
        new_ammo = max(0, current_ammo - amount)

        # Update tracking
        if unit_id not in self.ammo_tracking:
            self.ammo_tracking[unit_id] = {}

        self.ammo_tracking[unit_id][weapon_type] = new_ammo

        # Also update unit property if it exists
        try:
            ammo_key = f'ammo_{weapon_type}'
            if self.env.get_unit_property(unit_id, ammo_key) is not None:
                self.env.update_unit_property(unit_id, ammo_key, new_ammo)
        except:
            pass  # Property doesn't exist

        return new_ammo

    def _get_unit_at_position(self, position: Tuple[int, int]) -> Optional[int]:
        """Get unit at exact position if any."""
        x, y = position

        # Check state tensor for unit ID
        if 0 <= y < self.env.height and 0 <= x < self.env.width:
            unit_id = self.env.state_manager.state_tensor[y, x, 2]
            if unit_id > 0:
                return int(unit_id)

        return None

    def _calculate_effectiveness(self, results: Dict) -> float:
        """
        Calculate engagement effectiveness score.

        Args:
            results: Dictionary containing:
                - total_hits: Number of successful hits
                - ammo_expended: Total ammunition used
                - total_damage: Total damage dealt
                - suppression_level: Level of suppression achieved (0-1)

        Returns:
            Float effectiveness score (0-1)
        """
        # No effectiveness if no engagement occurred
        if results['ammo_expended'] == 0:
            return 0.0

        # Calculate hit ratio (accuracy component)
        hit_ratio = results['total_hits'] / results['ammo_expended']
        accuracy_score = min(1.0, hit_ratio * 2.0)  # Scale up but cap at 1.0

        # Calculate damage effectiveness
        max_expected_damage = results['ammo_expended'] * 50.0  # Assuming average max damage of 50 per round
        damage_ratio = results['total_damage'] / max_expected_damage
        damage_score = min(1.0, damage_ratio * 1.5)  # Scale up but cap at 1.0

        # Use suppression level directly (already 0-1)
        suppression_score = results['suppression_level']

        # Weight the components
        # - Accuracy is most important (40%)
        # - Damage dealt is next (35%)
        # - Suppression provides bonus effect (25%)
        effectiveness = (
                accuracy_score * 0.40 +
                damage_score * 0.35 +
                suppression_score * 0.25
        )

        return effectiveness

    def _record_engagement(self, unit_id: int, engagement_data: Dict) -> None:
        """Record engagement in history."""
        if unit_id not in self.engagement_history:
            self.engagement_history[unit_id] = []

        self.engagement_history[unit_id].append(engagement_data)

        # Limit history size
        if len(self.engagement_history[unit_id]) > 20:
            self.engagement_history[unit_id] = self.engagement_history[unit_id][-20:]


@dataclass
class EnvironmentConfig:
    """Configuration for environment initialization"""
    width: int = 20
    height: int = 20
    enable_terrain: bool = True
    enable_elevation: bool = True
    debug_level: int = 0  # 0=None, 1=Basic, 2=Detailed


class CoordinationType(Enum):
    """Types of coordination points."""
    PHASE_LINE = auto()  # General control measure
    SUPPORT_SET = auto()  # Support by fire position occupied
    ASSAULT_SET = auto()  # Assault position reached
    ATTACK_POSITION = auto()  # Final coordination before assault
    OBJECTIVE = auto()  # Final assault objective


class CoordinationLevel(Enum):
    """Level of approval required for coordination."""
    UNIT = auto()  # Unit-level coordination (fire team, squad)
    PLATOON = auto()  # Requires platoon leader approval
    COMPANY = auto()  # Requires company-level approval


@dataclass
class CoordinationPoint:
    """Defines a coordination point in the environment."""
    position: Tuple[int, int]
    coord_type: CoordinationType
    required_units: Set[int]  # Unit IDs required to coordinate
    level: CoordinationLevel
    conditions: List[str]  # Required conditions to be met
    priority: int  # Order of coordination (1 highest)

    def is_condition_met(self, unit_positions: Dict[int, Tuple[int, int]],
                         visibility_mgr: 'VisibilityManager') -> bool:
        """Check if coordination conditions are met."""
        for condition in self.conditions:
            if condition == "support_set":
                if not self._check_support_set(unit_positions, visibility_mgr):
                    return False
            elif condition == "mutual_support":
                if not self._check_mutual_support(unit_positions, visibility_mgr):
                    return False
            elif condition == "enemy_suppressed":
                if not self._check_enemy_suppression(unit_positions):
                    return False
        return True

    def _check_support_set(self, unit_positions: Dict[int, Tuple[int, int]],
                           visibility_mgr: 'VisibilityManager') -> bool:
        """Check if support element is in position with proper fields of fire."""
        for unit_id in self.required_units:
            if unit_id not in unit_positions:
                return False
            unit_pos = unit_positions[unit_id]
            # Use visibility manager to check fields of fire
            los_result = visibility_mgr.check_line_of_sight(unit_pos, self.position)
            if not los_result['has_los']:
                return False
        return True

    def _check_mutual_support(self, unit_positions: Dict[int, Tuple[int, int]],
                              visibility_mgr: 'VisibilityManager') -> bool:
        """Check if elements can provide mutual support."""
        positions = [unit_positions[unit_id] for unit_id in self.required_units
                     if unit_id in unit_positions]
        if len(positions) < 2:
            return False

        # Check for mutual fields of fire
        for i, pos1 in enumerate(positions[:-1]):
            for pos2 in positions[i + 1:]:
                los_result = visibility_mgr.check_line_of_sight(pos1, pos2)
                if not los_result['has_los']:
                    return False
        return True

    def _check_enemy_suppression(self, unit_positions: Dict[int, Tuple[int, int]]) -> bool:
        """Check if enemy positions are being effectively suppressed."""
        # This would typically integrate with the threat analysis system
        # For now, return True if required units are in position
        return all(unit_id in unit_positions for unit_id in self.required_units)


@dataclass
class CoordinationState:
    """Current state of coordination at a point."""
    position: Tuple[int, int]
    completed_units: Set[int] = None  # Units that have arrived
    conditions_met: Dict[str, bool] = None  # Status of conditions
    is_active: bool = True  # Whether point is still active

    def __post_init__(self):
        if self.completed_units is None:
            self.completed_units = set()
        if self.conditions_met is None:
            self.conditions_met = {}


class ActionType(Enum):
    """Types of actions units can take."""
    # Movement Actions
    MOVE = auto()  # Basic movement following route
    BOUND = auto()  # Bounding movement (one element moves while others provide security)
    HALT = auto()  # Stop movement

    # Combat Actions
    ENGAGE = auto()  # Engage identified enemy
    SUPPRESS = auto()  # Provide suppressing fire
    ASSAULT = auto()  # Close with and destroy enemy

    # Formation Actions
    CHANGE_FORMATION = auto()  # Change unit formation

    # Command & Control Actions
    REPORT = auto()  # Send report to higher
    COORDINATE = auto()  # Coordinate with adjacent units


@dataclass
class ActionParameters:
    """Parameters that can be passed with actions."""
    # Movement Parameters
    direction: Optional[Tuple[int, int]] = None  # Movement direction vector
    distance: Optional[int] = None  # Movement distance
    formation: Optional[str] = None  # Formation to change to

    # Combat Parameters
    target_pos: Optional[Tuple[int, int]] = None  # Position to engage/suppress
    weapon_type: Optional[str] = None  # Primary/secondary weapon selection

    # C2 Parameters
    report_type: Optional[str] = None  # Type of report to send
    coordination_point: Optional[Tuple[int, int]] = None  # Point to coordinate at


class MilitaryActionSpace:
    """Defines action spaces for different training and simulation phases."""

    def __init__(self, training_mode: bool = True):
        """
        Initialize action space based on mode.

        Args:
            training_mode: If True, use simplified action space for MARL training
        """
        self.training_mode = training_mode

        if training_mode:
            # Simplified discrete action space for MARL training
            # Focus on core combat actions for learning tactical behaviors
            self.action_space = spaces.Dict({
                'action_type': spaces.Discrete(4),  # MOVE, ENGAGE, SUPPRESS, ASSAULT
                'parameters': spaces.Dict({
                    'direction': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    'target_pos': spaces.Box(low=0, high=400, shape=(2,), dtype=np.int32),
                    'weapon_type': spaces.Discrete(2)  # Primary/Secondary
                })
            })
        else:
            # Full action space for simulation
            # Includes all possible actions and parameters
            self.action_space = spaces.Dict({
                'action_type': spaces.Discrete(len(ActionType)),
                'parameters': spaces.Dict({
                    # Movement parameters
                    'direction': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    'distance': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                    'formation': spaces.Discrete(10),  # Number of possible formations

                    # Combat parameters
                    'target_pos': spaces.Box(low=0, high=400, shape=(2,), dtype=np.int32),
                    'weapon_type': spaces.Discrete(2),

                    # C2 parameters
                    'report_type': spaces.Discrete(5),  # Different report types
                    'coordination_point': spaces.Box(low=0, high=400, shape=(2,), dtype=np.int32)
                })
            })

    def sample(self) -> Dict:
        """Sample random action from space."""
        return self.action_space.sample()

    def contains(self, action: Dict) -> bool:
        """Check if action is valid for current space."""
        return self.action_space.contains(action)

    def get_valid_actions(self, unit_state: Dict) -> List[Dict]:
        """
        Get list of valid actions based on unit's current state.

        Args:
            unit_state: Current state of unit including position, health, etc.

        Returns:
            List of valid action dictionaries
        """
        valid_actions = []

        # Basic movement always available if unit is alive
        if unit_state['health'] > 0:
            valid_actions.append({
                'action_type': ActionType.MOVE,
                'parameters': {
                    'direction': (0, 0),  # Will be set by policy
                    'distance': 0  # Will be set by policy
                }
            })

        # Combat actions available if enemies detected and weapons operational
        if unit_state.get('enemies_detected') and unit_state.get('weapons_operational'):
            valid_actions.append({
                'action_type': ActionType.ENGAGE,
                'parameters': {
                    'target_pos': None,  # Will be set by policy
                    'weapon_type': 'primary'
                }
            })

            # Suppression available if unit has automatic weapons
            if unit_state.get('has_automatic_weapons'):
                valid_actions.append({
                    'action_type': ActionType.SUPPRESS,
                    'parameters': {
                        'target_pos': None,
                        'weapon_type': 'primary'
                    }
                })

        # Formation changes available if unit is a leader
        if unit_state.get('is_leader'):
            valid_actions.append({
                'action_type': ActionType.CHANGE_FORMATION,
                'parameters': {
                    'formation': None  # Will be set by policy
                }
            })

        return valid_actions

    def convert_to_env_action(self, action_dict: Dict) -> Dict:
        """
        Convert high-level action to environment action format.

        Args:
            action_dict: High-level action dictionary

        Returns:
            Action formatted for environment execution
        """
        action_type = action_dict['action_type']
        params = action_dict['parameters']

        # Create standardized action parameters
        env_params = ActionParameters()

        # Convert based on action type
        if action_type == ActionType.MOVE:
            env_params.direction = params['direction']
            env_params.distance = params.get('distance', 1)

        elif action_type in [ActionType.ENGAGE, ActionType.SUPPRESS]:
            env_params.target_pos = params['target_pos']
            env_params.weapon_type = params.get('weapon_type', 'primary')

        elif action_type == ActionType.CHANGE_FORMATION:
            env_params.formation = params['formation']

        return {
            'action_type': action_type,
            'parameters': env_params
        }


class StateManager:
    """
    StateManager for Military Environment
    """

    def __init__(self, width: int, height: int):
        # Main state tensor: [terrain, elevation, unit_id, status]
        self.state_tensor = np.zeros((height, width, 4), dtype=np.int32)

        # Unit properties dictionary
        # unit_id -> {position, health, ammo, orientation, etc.}
        self.unit_properties = {}

        # Active unit tracking
        self.active_units: Set[int] = set()

        # Grid dimensions
        self.width = width
        self.height = height

        # Coordination tracking
        self.coordination_points: Dict[Tuple[int, int], CoordinationPoint] = {}
        self.coordination_states: Dict[Tuple[int, int], CoordinationState] = {}
        self.unit_coordination: Dict[int, Dict] = {}

    def add_unit(self, unit_id: int, properties: Dict) -> None:
        """Add new unit to state tracking"""
        # Convert role enum to value if it's not already
        if 'role' in properties and isinstance(properties['role'], Enum):
            properties['role'] = properties['role'].value

        self.unit_properties[unit_id] = properties
        self.active_units.add(unit_id)

        # Add unit ID to state tensor if position is provided
        if 'position' in properties:
            x, y = properties['position']
            self.state_tensor[y, x, 2] = unit_id

    def get_unit_position(self, unit_id: int) -> Tuple[int, int]:
        """Get current position of unit"""
        if unit_id not in self.unit_properties:
            raise ValueError(f"Unit {unit_id} not found")
        return self.unit_properties[unit_id]['position']

    def update_unit_position(self, unit_id: int, new_pos: Tuple[int, int]) -> None:
        """Update unit position in both properties and state tensor"""
        if unit_id not in self.unit_properties:
            raise ValueError(f"Unit {unit_id} not found")

        # Clear old position
        old_pos = self.unit_properties[unit_id]['position']
        self.state_tensor[old_pos[1], old_pos[0], 2] = 0

        # Update new position
        self.unit_properties[unit_id]['position'] = new_pos
        self.state_tensor[new_pos[1], new_pos[0], 2] = unit_id

    def get_unit_property(self, unit_id: int, property_name: str, default_value=None):
        """
        Get specific unit property with optional default value.

        Args:
            unit_id: ID of unit
            property_name: Name of property to get
            default_value: Value to return if property doesn't exist

        Returns:
            Property value or default if not found
        """
        if unit_id not in self.unit_properties:
            raise ValueError(f"Unit {unit_id} not found")

        return self.unit_properties[unit_id].get(property_name, default_value)

    def update_unit_property(self, unit_id: int, property_name: str, value) -> None:
        """Update specific unit property"""
        if unit_id not in self.unit_properties:
            raise ValueError(f"Unit {unit_id} not found")

        if property_name == 'role' and isinstance(value, Enum):
            value = value.value

        self.unit_properties[unit_id][property_name] = value

    def register_coordination_point(self, coord_point: CoordinationPoint) -> None:
        """Register a new coordination point."""
        self.coordination_points[coord_point.position] = coord_point
        self.coordination_states[coord_point.position] = CoordinationState(
            position=coord_point.position
        )

        # Initialize unit coordination tracking
        for unit_id in coord_point.required_units:
            if unit_id not in self.unit_coordination:
                self.unit_coordination[unit_id] = {
                    'current_coord_point': None,
                    'awaiting_approval': False,
                    'completed_points': set(),
                    'current_route_segment': 0
                }

            # Add coordination properties to unit
            self.update_unit_property(unit_id, 'at_coordination_point', False)
            self.update_unit_property(unit_id, 'awaiting_approval', False)
            self.update_unit_property(unit_id, 'current_route_segment', 0)

    def update_unit_at_coordination(self,
                                    unit_id: int,
                                    position: Tuple[int, int],
                                    at_point: bool = True) -> bool:
        """
        Update unit's coordination status.
        Returns True if coordination is complete at point.
        """
        if position not in self.coordination_points:
            return False

        coord_point = self.coordination_points[position]
        coord_state = self.coordination_states[position]

        if unit_id not in coord_point.required_units:
            return False

        # Update unit state
        self.update_unit_property(unit_id, 'at_coordination_point', at_point)
        self.update_unit_property(unit_id, 'awaiting_approval',
                                  coord_point.level != CoordinationLevel.UNIT)

        # Update coordination tracking
        if at_point:
            coord_state.completed_units.add(unit_id)
            self.unit_coordination[unit_id]['current_coord_point'] = position
            self.unit_coordination[unit_id]['awaiting_approval'] = (
                    coord_point.level != CoordinationLevel.UNIT
            )
        else:
            coord_state.completed_units.discard(unit_id)
            self.unit_coordination[unit_id]['current_coord_point'] = None
            self.unit_coordination[unit_id]['awaiting_approval'] = False

        # Check if coordination is complete
        return len(coord_state.completed_units) == len(coord_point.required_units)

    def update_coordination_condition(self,
                                      position: Tuple[int, int],
                                      condition: str,
                                      is_met: bool) -> None:
        """Update status of a coordination condition."""
        if position in self.coordination_states:
            self.coordination_states[position].conditions_met[condition] = is_met

    def check_coordination_complete(self, position: Tuple[int, int]) -> bool:
        """Check if coordination is complete at position."""
        if position not in self.coordination_points:
            return False

        coord_point = self.coordination_points[position]
        coord_state = self.coordination_states[position]

        # Check if all units present
        units_complete = (len(coord_state.completed_units) ==
                          len(coord_point.required_units))

        # Check if all conditions met
        conditions_complete = all(coord_state.conditions_met.values())

        return units_complete and conditions_complete

    def clear_coordination_point(self, position: Tuple[int, int]) -> None:
        """Clear coordination after point is complete."""
        if position not in self.coordination_states:
            return

        coord_state = self.coordination_states[position]
        coord_point = self.coordination_points[position]

        # Update all units at point
        for unit_id in coord_state.completed_units:
            # Update unit properties
            self.update_unit_property(unit_id, 'at_coordination_point', False)
            self.update_unit_property(unit_id, 'awaiting_approval', False)

            # Increment route segment
            current_segment = self.unit_coordination[unit_id]['current_route_segment']
            self.unit_coordination[unit_id]['current_route_segment'] = current_segment + 1
            self.update_unit_property(unit_id, 'current_route_segment', current_segment + 1)

            # Update coordination tracking
            self.unit_coordination[unit_id]['completed_points'].add(position)
            self.unit_coordination[unit_id]['current_coord_point'] = None
            self.unit_coordination[unit_id]['awaiting_approval'] = False

        # Mark coordination point as inactive
        coord_state.is_active = False
        coord_state.completed_units.clear()
        coord_state.conditions_met.clear()

    def get_coordination_state(self, position: Tuple[int, int]) -> Dict:
        """Get complete state of coordination at position."""
        if position not in self.coordination_points:
            return {}

        coord_point = self.coordination_points[position]
        coord_state = self.coordination_states[position]

        return {
            'type': coord_point.coord_type,
            'level': coord_point.level,
            'required_units': coord_point.required_units,
            'completed_units': coord_state.completed_units,
            'conditions_met': coord_state.conditions_met,
            'is_active': coord_state.is_active,
            'all_units_present': (len(coord_state.completed_units) ==
                                  len(coord_point.required_units)),
            'all_conditions_met': all(coord_state.conditions_met.values())
        }

    def get_unit_coordination_state(self, unit_id: int) -> Dict:
        """Get unit's current coordination state."""
        if unit_id not in self.unit_coordination:
            return {}

        coord_state = self.unit_coordination[unit_id]
        current_point = coord_state['current_coord_point']

        return {
            'at_coordination_point': current_point is not None,
            'current_point': current_point,
            'awaiting_approval': coord_state['awaiting_approval'],
            'completed_points': coord_state['completed_points'],
            'current_route_segment': coord_state['current_route_segment']
        }


class TerrainManager:
    """
    TerrainManager for Military Environment
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Reference to state tensor will be set by initialize_terrain
        self.state_tensor = None

        # Initialize terrain effects with default values
        self.terrain_effects = {
            TerrainType.BARE: {
                'movement_cost': 1.0,
                'visibility': 1.0,
                'cover': 0.0
            },
            TerrainType.SPARSE_VEG: {
                'movement_cost': 1.2,
                'visibility': 0.8,
                'cover': 0.2
            },
            TerrainType.DENSE_VEG: {
                'movement_cost': 1.5,
                'visibility': 0.6,
                'cover': 0.4
            },
            TerrainType.WOODS: {
                'movement_cost': 2.0,
                'visibility': 0.4,
                'cover': 0.6
            },
            TerrainType.STRUCTURE: {
                'movement_cost': 1.0,
                'visibility': 0.2,
                'cover': 0.8
            }
        }

    def initialize_terrain(self, state_tensor: np.ndarray) -> None:
        """Initialize terrain and elevation in state tensor"""
        self.state_tensor = state_tensor
        mid_x, mid_y = self.width // 2, self.height // 2

        for y in range(self.height):
            for x in range(self.width):
                if x < mid_x:
                    if y < mid_y:
                        terrain = TerrainType.BARE.value
                        elevation = ElevationType.GROUND_LEVEL.value
                    else:
                        terrain = TerrainType.WOODS.value
                        elevation = ElevationType.GROUND_LEVEL.value
                else:
                    if y < mid_y:
                        terrain = TerrainType.SPARSE_VEG.value
                        elevation = ElevationType.ELEVATED_LEVEL.value
                    else:
                        terrain = TerrainType.STRUCTURE.value
                        elevation = ElevationType.LOWER_LEVEL.value

                self.state_tensor[y, x, 0] = terrain
                self.state_tensor[y, x, 1] = elevation

    def load_from_csv(self, csv_path: str) -> None:
        """Load terrain data from CSV file."""
        if self.state_tensor is None:
            raise ValueError("State tensor not initialized. Call initialize_terrain first.")

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = int(row.get('x', 0))
                y = int(row.get('y', 0))

                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue

                # Update state tensor
                self.state_tensor[y, x, 0] = TerrainType[row.get('terrain_type', 'BARE')].value
                self.state_tensor[y, x, 1] = ElevationType[row.get('elevation_type', 'GROUND_LEVEL')].value

                # Update effects if provided
                terrain_type = TerrainType[row.get('terrain_type', 'BARE')]
                if all(key in row for key in ['movement_cost', 'visibility_factor', 'cover_bonus']):
                    self.terrain_effects[terrain_type] = {
                        'movement_cost': float(row['movement_cost']),
                        'visibility': float(row['visibility_factor']),
                        'cover': float(row['cover_bonus'])
                    }

    def get_movement_cost(self, position: Tuple[int, int]) -> float:
        """Get movement cost for a position."""
        x, y = position
        terrain_type = TerrainType(self.state_tensor[y, x, 0])
        return self.terrain_effects[terrain_type]['movement_cost']

    def get_visibility(self, position: Tuple[int, int]) -> float:
        """Get visibility factor for a position."""
        x, y = position
        terrain_type = TerrainType(self.state_tensor[y, x, 0])
        return self.terrain_effects[terrain_type]['visibility']

    def get_cover(self, position: Tuple[int, int]) -> float:
        """Get cover value for a position."""
        x, y = position
        terrain_type = TerrainType(self.state_tensor[y, x, 0])
        return self.terrain_effects[terrain_type]['cover']

    def get_terrain_type(self, position: Tuple[int, int]) -> TerrainType:
        """Get terrain type at position."""
        x, y = position
        return TerrainType(self.state_tensor[y, x, 0])

    def get_elevation_type(self, position: Tuple[int, int]) -> ElevationType:
        """Get elevation type at position."""
        x, y = position
        return ElevationType(self.state_tensor[y, x, 1])


class VisibilityManager:
    """
    VisibilityManager for Military Environment
    """

    def __init__(self, env: 'MilitaryEnvironment'):
        """Initialize with environment reference."""
        self.env = env
        self.width = env.width
        self.height = env.height
        self.terrain_manager = env.terrain_manager

        # Elevation effects
        self.elevation_observation_bonus = 0.1  # 10% observation range boost when higher
        self.elevation_cover_penalty = 0.1  # 10% cover reduction when engaged from higher ground

        # Terrain degradation factors per cell (normalized by total path length)
        self.terrain_degradation = {
            TerrainType.BARE: 0.0,  # No degradation
            TerrainType.SPARSE_VEG: 0.05,  # 5% per cell
            TerrainType.DENSE_VEG: 0.15,  # 15% per cell
            TerrainType.WOODS: 0.25,  # 25% per cell
            TerrainType.STRUCTURE: 1.0  # Complete blockage
        }

        # Track units that have fired
        self.fired_units: Set[int] = set()

    def has_elevation_advantage(self, position: Tuple[int, int], other_position: Tuple[int, int]) -> bool:
        """
        Check if position has elevation advantage over other position.

        Args:
            position: Position to check from
            other_position: Position to check against

        Returns:
            True if position has elevation advantage
        """
        pos_elevation = self.terrain_manager.get_elevation_type(position)
        other_elevation = self.terrain_manager.get_elevation_type(other_position)
        return pos_elevation.value > other_elevation.value

    def get_cover_bonus(self, position: Tuple[int, int], shooter_pos: Tuple[int, int] = None) -> float:
        """
        Get cover bonus (damage reduction) at position.

        Args:
            position: Position to check
            shooter_pos: Optional position of shooting unit for elevation check

        Returns:
            Cover bonus (0-1) where higher means more protection
        """
        # Get base cover from TerrainManager
        cover = self.terrain_manager.get_cover(position)

        # Apply elevation penalty if being shot at from higher ground
        if shooter_pos and self.has_elevation_advantage(shooter_pos, position):
            cover = max(0.0, cover - self.elevation_cover_penalty)

        return cover

    def calculate_observation_range(self, base_range: int, position: Tuple[int, int],
                                    target_area: Tuple[int, int] = None) -> int:
        """
        Calculate actual observation range considering terrain and elevation effects.

        Args:
            base_range: Base observation range in cells
            position: Unit's position
            target_area: Optional target area to check elevation advantage against

        Returns:
            Modified observation range in cells
        """
        # Get visibility from TerrainManager
        visibility = self.terrain_manager.get_visibility(position)

        # Apply elevation bonus if observing from higher ground
        if target_area and self.has_elevation_advantage(position, target_area):
            range_multiplier = 1.0 + self.elevation_observation_bonus
        else:
            range_multiplier = 1.0

        # Reduce range based on terrain and apply elevation multiplier
        return int(base_range * visibility * range_multiplier)

    def modify_hit_probability(self,
                               base_probability: float,
                               shooter_pos: Tuple[int, int],
                               target_pos: Tuple[int, int],
                               target_id: int) -> float:
        """
        Modify hit probability based on terrain and elevation effects.

        Args:
            base_probability: Base hit probability (0-1)
            shooter_pos: Position of shooting unit
            target_pos: Position of target
            target_id: ID of target unit

        Returns:
            Modified hit probability (0-1)
        """
        # Get shooter's field of view penalty
        direction = (
            target_pos[0] - shooter_pos[0],
            target_pos[1] - shooter_pos[1]
        )
        fov_penalty = self.get_field_of_view_penalty(shooter_pos, direction)

        # Get target's concealment bonus
        concealment = self.get_concealment_bonus(target_pos, target_id)

        # Calculate final probability
        modified_probability = base_probability * (1 - fov_penalty) * (1 - concealment)

        # Add small bonus for elevation advantage
        if self.has_elevation_advantage(shooter_pos, target_pos):
            modified_probability = min(1.0, modified_probability * (1.0 + self.elevation_observation_bonus))

        return modified_probability

    def modify_damage(self,
                      base_damage: float,
                      target_pos: Tuple[int, int],
                      shooter_pos: Tuple[int, int] = None) -> float:
        """
        Modify damage based on cover and elevation effects.

        Args:
            base_damage: Base damage amount
            target_pos: Position of target
            shooter_pos: Optional position of shooter for elevation check

        Returns:
            Modified damage amount
        """
        # Get cover with elevation consideration
        cover = self.get_cover_bonus(target_pos, shooter_pos)
        return base_damage * (1 - cover)

    def check_line_of_sight(self,
                            start: Tuple[int, int],
                            end: Tuple[int, int],
                            for_observation: bool = True) -> Dict:
        """
        Check line of sight between positions with enhanced terrain analysis.

        Args:
            start: Starting position
            end: Ending position
            for_observation: If True, use observation values, else engagement

        Returns:
            Dictionary containing:
            - has_los: Whether line of sight exists
            - los_quality: Quality of line of sight (0-1)
            - degradation: List of terrain causing degradation with distances
            - elevation_advantage: True if start has elevation advantage
        """
        # Get points along line using Bresenham's algorithm
        points = self._get_line_points(start[0], start[1], end[0], end[1])
        path_length = len(points)

        # Initialize tracking
        degradation_by_terrain = defaultdict(int)  # Track cells of each terrain type
        degradation_details = []
        total_degradation = 0.0

        # Get starting terrain for field of view
        start_visibility = self.terrain_manager.get_visibility(start)
        fov_penalty = 1.0 - start_visibility

        # Check elevation advantage
        has_elevation = self.has_elevation_advantage(start, end)
        if has_elevation:
            # Reduce degradation when observing from higher ground
            fov_penalty *= (1.0 - self.elevation_observation_bonus)

        # Analyze each point along the line
        current_terrain_run = None
        terrain_start_idx = 0

        for i, (x, y) in enumerate(points[1:-1]):  # Skip start and end points
            terrain_type = self.terrain_manager.get_terrain_type((x, y))

            # Check for complete blockage
            if terrain_type == TerrainType.STRUCTURE:
                return {
                    'has_los': False,
                    'los_quality': 0.0,
                    'degradation': ['Structure blocks line of sight'],
                    'elevation_advantage': has_elevation
                }

            # Track terrain runs
            if terrain_type != current_terrain_run:
                if current_terrain_run:
                    # Record previous terrain run
                    run_length = i - terrain_start_idx
                    degradation_by_terrain[current_terrain_run] += run_length
                    if run_length > 1:  # Only record significant runs
                        degradation_details.append(
                            f"{current_terrain_run.name} for {run_length} cells"
                        )
                current_terrain_run = terrain_type
                terrain_start_idx = i

            # Calculate degradation for this cell
            cell_degradation = self.terrain_degradation[terrain_type]
            if has_elevation:
                # Reduce terrain degradation when viewing from higher ground
                cell_degradation *= (1.0 - self.elevation_observation_bonus)

            total_degradation += cell_degradation

        # Record final terrain run
        if current_terrain_run:
            run_length = len(points) - 1 - terrain_start_idx
            degradation_by_terrain[current_terrain_run] += run_length
            if run_length > 1:
                degradation_details.append(
                    f"{current_terrain_run.name} for {run_length} cells"
                )

        # Normalize degradation by path length
        total_degradation = total_degradation / path_length

        # Apply field of view penalty from start position
        total_degradation += fov_penalty

        # Check if degradation exceeds threshold
        max_degradation = 0.8 if for_observation else 0.9
        if total_degradation >= max_degradation:
            return {
                'has_los': False,
                'los_quality': 0.0,
                'degradation': degradation_details,
                'elevation_advantage': has_elevation,
                'degradation_details': {
                    'total': total_degradation,
                    'by_terrain': dict(degradation_by_terrain),
                    'path_length': path_length,
                    'fov_penalty': fov_penalty
                }
            }

        # Calculate final LOS quality
        los_quality = max(0.0, 1.0 - total_degradation)

        # Apply elevation bonus if applicable
        if has_elevation:
            los_quality = min(1.0, los_quality * (1.0 + self.elevation_observation_bonus))

        return {
            'has_los': True,
            'los_quality': los_quality,
            'degradation': degradation_details,
            'elevation_advantage': has_elevation,
            'degradation_details': {
                'total': total_degradation,
                'by_terrain': dict(degradation_by_terrain),
                'path_length': path_length,
                'fov_penalty': fov_penalty
            }
        }

    def get_concealment_bonus(self, position: Tuple[int, int], unit_id: int) -> float:
        """
        Get concealment bonus at position for unit.

        Args:
            position: Position to check
            unit_id: ID of unit to check (for fired status)

        Returns:
            Concealment bonus (0-1) where higher means better concealed
        """
        # Get base visibility from TerrainManager
        visibility = self.terrain_manager.get_visibility(position)
        base_concealment = 1.0 - visibility

        # Reduce concealment if unit has fired
        if unit_id in self.fired_units:
            base_concealment *= 0.2  # 80% reduction

        return base_concealment

    def get_field_of_view_penalty(self, position: Tuple[int, int], direction: Tuple[int, int] = None) -> float:
        """
        Get field of view penalty at position.

        Args:
            position: Position to check
            direction: Optional direction to check (for edge of terrain check)

        Returns:
            Field of view penalty (0-1) where higher means more restricted
        """
        x, y = position
        # Get base visibility from TerrainManager
        visibility = self.terrain_manager.get_visibility(position)
        base_penalty = 1.0 - visibility

        # Check adjacent terrain if direction specified
        if direction:
            dx, dy = direction
            adj_x, adj_y = x + dx, y + dy

            # Check bounds
            if 0 <= adj_x < self.width and 0 <= adj_y < self.height:
                adj_visibility = self.terrain_manager.get_visibility((adj_x, adj_y))

                # If adjacent terrain has better visibility, reduce penalty
                if adj_visibility > visibility:
                    base_penalty *= 0.2  # 80% reduction

        return base_penalty

    def record_unit_fired(self, unit_id: int):
        """Record that a unit has fired (reducing concealment)."""
        self.fired_units.add(unit_id)

    def reset_fired_units(self):
        """Reset the fired units tracking."""
        self.fired_units.clear()

    def _get_line_points(self,
                         x1: int, y1: int,
                         x2: int, y2: int) -> List[Tuple[int, int]]:
        """Get points along line using Bresenham's algorithm."""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1

        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        err = dx - dy

        while True:
            points.append((x, y))
            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points


class MilitaryEnvironment(gym.Env):
    """
    MilitaryEnvironment Class
    """
    metadata = {'render.modes': ['human']}

    # Initialization and Core Environment Functions
    def __init__(self, config: EnvironmentConfig):
        """Initialize environment with configuration."""
        super().__init__()

        # Initialize dimensions
        self.config = config
        self.width = config.width
        self.height = config.height
        self.debug_level = config.debug_level
        self.current_step = 0
        self.max_steps = 500  # Default maximum episode length
        self.casualty_threshold = 3  # Default mission failure threshold

        # Initialize managers
        self.state_manager = StateManager(self.width, self.height)
        self.terrain_manager = TerrainManager(self.width, self.height)
        self.visibility_manager = VisibilityManager(self)
        self.combat_manager = CombatManager(self)

        # Define action space
        # [unit_id, action_type, params...]
        self.action_space = spaces.Dict({
            'unit_id': spaces.Discrete(10000),  # Maximum unit ID
            'action_type': spaces.Discrete(len(ActionType)),  # Number of possible actions
            'parameters': spaces.Dict({
                # Movement parameters
                'direction': spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),
                'distance': spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
                'formation': spaces.Discrete(10),  # Number of possible formations

                # Combat parameters
                'target_pos': spaces.Box(
                    low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32
                ),
                'weapon_type': spaces.Discrete(2),  # Primary/Secondary
                'suppress_only': spaces.Discrete(2),  # Boolean
                'max_rounds': spaces.Box(
                    low=1, high=100, shape=(1,), dtype=np.int32
                ),
                'rate_of_fire': spaces.Box(
                    low=0.1, high=2.0, shape=(1,), dtype=np.float32
                ),

                # C2 parameters
                'report_type': spaces.Discrete(5),  # Different report types
                'coordination_point': spaces.Box(
                    low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32
                )
            })
        })

        # Define observation space
        self.observation_space = spaces.Dict({
            # Main state tensor
            'state': spaces.Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(self.height, self.width, 4),
                dtype=np.int32
            ),
            # Unit properties for active units
            'units': spaces.Dict({
                'positions': spaces.Box(
                    low=0,
                    high=max(self.width, self.height),
                    shape=(100, 2),  # Maximum 100 units, (x,y) positions
                    dtype=np.int32
                ),
                'properties': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(100, 15),  # Maximum 100 units, 15 properties each
                    dtype=np.float32
                )
            }),
            # Combat state information
            'combat_state': spaces.Dict({
                'suppressed_units': spaces.Box(
                    low=0,
                    high=1,
                    shape=(100,),  # Suppression level per unit
                    dtype=np.float32
                ),
                'ammunition': spaces.Box(
                    low=0,
                    high=1000,
                    shape=(100, 2),  # Ammo levels for primary/secondary weapons
                    dtype=np.int32
                ),
                'team_effectiveness': spaces.Box(
                    low=0,
                    high=1,
                    shape=(20,),  # Combat effectiveness for up to 20 teams
                    dtype=np.float32
                )
            })
        })

        # Initialize ID counters
        self._next_unit_id = 1

        if self.debug_level > 0:
            print(f"Initialized {self.width}x{self.height} environment")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0

        # Clear state tracking
        self.state_manager = StateManager(self.width, self.height)

        # Initialize terrain if enabled
        self.terrain_manager.initialize_terrain(self.state_manager.state_tensor)

        # Load terrain data if provided in options
        if options and 'terrain_file' in options:
            try:
                self.terrain_manager.load_from_csv(options['terrain_file'])
            except Exception as e:
                print(f"Warning: Failed to load terrain: {e}")

        # Reset combat and visibility managers
        self.visibility_manager.reset_fired_units()
        self.combat_manager.reset()

        # Initialize units if provided in options
        if options and 'unit_init' in options:
            self._initialize_units(options['unit_init'])

        # Get initial observation
        observation = self._get_observation()

        if self.debug_level > 0:
            print("Environment reset")
            if self.debug_level > 1:
                print("Initial observation:", observation)

        return observation, {}

    def step(self, action: Dict):
        """
        Execute action and return new state.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit taking action
                - action_type: Type of action to take
                - parameters: Additional parameters for action

        Returns:
            observation: Current environment state
            reward: Reward for action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Increment step counter
        self.current_step += 1

        # Validate action
        if not self._validate_action(action):
            return self._get_observation(), -1, False, False, {"error": "Invalid action"}

        # Execute action
        reward = self._execute_action(action)

        # Update ongoing effects
        self.combat_manager.update_suppression_states()

        # Get updated observation
        observation = self._get_observation()

        # Check if episode is complete
        terminated = self._check_termination()
        truncated = self._check_truncation()

        # Compile info dict
        info = self._get_step_info()

        if self.debug_level > 1:
            print(f"Step {self.current_step} complete - Reward: {reward}")

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict:
        """Get current environment state as observation."""
        # Get positions and properties of active units
        active_units = list(self.state_manager.active_units)
        num_units = len(active_units)

        # Initialize arrays
        positions = np.zeros((100, 2), dtype=np.int32)
        properties = np.zeros((100, 15), dtype=np.float32)
        suppressed_units = np.zeros(100, dtype=np.float32)
        ammunition = np.zeros((100, 2), dtype=np.int32)
        team_effectiveness = np.zeros(20, dtype=np.float32)

        # Fill arrays with active unit data
        for i, unit_id in enumerate(active_units[:100]):  # Limit to 100 units
            # Get position
            pos = self.state_manager.get_unit_position(unit_id)
            positions[i] = pos

            # Get properties
            unit_props = self.state_manager.unit_properties[unit_id]
            properties[i] = self._format_unit_properties(unit_props)

            # Get combat state
            combat_state = self.combat_manager.get_unit_combat_state(unit_id)
            suppressed_units[i] = combat_state['suppression_level']
            ammunition[i, 0] = combat_state['ammo_primary']
            ammunition[i, 1] = combat_state['ammo_secondary']

        # Get team effectiveness for teams
        team_ids = [unit_id for unit_id in active_units if
                    self.get_unit_property(unit_id, 'type') in
                    [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM]]

        for i, team_id in enumerate(team_ids[:20]):  # Limit to 20 teams
            team_effectiveness[i] = self.combat_manager.get_team_combat_effectiveness(team_id)

        return {
            'state': self.state_manager.state_tensor,
            'units': {
                'positions': positions,
                'properties': properties
            },
            'combat_state': {
                'suppressed_units': suppressed_units,
                'ammunition': ammunition,
                'team_effectiveness': team_effectiveness
            }
        }

    def _validate_action(self, action: Dict) -> bool:
        """Validate action dictionary."""
        # Check required keys
        if not all(key in action for key in ['unit_id', 'action_type', 'parameters']):
            return False

        # Check unit exists
        if action['unit_id'] not in self.state_manager.active_units:
            return False

        # Check if unit can take action (not dead, etc.)
        health = self.get_unit_property(action['unit_id'], 'health', 0)
        if health <= 0:
            return False

        # Check unit suppression for certain actions
        if action['action_type'] in [ActionType.ENGAGE, ActionType.SUPPRESS]:
            combat_state = self.combat_manager.get_unit_combat_state(action['unit_id'])
            if combat_state['suppression_level'] > 0.8:
                # Unit too suppressed to engage
                return False

        # Check ammunition for engagement actions
        if action['action_type'] in [ActionType.ENGAGE, ActionType.SUPPRESS]:
            combat_state = self.combat_manager.get_unit_combat_state(action['unit_id'])
            if combat_state['ammo_primary'] <= 0:
                # No ammunition
                return False

        # Validate parameters for action type
        parameters = action['parameters']
        if action['action_type'] == ActionType.MOVE and 'direction' not in parameters:
            return False
        elif action['action_type'] in [ActionType.ENGAGE, ActionType.SUPPRESS] and 'target_pos' not in parameters:
            return False
        elif action['action_type'] == ActionType.CHANGE_FORMATION and 'formation' not in parameters:
            return False

        return True

    def _execute_action(self, action: Dict) -> float:
        """
        Execute action and return reward.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit taking action
                - action_type: Type of action to take
                - parameters: Additional parameters for action

        Returns:
            Reward value for the action
        """
        # Validate action
        if not self._validate_action(action):
            return -1.0  # Penalty for invalid action

        action_type = action['action_type']

        # Route to specific action handler
        if action_type == ActionType.MOVE:
            return self._execute_movement_action(action)
        elif action_type == ActionType.ENGAGE:
            return self._execute_engagement_action(action)
        elif action_type == ActionType.SUPPRESS:
            return self._execute_suppression_action(action)
        elif action_type == ActionType.BOUND:
            return self._execute_bounding_action(action)
        elif action_type == ActionType.CHANGE_FORMATION:
            return self._execute_formation_action(action)
        elif action_type == ActionType.REPORT:
            return self._execute_report_action(action)
        elif action_type == ActionType.COORDINATE:
            return self._execute_coordinate_action(action)
        elif action_type == ActionType.HALT:
            return self._execute_halt_action(action)
        else:
            if self.debug_level > 0:
                print(f"Warning: Unhandled action type {action_type}")
            return 0.0

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Mission success condition - objective secured
        if self._check_objective_secured():
            return True

        # Mission failure - casualties exceed threshold
        friendly_casualties = self._count_casualties(ForceType.FRIENDLY)
        if friendly_casualties >= self.casualty_threshold:
            return True

        # All enemies eliminated
        enemy_casualties = self._count_casualties(ForceType.ENEMY)
        total_enemies = sum(1 for unit_id in self.state_manager.active_units
                            if self.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY)
        if enemy_casualties >= total_enemies and total_enemies > 0:
            return True

        return False

    def _check_truncation(self) -> bool:
        """Check if episode should be truncated."""
        # Time limit exceeded
        if self.current_step >= self.max_steps:
            return True

        # Lost contact with enemy
        if self._lost_enemy_contact():
            return True

        # Ammunition exhausted
        if self._check_ammunition_exhausted():
            return True

        # LLM decision required - complex state transition
        if self._check_llm_trigger_conditions():
            return True

        return False

    def _get_step_info(self) -> Dict:
        """Get additional info about current step."""
        # Active units
        friendly_units = [unit_id for unit_id in self.state_manager.active_units
                          if self.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY]
        enemy_units = [unit_id for unit_id in self.state_manager.active_units
                       if self.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY]

        # Casualties
        friendly_casualties = self._count_casualties(ForceType.FRIENDLY)
        enemy_casualties = self._count_casualties(ForceType.ENEMY)

        # Combat state
        suppressed_friendly = sum(1 for unit_id in friendly_units
                                  if self.combat_manager.get_unit_combat_state(unit_id)['suppressed'])

        # Team effectiveness
        team_ids = [unit_id for unit_id in friendly_units if
                    self.get_unit_property(unit_id, 'type') in
                    [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM]]
        avg_team_effectiveness = 0.0
        if team_ids:
            team_eff_values = [self.combat_manager.get_team_combat_effectiveness(team_id)
                               for team_id in team_ids]
            avg_team_effectiveness = sum(team_eff_values) / len(team_eff_values)

        return {
            'step': self.current_step,
            'friendly_units': len(friendly_units),
            'enemy_units': len(enemy_units),
            'friendly_casualties': friendly_casualties,
            'enemy_casualties': enemy_casualties,
            'suppressed_friendly': suppressed_friendly,
            'avg_team_effectiveness': avg_team_effectiveness,
            'termination_reason': self._get_termination_reason() if self._check_termination() else None,
            'truncation_reason': self._get_truncation_reason() if self._check_truncation() else None
        }

    # Action Execution Functions
    def _execute_movement_action(self, action: Dict) -> float:
        """
        Handle unit movement action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit to move
                - parameters:
                    - direction: (dx, dy) movement vector
                    - distance: Movement distance

        Returns:
            Reward for movement action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract movement parameters
        direction = parameters.direction if hasattr(parameters, 'direction') else (0, 0)
        distance = parameters.distance if hasattr(parameters, 'distance') else 1

        # Get unit's initial position
        start_pos = self.get_unit_position(unit_id)

        # Get unit type to determine movement function
        unit_type = self.get_unit_property(unit_id, 'type')

        # Execute appropriate movement based on unit type
        if unit_type == UnitType.INFANTRY_TEAM or unit_type == UnitType.WEAPONS_TEAM:
            from US_Army_PLT_Composition_vTest import US_IN_execute_movement
            frames = US_IN_execute_movement(
                env=self,
                unit_id=unit_id,
                direction=direction,
                distance=distance,
                debug_level=self.debug_level
            )
        elif unit_type == UnitType.INFANTRY_SQUAD:
            from US_Army_PLT_Composition_vTest import execute_squad_movement
            frames = execute_squad_movement(
                env=self,
                squad_id=unit_id,
                direction=direction,
                distance=distance,
                debug_level=self.debug_level
            )
        else:
            # Default movement for other unit types
            self.move_unit(unit_id, direction, distance)
            frames = []

        # Get final position after movement
        end_pos = self.get_unit_position(unit_id)

        # Calculate reward based on movement
        reward = self._calculate_movement_reward(start_pos, end_pos)

        return reward

    def _execute_engagement_action(self, action: Dict) -> float:
        """
        Handle direct engagement/fire action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit engaging
                - parameters:
                    - target_pos: Position to engage
                    - weapon_type: Weapon to use
                    - max_rounds: Maximum rounds to expend

        Returns:
            Reward for engagement action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract engagement parameters
        target_pos = parameters.target_pos
        weapon_type = parameters.weapon_type if hasattr(parameters, 'weapon_type') else 'primary'
        max_rounds = parameters.max_rounds if hasattr(parameters, 'max_rounds') else 10

        # Create fire control object
        fire_control = FireControl(
            target_area=target_pos,
            max_rounds=max_rounds,
            rate_of_fire=1.0,
            suppress_only=False,
        )

        # Get unit type to determine engagement level
        unit_type = self.get_unit_property(unit_id, 'type')

        # Execute appropriate engagement based on unit type
        if unit_type == UnitType.INFANTRY_TEAM or unit_type == UnitType.WEAPONS_TEAM:
            results = self.combat_manager.execute_team_engagement(
                team_id=unit_id,
                target_pos=target_pos,
                engagement_type=EngagementType.POINT
            )
        elif unit_type == UnitType.INFANTRY_SQUAD:
            results = self.combat_manager.execute_squad_engagement(
                squad_id=unit_id,
                target_area=target_pos,
                engagement_type=EngagementType.POINT
            )
        else:
            # Individual unit engagement
            results = self.combat_manager.execute_engagement(unit_id, fire_control)

        # Calculate reward based on engagement results
        reward = self._calculate_engagement_reward(results)

        return reward

    def _execute_suppression_action(self, action: Dict) -> float:
        """
        Handle suppressive/area fire action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit providing suppression
                - parameters:
                    - target_pos: Position to suppress
                    - area_radius: Area of effect
                    - max_rounds: Maximum rounds to expend
                    - rate_of_fire: Rate of fire multiplier

        Returns:
            Reward for suppression action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract suppression parameters
        target_pos = parameters.target_pos
        area_radius = parameters.area_radius if hasattr(parameters, 'area_radius') else 3
        max_rounds = parameters.max_rounds if hasattr(parameters, 'max_rounds') else 30
        rate_of_fire = parameters.rate_of_fire if hasattr(parameters, 'rate_of_fire') else 1.5

        # Create fire control object
        fire_control = FireControl(
            target_area=target_pos,
            area_radius=area_radius,
            max_rounds=max_rounds,
            rate_of_fire=rate_of_fire,
            time_limit=8,
            suppress_only=True
        )

        # Get unit type to determine suppression level
        unit_type = self.get_unit_property(unit_id, 'type')

        # Execute appropriate suppression based on unit type
        if unit_type == UnitType.INFANTRY_TEAM:
            results = self.combat_manager.execute_team_engagement(
                team_id=unit_id,
                target_pos=target_pos,
                engagement_type=EngagementType.AREA
            )
        elif unit_type == UnitType.WEAPONS_TEAM:
            # Weapons teams are more effective at sustained suppression
            results = self.combat_manager.execute_team_engagement(
                team_id=unit_id,
                target_pos=target_pos,
                engagement_type=EngagementType.SUSTAINED
            )
        elif unit_type == UnitType.INFANTRY_SQUAD:
            results = self.combat_manager.execute_squad_engagement(
                squad_id=unit_id,
                target_area=target_pos,
                engagement_type=EngagementType.AREA
            )
        else:
            # Individual unit suppression
            results = self.combat_manager.execute_engagement(unit_id, fire_control)

        # Calculate reward based on suppression results
        reward = self._calculate_suppression_reward(results)

        return reward

    def _execute_bounding_action(self, action: Dict) -> float:
        """
        Handle bounding movement (one element moves while others provide security).

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit executing bounding movement
                - parameters:
                    - direction: Movement direction
                    - distance: Movement distance
                    - bounding_elements: Which elements bound first

        Returns:
            Reward for bounding action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract bounding parameters
        direction = parameters.direction
        distance = parameters.distance if hasattr(parameters, 'distance') else 5
        bounding_elements = parameters.bounding_elements if hasattr(parameters, 'bounding_elements') else None

        # Get unit's initial position
        start_pos = self.get_unit_position(unit_id)

        # Get unit type
        unit_type = self.get_unit_property(unit_id, 'type')

        # Execute appropriate bounding movement
        if unit_type == UnitType.INFANTRY_SQUAD:
            from US_Army_PLT_Composition_vTest import execute_squad_movement, MovementTechnique
            frames = execute_squad_movement(
                env=self,
                squad_id=unit_id,
                direction=direction,
                distance=distance,
                technique=MovementTechnique.BOUNDING,
                debug_level=self.debug_level
            )
        elif unit_type == UnitType.INFANTRY_PLATOON:
            # Platoon bounding requires specifying which squads move first
            # This would need to be implemented in the platoon movement functions
            frames = []
            # Placeholder for platoon bounding implementation
        else:
            # Teams don't bound internally - use regular movement
            from US_Army_PLT_Composition_vTest import US_IN_execute_movement
            frames = US_IN_execute_movement(
                env=self,
                unit_id=unit_id,
                direction=direction,
                distance=distance,
                debug_level=self.debug_level
            )

        # Get final position after movement
        end_pos = self.get_unit_position(unit_id)

        # Calculate reward based on movement with tactical bonus
        base_reward = self._calculate_movement_reward(start_pos, end_pos)
        tactical_bonus = 2.0  # Bonus for using appropriate tactics

        return base_reward + tactical_bonus

    def _execute_formation_action(self, action: Dict) -> float:
        """
        Handle formation change action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit changing formation
                - parameters:
                    - formation: New formation to apply

        Returns:
            Reward for formation change action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract formation parameter
        formation = parameters.formation

        # Get unit type
        unit_type = self.get_unit_property(unit_id, 'type')

        # Validate formation is appropriate for unit type
        from US_Army_PLT_Composition_vTest import US_IN_validate_formation
        if not US_IN_validate_formation(formation, unit_type):
            if self.debug_level > 0:
                print(f"Invalid formation {formation} for unit type {unit_type}")
            return -1.0  # Penalty for invalid formation

        # Apply formation
        from US_Army_PLT_Composition_vTest import US_IN_apply_formation
        US_IN_apply_formation(self, unit_id, formation)

        # Small positive reward for successful formation change
        return 0.5

    def _execute_report_action(self, action: Dict) -> float:
        """
        Handle report action (generates report to higher command / LLM).

        Args:
            action: Dictionary containing:
                - unit_id: ID of reporting unit
                - parameters:
                    - report_type: Type of report (SITREP, SPOTREP, etc.)

        Returns:
            Reward for report action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract report parameters
        report_type = parameters.report_type if hasattr(parameters, 'report_type') else 'SITREP'

        # Get unit position
        position = self.get_unit_position(unit_id)

        # Generate different report types
        if report_type == 'SITREP':
            # Situation report - unit status, position, surroundings
            from US_Army_PLT_Composition_vTest import OrderHandler

            # Create handler if needed
            if not hasattr(self, 'order_handler'):
                self.order_handler = OrderHandler(self)

            report = self.order_handler.handle_situation_report(unit_id, position)

            # Store report for LLM integration
            if not hasattr(self, 'pending_reports'):
                self.pending_reports = {}
            self.pending_reports[report.report_id] = report

            # Set LLM trigger flag if appropriate
            self.llm_decision_needed = True

        elif report_type == 'SPOTREP':
            # Spot report - enemy sighting
            enemies = self.get_visible_units(unit_id)
            if enemies:
                # Create spot report for enemies
                self.llm_decision_needed = True

        # Reward for successful reporting
        return 1.0

    def _execute_coordinate_action(self, action: Dict) -> float:
        """
        Handle coordination action (unit reaches coordination point).

        Args:
            action: Dictionary containing:
                - unit_id: ID of coordinating unit
                - parameters:
                    - coordination_point: Position to coordinate at

        Returns:
            Reward for coordination action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract coordination parameters
        coord_point = parameters.coordination_point

        # Check if this is a valid coordination point
        if coord_point not in self.state_manager.coordination_points:
            return -0.5  # Penalty for invalid coordination point

        # Update unit at coordination point
        coordination_complete = self.state_manager.update_unit_at_coordination(
            unit_id, coord_point, True)

        # Get coordination state
        coord_state = self.state_manager.get_coordination_state(coord_point)

        # If coordination requires higher approval, trigger LLM
        if coord_state['level'] != 'UNIT' and coordination_complete:
            # Generate situation report for higher-level decision
            from US_Army_PLT_Composition_vTest import OrderHandler
            if not hasattr(self, 'order_handler'):
                self.order_handler = OrderHandler(self)

            report = self.order_handler.handle_situation_report(unit_id, coord_point)

            # Store report for LLM integration
            if not hasattr(self, 'pending_reports'):
                self.pending_reports = {}
            self.pending_reports[report.report_id] = report

            # Set LLM trigger flag
            self.llm_decision_needed = True

            # Higher reward for completion requiring LLM decision
            return 2.0

        # Base reward for coordination
        reward = 1.0

        # Bonus if coordination is complete
        if coordination_complete:
            reward += 1.0

        return reward

    def _execute_halt_action(self, action: Dict) -> float:
        """
        Handle halt action (stop movement).

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit to halt

        Returns:
            Reward for halt action
        """
        unit_id = action['unit_id']

        # Get unit type
        unit_type = self.get_unit_property(unit_id, 'type')

        # Check if unit is currently moving
        if self.get_unit_property(unit_id, 'moving', False):
            # Stop movement
            self.update_unit_property(unit_id, 'moving', False)

            # If unit is part of route movement, update route state
            route_segment = self.get_unit_property(unit_id, 'current_route_segment', None)
            if route_segment is not None:
                # Pause route movement
                if unit_id in self.state_manager.unit_coordination:
                    self.state_manager.unit_coordination[unit_id]['route_paused'] = True

            # Small reward for responsive control
            return 0.5

        # No reward if unit wasn't moving
        return 0.0

    # Combat Functions
    def calculate_hit_probability(self, distance: float, weapon: BaseWeapon) -> float:
        """
        Calculate hit probability based on percentage of weapon's max range.

        Args:
            distance: Distance to target in cells
            weapon: BaseWeapon object with max_range and fire_rate attributes

        Returns:
            Float probability between 0.0 and 1.0
        """
        # Calculate what percentage of max range this distance represents
        range_percentage = (distance / weapon.max_range) * 100

        # Determine probability based on range percentage
        if range_percentage <= 50.0:  # Close range
            hit_prob = 0.9  # 90% chance to hit
        elif range_percentage <= 89.0:  # Medium range
            hit_prob = 0.7  # 70% chance to hit
        else:  # Long range
            hit_prob = 0.6  # 60% chance to hit

        # Apply weapon-specific modifiers
        if 'M249' in weapon.name or 'SAW' in weapon.name:
            # Slight penalty for automatic weapons at longer ranges
            hit_prob *= 0.9 if range_percentage > 70 else 1.0
        elif 'M240' in weapon.name:
            # Medium machine gun accuracy characteristics
            hit_prob *= 0.85 if range_percentage > 60 else 1.0
        elif 'Javelin' in weapon.name:
            # Guided weapons maintain accuracy at range
            hit_prob = max(hit_prob, 0.8)
        elif 'M320' in weapon.name or 'grenade' in weapon.name.lower():
            # Grenade launchers less accurate at all ranges
            hit_prob *= 0.7

        return hit_prob

    def calculate_damage(self, distance: float, weapon: BaseWeapon) -> float:
        """
        Calculate damage based on percentage of weapon's max range.

        Args:
            distance: Distance to target in cells
            weapon: BaseWeapon object with max_range and damage attributes

        Returns:
            Float damage amount
        """
        # Set base damage from weapon
        # NOTE: Each round does its own damage
        base_damage = weapon.damage

        # Calculate what percentage of max range this distance represents
        range_percentage = (distance / weapon.max_range) * 100

        # Determine damage multiplier based on range percentage
        if range_percentage <= 50.0:  # Close range
            damage_multiplier = 1.0  # 100% damage
        elif range_percentage <= 89.0:  # Medium range
            damage_multiplier = 0.9  # 90% damage
        else:  # Long range
            damage_multiplier = 0.8  # 80% damage

        # Apply weapon-specific effects
        if weapon.is_area_weapon:
            # Area weapons have falloff based on blast radius
            # Assume blast radius is 10% of max range for calculation
            blast_radius = weapon.max_range * 0.1
            if distance < blast_radius:
                # Within blast radius, damage scales from 100% at center to 60% at edge
                blast_multiplier = 1.0 - (0.4 * (distance / blast_radius))
                damage_multiplier *= blast_multiplier

        # Special handling for anti-tank weapons
        if 'Javelin' in weapon.name:
            # Maintains high damage even at longer ranges
            damage_multiplier = max(damage_multiplier, 0.9)

        return base_damage * damage_multiplier

    def apply_damage(self, unit_id: int, damage: float) -> None:
        """
        Apply damage to unit, handling casualties when health reaches zero.
        """
        if unit_id not in self.state_manager.active_units:
            return

        # Get current health and apply damage
        current_health = self.state_manager.get_unit_property(unit_id, 'health', 100)
        new_health = max(0, current_health - damage)
        self.state_manager.update_unit_property(unit_id, 'health', new_health)

        # If this is a soldier, also update parent team/squad health
        parent_id = self.state_manager.get_unit_property(unit_id, 'parent_id')
        if parent_id:
            # Calculate team health based on member health
            team_members = self.get_unit_children(parent_id)
            if team_members:
                total_health = sum(self.state_manager.get_unit_property(m, 'health', 0)
                                   for m in team_members)
                avg_health = total_health / len(team_members)
                self.state_manager.update_unit_property(parent_id, 'health', avg_health)

        # Handle casualty if health reached zero
        if new_health <= 0:
            self._handle_casualty(unit_id)

    def _handle_casualty(self, unit_id: int) -> None:
        """
        Handle unit becoming a casualty, including leadership succession if needed.

        Args:
            unit_id: ID of unit that became a casualty
        """
        # Skip if unit doesn't exist or is already handled
        if unit_id not in self.state_manager.active_units:
            return

        # Get unit information
        unit_type = self.get_unit_property(unit_id, 'type')
        parent_id = self.get_unit_property(unit_id, 'parent_id')
        is_leader = self.get_unit_property(unit_id, 'is_leader', False)
        force_type = self.get_unit_property(unit_id, 'force_type', ForceType.FRIENDLY)
        unit_string = self.get_unit_property(unit_id, 'string_id', str(unit_id))

        # Update unit status
        self.state_manager.update_unit_property(unit_id, 'status', 'KIA')

        # Update visual indicators in state tensor
        try:
            unit_pos = self.get_unit_position(unit_id)
            # Set casualty flag (bit 3)
            self.state_manager.state_tensor[unit_pos[1], unit_pos[0], 3] |= 8
        except:
            pass  # Position might be invalid

        if self.debug_level > 0:
            print(f"Unit {unit_string} ({unit_id}) is a casualty")

        # Handle leadership succession if unit is a leader and FRIENDLY force
        # Skip succession handling for enemy forces
        if is_leader and parent_id is not None and force_type == ForceType.FRIENDLY:
            if self.debug_level > 0:
                print(f"Initiating leadership succession for {unit_string}")

            try:
                # Import leadership succession handler from US Army module
                from US_Army_PLT_Composition_vTest import US_IN_handle_leader_casualty

                # Execute succession
                success = US_IN_handle_leader_casualty(self, parent_id)

                if self.debug_level > 0:
                    result = "successful" if success else "failed"
                    print(f"Leadership succession for {unit_string} was {result}")
            except Exception as e:
                # Handle succession failure gracefully
                if self.debug_level > 0:
                    print(f"Leadership succession for {unit_string} failed: {str(e)}")
        elif is_leader and force_type != ForceType.FRIENDLY:
            # Just log that enemy leader was eliminated, no succession handling
            if self.debug_level > 0:
                print(f"Enemy leader {unit_string} eliminated - no succession processing for enemy forces")

        # Special handling for different unit types
        if unit_type == UnitType.INFANTRY_TEAM or unit_type == UnitType.WEAPONS_TEAM:
            # Check if team is combat ineffective (50% casualties)
            team_members = self.get_unit_children(unit_id)
            active_members = sum(1 for m in team_members
                                 if self.get_unit_property(m, 'health', 0) > 0)

            if active_members < len(team_members) / 2:
                self.state_manager.update_unit_property(unit_id, 'combat_effective', False)
                if self.debug_level > 0:
                    print(f"Team {unit_string} is combat ineffective")

        # Notify parent unit of casualty
        if parent_id is not None:
            parent_casualties = self.get_unit_property(parent_id, 'casualties', 0)
            self.update_unit_property(parent_id, 'casualties', parent_casualties + 1)

    def check_unit_visibility(self, observer_id: int, target_id: int) -> Tuple[bool, float]:
        """
        Check if one unit can see another, accounting for terrain and distance.

        Args:
            observer_id: ID of observer unit
            target_id: ID of potential target unit

        Returns:
            Tuple of (can_see, visibility_factor)
        """
        # Verify both units exist
        if not (observer_id in self.state_manager.active_units and
                target_id in self.state_manager.active_units):
            return False, 0.0

        # Get positions
        observer_pos = self.state_manager.get_unit_position(observer_id)
        target_pos = self.state_manager.get_unit_position(target_id)

        # Get observer's observation range
        observer_range = self.state_manager.get_unit_property(
            observer_id, 'observation_range', 50)  # Default 500m

        # Check if in range
        distance = self._calculate_distance(observer_pos, target_pos)
        if distance > observer_range:
            return False, 0.0

        # Use visibility manager for detailed LOS check
        los_result = self.visibility_manager.check_line_of_sight(observer_pos, target_pos)

        if not los_result['has_los']:
            return False, 0.0

        # Get visibility factor
        visibility_factor = los_result['los_quality']

        # Apply target concealment if not firing
        if target_id not in self.visibility_manager.fired_units:
            # Get target's concealment from terrain
            target_concealment = self.visibility_manager.get_concealment_bonus(
                target_pos, target_id)
            # Reduce visibility by concealment
            visibility_factor *= (1.0 - target_concealment)

        # Apply distance falloff
        distance_factor = max(0.3, 1.0 - (distance / observer_range))
        visibility_factor *= distance_factor

        # Apply elevation advantage if any
        if los_result.get('elevation_advantage', False):
            visibility_factor = min(1.0, visibility_factor * 1.2)  # 20% bonus

        # Unit can be seen if visibility factor exceeds threshold
        can_see = visibility_factor > 0.05  # Minimum visibility threshold

        return can_see, visibility_factor

    def get_visible_units(self, observer_id: int, force_type: Optional[ForceType] = None) -> List[Tuple[int, float]]:
        """
        Get list of visible units and their visibility factors, optionally filtered by force type.

        Args:
            observer_id: ID of observing unit
            force_type: Optional filter for specific force type (FRIENDLY/ENEMY)

        Returns:
            List of tuples containing (unit_id, visibility_factor)
        """
        visible_units = []

        # Skip if observer doesn't exist
        if observer_id not in self.state_manager.active_units:
            return visible_units

        # Check each active unit
        for target_id in self.state_manager.active_units:
            # Skip self
            if target_id == observer_id:
                continue

            # Apply force type filter if specified
            if force_type is not None:
                target_force = self.get_unit_property(target_id, 'force_type', None)
                if target_force != force_type:
                    continue

            # Check visibility
            can_see, visibility = self.check_unit_visibility(observer_id, target_id)
            if can_see:
                visible_units.append((target_id, visibility))

        # Sort by visibility (most visible first)
        visible_units.sort(key=lambda x: x[1], reverse=True)

        return visible_units

    def can_engage(self, unit_id: int, target_pos: Tuple[int, int]) -> bool:
        """
        Check if unit can engage a target position based on range, LOS, and status.

        Args:
            unit_id: ID of unit attempting to engage
            target_pos: Position to engage

        Returns:
            Boolean indicating whether engagement is possible
        """
        # Check if unit exists and is active
        if unit_id not in self.state_manager.active_units:
            return False

        # Check if unit is alive and not heavily suppressed
        unit_health = self.get_unit_property(unit_id, 'health', 0)
        if unit_health <= 0:
            return False

        # Check if unit is suppressed (via combat manager)
        if hasattr(self, 'combat_manager'):
            if unit_id in self.combat_manager.suppressed_units:
                suppression = self.combat_manager.suppressed_units[unit_id]['level']
                if suppression >= 0.8:  # Heavily suppressed
                    return False

        # Get unit position and properties
        unit_pos = self.get_unit_position(unit_id)
        engagement_range = self.get_unit_property(unit_id, 'engagement_range', 40)  # Default 400m

        # Check range
        distance = self._calculate_distance(unit_pos, target_pos)
        if distance > engagement_range:
            return False

        # Check line of sight using visibility manager
        los_result = self.visibility_manager.check_line_of_sight(unit_pos, target_pos)
        if not los_result['has_los']:
            return False

        # Check ammunition
        if hasattr(self, 'combat_manager'):
            ammo = self.combat_manager._get_unit_ammo(unit_id, 'primary')
            if ammo <= 0:
                return False

        return True

    def execute_fire(self, unit_id: int, target_pos: Tuple[int, int],
                     weapon_type: str = 'primary') -> Tuple[bool, float]:
        """
        Execute basic firing action without combat manager.
        For backwards compatibility - prefer using combat_manager methods.

        Args:
            unit_id: ID of firing unit
            target_pos: Position to target
            weapon_type: Which weapon to use ('primary' or 'secondary')

        Returns:
            Tuple of (hit_successful, damage_dealt)
        """
        # Forward to combat manager if available
        if hasattr(self, 'combat_manager'):
            fire_control = FireControl(
                target_area=target_pos,
                max_rounds=1,
                rate_of_fire=1.0,
                suppress_only=False
            )
            results = self.combat_manager.execute_engagement(unit_id, fire_control)
            return results.hits > 0, results.damage_dealt

        # Legacy implementation if combat manager not available
        if not self.can_engage(unit_id, target_pos):
            return False, 0.0

        # Get unit properties
        unit_pos = self.get_unit_position(unit_id)

        # Get weapon
        weapon = self.get_unit_property(unit_id, f'{weapon_type}_weapon')
        if not weapon:
            return False, 0.0

        # Check ammo
        ammo_key = f'ammo_{weapon_type}'
        ammo = self.get_unit_property(unit_id, ammo_key, 0)
        if ammo <= 0:
            return False, 0.0

        # Calculate hit probability
        distance = self._calculate_distance(unit_pos, target_pos)
        base_hit_prob = self.calculate_hit_probability(distance, weapon)

        # Apply modifiers
        _, visibility = self.check_line_of_sight(unit_pos, target_pos)
        cover = self.terrain_manager.get_cover(target_pos)

        final_hit_prob = base_hit_prob * visibility * (1 - cover)

        # Determine hit
        hit = random.random() < final_hit_prob

        # Calculate and apply damage
        damage = 0.0
        if hit:
            damage = self.calculate_damage(distance, weapon)

            # Get unit at target position
            target_unit_id = self.state_manager.state_tensor[target_pos[1], target_pos[0], 2]
            if target_unit_id > 0:
                self.apply_damage(target_unit_id, damage)

        # Update ammo
        new_ammo = ammo - 1
        self.update_unit_property(unit_id, ammo_key, new_ammo)

        # Record firing for concealment effects
        self.visibility_manager.record_unit_fired(unit_id)

        if self.debug_level > 1:
            hit_result = "Hit" if hit else "Miss"
            print(f"Unit {unit_id} fired at {target_pos}: {hit_result}")
            if hit:
                print(f"Damage dealt: {damage:.1f}")

        return hit, damage

    # Unit Creation and Management Functions
    def create_unit(self, unit_type: UnitType, unit_id_str: str, start_position: Tuple[int, int]) -> int:
        """
        Create new unit in environment.

        Args:
            unit_type: Type of unit to create
            unit_id_str: String identifier (e.g., "1PLT-1SQD-ATM")
            start_position: Initial (x,y) position

        Returns:
            Numeric unit ID for environment tracking
        """
        # Generate numeric ID
        numeric_id = self._generate_unit_id()

        # Initialize basic properties
        properties = {
            'type': unit_type,
            'string_id': unit_id_str,
            'position': start_position,
            'health': 100,
            'orientation': 0,
            'formation': self._get_default_formation(unit_type),
            'parent_id': None,  # Will be set during hierarchy setup
            'force_type': ForceType.FRIENDLY  # Default to friendly force
        }

        # Add type-specific properties
        if unit_type == UnitType.INFANTRY_TEAM:
            properties.update({
                'observation_range': 50,
                'engagement_range': 40,
                'team_type': 'infantry',
                'ammo_primary': 210,  # Standard combat load
                'ammo_secondary': 0,  # No secondary by default
                'weapons_operational': True,
                'has_automatic_weapons': True  # Infantry teams have automatic weapons
            })
        elif unit_type == UnitType.WEAPONS_TEAM:
            properties.update({
                'observation_range': 70,
                'engagement_range': 60,
                'team_type': 'weapons',
                'ammo_primary': 1000,  # More ammo for weapons teams
                'ammo_secondary': 0,
                'weapons_operational': True,
                'has_automatic_weapons': True  # Weapons teams have automatic weapons
            })
        elif unit_type == UnitType.INFANTRY_SQUAD:
            properties.update({
                'observation_range': 70,
                'engagement_range': 60,
                'ammo_primary': 0,  # Squad itself doesn't have ammo
                'ammo_secondary': 0,
                'weapons_operational': True,
                'has_automatic_weapons': True
            })
        elif unit_type == UnitType.INFANTRY_PLATOON:
            properties.update({
                'observation_range': 100,
                'engagement_range': 80,
                'ammo_primary': 0,  # Platoon itself doesn't have ammo
                'ammo_secondary': 0,
                'weapons_operational': True,
                'has_automatic_weapons': True
            })

        # Initialize combat tracking
        if self.combat_manager:
            # Initialize ammunition tracking in combat manager
            if 'ammo_primary' in properties:
                self.combat_manager.ammo_tracking[numeric_id] = {
                    'primary': properties['ammo_primary'],
                    'secondary': properties['ammo_secondary']
                }

        # Add to state tracking
        self.state_manager.add_unit(numeric_id, properties)

        # Verify position is within bounds
        x, y = start_position
        if 0 <= x < self.width and 0 <= y < self.height:
            # Update state tensor
            self.state_manager.state_tensor[y, x, 2] = numeric_id  # Unit ID channel
            self.state_manager.state_tensor[y, x, 3] = 0  # Reset status flags

        if self.debug_level > 0:
            print(f"Created unit {unit_id_str} with ID {numeric_id}")

        return numeric_id

    def get_unit_position(self, unit_id: int) -> Tuple[int, int]:
        """
        Get current position of unit.

        Args:
            unit_id: ID of unit to locate

        Returns:
            (x, y) position tuple

        Raises:
            ValueError: If unit_id not found
        """
        return self.state_manager.get_unit_position(unit_id)

    def update_unit_position(self, unit_id: int, new_pos: Tuple[int, int]) -> None:
        """
        Update unit position in both properties and state tensor.

        Args:
            unit_id: ID of unit to move
            new_pos: New (x,y) position

        Raises:
            ValueError: If unit_id not found or position invalid
        """
        if unit_id not in self.state_manager.active_units:
            raise ValueError(f"Unit {unit_id} not found")

        # Validate position is within environment bounds
        x, y = new_pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Position {new_pos} is outside environment bounds")

        # Get current position and update state tensor
        old_pos = self.state_manager.get_unit_position(unit_id)
        old_x, old_y = old_pos

        # Get current status flags from old position
        status_flags = self.state_manager.state_tensor[old_y, old_x, 3]

        # Clear old position in unit ID channel
        self.state_manager.state_tensor[old_y, old_x, 2] = 0
        self.state_manager.state_tensor[old_y, old_x, 3] = 0

        # Update position in state manager
        self.state_manager.update_unit_position(unit_id, new_pos)

        # Update new position in state tensor
        self.state_manager.state_tensor[y, x, 2] = unit_id
        self.state_manager.state_tensor[y, x, 3] = status_flags  # Preserve status flags

        # Update suppression visualization if unit is suppressed
        if self.combat_manager and unit_id in self.combat_manager.suppressed_units:
            self.state_manager.state_tensor[y, x, 3] |= 1  # Set suppression bit

    def get_unit_property(self, unit_id: int, property_name: str, default_value=None):
        """
        Get specific unit property with optional default value.

        Args:
            unit_id: ID of unit
            property_name: Name of property to get
            default_value: Value to return if property doesn't exist

        Returns:
            Property value or default if not found

        Raises:
            ValueError: If unit_id not found and no default provided
        """
        try:
            # Special handling for ammunition
            if property_name in ['ammo_primary', 'ammo_secondary'] and hasattr(self, 'combat_manager'):
                weapon_type = property_name.split('_')[1]
                return self.combat_manager._get_unit_ammo(unit_id, weapon_type)

            return self.state_manager.get_unit_property(unit_id, property_name, default_value)
        except ValueError:
            if default_value is not None:
                return default_value
            raise

    def update_unit_property(self, unit_id: int, property_name: str, value) -> None:
        """
        Update specific unit property with integration for combat properties.

        Args:
            unit_id: ID of unit
            property_name: Name of property to update
            value: New value for property

        Raises:
            ValueError: If unit_id not found
        """
        if unit_id not in self.state_manager.active_units:
            raise ValueError(f"Unit {unit_id} not found")

        # Special handling for ammunition updates
        if property_name in ['ammo_primary', 'ammo_secondary'] and hasattr(self, 'combat_manager'):
            weapon_type = property_name.split('_')[1]

            # Initialize tracking if needed
            if unit_id not in self.combat_manager.ammo_tracking:
                self.combat_manager.ammo_tracking[unit_id] = {}

            self.combat_manager.ammo_tracking[unit_id][weapon_type] = value

        # Special handling for health - check for casualties
        if property_name == 'health' and value <= 0:
            old_health = self.state_manager.get_unit_property(unit_id, 'health', 100)
            if old_health > 0:
                # Unit has just become a casualty
                self._handle_casualty(unit_id)

        # Update in state manager
        self.state_manager.update_unit_property(unit_id, property_name, value)

        # Update status flags for visualization
        if property_name in ['health', 'suppressed']:
            try:
                pos = self.get_unit_position(unit_id)
                status_flags = self.state_manager.state_tensor[pos[1], pos[0], 3]

                if property_name == 'health' and value <= 0:
                    # Set casualty flag (bit 1)
                    status_flags |= 2
                elif property_name == 'suppressed' and value:
                    # Set suppression flag (bit 0)
                    status_flags |= 1
                elif property_name == 'suppressed' and not value:
                    # Clear suppression flag
                    status_flags &= ~1

                self.state_manager.state_tensor[pos[1], pos[0], 3] = status_flags
            except:
                pass  # Ignore errors updating visualization

    def get_unit_children(self, unit_id: int) -> List[int]:
        """
        Get IDs of all child units that have this unit as parent.

        Args:
            unit_id: Parent unit ID

        Returns:
            List of child unit IDs
        """
        return [uid for uid in self.state_manager.active_units
                if self.state_manager.get_unit_property(uid, 'parent_id') == unit_id]

    def set_unit_hierarchy(self, unit_id: int, parent_id: Optional[int]) -> None:
        """
        Set parent-child relationship between units.

        Args:
            unit_id: ID of child unit
            parent_id: ID of parent unit, or None to clear relationship

        Raises:
            ValueError: If either unit ID is invalid
        """
        if unit_id not in self.state_manager.active_units:
            raise ValueError(f"Unit {unit_id} not found")

        if parent_id and parent_id not in self.state_manager.active_units:
            raise ValueError(f"Parent unit {parent_id} not found")

        self.state_manager.update_unit_property(unit_id, 'parent_id', parent_id)

        # Update force type to match parent if appropriate
        if parent_id:
            parent_force = self.state_manager.get_unit_property(parent_id, 'force_type', None)
            if parent_force:
                self.state_manager.update_unit_property(unit_id, 'force_type', parent_force)

        if self.debug_level > 1:
            print(f"Set parent of {unit_id} to {parent_id}")

    def apply_formation(self, unit_id: int, formation_type: str, formation_template: Dict) -> None:
        """
        Apply formation to unit with proper positioning of all elements.

        Args:
            unit_id: ID of unit to form up
            formation_type: Name of formation to apply
            formation_template: Dict mapping roles/positions to relative coordinates

        Raises:
            ValueError: If unit not found or formation invalid
        """
        if unit_id not in self.state_manager.active_units:
            raise ValueError(f"Unit {unit_id} not found")

        # Get base position and orientation
        base_pos = self.state_manager.get_unit_position(unit_id)
        orientation = float(self.state_manager.get_unit_property(unit_id, 'orientation', 0))
        angle_rad = math.radians(orientation - 90)  # Convert to radians, adjust to make 0° face north

        if self.debug_level > 1:
            print(f"\nApplying formation {formation_type}")
            print(f"Base position: {base_pos}")
            print(f"Orientation: {orientation}°")

        # Process each child unit
        for child_id in self.get_unit_children(unit_id):
            # Get child properties
            role_value = self.state_manager.get_unit_property(child_id, 'role', None)
            string_id = self.state_manager.get_unit_property(child_id, 'string_id', '')

            # Look for a matching position in template
            offset = None
            template_key = None

            # Try exact matches first
            if string_id in formation_template:
                template_key = string_id
                offset = formation_template[template_key]

            # Try partial string matches
            if offset is None:
                for key in formation_template:
                    if isinstance(key, str) and key in string_id:
                        template_key = key
                        offset = formation_template[key]
                        break

            # Try role value matches
            if offset is None and role_value is not None:
                for key in formation_template:
                    if hasattr(key, 'value') and key.value == role_value:
                        template_key = key
                        offset = formation_template[key]
                        break

            if offset is not None:
                # Calculate rotated position
                rot_x = int(offset[0] * math.cos(angle_rad) -
                            offset[1] * math.sin(angle_rad))
                rot_y = int(offset[0] * math.sin(angle_rad) +
                            offset[1] * math.cos(angle_rad))

                # Calculate final position
                new_x = base_pos[0] + rot_x
                new_y = base_pos[1] + rot_y

                # Validate bounds
                new_x = max(0, min(new_x, self.width - 1))
                new_y = max(0, min(new_y, self.height - 1))

                # Update position
                self.update_unit_position(child_id, (new_x, new_y))

                # Update orientation
                self.update_unit_property(child_id, 'orientation', orientation)

                if self.debug_level > 1:
                    print(f"\nPositioned {string_id}:")
                    print(f"  Template key: {template_key}")
                    print(f"  Offset: {offset}")
                    print(f"  Final position: ({new_x}, {new_y})")
            else:
                if self.debug_level > 0:
                    print(f"Warning: No position found in template for {string_id} (Role {role_value})")

        # Update formation property after successful application
        self.state_manager.update_unit_property(unit_id, 'formation', formation_type)
        self.state_manager.update_unit_property(unit_id, 'formation_template', formation_template)

    def create_soldier(self, role: Enum, unit_id_str: str,
                       position: Tuple[int, int], is_leader: bool) -> int:
        """
        Create individual soldier with standardized role handling.

        Args:
            role: Soldier's role (from US_IN_Role enum)
            unit_id_str: String identifier for soldier
            position: Initial (x,y) position
            is_leader: Whether soldier is a leader

        Returns:
            Numeric unit ID for environment tracking
        """
        soldier_id = self._generate_unit_id()

        # Set up weapon properties based on role
        primary_weapon = None
        secondary_weapon = None
        observation_range = 50  # Default
        engagement_range = 40  # Default

        # Determine weapons and capabilities by role name
        role_name = role.name if hasattr(role, 'name') else str(role)

        if 'LEADER' in role_name:
            primary_weapon = 'M4'
            ammo_primary = 210
            observation_range = 60 if 'PLATOON' in role_name or 'SQUAD' in role_name else 50
        elif 'AUTO_RIFLEMAN' in role_name:
            primary_weapon = 'M249'
            ammo_primary = 600
            engagement_range = 60
        elif 'GRENADIER' in role_name:
            primary_weapon = 'M4'
            secondary_weapon = 'M320'
            ammo_primary = 210
            ammo_secondary = 12
        elif 'MACHINE_GUNNER' in role_name:
            primary_weapon = 'M240B'
            ammo_primary = 1000
            engagement_range = 80
            observation_range = 60
        elif 'ANTI_TANK' in role_name:
            primary_weapon = 'M4'
            secondary_weapon = 'Javelin'
            ammo_primary = 210
            ammo_secondary = 3
            engagement_range = 150  # Javelin has long range
        else:  # Default rifleman
            primary_weapon = 'M4'
            ammo_primary = 210

        # Create base soldier properties
        properties = {
            'type': 'soldier',
            'role': role.value if hasattr(role, 'value') else role,
            'role_name': role_name,
            'string_id': unit_id_str,
            'position': position,
            'is_leader': is_leader,
            'health': 100,
            'orientation': 0,
            'observation_range': observation_range,
            'engagement_range': engagement_range,
            'primary_weapon': primary_weapon,
            'secondary_weapon': secondary_weapon,
            'ammo_primary': ammo_primary,
            'ammo_secondary': ammo_secondary if secondary_weapon else 0,
            'force_type': ForceType.FRIENDLY,
            'weapons_operational': True,
            'has_automatic_weapons': 'M249' in primary_weapon or 'M240' in primary_weapon,
            'parent_id': None
        }

        # Add to state tracking
        self.state_manager.add_unit(soldier_id, properties)

        # Initialize ammunition tracking
        if hasattr(self, 'combat_manager'):
            self.combat_manager.ammo_tracking[soldier_id] = {
                'primary': ammo_primary,
                'secondary': ammo_secondary if secondary_weapon else 0
            }

        # Update state tensor
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.state_manager.state_tensor[y, x, 2] = soldier_id

        return soldier_id

    # Movement Functions
    def move_unit(self, unit_id: int, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """
        Move unit while maintaining formation.

        Args:
            unit_id: ID of unit to move
            direction: (dx, dy) movement vector
            distance: Number of spaces to move

        Returns:
            List of frames showing movement
        """
        if unit_id not in self.state_manager.active_units:
            return []

        # Get unit properties
        unit_type = self.get_unit_property(unit_id, 'type')
        current_formation = self.get_unit_property(unit_id, 'formation')
        current_pos = self.get_unit_position(unit_id)

        # Calculate movement frames based on unit type
        movement_frames = []
        if unit_type == UnitType.INFANTRY_TEAM or unit_type == UnitType.WEAPONS_TEAM:
            # Use team movement
            from US_Army_PLT_Composition_vTest import US_IN_execute_movement
            movement_frames = US_IN_execute_movement(
                self, unit_id, direction, distance, debug_level=0)

        elif unit_type == UnitType.INFANTRY_SQUAD:
            # Use squad movement
            from US_Army_PLT_Composition_vTest import execute_squad_movement
            movement_frames = execute_squad_movement(
                self, unit_id, direction, distance, debug_level=0)

        else:
            # Basic movement for other units
            dx, dy = direction
            magnitude = math.sqrt(dx * dx + dy * dy)
            if magnitude > 0:
                move_x = int((dx * distance) / magnitude)
                move_y = int((dy * distance) / magnitude)

                new_x = max(0, min(current_pos[0] + move_x, self.width - 1))
                new_y = max(0, min(current_pos[1] + move_y, self.height - 1))

                # Apply formation at new position
                if current_formation:
                    template = self.get_unit_property(unit_id, 'formation_template')
                    if template:
                        self.apply_formation(unit_id, current_formation, template)

                self.update_unit_position(unit_id, (new_x, new_y))

                # Create single frame for basic movement
                movement_frames = [self._capture_positions(unit_id)]

        # Update unit position in state manager after movement
        final_pos = self.get_unit_position(unit_id)
        self.state_manager.update_unit_position(unit_id, final_pos)

        # Check if unit moved through danger areas
        if len(movement_frames) > 0:
            self._check_movement_danger(unit_id, current_pos, final_pos)

        return movement_frames

    def calculate_rotation(self, current_orientation: int,
                           target_direction: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate rotation needed to face target direction.

        Args:
            current_orientation: Current orientation in degrees (0-359)
            target_direction: (dx, dy) tuple indicating target direction

        Returns:
            Tuple of (new_orientation, rotation_needed)
            - new_orientation: Final orientation in degrees (0-359)
            - rotation_needed: Degrees to rotate (-180 to 180)
        """
        dx, dy = target_direction
        if dx == 0 and dy == 0:
            return current_orientation, 0

        # Calculate target orientation (0° is North, increases clockwise)
        target_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)

        # Calculate shortest rotation to target (-180 to +180 degrees)
        rotation_needed = ((target_orientation - current_orientation + 180) % 360) - 180

        return target_orientation, rotation_needed

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def _check_movement_danger(self, unit_id: int, start_pos: Tuple[int, int],
                               end_pos: Tuple[int, int]) -> Dict:
        """
        Check if unit moved through danger areas and apply effects.

        Args:
            unit_id: ID of unit that moved
            start_pos: Starting position
            end_pos: Ending position

        Returns:
            Dictionary with danger assessment
        """
        danger_assessment = {
            'entered_fire_zone': False,
            'detected': False,
            'suppressed': False,
            'damage_taken': 0.0
        }

        # Get path between positions
        path_points = self._get_line_points(start_pos[0], start_pos[1],
                                            end_pos[0], end_pos[1])

        # Check each point for enemy observation
        for point in path_points:
            # Skip start and end
            if point == start_pos or point == end_pos:
                continue

            # Check if point is observable by enemies
            for enemy_id in self.state_manager.active_units:
                # Skip non-enemy units
                force_type = self.get_unit_property(enemy_id, 'force_type', None)
                if force_type != ForceType.ENEMY:
                    continue

                # Skip inactive enemies
                if self.get_unit_property(enemy_id, 'health', 0) <= 0:
                    continue

                enemy_pos = self.get_unit_position(enemy_id)

                # Check if enemy can see this point
                can_see, visibility = self.visibility_manager.check_line_of_sight(
                    enemy_pos, point, for_observation=True)

                if can_see:
                    danger_assessment['detected'] = True

                    # Check if in engagement range
                    engagement_range = self.get_unit_property(enemy_id, 'engagement_range', 0)
                    distance = self._calculate_distance(enemy_pos, point)

                    if distance <= engagement_range:
                        danger_assessment['entered_fire_zone'] = True

                        # Chance of suppression based on enemy effectiveness
                        enemy_effectiveness = 0.7  # Default value
                        if hasattr(self, 'combat_manager'):
                            enemy_team_id = self.get_unit_property(enemy_id, 'parent_id')
                            if enemy_team_id:
                                enemy_effectiveness = self.combat_manager.get_team_combat_effectiveness(
                                    enemy_team_id)

                        # Apply suppression with probability based on effectiveness
                        if random.random() < enemy_effectiveness * 0.3:
                            # Apply light suppression through combat manager
                            if hasattr(self, 'combat_manager'):
                                self.combat_manager._apply_suppression(
                                    unit_id, 0.3, duration=2)
                                danger_assessment['suppressed'] = True

                    # First detection is sufficient
                    break

        return danger_assessment

    # Reward and Evaluation Functions
    def _calculate_engagement_reward(self, results: Dict) -> float:
        """
        Calculate reward for engagement action based on results.

        Args:
            results: Engagement results (either EngagementResults or dict for team/squad)

        Returns:
            Float reward value
        """
        # Handle both individual and team/squad results
        if isinstance(results, dict):
            # Team or squad engagement results
            hits = results.get('total_hits', 0)
            damage = results.get('total_damage', 0.0)
            ammo = results.get('ammo_expended', 0)
            targets_hit = len(results.get('targets_hit', []))
            effectiveness = results.get('effectiveness', 0.0)
        else:
            # Individual engagement results
            hits = results.hits
            damage = results.damage_dealt
            ammo = results.rounds_expended
            targets_hit = len(results.targets_hit) if hasattr(results, 'targets_hit') else 0
            effectiveness = 0.0  # Not calculated for individual engagements

        # Base rewards
        hit_reward = hits * 2.0  # Reward per hit
        damage_reward = damage * 0.1  # Scaled reward for damage
        target_variety_reward = targets_hit * 1.5  # Reward for hitting multiple targets

        # Effectiveness bonus (if available)
        effectiveness_bonus = effectiveness * 5.0 if effectiveness > 0 else 0.0

        # Penalty for wasted ammunition (no hits)
        ammo_penalty = -0.1 * ammo if hits == 0 else 0.0

        # Calculate total reward
        total_reward = hit_reward + damage_reward + target_variety_reward + effectiveness_bonus + ammo_penalty

        # Cap reward to reasonable range
        total_reward = max(-10.0, min(total_reward, 50.0))

        if self.debug_level > 1:
            print(f"\nEngagement Reward Calculation:")
            print(f"Hits: {hits} × 2.0 = {hit_reward}")
            print(f"Damage: {damage:.1f} × 0.1 = {damage_reward:.1f}")
            print(f"Targets: {targets_hit} × 1.5 = {target_variety_reward}")
            print(f"Effectiveness: {effectiveness:.2f} × 5.0 = {effectiveness_bonus:.1f}")
            print(f"Ammo Penalty: {ammo_penalty:.1f}")
            print(f"Total Reward: {total_reward:.1f}")

        return total_reward

    def _calculate_suppression_reward(self, results: Dict) -> float:
        """
        Calculate reward for suppression action based on results.

        Args:
            results: Suppression results (either EngagementResults or dict for team/squad)

        Returns:
            Float reward value
        """
        # Handle both individual and team/squad results
        if isinstance(results, dict):
            # Team or squad suppression results
            suppression = results.get('suppression_level', 0.0)
            ammo = results.get('ammo_expended', 0)
            area_covered = results.get('area_radius', 3)  # Default if not specified
            time_sustained = results.get('time_steps', 0)
        else:
            # Individual suppression results
            suppression = results.suppression_effect
            ammo = results.rounds_expended
            area_covered = 1  # Single point for individual
            time_sustained = results.time_steps

        # Base suppression reward
        suppression_reward = suppression * 10.0  # Scale up suppression effect

        # Area coverage bonus
        area_bonus = area_covered * 0.5

        # Sustained fire bonus
        sustained_bonus = time_sustained * 0.5 if time_sustained > 3 else 0.0

        # Ammunition efficiency
        if ammo > 0:
            efficiency = suppression / ammo
            efficiency_factor = efficiency * 20.0  # Scale for reasonable values
        else:
            efficiency_factor = 0.0

        # Penalty for ineffective suppression (high ammo, low effect)
        ineffective_penalty = -5.0 if (ammo > 30 and suppression < 0.3) else 0.0

        # Calculate total reward
        total_reward = suppression_reward + area_bonus + sustained_bonus + efficiency_factor + ineffective_penalty

        # Cap reward to reasonable range
        total_reward = max(-10.0, min(total_reward, 30.0))

        if self.debug_level > 1:
            print(f"\nSuppression Reward Calculation:")
            print(f"Suppression: {suppression:.2f} × 10.0 = {suppression_reward:.1f}")
            print(f"Area Coverage: {area_covered} × 0.5 = {area_bonus}")
            print(f"Sustained Fire: {sustained_bonus}")
            print(f"Efficiency: {efficiency_factor:.1f}")
            print(f"Ineffective Penalty: {ineffective_penalty}")
            print(f"Total Reward: {total_reward:.1f}")

        return total_reward

    def _calculate_movement_reward(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> float:
        """
        Calculate reward for movement actions.

        Args:
            start_pos: Starting position
            end_pos: Ending position

        Returns:
            Float reward value
        """
        # No reward if position didn't change
        if start_pos == end_pos:
            return 0.0

        # Calculate distance moved
        distance_moved = self._calculate_distance(start_pos, end_pos)

        # Base movement reward
        movement_reward = distance_moved * 0.1

        # Determine if movement was toward objective
        objective_reward = 0.0
        if hasattr(self, 'objective'):
            start_to_objective = self._calculate_distance(start_pos, self.objective)
            end_to_objective = self._calculate_distance(end_pos, self.objective)

            # Reward movement toward objective, penalize movement away
            if end_to_objective < start_to_objective:
                # Moving toward objective
                approach_amount = start_to_objective - end_to_objective
                objective_reward = approach_amount * 0.5
            else:
                # Moving away from objective (smaller penalty)
                retreat_amount = end_to_objective - start_to_objective
                objective_reward = -retreat_amount * 0.2

        # Terrain movement cost penalty
        end_terrain_cost = self.terrain_manager.get_movement_cost(end_pos)
        terrain_penalty = -end_terrain_cost * 0.2

        # Exposure penalty - higher penalty for ending in exposed positions
        end_cover = self.terrain_manager.get_cover(end_pos)
        exposure_penalty = (1.0 - end_cover) * -0.5

        # Calculate threat exposure change
        threat_change_penalty = 0.0
        if hasattr(self, 'combat_manager') and hasattr(self.combat_manager, 'threat_analyzer'):
            # Get threat levels at start and end
            start_threat = self.combat_manager.threat_analyzer.total_threat_matrix[start_pos[1]][start_pos[0]]
            end_threat = self.combat_manager.threat_analyzer.total_threat_matrix[end_pos[1]][end_pos[0]]

            # Penalize movement into higher threat areas
            if end_threat > start_threat:
                threat_increase = end_threat - start_threat
                threat_change_penalty = -threat_increase * 3.0

        # Calculate total reward
        total_reward = movement_reward + objective_reward + terrain_penalty + exposure_penalty + threat_change_penalty

        # Cap reward to reasonable range
        total_reward = max(-10.0, min(total_reward, 10.0))

        if self.debug_level > 1:
            print(f"\nMovement Reward Calculation:")
            print(f"Distance Moved: {distance_moved:.1f} × 0.1 = {movement_reward:.1f}")
            print(f"Objective Approach: {objective_reward:.1f}")
            print(f"Terrain Penalty: {terrain_penalty:.1f}")
            print(f"Exposure Penalty: {exposure_penalty:.1f}")
            print(f"Threat Change: {threat_change_penalty:.1f}")
            print(f"Total Reward: {total_reward:.1f}")

        return total_reward

    def _calculate_formation_reward(self, unit_id: int, formation: str) -> float:
        """
        Calculate reward for formation change based on tactical appropriateness.

        Args:
            unit_id: Unit changing formation
            formation: Formation applied

        Returns:
            Float reward value
        """
        # Base reward for successful formation change
        base_reward = 0.5

        # Get unit type and position
        unit_type = self.get_unit_property(unit_id, 'type')
        position = self.get_unit_position(unit_id)

        # Get terrain at position
        terrain_type = self.terrain_manager.get_terrain_type(position)

        # Calculate tactical appropriateness of formation to terrain
        terrain_appropriateness = 0.0

        # Wedge formations are good in open terrain
        if 'wedge' in formation and terrain_type in [TerrainType.BARE, TerrainType.SPARSE_VEG]:
            terrain_appropriateness = 1.0
        # Line formations are good for assaults
        elif 'line' in formation:
            # Check if facing objective
            if hasattr(self, 'objective'):
                # Calculate orientation toward objective
                dx = self.objective[0] - position[0]
                dy = self.objective[1] - position[1]
                orientation = int((math.degrees(math.atan2(dy, dx)) + 90) % 360)

                # Get unit's current orientation
                unit_orientation = self.get_unit_property(unit_id, 'orientation', 0)

                # Reward for proper assault orientation
                if abs((orientation - unit_orientation + 180) % 360 - 180) < 45:
                    terrain_appropriateness = 1.5
        # Column formations are good in restricted terrain
        elif 'column' in formation and terrain_type in [TerrainType.DENSE_VEG, TerrainType.WOODS]:
            terrain_appropriateness = 1.0

        # Proximity to enemies affects formation appropriateness
        enemy_proximity_factor = 0.0
        if hasattr(self, 'combat_manager') and hasattr(self.combat_manager, 'threat_analyzer'):
            threat_level = self.combat_manager.threat_analyzer.total_threat_matrix[position[1]][position[0]]

            # Wedge/line better under higher threat
            if threat_level > 0.5 and ('wedge' in formation or 'line' in formation):
                enemy_proximity_factor = 1.0
            # Column better under lower threat
            elif threat_level < 0.3 and 'column' in formation:
                enemy_proximity_factor = 0.5

        # Calculate total reward
        total_reward = base_reward + terrain_appropriateness + enemy_proximity_factor

        if self.debug_level > 1:
            print(f"\nFormation Reward Calculation:")
            print(f"Base Reward: {base_reward}")
            print(f"Terrain Appropriateness: {terrain_appropriateness}")
            print(f"Enemy Proximity Factor: {enemy_proximity_factor}")
            print(f"Total Reward: {total_reward:.1f}")

        return total_reward

    def _calculate_coordination_reward(self, unit_id: int, coordination_point: Tuple[int, int],
                                       coordination_complete: bool) -> float:
        """
        Calculate reward for coordination actions.

        Args:
            unit_id: Coordinating unit
            coordination_point: Position of coordination
            coordination_complete: Whether coordination is complete

        Returns:
            Float reward value
        """
        # Base reward for attempting coordination
        base_reward = 1.0

        # Get coordination state
        coord_state = self.state_manager.get_coordination_state(coordination_point)
        if not coord_state:
            return 0.0  # Invalid coordination point

        # Bonus for completing all requirements
        completion_bonus = 3.0 if coordination_complete else 0.0

        # Progress bonus based on percentage of units present
        if len(coord_state['required_units']) > 0:
            progress_ratio = len(coord_state['completed_units']) / len(coord_state['required_units'])
            progress_bonus = progress_ratio * 2.0
        else:
            progress_bonus = 0.0

        # Condition satisfaction bonus
        condition_bonus = 0.0
        if 'conditions_met' in coord_state and coord_state['conditions_met']:
            condition_count = sum(1 for v in coord_state['conditions_met'].values() if v)
            condition_total = len(coord_state['conditions_met'])
            if condition_total > 0:
                condition_bonus = (condition_count / condition_total) * 2.0

        # Calculate total reward
        total_reward = base_reward + completion_bonus + progress_bonus + condition_bonus

        if self.debug_level > 1:
            print(f"\nCoordination Reward Calculation:")
            print(f"Base Reward: {base_reward}")
            print(f"Completion Bonus: {completion_bonus}")
            print(f"Progress Bonus: {progress_bonus:.1f}")
            print(f"Condition Bonus: {condition_bonus:.1f}")
            print(f"Total Reward: {total_reward:.1f}")

        return total_reward

    def _calculate_report_reward(self, report_type: str, report_quality: float = 1.0) -> float:
        """
        Calculate reward for reporting actions.

        Args:
            report_type: Type of report generated
            report_quality: Quality factor for report (0-1)

        Returns:
            Float reward value
        """
        # Base rewards by report type
        base_rewards = {
            'SITREP': 1.0,  # Situation report
            'SPOTREP': 2.0,  # Enemy spotted
            'ACK': 0.5,  # Acknowledgement
            'SALUTE': 2.5,  # Size/Activity/Location/Unit/Time/Equipment
            'MEDEVAC': 1.5  # Medical evacuation
        }

        # Get base reward for report type
        base_reward = base_rewards.get(report_type, 0.5)

        # Scale by quality
        quality_adjusted = base_reward * report_quality

        # Add LLM trigger bonus if applicable
        llm_trigger_bonus = 1.0 if self.llm_decision_needed else 0.0

        # Calculate total reward
        total_reward = quality_adjusted + llm_trigger_bonus

        if self.debug_level > 1:
            print(f"\nReport Reward Calculation:")
            print(f"Report Type: {report_type}")
            print(f"Base Reward: {base_reward}")
            print(f"Quality Adjustment: {quality_adjusted:.1f}")
            print(f"LLM Trigger Bonus: {llm_trigger_bonus}")
            print(f"Total Reward: {total_reward:.1f}")

        return total_reward

    # Utility Functions
    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """
        Get points along a line using Bresenham's algorithm.

        Args:
            x1, y1: Starting coordinates
            x2, y2: Ending coordinates

        Returns:
            List of (x,y) tuples representing points along the line
        """
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x2, y2))
        return points

    def _capture_positions(self, unit_id: int) -> Dict:
        """
        Capture positions of all unit elements for visualization.

        Args:
            unit_id: ID of unit to capture

        Returns:
            Dictionary with unit positions and properties
        """
        unit_type = self.get_unit_property(unit_id, 'type')
        unit_string = self.get_unit_property(unit_id, 'string_id')

        positions = []

        # Helper to find leader/members
        def get_unit_leader(unit_id):
            children = self.get_unit_children(unit_id)
            return next((child for child in children
                         if self.get_unit_property(child, 'is_leader', False)), None)

        def get_unit_members(unit_id):
            children = self.get_unit_children(unit_id)
            return [child for child in children
                    if not self.get_unit_property(child, 'is_leader', False)]

        # Add leader
        leader_id = get_unit_leader(unit_id)
        if leader_id:
            positions.append({
                'role': self.get_unit_property(leader_id, 'role'),
                'position': self.get_unit_position(leader_id),
                'is_leader': True,
                'health': self.get_unit_property(leader_id, 'health', 100),
                'suppressed': hasattr(self, 'combat_manager') and
                              leader_id in self.combat_manager.suppressed_units
            })

        # Add members
        for member_id in get_unit_members(unit_id):
            positions.append({
                'role': self.get_unit_property(member_id, 'role'),
                'position': self.get_unit_position(member_id),
                'is_leader': False,
                'health': self.get_unit_property(member_id, 'health', 100),
                'suppressed': hasattr(self, 'combat_manager') and
                              member_id in self.combat_manager.suppressed_units
            })

        return {
            'unit_type': unit_type,
            'unit_id': unit_string,
            'positions': positions,
            'timestamp': datetime.now().isoformat()
        }

    def _format_unit_properties(self, properties: Dict) -> np.ndarray:
        """
        Format unit properties into fixed-size array for ML/RL.

        Args:
            properties: Dictionary of unit properties

        Returns:
            Fixed-size numpy array with formatted properties
        """
        # Initialize array with default values
        props = np.zeros(10, dtype=np.float32)

        # Standard property mapping
        if 'type' in properties:
            props[0] = float(properties['type'].value if isinstance(properties['type'], Enum)
                             else properties['type'])
        if 'health' in properties:
            props[1] = float(properties['health'])
        if 'orientation' in properties:
            props[2] = float(properties['orientation'])
        if 'ammo_primary' in properties:
            props[3] = float(properties['ammo_primary'])
        if 'ammo_secondary' in properties:
            props[4] = float(properties['ammo_secondary'])
        if 'observation_range' in properties:
            props[5] = float(properties['observation_range'])
        if 'engagement_range' in properties:
            props[6] = float(properties['engagement_range'])

        # Combat status properties
        if hasattr(self, 'combat_manager'):
            # Get suppression level if available
            unit_id = properties.get('unit_id')
            if unit_id and unit_id in self.combat_manager.suppressed_units:
                props[7] = float(self.combat_manager.suppressed_units[unit_id]['level'])
                props[8] = float(self.combat_manager.suppressed_units[unit_id]['duration'])

        return props

    def _generate_unit_id(self) -> int:
        """Generate unique unit ID."""
        unit_id = self._next_unit_id
        self._next_unit_id += 1
        return unit_id

    def _get_default_formation(self, unit_type: UnitType) -> str:
        """
        Get default formation for unit type.

        Args:
            unit_type: Type of unit to get formation for

        Returns:
            String name of default formation
        """
        defaults = {
            UnitType.INFANTRY_TEAM: "team_wedge_right",
            UnitType.WEAPONS_TEAM: "team_column",
            UnitType.INFANTRY_SQUAD: "squad_column_team_wedge",
            UnitType.INFANTRY_PLATOON: "platoon_column"
        }
        return defaults.get(unit_type, "column")
