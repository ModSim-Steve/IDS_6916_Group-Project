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

# from US_Army_PLT_Composition_vTest import US_IN_Role


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
    AREA = auto()  # Area fire/suppression


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

        # Print detailed engagement results if debug mode is enabled
        if self.env.debug_level > 0:
            team_string = self.env.get_unit_property(team_id, 'string_id', str(team_id))
            print(f"\nTeam {team_id} ({team_string}) engagement complete:")
            print(f"- Total hits: {team_results['total_hits']}")
            print(f"- Total damage: {team_results['total_damage']:.1f}")
            print(f"- Suppression level: {team_results['suppression_level']:.2f}")
            print(f"- Ammunition expended: {team_results['ammo_expended']}")
            print(f"- Effectiveness: {team_results['effectiveness']:.2f}")
            print(f"- Targets hit: {team_results['targets_hit']}")
            print(f"- Time steps: {team_results['time_steps']}")

        return team_results

    def execute_squad_engagement(self, squad_id: int,
                                 target_area: Tuple[int, int],
                                 engagement_type: EngagementType = EngagementType.POINT) -> Dict:
        """
        Execute coordinated squad-level engagement by directing both fire teams plus squad leader.

        This simplified approach:
        1. Identifies the two fire teams (ATM and BTM)
        2. Directs both teams to engage the same target area
        3. Also has the squad leader engage if possible
        4. Aggregates the results

        Args:
            squad_id: ID of squad
            target_area: Central target area (x,y)
            engagement_type: POINT or AREA engagement type

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

        if self.env.debug_level > 0:
            print(f"\nExecuting squad engagement for squad {squad_id} at target {target_area}")

        # Find the two fire teams (ATM and BTM)
        alpha_team = None
        bravo_team = None
        squad_leader = None

        # Identify squad components
        for unit_id in self.env.get_unit_children(squad_id):
            unit_type = self.env.get_unit_property(unit_id, 'type')
            string_id = self.env.get_unit_property(unit_id, 'string_id', '')

            # Check for teams
            if 'ATM' in string_id or ('TEAM' in string_id and 'A' in string_id):
                alpha_team = unit_id
            elif 'BTM' in string_id or ('TEAM' in string_id and 'B' in string_id):
                bravo_team = unit_id

            # Find squad leader - direct child of squad that is flagged as leader
            if self.env.get_unit_property(unit_id, 'is_leader', False):
                squad_leader = unit_id

        if self.env.debug_level > 0:
            print(
                f"Squad components found: Alpha Team: {alpha_team}, Bravo Team: {bravo_team}, Squad Leader: {squad_leader}")

        # Control parameters for team engagements
        control_params = {
            'max_rounds': 15 if engagement_type == EngagementType.POINT else 30,
            'suppress_only': engagement_type == EngagementType.AREA,
            'area_radius': 0 if engagement_type == EngagementType.POINT else 3,
            'adjust_for_fire_rate': True
        }

        # Track participating teams
        participating_teams = []

        # Execute team engagements - Alpha Team
        if alpha_team:
            # Check if any member of Alpha Team can engage the target
            can_team_engage = False
            for member_id in self.env.get_unit_children(alpha_team):
                if self.validate_target(member_id, target_area):
                    can_team_engage = True
                    break

            if can_team_engage:
                if self.env.debug_level > 0:
                    print(f"Directing Alpha Team ({alpha_team}) to engage")

                alpha_results = self.execute_team_engagement(
                    team_id=alpha_team,
                    target_pos=target_area,
                    engagement_type=engagement_type,
                    control_params=control_params
                )

                # Only aggregate results and add to participating teams if there was actual engagement
                if alpha_results.get('ammo_expended', 0) > 0:
                    # Aggregate results
                    squad_results['total_hits'] += alpha_results.get('total_hits', 0)
                    squad_results['targets_hit'].update(alpha_results.get('targets_hit', []))
                    squad_results['total_damage'] += alpha_results.get('total_damage', 0.0)
                    squad_results['suppression_level'] = max(
                        squad_results['suppression_level'],
                        alpha_results.get('suppression_level', 0.0)
                    )
                    squad_results['ammo_expended'] += alpha_results.get('ammo_expended', 0)
                    squad_results['time_steps'] = max(
                        squad_results['time_steps'],
                        alpha_results.get('time_steps', 0)
                    )
                    squad_results['hit_locations'].extend(alpha_results.get('hit_locations', []))
                    participating_teams.append(alpha_team)
                elif self.env.debug_level > 0:
                    print(f"Alpha Team ({alpha_team}) did not expend any ammunition")
            elif self.env.debug_level > 0:
                print(f"Alpha Team ({alpha_team}) cannot engage target - no valid members")

        # Execute team engagements - Bravo Team
        if bravo_team:
            # Check if any member of Bravo Team can engage the target
            can_team_engage = False
            for member_id in self.env.get_unit_children(bravo_team):
                if self.validate_target(member_id, target_area):
                    can_team_engage = True
                    break

            if can_team_engage:
                if self.env.debug_level > 0:
                    print(f"Directing Bravo Team ({bravo_team}) to engage")

                bravo_results = self.execute_team_engagement(
                    team_id=bravo_team,
                    target_pos=target_area,
                    engagement_type=engagement_type,
                    control_params=control_params
                )

                # Only aggregate results and add to participating teams if there was actual engagement
                if bravo_results.get('ammo_expended', 0) > 0:
                    # Aggregate results
                    squad_results['total_hits'] += bravo_results.get('total_hits', 0)
                    squad_results['targets_hit'].update(bravo_results.get('targets_hit', []))
                    squad_results['total_damage'] += bravo_results.get('total_damage', 0.0)
                    squad_results['suppression_level'] = max(
                        squad_results['suppression_level'],
                        bravo_results.get('suppression_level', 0.0)
                    )
                    squad_results['ammo_expended'] += bravo_results.get('ammo_expended', 0)
                    squad_results['time_steps'] = max(
                        squad_results['time_steps'],
                        bravo_results.get('time_steps', 0)
                    )
                    squad_results['hit_locations'].extend(bravo_results.get('hit_locations', []))
                    participating_teams.append(bravo_team)
                elif self.env.debug_level > 0:
                    print(f"Bravo Team ({bravo_team}) did not expend any ammunition")
            elif self.env.debug_level > 0:
                print(f"Bravo Team ({bravo_team}) cannot engage target - no valid members")

        # Have the Squad Leader engage if possible
        if squad_leader:
            # Check if Squad Leader can engage the target
            can_engage = self.validate_target(squad_leader, target_area)

            if can_engage:
                if self.env.debug_level > 0:
                    print(f"Directing Squad Leader ({squad_leader}) to engage")

                # Create fire control object for squad leader
                fire_control = FireControl(
                    target_area=target_area,
                    area_radius=0 if engagement_type == EngagementType.POINT else 3,
                    max_rounds=10,  # Less rounds for individual
                    suppress_only=engagement_type == EngagementType.AREA,
                    adjust_for_fire_rate=True
                )

                # Execute squad leader engagement
                sl_results = self._execute_member_engagement(squad_leader, fire_control)

                # Only aggregate results if there was actual engagement
                if sl_results.rounds_expended > 0:
                    # Aggregate results
                    squad_results['total_hits'] += sl_results.hits
                    squad_results['targets_hit'].update(sl_results.targets_hit)
                    squad_results['total_damage'] += sl_results.damage_dealt
                    squad_results['suppression_level'] = max(
                        squad_results['suppression_level'],
                        sl_results.suppression_effect
                    )
                    squad_results['ammo_expended'] += sl_results.rounds_expended
                    squad_results['time_steps'] = max(
                        squad_results['time_steps'],
                        sl_results.time_steps
                    )
                    squad_results['hit_locations'].extend(sl_results.hit_locations)
                elif self.env.debug_level > 0:
                    print(f"Squad Leader ({squad_leader}) did not expend any ammunition")
            elif self.env.debug_level > 0:
                print(f"Squad Leader ({squad_leader}) cannot engage target")

        # Set participating teams
        squad_results['participating_teams'] = participating_teams

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

        # print(f"[DEBUG VALIDATE] Distance: {distance}, Engagement range: {engagement_range}, Observation range: {observation_range}")

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
        """
        Get unit's current ammunition with improved initialization.

        Args:
            unit_id: ID of unit
            weapon_type: Which weapon to get ammo for ('primary' or 'secondary')

        Returns:
            Current ammunition amount
        """
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

        # Get capacity from weapon
        capacity = weapon.ammo_capacity if hasattr(weapon, 'ammo_capacity') else 0

        # Store in tracking
        self.ammo_tracking[unit_id][weapon_type] = capacity

        return capacity

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
        """Get elevation type at position with safety checks."""
        try:
            x, y = position
            # Ensure integers and within bounds
            x, y = int(x), int(y)

            if not (0 <= x < self.width and 0 <= y < self.height):
                # Default to ground level for out-of-bounds
                return ElevationType.GROUND_LEVEL

            return ElevationType(self.state_tensor[y, x, 1])
        except Exception as e:
            print(f"[DEBUG] Error getting elevation type: {e}")
            return ElevationType.GROUND_LEVEL  # Default


class AgentManager:
    """
    Manages identification and state tracking of agents within the military environment.
    Handles mapping between agent actions and corresponding unit functions.
    """

    def __init__(self, env):
        """Initialize with reference to parent environment."""
        self.env = env
        self.agent_ids = []
        self.agent_types = {}  # Maps agent_id -> unit type (SQUAD, TEAM, WEAPONS_TEAM)
        self.agent_policies = {}  # For mapping different policies based on unit type
        self.original_agents = set()  # Track original agents for succession purposes

        # Add debug_level attribute, defaulting to parent environment's level if available
        self.debug_level = getattr(env, 'debug_level', 0)

        # Fields for consistent agent ID mapping
        self.role_to_agent_id = {}
        self.agent_id_to_role = {}
        self.unit_id_to_agent_id = {}
        self.initialized_role_mapping = False

    def initialize_agent_role_mapping(self):
        """
        Create a mapping from tactical roles to consistent agent IDs.
        Called once at the beginning of training.
        """
        # Only initialize if not already done
        if self.initialized_role_mapping:
            print("Role mapping already initialized, skipping...")
            return

        # Clear any existing mappings to ensure clean state
        self.role_to_agent_id = {}
        self.agent_id_to_role = {}
        self.unit_id_to_agent_id = {}

        # Define the consistent role-to-agent mapping
        self.role_to_agent_id = {
            "1SQD_Leader": 1,
            "2SQD_Leader": 2,
            "3SQD_Leader": 3,
            "GTM1_Leader": 4,
            "GTM2_Leader": 5,
            "JTM1_Leader": 6,
            "JTM2_Leader": 7
        }
        self.agent_id_to_role = {v: k for k, v in self.role_to_agent_id.items()}
        self.initialized_role_mapping = True

        print(f"Initialized role-to-agent mapping with {len(self.role_to_agent_id)} roles")

    def map_current_units_to_agent_ids(self, platoon_id=None):
        """
        After environment reset, map current unit IDs to consistent agent IDs.
        Fixed version that specifically targets weapons teams and properly sets is_agent flag.

        Args:
            platoon_id: ID of the platoon

        Returns:
            List of agent IDs
        """
        # Ensure mapping is initialized
        if not self.initialized_role_mapping:
            self.initialize_agent_role_mapping()

        # Clear previous mapping to avoid duplicates or stale entries
        self.unit_id_to_agent_id = {}

        # Find the platoon if not provided
        if platoon_id is None:
            for unit_id in self.env.state_manager.active_units:
                unit_type = self.env.get_unit_property(unit_id, 'type')
                unit_type_str = str(unit_type)
                if 'INFANTRY_PLATOON' in unit_type_str:
                    platoon_id = unit_id
                    break

        if not platoon_id:
            print("No platoon found, cannot map agents")
            return []

        # Get all platoon children first
        platoon_children = self.env.get_unit_children(platoon_id)
        print(f"Platoon has {len(platoon_children)} direct children")

        # Store all squad leaders and weapons team leaders
        squad_leaders = []
        weapons_teams = []

        # STEP 1: First pass - identify squads vs weapons teams
        for unit_id in platoon_children:
            unit_string = self.env.get_unit_property(unit_id, 'string_id', '')
            print(f"Checking unit: {unit_string} (ID: {unit_id})")

            # Check if this is a weapons team by string ID
            if 'GTM1' in unit_string:
                weapons_teams.append((unit_id, 'GTM1', 4))
                print(f"Found GTM1: {unit_string}")
            elif 'GTM2' in unit_string:
                weapons_teams.append((unit_id, 'GTM2', 5))
                print(f"Found GTM2: {unit_string}")
            elif 'JTM1' in unit_string:
                weapons_teams.append((unit_id, 'JTM1', 6))
                print(f"Found JTM1: {unit_string}")
            elif 'JTM2' in unit_string:
                weapons_teams.append((unit_id, 'JTM2', 7))
                print(f"Found JTM2: {unit_string}")
            # Check if this is a squad
            elif '1SQD' in unit_string:
                for child_id in self.env.get_unit_children(unit_id):
                    if self.env.get_unit_property(child_id, 'is_leader', False):
                        squad_leaders.append((child_id, '1SQD', 1))
                        print(f"Found 1SQD leader: {child_id}")
                        break
            elif '2SQD' in unit_string:
                for child_id in self.env.get_unit_children(unit_id):
                    if self.env.get_unit_property(child_id, 'is_leader', False):
                        squad_leaders.append((child_id, '2SQD', 2))
                        print(f"Found 2SQD leader: {child_id}")
                        break
            elif '3SQD' in unit_string:
                for child_id in self.env.get_unit_children(unit_id):
                    if self.env.get_unit_property(child_id, 'is_leader', False):
                        squad_leaders.append((child_id, '3SQD', 3))
                        print(f"Found 3SQD leader: {child_id}")
                        break

        # STEP 2: For weapons teams, find or designate their leaders
        weapons_leaders = []
        for team_id, team_type, agent_id in weapons_teams:
            team_string = self.env.get_unit_property(team_id, 'string_id', '')

            # First try to find a designated leader
            leader_found = False
            for child_id in self.env.get_unit_children(team_id):
                if self.env.get_unit_property(child_id, 'is_leader', False):
                    weapons_leaders.append((child_id, team_type, agent_id))
                    leader_found = True
                    print(f"Found designated leader for {team_string}: {child_id}")
                    break

            # If no designated leader, use the first member as leader
            if not leader_found:
                children = self.env.get_unit_children(team_id)
                if children:
                    weapons_leaders.append((children[0], team_type, agent_id))
                    print(f"Using first member as leader for {team_string}: {children[0]}")
                else:
                    # Last resort: use the team itself
                    weapons_leaders.append((team_id, team_type, agent_id))
                    print(f"Using team itself as agent for {team_string}: {team_id}")

        print(f"Final counts: {len(squad_leaders)} squad leaders, {len(weapons_leaders)} weapons team leaders")

        # STEP 3: Map all the agent IDs
        for leader_id, team_type, agent_id in squad_leaders:
            self.unit_id_to_agent_id[leader_id] = agent_id
            unit_string = self.env.get_unit_property(leader_id, 'string_id', '')
            print(f"Mapped squad unit {leader_id} ({unit_string}) to agent ID {agent_id}")

            # Set agent type
            self.agent_types[agent_id] = 'SQUAD'

            # Set the is_agent property
            self.env.update_unit_property(leader_id, 'is_agent', True)

        for leader_id, team_type, agent_id in weapons_leaders:
            self.unit_id_to_agent_id[leader_id] = agent_id
            unit_string = self.env.get_unit_property(leader_id, 'string_id', '')
            print(f"Mapped weapons team unit {leader_id} ({unit_string}) to agent ID {agent_id}")

            # Set agent type
            self.agent_types[agent_id] = 'WEAPONS_TEAM'

            # Set the is_agent property
            self.env.update_unit_property(leader_id, 'is_agent', True)

        # Create a list of agent IDs from the mapping
        agent_ids = sorted(set(self.unit_id_to_agent_id.values()))

        # Store agent IDs in the environment
        if hasattr(self.env, 'agent_ids'):
            self.env.agent_ids = agent_ids

        print(f"Mapped {len(agent_ids)} agents to consistent IDs: {agent_ids}")

        return agent_ids

    def get_current_unit_id(self, agent_id):
        """
        Get the current unit ID for a consistent agent ID.

        Args:
            agent_id: The consistent agent ID

        Returns:
            Current unit ID or None if not found
        """
        for unit_id, mapped_agent_id in self.unit_id_to_agent_id.items():
            if mapped_agent_id == agent_id:
                return unit_id
        return None

    def get_agent_id(self, unit_id):
        """
        Get the consistent agent ID for a unit ID.

        Args:
            unit_id: The current unit ID

        Returns:
            Consistent agent ID or None if not found
        """
        return self.unit_id_to_agent_id.get(unit_id)

    def identify_agents_from_platoon(self, platoon_id):
        """
        Identify agents from a platoon structure following US Army doctrine.
        For a standard infantry platoon, this would result in:
        - 3 Squad Leader agents
        - 4 Weapons Team Leader agents (2 gun teams, 2 javelin teams)

        Args:
            platoon_id: ID of platoon to extract agents from

        Returns:
            List of agent IDs
        """
        print(f"\n[DEBUG AGENTS] Identifying agents from platoon {platoon_id}")
        old_agent_ids = self.agent_ids.copy() if hasattr(self, 'agent_ids') and self.agent_ids else []
        print(f"[DEBUG AGENTS] Previous agent_ids: {old_agent_ids}")

        self.agent_ids = []
        self.agent_types = {}
        self.original_agents = set()

        # Find squads in platoon
        squads = []
        for unit_id in self.env.get_unit_children(platoon_id):
            unit_type = self.env.get_unit_property(unit_id, 'type')
            string_id = self.env.get_unit_property(unit_id, 'string_id', '')
            if unit_type == UnitType.INFANTRY_SQUAD or (
                    "1SQD" in string_id or "2SQD" in string_id or "3SQD" in string_id):
                squads.append(unit_id)

        if self.env.debug_level > 0:
            print(f"Found {len(squads)} squads in platoon {platoon_id}")

        # Find weapons teams in platoon
        weapons_teams = []
        for unit_id in self.env.get_unit_children(platoon_id):
            unit_type = self.env.get_unit_property(unit_id, 'type')
            string_id = self.env.get_unit_property(unit_id, 'string_id', '')
            if unit_type == UnitType.WEAPONS_TEAM or ('GTM' in string_id or 'JTM' in string_id):
                weapons_teams.append(unit_id)

        if self.env.debug_level > 0:
            print(f"Found {len(weapons_teams)} weapons teams in platoon {platoon_id}")

        from US_Army_PLT_Composition_vTest import US_IN_Role
        # For each squad, find the squad leader
        for squad_id in squads:
            squad_leader_found = False
            squad_string = self.env.get_unit_property(squad_id, 'string_id', str(squad_id))

            # First look directly in squad children
            for child_id in self.env.get_unit_children(squad_id):
                role = self.env.get_unit_property(child_id, 'role')
                is_leader = self.env.get_unit_property(child_id, 'is_leader', False)
                role_name = "Unknown"
                if isinstance(role, int):
                    try:
                        role_name = US_IN_Role(role).name
                    except:
                        role_name = str(role)
                else:
                    role_name = str(role)

                # Debug info
                if self.env.debug_level > 1:
                    print(f"Squad {squad_string} child {child_id}: role={role_name}, is_leader={is_leader}")

                # Check if this is the squad leader
                if is_leader and (
                        role == US_IN_Role.SQUAD_LEADER.value or 'SQUAD_LEADER' in role_name or 'SL' in str(role)):
                    self.agent_ids.append(child_id)
                    self.agent_types[child_id] = 'SQUAD'
                    self.original_agents.add(child_id)
                    squad_leader_found = True

                    if self.env.debug_level > 0:
                        print(f"Added Squad Leader {child_id} ({role_name}) as agent for {squad_string}")
                    break

            # If squad leader not found among direct children, check teams
            if not squad_leader_found:
                for team_id in self.env.get_unit_children(squad_id):
                    team_type = self.env.get_unit_property(team_id, 'type')
                    if team_type != UnitType.INFANTRY_TEAM:
                        continue

                    for member_id in self.env.get_unit_children(team_id):
                        role = self.env.get_unit_property(member_id, 'role')
                        is_leader = self.env.get_unit_property(member_id, 'is_leader', False)
                        role_name = "Unknown"
                        if isinstance(role, int):
                            try:
                                role_name = US_IN_Role(role).name
                            except:
                                role_name = str(role)
                        else:
                            role_name = str(role)

                        # Debug info
                        if self.env.debug_level > 1:
                            print(f"Checking team member {member_id}: role={role_name}, is_leader={is_leader}")

                        # Check if this could be a squad leader
                        if (role == US_IN_Role.SQUAD_LEADER.value or 'SQUAD_LEADER' in role_name or 'SL' in str(role)):
                            self.agent_ids.append(member_id)
                            self.agent_types[member_id] = 'SQUAD'
                            self.original_agents.add(member_id)
                            squad_leader_found = True

                            if self.env.debug_level > 0:
                                print(f"Added Squad Leader {member_id} ({role_name}) as agent for {squad_string}")
                            break

                    if squad_leader_found:
                        break

            if not squad_leader_found and self.env.debug_level > 0:
                print(f"WARNING: No Squad Leader found for squad {squad_string}")

        # For each weapons team, set the team leader as agent
        for team_id in weapons_teams:
            team_string = self.env.get_unit_property(team_id, 'string_id', str(team_id))
            team_leader_found = False

            # Try to find an existing leader
            members = self.env.get_unit_children(team_id)
            for member_id in members:
                is_leader = self.env.get_unit_property(member_id, 'is_leader', False)
                role = self.env.get_unit_property(member_id, 'role')
                role_name = "Unknown"
                if isinstance(role, int):
                    try:
                        role_name = US_IN_Role(role).name
                    except:
                        role_name = str(role)
                else:
                    role_name = str(role)

                # Debug info
                if self.env.debug_level > 1:
                    print(f"Weapons team {team_string} member {member_id}: role={role_name}, is_leader={is_leader}")

                if is_leader:
                    self.agent_ids.append(member_id)
                    self.agent_types[member_id] = 'WEAPONS_TEAM'
                    self.original_agents.add(member_id)
                    team_leader_found = True

                    if self.env.debug_level > 0:
                        print(f"Added Weapons Team Leader {member_id} ({role_name}) as agent for {team_string}")
                    break

            # If no leader exists, designate first member as leader
            if not team_leader_found and members:
                leader_id = members[0]

                # Make them the leader
                self.env.update_unit_property(leader_id, 'is_leader', True)

                role = self.env.get_unit_property(leader_id, 'role')
                role_name = "Unknown"
                if isinstance(role, int):
                    try:
                        role_name = US_IN_Role(role).name
                    except:
                        role_name = str(role)
                else:
                    role_name = str(role)

                # Add as agent
                self.agent_ids.append(leader_id)
                self.agent_types[leader_id] = 'WEAPONS_TEAM'
                self.original_agents.add(leader_id)

                if self.env.debug_level > 0:
                    print(f"Designated {leader_id} ({role_name}) as leader and agent for weapons team {team_string}")

        # Check if we can reuse any previously identified agents for continuity
        if old_agent_ids:
            print(f"[DEBUG AGENTS] Checking if we can maintain any previous agents")
            # Check which old agents still exist and are valid
            valid_old_agents = [agent_id for agent_id in old_agent_ids
                                if agent_id in self.env.state_manager.active_units]

            if valid_old_agents:
                # If we have both old and new valid agents, prefer keeping the old ones
                # for better policy training continuity
                if len(valid_old_agents) >= len(self.agent_ids) * 0.5:  # At least half are valid
                    print(f"[DEBUG AGENTS] Maintaining {len(valid_old_agents)} previous agents for continuity")
                    self.agent_ids = valid_old_agents
                    # Also update agent_types for these agents
                    for agent_id in self.agent_ids:
                        # Try to determine type if missing
                        if agent_id not in self.agent_types:
                            parent_id = self.env.get_unit_property(agent_id, 'parent_id')
                            if parent_id:
                                parent_type = self.env.get_unit_property(parent_id, 'type')
                                if parent_type == UnitType.INFANTRY_SQUAD:
                                    self.agent_types[agent_id] = 'SQUAD'
                                elif parent_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM]:
                                    self.agent_types[agent_id] = 'WEAPONS_TEAM'

        if self.env.debug_level > 0:
            print(f"Total identified agents: {len(self.agent_ids)}")

        # Mark all agents in the environment
        for agent_id in self.agent_ids:
            self.env.update_unit_property(agent_id, 'is_agent', True)

        print(f"[DEBUG AGENTS] Identified {len(self.agent_ids)} agents: {self.agent_ids}")

        return self.agent_ids

    def get_actions_for_agent(self, agent_id):
        """
        Get valid actions for an agent based on its type and state.

        Args:
            agent_id: ID of agent

        Returns:
            Dictionary of valid actions
        """
        valid_actions = []

        # Get agent type
        agent_type = self.agent_types.get(agent_id)

        # Check if agent exists and is alive
        if not agent_id in self.env.state_manager.active_units:
            return valid_actions

        if self.env.get_unit_property(agent_id, 'health', 0) <= 0:
            return valid_actions

        # Get unit properties
        parent_id = self.env.get_unit_property(agent_id, 'parent_id')
        if not parent_id:
            return valid_actions

        # Different action sets based on agent type
        if agent_type == 'SQUAD':
            # Squad leaders can execute squad movements and engagements
            valid_actions.extend([
                {'action_type': 0, 'name': 'MOVE'},  # Maps to execute_squad_movement with TRAVELING
                {'action_type': 1, 'name': 'ENGAGE'},  # Maps to execute_squad_engagement with POINT
                {'action_type': 2, 'name': 'SUPPRESS'},  # Maps to execute_squad_engagement with AREA
                {'action_type': 3, 'name': 'BOUND'},  # Maps to execute_squad_movement with BOUNDING
                {'action_type': 4, 'name': 'HALT'},  # Maps to halt action
                {'action_type': 5, 'name': 'CHANGE_FORMATION'}  # Maps to formation change action
            ])

        elif agent_type == 'WEAPONS_TEAM':
            # Weapons teams primarily focus on suppression and engagement
            valid_actions.extend([
                {'action_type': 0, 'name': 'MOVE'},  # Maps to US_IN_execute_movement
                {'action_type': 1, 'name': 'ENGAGE'},  # Maps to execute_team_engagement with POINT
                {'action_type': 2, 'name': 'SUPPRESS'},  # Maps to execute_team_engagement with AREA
                {'action_type': 4, 'name': 'HALT'},  # Maps to halt action
                {'action_type': 5, 'name': 'CHANGE_FORMATION'}  # Maps to formation change action
            ])

        elif agent_type == 'INFANTRY_TEAM':
            # Infantry teams have balanced movement and engagement capabilities
            valid_actions.extend([
                {'action_type': 0, 'name': 'MOVE'},  # Maps to US_IN_execute_movement
                {'action_type': 1, 'name': 'ENGAGE'},  # Maps to execute_team_engagement with POINT
                {'action_type': 2, 'name': 'SUPPRESS'},  # Maps to execute_team_engagement with AREA
                {'action_type': 4, 'name': 'HALT'},  # Maps to halt action
                {'action_type': 5, 'name': 'CHANGE_FORMATION'}  # Maps to formation change action
            ])

        return valid_actions

    def execute_agent_action(self, agent_id, action):
        """
        Execute an action for an agent by mapping to appropriate unit functions.
        Fixed version that properly handles routing for squad movements and other actions.
        Improved to handle errors gracefully.

        Args:
            agent_id: ID of agent
            action: Action dictionary from MARL action space

        Returns:
            Execution results
        """
        # Extract the action type
        action_type = action.get('action_type')

        # Debug info
        if self.env.debug_level > 0:
            print(f"DEBUG: Executing action {action_type} for agent {agent_id}")
            print(f"DEBUG: Action details: {action}")

        # Get agent type from the agent_types dictionary if available
        agent_type = self.agent_types.get(agent_id)

        # If agent_type is None, determine it from unit properties
        if agent_type is None:
            # Get unit type to determine agent type
            unit_type = self.env.get_unit_property(agent_id, 'type')
            unit_type_str = str(unit_type)

            # Try to convert to string if it's an enum
            if hasattr(unit_type, 'name'):
                unit_type_str = unit_type.name

            # Determine agent type based on unit type
            if 'INFANTRY_SQUAD' in unit_type_str or 'SQD' in unit_type_str:
                agent_type = 'SQUAD'
            elif 'WEAPONS_TEAM' in unit_type_str or 'GTM' in unit_type_str or 'JTM' in unit_type_str:
                agent_type = 'WEAPONS_TEAM'
            elif 'INFANTRY_TEAM' in unit_type_str or 'TM' in unit_type_str:
                agent_type = 'TEAM'
            else:
                # Check if this is a leader
                is_leader = self.env.get_unit_property(agent_id, 'is_leader', False)
                role = self.env.get_unit_property(agent_id, 'role')
                role_name = str(role)

                if hasattr(role, 'name'):
                    role_name = role.name

                if is_leader and 'SQUAD_LEADER' in role_name:
                    agent_type = 'SQUAD'
                elif is_leader and ('TEAM_LEADER' in role_name or 'TL' in role_name):
                    agent_type = 'TEAM'
                else:
                    # Default assumption based on parent
                    parent_id = self.env.get_unit_property(agent_id, 'parent_id')
                    if parent_id:
                        parent_type = self.env.get_unit_property(parent_id, 'type')
                        parent_type_str = str(parent_type)

                        if hasattr(parent_type, 'name'):
                            parent_type_str = parent_type.name

                        if 'INFANTRY_SQUAD' in parent_type_str:
                            agent_type = 'SQUAD'
                        elif 'TEAM' in parent_type_str:
                            agent_type = 'TEAM'

        # Determine if this agent is a leader position
        is_leader = self.env.get_unit_property(agent_id, 'is_leader', False)

        # Get parent information
        parent_id = self.env.get_unit_property(agent_id, 'parent_id')

        if self.env.debug_level > 0:
            print(f"DEBUG: Agent type: {agent_type}, Is leader: {is_leader}")

        # CRITICAL FIX: Determine the correct target unit based on agent type
        if agent_type == 'SQUAD' and is_leader and parent_id:
            # Squad leader's actions operate on the squad (parent)
            target_unit = parent_id
        elif (agent_type in ['WEAPONS_TEAM', 'TEAM']) and is_leader and parent_id:
            # Team leader's actions operate on the team (parent)
            target_unit = parent_id
        else:
            # Check if this is the unit itself (like a squad or team entity)
            unit_children = self.env.get_unit_children(agent_id)
            if unit_children:
                # This is a unit entity itself, not a position
                target_unit = agent_id
            elif parent_id:
                # This is a position, use parent
                target_unit = parent_id
            else:
                # Fallback to agent ID
                target_unit = agent_id

        if self.env.debug_level > 0:
            print(f"DEBUG: Target unit for action: {target_unit}")
            target_type = self.env.get_unit_property(target_unit, 'type')
            print(f"DEBUG: Unit type: {target_type}")

        # Store the result of the action execution
        result = None

        # Map action to appropriate function based on action type
        # MOVEMENT ACTIONS - Type 0 or 3
        if action_type in [0, 3]:  # MOVE or BOUND
            # Extract movement parameters
            movement_params = action.get('movement_params', {})
            direction = movement_params.get('direction', (0, 0))
            distance = movement_params.get('distance', [1])

            # Convert numpy array to tuple if needed
            if isinstance(direction, np.ndarray):
                direction = tuple(direction)

            # Convert distance to int if it's an array
            if isinstance(distance, (list, np.ndarray)):
                distance = int(distance[0]) if len(distance) > 0 else 1

            if self.env.debug_level > 0:
                print(f"DEBUG: Moving {target_unit} in direction {direction} for distance {distance}")

            try:
                # Determine unit type for proper movement function
                unit_type = self.env.get_unit_property(target_unit, 'type')
                unit_type_str = str(unit_type)

                # Try to convert to string if it's an enum
                if hasattr(unit_type, 'name'):
                    unit_type_str = unit_type.name

                # FIXED: Properly check for squad unit type using string comparison
                is_squad = 'INFANTRY_SQUAD' in unit_type_str or 'SQD' in unit_type_str

                # Determine movement technique based on action type
                if action_type == 3:  # BOUND
                    if is_squad:
                        from US_Army_PLT_Composition_vTest import execute_squad_movement, MovementTechnique
                        result = execute_squad_movement(
                            env=self.env,
                            squad_id=target_unit,
                            direction=direction,
                            distance=distance,
                            technique=MovementTechnique.BOUNDING,
                            debug_level=self.env.debug_level
                        )
                    else:
                        # Teams don't have bounding movement - use regular movement
                        from US_Army_PLT_Composition_vTest import US_IN_execute_movement
                        result = US_IN_execute_movement(
                            env=self.env,
                            unit_id=target_unit,
                            direction=direction,
                            distance=distance,
                            debug_level=self.env.debug_level
                        )
                else:  # MOVE
                    if is_squad:
                        from US_Army_PLT_Composition_vTest import execute_squad_movement, MovementTechnique
                        result = execute_squad_movement(
                            env=self.env,
                            squad_id=target_unit,
                            direction=direction,
                            distance=distance,
                            technique=MovementTechnique.TRAVELING,
                            debug_level=self.env.debug_level
                        )
                    else:
                        from US_Army_PLT_Composition_vTest import US_IN_execute_movement
                        result = US_IN_execute_movement(
                            env=self.env,
                            unit_id=target_unit,
                            direction=direction,
                            distance=distance,
                            debug_level=self.env.debug_level
                        )

                # NEW: Update parent unit position based on leader position after movement
                if unit_type_str in ["INFANTRY_TEAM", "INFANTRY_SQUAD",
                                     "WEAPONS_TEAM"] or "TEAM" in unit_type_str or "SQUAD" in unit_type_str:
                    # Find the leader
                    leader_id = None
                    for member_id in self.env.get_unit_children(target_unit):
                        if self.env.get_unit_property(member_id, 'is_leader', False):
                            leader_id = member_id
                            break

                    # Update unit position to match leader position
                    if leader_id:
                        leader_pos = self.env.get_unit_position(leader_id)
                        if self.env.debug_level > 0:
                            print(f"DEBUG: Updating unit {target_unit} position to match leader at {leader_pos}")
                        self.env.update_unit_position(target_unit, leader_pos)

            except Exception as e:
                print(f"Error executing movement action: {e}")
                import traceback
                traceback.print_exc()
                return None

        # ENGAGEMENT ACTIONS - Type 1 or 2
        elif action_type in [1, 2]:  # ENGAGE or SUPPRESS
            # Extract engagement parameters
            engagement_params = action.get('engagement_params', {})
            target_pos = engagement_params.get('target_pos')
            max_rounds = engagement_params.get('max_rounds', 10)
            suppress_only = bool(engagement_params.get('suppress_only', 0))
            adjust_for_fire_rate = bool(engagement_params.get('adjust_for_fire_rate', 1))

            # Convert numpy array to tuple if needed
            if isinstance(target_pos, np.ndarray):
                target_pos = tuple(target_pos)

            # Convert max_rounds to int if it's an array
            if isinstance(max_rounds, (list, np.ndarray)):
                max_rounds = int(max_rounds[0]) if len(max_rounds) > 0 else 10

            if self.env.debug_level > 0:
                print(f"DEBUG: Engaging from {target_unit} at position {target_pos}")
                print(
                    f"DEBUG: Engagement params: max_rounds={max_rounds}, suppress={suppress_only}, adjust_rate={adjust_for_fire_rate}")

            try:
                # Force suppress_only to True if action type is SUPPRESS
                if action_type == 2:  # SUPPRESS
                    suppress_only = True

                # Get unit type to determine proper engagement function
                unit_type = self.env.get_unit_property(target_unit, 'type')
                unit_type_str = str(unit_type)

                # Try to convert to string if it's an enum
                if hasattr(unit_type, 'name'):
                    unit_type_str = unit_type.name

                # FIXED: Properly check for squad unit type and use appropriate engagement function
                is_squad = 'INFANTRY_SQUAD' in unit_type_str or 'SQD' in unit_type_str or agent_type == 'SQUAD'

                if is_squad:
                    # Use squad engagement for squads
                    result = self.env.combat_manager.execute_squad_engagement(
                        squad_id=target_unit,  # Use squad ID
                        target_area=target_pos,
                        engagement_type=EngagementType.AREA if suppress_only else EngagementType.POINT
                    )
                else:
                    # Use team engagement for teams/weapons teams
                    result = self.env.combat_manager.execute_team_engagement(
                        team_id=target_unit,
                        target_pos=target_pos,
                        engagement_type=EngagementType.AREA if suppress_only else EngagementType.POINT,
                        control_params={
                            'max_rounds': max_rounds,
                            'suppress_only': suppress_only,
                            'area_radius': 3 if suppress_only else 0,
                            'sustained': suppress_only,
                            'adjust_for_fire_rate': adjust_for_fire_rate
                        }
                    )

                # Print detailed ammunition usage for debugging
                if self.env.debug_level > 0:
                    print("Ammunition usage:")
                    for member_id in self.env.get_unit_children(target_unit):
                        role = self.env.get_unit_property(member_id, 'role')
                        role_str = str(role)
                        if hasattr(role, 'name'):
                            role_str = role.name

                        initial_ammo = self.env.combat_manager.ammo_tracking.get(member_id, {}).get('primary', 0)
                        used_ammo = 0
                        if hasattr(self.env, 'combat_manager'):
                            current_ammo = self.env.combat_manager._get_unit_ammo(member_id, 'primary')
                            used_ammo = max(0, initial_ammo - current_ammo)

                        print(f"  Member {member_id} ({role}): Used {used_ammo} rounds")

                    # Print engagement results
                    if result:
                        print("Engagement results:")
                        if isinstance(result, dict):
                            print(f"  Hits: {result.get('total_hits', 0)}")
                            print(f"  Damage: {result.get('total_damage', 0.0)}")
                            print(f"  Rounds expended: {result.get('ammo_expended', 0)}")
                            if suppress_only:
                                print(f"  Suppression effect: {result.get('suppression_level', 0.0):.2f}")
                        else:
                            print(f"  Hits: {getattr(result, 'hits', 0)}")
                            print(f"  Damage: {getattr(result, 'damage_dealt', 0.0)}")
                            print(f"  Rounds expended: {getattr(result, 'rounds_expended', 0)}")
                            if suppress_only:
                                print(f"  Suppression effect: {getattr(result, 'suppression_effect', 0.0):.2f}")
            except Exception as e:
                print(f"Error executing engagement action: {e}")
                import traceback
                traceback.print_exc()
                return None

        # FORMATION CHANGE - Type 4
        elif action_type == 4:  # CHANGE_FORMATION
            try:
                # Get formation index
                formation_index = action.get('formation', 0)

                # Map formation index to formation name
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

                formation = formation_map.get(formation_index, "team_wedge_right")

                if self.env.debug_level > 0:
                    print(f"DEBUG: Changing formation of {target_unit} to {formation}")

                # Apply formation
                from US_Army_PLT_Composition_vTest import US_IN_apply_formation
                US_IN_apply_formation(self.env, target_unit, formation)

                result = {"formation_applied": formation}
            except Exception as e:
                print(f"Error executing formation action: {e}")
                import traceback
                traceback.print_exc()
                return None

        # HALT ACTION - Type 5
        elif action_type == 5:  # HALT
            try:
                # Mark unit as not moving
                self.env.update_unit_property(target_unit, 'moving', False)
                result = {"halted": True}
            except Exception as e:
                print(f"Error executing halt action: {e}")
                return None

        else:
            if self.env.debug_level > 0:
                print(f"DEBUG: Unrecognized action type {action_type}")
            return None

        return result

    def handle_agent_casualty(self, agent_id):
        """
        Enhanced function to handle agent casualty with proper position-based succession.
        Now includes squad consolidation checks after succession.

        Args:
            agent_id: ID of agent position that has a casualty

        Returns:
            The same agent_id since the position remains unchanged
        """
        print(f"\nDEBUG[AgentManager]: Starting agent casualty handling for position {agent_id}")
        print(f"DEBUG[AgentManager]: Current agent_ids: {self.agent_ids}")

        if hasattr(self, 'agent_types'):
            print(f"DEBUG[AgentManager]: Current agent_types: {self.agent_types}")

        # Check if agent exists
        if agent_id not in self.agent_ids:
            print(f"DEBUG[AgentManager]: Agent ID {agent_id} not in agent_ids list")
            return None

        # Get agent type and parent unit
        agent_type = "Unknown"
        if hasattr(self, 'agent_types'):
            agent_type = self.agent_types.get(agent_id, "Unknown")

        parent_id = self.env.get_unit_property(agent_id, 'parent_id')

        # Get current position status
        position_status = self.env.get_unit_property(agent_id, 'position_status', 'occupied')
        current_health = self.env.get_unit_property(agent_id, 'health', 0)
        casualty_handled = self.env.get_unit_property(agent_id, 'casualty_handled', False)
        vacancy_handled = self.env.get_unit_property(agent_id, 'vacancy_handled', False)

        print(f"DEBUG[AgentManager]: Agent {agent_id}, type {agent_type}")
        print(f"DEBUG[AgentManager]: Parent unit: {parent_id}")
        print(f"DEBUG[AgentManager]: Status: health={current_health}, position_status={position_status}, "
              f"casualty_handled={casualty_handled}, vacancy_handled={vacancy_handled}")

        # If this is a vacant position or already handled, just return
        if position_status == 'vacant' or vacancy_handled or casualty_handled or current_health > 0:
            print(
                f"DEBUG[AgentManager]: Position {agent_id} is vacant, already handled, or still alive - skipping agent succession")
            return agent_id

        if not parent_id:
            print(f"DEBUG[AgentManager]: Agent {agent_id} has no parent unit")
            return None

        # Print debugging information
        if self.env.debug_level > 0:
            print(f"DEBUG[AgentManager]: Handling agent casualty for agent position {agent_id}, type {agent_type}")
            print(f"DEBUG[AgentManager]: Parent unit: {parent_id}")

        # IMPORTANT: Don't remove the agent from agent_ids list, as we're
        # preserving the position (the agent_id) and just changing the soldier

        # Use the updated succession handling
        print(f"DEBUG[AgentManager]: Processing succession using updated handler")
        from US_Army_PLT_Composition_vTest import US_IN_handle_leader_casualty
        success = US_IN_handle_leader_casualty(self.env, parent_id)

        print(f"DEBUG[AgentManager]: US_IN_handle_leader_casualty returned {success}")

        if success:
            # Agent position remains the same, but is now occupied by a new soldier
            # Mark the position as an agent again
            self.env.update_unit_property(agent_id, 'is_agent', True)

            # Ensure the agent is in the agent_ids list
            if agent_id not in self.agent_ids:
                self.agent_ids.append(agent_id)

            # Ensure agent exists in env.agent_ids if it exists
            if hasattr(self.env, 'agent_ids') and agent_id not in self.env.agent_ids:
                self.env.agent_ids.append(agent_id)

            # Get updated health and position status after succession
            new_health = self.env.get_unit_property(agent_id, 'health', 0)
            new_soldier_id = self.env.get_unit_property(agent_id, 'soldier_id', None)
            new_position_status = self.env.get_unit_property(agent_id, 'position_status', 'occupied')

            print(f"DEBUG[AgentManager]: Position {agent_id} after succession: health={new_health}, "
                  f"soldier_id={new_soldier_id}, position_status={new_position_status}")

            if new_health > 0 and new_position_status == 'occupied':
                print(f"DEBUG[AgentManager]: Position {agent_id} is now occupied by a new soldier")
            else:
                print(f"DEBUG[AgentManager]: WARNING: Position {agent_id} still appears to be vacant "
                      f"or unhealthy after succession")

            # Preserve the agent type
            if hasattr(self, 'agent_types') and agent_id in self.agent_types:
                # No need to change it, just make sure it's still there
                agent_type = self.agent_types[agent_id]
                print(f"DEBUG[AgentManager]: Preserving agent type {agent_type} for position {agent_id}")

            # Check if squad needs consolidation - only for squad leaders
            if parent_id and agent_type == 'SQUAD' and self.env.get_unit_property(parent_id,
                                                                                  'type') == UnitType.INFANTRY_SQUAD:
                # Check squad health after succession
                if hasattr(self, 'check_squad_needs_consolidation'):
                    needs_consolidation, casualty_pct, _ = self.check_squad_needs_consolidation(parent_id)

                    if needs_consolidation:
                        print(f"DEBUG[AgentManager]: Squad {parent_id} needs consolidation after succession")
                        print(f"DEBUG[AgentManager]: Casualty percentage: {casualty_pct:.1f}%")
                        if hasattr(self, '_consolidate_squad_to_single_team'):
                            self._consolidate_squad_to_single_team(parent_id, agent_id)
                        else:
                            print(f"DEBUG[AgentManager]: Missing _consolidate_squad_to_single_team method")
                else:
                    print(f"DEBUG[AgentManager]: Missing check_squad_needs_consolidation method")

                if self.env.debug_level > 0:
                    soldier_id = self.env.get_unit_property(agent_id, 'soldier_id')
                    print(
                        f"DEBUG[AgentManager]: Agent succession complete: Position {agent_id} now occupied by soldier {soldier_id}")

                # Return the same agent_id since the position hasn't changed
                print(f"DEBUG[AgentManager]: Returning same agent_id: {agent_id}")
                return agent_id

    def check_squad_needs_consolidation(self, squad_id):
        """
        Determines if a squad needs to be consolidated based on casualty levels.
        Enhanced to count both active casualties and vacant positions from succession.

        Args:
            squad_id: ID of squad to check

        Returns:
            Tuple of (needs_consolidation, casualty_percentage, squad_leader_alive)
        """
        # Ensure we're working with the actual squad, not a team
        unit_type = self.env.get_unit_property(squad_id, 'type')
        unit_type_str = str(unit_type)
        if 'INFANTRY_SQUAD' not in unit_type_str and 'SQD' not in unit_type_str:
            # If this is a team, get its parent squad
            parent_id = self.env.get_unit_property(squad_id, 'parent_id')
            if parent_id:
                parent_type = self.env.get_unit_property(parent_id, 'type')
                parent_type_str = str(parent_type)
                if 'INFANTRY_SQUAD' in parent_type_str or 'SQD' in parent_type_str:
                    print(
                        f"DEBUG[Consolidation]: Redirecting consolidation check from team {squad_id} to squad {parent_id}")
                    squad_id = parent_id

        print(f"\nDEBUG[Consolidation]: Checking if squad {squad_id} needs consolidation")

        # Get all positions in the squad (including team members)
        all_positions = []
        team_ids = []

        # First find squad leader position
        squad_leader_pos = None
        for child_id in self.env.get_unit_children(squad_id):
            if self.env.get_unit_property(child_id, 'is_leader', False):
                squad_leader_pos = child_id
                all_positions.append(child_id)

            # Check if child is a team
            unit_type = self.env.get_unit_property(child_id, 'type')
            unit_type_str = str(unit_type)
            string_id = self.env.get_unit_property(child_id, 'string_id', '')

            is_team = False
            # Check by unit type
            if hasattr(unit_type, 'value') and hasattr(UnitType.INFANTRY_TEAM, 'value'):
                if unit_type.value == UnitType.INFANTRY_TEAM.value:
                    is_team = True
            elif 'INFANTRY_TEAM' in unit_type_str:
                is_team = True
            # Check by string ID
            elif 'ATM' in string_id or 'BTM' in string_id or 'TEAM' in string_id:
                is_team = True

            if is_team:
                team_ids.append(child_id)

                # Add all team members
                for member_id in self.env.get_unit_children(child_id):
                    all_positions.append(member_id)

        # Count casualties - improved by checking health, position_status AND soldier_id
        total_positions = len(all_positions)
        casualties = 0
        squad_leader_alive = True

        for pos_id in all_positions:
            health = self.env.get_unit_property(pos_id, 'health', 0)
            position_status = self.env.get_unit_property(pos_id, 'position_status', 'occupied')
            soldier_id = self.env.get_unit_property(pos_id, 'soldier_id', None)

            # Count as casualty if ANY of these conditions are met:
            # 1. Health <= 0 OR status is 'casualty'
            # 2. OR position is 'vacant' (to include succession vacancies)
            # 3. OR position has no soldier (casualties where soldier was removed)
            is_casualty = (health <= 0 or position_status == 'casualty' or
                           position_status == 'vacant' or soldier_id is None)

            if is_casualty:
                casualties += 1
                print(
                    f"DEBUG[Consolidation]: Position {pos_id} counted as casualty: health={health}, status={position_status}, soldier_id={soldier_id}")

                # Check specifically if squad leader is a casualty
                if pos_id == squad_leader_pos:
                    squad_leader_alive = False

        # Calculate casualty percentage
        if total_positions == 0:
            return False, 0, True

        casualty_percentage = (casualties / total_positions) * 100

        # Print detailed analysis
        print(f"DEBUG[Consolidation]: Squad {squad_id} analysis:")
        print(f"DEBUG[Consolidation]: Total positions: {total_positions}")
        print(f"DEBUG[Consolidation]: Casualties: {casualties} ({casualty_percentage:.1f}%)")
        print(f"DEBUG[Consolidation]: Squad leader alive: {squad_leader_alive}")

        # Specific threshold: 5 or more casualties in the squad
        specific_threshold = 5

        # Squad needs consolidation if:
        # 1. Five or more casualties, OR
        # 2. More than 50% casualties, OR
        # 3. Squad leader is dead AND casualties are > 30%
        needs_consolidation = (casualties >= specific_threshold or
                               casualty_percentage > 50 or
                               (not squad_leader_alive and casualty_percentage > 30))

        print(f"DEBUG[Consolidation]: Squad {squad_id} needs consolidation: {needs_consolidation}")
        if needs_consolidation:
            if casualties >= specific_threshold:
                print(f"DEBUG[Consolidation]: Trigger: 5+ casualties ({casualties})")
            elif casualty_percentage > 50:
                print(f"DEBUG[Consolidation]: Trigger: >50% casualties ({casualty_percentage:.1f}%)")
            elif not squad_leader_alive and casualty_percentage > 30:
                print(f"DEBUG[Consolidation]: Trigger: Squad leader KIA and >30% casualties")

        return needs_consolidation, casualty_percentage, squad_leader_alive

    def _consolidate_squad_to_single_team(self, squad_id, squad_leader_pos_id):
        """
        Enhanced squad consolidation that properly implements the outlined process:
        1. Identify which team has the most living members
        2. Move living soldiers (including leaders) to this team
        3. Update agent system appropriately maintaining consistent agent IDs
        4. Ensure all positions are correctly marked

        Args:
            squad_id: ID of the squad to consolidate
            squad_leader_pos_id: Position ID of the current squad leader
        """
        print(f"\nDEBUG[Consolidate]: Starting squad consolidation for squad {squad_id}")
        print(f"DEBUG[Consolidate]: Current squad leader position ID: {squad_leader_pos_id}")

        squad_string = self.env.get_unit_property(squad_id, 'string_id', str(squad_id))

        # Find the agent ID currently mapped to the squad leader position
        squad_leader_agent_id = None
        if hasattr(self, 'unit_id_to_agent_id'):
            squad_leader_agent_id = self.unit_id_to_agent_id.get(squad_leader_pos_id)
            if squad_leader_agent_id:
                print(f"DEBUG[Consolidate]: Found squad leader agent ID: {squad_leader_agent_id}")

        # Track positions we've already processed to avoid double-moves
        already_processed_positions = set()

        # Step 1: Identify all teams in the squad and count living members
        teams = []
        team_living_counts = {}

        for unit_id in self.env.get_unit_children(squad_id):
            # Try multiple ways to identify teams
            unit_type = self.env.get_unit_property(unit_id, 'type')
            string_id = self.env.get_unit_property(unit_id, 'string_id', '')

            # Check if it's a team
            is_team = False
            if hasattr(unit_type, 'value') and hasattr(UnitType.INFANTRY_TEAM, 'value'):
                if unit_type.value == UnitType.INFANTRY_TEAM.value:
                    is_team = True
            elif str(unit_type) == str(UnitType.INFANTRY_TEAM):
                is_team = True
            elif 'ATM' in string_id or 'BTM' in string_id or 'TEAM' in string_id:
                is_team = True

            if is_team:
                # Count living members in this team
                living_count = 0
                members = self.env.get_unit_children(unit_id)

                for member_id in members:
                    health = self.env.get_unit_property(member_id, 'health', 0)
                    position_status = self.env.get_unit_property(member_id, 'position_status', 'occupied')

                    if health > 0 and position_status == 'occupied':
                        living_count += 1

                teams.append(unit_id)
                team_living_counts[unit_id] = living_count
                print(f"DEBUG[Consolidate]: Team {unit_id} ({string_id}) has {living_count} living members")

        if not teams:
            print(f"DEBUG[Consolidate]: No viable teams found for consolidation")
            return

        # Step 2: Identify the best team to consolidate around
        # Prioritize Alpha team if it has any living members
        best_team_id = None
        highest_count = -1

        # First try to find Alpha team with living members
        for team_id in teams:
            team_string = self.env.get_unit_property(team_id, 'string_id', '')
            count = team_living_counts[team_id]

            if 'ATM' in team_string and count > 0:
                best_team_id = team_id
                print(f"DEBUG[Consolidate]: Selected Alpha team {team_id} ({team_string}) with {count} living members")
                break

        # If no Alpha team with living members, pick team with most survivors
        if best_team_id is None:
            for team_id, count in team_living_counts.items():
                if count > highest_count:
                    highest_count = count
                    best_team_id = team_id

                if best_team_id:
                    team_string = self.env.get_unit_property(best_team_id, 'string_id', '')
                    print(
                        f"DEBUG[Consolidate]: Selected team {best_team_id} ({team_string}) with most survivors ({highest_count})")

        if not best_team_id:
            print(f"DEBUG[Consolidate]: No viable team found with living members")
            return

        # Step 3: Find team leader position in target team
        team_leader_pos_id = None
        for pos_id in self.env.get_unit_children(best_team_id):
            is_leader = self.env.get_unit_property(pos_id, 'is_leader', False)
            if is_leader:
                team_leader_pos_id = pos_id
                team_leader_health = self.env.get_unit_property(pos_id, 'health', 0)
                team_leader_status = self.env.get_unit_property(pos_id, 'position_status', 'occupied')
                print(
                    f"DEBUG[Consolidate]: Found team leader position {pos_id}, health={team_leader_health}, status={team_leader_status}")
                break

        if not team_leader_pos_id:
            print(f"DEBUG[Consolidate]: No team leader position found for team {best_team_id}")
            # Try to find any position to use
            for pos_id in self.env.get_unit_children(best_team_id):
                team_leader_pos_id = pos_id
                print(f"DEBUG[Consolidate]: Using position {pos_id} as substitute team leader position")
                break

        if not team_leader_pos_id:
            print(f"DEBUG[Consolidate]: No viable positions found in team {best_team_id}")
            return

        # Step 4: Find all living soldiers across the squad to consolidate
        living_soldiers = []

        # First, check if squad leader is alive
        squad_leader_health = self.env.get_unit_property(squad_leader_pos_id, 'health', 0)
        squad_leader_status = self.env.get_unit_property(squad_leader_pos_id, 'position_status', 'occupied')
        squad_leader_soldier = self.env.get_unit_property(squad_leader_pos_id, 'soldier_id')

        if squad_leader_health > 0 and squad_leader_status == 'occupied' and squad_leader_soldier:
            living_soldiers.append({
                'position_id': squad_leader_pos_id,
                'soldier_id': squad_leader_soldier,
                'health': squad_leader_health,
                'is_leader': True,
                'is_agent': True,
                'primary_weapon': self.env.get_unit_property(squad_leader_pos_id, 'primary_weapon'),
                'secondary_weapon': self.env.get_unit_property(squad_leader_pos_id, 'secondary_weapon'),
                'type': 'SQUAD_LEADER'
            })
            print(
                f"DEBUG[Consolidate]: Squad leader position {squad_leader_pos_id} with soldier {squad_leader_soldier} is alive")

        # Then find all other living soldiers in all teams
        for team_id in teams:
            for pos_id in self.env.get_unit_children(team_id):
                # Skip if already processed
                if pos_id in already_processed_positions:
                    continue

                health = self.env.get_unit_property(pos_id, 'health', 0)
                status = self.env.get_unit_property(pos_id, 'position_status', 'occupied')
                soldier_id = self.env.get_unit_property(pos_id, 'soldier_id')
                is_leader = self.env.get_unit_property(pos_id, 'is_leader', False)
                is_agent = self.env.get_unit_property(pos_id, 'is_agent', False)

                if health > 0 and status == 'occupied' and soldier_id:
                    role_value = self.env.get_unit_property(pos_id, 'role')
                    role_name = str(role_value)
                    if hasattr(role_value, 'name'):
                        role_name = role_value.name

                    living_soldiers.append({
                        'position_id': pos_id,
                        'soldier_id': soldier_id,
                        'health': health,
                        'is_leader': is_leader,
                        'is_agent': is_agent,
                        'primary_weapon': self.env.get_unit_property(pos_id, 'primary_weapon'),
                        'secondary_weapon': self.env.get_unit_property(pos_id, 'secondary_weapon'),
                        'type': role_name
                    })
                    print(f"DEBUG[Consolidate]: Found living position {pos_id} with soldier {soldier_id} ({role_name})")

        print(f"DEBUG[Consolidate]: Found {len(living_soldiers)} living soldiers to consolidate")

        # Step 5: Get all available positions in target team
        all_team_positions = []
        available_positions = []
        for pos_id in self.env.get_unit_children(best_team_id):
            all_team_positions.append(pos_id)

            # Skip team leader position if it has a living soldier
            if pos_id == team_leader_pos_id and self.env.get_unit_property(pos_id, 'health', 0) > 0:
                continue

            # Check if position is usable (vacant or casualty)
            health = self.env.get_unit_property(pos_id, 'health', 0)
            status = self.env.get_unit_property(pos_id, 'position_status', 'occupied')

            # Either vacant or casualty position can be used
            if health <= 0 or status != 'occupied':
                available_positions.append(pos_id)
                print(f"DEBUG[Consolidate]: Position {pos_id} is available for consolidation")

        print(f"DEBUG[Consolidate]: Found {len(available_positions)} positions to fill in team {best_team_id}")

        # Step 6: Perform the consolidation
        # First, prioritize the squad leader (if alive) or a team leader for the team leader position
        team_leader_filled = False

        # Move the squad leader soldier to the team leader position first if it exists
        for i, soldier in enumerate(living_soldiers):
            if soldier['is_leader'] and soldier['position_id'] == squad_leader_pos_id:
                # Found squad leader - move to team leader position
                source_pos_id = soldier['position_id']
                target_pos_id = team_leader_pos_id

                print(
                    f"DEBUG[Consolidate]: Moving soldier {soldier['soldier_id']} from position {source_pos_id} to {target_pos_id}")

                # Update target position
                self.env.update_unit_property(target_pos_id, 'soldier_id', soldier['soldier_id'])
                self.env.update_unit_property(target_pos_id, 'health', soldier['health'])
                self.env.update_unit_property(target_pos_id, 'position_status', 'occupied')
                self.env.update_unit_property(target_pos_id, 'primary_weapon', soldier['primary_weapon'])
                self.env.update_unit_property(target_pos_id, 'secondary_weapon', soldier['secondary_weapon'])

                # This is critical: set the is_agent flag to True for the team leader position
                self.env.update_unit_property(target_pos_id, 'is_agent', True)
                self.env.update_unit_property(target_pos_id, 'is_leader', True)

                # Handle agent ID mapping
                if squad_leader_agent_id is not None:
                    print(
                        f"DEBUG[Consolidate]: Updated agent mapping: {squad_leader_agent_id} now maps to {target_pos_id}")

                    if hasattr(self, 'unit_id_to_agent_id'):
                        # Remove old mapping
                        if source_pos_id in self.unit_id_to_agent_id:
                            del self.unit_id_to_agent_id[source_pos_id]

                        # Add new mapping
                        self.unit_id_to_agent_id[target_pos_id] = squad_leader_agent_id

                # Update ammunition
                if hasattr(self.env, 'combat_manager'):
                    # Get ammunition from source position
                    primary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'primary')
                    secondary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'secondary')

                    # Set ammunition in target position
                    self.env.combat_manager.ammo_tracking[target_pos_id] = {
                        'primary': primary_ammo,
                        'secondary': secondary_ammo
                    }

                # Clear vacancy flags on target
                self.env.update_unit_property(target_pos_id, 'vacancy_handled', False)
                self.env.update_unit_property(target_pos_id, 'casualty_handled', False)

                # Mark source position as vacant
                self.env.update_unit_property(source_pos_id, 'position_status', 'vacant')
                self.env.update_unit_property(source_pos_id, 'soldier_id', None)
                self.env.update_unit_property(source_pos_id, 'health', 0)

                # Handle vacancy for source position
                self.env._handle_vacancy(source_pos_id)

                # Add to processed list
                already_processed_positions.add(source_pos_id)

                # Remove this soldier from the list
                living_soldiers.pop(i)
                team_leader_filled = True
                break

        # If no squad leader, find a team leader to fill the team leader position
        if not team_leader_filled:
            for i, soldier in enumerate(living_soldiers):
                if soldier['is_leader']:
                    # Found a team leader - move to team leader position
                    source_pos_id = soldier['position_id']
                    target_pos_id = team_leader_pos_id

                    print(
                        f"DEBUG[Consolidate]: Moving team leader {soldier['soldier_id']} from position {source_pos_id} to {target_pos_id}")

                    # Update target position
                    self.env.update_unit_property(target_pos_id, 'soldier_id', soldier['soldier_id'])
                    self.env.update_unit_property(target_pos_id, 'health', soldier['health'])
                    self.env.update_unit_property(target_pos_id, 'position_status', 'occupied')
                    self.env.update_unit_property(target_pos_id, 'primary_weapon', soldier['primary_weapon'])
                    self.env.update_unit_property(target_pos_id, 'secondary_weapon', soldier['secondary_weapon'])

                    # Since this was a team leader, check if it should be an agent
                    if squad_leader_agent_id is not None:
                        self.env.update_unit_property(target_pos_id, 'is_agent', True)
                        print(f"DEBUG[Consolidate]: Set team leader position {target_pos_id} as agent")

                    self.env.update_unit_property(target_pos_id, 'is_leader', True)

                    # Handle agent ID mapping if applicable
                    if squad_leader_agent_id is not None:
                        print(
                            f"DEBUG[Consolidate]: Updated agent mapping: {squad_leader_agent_id} now maps to {target_pos_id}")

                        if hasattr(self, 'unit_id_to_agent_id'):
                            # Remove old mapping
                            if source_pos_id in self.unit_id_to_agent_id:
                                del self.unit_id_to_agent_id[source_pos_id]

                            # Add new mapping
                            self.unit_id_to_agent_id[target_pos_id] = squad_leader_agent_id

                    # Update ammunition
                    if hasattr(self.env, 'combat_manager'):
                        # Get ammunition from source position
                        primary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'primary')
                        secondary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'secondary')

                        # Set ammunition in target position
                        self.env.combat_manager.ammo_tracking[target_pos_id] = {
                            'primary': primary_ammo,
                            'secondary': secondary_ammo
                        }

                    # Clear vacancy flags on target
                    self.env.update_unit_property(target_pos_id, 'vacancy_handled', False)
                    self.env.update_unit_property(target_pos_id, 'casualty_handled', False)

                    # Mark source position as vacant
                    self.env.update_unit_property(source_pos_id, 'position_status', 'vacant')
                    self.env.update_unit_property(source_pos_id, 'soldier_id', None)
                    self.env.update_unit_property(source_pos_id, 'health', 0)

                    # Handle vacancy for source position
                    self.env._handle_vacancy(source_pos_id)

                    # Add to processed list
                    already_processed_positions.add(source_pos_id)

                    # Remove this soldier from the list
                    living_soldiers.pop(i)
                    team_leader_filled = True
                    break

        # If still no team leader, use any available soldier
        if not team_leader_filled and living_soldiers:
            # Pick first available soldier
            soldier = living_soldiers.pop(0)
            source_pos_id = soldier['position_id']
            target_pos_id = team_leader_pos_id

            print(
                f"DEBUG[Consolidate]: Using soldier {soldier['soldier_id']} from position {source_pos_id} as team leader")

            # Update target position
            self.env.update_unit_property(target_pos_id, 'soldier_id', soldier['soldier_id'])
            self.env.update_unit_property(target_pos_id, 'health', soldier['health'])
            self.env.update_unit_property(target_pos_id, 'position_status', 'occupied')
            self.env.update_unit_property(target_pos_id, 'primary_weapon', soldier['primary_weapon'])
            self.env.update_unit_property(target_pos_id, 'secondary_weapon', soldier['secondary_weapon'])

            # Since this will become an agent, set the flag
            if squad_leader_agent_id is not None:
                self.env.update_unit_property(target_pos_id, 'is_agent', True)
                print(f"DEBUG[Consolidate]: Set team leader position {target_pos_id} as agent")

            # Set as leader
            self.env.update_unit_property(target_pos_id, 'is_leader', True)

            # Handle agent ID mapping if applicable
            if squad_leader_agent_id is not None:
                print(f"DEBUG[Consolidate]: Updated agent mapping: {squad_leader_agent_id} now maps to {target_pos_id}")

                if hasattr(self, 'unit_id_to_agent_id'):
                    # Remove old mapping
                    if source_pos_id in self.unit_id_to_agent_id:
                        del self.unit_id_to_agent_id[source_pos_id]

                    # Add new mapping
                    self.unit_id_to_agent_id[target_pos_id] = squad_leader_agent_id

            # Update ammunition
            if hasattr(self.env, 'combat_manager'):
                # Get ammunition from source position
                primary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'primary')
                secondary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'secondary')

                # Set ammunition in target position
                self.env.combat_manager.ammo_tracking[target_pos_id] = {
                    'primary': primary_ammo,
                    'secondary': secondary_ammo
                }

            # Clear vacancy flags on target
            self.env.update_unit_property(target_pos_id, 'vacancy_handled', False)
            self.env.update_unit_property(target_pos_id, 'casualty_handled', False)

            # Mark source position as vacant
            self.env.update_unit_property(source_pos_id, 'position_status', 'vacant')
            self.env.update_unit_property(source_pos_id, 'soldier_id', None)
            self.env.update_unit_property(source_pos_id, 'health', 0)

            # Handle vacancy for source position
            self.env._handle_vacancy(source_pos_id)

            # Add to processed list
            already_processed_positions.add(source_pos_id)
            team_leader_filled = True

        # Now move remaining soldiers to other available positions in team
        # Get updated list of available positions (skipping team leader if filled)
        available_positions = []
        for pos_id in all_team_positions:
            if pos_id == team_leader_pos_id and team_leader_filled:
                continue

            # Check if position is usable (vacant or casualty)
            health = self.env.get_unit_property(pos_id, 'health', 0)
            status = self.env.get_unit_property(pos_id, 'position_status', 'occupied')

            # Either vacant or casualty position can be used
            if health <= 0 or status != 'occupied':
                available_positions.append(pos_id)

        # Move as many remaining soldiers as possible to available positions
        for i, soldier in enumerate(living_soldiers):
            if i >= len(available_positions):
                print(
                    f"DEBUG[Consolidate]: Not enough positions for all soldiers - could only transfer {i} of {len(living_soldiers)}")
                break

            source_pos_id = soldier['position_id']
            target_pos_id = available_positions[i]

            # Skip if it's the same position
            if source_pos_id == target_pos_id:
                continue

            print(
                f"DEBUG[Consolidate]: Moving soldier {soldier['soldier_id']} from position {source_pos_id} to {target_pos_id}")

            # Update target position
            self.env.update_unit_property(target_pos_id, 'soldier_id', soldier['soldier_id'])
            self.env.update_unit_property(target_pos_id, 'health', soldier['health'])
            self.env.update_unit_property(target_pos_id, 'position_status', 'occupied')
            self.env.update_unit_property(target_pos_id, 'primary_weapon', soldier['primary_weapon'])
            self.env.update_unit_property(target_pos_id, 'secondary_weapon', soldier['secondary_weapon'])

            # Copy is_agent flag
            self.env.update_unit_property(target_pos_id, 'is_agent', soldier['is_agent'])

            # Not a leader unless it's the team leader position
            self.env.update_unit_property(target_pos_id, 'is_leader', target_pos_id == team_leader_pos_id)

            # Update ammunition
            if hasattr(self.env, 'combat_manager'):
                # Get ammunition from source position
                primary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'primary')
                secondary_ammo = self.env.combat_manager._get_unit_ammo(source_pos_id, 'secondary')

                # Set ammunition in target position
                self.env.combat_manager.ammo_tracking[target_pos_id] = {
                    'primary': primary_ammo,
                    'secondary': secondary_ammo
                }

            # Clear vacancy flags on target
            self.env.update_unit_property(target_pos_id, 'vacancy_handled', False)
            self.env.update_unit_property(target_pos_id, 'casualty_handled', False)

            # Mark source position as vacant
            self.env.update_unit_property(source_pos_id, 'position_status', 'vacant')
            self.env.update_unit_property(source_pos_id, 'soldier_id', None)
            self.env.update_unit_property(source_pos_id, 'health', 0)

            # Handle vacancy for source position
            self.env._handle_vacancy(source_pos_id)

            # Add to processed list
            already_processed_positions.add(source_pos_id)

        # Step 7: Update agent tracking
        if self.debug_level > 0:
            print(f"DEBUG[Consolidate]: Updating agent tracking")

        # Remove squad leader from agent_ids
        if squad_leader_pos_id in self.agent_ids:
            self.agent_ids.remove(squad_leader_pos_id)
            if self.debug_level > 0:
                print(f"DEBUG[Consolidate]: Removed squad leader position {squad_leader_pos_id} from agent_ids")

        # Add team leader to agent_ids
        if team_leader_pos_id not in self.agent_ids:
            self.agent_ids.append(team_leader_pos_id)
            if self.debug_level > 0:
                print(f"DEBUG[Consolidate]: Added team leader position {team_leader_pos_id} to agent_ids")

        # Update agent types
        if hasattr(self, 'agent_types'):
            # Remove squad leader from agent_types if present
            if squad_leader_pos_id in self.agent_types:
                del self.agent_types[squad_leader_pos_id]

            # Update type for team leader position - THIS IS THE CRITICAL CHANGE
            self.agent_types[team_leader_pos_id] = 'TEAM'
            if self.debug_level > 0:
                print(f"DEBUG[Consolidate]: Updated agent type for position {team_leader_pos_id} to TEAM")

            # Update by agent ID if using consistent agent IDs
            if squad_leader_agent_id is not None:
                self.agent_types[squad_leader_agent_id] = 'TEAM'
                print(f"DEBUG[Consolidate]: Updated agent type for agent ID {squad_leader_agent_id} to TEAM")

        # Update env.agent_ids if it exists
        if hasattr(self.env, 'agent_ids'):
            if squad_leader_pos_id in self.env.agent_ids:
                self.env.agent_ids.remove(squad_leader_pos_id)
            if team_leader_pos_id not in self.env.agent_ids:
                self.env.agent_ids.append(team_leader_pos_id)
            if self.debug_level > 0:
                print(f"DEBUG[Consolidate]: Updated environment agent_ids list")

        # Update the unit_id_to_agent_id mapping if we have it
        if hasattr(self, 'unit_id_to_agent_id') and squad_leader_agent_id is not None:
            # Remove the old mapping
            if squad_leader_pos_id in self.unit_id_to_agent_id:
                del self.unit_id_to_agent_id[squad_leader_pos_id]

            # Add the new mapping
            self.unit_id_to_agent_id[team_leader_pos_id] = squad_leader_agent_id
            print(
                f"DEBUG[Consolidate]: Updated unit-to-agent mapping: agent ID {squad_leader_agent_id} now maps to team leader position {team_leader_pos_id}")

        # Step 8: Apply team formation to consolidated team
        print(f"DEBUG[Consolidate]: Applying team formation to consolidated team")

        # Apply wedge formation to consolidated team
        try:
            from US_Army_PLT_Composition_vTest import US_IN_apply_formation
            US_IN_apply_formation(self.env, best_team_id, "team_wedge_left")
            print(f"DEBUG[Consolidate]: Applied team_wedge_left formation to team {best_team_id}")
        except Exception as e:
            print(f"DEBUG[Consolidate]: Error applying formation: {e}")

        print(f"DEBUG[Consolidate]: Squad {squad_string} (ID: {squad_id}) consolidated to team {best_team_id}")
        print(f"DEBUG[Consolidate]: Consolidation complete")

    def _verify_positions_after_consolidation(self, squad_id):
        """
        Verify that all positions in the squad have consistent states after consolidation.
        Enhanced to also check agent mapping consistency.

        Args:
            squad_id: ID of the consolidated squad
        """
        print(f"DEBUG[Consolidate]: Verifying position states after consolidation")

        # Helper to verify a single position
        def verify_single_position(pos_id):
            health = self.env.get_unit_property(pos_id, 'health', 0)
            position_status = self.env.get_unit_property(pos_id, 'position_status', 'unknown')
            soldier_id = self.env.get_unit_property(pos_id, 'soldier_id', None)
            is_agent = self.env.get_unit_property(pos_id, 'is_agent', False)

            # Check for inconsistencies
            if health <= 0 and position_status == 'occupied':
                print(f"ERROR[Verify]: Position {pos_id} has health {health} but status '{position_status}'")
                self.env.update_unit_property(pos_id, 'position_status', 'vacant')

            if health > 0 and position_status != 'occupied':
                print(f"ERROR[Verify]: Position {pos_id} has health {health} but status '{position_status}'")
                self.env.update_unit_property(pos_id, 'position_status', 'occupied')

            if position_status == 'vacant' and soldier_id is not None:
                print(f"ERROR[Verify]: Position {pos_id} is vacant but has soldier_id {soldier_id}")
                self.env.update_unit_property(pos_id, 'soldier_id', None)

            if position_status == 'occupied' and soldier_id is None:
                print(f"ERROR[Verify]: Position {pos_id} is occupied but has no soldier_id")

            # Verify agent status consistency
            if is_agent and health <= 0:
                print(f"ERROR[Verify]: Position {pos_id} is marked as agent but has no health")
                self.env.update_unit_property(pos_id, 'is_agent', False)

            # Verify ammunition consistency
            if hasattr(self.env, 'combat_manager'):
                if pos_id in self.env.combat_manager.ammo_tracking:
                    ammo = self.env.combat_manager.ammo_tracking[pos_id]
                    if health <= 0 and (ammo.get('primary', 0) > 0 or ammo.get('secondary', 0) > 0):
                        print(f"ERROR[Verify]: Position {pos_id} has health {health} but has ammunition")
                        self.env.combat_manager.ammo_tracking[pos_id] = {
                            'primary': 0,
                            'secondary': 0
                        }

        # Verify squad leader position
        for pos_id in self.env.get_unit_children(squad_id):
            if self.env.get_unit_property(pos_id, 'is_leader', False):
                print(f"DEBUG[Verify]: Checking squad leader position {pos_id}")
                verify_single_position(pos_id)

        # Verify all team positions
        for team_id in self.env.get_unit_children(squad_id):
            if self.env.get_unit_property(team_id, 'type') == UnitType.INFANTRY_TEAM:
                print(f"DEBUG[Verify]: Checking team {team_id}")

                # Check team leader
                for pos_id in self.env.get_unit_children(team_id):
                    if self.env.get_unit_property(pos_id, 'is_leader', False):
                        print(f"DEBUG[Verify]: Checking team leader position {pos_id}")
                        verify_single_position(pos_id)

                # Check team members
                for pos_id in self.env.get_unit_children(team_id):
                    if not self.env.get_unit_property(pos_id, 'is_leader', False):
                        print(f"DEBUG[Verify]: Checking team member position {pos_id}")
                        verify_single_position(pos_id)

        # Verify agent mapping consistency
        if hasattr(self, 'unit_id_to_agent_id') and self.unit_id_to_agent_id:
            print(f"DEBUG[Verify]: Checking agent mapping consistency")

            for unit_id, agent_id in list(self.unit_id_to_agent_id.items()):
                # Check if unit still exists and is valid
                if unit_id not in self.env.state_manager.active_units:
                    print(f"ERROR[Verify]: Unit {unit_id} in mapping doesn't exist in active units")
                    continue

                # Check health and position status
                health = self.env.get_unit_property(unit_id, 'health', 0)
                position_status = self.env.get_unit_property(unit_id, 'position_status', 'unknown')

                if health <= 0 or position_status != 'occupied':
                    print(
                        f"ERROR[Verify]: Agent {agent_id} is mapped to unit {unit_id} which has health={health}, status={position_status}")

                    # Check if there's a valid unit that should have this agent ID
                    found_replacement = False

                    # Look for replacement in the same squad/team structure
                    parent_id = self.env.get_unit_property(unit_id, 'parent_id')
                    if parent_id:
                        # Get the parent's parent (squad level for team members)
                        squad_id = parent_id
                        squad_type = self.env.get_unit_property(squad_id, 'type')
                        if squad_type != UnitType.INFANTRY_SQUAD:
                            squad_id = self.env.get_unit_property(parent_id, 'parent_id')

                        if squad_id:
                            # Look through the squad structure for a valid leader
                            for team_id in self.env.get_unit_children(squad_id):
                                if self.env.get_unit_property(team_id, 'type') == UnitType.INFANTRY_TEAM:
                                    for pos_id in self.env.get_unit_children(team_id):
                                        if self.env.get_unit_property(pos_id, 'is_leader', False):
                                            # Check if this is a valid leader position
                                            leader_health = self.env.get_unit_property(pos_id, 'health', 0)
                                            leader_status = self.env.get_unit_property(pos_id, 'position_status',
                                                                                       'unknown')
                                            if leader_health > 0 and leader_status == 'occupied' and pos_id != unit_id:
                                                # Found a potential replacement
                                                print(
                                                    f"DEBUG[Verify]: Found replacement leader position {pos_id} for agent {agent_id}")

                                                # Update the mapping
                                                del self.unit_id_to_agent_id[unit_id]
                                                self.unit_id_to_agent_id[pos_id] = agent_id
                                                print(
                                                    f"DEBUG[Verify]: Updated agent mapping: {agent_id} now maps to {pos_id}")

                                                found_replacement = True
                                                break

                                if found_replacement:
                                    break

            # Check agent_ids list for consistency with mapping
            if hasattr(self, 'agent_ids'):
                unit_list = list(self.unit_id_to_agent_id.keys())
                for unit_id in unit_list:
                    # Ensure all mapped units are in agent_ids
                    if unit_id not in self.agent_ids:
                        health = self.env.get_unit_property(unit_id, 'health', 0)
                        status = self.env.get_unit_property(unit_id, 'position_status', 'unknown')

                        if health > 0 and status == 'occupied':
                            print(f"ERROR[Verify]: Unit {unit_id} is in mapping but not in agent_ids list")
                            # Add to agent_ids
                            self.agent_ids.append(unit_id)
                            print(f"DEBUG[Verify]: Added unit {unit_id} to agent_ids list")

        print(f"DEBUG[Verify]: Position verification complete")

    def update_after_casualties(self, affected_unit_ids=None):
        """
        Updates agent tracking after casualties, performing consolidation if needed.
        Enhanced to properly detect and handle consolidation requirements.
        Should be called after casualties are processed.

        Args:
            affected_unit_ids: List of unit IDs that had casualties (optional)
                              If None, all squads will be checked
        """
        print(f"\nDEBUG[AgentMgr]: Starting update after casualties")

        # If no specific units provided, check all squads
        if affected_unit_ids is None:
            # Get all squad IDs
            squad_ids = []
            for unit_id in self.env.state_manager.active_units:
                unit_type = self.env.get_unit_property(unit_id, 'type')
                unit_type_str = str(unit_type)
                if 'INFANTRY_SQUAD' in unit_type_str or 'SQD' in unit_type_str:
                    squad_ids.append(unit_id)
                    print(f"DEBUG[AgentMgr]: Found squad {unit_id}")
        else:
            # Filter for squad IDs (ensuring we have the squad, not team IDs)
            squad_ids = []
            for unit_id in affected_unit_ids:
                unit_type = self.env.get_unit_property(unit_id, 'type')
                unit_type_str = str(unit_type)
                if 'INFANTRY_SQUAD' in unit_type_str or 'SQD' in unit_type_str:
                    squad_ids.append(unit_id)
                    print(f"DEBUG[AgentMgr]: Processing affected squad {unit_id}")
                else:
                    # Check parent if this is a member of a squad or team
                    parent_id = self.env.get_unit_property(unit_id, 'parent_id')
                    if parent_id:
                        parent_type = self.env.get_unit_property(parent_id, 'type')
                        parent_type_str = str(parent_type)

                        # If parent is squad, add it
                        if (
                                'INFANTRY_SQUAD' in parent_type_str or 'SQD' in parent_type_str) and parent_id not in squad_ids:
                            squad_ids.append(parent_id)
                            print(f"DEBUG[AgentMgr]: Processing parent squad {parent_id} of affected unit {unit_id}")
                        # If parent is team, check its parent
                        elif 'INFANTRY_TEAM' in parent_type_str or 'TEAM' in parent_type_str or 'TM' in parent_type_str:
                            squad_parent = self.env.get_unit_property(parent_id, 'parent_id')
                            if squad_parent:
                                squad_type = self.env.get_unit_property(squad_parent, 'type')
                                squad_type_str = str(squad_type)
                                if (
                                        'INFANTRY_SQUAD' in squad_type_str or 'SQD' in squad_type_str) and squad_parent not in squad_ids:
                                    squad_ids.append(squad_parent)
                                    print(f"DEBUG[AgentMgr]: Processing squad {squad_parent} from team {parent_id}")

        # Check each squad for consolidation needs
        for squad_id in squad_ids:
            # Skip squads that have already been consolidated
            if not squad_id in self.env.state_manager.active_units:
                continue

            # Check if squad needs consolidation using enhanced method
            needs_consolidation, casualty_pct, sl_alive = self.check_squad_needs_consolidation(squad_id)

            if needs_consolidation:
                print(f"\nDEBUG[AgentMgr]: Squad {squad_id} needs consolidation!")
                print(f"DEBUG[AgentMgr]: Casualty percentage: {casualty_pct:.1f}%")
                print(f"DEBUG[AgentMgr]: Squad leader alive: {sl_alive}")

                # Find squad leader position
                squad_leader_pos = None
                for child_id in self.env.get_unit_children(squad_id):
                    if self.env.get_unit_property(child_id, 'is_leader', False):
                        squad_leader_pos = child_id
                        break

                if squad_leader_pos:
                    # Perform consolidation
                    print(
                        f"DEBUG[AgentMgr]: Calling consolidation for squad {squad_id} with leader position {squad_leader_pos}")
                    self._consolidate_squad_to_single_team(squad_id, squad_leader_pos)
                else:
                    print(f"DEBUG[AgentMgr]: Cannot consolidate squad {squad_id} - squad leader position not found")
            else:
                print(f"DEBUG[AgentMgr]: Squad {squad_id} does not need consolidation")

        # Update unit_id_to_agent_id mapping to ensure it's current
        if hasattr(self, 'unit_id_to_agent_id') and self.initialized_role_mapping:
            updated_mapping = {}

            # Check all current mappings to see if they're still valid
            for unit_id, agent_id in self.unit_id_to_agent_id.items():
                if unit_id in self.env.state_manager.active_units:
                    # Check if this unit is still alive and valid
                    health = self.env.get_unit_property(unit_id, 'health', 0)
                    position_status = self.env.get_unit_property(unit_id, 'position_status', 'occupied')

                    if health > 0 and position_status == 'occupied':
                        # Still a valid unit, keep the mapping
                        updated_mapping[unit_id] = agent_id
                    else:
                        # Look for a replacement - check if there's a new leader in this position's parent
                        parent_id = self.env.get_unit_property(unit_id, 'parent_id')
                        if parent_id:
                            for child_id in self.env.get_unit_children(parent_id):
                                if self.env.get_unit_property(child_id, 'is_leader', False):
                                    # Found a new leader
                                    child_health = self.env.get_unit_property(child_id, 'health', 0)
                                    child_status = self.env.get_unit_property(child_id, 'position_status', 'occupied')

                                    if child_health > 0 and child_status == 'occupied' and child_id != unit_id:
                                        # This is a valid replacement - update the mapping
                                        updated_mapping[child_id] = agent_id
                                        print(
                                            f"Updating agent mapping: unit {unit_id} -> {child_id} for agent {agent_id}")
                                        break

            # Update the mapping
            self.unit_id_to_agent_id = updated_mapping

        print(f"DEBUG[AgentMgr]: Update after casualties complete")

    def debug_unit_structure(self, env, unit_id, indent=0):
        """
        Prints detailed hierarchical structure of a unit with position-based information.

        Args:
            env: Reference to environment
            unit_id: ID of unit to debug
            indent: Current indentation level for formatting
        """
        # Get basic unit properties
        unit_type = env.get_unit_property(unit_id, 'type')
        string_id = env.get_unit_property(unit_id, 'string_id', str(unit_id))
        position = env.get_unit_position(unit_id)
        health = env.get_unit_property(unit_id, 'health', 0)
        position_status = env.get_unit_property(unit_id, 'position_status', 'occupied')
        soldier_id = env.get_unit_property(unit_id, 'soldier_id', None)
        is_leader = env.get_unit_property(unit_id, 'is_leader', False)
        is_agent = env.get_unit_property(unit_id, 'is_agent', False)

        # Format type information
        if isinstance(unit_type, Enum):
            type_str = unit_type.name
        else:
            type_str = str(unit_type)

        # Create indentation
        indent_str = "  " * indent

        # Print unit information
        print(f"{indent_str}Unit: {string_id} (ID: {unit_id})")
        print(f"{indent_str}Type: {type_str}")
        print(f"{indent_str}Position: {position}")
        print(f"{indent_str}Soldier ID: {soldier_id}")
        print(f"{indent_str}Health: {health}")
        print(f"{indent_str}Status: {position_status}")
        print(f"{indent_str}Is Leader: {is_leader}")
        print(f"{indent_str}Is Agent: {is_agent}")

        # Get unit children
        children = env.get_unit_children(unit_id)
        if children:
            print(f"{indent_str}Children ({len(children)}):")
            for child_id in children:
                print(f"{indent_str}  Child {child_id}:")
                self.debug_unit_structure(env, child_id, indent + 2)
        else:
            print(f"{indent_str}No children")

        # Print a blank line after top-level units
        if indent == 0:
            print("")

    def debug_platoon_structure(self, platoon_id=None):
        """
        Print the full structure of a platoon to diagnose mapping issues.
        """
        # Find the platoon if not provided
        if platoon_id is None:
            for unit_id in self.env.state_manager.active_units:
                unit_type = self.env.get_unit_property(unit_id, 'type')
                unit_type_str = str(unit_type)
                if 'INFANTRY_PLATOON' in unit_type_str:
                    platoon_id = unit_id
                    break

        if not platoon_id:
            print("No platoon found")
            return

        print(f"\n=== PLATOON STRUCTURE (ID: {platoon_id}) ===")
        platoon_string = self.env.get_unit_property(platoon_id, 'string_id', str(platoon_id))
        print(f"Platoon: {platoon_string}\n")

        # Find all direct children (squads and teams)
        for unit_id in self.env.get_unit_children(platoon_id):
            unit_string = self.env.get_unit_property(unit_id, 'string_id', str(unit_id))
            unit_type = self.env.get_unit_property(unit_id, 'type')
            unit_type_str = str(unit_type)

            print(f"Unit: {unit_string} (ID: {unit_id}, Type: {unit_type_str})")

            # Print all team members
            for member_id in self.env.get_unit_children(unit_id):
                member_string = self.env.get_unit_property(member_id, 'string_id', str(member_id))
                role_value = self.env.get_unit_property(member_id, 'role')
                role_name = str(role_value)
                if hasattr(role_value, 'name'):
                    role_name = role_value.name

                is_leader = self.env.get_unit_property(member_id, 'is_leader', False)
                health = self.env.get_unit_property(member_id, 'health', 0)

                print(f"  - Member: {member_string} (ID: {member_id})")
                print(f"    Role: {role_name}, Is Leader: {is_leader}, Health: {health}")

            print("")  # Empty line between units

        print("=== END PLATOON STRUCTURE ===\n")

    def debug_agent_structure(self):
        """
        Prints detailed information about all agents in the system.
        Enhanced to properly display the consistent agent mapping with correct agent IDs.
        """
        print("=" * 60)
        print(" AGENT STRUCTURE OVERVIEW")
        print("=" * 60)

        # Get the consistent agent mapping if available
        agent_to_unit = {}
        unit_to_agent = {}

        if hasattr(self, 'unit_id_to_agent_id'):
            unit_to_agent = self.unit_id_to_agent_id
            # Create reverse mapping
            for unit_id, agent_id in unit_to_agent.items():
                agent_to_unit[agent_id] = unit_id

        # Print total count of agents
        if hasattr(self, 'agent_id_to_role'):
            print(f"Total Agents: {len(self.agent_id_to_role)}")
            print(f"Active Agent IDs: {sorted(self.agent_id_to_role.keys())}")
        else:
            print(f"Total Agents: {len(self.agent_ids)}")
            print(f"Agent IDs: {self.agent_ids}")

        # Display agents by their consistent agent IDs
        print("\n=== AGENTS BY CONSISTENT ID ===")

        # Get all valid agent IDs to display (sorted)
        agent_ids_to_display = []
        if hasattr(self, 'agent_id_to_role'):
            agent_ids_to_display = sorted(self.agent_id_to_role.keys())
        elif agent_to_unit:
            agent_ids_to_display = sorted(agent_to_unit.keys())
        else:
            # Fall back to numeric agent ids up to 10 (reasonable for our system)
            agent_ids_to_display = list(range(1, 8))  # Assume max 7 agents

        # Process each agent ID
        for agent_id in agent_ids_to_display:
            # Get the current unit ID for this agent
            unit_id = agent_to_unit.get(agent_id)

            if unit_id is None:
                # Try get_current_unit_id if available
                if hasattr(self, 'get_current_unit_id'):
                    unit_id = self.get_current_unit_id(agent_id)

            # Get role name for this agent
            role_name = "Unknown"
            if hasattr(self, 'agent_id_to_role'):
                role_name = self.agent_id_to_role.get(agent_id, "Unknown")

            print(f"\nAgent {agent_id}: ({role_name})")

            if unit_id is None:
                print("  No current unit mapping")
                continue

            # Skip if position doesn't exist
            if unit_id not in self.env.state_manager.active_units:
                print(f"  Position {unit_id} no longer exists")
                continue

            # Get agent details
            agent_type = "Unknown"
            if hasattr(self, 'agent_types'):
                # Try to get agent type directly by agent_id
                agent_type = self.agent_types.get(agent_id, "Unknown")
                # If not found, try by unit_id
                if agent_type == "Unknown" and unit_id in self.agent_types:
                    agent_type = self.agent_types[unit_id]

            # Get unit properties
            unit_string = self.env.get_unit_property(unit_id, 'string_id', str(unit_id))
            position = self.env.get_unit_position(unit_id)
            health = self.env.get_unit_property(unit_id, 'health', 0)
            position_status = self.env.get_unit_property(unit_id, 'position_status', 'occupied')
            soldier_id = self.env.get_unit_property(unit_id, 'soldier_id', None)
            is_leader = self.env.get_unit_property(unit_id, 'is_leader', False)

            # Get role name
            role_value = self.env.get_unit_property(unit_id, 'role', None)
            role_name = "Unknown"
            if role_value is not None:
                role_name = str(role_value)
                if hasattr(role_value, 'name'):
                    role_name = role_value.name

            print(f"  Current Unit ID: {unit_id} ({unit_string})")
            print(f"  Position: {position}")
            print(f"  Health: {health}")
            print(f"  Status: {position_status}")
            print(f"  Soldier ID: {soldier_id}")
            print(f"  Is Leader: {is_leader}")
            print(f"  Role: {role_name}")
            print(f"  Agent Type: {agent_type}")

            # Get parent information
            parent_id = self.env.get_unit_property(unit_id, 'parent_id')
            if parent_id:
                parent_type = self.env.get_unit_property(parent_id, 'type')
                parent_string = self.env.get_unit_property(parent_id, 'string_id', str(parent_id))

                # Handle different type formats
                parent_type_str = str(parent_type)
                if hasattr(parent_type, 'name'):
                    parent_type_str = parent_type.name

                print(f"  Parent Unit: {parent_string} (ID: {parent_id})")
                print(f"  Parent Type: {parent_type_str}")

                # Count siblings
                siblings = self.env.get_unit_children(parent_id)
                print(f"  Sibling positions: {len(siblings)}")

                # Count living siblings
                living_siblings = sum(1 for sibling_id in siblings
                                      if self.env.get_unit_property(sibling_id, 'health', 0) > 0)
                print(f"  Living sibling positions: {living_siblings}")
            else:
                print("  No parent unit")

        # Print the unit-to-agent mapping if available
        print("\nConsistent Unit-to-Agent Mapping:")
        if hasattr(self, 'unit_id_to_agent_id'):
            for unit_id, agent_id in sorted(self.unit_id_to_agent_id.items(), key=lambda x: x[1]):
                unit_string = self.env.get_unit_property(unit_id, 'string_id', str(unit_id))
                role_name = "Unknown"
                if hasattr(self, 'agent_id_to_role') and agent_id in self.agent_id_to_role:
                    role_name = self.agent_id_to_role[agent_id]
                print(f"  Unit {unit_id} ({unit_string}) → Agent {agent_id}" +
                      (f" ({role_name})" if role_name != "Unknown" else ""))
        else:
            print("  No consistent mapping available")

        print("\n" + "=" * 60)


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
        Check if position has elevation advantage over other position with robust type handling.

        Args:
            position: Position to check from
            other_position: Position to check against

        Returns:
            Boolean indicating if position has elevation advantage
        """
        try:
            # Convert numpy types and ensure integers
            x1, y1 = int(float(position[0])), int(float(position[1]))
            x2, y2 = int(float(other_position[0])), int(float(other_position[1]))

            # Ensure positions are within bounds
            if not (0 <= x1 < self.width and 0 <= y1 < self.height and
                    0 <= x2 < self.width and 0 <= y2 < self.height):
                return False

            # Get elevation types with safety checks
            pos_elevation = self.terrain_manager.get_elevation_type((x1, y1))
            other_elevation = self.terrain_manager.get_elevation_type((x2, y2))

            # Handle possible None values
            if pos_elevation is None or other_elevation is None:
                return False

            # Extract values safely - handle different ways the value might be stored
            if hasattr(pos_elevation, 'value'):
                pos_value = pos_elevation.value  # Enum case
            elif isinstance(pos_elevation, int):
                pos_value = pos_elevation  # Integer case
            else:
                # Try string representation as last resort
                try:
                    pos_value = int(str(pos_elevation).split('.')[-1])
                except:
                    return False  # Can't determine value

            if hasattr(other_elevation, 'value'):
                other_value = other_elevation.value  # Enum case
            elif isinstance(other_elevation, int):
                other_value = other_elevation  # Integer case
            else:
                # Try string representation as last resort
                try:
                    other_value = int(str(other_elevation).split('.')[-1])
                except:
                    return False  # Can't determine value

            # Compare the extracted values
            return pos_value > other_value

        except Exception as e:
            if hasattr(self, 'env') and hasattr(self.env, 'debug_level') and self.env.debug_level > 0:
                print(f"[DEBUG] Elevation advantage check error: {e}, positions: {position}, {other_position}")
            return False  # Default to no advantage on error

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

    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Get points along line using Bresenham's algorithm with safety checks."""
        # Safety check for very distant or invalid points
        max_distance = 1000  # Set a reasonable maximum
        if (abs(x2 - x1) > max_distance or abs(y2 - y1) > max_distance or
                not (0 <= x1 < self.width and 0 <= y1 < self.height and
                     0 <= x2 < self.width and 0 <= y2 < self.height)):
            # Create a simplified line with fewer points
            num_points = 100
            points = []
            for i in range(num_points + 1):
                t = i / num_points
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                # Ensure points are within bounds
                x = max(0, min(x, self.width - 1))
                y = max(0, min(y, self.height - 1))
                points.append((x, y))
            return points

        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1

        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        # Use a different implementation for steep vs shallow lines
        if dx > dy:
            err = dx / 2.0
            max_steps = dx + 1  # Add safety counter
            steps = 0

            while steps < max_steps:
                points.append((x, y))
                if x == x2 and y == y2:
                    break

                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
                steps += 1
        else:
            err = dy / 2.0
            max_steps = dy + 1  # Add safety counter
            steps = 0

            while steps < max_steps:
                points.append((x, y))
                if x == x2 and y == y2:
                    break

                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                steps += 1

        return points


class MilitaryEnvironment(gym.Env):
    """
    MilitaryEnvironment Class - Core Military Simulation Environment

    This environment serves as the foundation for tactical military simulations and
    reinforcement learning scenarios. It provides a grid-based world with terrain,
    units, and combat mechanics managed by specialized component managers.

    Key Components:
    - StateManager: Tracks all unit properties and positions
    - TerrainManager: Handles terrain types, movement costs, and elevation
    - VisibilityManager: Calculates line of sight and observation
    - CombatManager: Processes engagements, suppression, and casualties
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
                'adjust_for_fire_rate': spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.int32
                ),
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

        # Initialize reward tracking variables
        self._pre_action_ammo = {}
        self._previous_enemy_count = 0
        self._previous_friendly_count = 0

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

        # Initialize reward tracking variables
        self._previous_enemy_count = sum(1 for unit_id in self.state_manager.active_units
                                         if self.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY
                                         and self.get_unit_property(unit_id, 'health', 0) > 0)

        self._previous_friendly_count = sum(1 for unit_id in self.state_manager.active_units
                                            if self.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY
                                            and self.get_unit_property(unit_id, 'health', 0) > 0)

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
        # Track pre-action state for reward calculation
        if hasattr(self, 'combat_manager'):
            self._track_ammo_before_actions()

        # Increment step counter
        self.current_step += 1

        # Validate action
        if not self._validate_action(action):
            return self._get_observation(), -1, False, False, {"error": "Invalid action"}

        # Execute action
        self._execute_action(action)

        # Update ongoing effects
        self.combat_manager.update_suppression_states()

        # Track post-action state for reward calculation
        if hasattr(self, 'combat_manager'):
            self._track_ammo_after_actions()

        # Calculate reward
        reward = self._calculate_reward(action)

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
            suppressed_units[i] = combat_state.get('suppression_level', 0)
            ammunition[i, 0] = combat_state.get('ammo_primary', 0)
            ammunition[i, 1] = combat_state.get('ammo_secondary', 0)

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
            if combat_state.get('suppression_level', 0) > 0.8:
                # Unit too suppressed to engage
                return False

        # Check ammunition for engagement actions
        if action['action_type'] in [ActionType.ENGAGE, ActionType.SUPPRESS]:
            combat_state = self.combat_manager.get_unit_combat_state(action['unit_id'])
            if combat_state.get('ammo_primary', 0) <= 0:
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

    def _execute_action(self, action: Dict):
        """
        Execute action to update environment state.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit taking action
                - action_type: Type of action to take
                - parameters: Additional parameters for action
        """
        # Validate action
        if not self._validate_action(action):
            return  # Invalid action, just return without execution

        action_type = action['action_type']

        # Route to specific action handler
        if action_type == ActionType.MOVE:
            self._execute_movement_action(action)
        elif action_type == ActionType.ENGAGE:
            self._execute_engagement_action(action)
        elif action_type == ActionType.SUPPRESS:
            self._execute_suppression_action(action)
        elif action_type == ActionType.BOUND:
            self._execute_bounding_action(action)
        elif action_type == ActionType.CHANGE_FORMATION:
            self._execute_formation_action(action)
        elif action_type == ActionType.REPORT:
            self._execute_report_action(action)
        elif action_type == ActionType.HALT:
            self._execute_halt_action(action)
        else:
            if self.debug_level > 0:
                print(f"Warning: Unrecognized action type {action_type}")

    def _calculate_reward(self, action):
        """Calculate reward for a single agent action (for non-MARL version)."""
        # Basic reward components
        reward = 0.0

        # 1. Enemy elimination reward
        current_enemy_count = sum(1 for unit_id in self.state_manager.active_units
                                  if self.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY
                                  and self.get_unit_property(unit_id, 'health', 0) > 0)

        enemy_casualties = self._previous_enemy_count - current_enemy_count
        self._previous_enemy_count = current_enemy_count

        # Reward for eliminating enemies
        reward += enemy_casualties * 5.0

        # 2. Friendly survival penalty
        current_friendly_count = sum(1 for unit_id in self.state_manager.active_units
                                     if self.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY
                                     and self.get_unit_property(unit_id, 'health', 0) > 0)

        friendly_casualties = self._previous_friendly_count - current_friendly_count
        self._previous_friendly_count = current_friendly_count

        # Penalty for friendly casualties
        reward -= friendly_casualties * 10.0

        # 3. Objective progress reward (if objective exists)
        if hasattr(self, 'objective') and self.objective:
            # Check if action was movement toward objective
            if action.get('action_type') == ActionType.MOVE:
                unit_id = action.get('unit_id')
                parameters = action.get('parameters', {})

                # Get current position
                unit_pos = self.get_unit_position(unit_id)

                # Calculate movement vector
                direction = parameters.get('direction', (0, 0))
                distance = parameters.get('distance', 1)

                # Convert to scalar if needed
                if isinstance(distance, (list, np.ndarray)):
                    distance = distance[0] if len(distance) > 0 else 1

                # Calculate objective direction and movement alignment
                obj_vector = (self.objective[0] - unit_pos[0], self.objective[1] - unit_pos[1])
                obj_distance = math.sqrt(obj_vector[0] ** 2 + obj_vector[1] ** 2)

                # Normalize objective vector
                if obj_distance > 0:
                    obj_vector = (obj_vector[0] / obj_distance, obj_vector[1] / obj_distance)

                    # Calculate dot product for direction alignment
                    alignment = direction[0] * obj_vector[0] + direction[1] * obj_vector[1]

                    # Reward for moving toward objective
                    if alignment > 0:
                        reward += alignment * 0.5

        return reward

    def _track_ammo_before_actions(self):
        """Track ammunition before action execution."""
        # print("[DEBUG] Tracking pre-action ammo")
        self._pre_action_ammo = {}

        # Track ammo for all agents if the attribute exists
        if hasattr(self, 'agent_ids'):
            valid_agent_ids = [agent_id for agent_id in self.agent_ids
                               if agent_id in self.state_manager.active_units]
        else:
            # Fallback to tracking all active units if agent_ids doesn't exist
            valid_agent_ids = list(self.state_manager.active_units)

        for agent_id in valid_agent_ids:
            try:
                unit_type = self.get_unit_property(agent_id, 'type')

                if unit_type in [UnitType.INFANTRY_TEAM, UnitType.INFANTRY_SQUAD, UnitType.WEAPONS_TEAM]:
                    # Get all children for teams/squads
                    children = self.get_unit_children(agent_id)
                    for child_id in children:
                        if child_id in self.state_manager.active_units:
                            ammo = self.combat_manager._get_unit_ammo(child_id, 'primary')
                            self._pre_action_ammo[child_id] = ammo
                else:
                    # Individual unit
                    ammo = self.combat_manager._get_unit_ammo(agent_id, 'primary')
                    self._pre_action_ammo[agent_id] = ammo
            except Exception as e:
                print(f"[DEBUG] Error tracking ammo for agent {agent_id}: {e}")

    def _track_ammo_after_actions(self):
        """Track ammunition used after action execution."""
        self._ammo_used = {}

        # Calculate ammo used for each unit
        for unit_id, pre_ammo in self._pre_action_ammo.items():
            if unit_id in self.state_manager.active_units:
                post_ammo = self.combat_manager._get_unit_ammo(unit_id, 'primary')
                ammo_used = max(0, pre_ammo - post_ammo)
                self._ammo_used[unit_id] = ammo_used

    def _check_termination(self):
        """
        Check if episode should terminate.
        Returns a tuple of (terminated, reason)
        """
        # Mission success condition - objective secured
        if hasattr(self, 'objective') and self._check_objective_secured():
            return True, "objective_secured"

        # Mission failure - casualties exceed threshold
        friendly_casualties = self._count_casualties(ForceType.FRIENDLY)
        if friendly_casualties >= self.casualty_threshold:
            return True, "mission_failed_casualties"

        # All enemies eliminated
        enemy_casualties = self._count_casualties(ForceType.ENEMY)
        total_enemies = sum(1 for unit_id in self.state_manager.active_units
                            if self.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY)
        if enemy_casualties >= total_enemies and total_enemies > 0:
            return True, "all_enemies_eliminated"

        return False, "not_terminated"

    def _check_truncation(self):
        """
        Check if episode should be truncated.
        Returns a tuple of (truncated, reason)
        """
        # Time limit exceeded
        if self.current_step >= self.max_steps:
            return True, "time_limit"

        # Ammunition exhausted
        if self._check_ammunition_exhausted():
            return True, "ammunition_exhausted"

        return False, "not_truncated"

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
                                  if unit_id in self.combat_manager.suppressed_units)

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
    def _execute_movement_action(self, action: Dict):
        """
        Handle unit movement action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit to move
                - parameters:
                    - direction: (dx, dy) movement vector
                    - distance: Movement distance

        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract movement parameters
        direction = parameters.get('direction', (0, 0))
        distance = parameters.get('distance', 1)
        if isinstance(distance, np.ndarray):
            distance = int(distance[0])

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
            # Basic movement for other units
            self.move_unit(unit_id, direction, distance)
            frames = []

    def _execute_engagement_action(self, action: Dict):
        """
        Handle direct engagement/fire action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit engaging
                - parameters:
                    - target_pos: Position to engage
                    - weapon_type: Weapon to use
                    - max_rounds: Maximum rounds to expend
                    - adjust_for_fire_rate: Whether to adjust for weapon fire rate
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract engagement parameters
        target_pos = parameters.get('target_pos')
        weapon_type = parameters.get('weapon_type', 'primary')
        max_rounds = parameters.get('max_rounds', 10)
        adjust_for_fire_rate = parameters.get('adjust_for_fire_rate', False)

        if isinstance(max_rounds, np.ndarray):
            max_rounds = int(max_rounds[0])

        if isinstance(adjust_for_fire_rate, np.ndarray):
            adjust_for_fire_rate = bool(adjust_for_fire_rate[0])

        # Create fire control object
        fire_control = FireControl(
            target_area=target_pos,
            max_rounds=max_rounds,
            time_limit=5,
            suppress_only=False,
            adjust_for_fire_rate=adjust_for_fire_rate
        )

        # Get unit type to determine engagement level
        unit_type = self.get_unit_property(unit_id, 'type')

        # Execute appropriate engagement based on unit type
        if unit_type == UnitType.INFANTRY_TEAM or unit_type == UnitType.WEAPONS_TEAM:
            results = self.combat_manager.execute_team_engagement(
                team_id=unit_id,
                target_pos=target_pos,
                engagement_type=EngagementType.POINT,
                control_params={
                    'max_rounds': max_rounds,
                    'adjust_for_fire_rate': adjust_for_fire_rate
                }
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

    def _execute_suppression_action(self, action: Dict):
        """
        Handle suppressive/area fire action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit providing suppression
                - parameters:
                    - target_pos: Position to suppress
                    - area_radius: Area of effect
                    - max_rounds: Maximum rounds to expend
                    - adjust_for_fire_rate: Whether to adjust for weapon fire rate

        Returns:
            Reward for suppression action
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract suppression parameters
        target_pos = parameters.get('target_pos')
        area_radius = parameters.get('area_radius', 3)
        max_rounds = parameters.get('max_rounds', 30)
        adjust_for_fire_rate = parameters.get('adjust_for_fire_rate', True)

        if isinstance(max_rounds, np.ndarray):
            max_rounds = int(max_rounds[0])

        if isinstance(adjust_for_fire_rate, np.ndarray):
            adjust_for_fire_rate = bool(adjust_for_fire_rate[0])

        # Create fire control object
        fire_control = FireControl(
            target_area=target_pos,
            area_radius=area_radius,
            max_rounds=max_rounds,
            time_limit=8,
            suppress_only=True,
            adjust_for_fire_rate=adjust_for_fire_rate
        )

        # Get unit type to determine suppression level
        unit_type = self.get_unit_property(unit_id, 'type')

        # Execute appropriate suppression based on unit type
        if unit_type == UnitType.INFANTRY_TEAM:
            results = self.combat_manager.execute_team_engagement(
                team_id=unit_id,
                target_pos=target_pos,
                engagement_type=EngagementType.AREA,
                control_params={
                    'max_rounds': max_rounds,
                    'area_radius': area_radius,
                    'suppress_only': True,
                    'adjust_for_fire_rate': adjust_for_fire_rate
                }
            )
        elif unit_type == UnitType.WEAPONS_TEAM:
            # Weapons teams are more effective at sustained suppression
            results = self.combat_manager.execute_team_engagement(
                team_id=unit_id,
                target_pos=target_pos,
                engagement_type=EngagementType.AREA,
                control_params={
                    'max_rounds': max_rounds,
                    'area_radius': area_radius,
                    'suppress_only': True,
                    'sustained': True,
                    'adjust_for_fire_rate': adjust_for_fire_rate
                }
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

    def _execute_bounding_action(self, action: Dict):
        """
        Handle bounding movement (one element moves while others provide security).

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit executing bounding movement
                - parameters:
                    - direction: Movement direction
                    - distance: Movement distance
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract bounding parameters
        direction = parameters.get('direction')
        distance = parameters.get('distance', 5)

        if isinstance(distance, np.ndarray):
            distance = int(distance[0])

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
            # Placeholder for platoon bounding implementation
            frames = []
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

    def _execute_formation_action(self, action: Dict):
        """
        Handle formation change action.

        Args:
            action: Dictionary containing:
                - unit_id: ID of unit changing formation
                - parameters:
                    - formation: New formation to apply
        """
        unit_id = action['unit_id']
        parameters = action['parameters']

        # Extract formation parameter
        formation = parameters.get('formation')
        if isinstance(formation, np.ndarray):
            formation = int(formation[0])

        # Map integer formation index to string formation name
        formation_map = {
            0: "team_wedge_right",
            1: "team_wedge_left",
            2: "team_line_right",
            3: "team_line_left",
            4: "team_column",
            5: "squad_column_team_wedge",
            6: "squad_column_team_column",
            7: "squad_line_team_wedge",
            8: "platoon_column",
            9: "platoon_wedge"
        }

        if isinstance(formation, int):
            formation = formation_map.get(formation, "team_wedge_right")

        # Get unit type
        unit_type = self.get_unit_property(unit_id, 'type')

        # Validate formation is appropriate for unit type
        from US_Army_PLT_Composition_vTest import US_IN_validate_formation
        if not US_IN_validate_formation(formation, unit_type):
            if self.debug_level > 0:
                print(f"Invalid formation {formation} for unit type {unit_type}")
            return  # 0.0  # No reward or penalty

        # Apply formation
        from US_Army_PLT_Composition_vTest import US_IN_apply_formation
        US_IN_apply_formation(self, unit_id, formation)

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

        # Check if unit is currently moving
        if self.get_unit_property(unit_id, 'moving', False):
            # Stop movement
            self.update_unit_property(unit_id, 'moving', False)

            # Small reward for responsive control
            return 0.5

        # No reward if unit wasn't moving
        return 0.0

    def _execute_report_action(self, action: Dict) -> float:
        """
        Handle basic report action (simplified from original).

        Args:
            action: Dictionary containing:
                - unit_id: ID of reporting unit
                - parameters:
                    - report_type: Type of report

        Returns:
            Reward for report action
        """
        # Simplified report action - just returns a small reward
        # This has been deprecated from the full system
        return 0.1

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
            'force_type': ForceType.FRIENDLY,  # Default to friendly force
            'position_status': 'occupied'  # Initialize position as occupied
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
        if hasattr(self, 'combat_manager'):
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
        Enhanced with proper boundary checking.

        Args:
            unit_id: ID of unit to move
            new_pos: New (x,y) position

        Raises:
            ValueError: If unit_id not found or position invalid
        """
        if unit_id not in self.state_manager.active_units:
            raise ValueError(f"Unit {unit_id} not found")

        # Validate and convert position to integers
        try:
            x, y = new_pos
            x = int(x)
            y = int(y)

            # Ensure position is within environment bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            new_pos = (x, y)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid position {new_pos}: {e}")

        # Get current position and update state tensor
        old_pos = self.state_manager.get_unit_position(unit_id)
        old_x, old_y = old_pos

        # Ensure old position is also valid
        if not (0 <= old_x < self.width and 0 <= old_y < self.height):
            # Just update the position in properties without touching the state tensor
            self.state_manager.unit_properties[unit_id]['position'] = new_pos
            return

        # Get current status flags from old position
        status_flags = self.state_manager.state_tensor[old_y, old_x, 3]

        # Clear old position in unit ID channel
        self.state_manager.state_tensor[old_y, old_x, 2] = 0
        self.state_manager.state_tensor[old_y, old_x, 3] = 0

        # Update position in state manager properties
        self.state_manager.unit_properties[unit_id]['position'] = new_pos

        # Update new position in state tensor
        self.state_manager.state_tensor[y, x, 2] = unit_id
        self.state_manager.state_tensor[y, x, 3] = status_flags  # Preserve status flags

        # Update suppression visualization if unit is suppressed
        if hasattr(self, 'combat_manager') and unit_id in self.combat_manager.suppressed_units:
            self.state_manager.state_tensor[y, x, 3] |= 1  # Set suppression bit

    def get_unit_property(self, unit_id: int, property_name: str, default_value=None):
        """
        Get specific unit property with optional default value.
        Works with position-based soldier properties.

        Args:
            unit_id: ID of unit position
            property_name: Name of property to get
            default_value: Value to return if property doesn't exist

        Returns:
            Property value or default if not found
        """
        try:
            # Special handling for ammunition
            if property_name in ['ammo_primary', 'ammo_secondary'] and hasattr(self, 'combat_manager'):
                weapon_type = property_name.split('_')[1]
                return self.combat_manager._get_unit_ammo(unit_id, weapon_type)

            # Special handling for soldier properties - would go here if needed
            # Currently all soldier props are stored with the position, so no special handling needed

            return self.state_manager.get_unit_property(unit_id, property_name, default_value)
        except ValueError:
            if default_value is not None:
                return default_value
            raise

    def update_unit_property(self, unit_id: int, property_name: str, value) -> None:
        """
        Update specific unit property with improved position-based system integration.
        Distinguishes between casualties and vacancies in position-based succession.

        Args:
            unit_id: ID of unit position
            property_name: Name of property to update
            value: New value for property

        Raises:
            ValueError: If unit_id not found
        """
        if unit_id not in self.state_manager.active_units:
            raise ValueError(f"Unit {unit_id} not found")

        # Special handling for position_status property
        if property_name == 'position_status':
            # Direct update to state manager
            self.state_manager.update_unit_property(unit_id, property_name, value)

            # If changing to vacant, handle vacancy specifically
            if value == 'vacant':
                # Check if this position needs vacancy handling
                if not self.get_unit_property(unit_id, 'vacancy_handled', False):
                    self._handle_vacancy(unit_id)

            return

        # Special handling for ammunition updates
        if property_name in ['ammo_primary', 'ammo_secondary'] and hasattr(self, 'combat_manager'):
            weapon_type = property_name.split('_')[1]

            # Initialize tracking if needed
            if unit_id not in self.combat_manager.ammo_tracking:
                self.combat_manager.ammo_tracking[unit_id] = {}

            self.combat_manager.ammo_tracking[unit_id][weapon_type] = value

        # Special handling for health - improved to check status flags first
        if property_name == 'health' and value <= 0:
            # Get current properties for comparison
            old_health = self.state_manager.get_unit_property(unit_id, 'health', 100)
            position_status = self.state_manager.get_unit_property(unit_id, 'position_status', 'occupied')
            already_handled = self.get_unit_property(unit_id, 'casualty_handled', False)
            vacancy_handled = self.get_unit_property(unit_id, 'vacancy_handled', False)

            print(f"\nDEBUG[update_unit_property]: Updating health for position {unit_id} to {value}")
            print(f"DEBUG[update_unit_property]: Current status: position_status={position_status}, "
                  f"old_health={old_health}, already_handled={already_handled}, "
                  f"vacancy_handled={vacancy_handled}")

            # Check each case in a specific order to ensure consistent behavior

            # Case 1: Check vacancy first - if this is a vacant position, just update the health
            if position_status == 'vacant' or vacancy_handled:
                print(
                    f"DEBUG[update_unit_property]: Position {unit_id} is vacant - updating health without casualty handling")
                self.state_manager.update_unit_property(unit_id, property_name, value)
                return

            # Case 2: Check if already handled as casualty
            elif position_status == 'casualty' or already_handled:
                print(
                    f"DEBUG[update_unit_property]: Position {unit_id} casualty already handled - just updating health")
                self.state_manager.update_unit_property(unit_id, property_name, value)
                return

            # Case 3: Handle new casualty - only if health was previously > 0
            elif old_health > 0:
                # Mark as handled BEFORE proceeding to prevent recursion
                print(f"DEBUG[update_unit_property]: Handling NEW casualty for position {unit_id}")

                # Update health in state manager first
                self.state_manager.update_unit_property(unit_id, property_name, value)

                # Mark position as casualty (not vacant)
                self.state_manager.update_unit_property(unit_id, 'position_status', 'casualty')

                # Then handle casualty
                self._handle_casualty(unit_id)
                self.state_manager.update_unit_property(unit_id, 'casualty_handled', True)
                return
            else:
                # Health already 0, just update value
                print(f"DEBUG[update_unit_property]: Position {unit_id} health already 0 - "
                      f"just updating health")
                self.state_manager.update_unit_property(unit_id, property_name, value)

        # Default case: update in state manager
        self.state_manager.update_unit_property(unit_id, property_name, value)

        # Update status flags in state tensor if appropriate
        if property_name == 'health' and value <= 0:
            try:
                pos = self.get_unit_position(unit_id)
                # Set casualty flag (bit 3)
                self.state_manager.state_tensor[pos[1], pos[0], 3] |= 8
            except:
                pass  # Position might be invalid
        elif property_name == 'suppressed' and value:
            try:
                pos = self.get_unit_position(unit_id)
                # Set suppression flag (bit 0)
                self.state_manager.state_tensor[pos[1], pos[0], 3] |= 1
            except:
                pass  # Position might be invalid

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

        # Get current parent for debugging
        old_parent = self.state_manager.get_unit_property(unit_id, 'parent_id')

        # Update in state manager
        self.state_manager.update_unit_property(unit_id, 'parent_id', parent_id)

        # Update force type to match parent if appropriate
        if parent_id:
            parent_force = self.state_manager.get_unit_property(parent_id, 'force_type', None)
            if parent_force:
                self.state_manager.update_unit_property(unit_id, 'force_type', parent_force)

        if self.debug_level > 1:
            print(f"Changed parent of {unit_id} from {old_parent} to {parent_id}")

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
        Initializes position_status property.

        Args:
            role: Soldier's role (from US_IN_Role enum)
            unit_id_str: String identifier for soldier
            position: Initial (x,y) position
            is_leader: Whether soldier is a leader

        Returns:
            Numeric unit ID for environment tracking
        """
        from US_Army_PLT_Composition_vTest import (
            US_IN_M4, US_IN_M249, US_IN_M320, US_IN_M240B, US_IN_JAVELIN
        )

        soldier_id = self._generate_unit_id()

        # Set up weapon properties based on role
        primary_weapon = None
        secondary_weapon = None
        observation_range = 50  # Default
        engagement_range = 40  # Default
        ammo_primary = 0
        ammo_secondary = 0  # Initialize to prevent reference-before-assignment errors

        # Determine weapons and capabilities by role name
        role_name = role.name if hasattr(role, 'name') else str(role)

        if 'LEADER' in role_name:
            primary_weapon = US_IN_M4
            ammo_primary = US_IN_M4.ammo_capacity
            observation_range = 60 if 'PLATOON' in role_name or 'SQUAD' in role_name else 50
        elif 'AUTO_RIFLEMAN' in role_name:
            primary_weapon = US_IN_M249
            ammo_primary = US_IN_M249.ammo_capacity
            engagement_range = US_IN_M249.max_range
        elif 'GRENADIER' in role_name:
            primary_weapon = US_IN_M4
            secondary_weapon = US_IN_M320
            ammo_primary = US_IN_M4.ammo_capacity
            ammo_secondary = US_IN_M320.ammo_capacity
        elif 'MACHINE_GUNNER' in role_name:
            primary_weapon = US_IN_M240B
            ammo_primary = US_IN_M240B.ammo_capacity
            engagement_range = US_IN_M240B.max_range
            observation_range = 60
        elif 'ANTI_TANK' in role_name:
            primary_weapon = US_IN_M4
            secondary_weapon = US_IN_JAVELIN
            ammo_primary = US_IN_M4.ammo_capacity
            ammo_secondary = US_IN_JAVELIN.ammo_capacity
            engagement_range = US_IN_JAVELIN.max_range  # Javelin has long range
        else:  # Default rifleman
            primary_weapon = US_IN_M4
            ammo_primary = US_IN_M4.ammo_capacity

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
            'ammo_secondary': ammo_secondary,
            'force_type': ForceType.FRIENDLY,
            'weapons_operational': True,
            'has_automatic_weapons': primary_weapon == US_IN_M249 or primary_weapon == US_IN_M240B,
            'parent_id': None,
            'position_status': 'occupied'  # Initialize position as occupied
        }

        # Add to state tracking
        self.state_manager.add_unit(soldier_id, properties)

        # Initialize ammunition tracking
        if hasattr(self, 'combat_manager'):
            self.combat_manager.ammo_tracking[soldier_id] = {
                'primary': ammo_primary,
                'secondary': ammo_secondary
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

                # Update position
                self.update_unit_position(unit_id, (new_x, new_y))

                # Create single frame for basic movement
                movement_frames = [self._capture_positions(unit_id)]

        # Update unit position in state manager after movement
        final_pos = self.get_unit_position(unit_id)
        self.state_manager.update_unit_position(unit_id, final_pos)

        # FIX: Safely reapply formation if needed
        if current_formation:
            # Check if current_formation is a string (formation name)
            if isinstance(current_formation, str):
                # Try to get formation template if needed
                from US_Army_PLT_Composition_vTest import US_IN_apply_formation
                try:
                    # Reapply the formation by name without needing the template
                    US_IN_apply_formation(self, unit_id, current_formation)
                except Exception as e:
                    print(f"Warning: Could not reapply formation after movement: {e}")
            else:
                print(f"Warning: Formation type mismatch, got {type(current_formation)} but expected string")

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
        path_points = self.visibility_manager._get_line_points(
            start_pos[0], start_pos[1], end_pos[0], end_pos[1])

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
                los_result = self.visibility_manager.check_line_of_sight(
                    enemy_pos, point, for_observation=True)
                can_see = los_result['has_los']
                visibility = los_result['los_quality']

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
                # Remove rate_of_fire parameter which doesn't exist
                suppress_only=False
            )
            results = self.combat_manager.execute_engagement(unit_id, fire_control)
            return results.hits > 0, results.damage_dealt

        # Legacy implementation if combat manager not available
        # Replace can_engage with more basic validation
        if not self._validate_fire(unit_id, target_pos):
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

        # Calculate hit probability - ensure weapon is a BaseWeapon
        distance = self._calculate_distance(unit_pos, target_pos)
        if not isinstance(weapon, BaseWeapon):
            # Default values if weapon is not proper type
            base_hit_prob = 0.7 if distance < 10 else 0.4
        else:
            base_hit_prob = self.calculate_hit_probability(distance, weapon)

        # Apply modifiers
        # Fix check_line_of_sight reference
        los_result = self.visibility_manager.check_line_of_sight(unit_pos, target_pos)
        visibility = los_result.get('los_quality', 0.5)
        cover = self.terrain_manager.get_cover(target_pos)

        final_hit_prob = base_hit_prob * visibility * (1 - cover)

        # Determine hit
        hit = random.random() < final_hit_prob

        # Calculate and apply damage
        damage = 0.0
        if hit:
            # Fix damage calculation when weapon is not proper type
            if not isinstance(weapon, BaseWeapon):
                # Default damage values
                damage = 25.0 if distance < 10 else 15.0
            else:
                damage = self.calculate_damage(distance, weapon)

            # Get unit at target position
            target_y, target_x = min(target_pos[1], self.height - 1), min(target_pos[0], self.width - 1)
            target_unit_id = self.state_manager.state_tensor[target_y, target_x, 2]

            # Fix target unit ID processing
            if isinstance(target_unit_id, np.ndarray):
                target_unit_id = int(target_unit_id.item())

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

    def _validate_fire(self, unit_id: int, target_pos: Tuple[int, int]) -> bool:
        """
        Basic validation for fire action (replacing can_engage).

        Args:
            unit_id: ID of firing unit
            target_pos: Position to target

        Returns:
            Boolean indicating if fire is valid
        """
        # Check if unit exists
        if unit_id not in self.state_manager.active_units:
            return False

        # Check if unit is alive
        health = self.get_unit_property(unit_id, 'health', 0)
        if health <= 0:
            return False

        # Check if target position is in bounds
        if not (0 <= target_pos[0] < self.width and 0 <= target_pos[1] < self.height):
            return False

        # Check if weapon is operational
        weapons_operational = self.get_unit_property(unit_id, 'weapons_operational', True)
        if not weapons_operational:
            return False

        # Get positions and range
        unit_pos = self.get_unit_position(unit_id)
        distance = self._calculate_distance(unit_pos, target_pos)
        engagement_range = self.get_unit_property(unit_id, 'engagement_range', 40)

        # Check range
        if distance > engagement_range:
            return False

        # Check line of sight
        los_result = self.visibility_manager.check_line_of_sight(unit_pos, target_pos)
        if not los_result['has_los']:
            return False

        return True

    def apply_damage(self, unit_id: int, damage: float) -> None:
        """
        Apply damage to unit position, handling casualties when health reaches zero.
        Works with position-based soldier system.

        Args:
            unit_id: ID of position taking damage
            damage: Amount of damage to apply
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
            # Calculate team health based on member position health
            team_members = self.get_unit_children(parent_id)
            if team_members:
                total_health = sum(self.state_manager.get_unit_property(m, 'health', 0)
                                   for m in team_members)
                avg_health = total_health / len(team_members)
                self.state_manager.update_unit_property(parent_id, 'health', avg_health)

        # Handle casualty if health reached zero
        if new_health <= 0:
            self._handle_casualty(unit_id)

        # Update status indicators in state tensor if health is critically low
        if new_health < 25 and new_health > 0:
            try:
                pos = self.get_unit_position(unit_id)
                # Set wounded flag (bit 2) - Binary: 0100
                self.state_manager.state_tensor[pos[1], pos[0], 3] |= 4
            except:
                pass  # Position might be invalid

    # Utility Functions
    def _calculate_distance(self, pos1, pos2):
        """Calculate distance between two points with robust type handling."""
        try:
            # Convert inputs to float first to handle various numeric types
            x1, y1 = float(pos1[0]), float(pos1[1])
            x2, y2 = float(pos2[0]), float(pos2[1])

            # Calculate Euclidean distance
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Check for invalid results
            if not math.isfinite(distance):
                print(f"[DEBUG] Invalid distance calculation result: {distance}")
                return float('inf')  # Return large distance as fallback

            return distance
        except Exception as e:
            print(f"[DEBUG] Error in distance calculation: {e}")
            return float('inf')  # Return large distance on error

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
            'positions': positions
        }

    def _handle_casualty(self, unit_id):
        """
        Enhanced casualty handler that properly distinguishes between different
        casualty types and follows a clear, structured process for each.

        Args:
            unit_id: ID of position that has a casualty
        """
        # Skip if unit doesn't exist
        if unit_id not in self.state_manager.active_units:
            return

        # Get unit information for context
        unit_type = self.get_unit_property(unit_id, 'type')
        parent_id = self.get_unit_property(unit_id, 'parent_id')
        is_leader = self.get_unit_property(unit_id, 'is_leader', False)
        force_type = self.get_unit_property(unit_id, 'force_type', ForceType.FRIENDLY)
        unit_string = self.get_unit_property(unit_id, 'string_id', str(unit_id))
        is_agent = self.get_unit_property(unit_id, 'is_agent', False)
        soldier_id = self.get_unit_property(unit_id, 'soldier_id', None)
        health = self.get_unit_property(unit_id, 'health', 0)
        position_status = self.get_unit_property(unit_id, 'position_status', 'occupied')

        # Additional logging information
        if self.debug_level > 0:
            print(f"\nDEBUG[handle_casualty]: Processing position {unit_id} ({unit_string})")
            print(f"DEBUG[handle_casualty]: Status: health={health}, position_status={position_status}")
            print(
                f"DEBUG[handle_casualty]: Attributes: is_leader={is_leader}, is_agent={is_agent}, soldier_id={soldier_id}")

        # Skip if this is a vacant position from succession
        if position_status == 'vacant' or self.get_unit_property(unit_id, 'vacancy_handled', False):
            if self.debug_level > 0:
                print(
                    f"DEBUG[handle_casualty]: Position {unit_id} is vacant from succession, not a casualty - skipping")
            return

        # Skip if the position is already alive or has already been handled as a casualty
        if health > 0:
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: Position {unit_id} has health {health}, not a casualty - skipping")
            return

        if self.get_unit_property(unit_id, 'casualty_handled', False):
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: Position {unit_id} casualty already handled - skipping")
            return

        # Now we know this is a real casualty that needs processing
        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Confirmed this is a NEW casualty that needs processing")

        # Add casualty to force type casualty list for logging if available
        self._add_to_casualty_list(soldier_id, force_type)

        # Mark position as casualty - do this FIRST before any further processing
        self.state_manager.update_unit_property(unit_id, 'position_status', 'casualty')

        # CASE 1: If it's a regular soldier (not a leader)
        if not is_leader:
            self._handle_regular_casualty(unit_id, soldier_id)

        # CASE 2: If it's a team leader
        elif is_leader and parent_id and not is_agent:
            self._handle_team_leader_casualty(unit_id, soldier_id, parent_id)

        # CASE 3: If it's a squad leader (or other agent)
        elif is_leader and parent_id and is_agent:
            self._handle_squad_leader_casualty(unit_id, soldier_id, parent_id)

        # CASE 4: Check for squad consolidation after casualties
        if parent_id:
            # Find the squad the unit belongs to (might be a team's parent)
            squad_id = parent_id
            squad_type = self.get_unit_property(squad_id, 'type')
            squad_type_str = str(squad_type)

            # If this is a team, get its parent squad
            if 'INFANTRY_SQUAD' not in squad_type_str and 'SQD' not in squad_type_str:
                squad_parent = self.get_unit_property(squad_id, 'parent_id')
                if squad_parent:
                    parent_type = self.get_unit_property(squad_parent, 'type')
                    parent_type_str = str(parent_type)
                    if 'INFANTRY_SQUAD' in parent_type_str or 'SQD' in parent_type_str:
                        squad_id = squad_parent

            # Check if squad needs consolidation
            if hasattr(self, 'agent_manager') and hasattr(self.agent_manager, 'check_squad_needs_consolidation'):
                needs_consolidation, casualty_pct, squad_leader_alive = self.agent_manager.check_squad_needs_consolidation(
                    squad_id)

                if needs_consolidation:
                    if self.debug_level > 0:
                        print(
                            f"DEBUG[handle_casualty]: Squad {squad_id} needs consolidation ({casualty_pct:.1f}% casualties)")

                    # Find squad leader position for consolidation
                    squad_leader_pos = None
                    for child_id in self.get_unit_children(squad_id):
                        if self.get_unit_property(child_id, 'is_leader', False):
                            squad_leader_pos = child_id
                            break

                    if squad_leader_pos and hasattr(self.agent_manager, '_consolidate_squad_to_single_team'):
                        self.agent_manager._consolidate_squad_to_single_team(squad_id, squad_leader_pos)
                        if self.debug_level > 0:
                            print(f"DEBUG[handle_casualty]: Squad {squad_id} consolidated")

                    elif hasattr(self.agent_manager, 'update_after_casualties'):
                        # If we don't have direct access to consolidation, call update_after_casualties
                        self.agent_manager.update_after_casualties([squad_id])
                        if self.debug_level > 0:
                            print(f"DEBUG[handle_casualty]: Triggered consolidation check for squad {squad_id}")

    def _handle_regular_casualty(self, unit_id, soldier_id):
        """
        Handle a regular (non-leader) casualty.

        Args:
            unit_id: ID of position with casualty
            soldier_id: ID of soldier in the position
        """
        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Processing regular (non-leader) casualty")

        # Position is already marked as 'casualty' by the parent function

        # Remove soldier from position
        self.state_manager.update_unit_property(unit_id, 'soldier_id', None)

        # Ensure health is zero
        self.state_manager.update_unit_property(unit_id, 'health', 0)

        # Mark as handled to prevent reprocessing
        self.state_manager.update_unit_property(unit_id, 'casualty_handled', True)
        self.state_manager.update_unit_property(unit_id, 'vacancy_handled', False)

        # Reset ammunition
        if hasattr(self, 'combat_manager'):
            if unit_id in self.combat_manager.ammo_tracking:
                self.combat_manager.ammo_tracking[unit_id] = {
                    'primary': 0,
                    'secondary': 0
                }

        # Update status flags in state tensor
        try:
            pos = self.get_unit_position(unit_id)
            # Set casualty flag (bit 3)
            self.state_manager.state_tensor[pos[1], pos[0], 3] |= 8
        except:
            pass  # Position might be invalid

        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Regular casualty processing complete for {unit_id}")

    def _add_to_casualty_list(self, soldier_id, force_type):
        """
        Add a casualty to the appropriate force type casualty list for logging.

        Args:
            soldier_id: ID of the soldier casualty
            force_type: Type of force (FRIENDLY/ENEMY)
        """
        # Initialize casualty tracking if not already present
        if not hasattr(self, 'casualty_lists'):
            self.casualty_lists = {
                ForceType.FRIENDLY: [],
                ForceType.ENEMY: []
            }

        # Add to appropriate list
        if soldier_id is not None:
            self.casualty_lists[force_type].append(soldier_id)
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: Added soldier {soldier_id} to {force_type} casualty list")

    def _handle_team_leader_casualty(self, unit_id, soldier_id, parent_id):
        """
        Improved handler for team leader casualties that properly maintains position status
        during the succession process.

        Args:
            unit_id: ID of team leader position
            soldier_id: ID of the soldier
            parent_id: ID of the parent team
        """
        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Processing team leader casualty for position {unit_id}")

        # Initial casualty processing - do NOT mark as vacant yet
        # Remove soldier from position
        self.state_manager.update_unit_property(unit_id, 'soldier_id', None)

        # Ensure health is zero
        self.state_manager.update_unit_property(unit_id, 'health', 0)

        # Reset ammunition
        if hasattr(self, 'combat_manager'):
            if unit_id in self.combat_manager.ammo_tracking:
                self.combat_manager.ammo_tracking[unit_id] = {
                    'primary': 0,
                    'secondary': 0
                }

        # Update status flags in state tensor
        try:
            pos = self.get_unit_position(unit_id)
            # Set casualty flag (bit 3)
            self.state_manager.state_tensor[pos[1], pos[0], 3] |= 8
        except:
            pass  # Position might be invalid

        # Mark as NOT handled yet (succession will complete handling)
        self.state_manager.update_unit_property(unit_id, 'casualty_handled', False)
        self.state_manager.update_unit_property(unit_id, 'vacancy_handled', False)

        # Find and appoint successor
        successor_found = self._find_and_appoint_team_leader_successor(unit_id, parent_id)

        if successor_found:
            # After succession, mark as fully handled
            self.state_manager.update_unit_property(unit_id, 'casualty_handled', True)
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: Team leader succession completed for position {unit_id}")
        else:
            # If no successor found, just mark as handled and vacant
            self.state_manager.update_unit_property(unit_id, 'casualty_handled', True)
            self.state_manager.update_unit_property(unit_id, 'position_status', 'vacant')
            self.state_manager.update_unit_property(unit_id, 'vacancy_handled', True)
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: No successor found for team leader position {unit_id}")

    def _find_and_appoint_team_leader_successor(self, team_leader_pos, team_id):
        """
        Improved successor finder that properly maintains position status throughout.

        Args:
            team_leader_pos: Position ID of the team leader
            team_id: ID of the team

        Returns:
            Boolean indicating if a successor was found and appointed
        """
        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Finding successor for team leader position {team_leader_pos}")

        # Get all team members
        team_members = self.get_unit_children(team_id)

        # Create a priority order for succession based on role
        succession_candidates = []

        # First check: Find all living riflemen
        for member_id in team_members:
            if member_id == team_leader_pos:
                continue  # Skip the vacant team leader position

            # Skip if already a casualty or vacant
            if self.get_unit_property(member_id, 'health', 0) <= 0:
                continue

            position_status = self.get_unit_property(member_id, 'position_status', 'occupied')
            if position_status != 'occupied':
                continue

            # Check if it's a rifleman
            role = self.get_unit_property(member_id, 'role')
            role_name = ""

            if isinstance(role, int):
                try:
                    from US_Army_PLT_Composition_vTest import US_IN_Role
                    role_name = US_IN_Role(role).name
                except:
                    role_name = str(role)
            else:
                role_name = str(role)

            # Add to candidates list with priority based on role
            priority = 999  # Default low priority

            if 'RIFLEMAN' in role_name and 'AUTOMATIC' not in role_name:
                priority = 1
            elif 'GRENADIER' in role_name:
                priority = 2
            elif 'AUTOMATIC' in role_name or 'AUTO_RIFLEMAN' in role_name:
                priority = 3

            succession_candidates.append({
                'position_id': member_id,
                'priority': priority,
                'role_name': role_name
            })

        # Sort by priority
        succession_candidates.sort(key=lambda x: x['priority'])

        # If no candidates found, return False
        if not succession_candidates:
            if self.debug_level > 0:
                print(
                    f"DEBUG[handle_casualty]: No succession candidates found for team leader position {team_leader_pos}")
            return False

        # Select the highest priority candidate
        successor = succession_candidates[0]
        successor_pos = successor['position_id']

        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Selected successor: Position {successor_pos} ({successor['role_name']})")

        # Move soldier to leader position, maintaining proper position status
        return self._move_soldier_to_leader_position(
            source_pos=successor_pos,
            target_pos=team_leader_pos,
            is_agent=False  # Team leaders are not agents by default
        )

    def _handle_squad_leader_casualty(self, unit_id, soldier_id, parent_id):
        """
        Improved squad leader casualty handler that ensures proper status transitions.

        Args:
            unit_id: ID of squad leader position
            soldier_id: ID of the soldier
            parent_id: ID of the parent squad
        """
        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Processing squad leader casualty for position {unit_id}")

        # Initial casualty processing - do NOT mark as vacant yet
        # Remove soldier from position
        self.state_manager.update_unit_property(unit_id, 'soldier_id', None)

        # Ensure health is zero
        self.state_manager.update_unit_property(unit_id, 'health', 0)

        # Reset ammunition
        if hasattr(self, 'combat_manager'):
            if unit_id in self.combat_manager.ammo_tracking:
                self.combat_manager.ammo_tracking[unit_id] = {
                    'primary': 0,
                    'secondary': 0
                }

        # Update status flags in state tensor
        try:
            pos = self.get_unit_position(unit_id)
            # Set casualty flag (bit 3)
            self.state_manager.state_tensor[pos[1], pos[0], 3] |= 8
        except:
            pass  # Position might be invalid

        # Mark as NOT handled yet (succession will complete handling)
        self.state_manager.update_unit_property(unit_id, 'casualty_handled', False)
        self.state_manager.update_unit_property(unit_id, 'vacancy_handled', False)

        # Find and appoint successor
        successor_found = self._find_and_appoint_squad_leader_successor(unit_id, parent_id)

        if successor_found:
            # Mark as fully handled after succession
            self.state_manager.update_unit_property(unit_id, 'casualty_handled', True)
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: Squad leader succession completed for position {unit_id}")
        else:
            # If no successor found, just mark as handled and vacant
            self.state_manager.update_unit_property(unit_id, 'casualty_handled', True)
            self.state_manager.update_unit_property(unit_id, 'position_status', 'vacant')
            self.state_manager.update_unit_property(unit_id, 'vacancy_handled', True)
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: No successor found for squad leader position {unit_id}")

        # Special case for agents
        if hasattr(self, 'agent_manager'):
            # Ensure agent remains in agent_ids list
            if unit_id not in self.agent_manager.agent_ids:
                self.agent_manager.agent_ids.append(unit_id)
                if self.debug_level > 0:
                    print(f"DEBUG[handle_casualty]: Added position {unit_id} back to agent_ids list")

            # Set agent type if applicable
            if hasattr(self.agent_manager, 'agent_types'):
                self.agent_manager.agent_types[unit_id] = 'SQUAD'
                if self.debug_level > 0:
                    print(f"DEBUG[handle_casualty]: Set agent type for position {unit_id} to SQUAD")

        # Also update environment's agent_ids list if it exists
        if hasattr(self, 'agent_ids'):
            agent_ids = getattr(self, 'agent_ids', [])
            if unit_id not in agent_ids:
                agent_ids.append(unit_id)
            if self.debug_level > 0:
                print(f"DEBUG[handle_casualty]: Added position {unit_id} to environment agent_ids list")

        # Explicitly trigger consolidation check
        if hasattr(self, 'agent_manager') and hasattr(self.agent_manager, 'update_after_casualties'):
            print(f"DEBUG[handle_casualty]: Triggering squad consolidation check for squad {parent_id}")
            self.agent_manager.update_after_casualties([parent_id])

    def _find_and_appoint_squad_leader_successor(self, squad_leader_pos, squad_id):
        """
        Enhanced method to find a successor for the squad leader position following the defined process:
        1. First try to find team leaders within the squad
        2. If no team leaders, look for any living member

        This version maintains proper position status throughout the process and triggers team leader succession.

        Args:
            squad_leader_pos: Position ID of the squad leader
            squad_id: ID of the squad

        Returns:
            Boolean indicating if a successor was found and appointed
        """
        print(f"\nDEBUG[handle_casualty]: Finding successor for squad leader position {squad_leader_pos}")

        # Get all teams in the squad - enhanced to handle different type formats
        teams = []
        for unit_id in self.get_unit_children(squad_id):
            # Try multiple ways to identify teams
            unit_type = self.get_unit_property(unit_id, 'type')
            string_id = self.get_unit_property(unit_id, 'string_id', '')

            # Check if it's a team using type comparison
            is_team = False

            # Method 1: Direct enum comparison
            if hasattr(unit_type, 'value') and hasattr(UnitType.INFANTRY_TEAM, 'value'):
                if unit_type.value == UnitType.INFANTRY_TEAM.value:
                    is_team = True

            # Method 2: String comparison
            elif str(unit_type) == str(UnitType.INFANTRY_TEAM):
                is_team = True

            # Method 3: Check string_id for team identifiers
            if 'ATM' in string_id or 'BTM' in string_id or 'TEAM' in string_id:
                is_team = True

            # Method 4: Check if it has team members
            members = self.get_unit_children(unit_id)
            if members and len(members) >= 2:  # Teams typically have multiple members
                roles = [self.get_unit_property(m, 'role') for m in members]
                # If there are roles like rifleman, grenadier, etc., it's likely a team
                if roles and any('RIFLEMAN' in str(r) or 'GRENADIER' in str(r) for r in roles):
                    is_team = True

            if is_team:
                teams.append(unit_id)
                print(f"DEBUG[handle_casualty]: Found team {unit_id} ({string_id})")

        print(f"DEBUG[handle_casualty]: Found {len(teams)} teams in squad {squad_id}")

        # Step 1: First try to find a team leader to succeed
        for team_id in teams:
            for member_id in self.get_unit_children(team_id):
                # Check if this is a team leader
                is_leader = self.get_unit_property(member_id, 'is_leader', False)
                if not is_leader:
                    continue

                # Check if this leader is alive
                health = self.get_unit_property(member_id, 'health', 0)
                position_status = self.get_unit_property(member_id, 'position_status', 'occupied')

                if health > 0 and position_status == 'occupied':
                    print(f"DEBUG[handle_casualty]: Found team leader at position {member_id} to succeed")

                    # Transfer team leader to squad leader position
                    success = self._move_soldier_to_leader_position(
                        source_pos=member_id,
                        target_pos=squad_leader_pos,
                        is_agent=True  # Squad leaders are agents
                    )

                    if success:
                        # IMPORTANT NEW ADDITION: Trigger team leader succession for the team that lost its leader
                        print(f"DEBUG[handle_casualty]: Initiating team leader succession for team {team_id}")
                        from US_Army_PLT_Composition_vTest import US_IN_handle_leader_casualty
                        US_IN_handle_leader_casualty(self, team_id)

                    return success

        # Step 2: If no team leaders found, try to find any living member
        print(f"DEBUG[handle_casualty]: No team leaders found. Looking for any living member.")

        for team_id in teams:
            for member_id in self.get_unit_children(team_id):
                # Check if member is alive
                health = self.get_unit_property(member_id, 'health', 0)
                position_status = self.get_unit_property(member_id, 'position_status', 'occupied')

                if health > 0 and position_status == 'occupied':
                    print(f"DEBUG[handle_casualty]: Found member at position {member_id} to succeed")

                    # Transfer member to squad leader position
                    return self._move_soldier_to_leader_position(
                        source_pos=member_id,
                        target_pos=squad_leader_pos,
                        is_agent=True  # Squad leaders are agents
                    )

        # Step 3: As a last resort, check any direct children of the squad
        print(f"DEBUG[handle_casualty]: No team members found. Checking direct squad children as last resort.")

        # Try direct squad children as a fallback
        for member_id in self.get_unit_children(squad_id):
            # Skip the squad leader position itself
            if member_id == squad_leader_pos:
                continue

            # Check if the position is a healthy soldier (not a team)
            health = self.get_unit_property(member_id, 'health', 0)
            position_status = self.get_unit_property(member_id, 'position_status', 'occupied')
            soldier_id = self.get_unit_property(member_id, 'soldier_id')

            # Make sure this is a position with a soldier, not a team
            if soldier_id and health > 0 and position_status == 'occupied':
                print(f"DEBUG[handle_casualty]: Found direct squad child at position {member_id} to succeed")

                # Transfer to squad leader position
                return self._move_soldier_to_leader_position(
                    source_pos=member_id,
                    target_pos=squad_leader_pos,
                    is_agent=True  # Squad leaders are agents
                )

        print(f"DEBUG[handle_casualty]: No viable successor found for squad leader position {squad_leader_pos}")
        return False

    def _move_soldier_to_leader_position(self, source_pos, target_pos, is_agent=False):
        """
        Enhanced method to move a soldier from source position to target leadership position,
        correctly maintaining position status throughout.

        Args:
            source_pos: Position ID of the source (successor)
            target_pos: Position ID of the target leadership position
            is_agent: Whether the target position is an agent

        Returns:
            Boolean indicating success
        """
        if self.debug_level > 0:
            print(f"DEBUG[handle_casualty]: Moving soldier from position {source_pos} to {target_pos}")

        # Get source position properties to transfer
        soldier_id = self.get_unit_property(source_pos, 'soldier_id')
        health = self.get_unit_property(source_pos, 'health', 0)
        primary_weapon = self.get_unit_property(source_pos, 'primary_weapon')
        secondary_weapon = self.get_unit_property(source_pos, 'secondary_weapon')

        # Get ammunition if combat manager exists
        primary_ammo = 0
        secondary_ammo = 0

        if hasattr(self, 'combat_manager'):
            primary_ammo = self.combat_manager._get_unit_ammo(source_pos, 'primary')
            secondary_ammo = self.combat_manager._get_unit_ammo(source_pos, 'secondary')

        # Transfer properties to target position
        self.state_manager.update_unit_property(target_pos, 'soldier_id', soldier_id)
        self.state_manager.update_unit_property(target_pos, 'health', health)
        self.state_manager.update_unit_property(target_pos, 'primary_weapon', primary_weapon)
        self.state_manager.update_unit_property(target_pos, 'secondary_weapon', secondary_weapon)
        self.state_manager.update_unit_property(target_pos, 'position_status', 'occupied')

        # Update leader and agent status
        self.state_manager.update_unit_property(target_pos, 'is_leader', True)
        self.state_manager.update_unit_property(target_pos, 'is_agent', is_agent)

        # Update ammunition tracking
        if hasattr(self, 'combat_manager'):
            self.combat_manager.ammo_tracking[target_pos] = {
                'primary': primary_ammo,
                'secondary': secondary_ammo
            }

        # Verify the transfer was successful
        if self.debug_level > 0:
            verification = self._verify_position_transfer(target_pos, soldier_id, health)
            if verification:
                print(f"DEBUG[handle_casualty]: Soldier successfully moved to position {target_pos}")
            else:
                print(f"DEBUG[handle_casualty]: Failed to verify soldier transfer to position {target_pos}")
                return False

        # Mark target position as no longer a casualty
        self.state_manager.update_unit_property(target_pos, 'casualty_handled', True)
        self.state_manager.update_unit_property(target_pos, 'vacancy_handled', False)

        # IMPORTANT: Mark original position as vacant only AFTER successful transfer
        self.state_manager.update_unit_property(source_pos, 'position_status', 'vacant')
        self.state_manager.update_unit_property(source_pos, 'soldier_id', None)
        self.state_manager.update_unit_property(source_pos, 'health', 0)

        # Update ammunition for vacant position
        if hasattr(self, 'combat_manager'):
            self.combat_manager.ammo_tracking[source_pos] = {
                'primary': 0,
                'secondary': 0
            }

        # Mark source position as properly handled to prevent re-processing
        self.state_manager.update_unit_property(source_pos, 'casualty_handled', True)
        self.state_manager.update_unit_property(source_pos, 'vacancy_handled', True)

        # Update status flags in state tensor for source position
        try:
            pos = self.get_unit_position(source_pos)
            # Use status flag bit 4 for vacant (different from casualty which uses bit 3)
            self.state_manager.state_tensor[pos[1], pos[0], 3] |= 16  # Binary: 10000
        except:
            pass  # Position might be invalid

        return True

    def _verify_position_transfer(self, position_id, expected_soldier_id, expected_health):
        """
        Enhanced verification for soldier transfers during succession.

        Args:
            position_id: Position to verify
            expected_soldier_id: Expected soldier ID
            expected_health: Expected health value

        Returns:
            Boolean indicating verification success
        """
        # Check soldier ID
        actual_soldier_id = self.get_unit_property(position_id, 'soldier_id')
        if actual_soldier_id != expected_soldier_id:
            print(
                f"DEBUG[verify]: Position {position_id} has wrong soldier ID: {actual_soldier_id} vs {expected_soldier_id}")
            return False

        # Check health
        actual_health = self.get_unit_property(position_id, 'health', 0)
        if actual_health != expected_health:
            print(f"DEBUG[verify]: Position {position_id} has wrong health: {actual_health} vs {expected_health}")
            return False

        # Check position status
        position_status = self.get_unit_property(position_id, 'position_status', 'unknown')
        if position_status != 'occupied':
            print(f"DEBUG[verify]: Position {position_id} has wrong status: {position_status} vs 'occupied'")
            return False

        # Verify weapons if original position had them
        primary_weapon = self.get_unit_property(position_id, 'primary_weapon', None)
        if primary_weapon is None:
            print(f"DEBUG[verify]: Position {position_id} missing primary weapon")
            return False

        # Check ammo if combat manager exists
        if hasattr(self, 'combat_manager') and hasattr(self.combat_manager, 'ammo_tracking'):
            ammo = self.combat_manager.ammo_tracking.get(position_id, {})
            if 'primary' not in ammo or ammo['primary'] <= 0:
                print(f"DEBUG[verify]: Position {position_id} has no ammunition")
                return False

        print(f"DEBUG[verify]: Position {position_id} transfer verified successfully")
        return True

    def _handle_vacancy(self, unit_id: int) -> None:
        """
        Enhanced handler for a position that is vacant due to succession (not a casualty).
        This function specifically addresses positions that become vacant when
        soldiers are moved to leadership positions.

        Args:
            unit_id: ID of position that is vacant
        """
        # Skip if this position has already been handled as vacant
        if self.get_unit_property(unit_id, 'vacancy_handled', False):
            print(f"DEBUG[Environment]: Position {unit_id} already handled as vacant - skipping")
            return

        print(f"\nDEBUG[Environment]: Handling vacancy for position {unit_id}")

        # Get unit information for debug logging
        unit_string = self.get_unit_property(unit_id, 'string_id', str(unit_id))
        role_value = self.get_unit_property(unit_id, 'role')
        role_name = "Unknown"
        if isinstance(role_value, int):
            try:
                from US_Army_PLT_Composition_vTest import US_IN_Role
                role_name = US_IN_Role(role_value).name
            except:
                role_name = str(role_value)
        else:
            role_name = str(role_value)

        print(f"DEBUG[Environment]: Vacant position is {unit_string} with role {role_name}")

        # Get parent information if available
        parent_id = self.get_unit_property(unit_id, 'parent_id')
        parent_string = "None"
        if parent_id:
            parent_string = self.get_unit_property(parent_id, 'string_id', str(parent_id))

        print(f"DEBUG[Environment]: Vacant position belongs to unit {parent_string}")

        # Mark the position as "vacant" to distinguish from casualties
        self.state_manager.update_unit_property(unit_id, 'position_status', 'vacant')

        # Clear soldier ID if present
        if self.get_unit_property(unit_id, 'soldier_id') is not None:
            self.state_manager.update_unit_property(unit_id, 'soldier_id', None)

        # Update health directly in state manager (avoiding casualty handler)
        self.state_manager.update_unit_property(unit_id, 'health', 0)

        # Reset weapon status and ammo if needed
        if hasattr(self, 'combat_manager'):
            # Reset ammo tracking
            if unit_id in self.combat_manager.ammo_tracking:
                self.combat_manager.ammo_tracking[unit_id] = {
                    'primary': 0,
                    'secondary': 0
                }

        # Visual indicators in state tensor - use a different flag for vacant vs casualty
        try:
            pos = self.get_unit_position(unit_id)
            # Use status flag bit 4 for vacant (different from casualty which uses bit 3)
            self.state_manager.state_tensor[pos[1], pos[0], 3] |= 16  # Binary: 10000
        except:
            pass  # Position might be invalid

        # Flag to prevent future casualty handling for this position
        self.state_manager.update_unit_property(unit_id, 'vacancy_handled', True)

        # Also set casualty_handled flag to prevent double-handling
        self.state_manager.update_unit_property(unit_id, 'casualty_handled', True)

        print(f"DEBUG[Environment]: Position {unit_id} is now properly marked as vacant")

    def _format_unit_properties(self, properties: Dict) -> np.ndarray:
        """
        Format unit properties into fixed-size array for ML/RL.

        Args:
            properties: Dictionary of unit properties

        Returns:
            Fixed-size numpy array with formatted properties
        """
        # Initialize array with default values
        props = np.zeros(15, dtype=np.float32)

        # Standard property mapping
        if 'type' in properties:
            # Handle string type values (like 'soldier')
            if isinstance(properties['type'], str):
                # Map string types to numeric values
                type_map = {'soldier': 100.0}  # You can extend this map for other string types
                props[0] = type_map.get(properties['type'], 0.0)
            elif isinstance(properties['type'], Enum):
                props[0] = float(properties['type'].value)
            else:
                # Handle any other case
                try:
                    props[0] = float(properties['type'])
                except (ValueError, TypeError):
                    props[0] = 0.0  # Default value

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

    # Termination and Truncation Functions
    def _check_objective_secured(self) -> bool:
        """Check if objective is secured (all enemy units eliminated and all friendly units at objective)."""
        if not hasattr(self, 'objective'):
            return False

        # Check if all enemies in objective area are eliminated
        objective_radius = 15  # 150m

        # Check for living enemies in objective area
        for unit_id in self.state_manager.active_units:
            # Skip non-enemy units
            if self.get_unit_property(unit_id, 'force_type') != ForceType.ENEMY:
                continue

            # Check if in objective area
            unit_pos = self.get_unit_position(unit_id)
            distance = self._calculate_distance(unit_pos, self.objective)

            if distance <= objective_radius:
                # Check if alive
                health = self.get_unit_property(unit_id, 'health', 0)
                if health > 0:
                    return False  # Found living enemy in objective area

        # Check if all friendly tactical units are in objective area
        tactical_units = [uid for uid in self.state_manager.active_units
                          if self.get_unit_property(uid, 'type') in
                          [UnitType.INFANTRY_SQUAD, UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM]]

        friendly_units = [uid for uid in tactical_units
                          if self.get_unit_property(uid, 'force_type') == ForceType.FRIENDLY]

        # Count how many friendly units are at objective
        units_at_objective = 0
        for unit_id in friendly_units:
            unit_pos = self.get_unit_position(unit_id)
            distance = self._calculate_distance(unit_pos, self.objective)

            if distance <= objective_radius:
                units_at_objective += 1

        # Require at least 75% of friendly units at objective
        return units_at_objective >= len(friendly_units) * 0.75

    def _check_ammunition_exhausted(self) -> bool:
        """Check if all friendly units have exhausted ammunition."""
        # Get all friendly units
        friendly_units = [uid for uid in self.state_manager.active_units
                          if self.get_unit_property(uid, 'force_type') == ForceType.FRIENDLY]

        # Calculate total ammo across all units
        total_ammo = 0
        for unit_id in friendly_units:
            ammo = self.get_unit_property(unit_id, 'ammo_primary', 0)
            total_ammo += ammo

        # Consider ammunition exhausted if less than 5% remains
        return total_ammo < 50  # Arbitrary low threshold

    def _count_casualties(self, force_type: ForceType) -> int:
        """Count casualties for a specific force type."""
        casualties = 0

        for unit_id in self.state_manager.active_units:
            unit_force = self.get_unit_property(unit_id, 'force_type')
            if unit_force == force_type:
                health = self.get_unit_property(unit_id, 'health', 0)
                if health <= 0:
                    casualties += 1

        return casualties

    def _get_termination_reason(self) -> str:
        """Get reason for termination."""
        if hasattr(self, 'objective') and self._check_objective_secured():
            return "objective_secured"

        friendly_casualties = self._count_casualties(ForceType.FRIENDLY)
        if friendly_casualties >= self.casualty_threshold:
            return "mission_failed_casualties"

        enemy_casualties = self._count_casualties(ForceType.ENEMY)
        total_enemies = sum(1 for unit_id in self.state_manager.active_units
                            if self.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY)
        if enemy_casualties >= total_enemies and total_enemies > 0:
            return "all_enemies_eliminated"

        return "unknown"

    def _get_truncation_reason(self) -> str:
        """Get reason for truncation."""
        if self.current_step >= self.max_steps:
            return "time_limit"

        if self._check_ammunition_exhausted():
            return "ammunition_exhausted"

        return "unknown"

    def _initialize_units(self, unit_init: Dict) -> None:
        """
        Initialize units based on provided configuration.

        Args:
            unit_init: Dictionary with unit initialization parameters
        """
        # Create platoon if specified
        if 'platoon' in unit_init:
            from US_Army_PLT_Composition_vTest import US_IN_create_platoon

            plt_pos = unit_init['platoon'].get('position', (10, 10))
            plt_num = unit_init['platoon'].get('number', 1)

            plt_id = US_IN_create_platoon(self, plt_num, plt_pos)

            if 'objective' in unit_init:
                self.objective = unit_init['objective']

        # Create enemy units if specified
        if 'enemies' in unit_init:
            for enemy in unit_init['enemies']:
                self._create_enemy_unit(enemy)

    def _create_enemy_unit(self, enemy_config: Dict) -> int:
        """
        Create enemy unit based on configuration.

        Args:
            enemy_config: Dictionary with enemy unit parameters

        Returns:
            ID of created enemy unit
        """
        unit_type = enemy_config.get('type', UnitType.INFANTRY_TEAM)
        position = enemy_config.get('position', (0, 0))
        unit_id_str = enemy_config.get('id_str', 'ENEMY')

        # Create basic unit
        unit_id = self.create_unit(unit_type, unit_id_str, position)

        # Set force type to ENEMY
        self.update_unit_property(unit_id, 'force_type', ForceType.ENEMY)

        # Set health if specified
        if 'health' in enemy_config:
            self.update_unit_property(unit_id, 'health', enemy_config['health'])

        # Set weapons if specified
        if 'weapons' in enemy_config:
            weapons = enemy_config['weapons']
            if 'primary' in weapons:
                self.update_unit_property(unit_id, 'primary_weapon', weapons['primary'])
            if 'secondary' in weapons:
                self.update_unit_property(unit_id, 'secondary_weapon', weapons['secondary'])

        # Create members if team or squad
        if unit_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM, UnitType.INFANTRY_SQUAD]:
            self._create_enemy_members(unit_id, enemy_config)

        return unit_id

    def _create_enemy_members(self, parent_id: int, config: Dict) -> None:
        """
        Create members for enemy team or squad.

        Args:
            parent_id: ID of parent unit
            config: Unit configuration
        """
        unit_type = self.get_unit_property(parent_id, 'type')
        position = self.get_unit_position(parent_id)

        # Default member counts
        member_count = 4 if unit_type == UnitType.INFANTRY_TEAM else 2  # 2 for weapons team

        # Override with config if provided
        if 'member_count' in config:
            member_count = config['member_count']

        # Create basic members
        for i in range(member_count):
            # Calculate offset position
            offset_x = i * 2
            offset_y = -2
            member_pos = (position[0] + offset_x, position[1] + offset_y)

            # Create as soldier
            from US_Army_PLT_Composition_vTest import US_IN_Role
            role = US_IN_Role.RIFLEMAN

            # Leader for first member
            is_leader = (i == 0)
            if is_leader:
                role = US_IN_Role.TEAM_LEADER

            # Create soldier
            member_id = self.create_soldier(
                role=role,
                unit_id_str=f"{self.get_unit_property(parent_id, 'string_id')}-{i + 1}",
                position=member_pos,
                is_leader=is_leader
            )

            # Set as enemy
            self.update_unit_property(member_id, 'force_type', ForceType.ENEMY)

            # Set parent
            self.set_unit_hierarchy(member_id, parent_id)


class MARLMilitaryEnvironment(MilitaryEnvironment):
    """
    Multi-Agent Reinforcement Learning adaptation of the Military Environment.

    This environment extends the base MilitaryEnvironment to provide a multi-agent
    interface compatible with MARL algorithms like PPO. It handles agent-level
    observations, rewards, and actions while leveraging the full tactical simulation
    capabilities of the base environment.

    Key Features:
    - Agent-level observations with partial observability
    - Cooperative reward structure that incentivizes team coordination
    - Simplified action space for effective learning
    - Integration with PPO and other MARL algorithms
    """

    def __init__(self, config: EnvironmentConfig, objective_position: Tuple[int, int] = None):
        """
        Initialize MARL environment with base environment and MARL-specific parameters.

        Args:
            config: Base environment configuration
            objective_position: Position of mission objective
        """
        super().__init__(config)

        # Set objective
        self.objective = objective_position

        # Initialize agent manager
        self.agent_manager = AgentManager(self)

        # Initialize agent_ids attribute
        self.agent_ids = []

        # MARL specific parameters
        self.max_steps = 5000  # Maximum episode length
        self.casualty_threshold = 3  # Mission failure threshold

        # Initialize tracking fields
        self._previous_positions = {}
        self._previous_avg_distance_to_objective = float('inf')
        self._previous_enemy_count = 0
        self._previous_friendly_count = 0
        self._ammo_used_this_step = {}
        self._progress_history = []
        self.current_episode = 0

        # Initialize action space
        self.marl_action_space = self._create_marl_action_space()

        # Initialize observation space
        self.marl_observation_space = self._create_marl_observation_space()

    def reset(self, seed=None, options=None):
        """
        Reset environment with proper agent re-initialization.
        Enhanced to support different learning algorithms.

        Args:
            seed: Random seed
            options: Dictionary of reset options

        Returns:
            observations: Dictionary mapping agent IDs to observations
            info: Dictionary with reset information
        """
        # Increment episode counter
        self.current_episode += 1

        # Store original agent IDs if any exist - we'll try to maintain consistency
        original_agent_ids = self.agent_ids.copy() if hasattr(self, 'agent_ids') and self.agent_ids else []

        # Call parent reset
        observation, _ = super().reset(seed=seed, options=options)

        # After reset, identify agents using consistent mapping
        platoon_id = None
        if options and 'platoon_id' in options:
            platoon_id = options['platoon_id']
        elif 'unit_init' in options and 'platoon' in options['unit_init']:
            # Try to find a newly created platoon
            platoon_ids = [unit_id for unit_id in self.state_manager.active_units
                           if self.get_unit_property(unit_id, 'type') == UnitType.INFANTRY_PLATOON]

            if platoon_ids:
                platoon_id = platoon_ids[0]

        # Use the consistent agent mapping
        self.agent_ids = self.agent_manager.map_current_units_to_agent_ids(platoon_id)

        # print(f"[DEBUG] Reset complete with {len(self.agent_ids)} mapped agents")
        # print(f"[DEBUG] Agent IDs: {self.agent_ids}")

        # Initialize reward tracking with new agent IDs
        self._previous_enemy_count = self._count_living_enemies()
        self._previous_friendly_count = self._count_living_friendlies()
        self._previous_avg_distance_to_objective = self._get_avg_distance_to_objective()
        self._previous_positions = {agent_id: self.get_unit_position(self.agent_manager.get_current_unit_id(agent_id))
                                    for agent_id in self.agent_ids
                                    if self.agent_manager.get_current_unit_id(agent_id) is not None}

        # Initialize milestone tracking
        self._objective_milestones_reached = set()
        self._progress_history = []

        # If the environment has a marl_algorithm reference, initialize its episode
        # Using hasattr to check safely
        marl_algorithm = getattr(self, 'marl_algorithm', None)
        if marl_algorithm is not None:
            # Check if the algorithm has the initialize_episode method
            if hasattr(marl_algorithm, 'initialize_episode'):
                marl_algorithm.initialize_episode(self.agent_ids)

        # Return observations for new agents
        observations = {}
        for agent_id in self.agent_ids:
            try:
                unit_id = self.agent_manager.get_current_unit_id(agent_id)
                if unit_id:
                    observations[agent_id] = self._get_observation_for_agent(unit_id)
            except Exception as e:
                print(f"[DEBUG] Error getting observation for agent {agent_id}: {e}")

        return observations, {}

    def step(self, actions):
        """
        Execute actions for all agents and return results.
        Enhanced with improved reward distribution to ensure all agents receive rewards.

        This method is algorithm-agnostic but will integrate with any MARL algorithm
        that uses the environment's marl_algorithm attribute.

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            observations: Dictionary mapping agent IDs to observations
            rewards: Dictionary mapping agent IDs to rewards
            terminated_map: Dictionary mapping agent IDs to termination flags
            truncated_map: Dictionary mapping agent IDs to truncation flags
            infos: Dictionary mapping agent IDs to info dictionaries
        """
        # Track pre-action state for reward calculation
        self._track_ammo_before_actions()
        self._track_previous_positions()

        print(f"\n[DEBUG] Starting step with {len(actions)} actions for agents: {list(actions.keys())}")

        # Map actions from consistent agent IDs to current unit IDs
        unit_actions = {}
        for agent_id, action in actions.items():
            unit_id = self.agent_manager.get_current_unit_id(agent_id)
            if unit_id:
                unit_actions[unit_id] = action
            else:
                print(f"[DEBUG] Warning: No current unit ID found for agent {agent_id}")

        # Execute actions through agent manager with enhanced handling
        for unit_id, action in unit_actions.items():
            # Convert to format expected by core environment
            core_action = self._convert_action_to_core_format(unit_id, action)

            # Execute through agent manager
            try:
                self.agent_manager.execute_agent_action(unit_id, core_action)
            except Exception as e:
                print(f"Error executing action for unit {unit_id}: {e}")

        # Update ongoing effects
        if hasattr(self, 'combat_manager'):
            self.combat_manager.update_suppression_states()

        # Track post-action state for reward calculation
        self._track_ammo_after_actions()

        # Get observations for all agents
        observations = {}
        for agent_id in self.agent_ids:
            try:
                unit_id = self.agent_manager.get_current_unit_id(agent_id)
                if unit_id:
                    observations[agent_id] = self._get_observation_for_agent(unit_id)
            except Exception as e:
                print(f"[DEBUG] Error getting observation for agent {agent_id}: {e}")

        # Calculate rewards using consistent agent IDs
        rewards = {}
        core_rewards = self._calculate_marl_rewards(unit_actions)

        # Map rewards from unit IDs back to consistent agent IDs
        for unit_id, reward in core_rewards.items():
            agent_id = self.agent_manager.get_agent_id(unit_id)
            if agent_id:
                rewards[agent_id] = reward
            else:
                # If we can't find a mapping, just use the unit_id (fallback)
                rewards[unit_id] = reward

        # Check termination and truncation
        terminated, termination_reason = self._check_termination()
        truncated, truncation_reason = self._check_truncation()

        # Create terminal and truncated maps for all agents
        terminated_map = {agent_id: terminated for agent_id in self.agent_ids}
        truncated_map = {agent_id: truncated for agent_id in self.agent_ids}

        # Compile info dict
        infos = {}
        for agent_id in self.agent_ids:
            infos[agent_id] = {
                'termination_reason': termination_reason if terminated else None,
                'truncation_reason': truncation_reason if truncated else None
            }

        # If the environment has a marl_algorithm reference, allow it to store rewards/terminals
        # Safely get the marl_algorithm reference
        marl_algorithm = getattr(self, 'marl_algorithm', None)
        if marl_algorithm is not None:
            # Check if the algorithm has the store_rewards_and_terminals method
            if hasattr(marl_algorithm, 'store_rewards_and_terminals'):
                marl_algorithm.store_rewards_and_terminals(rewards, terminated_map, truncated_map)

        # Increment step counter
        self.current_step += 1

        print(f"[DEBUG] Step complete, current_step={self.current_step}")

        return observations, rewards, terminated_map, truncated_map, infos

    def _create_marl_action_space(self):
        """Create revised action space for MARL training with tactical decisions."""
        return spaces.Dict({
            # Basic action types: MOVE (0), ENGAGE (1), SUPPRESS (2), BOUND (3), HALT/FORMATION_CHANGE (4)
            'action_type': spaces.Discrete(5),

            # Movement
            'movement_params': spaces.Dict({
                'direction': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                'distance': spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)
            }),

            # Engagement
            'engagement_params': spaces.Dict({
                'target_pos': spaces.Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32),
                'max_rounds': spaces.Box(low=1, high=30, shape=(1,), dtype=np.int32),
                'suppress_only': spaces.Discrete(2),  # Boolean for suppression fire
                'adjust_for_fire_rate': spaces.Discrete(2)  # Boolean for automatic fire rate adjustment
            }),

            # Formation
            # 0-4: Team formations (wedge_right, wedge_left, line_right, line_left, column)
            # 5-8: Squad formations (column_team_wedge, column_team_column, line_team_wedge, vee_team_wedge)
            'formation': spaces.Discrete(8)  # Index into formation options
        })

    def _convert_action_to_core_format(self, agent_id, action):
        """
        Convert MARL action to core environment format.
        Handles mapping of formation indices to formation names.

        Args:
            agent_id: ID of agent
            action: MARL action dictionary

        Returns:
            Action dictionary in core environment format
        """
        action_type = action['action_type']

        # Initialize core action
        core_action = {
            'action_type': action_type,
            'movement_params': {},
            'engagement_params': {},
            'formation_params': {}
        }

        # Handle movement actions
        if action_type in [0, 3]:  # MOVE or BOUND
            core_action['movement_params'] = {
                'direction': action['movement_params']['direction'],
                'distance': action['movement_params']['distance']
            }

            # Add technique if provided
            if 'technique' in action['movement_params']:
                core_action['movement_params']['technique'] = action['movement_params']['technique']

        # Handle engagement actions
        elif action_type in [1, 2]:  # ENGAGE or SUPPRESS
            # Default engagement parameters
            core_action['engagement_params'] = {
                'target_pos': action['engagement_params']['target_pos']
            }

            # Add optional parameters if provided
            if 'max_rounds' in action['engagement_params']:
                core_action['engagement_params']['max_rounds'] = action['engagement_params']['max_rounds']

            if 'adjust_for_fire_rate' in action['engagement_params']:
                core_action['engagement_params']['adjust_for_fire_rate'] = action['engagement_params'][
                    'adjust_for_fire_rate']

            if 'engagement_type' in action['engagement_params']:
                core_action['engagement_params']['engagement_type'] = action['engagement_params']['engagement_type']

        # Handle formation change
        elif action_type == 5:  # CHANGE_FORMATION
            # Get formation type index and orientation
            formation_index = action['formation_params']['formation_type']
            orientation = action['formation_params']['orientation'][0]

            # Comprehensive formation mapping
            # Team formations
            team_formation_map = {
                0: "team_wedge_right",
                1: "team_wedge_left",
                2: "team_line_right",
                3: "team_line_left",
                4: "team_column"
            }

            # Squad formations
            squad_formation_map = {
                5: "squad_column_team_wedge",
                6: "squad_column_team_column",
                7: "squad_line_team_wedge",
                8: "squad_vee_team_wedge"
            }

            # Platoon formations
            platoon_formation_map = {
                9: "platoon_column",
                10: "platoon_wedge"
            }

            # Combine all formation maps
            all_formations = {**team_formation_map, **squad_formation_map, **platoon_formation_map}

            # Get formation based on index
            formation = all_formations.get(formation_index, "team_wedge_right")  # Default if index not found

            # Set formation parameters
            core_action['formation_params'] = {
                'formation': formation,
                'orientation': orientation
            }

            # Get agent type to determine appropriate formations
            agent_type = self.agent_manager.agent_types.get(agent_id)

            # Validate formation is appropriate for unit type
            if agent_type == 'SQUAD':
                # Squad should only use squad formations
                if formation_index < 5 or formation_index > 8:
                    # Adjust to a default squad formation
                    core_action['formation_params']['formation'] = "squad_column_team_wedge"
            elif agent_type in ['INFANTRY_TEAM', 'WEAPONS_TEAM']:
                # Teams should only use team formations
                if formation_index >= 5:
                    # Adjust to a default team formation
                    core_action['formation_params']['formation'] = "team_wedge_right"

        return core_action

    def _create_marl_observation_space(self):
        """Create simplified observation space for MARL training."""
        return spaces.Dict({
            # Agent state
            'agent_state': spaces.Dict({
                'position': spaces.Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32),
                'health': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                'ammo': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
                'suppressed': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            }),
            # Tactical information
            'tactical_info': spaces.Dict({
                'formation': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
                'orientation': spaces.Box(low=0, high=359, shape=(1,), dtype=np.int32),
                'unit_type': spaces.Box(low=0, high=2, shape=(1,), dtype=np.int32)
            }),
            # Knowledge of friendly units (via radio)
            'friendly_units': spaces.Box(
                low=0,
                high=max(self.width, self.height),
                shape=(10, 2),  # Positions of up to 10 friendly units
                dtype=np.int32
            ),
            # Known enemy positions
            'known_enemies': spaces.Box(
                low=0,
                high=max(self.width, self.height),
                shape=(10, 2),  # Positions of up to 10 known enemies
                dtype=np.int32
            ),
            # Objective position
            'objective': spaces.Box(
                low=0,
                high=max(self.width, self.height),
                shape=(2,),
                dtype=np.int32
            ),
            # Enhanced objective information
            'objective_info': spaces.Dict({
                'direction': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                'distance': spaces.Box(low=0.0, high=float('inf'), shape=(1,), dtype=np.float32)
            })
        })

    def _get_observation_for_agent(self, agent_id):
        """
        Get simplified observation for specific agent.
        """
        # Get agent position and properties
        agent_pos = self.get_unit_position(agent_id)
        agent_health = self.get_unit_property(agent_id, 'health', 100)
        agent_ammo = self.get_unit_property(agent_id, 'ammo_primary', 0)
        agent_suppressed = 0.0
        if hasattr(self, 'combat_manager') and agent_id in self.combat_manager.suppressed_units:
            agent_suppressed = self.combat_manager.suppressed_units[agent_id]['level']

        # Get tactical info
        parent_id = self.get_unit_property(agent_id, 'parent_id')
        formation = "unknown"
        orientation = 0
        if parent_id:
            formation = self.get_unit_property(parent_id, 'formation', "unknown")
            orientation = self.get_unit_property(parent_id, 'orientation', 0)

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

        # Determine agent type
        agent_type = self.agent_manager.agent_types.get(agent_id) if hasattr(self, 'agent_manager') else None

        # Map agent type to index
        unit_type_index = 0  # Default to INFANTRY_TEAM
        if agent_type == 'WEAPONS_TEAM':
            unit_type_index = 1
        elif agent_type == 'SQUAD':
            unit_type_index = 2

        # Get visible enemies
        known_enemies = self._get_visible_enemies(agent_id)

        # Get friendly units (via "radio")
        friendly_positions = self._get_friendly_positions(agent_id)

        # Calculate objective information
        if self.objective:
            # Direction to objective (normalized vector)
            direction_to_objective = self._calculate_direction_to_objective(agent_pos)
            # Distance to objective (in cells)
            distance_to_objective = self._calculate_distance(agent_pos, self.objective)
        else:
            direction_to_objective = np.array([0.0, 0.0], dtype=np.float32)
            distance_to_objective = np.array([0.0], dtype=np.float32)

        # Create simplified observation dictionary
        observation = {
            'agent_state': {
                'position': np.array(agent_pos, dtype=np.int32),
                'health': np.array([agent_health], dtype=np.float32),
                'ammo': np.array([agent_ammo], dtype=np.int32),
                'suppressed': np.array([agent_suppressed], dtype=np.float32)
            },
            'tactical_info': {
                'formation': np.array([formation_index], dtype=np.int32),
                'orientation': np.array([orientation], dtype=np.int32),
                'unit_type': np.array([unit_type_index], dtype=np.int32)
            },
            'friendly_units': friendly_positions,
            'known_enemies': known_enemies,
            'objective': np.array(self.objective if self.objective else [0, 0], dtype=np.int32),
            'objective_info': {
                'direction': direction_to_objective,
                'distance': np.array([distance_to_objective], dtype=np.float32)
            }
        }

        return observation

    def _get_friendly_positions(self, agent_id):
        """
        Get positions of friendly units via simulated radio comms.

        Args:
            agent_id: ID of agent requesting information

        Returns:
            Numpy array of friendly unit positions
        """
        # Get all friendly units
        friendly_units = [uid for uid in self.state_manager.active_units
                          if self.get_unit_property(uid, 'force_type') == ForceType.FRIENDLY
                          and uid != agent_id]

        # Calculate radio range (e.g., 100 cells)
        radio_range = 100

        # Get agent position
        agent_pos = self.get_unit_position(agent_id)

        # Initialize positions array
        positions = np.zeros((10, 2), dtype=np.int32)

        # Fill with friendly positions within radio range
        count = 0
        for uid in friendly_units:
            if count >= 10:  # Limit to 10 units
                break

            unit_pos = self.get_unit_position(uid)
            distance = self._calculate_distance(agent_pos, unit_pos)

            # Check if within radio range
            if distance <= radio_range:
                positions[count] = unit_pos
                count += 1

        return positions

    def _get_visible_enemies(self, agent_id):
        """
        Get positions of visible enemy units.

        Args:
            agent_id: ID of agent

        Returns:
            Numpy array of visible enemy positions
        """
        # Initialize positions array
        positions = np.zeros((10, 2), dtype=np.int32)

        # Get agent position and observation range
        agent_pos = self.get_unit_position(agent_id)
        observation_range = self.get_unit_property(agent_id, 'observation_range', 50)

        # Get all enemy units
        enemy_units = [uid for uid in self.state_manager.active_units
                       if self.get_unit_property(uid, 'force_type') == ForceType.ENEMY
                       and self.get_unit_property(uid, 'health', 0) > 0]

        # Find visible enemies
        count = 0
        for enemy_id in enemy_units:
            if count >= 10:  # Limit to 10 units
                break

            enemy_pos = self.get_unit_position(enemy_id)
            distance = self._calculate_distance(agent_pos, enemy_pos)

            # Check if within observation range
            if distance <= observation_range:
                # Check line of sight
                los_result = self.visibility_manager.check_line_of_sight(agent_pos, enemy_pos)
                if los_result['has_los']:
                    positions[count] = enemy_pos
                    count += 1

        return positions

    def _get_detailed_unit_state(self, unit_id):
        """
        Get detailed state information for a unit, including all members.
        Used for detailed state logging and debugging.

        Args:
            unit_id: ID of unit

        Returns:
            Dictionary with detailed state information
        """
        # Get basic unit info
        unit_type = self.get_unit_property(unit_id, 'type')
        string_id = self.get_unit_property(unit_id, 'string_id', str(unit_id))
        position = self.get_unit_position(unit_id)
        orientation = self.get_unit_property(unit_id, 'orientation', 0)
        formation = self.get_unit_property(unit_id, 'formation', None)

        # Get members
        members = []
        for member_id in self.get_unit_children(unit_id):
            # Get member properties
            member_pos = self.get_unit_position(member_id)
            member_role = self.get_unit_property(member_id, 'role')
            member_health = self.get_unit_property(member_id, 'health', 0)
            member_weapon = self.get_unit_property(member_id, 'primary_weapon', None)
            member_ammo = 0

            if hasattr(self, 'combat_manager'):
                member_ammo = self.combat_manager._get_unit_ammo(member_id, 'primary')

            # Add to members list
            members.append({
                'id': member_id,
                'role': member_role,
                'position': member_pos,
                'health': member_health,
                'weapon': member_weapon.name if hasattr(member_weapon, 'name') else str(member_weapon),
                'ammo': member_ammo,
                'is_leader': self.get_unit_property(member_id, 'is_leader', False)
            })

        # Create detailed state
        detailed_state = {
            'id': unit_id,
            'type': unit_type,
            'string_id': string_id,
            'position': position,
            'orientation': orientation,
            'formation': formation,
            'members_count': len(members),
            'members': members
        }

        return detailed_state

    # Reward related functions
    def _calculate_marl_rewards(self, actions):
        """
        Calculate rewards for all agents with improved consistency.
        Ensures all agents receive rewards.

        This method is algorithm-agnostic and focuses on proper reward distribution.

        Args:
            actions: Dictionary mapping unit IDs to actions

        Returns:
            Dictionary mapping unit IDs to rewards
        """
        # print("\n[DEBUG REWARD] Starting reward calculation")

        # Calculate shared team reward
        team_reward = self._calculate_team_reward()
        # print(f"[DEBUG REWARD] Team reward: {team_reward}")

        # Calculate individual agent rewards and combine with team reward
        rewards = {}

        # Get unit IDs mapped from agent IDs for consistent rewarding
        units_to_reward = {}
        for agent_id in self.agent_ids:
            unit_id = self.agent_manager.get_current_unit_id(agent_id)
            if unit_id:
                units_to_reward[unit_id] = agent_id

        # First, assign rewards to units that took actions
        for unit_id, action in actions.items():
            try:
                individual_reward = self._calculate_agent_reward(unit_id, action)
                rewards[unit_id] = team_reward + individual_reward

                # Check for infinite values
                if not np.isfinite(rewards[unit_id]):
                    # print(f"[DEBUG REWARD] WARNING: Infinite reward for unit {unit_id}!")
                    # print(f"[DEBUG REWARD] Team component: {team_reward}, Individual: {individual_reward}")
                    # Set to a safe maximum value
                    rewards[unit_id] = 100.0
            except Exception as e:
                # print(f"[DEBUG REWARD] Error calculating reward for unit {unit_id}: {e}")
                rewards[unit_id] = team_reward

        # Then, assign team reward to any units that didn't take actions but should receive rewards
        for unit_id, agent_id in units_to_reward.items():
            if unit_id not in rewards:
                rewards[unit_id] = team_reward
                # if self.debug_level > 0:
                    # print(f"[DEBUG REWARD] Assigned team reward to unit {unit_id} (mapped from agent {agent_id})")

        # Print debug info
        # print(f"[DEBUG REWARD] Enemy count: {self._previous_enemy_count} -> {self._count_living_enemies()}")
        # print(f"[DEBUG REWARD] Friendly count: {self._previous_friendly_count} -> {self._count_living_friendlies()}")
        # if hasattr(self, 'objective') and self.objective:
            # print(f"[DEBUG REWARD] Average distance to objective: {self._previous_avg_distance_to_objective}")

        # print(f"[DEBUG REWARD] Final rewards: {rewards}")

        return rewards

    def _calculate_team_reward(self):
        """
        Calculate team reward with enhanced graduated objective bonuses.
        Provides stronger incentives for making progress toward the objective.
        """
        total_reward = 0.0

        # 1. Enemy Elimination Reward
        current_enemy_count = self._count_living_enemies()
        enemy_casualties = max(0, self._previous_enemy_count - current_enemy_count)  # Ensure non-negative
        self._previous_enemy_count = current_enemy_count

        # Significant reward for eliminating enemies
        enemy_reward = enemy_casualties * 10.0
        total_reward += enemy_reward

        # 2. Enhanced Objective Progress Reward
        if hasattr(self, 'objective') and self.objective:
            try:
                current_avg_distance = self._get_avg_distance_to_objective()

                # Safety check for valid distances
                if math.isfinite(current_avg_distance) and math.isfinite(self._previous_avg_distance_to_objective):
                    # Improved distance change reward (scaled higher)
                    distance_improvement = self._previous_avg_distance_to_objective - current_avg_distance
                    distance_reward = max(-5.0,
                                          min(10.0, distance_improvement * 5.0))  # Increased scaling and max reward

                    # Apply graduated bonuses based on proximity to objective
                    proximity_bonus = self._calculate_proximity_bonus(current_avg_distance)

                    # Track if we reached new milestone for logging
                    self._track_objective_milestone(current_avg_distance)

                    # Total objective reward
                    objective_reward = distance_reward + proximity_bonus
                    total_reward += objective_reward

                    # Update for next step
                    self._previous_avg_distance_to_objective = current_avg_distance

                    # Log reward components for debugging
                    if self.debug_level > 0:
                        print(
                            f"[REWARD] Distance improvement: {distance_improvement:.2f}, reward: {distance_reward:.2f}")
                        print(f"[REWARD] Proximity bonus: {proximity_bonus:.2f}")

            except Exception as e:
                print(f"[DEBUG] Error in objective reward calculation: {e}")

        # 3. Survival Penalty
        current_friendly_count = self._count_living_friendlies()
        friendly_casualties = max(0, self._previous_friendly_count - current_friendly_count)  # Ensure non-negative
        self._previous_friendly_count = current_friendly_count

        # Significant penalty for losing friendly units
        survival_penalty = friendly_casualties * -20.0
        total_reward += survival_penalty

        # Ensure reward is finite
        if not math.isfinite(total_reward):
            print(f"[DEBUG] Infinite team reward detected! Using 0 instead.")
            total_reward = 0.0

        return total_reward

    def _calculate_agent_reward(self, agent_id, action):
        """
        Calculate individual reward component for an agent with improved robustness.
        Removed any reliance on tactical features.
        """
        try:
            individual_reward = 0.0

            # Get action type
            action_type = action['action_type']

            # Get agent state and position
            agent_pos = self.get_unit_position(agent_id)

            # 1. Reward for engagement when enemies are visible
            if action_type in [1, 2]:  # ENGAGE or SUPPRESS
                observation = self._get_observation_for_agent(agent_id)
                known_enemies = observation['known_enemies']
                visible_enemies = np.any(known_enemies.sum(axis=1) > 0)

                if visible_enemies:
                    # Reward for engaging visible enemies
                    individual_reward += 0.5

                    # Check if ammo was used (effective fire)
                    ammo_used = self._ammo_used_this_step.get(agent_id, 0)
                    if ammo_used > 0:
                        individual_reward += min(0.5, 0.1 * min(ammo_used, 10))  # Cap reward
                else:
                    # Small penalty for engaging with no visible enemies
                    individual_reward -= 0.5

            # 2. Reward for moving toward objective when appropriate
            elif action_type == 0:  # MOVE
                if hasattr(self, 'objective') and self.objective:
                    # Calculate previous and current distance to objective
                    prev_pos = self._previous_positions.get(agent_id, agent_pos)
                    prev_dist = self._calculate_distance(prev_pos, self.objective)
                    curr_dist = self._calculate_distance(agent_pos, self.objective)

                    # Reward for moving closer to objective
                    if curr_dist < prev_dist:
                        distance_improvement = prev_dist - curr_dist
                        # Limit reward for very large movements
                        individual_reward += min(1.0, max(0.0, distance_improvement * 0.2))

            # Ensure reward is finite and within reasonable bounds
            if not np.isfinite(individual_reward):
                individual_reward = 0.0

            # Clamp to reasonable range
            individual_reward = max(-2.0, min(2.0, individual_reward))

            return individual_reward

        except Exception as e:
            return 0.0  # Return neutral reward on error

    def _calculate_proximity_bonus(self, distance_to_objective):
        """
        Calculate bonus reward based on proximity to objective.
        Uses graduated bonuses at different distance thresholds.

        Args:
            distance_to_objective: Current distance to objective

        Returns:
            Bonus reward value
        """
        # Define distance thresholds and corresponding bonuses
        # Format: (distance_threshold, bonus_value)
        graduated_bonuses = [
            (5, 10.0),  # Within 5 cells: +10.0 reward
            (10, 9.0),  # Within 10 cells: +9.0 reward
            (15, 8.0),  # Within 15 cells: +8.0 reward
            (20, 7.0),  # Within 20 cells: +7.0 reward
            (25, 6.0),  # Within 25 cells: +6.0 reward
            (30, 5.0),  # Within 30 cells: +5.0 reward
            (35, 4.0),  # Within 35 cells: +4.0 reward
            (40, 3.0),  # Within 40 cells: +3.0 reward
            (45, 2.5),  # Within 45 cells: +2.5 reward
            (50, 2.0),  # Within 50 cells: +2.0 reward
            (55, 1.5),  # Within 55 cells: +1.5 reward
            (60, 1.0),  # Within 60 cells: +1.0 reward
        ]

        # Find the highest applicable bonus
        for threshold, bonus in graduated_bonuses:
            if distance_to_objective <= threshold:
                return bonus

        # No bonus if beyond all thresholds
        return 0.0

    def _track_objective_milestone(self, current_distance):
        """
        Track and log when agents reach new objective distance milestones.

        Args:
            current_distance: Current distance to objective
        """
        # Initialize milestone tracking if not already done
        if not hasattr(self, '_objective_milestones_reached'):
            self._objective_milestones_reached = set()

        # Initialize progress history if not already done
        if not hasattr(self, '_progress_history'):
            self._progress_history = []

        # Define milestones to track
        milestones = [60, 50, 40, 30, 20, 15, 10, 5]

        # Check each milestone
        for milestone in milestones:
            if current_distance <= milestone and milestone not in self._objective_milestones_reached:
                self._objective_milestones_reached.add(milestone)
                print(
                    f"[MILESTONE] Agents reached within {milestone} cells of objective! (Episode {getattr(self, 'current_episode', 0)}, Step {self.current_step})")

                # Log the milestone for later analysis
                self._log_milestone(milestone, current_distance)

                # Record in episode info if we have a progress tracker
                self._progress_history.append({
                    'episode': getattr(self, 'current_episode', 0),
                    'step': self.current_step,
                    'milestone': milestone,
                    'distance': current_distance
                })

    def _log_milestone(self, milestone, distance):
        """Log milestone achievement to a file."""
        # Create logs directory if it doesn't exist
        import os
        os.makedirs('./logs', exist_ok=True)

        # Log to file
        with open('./logs/objective_milestones.csv', 'a') as f:
            # Create header if file is new
            if os.path.getsize('./logs/objective_milestones.csv') == 0:
                f.write('timestamp,episode,step,milestone,distance\n')

            # Add log entry
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{timestamp},{getattr(self, "current_episode", 0)},{self.current_step},{milestone},{distance}\n')

    def create_milestone_visualization(self, output_dir='./logs'):
        """Create visualization of milestone progress over training."""
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        # Check if milestone file exists
        milestone_file = os.path.join(output_dir, 'objective_milestones.csv')
        if not os.path.exists(milestone_file):
            print(f"No milestone file found at {milestone_file}")
            return

        # Load milestone data
        df = pd.read_csv(milestone_file)

        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Plot milestones over episodes
        plt.figure(figsize=(12, 8))

        # Define milestone markers and colors
        milestones = [60, 50, 40, 30, 20, 15, 10, 5]
        colors = plt.cm.viridis(np.linspace(0, 1, len(milestones)))

        # Plot each milestone
        for i, milestone in enumerate(milestones):
            milestone_data = df[df['milestone'] == milestone]
            if not milestone_data.empty:
                plt.scatter(
                    milestone_data['episode'],
                    milestone_data['step'],
                    label=f'Within {milestone} cells',
                    color=colors[i],
                    s=100,
                    alpha=0.7
                )

        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Achievement of Objective Distance Milestones During Training')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add trendline for average progress
        if len(df) > 1:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(df['episode'], df['step'])
                plt.plot(df['episode'], intercept + slope * df['episode'], 'r--',
                         label=f'Trend (R²={r_value ** 2:.2f})')
                plt.legend()
            except:
                pass  # Skip trendline if scipy not available

        # Save plot
        plt.savefig(os.path.join(output_dir, 'milestone_progress.png'))
        plt.close()

        # Create episode time to achieve milestones plot
        plt.figure(figsize=(12, 8))

        # Group by episode and milestone to get first occurrence
        if 'episode' in df.columns:
            episode_milestones = df.groupby(['episode', 'milestone']).first().reset_index()

            # For each milestone, plot which episode it was achieved
            for milestone in milestones:
                milestone_episodes = episode_milestones[episode_milestones['milestone'] == milestone]
                if not milestone_episodes.empty:
                    plt.plot(
                        milestone_episodes['episode'],
                        [milestone] * len(milestone_episodes),
                        'o-',
                        label=f'Within {milestone} cells',
                        alpha=0.7
                    )

            plt.xlabel('Episode')
            plt.ylabel('Milestone (cells to objective)')
            plt.title('First Episode When Each Milestone Was Achieved')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            # Invert y-axis so closer distances are higher on the chart
            plt.gca().invert_yaxis()

            # Save plot
            plt.savefig(os.path.join(output_dir, 'milestone_episodes.png'))
            plt.close()

    def _calculate_direction_to_objective(self, position):
        """
        Calculate normalized direction vector from position to objective.

        Args:
            position: Starting position

        Returns:
            Normalized 2D direction vector as numpy array
        """
        if not hasattr(self, 'objective') or self.objective is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Calculate vector from position to objective
        dx = self.objective[0] - position[0]
        dy = self.objective[1] - position[1]

        # Calculate magnitude
        magnitude = math.sqrt(dx * dx + dy * dy)

        # Normalize vector (handle zero magnitude case)
        if magnitude > 0:
            return np.array([dx / magnitude, dy / magnitude], dtype=np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

    def _get_avg_distance_to_objective(self):
        """
        Calculate average distance of friendly units to objective.
        Uses consistent agent IDs to find units.
        """
        if not hasattr(self, 'objective') or self.objective is None:
            return float('inf')

        # Get friendly tactical units using consistent agent IDs
        friendly_units = []
        for agent_id in self.agent_ids:
            unit_id = self.agent_manager.get_current_unit_id(agent_id)
            if unit_id:
                friendly_units.append(unit_id)

        if not friendly_units:
            return float('inf')

        # Calculate average distance
        total_distance = 0
        for unit_id in friendly_units:
            unit_pos = self.get_unit_position(unit_id)
            distance = self._calculate_distance(unit_pos, self.objective)
            total_distance += distance

        return total_distance / len(friendly_units)

    def _track_previous_positions(self):
        """Track positions before action execution using consistent agent IDs."""
        if not hasattr(self, '_previous_positions'):
            self._previous_positions = {}

        # Track positions for each agent using their current unit IDs
        for agent_id in self.agent_ids:
            unit_id = self.agent_manager.get_current_unit_id(agent_id)
            if unit_id:
                try:
                    self._previous_positions[agent_id] = self.get_unit_position(unit_id)
                except:
                    pass  # Skip if we can't get position

    def _track_ammo_before_actions(self):
        """Track ammunition before action execution."""
        # print("[DEBUG] Tracking pre-action ammo")
        self._pre_action_ammo = {}

        # Track ammo for all agents
        valid_agent_ids = [agent_id for agent_id in self.agent_ids
                           if agent_id in self.state_manager.active_units]

        for agent_id in valid_agent_ids:
            try:
                unit_type = self.get_unit_property(agent_id, 'type')

                if unit_type in [UnitType.INFANTRY_TEAM, UnitType.INFANTRY_SQUAD, UnitType.WEAPONS_TEAM]:
                    # Get all children for teams/squads
                    children = self.get_unit_children(agent_id)
                    for child_id in children:
                        if child_id in self.state_manager.active_units:
                            ammo = self.combat_manager._get_unit_ammo(child_id, 'primary')
                            self._pre_action_ammo[child_id] = ammo
                else:
                    # Individual unit
                    ammo = self.combat_manager._get_unit_ammo(agent_id, 'primary')
                    self._pre_action_ammo[agent_id] = ammo
            except Exception as e:
                print(f"[DEBUG] Error tracking ammo for agent {agent_id}: {e}")

    def _track_ammo_after_actions(self):
        """Track ammunition used after action execution."""
        self._ammo_used_this_step = {agent_id: 0 for agent_id in self.agent_ids}

        # Calculate ammo used for each unit and aggregate to agent level
        for unit_id, pre_ammo in self._pre_action_ammo.items():
            post_ammo = self.combat_manager._get_unit_ammo(unit_id, 'primary')
            ammo_used = pre_ammo - post_ammo

            # Ensure non-negative
            ammo_used = max(0, ammo_used)

            # Find parent agent
            parent_id = self.get_unit_property(unit_id, 'parent_id')
            if parent_id in self.agent_ids:
                self._ammo_used_this_step[parent_id] += ammo_used
            elif unit_id in self.agent_ids:
                self._ammo_used_this_step[unit_id] += ammo_used

    def _count_living_enemies(self):
        """Count the number of living enemy units."""
        return sum(1 for unit_id in self.state_manager.active_units
                   if self.get_unit_property(unit_id, 'force_type') == ForceType.ENEMY
                   and self.get_unit_property(unit_id, 'health', 0) > 0)

    def _count_living_friendlies(self):
        """Count the number of living friendly units."""
        return sum(1 for unit_id in self.state_manager.active_units
                   if self.get_unit_property(unit_id, 'force_type') == ForceType.FRIENDLY
                   and self.get_unit_property(unit_id, 'health', 0) > 0)
