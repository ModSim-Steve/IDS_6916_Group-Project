"""
Russian Assault Detachment Capture Squad Composition

Purpose:

This file defines the structure and composition of a Russian Assault Detachment Capture Squad
for use in a military simulation environment. It includes detailed representations of
the machine gun team, grenade launcher team, and assault team along with their equipment and formations.



General Composition:

1 Squad consists of:
- 1 Squad Leader (SL)
- 1 Machine Gun Team (2 soldiers)
- 1 Grenade Launcher Team (2 soldiers)
- 1 Assault Team (4 soldiers including TL)
Total: 9 soldiers



Equipment Details:

AK-12 (Primary weapon for most soldiers):

- Range: 300 meters (30 spaces)
- Ammo: 210 rounds (7 magazines of 30 rounds each)
- Fire rate: 1 round per action
- Effect: Precision (straight red line)

PKM (Primary weapon for MG Gunner):

- Range: 1100 meters (110 spaces)
- Ammo: 600 rounds (3 belts of 200 rounds each)
- Fire rate: 6 rounds per action
- Effect: Area weapon (red narrow transparent vector, 6 spaces wide at max range)

RPG-7V2 (Primary weapon for GL Gunner):

- Range: 700 meters (70 spaces)
- Ammo: 4 rounds
- Fire rate: 1 round per action
- Effect: Area damage (100 hp at target, 70 hp 1 space away, 30 hp 2 spaces away)


Weapon Damage and Hit Probability:

Damage for weapons varies based on target distance (see in-code comments)
Hit probability for all weapons:

100% if target is ≤ 2/4 of max range
80% if target is between 2/4 and 3/4 of max range
70% if target is ≥ 3/4 of max range
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class HealthStatus(Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    BLACK = "black"


class AnimationState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    FIRING = "firing"
    RELOADING = "reloading"


class MovementTechnique(Enum):
    TRAVELING = "traveling"
    BOUNDING = "bounding"


class FormationType(Enum):
    WAGON_WHEEL = "wagon_wheel"
    FOLLOW_LEADER = "follow_leader"


def get_formation_type(formation_name: str) -> FormationType:
    """
    Determine formation type based on formation name.

    Args:
        formation_name: String identifier for the formation

    Returns:
        FormationType.WAGON_WHEEL for formations that rotate around pivot point
        FormationType.FOLLOW_LEADER for column formations
    """
    wagon_wheel_formations = {
        "team_wedge",
        "assault_team_wedge",
        "mg_team_spread",
        "gl_team_spread",
        "squad_wedge",
        "squad_line"
    }

    return (FormationType.WAGON_WHEEL
            if formation_name in wagon_wheel_formations
            else FormationType.FOLLOW_LEADER)


@dataclass
class Weapon:
    name: str
    range: int  # in spaces (1 space = 10 meters)
    ammo_count: int
    fire_rate: int  # rounds per action
    firing_animation: str
    is_area_weapon: bool
    area_effect_width: int = 0  # Only applicable for area weapons

    def calculate_hit_probability(self, distance: int) -> float:
        range_ratio = distance / self.range
        if range_ratio <= 0.5:
            return 1.0
        elif 0.5 < range_ratio < 0.75:
            return 0.8
        else:
            return 0.7

    def calculate_damage(self, distance: int) -> int:
        range_ratio = distance / self.range
        if range_ratio >= 0.75:
            return 80 if random.random() < 0.1 else 30
        elif 0.5 <= range_ratio < 0.75:
            return 80 if random.random() < 0.1 else 50
        else:
            return 90 if random.random() < 0.1 else 80


@dataclass
class AreaDamageWeapon(Weapon):
    def calculate_damage(self, distance: int) -> Tuple[int, int, int]:
        return 100, 70, 30  # Damage at target, 1 space away, and 2 spaces away


@dataclass
class Soldier:
    role: str
    health: int
    max_health: int
    primary_weapon: Weapon
    secondary_weapon: Optional[Weapon]
    observation_range: int
    engagement_range: int
    position: Tuple[int, int]
    is_leader: bool
    animation_state: AnimationState = AnimationState.IDLE

    @property
    def is_alive(self):
        return self.health > 0

    @property
    def health_status(self) -> HealthStatus:
        health_percentage = self.health / self.max_health
        if health_percentage > 0.8:
            return HealthStatus.GREEN
        elif health_percentage > 0.6:
            return HealthStatus.AMBER
        elif health_percentage > 0:
            return HealthStatus.RED
        else:
            return HealthStatus.BLACK

    def update_animation(self, new_state: AnimationState):
        self.animation_state = new_state

    def move_to(self, new_position: Tuple[int, int]):
        self.position = new_position
        self.update_animation(AnimationState.MOVING)

    def fire_weapon(self, is_primary: bool = True):
        weapon = self.primary_weapon if is_primary else self.secondary_weapon
        if weapon and weapon.ammo_count >= weapon.fire_rate:
            weapon.ammo_count -= weapon.fire_rate
            self.update_animation(AnimationState.FIRING)
            return True
        return False


@dataclass
class TeamMember:
    soldier: Soldier
    team_id: str


@dataclass
class Team:
    team_id: str
    leader: TeamMember
    members: List[TeamMember] = field(default_factory=list)
    orientation: int = 0  # Orientation in degrees, 0 is North
    formation: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_formation: str = "team_wedge"  # Formation name
    formation_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    parent_unit: Optional['Squad'] = None

    @property
    def all_members(self):
        return [self.leader] + self.members

    @property
    def alive_members(self):
        return [member for member in self.all_members if member.soldier.is_alive]

    def add_member(self, role: str, primary_weapon: Weapon, secondary_weapon: Optional[Weapon],
                   observation_range: int, engagement_range: int, position: Tuple[int, int]):
        soldier = Soldier(
            role=role,
            health=100,
            max_health=100,
            primary_weapon=primary_weapon,
            secondary_weapon=secondary_weapon,
            observation_range=observation_range,
            engagement_range=engagement_range,
            position=position,
            is_leader=(role == "Team Leader")
        )
        member = TeamMember(soldier, self.team_id)

        if role == "Team Leader":
            if self.leader is not None:
                raise ValueError(f"Team {self.team_id} already has a leader")
            self.leader = member
        else:
            self.members.append(member)

    def check_and_replace_leader(self):
        if not self.leader.soldier.is_alive:
            alive_members = [m for m in self.members if m.soldier.is_alive]
            if alive_members:
                new_leader = alive_members[0]
                new_leader.soldier.is_leader = True
                self.leader = new_leader
                self.members.remove(new_leader)
                print(f"New leader for {self.team_id}: {new_leader.soldier.role}")
            else:
                print(f"All members of {self.team_id} are incapacitated.")

    def set_formation(self, formation_positions: Dict[str, Tuple[int, int]], formation_name: str):
        """Set team formation positions and name."""
        self.formation_positions = formation_positions
        self.current_formation = formation_name
        self.apply_formation(self.leader.soldier.position)

    def apply_formation(self, base_position: Tuple[int, int] = (0, 0)):
        """Apply current formation from the given base position."""
        base_x, base_y = base_position

        # Position leader
        leader_rel_x, leader_rel_y = self.formation_positions.get("Team Leader", (0, 0))
        self.leader.soldier.position = (base_x + leader_rel_x, base_y + leader_rel_y)

        # Position other members
        for member in self.members:
            rel_x, rel_y = self.formation_positions.get(member.soldier.role, (0, 0))
            member.soldier.position = (base_x + rel_x, base_y + rel_y)

    def execute_movement(self, direction: Tuple[int, int], distance: int,
                         technique: MovementTechnique = MovementTechnique.TRAVELING) -> List[Dict]:
        """Execute movement maintaining assault team formation."""
        frames = []
        steps = 10

        formation_type = get_formation_type(self.current_formation)
        if formation_type == FormationType.WAGON_WHEEL:
            frames = self._execute_wagon_wheel(direction, distance)
        else:  # FormationType.FOLLOW_LEADER
            frames = self._execute_follow_leader_movement(direction, distance)

        return frames

    def _execute_wagon_wheel(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute wagon wheel movement with rotation and translation."""
        frames = []
        steps = 10
        rotation_steps = 4

        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        # Calculate target orientation
        target_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        rotation_needed = ((target_orientation - self.orientation + 180) % 360) - 180

        # Always rotate for direction change
        if dx != 0 or dy != 0:
            rotation_per_step = rotation_needed // rotation_steps
            for step in range(rotation_steps):
                self._rotate_formation(rotation_per_step)
                frames.append(self._capture_current_positions())

        # Execute movement
        step_distance = distance // steps
        for step in range(steps):
            # Move entire formation
            for member in [self.leader] + self.members:
                new_x = member.soldier.position[0] + (dx * step_distance) // math.sqrt(dx * dx + dy * dy)
                new_y = member.soldier.position[1] + (dy * step_distance) // math.sqrt(dx * dx + dy * dy)
                member.soldier.position = (new_x, new_y)
            frames.append(self._capture_current_positions())

        return frames

    def _execute_follow_leader_movement(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute follow-the-leader movement appropriate for assault teams."""
        frames = []
        steps = 10

        # Calculate movement parameters
        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        step_distance = distance // steps
        move_dx = (dx * step_distance) // math.sqrt(dx * dx + dy * dy)
        move_dy = (dy * step_distance) // math.sqrt(dx * dx + dy * dy)

        # Store path history for followers
        path_history = [self.leader.soldier.position]

        for step in range(steps):
            # Move leader
            new_x = self.leader.soldier.position[0] + move_dx
            new_y = self.leader.soldier.position[1] + move_dy
            self.leader.soldier.position = (new_x, new_y)
            path_history.append((new_x, new_y))

            # Move followers in echelon
            for i, member in enumerate(self.members):
                if len(path_history) > i + 2:
                    member.soldier.position = path_history[-(i + 2)]

            frames.append(self._capture_current_positions())

        return frames

    def _rotate_formation(self, angle: int):
        """Rotate formation around leader."""
        leader_pos = self.leader.soldier.position
        angle_rad = math.radians(angle)

        for member in self.members:
            rel_x = member.soldier.position[0] - leader_pos[0]
            rel_y = member.soldier.position[1] - leader_pos[1]

            # Rotate relative position
            new_rel_x = int(rel_x * math.cos(angle_rad) - rel_y * math.sin(angle_rad))
            new_rel_y = int(rel_x * math.sin(angle_rad) + rel_y * math.cos(angle_rad))

            # Apply new position
            member.soldier.position = (leader_pos[0] + new_rel_x, leader_pos[1] + new_rel_y)

        self.orientation = (self.orientation + angle) % 360

    def _capture_current_positions(self) -> Dict:
        """Capture current positions of all team members for animation."""
        return {
            'unit_type': 'Team',
            'team_id': self.team_id,
            'positions': [
                             {
                                 'role': self.leader.soldier.role,
                                 'position': self.leader.soldier.position,
                                 'is_leader': True
                             }
                         ] + [
                             {
                                 'role': member.soldier.role,
                                 'position': member.soldier.position,
                                 'is_leader': False
                             } for member in self.members
                         ]
        }


@dataclass
class SpecialTeam:
    team_id: str
    members: List[TeamMember] = field(default_factory=list)
    orientation: int = 0
    formation: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_formation: str = "special_team"
    formation_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    parent_unit: Optional['Squad'] = None

    @property
    def leader(self) -> TeamMember:
        return self.members[0] if self.members else None

    @property
    def all_members(self):
        return self.members

    @property
    def alive_members(self):
        return [member for member in self.members if member.soldier.is_alive]

    def add_member(self, role: str, primary_weapon: Weapon, secondary_weapon: Optional[Weapon],
                   observation_range: int, engagement_range: int, position: Tuple[int, int]):
        soldier = Soldier(
            role=role,
            health=100,
            max_health=100,
            primary_weapon=primary_weapon,
            secondary_weapon=secondary_weapon,
            observation_range=observation_range,
            engagement_range=engagement_range,
            position=position,
            is_leader=(len(self.members) == 0)  # First member added is the leader
        )
        member = TeamMember(soldier, self.team_id)
        self.members.append(member)

    def set_formation(self, formation_positions: Dict[str, Tuple[int, int]], formation_name: str):
        """Set formation positions for two-man special teams."""
        self.formation_positions = formation_positions
        self.current_formation = formation_name
        if self.leader:
            self.apply_formation(self.leader.soldier.position)

    def apply_formation(self, base_position: Tuple[int, int] = (0, 0)):
        """Apply current formation from the given base position."""
        base_x, base_y = base_position

        # Position both team members
        for i, member in enumerate(self.members):
            rel_x, rel_y = self.formation_positions.get(member.soldier.role, (0, 0))
            member.soldier.position = (base_x + rel_x, base_y + rel_y)

    def execute_movement(self, direction: Tuple[int, int], distance: int,
                         technique: MovementTechnique = MovementTechnique.TRAVELING) -> List[Dict]:
        """Execute movement for two-man special teams."""
        frames = []
        steps = 10

        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        # Calculate movement parameters
        step_distance = distance // steps
        magnitude = math.sqrt(dx*dx + dy*dy)
        move_dx = (dx * step_distance) // magnitude
        move_dy = (dy * step_distance) // magnitude

        # Special teams maintain fixed positions relative to each other
        for step in range(steps):
            # Move both members maintaining their relative positions
            for member in self.members:
                new_x = member.soldier.position[0] + move_dx
                new_y = member.soldier.position[1] + move_dy
                member.soldier.position = (new_x, new_y)

            frames.append(self._capture_current_positions())

        return frames

    def _capture_current_positions(self) -> Dict:
        """Capture current positions of all team members for animation."""
        return {
            'unit_type': 'Team',
            'team_id': self.team_id,
            'positions': [
                             {
                                 'role': self.leader.soldier.role,
                                 'position': self.leader.soldier.position,
                                 'is_leader': True
                             }
                         ] + [
                             {
                                 'role': member.soldier.role,
                                 'position': member.soldier.position,
                                 'is_leader': False
                             } for member in self.members
                         ]
        }


@dataclass
class Squad:
    squad_id: str
    leader: Soldier
    mg_team: SpecialTeam
    gl_team: SpecialTeam
    assault_team: Team
    orientation: int = 0
    formation: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_formation: str = "squad_wedge"
    formation_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def all_members(self):
        return [self.leader] + self.mg_team.all_members + self.gl_team.all_members + self.assault_team.all_members

    @property
    def alive_members(self):
        return [member for member in self.all_members if member.is_alive]

    def check_and_replace_leader(self):
        if not self.leader.is_alive:
            # Try assault team leader first
            if self.assault_team.leader.soldier.is_alive:
                self.leader = self.assault_team.leader.soldier
                self.leader.is_leader = True
                print(f"New squad leader from assault team")
                self.assault_team.check_and_replace_leader()
            # Then try MG team leader
            elif self.mg_team.leader.soldier.is_alive:
                self.leader = self.mg_team.leader.soldier
                self.leader.is_leader = True
                print(f"New squad leader from MG team")
            # Finally try GL team leader
            elif self.gl_team.leader.soldier.is_alive:
                self.leader = self.gl_team.leader.soldier
                self.leader.is_leader = True
                print(f"New squad leader from GL team")
            else:
                print(f"No available leaders for {self.squad_id}")

    def set_formation(self, formation_positions: Dict[str, Tuple[int, int]],
                      team_formation_alpha: Dict[str, Tuple[int, int]],
                      team_formation_bravo: Dict[str, Tuple[int, int]],
                      team_formation_charlie: Dict[str, Tuple[int, int]],
                      formation_name: str):
        """Set formations for entire Russian squad."""
        self.formation_positions = formation_positions
        self.current_formation = formation_name

        # Set formations for component teams
        self.mg_team.set_formation(team_formation_alpha, "mg_team")
        self.gl_team.set_formation(team_formation_bravo, "gl_team")
        self.assault_team.set_formation(team_formation_charlie, "assault_team")

    def apply_formation(self, base_position: Tuple[int, int] = (0, 0)):
        """Apply current formation from the given base position."""
        base_x, base_y = base_position

        # Position squad leader
        sl_x, sl_y = self.formation_positions.get("Squad Leader", (0, 0))
        self.leader.position = (base_x + sl_x, base_y + sl_y)

        # Position teams
        mg_x, mg_y = self.formation_positions.get("MG Team", (0, 0))
        self.mg_team.apply_formation((base_x + mg_x, base_y + mg_y))

        gl_x, gl_y = self.formation_positions.get("GL Team", (0, 0))
        self.gl_team.apply_formation((base_x + gl_x, base_y + gl_y))

        assault_x, assault_y = self.formation_positions.get("Assault Team", (0, 0))
        self.assault_team.apply_formation((base_x + assault_x, base_y + assault_y))

    def execute_movement(self, direction: Tuple[int, int], distance: int,
                         technique: MovementTechnique = MovementTechnique.TRAVELING) -> List[Dict]:
        """Execute coordinated squad movement."""
        frames = []

        if technique == MovementTechnique.TRAVELING:
            formation_type = get_formation_type(self.current_formation)
            if formation_type == FormationType.WAGON_WHEEL:
                frames = self._execute_wagon_wheel(direction, distance)
            else:
                frames = self._execute_follow_leader(direction, distance)
        else:  # BOUNDING
            frames = self._execute_bounding(direction, distance)

        return frames

    def _execute_wagon_wheel(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute wagon wheel movement for entire squad."""
        frames = []
        steps = 10
        rotation_steps = 4

        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        # Calculate rotation needed
        target_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        rotation_needed = ((target_orientation - self.orientation + 180) % 360) - 180

        # Execute rotation if needed
        if abs(rotation_needed) > 1:
            rotation_per_step = rotation_needed // rotation_steps
            for step in range(rotation_steps):
                self._rotate_squad(rotation_per_step)
                frames.append(self._capture_current_positions())

        # Execute movement
        step_distance = distance // steps
        for step in range(steps):
            self._move_squad(direction, step_distance)
            frames.append(self._capture_current_positions())

        return frames

    def _execute_follow_leader(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute follow-the-leader movement for entire squad."""
        frames = []
        path_history = [self.leader.position]

        # Calculate movement parameters
        steps = 10
        step_distance = distance // steps
        dx, dy = direction

        for step in range(steps):
            # Move squad leader
            new_x = self.leader.position[0] + (dx * step_distance) // math.sqrt(dx * dx + dy * dy)
            new_y = self.leader.position[1] + (dy * step_distance) // math.sqrt(dx * dx + dy * dy)
            self.leader.position = (new_x, new_y)
            path_history.append((new_x, new_y))

            # Move teams through previous positions
            if len(path_history) > 3:
                self.assault_team.leader.soldier.position = path_history[-3]
            if len(path_history) > 5:
                self.mg_team.leader.soldier.position = path_history[-5]
            if len(path_history) > 7:
                self.gl_team.leader.soldier.position = path_history[-7]

            # Update team formations
            self.assault_team.apply_formation(self.assault_team.leader.soldier.position)
            self.mg_team.apply_formation(self.mg_team.leader.soldier.position)
            self.gl_team.apply_formation(self.gl_team.leader.soldier.position)

            frames.append(self._capture_current_positions())

        return frames

    def _execute_bounding(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute bounding movement technique."""
        frames = []
        current_distance = 0
        bound_distance = 50  # Standard bound distance

        while current_distance < distance:
            # First bound: MG Team provides overwatch while others move
            frames.extend(self._bound_element(self.assault_team, direction, bound_distance))
            frames.extend(self._bound_element(self.gl_team, direction, bound_distance))

            # Second bound: Others provide overwatch while MG team moves
            frames.extend(self._bound_element(self.mg_team, direction, bound_distance))

            current_distance += bound_distance

        return frames

    def _bound_element(self, team: Union[Team, SpecialTeam], direction: Tuple[int, int],
                       distance: int) -> List[Dict]:
        """Execute single bound movement for a team."""
        return team.execute_movement(direction, distance, MovementTechnique.TRAVELING)

    def _rotate_squad(self, angle: int):
        """Rotate entire squad formation."""
        # Rotate squad orientation
        self.orientation = (self.orientation + angle) % 360

        # Rotate each component
        self.assault_team._rotate_formation(angle)

        # Special teams maintain relative positions during rotation
        center = self.leader.position
        angle_rad = math.radians(angle)

        for team in [self.mg_team, self.gl_team]:
            for member in team.members:
                rel_x = member.soldier.position[0] - center[0]
                rel_y = member.soldier.position[1] - center[1]

                new_rel_x = int(rel_x * math.cos(angle_rad) - rel_y * math.sin(angle_rad))
                new_rel_y = int(rel_x * math.sin(angle_rad) + rel_y * math.cos(angle_rad))

                member.soldier.position = (center[0] + new_rel_x, center[1] + new_rel_y)

    def _move_squad(self, direction: Tuple[int, int], distance: int):
        """
        Move entire squad maintaining formation.

        Args:
            direction: (dx, dy) tuple indicating movement direction
            distance: Distance to move in spaces
        """
        dx, dy = direction
        magnitude = math.sqrt(dx * dx + dy * dy)
        move_dx = (dx * distance) // magnitude
        move_dy = (dy * distance) // magnitude

        # Move squad leader
        new_sl_x = self.leader.position[0] + move_dx
        new_sl_y = self.leader.position[1] + move_dy
        self.leader.position = (int(new_sl_x), int(new_sl_y))

        # Move MG team maintaining formation
        mg_team_pos = self.formation_positions.get("MG Team", (0, 0))
        mg_x = int(new_sl_x + mg_team_pos[0])
        mg_y = int(new_sl_y + mg_team_pos[1])
        self.mg_team.apply_formation((mg_x, mg_y))

        # Move GL team maintaining formation
        gl_team_pos = self.formation_positions.get("GL Team", (0, 0))
        gl_x = int(new_sl_x + gl_team_pos[0])
        gl_y = int(new_sl_y + gl_team_pos[1])
        self.gl_team.apply_formation((gl_x, gl_y))

        # Move assault team maintaining formation
        assault_team_pos = self.formation_positions.get("Assault Team", (0, 0))
        assault_x = int(new_sl_x + assault_team_pos[0])
        assault_y = int(new_sl_y + assault_team_pos[1])
        self.assault_team.apply_formation((assault_x, assault_y))

    def _capture_current_positions(self) -> Dict:
        """Capture current positions of all squad members for animation."""
        return {
            'leader': {'position': self.leader.position},
            'mg_team': [
                {
                    'role': member.soldier.role,
                    'position': member.soldier.position
                } for member in self.mg_team.members
            ],
            'gl_team': [
                {
                    'role': member.soldier.role,
                    'position': member.soldier.position
                } for member in self.gl_team.members
            ],
            'assault_team': {
                'leader': self.assault_team.leader.soldier.position,
                'members': [
                    member.soldier.position for member in self.assault_team.members
                ]
            }
        }


# Weapon definitions
AK12 = Weapon("AK-12", 30, 210, 1, "rifle_fire", False)
PKM = Weapon("PKM", 110, 600, 6, "mg_fire", True, 6)
RPG7V2 = AreaDamageWeapon("RPG-7V2", 70, 4, 1, "rpg_fire", True, 5)


# Formation definitions
def mg_team_formation() -> Dict[str, Tuple[int, int]]:
    return {
        "Machine Gunner": (0, 0),
        "Assistant Gunner": (2, 0)
    }


def gl_team_formation() -> Dict[str, Tuple[int, int]]:
    return {
        "Grenade Launcher Gunner": (0, 0),
        "Ammo Handler": (2, 0)
    }


def assault_team_wedge() -> Dict[str, Tuple[int, int]]:
    return {
        "Team Leader": (0, 0),
        "Rifleman 1": (-2, -2),
        "Rifleman 2": (2, -2),
        "Rifleman 3": (0, -4)
    }


def squad_wedge() -> Dict[str, Tuple[int, int]]:
    return {
        "Squad Leader": (0, 0),
        "MG Team": (-4, -4),
        "GL Team": (4, -4),
        "Assault Team": (0, -6)
    }


def create_russian_squad(squad_id: str, start_position: Tuple[int, int]) -> Squad:
    """Create a Russian Assault Detachment Capture Squad."""
    x, y = start_position

    # Create Squad Leader
    sl = Soldier("Squad Leader", 100, 100, AK12, None, 48, 30, (x, y), True)

    # Create MG Team
    mg_team = SpecialTeam("MG Team")
    mg_team.add_member("Machine Gunner", PKM, None, 48, 110, (x - 4, y - 4))
    mg_team.add_member("Assistant Gunner", AK12, None, 48, 30, (x - 2, y - 4))

    # Create GL Team
    gl_team = SpecialTeam("GL Team")
    gl_team.add_member("Grenade Launcher Gunner", RPG7V2, AK12, 48, 70, (x + 4, y - 4))
    gl_team.add_member("Ammo Handler", AK12, None, 48, 30, (x + 6, y - 4))

    # Create Assault Team
    tl = Soldier("Team Leader", 100, 100, AK12, None, 48, 30, (x, y - 6), True)
    assault_team = Team("Assault Team", TeamMember(tl, "Assault"))
    assault_team.add_member("Rifleman 1", AK12, None, 48, 30, (x - 2, y - 8))
    assault_team.add_member("Rifleman 2", AK12, None, 48, 30, (x + 2, y - 8))
    assault_team.add_member("Rifleman 3", AK12, None, 48, 30, (x, y - 10))

    return Squad(squad_id, sl, mg_team, gl_team, assault_team)


def test_squad_creation():
    print("\n=== Testing Russian Squad Creation ===")
    squad = create_russian_squad("1stSquad", (100, 100))

    print("\nSquad Composition:")
    print(f"Squad Leader: {squad.leader.role}")

    print("\nMG Team:")
    for member in squad.mg_team.all_members:
        print(f"  {member.soldier.role} - {member.soldier.primary_weapon.name}")

    print("\nGL Team:")
    for member in squad.gl_team.all_members:
        print(f"  {member.soldier.role} - {member.soldier.primary_weapon.name}")

    print("\nAssault Team:")
    print(f"  {squad.assault_team.leader.soldier.role} - {squad.assault_team.leader.soldier.primary_weapon.name}")
    for member in squad.assault_team.members:
        print(f"  {member.soldier.role} - {member.soldier.primary_weapon.name}")

    print("\nTotal members:", len(squad.all_members))


def test_succession_of_command():
    print("\n=== Testing Succession of Command ===")
    squad = create_russian_squad("1stSquad", (100, 100))

    print("\nInitial leadership:")
    print(f"Squad Leader: {squad.leader.role}")
    print(f"Assault Team Leader: {squad.assault_team.leader.soldier.role}")
    print(f"MG Team Leader: {squad.mg_team.leader.soldier.role}")
    print(f"GL Team Leader: {squad.gl_team.leader.soldier.role}")

    # Test Assault Team leader succession
    print("\nTesting Assault Team leader succession:")
    old_tl = squad.assault_team.leader
    old_tl.soldier.health = 0
    squad.assault_team.check_and_replace_leader()
    print(f"New Assault Team leader: {squad.assault_team.leader.soldier.role}")

    # Test Squad Leader succession
    print("\nTesting Squad Leader succession:")
    old_sl = squad.leader
    old_sl.health = 0
    squad.check_and_replace_leader()
    print(f"New Squad Leader: {squad.leader.role}")


def test_combat_mechanics():
    print("\n=== Testing Combat Mechanics ===")
    squad = create_russian_squad("1stSquad", (100, 100))

    # Test PKM
    print("\nTesting PKM:")
    mg = next(member for member in squad.mg_team.all_members
              if member.soldier.role == "Machine Gunner")
    initial_ammo = mg.soldier.primary_weapon.ammo_count

    for _ in range(3):
        mg.soldier.fire_weapon()

    print(f"Initial ammo: {initial_ammo}")
    print(f"After 3 bursts: {mg.soldier.primary_weapon.ammo_count}")
    print(f"Rounds per burst: {mg.soldier.primary_weapon.fire_rate}")

    # Test RPG-7V2
    print("\nTesting RPG-7V2:")
    gl = next(member for member in squad.gl_team.all_members
              if member.soldier.role == "Grenade Launcher Gunner")
    initial_ammo = gl.soldier.primary_weapon.ammo_count

    gl.soldier.fire_weapon()

    print(f"Initial ammo: {initial_ammo}")
    print(f"After 1 shot: {gl.soldier.primary_weapon.ammo_count}")

    # Test AK-12
    print("\nTesting AK-12:")
    rifleman = squad.assault_team.members[0]
    initial_ammo = rifleman.soldier.primary_weapon.ammo_count

    for _ in range(5):
        rifleman.soldier.fire_weapon()

    print(f"Initial ammo: {initial_ammo}")
    print(f"After 5 shots: {rifleman.soldier.primary_weapon.ammo_count}")


def plot_formation(squad: Squad, formation_name: str):
    """Visualize squad formation."""
    plt.figure(figsize=(10, 10))

    # Plot Squad Leader
    x, y = squad.leader.position
    plt.plot(x, y, 'ro', markersize=12, label='Squad Leader')
    plt.text(x + 0.5, y + 0.5, 'SL', fontsize=10)

    # Plot MG Team
    for member in squad.mg_team.all_members:
        x, y = member.soldier.position
        plt.plot(x, y, 'bo', markersize=10)
        label = 'MG' if member.soldier.role == "Machine Gunner" else 'AG'
        plt.text(x + 0.5, y + 0.5, label, fontsize=8)

    # Plot GL Team
    for member in squad.gl_team.all_members:
        x, y = member.soldier.position
        plt.plot(x, y, 'go', markersize=10)
        label = 'GL' if member.soldier.role == "Grenade Launcher Gunner" else 'AH'
        plt.text(x + 0.5, y + 0.5, label, fontsize=8)

    # Plot Assault Team
    x, y = squad.assault_team.leader.soldier.position
    plt.plot(x, y, 'mo', markersize=10, label='Assault Team')
    plt.text(x + 0.5, y + 0.5, 'TL', fontsize=8)

    for member in squad.assault_team.members:
        x, y = member.soldier.position
        plt.plot(x, y, 'mo', markersize=8)
        plt.text(x + 0.5, y + 0.5, 'R', fontsize=8)

    plt.grid(True)
    plt.title(f"Russian Squad Formation: {formation_name}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.axis('equal')

    # Set view limits
    all_positions = [squad.leader.position]
    all_positions.extend(m.soldier.position for m in squad.mg_team.all_members)
    all_positions.extend(m.soldier.position for m in squad.gl_team.all_members)
    all_positions.extend(m.soldier.position for m in squad.assault_team.all_members)

    min_x = min(pos[0] for pos in all_positions) - 2
    max_x = max(pos[0] for pos in all_positions) + 2
    min_y = min(pos[1] for pos in all_positions) - 2
    max_y = max(pos[1] for pos in all_positions) + 2
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.show()


def test_formations():
    print("\n=== Testing Squad Formations ===")
    squad = create_russian_squad("1stSquad", (100, 100))

    # Test wedge formation
    print("\nTesting Wedge Formation:")
    squad.set_formation(
        formation_positions=squad_wedge(),
        team_formation_alpha=mg_team_formation(),
        team_formation_bravo=gl_team_formation(),
        team_formation_charlie=assault_team_wedge(),
        formation_name="squad_wedge"
    )
    squad.apply_formation((100, 100))
    plot_formation(squad, "Wedge")

    print("Squad positions after wedge formation:")
    print(f"SL: {squad.leader.position}")
    print(f"MG Team: {[m.soldier.position for m in squad.mg_team.all_members]}")
    print(f"GL Team: {[m.soldier.position for m in squad.gl_team.all_members]}")
    print(f"Assault Team: {squad.assault_team.leader.soldier.position}, "
          f"{[m.soldier.position for m in squad.assault_team.members]}")


def create_movement_animation(frames, title, save_path, view_bounds):
    """Create animation of squad movement."""
    fig, ax = plt.subplots(figsize=(15, 15))

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()
        ax.grid(True)

        # Plot squad leader
        sl_pos = frame['leader']['position']
        ax.plot(sl_pos[0], sl_pos[1], 'ro', markersize=12, label='Squad Leader')
        ax.text(sl_pos[0], sl_pos[1], 'SL', ha='center', va='center')

        # Plot MG Team
        for member in frame['mg_team']:
            pos = member['position']
            ax.plot(pos[0], pos[1], 'bo', markersize=10)
            label = 'MG' if member['role'] == "Machine Gunner" else 'AG'
            ax.text(pos[0], pos[1], label, ha='center', va='center')

        # Plot GL Team
        for member in frame['gl_team']:
            pos = member['position']
            ax.plot(pos[0], pos[1], 'go', markersize=10)
            label = 'GL' if member['role'] == "Grenade Launcher Gunner" else 'AH'
            ax.text(pos[0], pos[1], label, ha='center', va='center')

        # Plot Assault Team
        assault_leader = frame['assault_team']['leader']
        ax.plot(assault_leader[0], assault_leader[1], 'mo', markersize=10, label='Assault Team')
        ax.text(assault_leader[0], assault_leader[1], 'TL', ha='center', va='center')

        for member in frame['assault_team']['members']:
            ax.plot(member[0], member[1], 'mo', markersize=8)
            ax.text(member[0], member[1], 'R', ha='center', va='center')

        ax.set_xlim(view_bounds[0], view_bounds[1])
        ax.set_ylim(view_bounds[2], view_bounds[3])
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nFrame {frames.index(frame) + 1}/{len(frames)}')

        return []

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                   interval=200, blit=True)
    anim.save(save_path, writer='pillow')
    plt.close()


def test_movements():
    print("\n=== Testing Squad Movements ===")
    squad = create_russian_squad("1stSquad", (100, 100))

    # Test wedge formation movement
    print("\nTesting Wedge Formation Movement:")
    squad.set_formation(
        formation_positions=squad_wedge(),
        team_formation_alpha=mg_team_formation(),
        team_formation_bravo=gl_team_formation(),
        team_formation_charlie=assault_team_wedge(),
        formation_name="squad_wedge"
    )
    squad.apply_formation((100, 100))

    frames = squad.execute_movement(
        direction=(1, 1),
        distance=50,
        technique=MovementTechnique.TRAVELING
    )

    create_movement_animation(
        frames=frames,
        title="Russian Squad Wedge Movement",
        save_path="russian_squad_movement.gif",
        view_bounds=(90, 160, 90, 160)
    )

    print("Movement animation saved as 'russian_squad_movement.gif'")


if __name__ == "__main__":
    # Run all tests
    try:
        print("Starting Russian Squad Composition Tests...")

        test_squad_creation()
        test_succession_of_command()
        test_combat_mechanics()
        test_formations()
        test_movements()

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback

        traceback.print_exc()
