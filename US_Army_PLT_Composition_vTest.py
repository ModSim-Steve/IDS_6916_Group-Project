"""
US Army Infantry Platoon Composition Interface

This file defines the interface between US Army Infantry platoon concepts and the
WarGamingEnvironment state management system. It provides functions for creating
and managing US Army Infantry units within the environment.

Key Components:
- Unit creation and hierarchy establishment
- Formation definitions and management
- Equipment and weapon definitions
- Movement pattern implementations
- Standard operating procedures

Note: This file handles ONLY US Army Infantry Platoon composition.
Other unit types (Armor, Artillery, etc.) should be in separate files.
Enemy forces should be in their own composition files.

State Tensor Channel Mapping:
Channel 0: Terrain type
Channel 1: Elevation
Channel 2: Unit ID
Channel 3: Status flags

Unit Property Mapping:
0: Unit type
1: Formation index
2: Base position X
3: Base position Y
4: Orientation
5: Parent unit ID
6: Unit status

Soldier Property Mapping:
0: Role ID
1: Health
2: Primary weapon ammo
3: Secondary weapon ammo
4: Observation range
5: Engagement range
6: Orientation

Unit ID Structure (PSTP format):
P: Platoon number (1-9)
S: Squad number (0-9, 0 = PLT HQ)
T: Team number (0-9):
    0 = Squad/PLT HQ
    1 = Alpha Team
    2 = Bravo Team
    3-4 = Gun Teams
    5-6 = Javelin Teams
P: Position number (0-7):
    0 = Leader
    1 = Auto Rifleman
    2 = Grenadier
    3 = Rifleman
    4 = Machine Gunner
    5 = Assistant Gunner
    6 = Anti-Tank
    7 = Anti-Tank Helper
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any
from enum import Enum, auto
from datetime import datetime
import math

from WarGamingEnvironment_vTest import (
    MilitaryEnvironment,
    UnitType,
    BaseWeapon,
    SectorOfFire,
    CoordinationPoint,
    CoordinationLevel
)


class US_IN_Role(Enum):
    """US Army Infantry roles with their numeric values for soldier properties"""
    PLATOON_LEADER = 0
    SQUAD_LEADER = 1
    TEAM_LEADER = 2
    AUTO_RIFLEMAN = 3
    GRENADIER = 4
    RIFLEMAN = 5
    MACHINE_GUNNER = 6
    ASSISTANT_GUNNER = 7
    ANTI_TANK = 8
    ANTI_TANK_HELPER = 9


class US_IN_TeamType(Enum):
    """Types of teams that can exist"""
    FIRE = "FIRE"
    GUN = "GUN"
    JAVELIN = "JAVELIN"


class US_IN_UnitDesignator(Enum):
    """Standard designators for units"""
    # Teams
    ALPHA_TEAM = "ATM"
    BRAVO_TEAM = "BTM"
    GUN_TEAM_1 = "GTM1"
    GUN_TEAM_2 = "GTM2"
    JAVELIN_TEAM_1 = "JTM1"
    JAVELIN_TEAM_2 = "JTM2"

    # Squads
    SQUAD_1 = "1SQD"
    SQUAD_2 = "2SQD"
    SQUAD_3 = "3SQD"

    @classmethod
    def get_formation_key(cls, designator: 'US_IN_UnitDesignator') -> str:
        """Convert designator to formation template key"""
        formation_keys = {
            cls.ALPHA_TEAM: "Alpha Team",
            cls.BRAVO_TEAM: "Bravo Team",
            cls.GUN_TEAM_1: "Gun Team 1",
            cls.GUN_TEAM_2: "Gun Team 2",
            cls.JAVELIN_TEAM_1: "Javelin Team 1",
            cls.JAVELIN_TEAM_2: "Javelin Team 2",
            cls.SQUAD_1: "1st Squad",
            cls.SQUAD_2: "2nd Squad",
            cls.SQUAD_3: "3rd Squad"
        }
        return formation_keys[designator]


class MovementTechnique(Enum):
    """Types of movement techniques that can be used."""
    TRAVELING = "traveling"  # Units follow each other in sequence
    BOUNDING = "bounding"  # Units move one at a time, providing overwatch


class FormationType(Enum):
    """Types of formations based on movement characteristics."""
    WAGON_WHEEL = "wagon_wheel"  # Formations that rotate around pivot point
    FOLLOW_LEADER = "follow_leader"  # Column formations that follow path of leader


@dataclass
class RouteWaypoint:
    """Represents a waypoint in a movement route."""
    position: Tuple[int, int]  # (x, y) coordinates
    formation: Optional[str] = None  # Formation to use at this waypoint
    hold_time: int = 0  # Number of steps to hold at waypoint
    required_actions: List[str] = None  # Actions to perform at waypoint


@dataclass
class MovementRoute:
    """Represents a complete movement route with waypoints."""
    waypoints: List[RouteWaypoint]
    current_waypoint: int = 0
    technique: MovementTechnique = MovementTechnique.TRAVELING
    completed: bool = False


def get_formation_type(formation_name: str) -> FormationType:
    """Get the movement type for a given formation."""
    wagon_wheel_formations = [
        "team_wedge_right", "team_wedge_left", "team_line_right", "team_line_left", "gun_team", "javelin_team",
        "squad_line_team_wedge", "platoon_line_squad_line"
    ]
    return (FormationType.WAGON_WHEEL
            if formation_name in wagon_wheel_formations
            else FormationType.FOLLOW_LEADER)


# Standard US Army Infantry weapon definitions
US_IN_M4 = BaseWeapon("M4 Carbine", 50, 210, 1, 40)
US_IN_M249 = BaseWeapon("M249 SAW", 80, 600, 6, 45, True)
US_IN_M320 = BaseWeapon("M320 GL", 35, 12, 1, 80, True)
US_IN_M240B = BaseWeapon("M240B", 100, 1000, 9, 50, True)
US_IN_JAVELIN = BaseWeapon("Javelin", 200, 3, 1, 1000, True)


def US_IN_generate_soldier_id(plt_num: int, squad_num: int, team_num: int, pos_num: int) -> int:
    """
    Generate unique soldier ID in PSTP format.

    Args:
        plt_num: Platoon number (1-9)
        squad_num: Squad number (0-9, 0 = PLT HQ)
        team_num: Team number (0-9):
            0 = Squad/PLT HQ
            1 = Alpha Team
            2 = Bravo Team
            3-4 = Gun Teams
            5-6 = Javelin Teams
        pos_num: Position number (0-7):
            0 = Leader
            1 = Auto Rifleman
            2 = Grenadier
            3 = Rifleman
            4 = Machine Gunner
            5 = Assistant Gunner
            6 = Anti-Tank
            7 = Anti-Tank Helper

    Returns:
        Integer ID in format: PSTP
    """
    return (plt_num * 1000 + squad_num * 100 + team_num * 10 + pos_num)


def US_IN_create_unit_id(plt_num: int, squad_num: int = None, team_designator: US_IN_UnitDesignator = None) -> str:
    """Create standardized unit ID string"""
    if team_designator:
        return f"{plt_num}PLT-{squad_num}SQD-{team_designator.value}"
    elif squad_num:
        return f"{plt_num}PLT-{squad_num}SQD"
    else:
        return f"{plt_num}PLT"


def US_IN_create_team(env: MilitaryEnvironment, plt_num: int, squad_num: int,
                      designator: US_IN_UnitDesignator, start_position: Tuple[int, int]) -> int:
    """Create a US Army Infantry team."""
    # Create team unit
    team_id_str = US_IN_create_unit_id(plt_num, squad_num, designator)
    team_id = env.create_unit(
        unit_type=UnitType.INFANTRY_TEAM,
        unit_id_str=team_id_str,
        start_position=start_position
    )

    # Determine team type and roles based on designator
    if designator in [US_IN_UnitDesignator.ALPHA_TEAM, US_IN_UnitDesignator.BRAVO_TEAM]:
        member_roles = [
            (US_IN_Role.TEAM_LEADER, 'TL'),
            (US_IN_Role.AUTO_RIFLEMAN, 'AR'),
            (US_IN_Role.GRENADIER, 'GRN'),
            (US_IN_Role.RIFLEMAN, 'RFLM')
        ]
        team_type = US_IN_TeamType.FIRE
    elif 'GTM' in designator.value:
        member_roles = [
            (US_IN_Role.MACHINE_GUNNER, 'MG'),
            (US_IN_Role.ASSISTANT_GUNNER, 'AG')
        ]
        team_type = US_IN_TeamType.GUN
    elif 'JTM' in designator.value:
        member_roles = [
            (US_IN_Role.ANTI_TANK, 'AT'),
            (US_IN_Role.ANTI_TANK_HELPER, 'AH')
        ]
        team_type = US_IN_TeamType.JAVELIN
    else:
        raise ValueError(f"Invalid team designator: {designator}")

    # Store team type
    env.update_unit_property(team_id, 'team_type', team_type.value)

    # Calculate offset positions for members
    member_positions = []
    for i in range(len(member_roles)):
        if i == 0:  # Leader stays at base position
            member_positions.append(start_position)
        else:  # Other members get offset positions
            offset_x = 2 * (i - 1)
            offset_y = -2
            member_positions.append((start_position[0] + offset_x, start_position[1] + offset_y))

    # Create members with offset positions
    for idx, ((role, suffix), position) in enumerate(zip(member_roles, member_positions)):
        is_leader = idx == 0  # First role is always the leader

        # Determine weapons based on role
        primary_weapon = None
        secondary_weapon = None

        if role == US_IN_Role.TEAM_LEADER:
            primary_weapon = US_IN_M4
        elif role == US_IN_Role.AUTO_RIFLEMAN:
            primary_weapon = US_IN_M249
        elif role == US_IN_Role.GRENADIER:
            primary_weapon = US_IN_M4
            secondary_weapon = US_IN_M320
        elif role == US_IN_Role.RIFLEMAN:
            primary_weapon = US_IN_M4
        elif role == US_IN_Role.MACHINE_GUNNER:
            primary_weapon = US_IN_M240B
        elif role == US_IN_Role.ANTI_TANK:
            primary_weapon = US_IN_M4
            secondary_weapon = US_IN_JAVELIN
        else:
            primary_weapon = US_IN_M4

        member_id = env.create_soldier(
            role=role,
            unit_id_str=f"{team_id_str}-{suffix}",
            position=position,
            is_leader=is_leader
        )

        # Set weapons and ranges
        if primary_weapon:
            env.update_unit_property(member_id, 'primary_weapon', primary_weapon)
            env.update_unit_property(member_id, 'engagement_range', primary_weapon.max_range)
        if secondary_weapon:
            env.update_unit_property(member_id, 'secondary_weapon', secondary_weapon)

        # Set observation range based on team type
        if team_type == US_IN_TeamType.FIRE:
            obs_range = 50
        else:  # Gun and Javelin teams have better observation
            obs_range = 70
        env.update_unit_property(member_id, 'observation_range', obs_range)

        # Set hierarchy
        env.set_unit_hierarchy(member_id, team_id)

    # Set initial formation based on team type
    if team_type == US_IN_TeamType.FIRE and designator == US_IN_UnitDesignator.ALPHA_TEAM:
        US_IN_apply_formation(env, team_id, "team_wedge_left")
    elif team_type == US_IN_TeamType.FIRE and designator == US_IN_UnitDesignator.BRAVO_TEAM:
        US_IN_apply_formation(env, team_id, "team_wedge_right")
    elif team_type == US_IN_TeamType.GUN:
        US_IN_apply_formation(env, team_id, "gun_team")
    else:  # JAVELIN
        US_IN_apply_formation(env, team_id, "javelin_team")

    return team_id


def US_IN_create_squad(env: MilitaryEnvironment, plt_num: int, squad_num: int,
                       start_position: Tuple[int, int]) -> int:
    """Create a US Army Infantry squad."""
    # Create squad unit
    squad_id_str = US_IN_create_unit_id(plt_num, squad_num)
    squad_id = env.create_unit(
        unit_type=UnitType.INFANTRY_SQUAD,
        unit_id_str=squad_id_str,
        start_position=start_position
    )

    # Create Squad Leader
    sl_id = env.create_soldier(
        role=US_IN_Role.SQUAD_LEADER,
        unit_id_str=f"{squad_id_str}-SL",
        position=start_position,
        is_leader=True
    )

    # Set SL weapons and properties
    env.update_unit_property(sl_id, 'primary_weapon', US_IN_M4)
    env.update_unit_property(sl_id, 'engagement_range', US_IN_M4.max_range)
    env.update_unit_property(sl_id, 'observation_range', 60)  # Enhanced observation for SL
    env.set_unit_hierarchy(sl_id, squad_id)

    # Calculate team positions based on squad formation
    alpha_pos = (start_position[0], start_position[1] + 8)  # Alpha forward
    bravo_pos = (start_position[0], start_position[1] - 3)  # Bravo back

    # Create teams
    alpha_id = US_IN_create_team(
        env=env,
        plt_num=plt_num,
        squad_num=squad_num,
        designator=US_IN_UnitDesignator.ALPHA_TEAM,
        start_position=alpha_pos
    )
    env.set_unit_hierarchy(alpha_id, squad_id)

    bravo_id = US_IN_create_team(
        env=env,
        plt_num=plt_num,
        squad_num=squad_num,
        designator=US_IN_UnitDesignator.BRAVO_TEAM,
        start_position=bravo_pos
    )
    env.set_unit_hierarchy(bravo_id, squad_id)

    # Set initial formation
    US_IN_apply_formation(env, squad_id, "squad_column_team_wedge")

    if env.debug_level > 0:
        print(f"Created squad {squad_id_str} with SL and 2 teams")
        child_units = env.get_unit_children(squad_id)
        print(f"Squad has {len(child_units)} direct children")

    return squad_id


def US_IN_create_platoon(env: MilitaryEnvironment, plt_num: int,
                         start_position: Tuple[int, int]) -> int:
    """Create a complete US Army Infantry platoon."""
    # Create platoon unit
    plt_id_str = US_IN_create_unit_id(plt_num)
    plt_id = env.create_unit(
        unit_type=UnitType.INFANTRY_PLATOON,
        unit_id_str=plt_id_str,
        start_position=start_position
    )

    # Create platoon leader
    pl_id = env.create_soldier(
        role=US_IN_Role.PLATOON_LEADER,
        unit_id_str=f"{plt_id_str}-PL",
        position=start_position,
        is_leader=True
    )

    # Set PL weapons and properties
    env.update_unit_property(pl_id, 'primary_weapon', US_IN_M4)
    env.update_unit_property(pl_id, 'engagement_range', US_IN_M4.max_range)
    env.update_unit_property(pl_id, 'observation_range', 70)  # Enhanced observation for PL
    env.set_unit_hierarchy(pl_id, plt_id)

    # Calculate squad positions
    squad_positions = [
        (start_position[0], start_position[1] + 11),  # 1st squad forward
        (start_position[0], start_position[1] - 15),  # 2nd squad middle
        (start_position[0], start_position[1] - 36)  # 3rd squad rear
    ]

    # Create rifle squads
    for idx, squad_pos in enumerate(squad_positions, 1):
        squad_id = US_IN_create_squad(
            env=env,
            plt_num=plt_num,
            squad_num=idx,
            start_position=squad_pos
        )
        env.set_unit_hierarchy(squad_id, plt_id)

    # Calculate weapons team positions
    wpns_base = (start_position[0], start_position[1] - 2)  # Weapons base position
    gun_positions = [
        (wpns_base[0] + 2, wpns_base[1]),  # Gun team 1 right
        (wpns_base[0] - 4, wpns_base[1] - 23)  # Gun team 2 left
    ]
    javelin_positions = [
        (wpns_base[0] - 4, wpns_base[1]),  # Javelin team 1 left
        (wpns_base[0] + 2, wpns_base[1] - 23)  # Javelin team 2 right
    ]

    # Create gun teams
    for idx, pos in enumerate(gun_positions, 1):
        team_id = US_IN_create_team(
            env=env,
            plt_num=plt_num,
            squad_num=0,  # 0 indicates platoon HQ
            designator=getattr(US_IN_UnitDesignator, f'GUN_TEAM_{idx}'),
            start_position=pos
        )
        env.set_unit_hierarchy(team_id, plt_id)

    # Create javelin teams
    for idx, pos in enumerate(javelin_positions, 1):
        team_id = US_IN_create_team(
            env=env,
            plt_num=plt_num,
            squad_num=0,  # 0 indicates platoon HQ
            designator=getattr(US_IN_UnitDesignator, f'JAVELIN_TEAM_{idx}'),
            start_position=pos
        )
        env.set_unit_hierarchy(team_id, plt_id)

    # Set initial formation
    US_IN_apply_formation(env, plt_id, "platoon_column")

    if env.debug_level > 0:
        print(f"Created platoon {plt_id_str}")
        child_units = env.get_unit_children(plt_id)
        print(f"Platoon has {len(child_units)} direct children")
        squads = [u for u in child_units if env.get_unit_property(u, 'type') == UnitType.INFANTRY_SQUAD]
        gun_teams = [u for u in child_units if 'GTM' in env.get_unit_property(u, 'string_id')]
        javelin_teams = [u for u in child_units if 'JTM' in env.get_unit_property(u, 'string_id')]
        print(f"Composition: {len(squads)} squads, {len(gun_teams)} gun teams, {len(javelin_teams)} javelin teams")

    return plt_id


# Formation definitions for US Army Infantry units
"""
US Army Infantry Formation Definitions and Application.

This module defines standard formations for US Army Infantry units.
All individual soldier roles use the enum format (TEAM_LEADER, AUTO_RIFLEMAN, etc.).
Team and Squad identifiers use readable strings for clarity.
"""


def US_IN_team_wedge_right() -> Dict[US_IN_Role, Tuple[int, int]]:
    """
    Standard right-oriented team wedge formation.

    Configuration:
             TL
        AR        GRN
                        RFLM
    """
    return {
        US_IN_Role.TEAM_LEADER: (0, 0),  # At base position
        US_IN_Role.AUTO_RIFLEMAN: (-2, -2),  # Back left
        US_IN_Role.GRENADIER: (2, -2),  # Back right
        US_IN_Role.RIFLEMAN: (4, -4)  # Far back right
    }


def US_IN_team_wedge_left() -> Dict[US_IN_Role, Tuple[int, int]]:
    """
    Standard left-oriented team wedge formation.

    Configuration:
                 TL
           GRN        AR
    RFLM
    """
    return {
        US_IN_Role.TEAM_LEADER: (0, 0),  # At base position
        US_IN_Role.AUTO_RIFLEMAN: (2, -2),  # Back right
        US_IN_Role.GRENADIER: (-2, -2),  # Back left
        US_IN_Role.RIFLEMAN: (-4, -4)  # Far back left
    }


def US_IN_team_line_right() -> Dict[US_IN_Role, Tuple[int, int]]:
    """
    Right-oriented team line formation.

    Configuration:
    TL  AR  GRN  RFLM
    """
    return {
        US_IN_Role.TEAM_LEADER: (0, 0),  # At base position
        US_IN_Role.AUTO_RIFLEMAN: (-2, 0),  # Left
        US_IN_Role.GRENADIER: (2, 0),  # Right
        US_IN_Role.RIFLEMAN: (4, 0)  # Far right
    }


def US_IN_team_line_left() -> Dict[US_IN_Role, Tuple[int, int]]:
    """
    Left-oriented team line formation.

    Configuration:
    RFLM  GRN  AR  TL
    """
    return {
        US_IN_Role.TEAM_LEADER: (0, 0),  # At base position
        US_IN_Role.AUTO_RIFLEMAN: (2, 0),  # Right
        US_IN_Role.GRENADIER: (-2, 0),  # Left
        US_IN_Role.RIFLEMAN: (-4, 0)  # Far left
    }


def US_IN_team_column() -> Dict[US_IN_Role, Tuple[int, int]]:
    """
    Team column formation.

    Configuration:
    TL
    AR
    GRN
    RFLM
    """
    return {
        US_IN_Role.TEAM_LEADER: (0, 0),  # Front
        US_IN_Role.AUTO_RIFLEMAN: (0, -2),  # Second
        US_IN_Role.GRENADIER: (0, -4),  # Third
        US_IN_Role.RIFLEMAN: (0, -6)  # Last
    }


def US_IN_gun_team() -> Dict[US_IN_Role, Tuple[int, int]]:
    """
    Standard machine gun team formation.

    Configuration:
    MG  AG
    """
    return {
        US_IN_Role.MACHINE_GUNNER: (0, 0),  # At base position
        US_IN_Role.ASSISTANT_GUNNER: (2, 0)  # Right of gunner
    }


def US_IN_javelin_team() -> Dict[US_IN_Role, Tuple[int, int]]:
    """
    Standard Javelin anti-tank team formation.

    Configuration:
    AT  AH
    """
    return {
        US_IN_Role.ANTI_TANK: (0, 0),  # At base position
        US_IN_Role.ANTI_TANK_HELPER: (2, 0)  # Right of gunner
    }


def US_IN_squad_column_team_wedge() -> Dict[str, Tuple[int, int]]:
    """
    Squad column formation with teams in wedge.

    Configuration:
          ATM (wedge)

          SL

          BTM (wedge)
    """
    return {
        'ATM': (0, 8),
        US_IN_Role.SQUAD_LEADER: (0, 0),
        US_IN_UnitDesignator.BRAVO_TEAM.value: (0, -3)
    }


def US_IN_squad_column_team_column() -> Dict[str, Tuple[int, int]]:
    """
    Squad column formation with teams in column.

    Configuration:
          ATM (column)

          SL

          BTM (column)
    """
    return {
        US_IN_Role.SQUAD_LEADER: (0, 0),  # Center
        'ATM': (0, 8),  # Front
        US_IN_UnitDesignator.BRAVO_TEAM.value: (0, -3)  # Back
    }


def US_IN_squad_line_team_wedge() -> Dict[str, Tuple[int, int]]:
    """
    Squad line formation with teams in wedge.

    Configuration:
    ATM (wedge)  SL  BTM (wedge)
    """
    return {
        US_IN_Role.SQUAD_LEADER: (0, 0),  # Center
        'ATM': (-4, 4),  # Left
        US_IN_UnitDesignator.BRAVO_TEAM.value: (4, 4)  # Right
    }


def US_IN_platoon_column() -> Dict[str, Tuple[int, int]]:
    """
    Platoon column formation.

    Configuration:
    1st Squad

    PL
    GTM1  JTM1

    2nd Squad

    GTM2  JTM2

    3rd Squad
    """
    return {
        US_IN_Role.PLATOON_LEADER: (0, 0),  # Center
        US_IN_UnitDesignator.SQUAD_1.value: (0, 11),  # Front
        US_IN_UnitDesignator.GUN_TEAM_1.value: (2, -2),  # Back right
        US_IN_UnitDesignator.JAVELIN_TEAM_1.value: (-4, -2),  # Back left
        US_IN_UnitDesignator.SQUAD_2.value: (0, -15),  # Further back
        US_IN_UnitDesignator.GUN_TEAM_2.value: (-4, -25),  # Far back left
        US_IN_UnitDesignator.JAVELIN_TEAM_2.value: (2, -25),  # Far back right
        US_IN_UnitDesignator.SQUAD_3.value: (0, -36)  # Last
    }


def US_IN_platoon_line() -> Dict[str, Tuple[int, int]]:
    """
    Platoon line formation with squads in line.

    Configuration:
    1SQD  2SQD  3SQD
    GTM1  PL   GTM2
    JTM1       JTM2
    """
    return {
        US_IN_Role.PLATOON_LEADER: (0, 0),  # Center
        US_IN_UnitDesignator.SQUAD_1.value: (-19, 3),  # Left
        US_IN_UnitDesignator.SQUAD_2.value: (0, 3),  # Center
        US_IN_UnitDesignator.SQUAD_3.value: (19, 3),  # Right
        US_IN_UnitDesignator.GUN_TEAM_1.value: (-15, -1),  # Left rear
        US_IN_UnitDesignator.GUN_TEAM_2.value: (13, -1),  # Right rear
        US_IN_UnitDesignator.JAVELIN_TEAM_1.value: (-9, -1),  # Left center rear
        US_IN_UnitDesignator.JAVELIN_TEAM_2.value: (7, -1)  # Right center rear
    }


def US_IN_platoon_wedge() -> Dict[str, Tuple[int, int]]:
    """
    Platoon wedge formation with squads in column.

    Configuration:
         1SQD

    2SQD  PL  3SQD
    GTM1      GTM2
    JTM1      JTM2
    """
    return {
        US_IN_Role.PLATOON_LEADER: (0, 0),  # Center
        US_IN_UnitDesignator.SQUAD_1.value: (0, 8),  # Front
        US_IN_UnitDesignator.SQUAD_2.value: (-8, -4),  # Left rear
        US_IN_UnitDesignator.SQUAD_3.value: (8, -4),  # Right rear
        US_IN_UnitDesignator.GUN_TEAM_1.value: (-4, -2),  # Left center
        US_IN_UnitDesignator.GUN_TEAM_2.value: (4, -2),  # Right center
        US_IN_UnitDesignator.JAVELIN_TEAM_1.value: (-2, -6),  # Left rear
        US_IN_UnitDesignator.JAVELIN_TEAM_2.value: (2, -6)  # Right rear
    }


def US_IN_apply_formation(env: MilitaryEnvironment, unit_id: int, formation_type: str) -> None:
    """
    Apply US Army Infantry formation to unit.
    Uses standard coordinate system where 0° = East.

    Coordinate System Standard:
    - 0° = East (right)
    - 90° = South (down)
    - 180° = West (left)
    - 270° = North (up)

    This matches Pygame's coordinate system where (0,0) is top-left.

    Args:
        env: Reference to environment
        unit_id: ID of unit to form up
        formation_type: Name of formation to apply
    """
    # print(f"\n[DEBUG APPLY FORMATION] US_IN_apply_formation called for unit {unit_id} with formation {formation_type}")
    # print(f"[DEBUG APPLY FORMATION] Current orientation: {env.get_unit_property(unit_id, 'orientation')}")

    # Map formation names to functions
    formation_funcs = {
        "team_wedge_left": US_IN_team_wedge_left,
        "team_wedge_right": US_IN_team_wedge_right,
        "team_line_left": US_IN_team_line_left,
        "team_line_right": US_IN_team_line_right,
        "team_column": US_IN_team_column,
        "gun_team": US_IN_gun_team,
        "javelin_team": US_IN_javelin_team,
        "squad_column_team_wedge": US_IN_squad_column_team_wedge,
        "squad_column_team_column": US_IN_squad_column_team_column,
        "squad_line_team_wedge": US_IN_squad_line_team_wedge,
        "platoon_column": US_IN_platoon_column,
        "platoon_line": US_IN_platoon_line,
        "platoon_wedge": US_IN_platoon_wedge
    }

    if env.debug_level > 1:
        print(f"\nApplying formation {formation_type} to unit {unit_id}")
        unit_type = env.get_unit_property(unit_id, 'type')
        print(f"Unit type: {unit_type}")

    # Validate formation exists
    if formation_type not in formation_funcs:
        raise ValueError(f"Unknown formation type: {formation_type}")

    # Get formation template
    formation_func = formation_funcs[formation_type]
    template = formation_func()

    if env.debug_level > 1:
        print(f"Original template: {template}")

    # Convert template to format environment expects
    converted_template = {}
    for key, value in template.items():
        if isinstance(key, US_IN_Role):
            # Keep role enums as is
            converted_template[key] = value
        elif isinstance(key, str) and any(key == designator.value for designator in US_IN_UnitDesignator):
            # Handle any unit designator value (ATM, BTM, 1SQD, GTM1, etc.)
            converted_template[key] = value
        else:
            # Keep any other keys as-is
            converted_template[key] = value

    if env.debug_level > 1:
        print(f"Converted template: {converted_template}")
        print(f"Unit type: {env.get_unit_property(unit_id, 'type')}")
        for child in env.get_unit_children(unit_id):
            print(f"  Child unit: {env.get_unit_property(child, 'string_id')}")
            print(f"    Current position: {env.get_unit_position(child)}")

    # Store formation properties
    env.state_manager.update_unit_property(unit_id, 'formation_type', formation_type)
    env.state_manager.update_unit_property(unit_id, 'formation_template', template)
    env.state_manager.update_unit_property(unit_id, 'converted_template', converted_template)

    # Apply formation through environment
    env.apply_formation(unit_id, formation_type, converted_template)

    # Now store the rotated sectors for each member
    orientation = env.get_unit_property(unit_id, 'orientation', 0)
    for member_id in env.get_unit_children(unit_id):
        role = env.get_unit_property(member_id, 'role')
        is_leader = env.get_unit_property(member_id, 'is_leader', False)

        # Get base sectors
        if isinstance(role, int):
            role = US_IN_Role(role)
        primary_sector, secondary_sector = get_unit_sectors(role, formation_type, is_leader)

        # Add detailed debug prints
        # print(f"[DEBUG APPLY FORMATION] Calculating sectors for member {member_id}, role: {role}")
        # print(f"[DEBUG APPLY FORMATION] Unit orientation: {orientation}°")

        # Store both base and rotated sector angles in state manager
        if primary_sector:
            # print(f"[DEBUG APPLY FORMATION] Primary sector: base {primary_sector.start_angle}°-{primary_sector.end_angle}°")
            # print(f"[DEBUG APPLY FORMATION] Primary sector: rotated {(primary_sector.start_angle + orientation) % 360}°-{(primary_sector.end_angle + orientation) % 360}°")

            env.update_unit_property(member_id, 'primary_sector_base_start', primary_sector.start_angle)
            env.update_unit_property(member_id, 'primary_sector_base_end', primary_sector.end_angle)
            env.update_unit_property(member_id, 'primary_sector_rotated_start',
                                     (primary_sector.start_angle + orientation) % 360)
            env.update_unit_property(member_id, 'primary_sector_rotated_end',
                                     (primary_sector.end_angle + orientation) % 360)

            if env.debug_level > 1:
                print(f"  Member {role.name}: Primary sector rotated from "
                      f"{primary_sector.start_angle}°-{primary_sector.end_angle}° to "
                      f"{(primary_sector.start_angle + orientation) % 360}°-{(primary_sector.end_angle + orientation) % 360}°")

        if secondary_sector:
            env.update_unit_property(member_id, 'secondary_sector_base_start', secondary_sector.start_angle)
            env.update_unit_property(member_id, 'secondary_sector_base_end', secondary_sector.end_angle)
            env.update_unit_property(member_id, 'secondary_sector_rotated_start',
                                     (secondary_sector.start_angle + orientation) % 360)
            env.update_unit_property(member_id, 'secondary_sector_rotated_end',
                                     (secondary_sector.end_angle + orientation) % 360)

            if env.debug_level > 1:
                print(f"  Member {role.name}: Secondary sector rotated from "
                      f"{secondary_sector.start_angle}°-{secondary_sector.end_angle}° to "
                      f"{(secondary_sector.start_angle + orientation) % 360}°-{(secondary_sector.end_angle + orientation) % 360}°")

    # Handle subordinate unit formations
    unit_type = env.get_unit_property(unit_id, 'type')
    if unit_type in [UnitType.INFANTRY_PLATOON, UnitType.INFANTRY_SQUAD]:
        if env.debug_level > 1:
            print(f"\nHandling subordinate formations for {unit_type}")

        # Get child units
        child_units = env.get_unit_children(unit_id)

        # For platoons, handle both squads and weapons teams
        if unit_type == UnitType.INFANTRY_PLATOON:
            for child_id in child_units:
                child_type = env.get_unit_property(child_id, 'type')
                child_formation = env.get_unit_property(child_id, 'formation_type', None)
                if child_formation:
                    if env.debug_level > 1:
                        print(f"Reapplying {child_formation} to {child_type}")
                    US_IN_apply_formation(env, child_id, child_formation)

        # For squads, handle teams
        elif unit_type == UnitType.INFANTRY_SQUAD:
            for child_id in child_units:
                if env.get_unit_property(child_id, 'type') == UnitType.INFANTRY_TEAM:
                    child_formation = env.get_unit_property(child_id, 'formation_type', None)
                    if child_formation:
                        if env.debug_level > 1:
                            print(f"Reapplying {child_formation} to team")
                        US_IN_apply_formation(env, child_id, child_formation)


def US_IN_validate_formation(formation_type: str, unit_type: UnitType) -> bool:
    """
    Validate formation type is appropriate for unit type.
    """
    valid_formations = {
        UnitType.INFANTRY_TEAM: {
            "team_wedge_left", "team_wedge_right",
            "team_line_left", "team_line_right",
            "team_column"
        },
        UnitType.WEAPONS_TEAM: {
            "gun_team", "javelin_team"
        },
        UnitType.INFANTRY_SQUAD: {
            "squad_column_team_wedge",
            "squad_column_team_column",
            "squad_line_team_wedge"
        },
        UnitType.INFANTRY_PLATOON: {
            "platoon_column",
            "platoon_line",
            "platoon_wedge"
        }
    }
    return formation_type in valid_formations.get(unit_type, set())


# Sectors of Fire
"""
Defines sectors of fire for infantry unit formations.

Sectors are defined for each position within various formations:
- Individual sectors of fire (primary and secondary)
- Integration with formations and orientations
- Used in engagement validation and target selection
"""


def get_unit_sectors(role: 'US_IN_Role', formation: str, is_leader: bool) -> Tuple[SectorOfFire, SectorOfFire]:
    """Get US Army unit sectors based on role and formation."""
    # Handle squad leaders separately
    if is_leader and role == US_IN_Role.SQUAD_LEADER:
        return _get_squad_leader_sectors(formation)

    # Handle weapons teams
    if role in [US_IN_Role.MACHINE_GUNNER, US_IN_Role.ASSISTANT_GUNNER,
                US_IN_Role.ANTI_TANK, US_IN_Role.ANTI_TANK_HELPER]:
        return _get_weapons_team_sectors(role)

    # Handle fire team formations
    if 'column' in formation.lower():
        return _get_team_column_sectors(role)
    elif 'wedge_right' in formation.lower() or 'line_right' in formation.lower():
        return _get_team_wedge_right_sectors(role)
    elif 'wedge_left' in formation.lower() or 'line_left' in formation.lower():
        return _get_team_wedge_left_sectors(role)

    # Default to forward sectors if formation not recognized
    return (
        SectorOfFire(315, 45, True),  # Primary: front
        SectorOfFire(165, 205, False)  # Secondary: rear
    )


def _get_squad_leader_sectors(formation: str) -> Tuple[SectorOfFire, SectorOfFire]:
    """Get US Army squad leader sectors."""
    if 'column' in formation.lower():
        return (
            SectorOfFire(225, 315, True),  # Left
            SectorOfFire(45, 135, False)  # Right
        )
    else:  # line formation
        return (
            SectorOfFire(335, 25, True),  # Front
            SectorOfFire(165, 205, False)  # Rear
        )


def _get_weapons_team_sectors(role: 'US_IN_Role') -> Tuple[SectorOfFire, SectorOfFire]:
    """Get US Army weapons team sectors."""
    return (
        SectorOfFire(335, 25, True),  # Front
        SectorOfFire(165, 205, False)  # Rear
    )


def _get_team_wedge_right_sectors(role: 'US_IN_Role') -> Tuple[SectorOfFire, SectorOfFire]:
    """
    Get US Army team wedge right formation sectors.
    Using standard coordinate system where 0° = East.

    Coordinate System Standard:
    - 0° = East (right)
    - 90° = South (down)
    - 180° = West (left)
    - 270° = North (up)

    This matches Pygame's coordinate system where (0,0) is top-left.
    """
    sectors = {
        US_IN_Role.AUTO_RIFLEMAN: (
            SectorOfFire(350, 90, True),  # Primary: Southwest
            SectorOfFire(90, 190, False)  # Secondary: Same
        ),
        US_IN_Role.TEAM_LEADER: (
            SectorOfFire(335, 25, True),    # Primary: North to Northeast
            SectorOfFire(165, 205, False)   # Secondary: South
        ),
        US_IN_Role.GRENADIER: (
            SectorOfFire(335, 25, True),    # Primary: North to Northeast
            SectorOfFire(165, 205, False)   # Secondary: South
        ),
        US_IN_Role.RIFLEMAN: (
            SectorOfFire(270, 10, True),    # Primary: Northeast to Southeast
            SectorOfFire(170, 270, False)    # Secondary: Same
        )
    }
    return sectors.get(role, (None, None))


def _get_team_wedge_left_sectors(role: 'US_IN_Role') -> Tuple[SectorOfFire, SectorOfFire]:
    """
    Get US Army team wedge left formation sectors.
    Using standard coordinate system where 0° = East.

    Coordinate System Standard:
    - 0° = East (right)
    - 90° = South (down)
    - 180° = West (left)
    - 270° = North (up)

    This matches Pygame's coordinate system where (0,0) is top-left.
    """
    sectors = {
        US_IN_Role.RIFLEMAN: (
            SectorOfFire(350, 90, True),   # Primary: Southeast to Southwest
            SectorOfFire(90, 190, False)   # Secondary: Same
        ),
        US_IN_Role.GRENADIER: (
            SectorOfFire(335, 25, True),    # Primary: North to Northeast
            SectorOfFire(165, 205, False)   # Secondary: South
        ),
        US_IN_Role.TEAM_LEADER: (
            SectorOfFire(335, 25, True),    # Primary: North to Northeast
            SectorOfFire(165, 205, False)   # Secondary: South
        ),
        US_IN_Role.AUTO_RIFLEMAN: (
            SectorOfFire(270, 10, True),    # Primary: Northeast to Southeast
            SectorOfFire(170, 270, False)    # Secondary: Same
        )
    }
    return sectors.get(role, (None, None))


def _get_team_column_sectors(role: 'US_IN_Role') -> Tuple[SectorOfFire, SectorOfFire]:
    """
    Get US Army team column formation sectors.
    Using standard coordinate system where 0° = East.

    Coordinate System Standard:
    - 0° = East (right)
    - 90° = South (down)
    - 180° = West (left)
    - 270° = North (up)

    This matches Pygame's coordinate system where (0,0) is top-left.
    """
    sectors = {
        US_IN_Role.TEAM_LEADER: (
            SectorOfFire(315, 45, True),    # Primary: North to Northeast
            SectorOfFire(315, 45, False)    # Secondary: Same
        ),
        US_IN_Role.AUTO_RIFLEMAN: (
            SectorOfFire(45, 135, True),    # Primary: Northeast to Southeast
            SectorOfFire(225, 315, False)   # Secondary: Southwest to Northwest
        ),
        US_IN_Role.GRENADIER: (
            SectorOfFire(225, 315, True),   # Primary: Southwest to Northwest
            SectorOfFire(45, 135, False)    # Secondary: Northeast to Southeast
        ),
        US_IN_Role.RIFLEMAN: (
            SectorOfFire(135, 225, True),   # Primary: Southeast to Southwest
            SectorOfFire(135, 225, False)   # Secondary: Same
        )
    }
    return sectors.get(role, (None, None))


# Succession of Command Process
"""
US Army Infantry Succession of Command Rules

This module handles succession of command when leaders are killed in combat.
Different rules apply at different unit levels:

Team Level:
- Next senior soldier becomes Team Leader
- Standard order: AR -> GRN -> RFLM

Squad Level:
- Most senior Team Leader becomes Squad Leader
- Member of that team becomes new Team Leader
- Alpha TL has seniority over Bravo TL

Platoon Level:
- Most senior Squad Leader becomes Platoon Leader
- Their Team Leader becomes new Squad Leader
- Member of that team becomes new Team Leader
"""


def US_IN_handle_leader_casualty(env: MilitaryEnvironment, unit_id: int,
                                 formation_update: bool = True) -> bool:
    """
    Handle succession of command when a leader becomes a casualty.

    Args:
        env: Reference to environment
        unit_id: ID of unit that lost its leader
        formation_update: Whether to update formation after succession

    Returns:
        True if succession was successful, False if unit is combat ineffective
    """
    # Get unit type from environment
    unit_type = env.get_unit_property(unit_id, 'type')

    # Call appropriate handler
    if unit_type == UnitType.INFANTRY_TEAM:
        success = _US_IN_handle_team_succession(env, unit_id)
    elif unit_type == UnitType.INFANTRY_SQUAD:
        success = _US_IN_handle_squad_succession(env, unit_id)
    elif unit_type == UnitType.INFANTRY_PLATOON:
        success = _US_IN_handle_platoon_succession(env, unit_id)
    else:
        raise ValueError(f"Invalid unit type for succession: {unit_type}")

    # Update formation if requested and succession was successful
    if success and formation_update:
        current_formation = env.get_unit_property(unit_id, 'formation')
        US_IN_apply_formation(env, unit_id, current_formation)

    return success


def _US_IN_handle_team_succession(env: MilitaryEnvironment, team_id: int) -> bool:
    """
    Handle succession at team level.
    Returns True if successful, False if team combat ineffective.
    """
    # Get team info
    team_type = env.get_unit_property(team_id, 'team_type')
    member_ids = env.get_unit_children(team_id)

    # Get current roles and find alive members
    alive_members = []
    for member_id in member_ids:
        if env.get_unit_property(member_id, 'health') > 0:
            role = env.get_unit_property(member_id, 'role')
            alive_members.append((member_id, US_IN_Role(role)))

    if not alive_members:
        return False  # No one left alive

    # Different succession orders based on team type
    if team_type == US_IN_TeamType.FIRE.value:
        succession_order = [
            US_IN_Role.RIFLEMAN,
            US_IN_Role.GRENADIER,
            US_IN_Role.AUTO_RIFLEMAN
        ]
    elif team_type in [US_IN_TeamType.GUN.value, US_IN_TeamType.JAVELIN.value]:
        succession_order = [
            US_IN_Role.ASSISTANT_GUNNER,
        ]
    else:
        raise ValueError(f"Invalid team type: {team_type}")

    # Find next in succession order
    for desired_role in succession_order:
        for member_id, current_role in alive_members:
            if current_role == desired_role:
                # Update to leader
                env.update_unit_property(member_id, 'is_leader', True)
                env.update_unit_property(member_id, 'role', US_IN_Role.TEAM_LEADER.value)
                print(f"New team leader selected: {member_id}")
                return True

    return False


def _US_IN_handle_squad_succession(env: MilitaryEnvironment, squad_id: int) -> bool:
    """
    Handle succession at squad level.
    Returns True if successful, False if squad combat ineffective.

    Process:
    1. Alpha TL becomes SL if alive
    2. If not, Bravo TL becomes SL
    3. Promote member of TL's former team to TL
    """
    if env.debug_level > 0:
        print("\nHandling squad succession...")

    # Get teams
    team_ids = [tid for tid in env.get_unit_children(squad_id)
                if env.get_unit_property(tid, 'type') == UnitType.INFANTRY_TEAM]

    # Find Alpha and Bravo teams
    alpha_id = next((tid for tid in team_ids
                     if US_IN_UnitDesignator.ALPHA_TEAM.value in env.get_unit_property(tid, 'string_id')), None)
    bravo_id = next((tid for tid in team_ids
                     if US_IN_UnitDesignator.BRAVO_TEAM.value in env.get_unit_property(tid, 'string_id')), None)

    if alpha_id is None or bravo_id is None:
        if env.debug_level > 0:
            print("Could not find both Alpha and Bravo teams")
        return False

    # Try Alpha TL first
    alpha_tl = _find_team_leader(env, alpha_id)
    if alpha_tl and env.get_unit_property(alpha_tl, 'health') > 0:
        if env.debug_level > 0:
            print("Promoting Alpha Team Leader to Squad Leader")

        # Update Alpha TL to SL
        env.update_unit_property(alpha_tl, 'role', US_IN_Role.SQUAD_LEADER.value)
        env.update_unit_property(alpha_tl, 'observation_range', 60)  # Enhanced observation for SL
        env.set_unit_hierarchy(alpha_tl, squad_id)

        # Handle Alpha team succession
        return _US_IN_handle_team_succession(env, alpha_id)

    # Try Bravo TL if Alpha TL unavailable
    bravo_tl = _find_team_leader(env, bravo_id)
    if bravo_tl and env.get_unit_property(bravo_tl, 'health') > 0:
        if env.debug_level > 0:
            print("Promoting Bravo Team Leader to Squad Leader")

        # Update Bravo TL to SL
        env.update_unit_property(bravo_tl, 'role', US_IN_Role.SQUAD_LEADER.value)
        env.update_unit_property(bravo_tl, 'observation_range', 60)  # Enhanced observation for SL
        env.set_unit_hierarchy(bravo_tl, squad_id)

        # Handle Bravo team succession
        return _US_IN_handle_team_succession(env, bravo_id)

    if env.debug_level > 0:
        print("No valid successors found for Squad Leader position")
    return False


def _US_IN_handle_platoon_succession(env: MilitaryEnvironment, plt_id: int) -> bool:
    """
    Handle succession at platoon level.
    Returns True if successful, False if platoon combat ineffective.

    Process:
    1. Most senior Squad Leader becomes PL (1st squad, then 2nd, then 3rd)
    2. Handle succession for the squad that lost its leader
    """
    if env.debug_level > 0:
        print("\nHandling platoon succession...")

    # Get all squads in seniority order
    squads = [u for u in env.get_unit_children(plt_id)
              if env.get_unit_property(u, 'type') == UnitType.INFANTRY_SQUAD]

    # Sort by squad number (extracted from string_id)
    squads.sort(key=lambda x: int(env.get_unit_property(x, 'string_id').split('SQD')[0][-1]))

    if env.debug_level > 0:
        print(f"Found {len(squads)} squads")

    # Try each squad leader in order
    for squad_id in squads:
        # Find squad leader
        sl = _find_squad_leader(env, squad_id)

        if sl and env.get_unit_property(sl, 'health') > 0:
            if env.debug_level > 0:
                squad_num = env.get_unit_property(squad_id, 'string_id').split('SQD')[0][-1]
                print(f"Promoting {squad_num} Squad Leader to Platoon Leader")

            # Update SL to PL
            env.update_unit_property(sl, 'role', US_IN_Role.PLATOON_LEADER.value)
            env.update_unit_property(sl, 'observation_range', 70)  # Enhanced observation for PL
            env.set_unit_hierarchy(sl, plt_id)

            # Handle squad succession since we took their leader
            return _US_IN_handle_squad_succession(env, squad_id)

    if env.debug_level > 0:
        print("No valid successors found for Platoon Leader position")
    return False


def _find_team_leader(env: MilitaryEnvironment, team_id: int) -> Optional[int]:
    """Helper function to find team leader ID."""
    members = env.get_unit_children(team_id)
    return next((mid for mid in members
                 if env.get_unit_property(mid, 'role') == US_IN_Role.TEAM_LEADER.value), None)


def _find_squad_leader(env: MilitaryEnvironment, squad_id: int) -> Optional[int]:
    """Helper function to find squad leader ID."""
    members = env.get_unit_children(squad_id)
    return next((mid for mid in members
                 if env.get_unit_property(mid, 'role') == US_IN_Role.SQUAD_LEADER.value), None)


# Movement mechanics for US Army Infantry units


def print_unit_state(env: MilitaryEnvironment, unit_id: int, label: str = "", debug_level: int = 0):
    """Print current state of a unit and its members based on debug level."""
    if debug_level <= 0:
        return

    if label:
        print(f"\n{label}:")

    leader_id = _get_unit_leader(env, unit_id)
    members = _get_unit_members(env, unit_id)

    leader_pos = env.get_unit_position(leader_id)
    orientation = env.get_unit_property(unit_id, 'orientation')

    print(f"  Leader at {leader_pos}, Orientation: {orientation}°")
    for m in members:
        role = US_IN_Role(env.get_unit_property(m, 'role'))
        pos = env.get_unit_position(m)
        print(f"  {role.name}: {pos}")


def US_IN_execute_movement(env: MilitaryEnvironment, unit_id: int,
                           direction: Tuple[int, int], distance: int,
                           technique: MovementTechnique = MovementTechnique.TRAVELING,
                           debug_level: int = 1) -> List[Dict]:
    """
    Execute movement for any unit type.

    Debug Levels:
    0 - No output
    1 - Basic movement info (start/end positions, phases)
    2 - Detailed movement info (step by step calculations)
    """
    frames = []  # Initialize frames list

    if debug_level >= 1:
        print(f"\n{'=' * 60}")
        print(f"MOVEMENT EXECUTION - Unit {unit_id}")
        print(f"{'=' * 60}")
        print(f"Direction: {direction}, Distance: {distance} spaces")

    formation_name = env.get_unit_property(unit_id, 'formation')
    formation_type = get_formation_type(formation_name)

    if debug_level >= 1:
        print(f"Formation: {formation_name} ({formation_type.value})")
        print_unit_state(env, unit_id, "Initial Positions", debug_level)

    # Handle rotation if needed
    current_orientation = env.get_unit_property(unit_id, 'orientation', 0)
    new_orientation, rotation_needed = env.calculate_rotation(current_orientation, direction)

    if abs(rotation_needed) > 1:
        if debug_level >= 1:
            print(f"\nROTATION PHASE: {rotation_needed}° needed")

        rotation_steps = 4
        rotation_per_step = rotation_needed // rotation_steps

        for step in range(rotation_steps):
            if debug_level >= 2:
                print(f"\nRotation Step {step + 1}/{rotation_steps}")

            current_orientation = env.get_unit_property(unit_id, 'orientation', 0)
            new_step_orientation = (current_orientation + rotation_per_step) % 360
            env.update_unit_property(unit_id, 'orientation', new_step_orientation)

            if debug_level >= 2:
                print(f"Updated orientation to {new_step_orientation}°")

            formation = env.get_unit_property(unit_id, 'formation')
            template = env.get_unit_property(unit_id, 'formation_template')
            env.apply_formation(unit_id, formation, template)

            frames.append(_capture_positions(env, unit_id))

        if debug_level >= 1:
            print_unit_state(env, unit_id, "After Rotation", debug_level)

    # Calculate movement steps
    if debug_level >= 1:
        print(f"\nMOVEMENT PHASE:")

    steps = 10
    dx, dy = direction
    magnitude = math.sqrt(dx * dx + dy * dy)
    if magnitude > 0:
        # Calculate total movement
        total_dx = (dx * distance) // magnitude
        total_dy = (dy * distance) // magnitude

        if debug_level >= 1:
            print(f"Total movement vector: ({total_dx}, {total_dy})")

        # Generate all intermediate positions
        x_positions = [int(i * total_dx / steps) for i in range(steps + 1)]
        y_positions = [int(i * total_dy / steps) for i in range(steps + 1)]

        # Calculate step movements from position differences
        step_movements = []
        for i in range(steps):
            step_dx = x_positions[i + 1] - x_positions[i]
            step_dy = y_positions[i + 1] - y_positions[i]
            step_movements.append((step_dx, step_dy))

        if debug_level >= 2:
            print("Step movements:")
            for i, (mdx, mdy) in enumerate(step_movements):
                print(f"  Step {i + 1}: ({mdx}, {mdy})")

        # Execute movement based on formation type
        if formation_type == FormationType.WAGON_WHEEL:
            for step_vector in step_movements:
                step_frames = _execute_wagon_wheel_movement(
                    env, unit_id, step_vector, 1, debug_level)
                if step_frames:
                    frames.extend(step_frames)
        else:  # FOLLOW_LEADER
            for step_vector in step_movements:
                step_frames = _execute_follow_leader_movement(
                    env, unit_id, step_vector, 1, debug_level)
                if step_frames:
                    frames.extend(step_frames)

        # Check if we've reached total movement
        final_x = x_positions[-1]
        final_y = y_positions[-1]
        remainder_dx = total_dx - final_x
        remainder_dy = total_dy - final_y

        # Handle any remainder
        if remainder_dx != 0 or remainder_dy != 0:
            if debug_level >= 2:
                print(f"\nFinal adjustment to reach target: ({remainder_dx}, {remainder_dy})")
            if formation_type == FormationType.WAGON_WHEEL:
                remainder_frames = _execute_wagon_wheel_movement(
                    env, unit_id, (remainder_dx, remainder_dy), 1, debug_level)
                if remainder_frames:
                    frames.extend(remainder_frames)
            else:  # FOLLOW_LEADER
                remainder_frames = _execute_follow_leader_movement(
                    env, unit_id, (remainder_dx, remainder_dy), 1, debug_level)
                if remainder_frames:
                    frames.extend(remainder_frames)

    if debug_level >= 1:
        print_unit_state(env, unit_id, "\nFinal Positions", debug_level)
        print(f"\nMovement complete - {len(frames)} frames captured")
        print(f"{'=' * 60}\n")

    # Always return frames list, even if empty
    return frames


def _execute_wagon_wheel_movement(env: MilitaryEnvironment, unit_id: int,
                                  move_vector: Tuple[int, int],
                                  steps: int,
                                  debug_level: int = 0) -> List[Dict]:
    """Execute movement maintaining wedge/line formation shape."""
    frames = []
    move_dx, move_dy = move_vector

    # Get formation and template info
    formation = env.get_unit_property(unit_id, 'formation')
    template = env.get_unit_property(unit_id, 'formation_template')

    if debug_level >= 2:
        print(f"\n=== Starting Wagon Wheel Movement ===")
        print(f"Movement vector: ({move_dx}, {move_dy})")

    # Get leader and members
    leader_id = _get_unit_leader(env, unit_id)
    members = _get_unit_members(env, unit_id)
    start_pos = env.get_unit_position(leader_id)
    orientation = env.get_unit_property(unit_id, 'orientation')

    if debug_level >= 2:
        print_unit_state(env, unit_id, "Initial Positions", debug_level)

    # Store initial positions for interpolation
    initial_positions = {leader_id: start_pos}
    for member_id in members:
        initial_positions[member_id] = env.get_unit_position(member_id)

    # Calculate final positions
    final_leader_x = start_pos[0] + move_dx
    final_leader_y = start_pos[1] + move_dy
    final_leader_pos = (final_leader_x, final_leader_y)

    # Calculate interpolation steps
    for step in range(steps):
        if debug_level >= 2:
            print(f"\n--- Step {step + 1}/{steps} ---")

        # Interpolate leader position
        progress = (step + 1) / steps
        current_x = start_pos[0] + int(move_dx * progress)
        current_y = start_pos[1] + int(move_dy * progress)
        current_pos = (current_x, current_y)

        # Move leader
        if debug_level >= 2:
            print(f"Leader moved: {env.get_unit_position(leader_id)} -> {current_pos}")
        env.update_unit_position(leader_id, current_pos)

        # Move members maintaining formation
        angle_rad = math.radians(orientation - 90)

        for member_id in members:
            role = US_IN_Role(env.get_unit_property(member_id, 'role'))
            if role in template:
                if debug_level >= 2:
                    print(f"\n  Moving {role.name}:")

                # Calculate rotated offset from template
                offset = template[role]
                rot_x = int(offset[0] * math.cos(angle_rad) - offset[1] * math.sin(angle_rad))
                rot_y = int(offset[0] * math.sin(angle_rad) + offset[1] * math.cos(angle_rad))

                # Calculate new position
                new_pos = (current_x + rot_x, current_y + rot_y)

                if debug_level >= 2:
                    old_pos = env.get_unit_position(member_id)
                    print(f"    {old_pos} -> {new_pos}")

                env.update_unit_position(member_id, new_pos)

        # Capture frame after moving everyone
        frames.append(_capture_positions(env, unit_id))

    if debug_level >= 2:
        print_unit_state(env, unit_id, "Final Positions", debug_level)

    return frames


def _execute_follow_leader_movement(env: MilitaryEnvironment, unit_id: int,
                                    move_vector: Tuple[int, int],
                                    steps: int,
                                    debug_level: int = 0) -> List[Dict]:
    """Execute movement where elements follow leader's path maintaining formation."""
    frames = []
    move_dx, move_dy = move_vector

    # Get formation and template info
    formation = env.get_unit_property(unit_id, 'formation')
    template = env.get_unit_property(unit_id, 'formation_template')

    if debug_level >= 2:
        print(f"\n=== Starting Follow Leader Movement ===")
        print(f"Movement vector: ({move_dx}, {move_dy})")

    # Get leader and members
    leader_id = _get_unit_leader(env, unit_id)
    members = _get_unit_members(env, unit_id)
    start_pos = env.get_unit_position(leader_id)
    orientation = env.get_unit_property(unit_id, 'orientation')

    if debug_level >= 2:
        print_unit_state(env, unit_id, "Initial Positions", debug_level)

    # Store initial positions for interpolation
    initial_positions = {leader_id: start_pos}
    for member_id in members:
        initial_positions[member_id] = env.get_unit_position(member_id)

    # Calculate final positions
    final_leader_x = start_pos[0] + move_dx
    final_leader_y = start_pos[1] + move_dy
    final_leader_pos = (final_leader_x, final_leader_y)

    # Calculate interpolation steps
    for step in range(steps):
        if debug_level >= 2:
            print(f"\n--- Step {step + 1}/{steps} ---")

        # Interpolate leader position
        progress = (step + 1) / steps
        current_x = start_pos[0] + int(move_dx * progress)
        current_y = start_pos[1] + int(move_dy * progress)
        current_pos = (current_x, current_y)

        # Move leader
        if debug_level >= 2:
            print(f"Leader moved: {env.get_unit_position(leader_id)} -> {current_pos}")
        env.update_unit_position(leader_id, current_pos)

        # Move members maintaining formation
        angle_rad = math.radians(orientation - 90)

        for member_id in members:
            role = US_IN_Role(env.get_unit_property(member_id, 'role'))
            if role in template:
                if debug_level >= 2:
                    print(f"\n  Moving {role.name}:")

                # Calculate rotated offset from template
                offset = template[role]
                rot_x = int(offset[0] * math.cos(angle_rad) - offset[1] * math.sin(angle_rad))
                rot_y = int(offset[0] * math.sin(angle_rad) + offset[1] * math.cos(angle_rad))

                # Calculate new position
                new_pos = (current_x + rot_x, current_y + rot_y)

                if debug_level >= 2:
                    old_pos = env.get_unit_position(member_id)
                    print(f"    {old_pos} -> {new_pos}")

                env.update_unit_position(member_id, new_pos)

        # Capture frame after moving everyone
        frames.append(_capture_positions(env, unit_id))

    if debug_level >= 2:
        print_unit_state(env, unit_id, "Final Positions", debug_level)

    return frames


def _capture_positions(env: MilitaryEnvironment, unit_id: int) -> Dict:
    """Capture positions of all unit elements."""
    unit_type = env.get_unit_property(unit_id, 'type')
    unit_string = env.get_unit_property(unit_id, 'string_id')

    positions = []

    # Add leader
    leader_id = _get_unit_leader(env, unit_id)
    if leader_id:
        positions.append({
            'role': env.get_unit_property(leader_id, 'role'),
            'position': env.get_unit_position(leader_id),
            'is_leader': True
        })

    # Add members
    for member_id in _get_unit_members(env, unit_id):
        positions.append({
            'role': env.get_unit_property(member_id, 'role'),
            'position': env.get_unit_position(member_id),
            'is_leader': False
        })

    return {
        'unit_type': unit_type,
        'unit_id': unit_string,
        'positions': positions
    }


def _get_unit_leader(env: MilitaryEnvironment, unit_id: int) -> Optional[int]:
    """Get the leader of a unit."""
    children = env.get_unit_children(unit_id)
    return next((child for child in children
                 if env.get_unit_property(child, 'is_leader', False)), None)


def _get_unit_members(env: MilitaryEnvironment, unit_id: int) -> List[int]:
    """Get non-leader members of a unit."""
    children = env.get_unit_children(unit_id)
    return [child for child in children
            if not env.get_unit_property(child, 'is_leader', False)]


def US_IN_execute_route_movement(env: MilitaryEnvironment,
                                 unit_id: int,
                                 route: MovementRoute,
                                 debug_level: int = 0) -> List[Dict]:
    """
    Execute movement along a defined route, handling both position and orientation.

    Args:
        env: Reference to environment
        unit_id: ID of unit to move
        route: MovementRoute object defining the path
        debug_level: Level of debug output (0=None, 1=Basic, 2=Detailed)

    Returns:
        List of animation frames showing movement
    """
    all_frames = []
    steps_per_move = 10  # Number of interpolation steps between waypoints

    if debug_level >= 1:
        print(f"\nExecuting route movement for unit {unit_id}")
        print(f"Total waypoints: {len(route.waypoints)}")
        print(f"Current waypoint: {route.current_waypoint}")
        print(f"Movement technique: {route.technique.value}")

    # Process each remaining waypoint
    while route.current_waypoint < len(route.waypoints):
        current_waypoint = route.waypoints[route.current_waypoint]
        current_pos = env.get_unit_position(unit_id)
        current_orientation = env.get_unit_property(unit_id, 'orientation')

        if debug_level >= 1:
            print(f"\nMoving to waypoint {route.current_waypoint + 1}")
            print(f"Current position: {current_pos}")
            print(f"Current orientation: {current_orientation}°")
            print(f"Target position: {current_waypoint.position}")

        # First apply any formation change
        if current_waypoint.formation:
            if debug_level >= 1:
                print(f"Changing formation to: {current_waypoint.formation}")
            US_IN_apply_formation(env, unit_id, current_waypoint.formation)
            all_frames.append(_capture_positions(env, unit_id))

        # Calculate movement to waypoint
        target_x, target_y = current_waypoint.position
        start_x, start_y = current_pos
        total_dx = target_x - start_x
        total_dy = target_y - start_y

        if total_dx != 0 or total_dy != 0:
            # Calculate new orientation based on movement direction
            new_orientation = int((math.degrees(math.atan2(total_dy, total_dx)) + 360) % 360)
            rotation_needed = ((new_orientation - current_orientation + 180) % 360) - 180

            if debug_level >= 1:
                print(f"Movement vector: ({total_dx}, {total_dy})")
                print(f"New orientation: {new_orientation}°")
                print(f"Rotation needed: {rotation_needed}°")

            # Handle rotation if needed
            if abs(rotation_needed) > 1:
                rotation_steps = 4
                rotation_per_step = rotation_needed / rotation_steps

                for step in range(rotation_steps):
                    # Update orientation
                    current_orientation = (current_orientation + rotation_per_step) % 360
                    env.update_unit_property(unit_id, 'orientation', current_orientation)

                    # Apply formation at new orientation
                    if current_waypoint.formation:
                        US_IN_apply_formation(env, unit_id, current_waypoint.formation)
                    all_frames.append(_capture_positions(env, unit_id))

                    if debug_level >= 2:
                        print(f"Rotation step {step + 1}: {current_orientation}°")

            # Calculate step sizes for movement
            step_x = total_dx / steps_per_move
            step_y = total_dy / steps_per_move

            # Move in steps
            for step in range(steps_per_move):
                # Calculate new position
                new_x = int(start_x + step_x * (step + 1))
                new_y = int(start_y + step_y * (step + 1))

                # Update unit position
                env.update_unit_position(unit_id, (new_x, new_y))

                # Apply formation at new position
                if current_waypoint.formation:
                    US_IN_apply_formation(env, unit_id, current_waypoint.formation)

                # Capture frame
                all_frames.append(_capture_positions(env, unit_id))

                if debug_level >= 2:
                    print(f"Step {step + 1}: Position updated to ({new_x}, {new_y})")

            # Ensure final position is exactly at waypoint
            env.update_unit_position(unit_id, current_waypoint.position)
            if current_waypoint.formation:
                US_IN_apply_formation(env, unit_id, current_waypoint.formation)
            all_frames.append(_capture_positions(env, unit_id))

            if debug_level >= 1:
                final_pos = env.get_unit_position(unit_id)
                final_orientation = env.get_unit_property(unit_id, 'orientation')
                print(f"Movement complete. Position: {final_pos}, Orientation: {final_orientation}°")

        # Handle hold time if specified
        if current_waypoint.hold_time > 0:
            if debug_level >= 1:
                print(f"Holding at waypoint for {current_waypoint.hold_time} steps")
            hold_frame = _capture_positions(env, unit_id)
            for _ in range(current_waypoint.hold_time):
                all_frames.append(hold_frame)

        # Move to next waypoint
        route.current_waypoint += 1
        if debug_level >= 1:
            current_pos = env.get_unit_position(unit_id)
            current_orientation = env.get_unit_property(unit_id, 'orientation')
            print(f"Moving to next waypoint. Position: {current_pos}, Orientation: {current_orientation}°")

    route.completed = True

    if debug_level >= 1:
        final_pos = env.get_unit_position(unit_id)
        final_orientation = env.get_unit_property(unit_id, 'orientation')
        print("\nRoute movement completed")
        print(f"Final position: {final_pos}")
        print(f"Final orientation: {final_orientation}°")
        print(f"Total frames: {len(all_frames)}")

    return all_frames


def US_IN_create_route(waypoints: List[Tuple[int, int]],
                       technique: MovementTechnique = MovementTechnique.TRAVELING,
                       formations: List[str] = None,
                       hold_times: List[int] = None,
                       required_actions: List[List[str]] = None) -> MovementRoute:
    """
    Create a movement route from a list of waypoints and optional parameters.

    Args:
        waypoints: List of (x, y) positions defining the route
        technique: Movement technique to use (TRAVELING or BOUNDING)
        formations: Optional list of formations to use at each waypoint
        hold_times: Optional list of hold times at each waypoint
        required_actions: Optional list of required actions at each waypoint

    Returns:
        MovementRoute object
    """
    route_waypoints = []

    for i, pos in enumerate(waypoints):
        # Default values
        formation = None
        hold_time = 0
        actions = None

        # Get optional parameters if provided
        if formations and i < len(formations):
            formation = formations[i]
        if hold_times and i < len(hold_times):
            hold_time = hold_times[i]
        if required_actions and i < len(required_actions):
            actions = required_actions[i]

        # Create waypoint with explicit values
        waypoint = RouteWaypoint(
            position=pos,
            formation=formation,
            hold_time=hold_time,
            required_actions=actions
        )
        route_waypoints.append(waypoint)

    return MovementRoute(
        waypoints=route_waypoints,
        technique=technique
    )


def _validate_route(env: MilitaryEnvironment, unit_id: int, route: MovementRoute) -> bool:
    """
    Validate a movement route.

    Checks:
    - All waypoints are within environment bounds
    - All formations (if specified) are valid for unit type
    - Path doesn't cross impassable terrain (future)
    - Route is achievable given unit capabilities (future)

    Args:
        env: Reference to environment
        unit_id: ID of unit that will execute route
        route: MovementRoute to validate

    Returns:
        True if route is valid, False otherwise
    """
    # Get environment dimensions
    width = env.width
    height = env.height

    # Get unit type
    unit_type = env.get_unit_property(unit_id, 'type')

    # Check each waypoint
    for wp in route.waypoints:
        # Check position bounds
        if not (0 <= wp.position[0] < width and 0 <= wp.position[1] < height):
            return False

        # Check formation validity if specified
        if wp.formation and not US_IN_validate_formation(wp.formation, unit_type):
            return False

    return True


# Squad related movement functions

def _print_squad_state(env: MilitaryEnvironment, squad_id: int):
    """Print detailed squad state including all positions and orientations."""
    sl = _get_squad_leader(env, squad_id)
    alpha = _get_alpha_team(env, squad_id)
    bravo = _get_bravo_team(env, squad_id)

    print("\nSquad State:")
    print("=" * 50)

    # Squad level info
    sl_pos = env.get_unit_position(sl)
    squad_orientation = env.get_unit_property(squad_id, 'orientation')
    print(f"Squad Leader Position: {sl_pos}")
    print(f"Squad Orientation: {squad_orientation}°")

    # Team level info
    for team_id, team_name in [(alpha, "Alpha"), (bravo, "Bravo")]:
        tl = _get_team_leader(env, team_id)
        tl_pos = env.get_unit_position(tl)
        team_orientation = env.get_unit_property(team_id, 'orientation')
        dist_to_sl = _calculate_distance(sl_pos, tl_pos)

        print(f"\n{team_name} Team:")
        print("-" * 30)
        print(f"Team Leader Position: {tl_pos}")
        print(f"Team Orientation: {team_orientation}°")
        print(f"Distance to SL: {dist_to_sl:.1f}")

        # Member info
        members = _get_unit_members(env, team_id)
        for member in members:
            role = US_IN_Role(env.get_unit_property(member, 'role')).name
            pos = env.get_unit_position(member)
            dist_to_tl = _calculate_distance(tl_pos, pos)
            print(f"  {role}:")
            print(f"    Position: {pos}")
            print(f"    Distance to TL: {dist_to_tl:.1f}")


def execute_squad_movement(env: MilitaryEnvironment, squad_id: int,
                           direction: Tuple[int, int], distance: int,
                           technique: MovementTechnique = MovementTechnique.TRAVELING,
                           debug_level: int = 1,
                           route: Optional[MovementRoute] = None) -> List[Dict]:
    """
    Execute squad movement - either a single movement or a sequence of waypoints.

    Args:
        env: Environment reference
        squad_id: ID of squad to move
        direction: (dx, dy) movement vector
        distance: Total distance to move
        technique: TRAVELING or BOUNDING movement technique
        debug_level: Level of debug output
        route: Optional MovementRoute for multi-waypoint movement

    Returns:
        List of animation frames
    """
    all_frames = []

    if route is None:
        # Single point movement
        if debug_level >= 1:
            print(f"\n=== Squad Movement Execution ===")
            print(f"Squad: {env.get_unit_property(squad_id, 'string_id')}")
            print(f"Direction: {direction}, Distance: {distance}")
            print(f"Technique: {technique.value}")

        # Get squad's current orientation and starting position
        current_orientation = env.get_unit_property(squad_id, 'orientation')
        squad_leader = _get_squad_leader(env, squad_id)
        start_pos = env.get_unit_position(squad_leader)
        if debug_level >= 1:
            print(f"Starting position: {start_pos}")

        # Calculate new orientation and rotation
        dx, dy = direction
        if dx != 0 or dy != 0:
            new_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
            rotation_needed = ((new_orientation - current_orientation + 180) % 360) - 180

            if debug_level >= 1:
                print(f"Current orientation: {current_orientation}°")
                print(f"New orientation: {new_orientation}°")
                print(f"Rotation needed: {rotation_needed}°")

            # Execute rotation if needed
            if abs(rotation_needed) > 1:
                rotation_frames = _execute_squad_rotation(env, squad_id, rotation_needed, debug_level)
                all_frames.extend(rotation_frames)

        # Execute movement
        if technique == MovementTechnique.TRAVELING:
            movement_frames = _execute_squad_traveling(env, squad_id, direction, distance, debug_level)
            all_frames.extend(movement_frames)
        else:  # BOUNDING
            movement_frames = _execute_squad_bounding(env, squad_id, direction, distance, debug_level)
            all_frames.extend(movement_frames)

    else:
        # Route movement
        if debug_level >= 1:
            print(f"\n=== Starting Squad Route Movement ===")
            print(f"Total waypoints: {len(route.waypoints)}")
            print(f"Current waypoint: {route.current_waypoint}")
            print(f"Movement technique: {technique.value}")

        # Process each waypoint
        while route.current_waypoint < len(route.waypoints):
            current_waypoint = route.waypoints[route.current_waypoint]
            squad_pos = env.get_unit_position(_get_squad_leader(env, squad_id))

            if debug_level >= 1:
                print(f"\nMoving to waypoint {route.current_waypoint + 1}")
                print(f"Current position: {squad_pos}")
                print(f"Target position: {current_waypoint.position}")

            # Apply formation if specified
            if current_waypoint.formation:
                if debug_level >= 1:
                    print(f"Changing formation to: {current_waypoint.formation}")
                US_IN_apply_formation(env, squad_id, current_waypoint.formation)
                all_frames.append(_capture_squad_positions(env, squad_id))

            # Get current position for this leg
            squad_leader = _get_squad_leader(env, squad_id)
            current_pos = env.get_unit_position(squad_leader)

            # Calculate movement to waypoint
            target_x, target_y = current_waypoint.position
            dx = target_x - current_pos[0]
            dy = target_y - current_pos[1]
            waypoint_distance = int(math.sqrt(dx * dx + dy * dy))

            if debug_level >= 1:
                print(f"Current position: {current_pos}")
                print(f"Target position: {current_waypoint.position}")
                print(f"Movement vector: ({dx}, {dy})")
                print(f"Distance: {waypoint_distance}")

            if waypoint_distance > 0:
                # Execute movement to this waypoint
                waypoint_frames = execute_squad_movement(
                    env=env,
                    squad_id=squad_id,
                    direction=(dx, dy),
                    distance=waypoint_distance,
                    technique=technique,
                    debug_level=debug_level
                )
                all_frames.extend(waypoint_frames)

            # Handle hold time
            if current_waypoint.hold_time > 0:
                if debug_level >= 1:
                    print(f"\nHolding at waypoint for {current_waypoint.hold_time} steps")
                hold_frame = _capture_squad_positions(env, squad_id)
                for _ in range(current_waypoint.hold_time):
                    all_frames.append(hold_frame)

            # Store final positions for next waypoint
            final_sl_pos = env.get_unit_position(squad_leader)
            final_orientation = env.get_unit_property(squad_id, 'orientation')

            # Update state for next waypoint
            env.state_manager.update_unit_position(squad_id, final_sl_pos)
            env.state_manager.update_unit_property(squad_id, 'position', final_sl_pos)
            env.state_manager.update_unit_property(squad_id, 'orientation', final_orientation)

            if debug_level >= 1:
                print(f"\nWaypoint complete")
                print(f"Final position: {final_sl_pos}")
                print(f"Final orientation: {final_orientation}°")

            # Move to next waypoint
            route.current_waypoint += 1

        route.completed = True

    return all_frames


def _execute_squad_rotation(env: MilitaryEnvironment, squad_id: int,
                            rotation_angle: int, debug_level: int = 1) -> List[Dict]:
    """Handle coordinated rotation of entire squad with proper team rotations."""
    frames = []
    steps = 4  # Number of rotation steps
    rotation_per_step = rotation_angle // steps

    # Get squad components and initial positions
    squad_leader = _get_squad_leader(env, squad_id)
    alpha_team = _get_alpha_team(env, squad_id)
    bravo_team = _get_bravo_team(env, squad_id)
    sl_pos = env.get_unit_position(squad_leader)

    # Track orientations separately
    squad_orientation = env.get_unit_property(squad_id, 'orientation', 0)
    alpha_orientation = env.get_unit_property(alpha_team, 'orientation', 0)
    bravo_orientation = env.get_unit_property(bravo_team, 'orientation', 0)

    # Get templates and validate
    squad_template = env.get_unit_property(squad_id, 'formation_template')
    alpha_template = env.get_unit_property(alpha_team, 'formation_template')
    bravo_template = env.get_unit_property(bravo_team, 'formation_template')

    # Validate and get template keys
    template_info = _validate_template_distances(env, squad_id, debug_level)
    alpha_key = template_info['alpha_key']
    bravo_key = template_info['bravo_key']

    if debug_level >= 1:
        print("\nInitial orientations:")
        print(f"Squad: {squad_orientation}°")
        print(f"Alpha: {alpha_orientation}°")
        print(f"Bravo: {bravo_orientation}°")

        if not template_info['squad_valid']:
            print("WARNING: Initial formation distances not valid")

    # Execute rotation steps
    for step in range(steps):
        # Update orientations
        squad_orientation = (squad_orientation + rotation_per_step) % 360
        alpha_orientation = squad_orientation  # Teams rotate with squad
        bravo_orientation = squad_orientation

        # Update orientation in environment
        env.update_unit_property(squad_id, 'orientation', squad_orientation)
        env.update_unit_property(alpha_team, 'orientation', alpha_orientation)
        env.update_unit_property(bravo_team, 'orientation', bravo_orientation)

        # Calculate rotated positions for teams
        squad_angle_rad = math.radians(squad_orientation - 90)

        # Position Alpha Team
        if alpha_key in squad_template:
            alpha_offset = squad_template[alpha_key]
            alpha_x = sl_pos[0] + int(alpha_offset[0] * math.cos(squad_angle_rad) -
                                      alpha_offset[1] * math.sin(squad_angle_rad))
            alpha_y = sl_pos[1] + int(alpha_offset[0] * math.sin(squad_angle_rad) +
                                      alpha_offset[1] * math.cos(squad_angle_rad))
            alpha_tl = _get_team_leader(env, alpha_team)
            env.update_unit_position(alpha_tl, (alpha_x, alpha_y))

            # Position Alpha Team members
            alpha_angle_rad = math.radians(alpha_orientation - 90)
            for member in _get_unit_members(env, alpha_team):
                role = US_IN_Role(env.get_unit_property(member, 'role'))
                if role in alpha_template:
                    member_offset = alpha_template[role]
                    member_x = alpha_x + int(member_offset[0] * math.cos(alpha_angle_rad) -
                                             member_offset[1] * math.sin(alpha_angle_rad))
                    member_y = alpha_y + int(member_offset[0] * math.sin(alpha_angle_rad) +
                                             member_offset[1] * math.cos(alpha_angle_rad))
                    env.update_unit_position(member, (member_x, member_y))

        # Position Bravo Team
        if bravo_key in squad_template:
            bravo_offset = squad_template[bravo_key]
            bravo_x = sl_pos[0] + int(bravo_offset[0] * math.cos(squad_angle_rad) -
                                      bravo_offset[1] * math.sin(squad_angle_rad))
            bravo_y = sl_pos[1] + int(bravo_offset[0] * math.sin(squad_angle_rad) +
                                      bravo_offset[1] * math.cos(squad_angle_rad))
            bravo_tl = _get_team_leader(env, bravo_team)
            env.update_unit_position(bravo_tl, (bravo_x, bravo_y))

            # Position Bravo Team members
            bravo_angle_rad = math.radians(bravo_orientation - 90)
            for member in _get_unit_members(env, bravo_team):
                role = US_IN_Role(env.get_unit_property(member, 'role'))
                if role in bravo_template:
                    member_offset = bravo_template[role]
                    member_x = bravo_x + int(member_offset[0] * math.cos(bravo_angle_rad) -
                                             member_offset[1] * math.sin(bravo_angle_rad))
                    member_y = bravo_y + int(member_offset[0] * math.sin(bravo_angle_rad) +
                                             member_offset[1] * math.cos(bravo_angle_rad))
                    env.update_unit_position(member, (member_x, member_y))

        # Validate formation after rotation
        if debug_level >= 2:
            rotation_valid = _validate_template_distances(env, squad_id, debug_level)
            if not rotation_valid['squad_valid']:
                print(f"WARNING: Formation distances invalid after rotation step {step + 1}")

        # Capture intermediate state
        frames.append(_capture_squad_positions(env, squad_id))

        if debug_level >= 1:
            print(f"\nRotation step {step + 1}:")
            print(f"Squad orientation: {squad_orientation}°")
            print(f"Alpha orientation: {alpha_orientation}°")
            print(f"Bravo orientation: {bravo_orientation}°")

    return frames


def _execute_squad_traveling(env: MilitaryEnvironment, squad_id: int,
                             direction: Tuple[int, int], distance: int,
                             debug_level: int = 1) -> List[Dict]:
    """Execute squad movement while maintaining formations."""
    frames = []
    steps = 10  # Number of interpolation steps

    if debug_level >= 1:
        print(f"\n=== Executing Squad Traveling Movement ===")
        print(f"Total movement vector: ({direction[0]}, {direction[1]})")
        print(f"Total distance: {distance}")
        print(f"Steps: {steps}")

    # Get squad components and orientations
    squad_leader = _get_squad_leader(env, squad_id)
    alpha_team = _get_alpha_team(env, squad_id)
    bravo_team = _get_bravo_team(env, squad_id)

    squad_orientation = env.get_unit_property(squad_id, 'orientation')
    alpha_orientation = env.get_unit_property(alpha_team, 'orientation')
    bravo_orientation = env.get_unit_property(bravo_team, 'orientation')

    # Get templates
    squad_template = env.get_unit_property(squad_id, 'formation_template')
    alpha_template = env.get_unit_property(alpha_team, 'formation_template')
    bravo_template = env.get_unit_property(bravo_team, 'formation_template')

    # Validate and get template keys
    template_info = _validate_template_distances(env, squad_id, debug_level)
    alpha_key = template_info['alpha_key']
    bravo_key = template_info['bravo_key']

    if debug_level >= 1 and not template_info['squad_valid']:
        print("WARNING: Initial formation distances not valid")

    # Calculate movement increments
    dx, dy = direction
    magnitude = math.sqrt(dx * dx + dy * dy)
    if magnitude > 0:
        step_distance = distance // steps
        step_dx = int((dx * step_distance) / magnitude)
        step_dy = int((dy * step_distance) / magnitude)
    else:
        step_dx = step_dy = 0

    # Get starting position
    sl_pos = env.get_unit_position(squad_leader)
    current_x, current_y = sl_pos

    # Execute movement steps
    for step in range(steps):
        # Move squad leader
        new_sl_x = current_x + step_dx
        new_sl_y = current_y + step_dy

        # Validate position is within bounds
        new_sl_x = int(max(0, min(new_sl_x, env.width - 1)))
        new_sl_y = int(max(0, min(new_sl_y, env.height - 1)))

        # Update squad leader position
        env.update_unit_position(squad_leader, (new_sl_x, new_sl_y))

        # Update team positions based on new SL position
        squad_angle_rad = math.radians(squad_orientation - 90)

        # Move Alpha Team
        if alpha_key in squad_template:
            alpha_offset = squad_template[alpha_key]
            alpha_x = new_sl_x + int(alpha_offset[0] * math.cos(squad_angle_rad) -
                                     alpha_offset[1] * math.sin(squad_angle_rad))
            alpha_y = new_sl_y + int(alpha_offset[0] * math.sin(squad_angle_rad) +
                                     alpha_offset[1] * math.cos(squad_angle_rad))

            # Validate team position is within bounds
            alpha_x = int(max(0, min(alpha_x, env.width - 1)))
            alpha_y = int(max(0, min(alpha_y, env.height - 1)))

            alpha_tl = _get_team_leader(env, alpha_team)
            env.update_unit_position(alpha_tl, (alpha_x, alpha_y))

            # Move Alpha Team members
            alpha_angle_rad = math.radians(alpha_orientation - 90)
            for member in _get_unit_members(env, alpha_team):
                role = US_IN_Role(env.get_unit_property(member, 'role'))
                if role in alpha_template:
                    member_offset = alpha_template[role]
                    member_x = alpha_x + int(member_offset[0] * math.cos(alpha_angle_rad) -
                                             member_offset[1] * math.sin(alpha_angle_rad))
                    member_y = alpha_y + int(member_offset[0] * math.sin(alpha_angle_rad) +
                                             member_offset[1] * math.cos(alpha_angle_rad))

                    # Validate member position is within bounds
                    member_x = int(max(0, min(member_x, env.width - 1)))
                    member_y = int(max(0, min(member_y, env.height - 1)))

                    env.update_unit_position(member, (member_x, member_y))

        # Move Bravo Team
        if bravo_key in squad_template:
            bravo_offset = squad_template[bravo_key]
            bravo_x = new_sl_x + int(bravo_offset[0] * math.cos(squad_angle_rad) -
                                     bravo_offset[1] * math.sin(squad_angle_rad))
            bravo_y = new_sl_y + int(bravo_offset[0] * math.sin(squad_angle_rad) +
                                     bravo_offset[1] * math.cos(squad_angle_rad))

            # Validate team position is within bounds
            bravo_x = int(max(0, min(bravo_x, env.width - 1)))
            bravo_y = int(max(0, min(bravo_y, env.height - 1)))

            bravo_tl = _get_team_leader(env, bravo_team)
            env.update_unit_position(bravo_tl, (bravo_x, bravo_y))

            # Move Bravo Team members
            bravo_angle_rad = math.radians(bravo_orientation - 90)
            for member in _get_unit_members(env, bravo_team):
                role = US_IN_Role(env.get_unit_property(member, 'role'))
                if role in bravo_template:
                    member_offset = bravo_template[role]
                    member_x = bravo_x + int(member_offset[0] * math.cos(bravo_angle_rad) -
                                             member_offset[1] * math.sin(bravo_angle_rad))
                    member_y = bravo_y + int(member_offset[0] * math.sin(bravo_angle_rad) +
                                             member_offset[1] * math.cos(bravo_angle_rad))

                    # Validate member position is within bounds
                    member_x = int(max(0, min(member_x, env.width - 1)))
                    member_y = int(max(0, min(member_y, env.height - 1)))

                    env.update_unit_position(member, (member_x, member_y))

        # Update current position for next step
        current_x = new_sl_x
        current_y = new_sl_y

        # Validate formation after movement
        if debug_level >= 2:
            validation_results = _validate_template_distances(env, squad_id, debug_level)
            if not validation_results['squad_valid']:
                print(f"WARNING: Formation distances invalid after step {step + 1}")

        # Capture intermediate state
        frames.append(_capture_squad_positions(env, squad_id))

        if debug_level >= 1:
            print(f"\nMovement step {step + 1}:")
            _print_squad_state(env, squad_id)

    return frames


def _execute_squad_bounding(env: MilitaryEnvironment, squad_id: int,
                            direction: Tuple[int, int], distance: int,
                            debug_level: int = 1) -> List[Dict]:
    """
    Execute bounding movement where teams alternate moving.
    One team moves while the other provides security.
    """
    frames = []
    bound_distance = min(50, distance)  # Maximum single bound distance
    steps = 10  # Steps per bound for smooth movement
    current_distance = 0

    # Get squad components
    squad_leader = _get_squad_leader(env, squad_id)
    alpha_team = _get_alpha_team(env, squad_id)
    bravo_team = _get_bravo_team(env, squad_id)

    # Get current orientations
    squad_orientation = env.get_unit_property(squad_id, 'orientation')
    alpha_orientation = env.get_unit_property(alpha_team, 'orientation')
    bravo_orientation = env.get_unit_property(bravo_team, 'orientation')

    # Get templates
    squad_template = env.get_unit_property(squad_id, 'formation_template')
    alpha_template = env.get_unit_property(alpha_team, 'formation_template')
    bravo_template = env.get_unit_property(bravo_team, 'formation_template')

    # Validate templates
    template_info = _validate_template_distances(env, squad_id, debug_level)
    alpha_key = template_info['alpha_key']
    bravo_key = template_info['bravo_key']

    if debug_level >= 1:
        print("\n=== Executing Squad Bounding Movement ===")
        print(f"Total distance: {distance}")
        print(f"Bound distance: {bound_distance}")
        print(f"Initial orientations:")
        print(f"Squad: {squad_orientation}°")
        print(f"Alpha: {alpha_orientation}°")
        print(f"Bravo: {bravo_orientation}°")

    # Start with Alpha team bounding
    moving_team = alpha_team
    moving_key = alpha_key
    moving_template = alpha_template
    moving_team_id = "Alpha"
    moving_sl_offset = squad_template.get('Squad Leader', (4, -4))

    overwatch_team = bravo_team
    overwatch_key = bravo_key
    overwatch_template = bravo_template
    overwatch_team_id = "Bravo"
    bound_number = 1

    while current_distance < distance:
        if debug_level >= 1:
            print(f"\n=== Bound {bound_number} ===")
            print(f"Moving team: {moving_team_id}")
            print(f"Overwatch team: {overwatch_team_id}")

        # Calculate current bound
        remaining = distance - current_distance
        current_bound = min(bound_distance, remaining)

        # Calculate movement increments for this bound
        dx, dy = direction
        magnitude = math.sqrt(dx * dx + dy * dy)
        if magnitude > 0:
            step_distance = current_bound // steps
            step_dx = int((dx * step_distance) / magnitude)
            step_dy = int((dy * step_distance) / magnitude)
        else:
            step_dx = step_dy = 0

        # Execute bound in steps
        for step in range(steps):
            # Get current positions
            moving_tl = _get_team_leader(env, moving_team)
            tl_pos = env.get_unit_position(moving_tl)

            # Calculate new team leader position
            new_tl_x = tl_pos[0] + step_dx
            new_tl_y = tl_pos[1] + step_dy

            # Validate position is within bounds
            new_tl_x = int(max(0, min(new_tl_x, env.width - 1)))
            new_tl_y = int(max(0, min(new_tl_y, env.height - 1)))

            # Move team leader
            env.update_unit_position(moving_tl, (new_tl_x, new_tl_y))

            # Move team members maintaining formation
            team_angle_rad = math.radians(squad_orientation - 90)
            for member in _get_unit_members(env, moving_team):
                role = US_IN_Role(env.get_unit_property(member, 'role'))
                if role in moving_template:
                    member_offset = moving_template[role]
                    member_x = new_tl_x + int(member_offset[0] * math.cos(team_angle_rad) -
                                              member_offset[1] * math.sin(team_angle_rad))
                    member_y = new_tl_y + int(member_offset[0] * math.sin(team_angle_rad) +
                                              member_offset[1] * math.cos(team_angle_rad))

                    # Validate member position is within bounds
                    member_x = int(max(0, min(member_x, env.width - 1)))
                    member_y = int(max(0, min(member_y, env.height - 1)))

                    env.update_unit_position(member, (member_x, member_y))

            # Move squad leader with proper offset from moving team
            if moving_team == alpha_team:
                moving_tl_pos = (new_tl_x, new_tl_y)
                sl_x = moving_tl_pos[0] + int(moving_sl_offset[0] * math.cos(team_angle_rad) -
                                              moving_sl_offset[1] * math.sin(team_angle_rad))
                sl_y = moving_tl_pos[1] + int(moving_sl_offset[0] * math.sin(team_angle_rad) +
                                              moving_sl_offset[1] * math.cos(team_angle_rad))

                # Validate SL position is within bounds
                sl_x = int(max(0, min(sl_x, env.width - 1)))
                sl_y = int(max(0, min(sl_y, env.height - 1)))

                env.update_unit_position(squad_leader, (sl_x, sl_y))

            # Hold overwatch position
            if step == 0:  # Only need to add overwatch position once per step
                overwatch_tl = _get_team_leader(env, overwatch_team)
                overwatch_tl_pos = env.get_unit_position(overwatch_tl)
                overwatch_angle_rad = math.radians(squad_orientation - 90)

                # Move overwatch members to maintain formation
                for member in _get_unit_members(env, overwatch_team):
                    role = US_IN_Role(env.get_unit_property(member, 'role'))
                    if role in overwatch_template:
                        member_offset = overwatch_template[role]
                        member_x = overwatch_tl_pos[0] + int(member_offset[0] * math.cos(overwatch_angle_rad) -
                                                             member_offset[1] * math.sin(overwatch_angle_rad))
                        member_y = overwatch_tl_pos[1] + int(member_offset[0] * math.sin(overwatch_angle_rad) +
                                                             member_offset[1] * math.cos(overwatch_angle_rad))

                        # Validate bounds
                        member_x = int(max(0, min(member_x, env.width - 1)))
                        member_y = int(max(0, min(member_y, env.height - 1)))

                        env.update_unit_position(member, (member_x, member_y))

            # Validate formation after movement
            if debug_level >= 2:
                formation_valid = _validate_template_distances(env, squad_id, debug_level)
                if not formation_valid['squad_valid']:
                    print(f"WARNING: Formation distances invalid during bound {bound_number}, step {step + 1}")

            # Add frame showing both teams
            frames.append(_capture_squad_positions(env, squad_id))

        # Update distance covered
        current_distance += current_bound

        if debug_level >= 1:
            print(f"Bound {bound_number} complete")
            print(f"Distance covered: {current_distance}/{distance}")
            _print_squad_state(env, squad_id)

        # Store final positions for state management
        final_sl_pos = env.get_unit_position(squad_leader)
        final_orientation = env.get_unit_property(squad_id, 'orientation')

        # Update state for next bound
        env.state_manager.update_unit_position(squad_id, final_sl_pos)
        env.state_manager.update_unit_property(squad_id, 'position', final_sl_pos)
        env.state_manager.update_unit_property(squad_id, 'orientation', final_orientation)

        # Swap roles if not finished
        if current_distance < distance:
            moving_team, overwatch_team = overwatch_team, moving_team
            moving_key, overwatch_key = overwatch_key, moving_key
            moving_template, overwatch_template = overwatch_template, moving_template
            moving_team_id, overwatch_team_id = overwatch_team_id, moving_team_id
            bound_number += 1

    return frames


def _get_squad_leader(env: MilitaryEnvironment, squad_id: int) -> Optional[int]:
    """Get squad leader ID."""
    members = env.get_unit_children(squad_id)
    return next((mid for mid in members
                 if env.get_unit_property(mid, 'role') == US_IN_Role.SQUAD_LEADER.value), None)


def _get_team_leader(env: MilitaryEnvironment, team_id: int) -> Optional[int]:
    """Get team leader ID."""
    members = env.get_unit_children(team_id)
    return next((mid for mid in members
                 if env.get_unit_property(mid, 'is_leader')), None)


def _get_alpha_team(env: MilitaryEnvironment, squad_id: int) -> Optional[int]:
    """Get Alpha team ID."""
    teams = [tid for tid in env.get_unit_children(squad_id)
             if env.get_unit_property(tid, 'type') == UnitType.INFANTRY_TEAM]
    return next((tid for tid in teams
                 if US_IN_UnitDesignator.ALPHA_TEAM.value in
                 env.get_unit_property(tid, 'string_id')), None)


def _get_bravo_team(env: MilitaryEnvironment, squad_id: int) -> Optional[int]:
    """Get Bravo team ID."""
    teams = [tid for tid in env.get_unit_children(squad_id)
             if env.get_unit_property(tid, 'type') == UnitType.INFANTRY_TEAM]
    return next((tid for tid in teams
                 if US_IN_UnitDesignator.BRAVO_TEAM.value in
                 env.get_unit_property(tid, 'string_id')), None)


def _capture_squad_positions(env: MilitaryEnvironment, squad_id: int) -> Dict:
    """Capture current positions of all squad elements."""
    squad_string = env.get_unit_property(squad_id, 'string_id')
    positions = []

    # Add squad leader
    sl_id = _get_squad_leader(env, squad_id)
    if sl_id:
        positions.append({
            'role': env.get_unit_property(sl_id, 'role'),
            'position': env.get_unit_position(sl_id),
            'is_leader': True
        })

    # Add team members
    for team_id in [_get_alpha_team(env, squad_id), _get_bravo_team(env, squad_id)]:
        if team_id:
            team_members = env.get_unit_children(team_id)
            for member_id in team_members:
                positions.append({
                    'role': env.get_unit_property(member_id, 'role'),
                    'position': env.get_unit_position(member_id),
                    'is_leader': env.get_unit_property(member_id, 'is_leader', False)
                })

    return {
        'unit_type': 'Squad',
        'unit_id': squad_string,
        'positions': positions
    }


def _validate_template_distances(env: MilitaryEnvironment, squad_id: int,
                                 debug_level: int = 1) -> Dict[str, bool]:
    """
    Validate actual distances against template distances at squad and team levels.

    Args:
        env: Military environment instance
        squad_id: ID of squad to validate
        debug_level: Level of debug output

    Returns:
        Dictionary with validation results and template keys for each level
    """
    validation_results = {
        'squad_valid': True,
        'alpha_valid': True,
        'bravo_valid': True,
        'alpha_key': None,
        'bravo_key': None
    }

    if debug_level >= 1:
        print("\n=== Template Distance Validation ===")

    # Get squad components
    squad_leader = _get_squad_leader(env, squad_id)
    alpha_team = _get_alpha_team(env, squad_id)
    bravo_team = _get_bravo_team(env, squad_id)

    # Get squad template and current positions
    squad_template = env.get_unit_property(squad_id, 'formation_template')
    sl_pos = env.get_unit_position(squad_leader)

    # Validate squad-level distances and get template keys
    for team_id, team_string in [(alpha_team, 'ATM'), (bravo_team, 'BTM')]:
        # Find matching template key
        template_key = None
        team_id_str = env.get_unit_property(team_id, 'string_id')

        if team_string in team_id_str:
            template_key = team_string
            if team_string == 'ATM':
                validation_results['alpha_key'] = template_key
            else:
                validation_results['bravo_key'] = template_key

        if template_key and template_key in squad_template:
            # Get template offset
            template_offset = squad_template[template_key]
            template_distance = math.sqrt(template_offset[0] ** 2 + template_offset[1] ** 2)

            # Get actual distance
            team_leader = _get_team_leader(env, team_id)
            tl_pos = env.get_unit_position(team_leader)
            actual_distance = _calculate_distance(sl_pos, tl_pos)

            # Compare distances with tolerance
            tolerance = 1.5  # Allow for some rounding error
            distance_valid = abs(template_distance - actual_distance) <= tolerance

            if debug_level >= 1:
                print(f"\n{team_string}:")
                print(f"  Template key: {template_key}")
                print(f"  Template distance: {template_distance:.1f}")
                print(f"  Actual distance: {actual_distance:.1f}")
                print(f"  Valid: {distance_valid}")

            # Update validation results
            if team_string == 'ATM':
                validation_results['alpha_valid'] = distance_valid
            else:
                validation_results['bravo_valid'] = distance_valid

            validation_results['squad_valid'] &= distance_valid

    return validation_results


def _calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate distance between two points."""
    return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)


# Orders
class OrderType(Enum):
    """Types of orders."""
    OPORD = auto()  # Operation Order
    FRAGO = auto()  # Fragmentary Order
    WARNO = auto()  # Warning Order


class ReportType(Enum):
    """Types of tactical reports."""
    SITREP = auto()  # Situation Report
    SPOTREP = auto()  # Spot Report (Enemy Contact)
    ACK = auto()  # Acknowledgement


@dataclass
class TacticalOrder:
    """Base class for tactical orders."""
    # Required fields (non-default) come first
    order_id: str
    order_type: OrderType
    issuing_unit: int
    receiving_units: Set[int]
    datetime_issued: datetime
    mission_type: str

    # Make original_order_id optional with None default
    original_order_id: Optional[str] = None

    # Default fields come last
    situation: Dict[str, Any] = field(default_factory=dict)
    mission: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    sustainment: Dict[str, Any] = field(default_factory=dict)
    command_signal: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MovementOrder(TacticalOrder):
    """Specific order for movement operations."""
    # Only adding fields with defaults
    routes: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    coordination_points: Dict[Tuple[int, int], Dict[str, Any]] = field(default_factory=dict)

    def get_unit_route(self, unit_id: int) -> Optional[List[Dict[str, Any]]]:
        """Get route for specific unit."""
        return self.routes.get(unit_id)

    def get_coordination_requirements(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """Get coordination requirements at position."""
        return self.coordination_points.get(position, {})


@dataclass
class FragmentaryOrder(TacticalOrder):
    """Order modifying existing operation."""
    # Override original_order_id to make it required for FRAGOs
    original_order_id: str
    # Additional fields with defaults
    changes: Dict[str, Any] = field(default_factory=dict)

    def get_route_changes(self, unit_id: int) -> Optional[List[Dict[str, Any]]]:
        """Get route changes for unit."""
        route_changes = self.changes.get('routes', {})
        return route_changes.get(unit_id)


@dataclass
class TacticalReport:
    """Base class for tactical reports."""
    report_id: str
    report_type: ReportType
    reporting_unit: int
    datetime_reported: datetime
    position: Tuple[int, int]
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SituationReport(TacticalReport):
    """Detailed situation report."""
    unit_status: Dict[str, Any] = field(default_factory=dict)
    coordination_status: Dict[str, bool] = field(default_factory=dict)
    conditions_status: Dict[str, bool] = field(default_factory=dict)


class OrderHandler:
    """Manages order dissemination and execution."""

    def __init__(self, env):
        self.env = env
        self.active_orders: Dict[str, TacticalOrder] = {}
        self.pending_reports: Dict[str, TacticalReport] = {}
        self.order_counter = 0
        self.report_counter = 0

    def create_movement_order(self,
                              issuing_unit: int,
                              receiving_units: Set[int],
                              routes: Dict[int, List[Dict[str, Any]]],
                              coordination_points: Dict[Tuple[int, int], Dict[str, Any]]
                              ) -> MovementOrder:
        """Create new movement order."""
        order_id = f"OPORD_{self.order_counter}"
        self.order_counter += 1

        order = MovementOrder(
            order_id=order_id,
            order_type=OrderType.OPORD,
            issuing_unit=issuing_unit,
            receiving_units=receiving_units,
            datetime_issued=datetime.now(),
            mission_type="Movement",
            routes=routes,
            coordination_points=coordination_points
        )

        self.active_orders[order_id] = order
        return order

    def create_fragmentary_order(self,
                                 original_order: MovementOrder,
                                 changes: Dict[str, Any]
                                 ) -> FragmentaryOrder:
        """Create FRAGO modifying existing order."""
        order_id = f"FRAGO_{self.order_counter}"
        self.order_counter += 1

        frago = FragmentaryOrder(
            order_id=order_id,
            order_type=OrderType.FRAGO,
            issuing_unit=original_order.issuing_unit,
            receiving_units=original_order.receiving_units,
            datetime_issued=datetime.now(),
            mission_type=original_order.mission_type,
            original_order_id=original_order.order_id,
            changes=changes
        )

        self.active_orders[order_id] = frago
        return frago

    def handle_situation_report(self,
                                unit_id: int,
                                position: Tuple[int, int]
                                ) -> SituationReport:
        """Process situation report from unit."""
        report_id = f"SITREP_{self.report_counter}"
        self.report_counter += 1

        # Get unit state from environment
        unit_state = self.env.state_manager.get_unit_coordination_state(unit_id)
        coord_state = self.env.state_manager.get_coordination_state(position)

        sitrep = SituationReport(
            report_id=report_id,
            report_type=ReportType.SITREP,
            reporting_unit=unit_id,
            datetime_reported=datetime.now(),
            position=position,
            summary=f"Unit {unit_id} reporting from {position}",
            unit_status=unit_state,
            coordination_status=coord_state.get('all_units_present', False),
            conditions_status=coord_state.get('conditions_met', {})
        )

        self.pending_reports[report_id] = sitrep
        return sitrep


class MovementExecutor:
    """Handles doctrinal movement execution."""

    def __init__(self, env):
        self.env = env
        self.order_handler = OrderHandler(env)
        self.active_movements: Dict[int, Dict[str, Any]] = {}

    def start_movement(self,
                       unit_id: int,
                       order: MovementOrder
                       ) -> bool:
        """Start unit movement under order."""
        route = order.get_unit_route(unit_id)
        if not route:
            return False

        # Initialize movement tracking
        self.active_movements[unit_id] = {
            'order_id': order.order_id,
            'current_segment': 0,
            'paused_for_coordination': False,
            'route': route
        }

        # Update unit state
        self.env.state_manager.update_unit_property(
            unit_id, 'current_route_segment', 0)
        self.env.state_manager.update_unit_property(
            unit_id, 'moving', True)

        return True

    def handle_coordination_point(self,
                                  unit_id: int,
                                  position: Tuple[int, int]
                                  ) -> Optional[SituationReport]:
        """Handle unit reaching coordination point."""
        if unit_id not in self.active_movements:
            return None

        # Update unit at coordination point
        coordination_complete = self.env.state_manager.update_unit_at_coordination(
            unit_id, position, True)

        # Get coordination requirements
        coord_state = self.env.state_manager.get_coordination_state(position)
        if not coord_state:
            return None

        # Pause movement if awaiting approval
        if coord_state['level'] != CoordinationLevel.UNIT:
            self.active_movements[unit_id]['paused_for_coordination'] = True

            # Generate situation report for higher approval
            return self.order_handler.handle_situation_report(unit_id, position)

        # For unit-level coordination, continue if complete
        if coordination_complete:
            self._continue_movement(unit_id, position)

        return None

    def approve_continuation(self,
                             position: Tuple[int, int],
                             changes: Optional[Dict[str, Any]] = None
                             ) -> Optional[FragmentaryOrder]:
        """Approve units to continue past coordination point."""
        coord_state = self.env.state_manager.get_coordination_state(position)
        if not coord_state:
            return None

        # Get current order for units
        order_id = None
        affected_units = set()
        for unit_id in coord_state['completed_units']:
            if unit_id in self.active_movements:
                order_id = self.active_movements[unit_id]['order_id']
                affected_units.add(unit_id)

        if not order_id or not affected_units:
            return None

        # Get original order
        original_order = self.order_handler.active_orders.get(order_id)
        if not original_order:
            return None

        # Create FRAGO if changes provided
        frago = None
        if changes:
            frago = self.order_handler.create_fragmentary_order(
                original_order, changes)

        # Clear coordination point and continue movement
        self.env.state_manager.clear_coordination_point(position)

        for unit_id in affected_units:
            self._continue_movement(unit_id, position)

        return frago

    def _continue_movement(self, unit_id: int, position: Tuple[int, int]):
        """Continue unit movement after coordination."""
        if unit_id not in self.active_movements:
            return

        movement_state = self.active_movements[unit_id]

        # Increment segment and unpause
        movement_state['current_segment'] += 1
        movement_state['paused_for_coordination'] = False

        # Update unit state
        self.env.state_manager.update_unit_property(
            unit_id, 'current_route_segment',
            movement_state['current_segment'])


def create_movement_order_from_routes(
        env,
        issuing_unit: int,
        unit_routes: Dict[int, List[Dict[str, Any]]],
        coordination_points: Dict[Tuple[int, int], Dict[str, Any]]) -> MovementOrder:
    """Create movement order from route analyzer output."""
    # Create order handler if needed
    if not hasattr(env, 'order_handler'):
        env.order_handler = OrderHandler(env)

    # Get receiving units from routes
    receiving_units = set(unit_routes.keys())

    # Create order
    order = env.order_handler.create_movement_order(
        issuing_unit=issuing_unit,
        receiving_units=receiving_units,
        routes=unit_routes,
        coordination_points=coordination_points
    )

    # Register coordination points with state manager
    for pos, coord_info in coordination_points.items():
        coord_point = CoordinationPoint(
            position=pos,
            coord_type=coord_info['type'],
            required_units=coord_info['required_units'],
            level=coord_info['level'],
            conditions=coord_info['conditions'],
            priority=coord_info['priority']
        )
        env.state_manager.register_coordination_point(coord_point)

    return order
