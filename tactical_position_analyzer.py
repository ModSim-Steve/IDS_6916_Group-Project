"""
Tactical Position Filter

Purpose:
    Filters the operational environment to identify tactically suitable positions
    based on unit type, mission role, and threat considerations.

Key Components:
    - Position requirements based on unit type (team/squad) and role (assault/support/reserve)
    - Threat-based filtering using enemy capabilities
    - Terrain analysis aligned with movement requirements
    - Integration with US Army unit composition and environment state management
"""
from __future__ import annotations

import traceback

import numpy as np
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

from Russian_AF_ASLT_DET_Cap_SQD import Squad

from US_Army_PLT_Composition_vTest import (
    # US_IN_Role,
    # US_IN_TeamType,
    # US_IN_UnitDesignator,
    UnitType
)

from WarGamingEnvironment_vTest import (
    MilitaryEnvironment,
    EnvironmentConfig,
    TerrainType,
    ElevationType, BaseWeapon, VisibilityManager,
    # ForceType
)


class UnitSize(Enum):
    """Size of unit for position requirements."""
    TEAM = "TEAM"  # Fire team or weapons team
    SQUAD = "SQUAD"  # Full squad


# Tactical Position data structures
@dataclass
class TacticalPosition:
    """Position that meets tactical requirements."""
    # Basic position information
    position: Tuple[int, int]  # Center position
    cells: List[Tuple[int, int]]  # All cells in position area
    position_type: PositionPurpose  # Purpose of position (assault, support, etc)
    unit_size: UnitSize  # Size of unit position is for (team/squad)

    # Engagement characteristics
    coverage_arc: Tuple[float, float]  # Main engagement arc in degrees
    max_range: int  # Maximum engagement range in meters
    covered_threats: List[Dict]  # List of threats that can be engaged
    engagement_quality: float  # Quality of engagement capability (0-1)

    # Cover and concealment
    cover_score: float  # Average cover value (0-1)
    concealment_score: float  # Average concealment value (0-1)

    # Movement characteristics
    movement_score: float  # Ease of movement in/out of position (0-1)
    approach_routes: List[Tuple[int, int]]  # Possible routes to position
    withdrawal_routes: List[Tuple[int, int]]  # Disengagement routes
    has_elevation: bool  # Whether position has elevation advantage

    # Observation
    observation_arc: Tuple[float, float]  # Main observation arc in degrees
    observation_range: int  # Maximum observation range in meters
    observation_quality: float  # Quality of observation capability (0-1)

    # Threat exposure
    threat_exposure: float  # Level of exposure to enemy (0-1)
    primary_threats: List[Tuple[int, int]]  # Most dangerous enemy positions

    # Support
    supporting_positions: List[Tuple[int, int]]  # Positions that can provide support
    requires_support: bool  # Whether position needs supporting positions
    mutual_support_positions: List[Tuple[int, int]]  # Positions for mutual support

    # Overall quality
    quality_score: float  # Overall position quality (0-1)
    quality_breakdown: Dict[str, float]  # Detailed quality scores by category

    @property
    def engagement_score(self) -> float:
        """Get overall engagement effectiveness score."""
        return self.engagement_quality

    @property
    def primary_arc(self) -> Tuple[float, float]:
        """Get primary engagement arc."""
        return self.coverage_arc

    def __post_init__(self):
        """Validate position after initialization."""
        if not self.cells:
            raise ValueError("Position must have at least one cell")

        if not 0 <= self.quality_score <= 1:
            raise ValueError("Quality score must be between 0 and 1")

        if not isinstance(self.coverage_arc, tuple) or len(self.coverage_arc) != 2:
            raise ValueError("Coverage arc must be tuple of (start_angle, end_angle)")

        if not isinstance(self.covered_threats, list):
            raise ValueError("Covered threats must be a list")

        if not 0 <= self.cover_score <= 1:
            raise ValueError("Cover score must be between 0 and 1")

        if not 0 <= self.concealment_score <= 1:
            raise ValueError("Concealment score must be between 0 and 1")

        if not 0 <= self.movement_score <= 1:
            raise ValueError("Movement score must be between 0 and 1")

        if not 0 <= self.observation_quality <= 1:
            raise ValueError("Observation quality must be between 0 and 1")

        if not 0 <= self.threat_exposure <= 1:
            raise ValueError("Threat exposure must be between 0 and 1")


@dataclass
class TerrainRequirements:
    """Physical terrain requirements for a tactical position."""
    min_cover: float  # Minimum cover value (0-1)
    min_concealment: float  # Minimum concealment value (0-1)
    max_movement_cost: float  # Maximum allowed movement difficulty
    allowed_terrain_types: List[TerrainType]  # Terrain types that can be used
    preferred_elevation: Optional[ElevationType] = None  # Best elevation type (not required)
    requires_elevation_advantage: bool = False  # Whether elevation advantage is required


@dataclass
class ThreatRequirements:
    """Requirements related to enemy threat exposure and handling."""
    max_threat_exposure: float  # Maximum acceptable threat level (0-1)
    min_covered_threats: int  # Minimum number of threats position must control
    max_observable_threats: int  # Maximum threats that can observe position
    requires_fallback_position: bool  # Whether position needs a backup position
    max_flanking_exposure: float  # Maximum exposure to flanking fire (0-1)


@dataclass
class EngagementRequirements:
    """Requirements for engaging enemy forces from position."""
    min_range: int  # Minimum range to target in cells (10m per cell)
    max_range: int  # Maximum range to target
    min_engagement_quality: float  # Minimum quality score (0-1)
    required_arc_width: float  # Required fire arc in degrees
    requires_mutual_support: bool  # Whether position needs supporting position
    min_covered_arc: float  # Minimum arc that must be covered


@dataclass
class MovementRequirements:
    """Requirements for movement to and from position."""
    min_approach_routes: int  # Minimum number of routes to position
    min_withdrawal_routes: int  # Minimum number of exit routes
    max_route_exposure: float  # Maximum exposure during movement (0-1)
    requires_covered_approach: bool  # Whether approach must be covered
    requires_defilade_movement: bool  # Whether defilade movement required


class PositionPurpose(Enum):
    """Tactical purposes for positions with integrated requirements and scoring."""
    ASSAULT = auto()
    SUPPORT = auto()
    RESERVE = auto()

    def get_requirements(self) -> Dict:
        """
        Get complete set of requirements for position purpose.
        Includes both minimum requirements and terrain requirements.
        """
        # Get base minimum requirements
        min_reqs = self._get_minimum_requirements()

        # Define terrain requirements based on purpose
        if self == PositionPurpose.SUPPORT:
            terrain_reqs = TerrainRequirements(
                min_cover=min_reqs['min_cover'],
                min_concealment=min_reqs['min_concealment'],
                max_movement_cost=2.0,
                allowed_terrain_types=[
                    TerrainType.SPARSE_VEG,
                    TerrainType.DENSE_VEG,
                    TerrainType.WOODS
                ],
                preferred_elevation=ElevationType.ELEVATED_LEVEL,
                requires_elevation_advantage=False  # Changed to bonus rather than requirement
            )

            engagement_reqs = EngagementRequirements(
                min_range=40,  # 400m minimum
                max_range=80,  # 800m maximum
                min_engagement_quality=0.3,  # Moderate requirement
                required_arc_width=60.0,  # Standard arc
                requires_mutual_support=False,  # Changed to bonus
                min_covered_arc=45.0  # Minimum arc
            )

        elif self == PositionPurpose.ASSAULT:
            terrain_reqs = TerrainRequirements(
                min_cover=min_reqs['min_cover'],
                min_concealment=min_reqs['min_concealment'],
                max_movement_cost=2.0,
                allowed_terrain_types=[
                    TerrainType.DENSE_VEG,
                    TerrainType.WOODS,
                    TerrainType.STRUCTURE
                ],
                preferred_elevation=None,
                requires_elevation_advantage=False
            )

            engagement_reqs = EngagementRequirements(
                min_range=0,  # Close combat
                max_range=30,  # 300m max
                min_engagement_quality=0.2,  # Lower requirement
                required_arc_width=90.0,  # Wide arc needed
                requires_mutual_support=False,  # Changed to bonus
                min_covered_arc=60.0  # Minimum arc
            )

        else:  # RESERVE
            terrain_reqs = TerrainRequirements(
                min_cover=min_reqs['min_cover'],
                min_concealment=min_reqs['min_concealment'],
                max_movement_cost=2.5,
                allowed_terrain_types=[
                    TerrainType.SPARSE_VEG,
                    TerrainType.DENSE_VEG,
                    TerrainType.WOODS,
                    TerrainType.STRUCTURE
                ],
                preferred_elevation=None,
                requires_elevation_advantage=False
            )

            engagement_reqs = EngagementRequirements(
                min_range=50,  # 500m minimum
                max_range=100,  # 1000m maximum
                min_engagement_quality=0.2,  # Lower requirement
                required_arc_width=60.0,  # Standard arc
                requires_mutual_support=False,  # Not required
                min_covered_arc=45.0  # Minimum arc
            )

        # Create threat requirements
        threat_reqs = ThreatRequirements(
            max_threat_exposure=min_reqs['max_threat'],
            min_covered_threats=min_reqs['min_threats'],
            max_observable_threats=3,
            requires_fallback_position=True,
            max_flanking_exposure=0.5
        )

        # Create movement requirements - now used for scoring rather than hard requirements
        movement_reqs = MovementRequirements(
            min_approach_routes=1,
            min_withdrawal_routes=1,
            max_route_exposure=0.8,  # More lenient threshold
            requires_covered_approach=self == PositionPurpose.ASSAULT,
            requires_defilade_movement=self == PositionPurpose.ASSAULT
        )

        return {
            'terrain': terrain_reqs,
            'threat': threat_reqs,
            'engagement': engagement_reqs,
            'movement': movement_reqs,
            'min_requirements': min_reqs
        }

    def _get_minimum_requirements(self) -> Dict[str, float]:
        """Get minimum requirements based on position purpose."""
        if self == PositionPurpose.SUPPORT:
            return {
                'min_cover': 0.3,
                'min_concealment': 0.2,
                'min_threats': 1,
                'min_engagement': 0.3,
                'max_threat': 0.8,
                'min_score': 0.4
            }
        elif self == PositionPurpose.ASSAULT:
            return {
                'min_cover': 0.4,
                'min_concealment': 0.3,
                'min_threats': 1,
                'min_engagement': 0.2,
                'max_threat': 0.7,
                'min_score': 0.5
            }
        else:  # RESERVE
            return {
                'min_cover': 0.3,
                'min_concealment': 0.3,
                'min_threats': 1,
                'min_engagement': 0.2,
                'max_threat': 0.6,
                'min_score': 0.4
            }

    def validate_position_requirements(self, position: TacticalPosition,
                                       env_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate position with purpose-specific criteria.
        Routes and elevation are treated as score modifiers rather than requirements.
        """
        failures = []

        # Core Requirements (must pass these)
        min_requirements = self._get_minimum_requirements()

        # 1. Basic position characteristics
        if position.cover_score < min_requirements['min_cover']:
            failures.append(f"Insufficient cover: {position.cover_score:.2f} < {min_requirements['min_cover']}")

        if position.concealment_score < min_requirements['min_concealment']:
            failures.append(
                f"Insufficient concealment: {position.concealment_score:.2f} < {min_requirements['min_concealment']}")

        # 2. Engagement capabilities
        if len(position.covered_threats) < min_requirements['min_threats']:
            failures.append(
                f"Too few covered threats: {len(position.covered_threats)} < {min_requirements['min_threats']}")

        if position.engagement_quality < min_requirements['min_engagement']:
            failures.append(
                f"Poor engagement quality: {position.engagement_quality:.2f} < {min_requirements['min_engagement']}")

        # 3. Threat exposure
        if position.threat_exposure > min_requirements['max_threat']:
            failures.append(
                f"Excessive threat exposure: {position.threat_exposure:.2f} > {min_requirements['max_threat']}")

        # Calculate scores
        base_score = self._calculate_base_position_score(position)
        route_score = self._calculate_route_score(position)
        final_score = base_score * (1.0 + route_score)  # Routes can boost score up to 20%

        # Update position's quality score
        position.quality_score = final_score

        # Position passes if it meets core requirements and has acceptable final score
        passes = len(failures) == 0 and final_score >= min_requirements['min_score']

        return passes, failures

    def _calculate_base_position_score(self, position: TacticalPosition) -> float:
        """Calculate base position score without route considerations."""
        if self == PositionPurpose.SUPPORT:
            score = (
                    position.cover_score * 0.25 +  # Cover from fire
                    position.concealment_score * 0.15 +  # Concealment from observation
                    (1.0 - position.threat_exposure) * 0.2 +  # Protection from threats
                    position.engagement_quality * 0.3 +  # Ability to engage
                    (len(position.covered_threats) / max(1, len(position.primary_threats))) * 0.1  # Target coverage
            )
        elif self == PositionPurpose.ASSAULT:
            score = (
                    position.cover_score * 0.3 +  # More emphasis on cover
                    position.concealment_score * 0.25 +  # More emphasis on concealment
                    (1.0 - position.threat_exposure) * 0.25 +  # More emphasis on protection
                    position.engagement_quality * 0.1 +  # Less emphasis on engagement
                    position.movement_score * 0.1  # Some emphasis on movement
            )
        else:  # RESERVE
            score = (
                    position.cover_score * 0.25 +  # Balanced cover
                    position.concealment_score * 0.25 +  # Strong concealment
                    (1.0 - position.threat_exposure) * 0.3 +  # Strong protection
                    position.engagement_quality * 0.1 +  # Minimal engagement
                    position.movement_score * 0.1  # Some movement
            )

        # Elevation bonus
        if position.has_elevation:
            score *= 1.1  # 10% bonus for elevation advantage

        return score

    def _calculate_route_score(self, position: TacticalPosition) -> float:
        """
        Calculate route quality modifier.
        Returns value from 0.0 to 0.2 (can boost score up to 20%)
        """
        route_score = 0.0

        # Approach routes bonus (up to 10%)
        if len(position.approach_routes) > 0:
            route_score += 0.1 * min(1.0, len(position.approach_routes) / 2)

        # Withdrawal routes bonus (up to 10%)
        if len(position.withdrawal_routes) > 0:
            route_score += 0.1 * min(1.0, len(position.withdrawal_routes) / 2)

        return route_score


# Threat data structures
@dataclass
class Threat:
    """Represents an enemy unit with its threat characteristics using VisibilityManager."""
    position: Tuple[int, int]  # (x, y) coordinates
    unit: Optional[Squad]  # Now properly typed
    observation_range: int  # Range in cells
    engagement_range: int  # Range in cells
    suspected_accuracy: float  # 0.0 to 1.0

    def evaluate_threat_to_position(self, env: MilitaryEnvironment, position: Tuple[int, int]) -> Dict:
        """
        Evaluate threat to a specific position using VisibilityManager.

        Args:
            env: Military environment reference
            position: Position to evaluate

        Returns:
            Dictionary containing:
            - can_observe: Whether position can be observed
            - can_engage: Whether position can be engaged
            - observation_quality: Quality of observation (0-1)
            - engagement_quality: Quality of engagement (0-1)
            - total_threat: Combined threat score (0-1)
        """
        # Create visibility manager instance for this evaluation
        visibility_mgr = VisibilityManager(env)

        # Get line of sight check
        los_result = visibility_mgr.check_line_of_sight(
            self.position,
            position
        )

        if not los_result['has_los']:
            return {
                'can_observe': False,
                'can_engage': False,
                'observation_quality': 0.0,
                'engagement_quality': 0.0,
                'total_threat': 0.0
            }

        # Calculate distance
        distance = math.sqrt(
            (position[0] - self.position[0]) ** 2 +
            (position[1] - self.position[1]) ** 2
        )

        # Check observation capability
        can_observe = distance <= self.observation_range
        observation_quality = 0.0
        if can_observe:
            # Get observation quality factoring in LOS and distance
            observation_quality = los_result['los_quality'] * (
                    1.0 - (distance / self.observation_range)
            )

        # Check engagement capability
        can_engage = distance <= self.engagement_range
        engagement_quality = 0.0
        if can_engage:
            # Get base weapon characteristics
            weapon = BaseWeapon("Generic", self.engagement_range, 30, 1, 40)

            # Calculate hit probability using public method
            base_hit_prob = env.calculate_hit_probability(distance, weapon)
            test_unit_id = 1  # Temporary ID for threat unit

            # Get modified hit probability using visibility manager
            engagement_quality = visibility_mgr.modify_hit_probability(
                base_hit_prob,
                self.position,
                position,
                test_unit_id
            )

            # Factor in suspected accuracy
            engagement_quality *= self.suspected_accuracy

        # Calculate total threat score
        # Weight engagement higher than observation
        total_threat = max(
            engagement_quality * 0.7,
            observation_quality * 0.3
        )

        return {
            'can_observe': can_observe,
            'can_engage': can_engage,
            'observation_quality': observation_quality,
            'engagement_quality': engagement_quality,
            'total_threat': total_threat
        }


class ThreatAnalysis:
    """Enhanced threat analysis using VisibilityManager."""

    def __init__(self, env: MilitaryEnvironment):
        self.env = env
        self.threats: List[Threat] = []
        self.width = env.width
        self.height = env.height

        # Initialize matrices for threat analysis
        self.observation_matrix = np.zeros((self.height, self.width))
        self.engagement_matrix = np.zeros((self.height, self.width))
        self.total_threat_matrix = np.zeros((self.height, self.width))

    def add_threat(self, threat: Threat):
        """Add threat and update threat matrices."""
        self.threats.append(threat)
        self._update_threat_matrices(threat)

    def _update_threat_matrices(self, threat: Threat):
        """Update threat matrices for new threat using VisibilityManager."""
        # For each cell in environment
        for y in range(self.height):
            for x in range(self.width):
                position = (x, y)

                # Get threat evaluation
                threat_eval = threat.evaluate_threat_to_position(self.env, position)

                # Update matrices with maximum values
                if threat_eval['can_observe']:
                    self.observation_matrix[y, x] = np.maximum(
                        self.observation_matrix[y, x],
                        threat_eval['observation_quality']
                    )

                if threat_eval['can_engage']:
                    self.engagement_matrix[y, x] = np.maximum(
                        self.engagement_matrix[y, x],
                        threat_eval['engagement_quality']
                    )

                self.total_threat_matrix[y, x] = np.maximum(
                    self.total_threat_matrix[y, x],
                    threat_eval['total_threat']
                )

    def analyze_position(self, position: Tuple[int, int]) -> Dict:
        """
        Analyze threats to a specific position.

        Returns:
            Dictionary containing:
            - total_threat: Combined threat level
            - observation_threat: Observation exposure
            - engagement_threat: Engagement exposure
            - threatening_enemies: List of threats affecting position
            - primary_threat: Most dangerous threat position
        """
        threats_affecting = []
        max_threat = 0.0
        primary_threat = None

        for threat in self.threats:
            threat_eval = threat.evaluate_threat_to_position(self.env, position)

            if threat_eval['total_threat'] > 0:
                threats_affecting.append({
                    'position': threat.position,
                    'threat_level': threat_eval
                })

                if threat_eval['total_threat'] > max_threat:
                    max_threat = threat_eval['total_threat']
                    primary_threat = threat.position

        return {
            'total_threat': self.total_threat_matrix[position[1], position[0]],
            'observation_threat': self.observation_matrix[position[1], position[0]],
            'engagement_threat': self.engagement_matrix[position[1], position[0]],
            'threatening_enemies': threats_affecting,
            'primary_threat': primary_threat
        }


# Direct Fire engagement data structures
@dataclass
class UnitFireCapabilities:
    """Defines unit's direct fire capabilities."""
    observation_range: int  # Cells (10m per cell)
    engagement_range: int  # Maximum effective range
    primary_weapon_type: str  # For weapon-specific effects
    min_range: Optional[int] = None  # Minimum engagement range if applicable


class DirectFireAnalysis:
    """Enhanced direct fire analysis using VisibilityManager."""

    def __init__(self, env: MilitaryEnvironment):
        self.env = env
        self.width = env.width
        self.height = env.height

    def analyze_position(self,
                         position: Tuple[int, int],
                         capabilities: UnitFireCapabilities,
                         targets: List[Tuple[int, int]],
                         position_purpose: PositionPurpose) -> Dict:
        """
        Enhanced direct fire analysis with improved engagement calculations.
        """
        # Create visibility manager instance
        visibility_mgr = VisibilityManager(self.env)

        # Create weapon based on capabilities
        weapon = BaseWeapon(
            capabilities.primary_weapon_type,
            capabilities.engagement_range,
            100,  # Default ammo
            1,  # Default fire rate
            40  # Default damage
        )

        covered_targets = []
        observation_arc = {}
        engagement_arc = {}

        # Get base position quality scores
        cover_score = visibility_mgr.get_cover_bonus(position)
        test_unit_id = 1
        concealment_score = visibility_mgr.get_concealment_bonus(position, test_unit_id)

        # Calculate movement score
        movement_cost = self.env.terrain_manager.get_movement_cost(position)
        movement_score = 1.0 - (movement_cost / 3.0)  # Normalize to 0-1 range

        # Initialize engagement tracking
        total_engagement_quality = 0.0
        total_los_quality = 0.0
        valid_targets = 0
        max_hit_prob = 0.0  # Track the best engagement opportunity

        # Analyze each target
        for target in targets:
            # Check line of sight
            los_result = visibility_mgr.check_line_of_sight(
                position,
                target,
                for_observation=False  # Use engagement parameters
            )

            if los_result['has_los']:
                # Calculate distance
                distance = math.sqrt(
                    (target[0] - position[0]) ** 2 +
                    (target[1] - position[1]) ** 2
                )

                # Record observation quality
                observation_quality = los_result['los_quality']
                observation_arc[target] = observation_quality
                total_los_quality += observation_quality

                # Check engagement capability
                if distance <= capabilities.engagement_range:
                    # Calculate engagement effectiveness
                    base_hit_prob = self.env.calculate_hit_probability(distance, weapon)

                    # Get modified hit probability considering terrain and visibility
                    hit_prob = visibility_mgr.modify_hit_probability(
                        base_hit_prob,
                        position,
                        target,
                        test_unit_id
                    )

                    # Calculate potential damage
                    base_damage = self.env.calculate_damage(distance, weapon)
                    damage_potential = visibility_mgr.modify_damage(
                        base_damage,
                        target,
                        position
                    )

                    # Update quality tracking
                    engagement_quality = hit_prob * (los_result['los_quality'] + 0.5)  # Weight LOS quality
                    total_engagement_quality += engagement_quality
                    max_hit_prob = max(max_hit_prob, hit_prob)
                    valid_targets += 1

                    # Record engagement arc
                    engagement_arc[target] = engagement_quality

                    # Add detailed target data
                    covered_targets.append({
                        'position': target,
                        'distance': distance * 10,  # Convert to meters
                        'hit_probability': hit_prob,
                        'damage_potential': damage_potential,
                        'los_quality': los_result['los_quality'],
                        'engagement_quality': engagement_quality
                    })

                    if self.env.debug_level > 1:
                        print(f"\nTarget at {target}:")
                        print(f"Distance: {distance * 10:.1f}m")
                        print(f"Base Hit Prob: {base_hit_prob:.2f}")
                        print(f"Modified Hit Prob: {hit_prob:.2f}")
                        print(f"LOS Quality: {los_result['los_quality']:.2f}")
                        print(f"Engagement Quality: {engagement_quality:.2f}")

        # Calculate coverage and quality scores
        num_targets = len(targets)
        if num_targets > 0:
            coverage_score = len(covered_targets) / num_targets
        else:
            coverage_score = 0.0

        if valid_targets > 0:
            avg_engagement_quality = total_engagement_quality / valid_targets
            avg_los_quality = total_los_quality / num_targets
        else:
            avg_engagement_quality = 0.0
            avg_los_quality = 0.0

        # Calculate position quality based on purpose
        if position_purpose == PositionPurpose.SUPPORT:
            position_score = (
                    cover_score * 0.3 +
                    concealment_score * 0.2 +
                    movement_score * 0.1 +
                    max_hit_prob * 0.4  # Use the best engagement opportunity
            )
        else:  # ASSAULT
            position_score = (
                    cover_score * 0.4 +
                    concealment_score * 0.3 +
                    movement_score * 0.2 +
                    max_hit_prob * 0.1
            )

        # Calculate overall quality with engagement emphasis
        overall_quality = (
                position_score * 0.4 +
                coverage_score * 0.3 +
                avg_engagement_quality * 0.3
        )

        # Compile quality scores
        quality_scores = {
            'overall_quality': overall_quality,
            'position_score': position_score,
            'coverage_score': coverage_score,
            'cover_score': cover_score,
            'concealment_score': concealment_score,
            'movement_score': movement_score,
            'engagement_quality': avg_engagement_quality,
            'los_quality': avg_los_quality,
            'max_hit_prob': max_hit_prob
        }

        # Check if position meets minimum requirements
        meets_requirements = (
                overall_quality >= 0.5 and  # Minimum overall quality
                coverage_score > 0.0 and  # Must cover at least one target
                max_hit_prob >= 0.3  # Must have reasonable engagement capability
        )

        if self.env.debug_level > 1:
            print(f"\nPosition {position} Analysis:")
            print(f"Overall Quality: {overall_quality:.2f}")
            print(f"Coverage Score: {coverage_score:.2f}")
            print(f"Engagement Quality: {avg_engagement_quality:.2f}")
            print(f"Max Hit Probability: {max_hit_prob:.2f}")
            print(f"Meets Requirements: {meets_requirements}")

        return {
            'covered_threats': covered_targets if 'covered_targets' in locals() else [],
            'quality_scores': quality_scores,
            'observation_arc': observation_arc if 'observation_arc' in locals() else {},
            'engagement_arc': engagement_arc if 'engagement_arc' in locals() else {},
            'has_elevation': self.env.terrain_manager.get_elevation_type(position) == ElevationType.ELEVATED_LEVEL,
            'meets_requirements': meets_requirements if 'meets_requirements' in locals() else False
        }


# Main tactical position identifier
class TacticalFilter:
    """
    Progressive filter for finding tactical positions. Uses kernel-based filtering and detailed analysis.

    Filtering Process:
    1. Range-based area reduction from objective
    2. Spatial filtering using unit-size kernel
    3. Pooled analysis using ThreatAnalysis and DirectFireAnalysis
    4. Position scoring with purpose-specific weights
    """

    def __init__(self, env: MilitaryEnvironment):
        """Initialize filter with environment reference."""
        self.env = env
        self.width = env.width
        self.height = env.height
        self.objective = None

        # Initialize analysis components
        self.threat_analyzer = ThreatAnalysis(env)
        self.direct_fire_analyzer = DirectFireAnalysis(env)

        # Unit spacing requirements (in cells)
        self.unit_spacing = {
            UnitType.INFANTRY_TEAM: 5,  # 50m between teams
            UnitType.WEAPONS_TEAM: 7,  # 70m between weapons teams
            UnitType.INFANTRY_SQUAD: 10  # 100m between squads
        }

    def find_positions(self,
                       objective: Tuple[int, int],
                       unit_type: UnitType,
                       position_purpose: PositionPurpose,
                       start_positions: List[Tuple[int, int]],
                       enemy_threats: List[Dict],
                       max_positions: int = 5) -> Dict:
        """
        Find tactical positions using progressive filtering and analysis.

        Args:
            objective: Target position (x,y)
            unit_type: Type of unit (team/squad)
            position_purpose: Tactical purpose (assault/support/reserve)
            start_positions: Possible starting positions
            enemy_threats: Known enemy positions and capabilities
            max_positions: Number of positions to return

        Returns:
            Dictionary containing analysis results and positions
        """
        print(f"\n{'=' * 80}")
        print(f"TACTICAL POSITION ANALYSIS: {unit_type.name} {position_purpose.name}")
        print(f"{'=' * 80}")

        # Store objective and enemy locations for analysis
        self.objective = objective
        self._initialize_threat_analysis(enemy_threats)

        # Step 1: Initialize spatial filter
        filter_size = self._get_unit_dimensions(unit_type)
        stride = 2 if unit_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM] else 4
        print(f"\nSTEP 1: Initialization:")
        print(f"- Filter size: {filter_size}x{filter_size} cells")
        print(f"- Filter stride: {stride} cells")
        print(f"- Objective location: {objective}")
        min_range, max_range = self._get_position_ranges(unit_type, position_purpose)
        print(f"- Range band: {min_range * 10}-{max_range * 10}m (based on unit capabilities)")
        print(f"- Enemy threats: {len(enemy_threats)}")

        # Step 2: Get range-based area of interest
        print("\nSTEP 2: Range-Based Filtering")
        initial_positions = self._get_range_based_positions(
            objective, min_range, max_range)
        print(f"- Initial search space: {self.width * self.height} cells")
        print(f"- Spaces in range band: {len(initial_positions)} cells")

        # Step 3: Apply kernel filter
        print("\nSTEP 3: Kernel-Based Filtering")
        filtered_centers = self._apply_kernel_filter(
            initial_positions, filter_size, stride)
        print(f"- Valid position centers after kernel filtering: {len(filtered_centers)}")

        # Calculate theoretical maximum positions
        max_theoretical = ((self.width // stride) * (self.height // stride))
        print(f"- Theoretical maximum positions: {max_theoretical}")
        print(f"- Filter coverage per position: {filter_size * filter_size} cells")

        # Step 4: Analyze filtered positions
        print("\nSTEP 4: Detailed Position Analysis")
        analyzed_positions = []
        progress_step = max(1, len(filtered_centers) // 10)

        for i, center in enumerate(filtered_centers):
            if i % progress_step == 0:
                print(f"- Analyzing position {i + 1}/{len(filtered_centers)}...")
                print(f"  Progress: {((i + 1) / len(filtered_centers)) * 100:.1f}%")

            # Get all cells within filter bounds
            area_cells = self._get_filter_cells(center, filter_size)

            # Get pooled metrics for filter area with progress updates
            area_metrics = self._analyze_filter_area(
                center,
                area_cells,
                objective,
                start_positions,
                show_progress=(i % progress_step == 0)
            )

            # Apply minimum quality threshold
            if area_metrics['avg_quality'] > 0.2:
                analyzed_positions.append((center, area_metrics))
                if i % progress_step == 0:
                    print(f"  Position passed quality threshold: Score = {area_metrics['avg_quality']:.2f}")

        print(f"- Positions passing quality threshold: {len(analyzed_positions)}")

        # Step 5: Score positions
        print("\nSTEP 5: Position Scoring")
        scored_positions = []
        for i, (pos, metrics) in enumerate(analyzed_positions):
            if i % progress_step == 0:
                print(f"- Scoring position {i + 1}/{len(analyzed_positions)}...")

            # Add accessibility evaluation
            metrics['accessibility'] = self._evaluate_position_accessibility(
                pos, start_positions)

            # Calculate comprehensive score
            score_data = self._calculate_position_score(metrics, position_purpose)

            scored_positions.append({
                'position': pos,
                'metrics': metrics,
                'score_data': score_data,
                'filter_cells': self._get_filter_cells(pos, filter_size)  # Store cells for visualization
            })

        # Sort and get top positions
        scored_positions.sort(key=lambda x: x['score_data']['final_score'], reverse=True)
        top_positions = scored_positions[:max_positions]

        # Print analysis summary
        print(f"\n{'-' * 80}")
        print(f"FINAL RESULTS:")
        print(f"{'-' * 80}")
        print(f"Total positions evaluated: {len(filtered_centers)}")
        print(f"Positions passing analysis: {len(analyzed_positions)}")
        print(f"Final positions selected: {len(top_positions)}")

        # Print detailed results for top positions
        print("\nTop Position Details:")
        for i, pos_data in enumerate(top_positions, 1):
            print(f"\nPosition {i}:")
            print(f"- Location: {pos_data['position']}")
            print(f"- Quality Score: {pos_data['score_data']['final_score']:.2f}")
            print(f"- Cover Score: {pos_data['metrics']['avg_cover']:.2f}")
            print(f"- Concealment Score: {pos_data['metrics']['avg_concealment']:.2f}")
            print(f"- Threat Exposure: {pos_data['metrics']['avg_threat']:.2f}")
            if pos_data['metrics']['has_elevation']:
                print("- Has elevation advantage")

        # Create results structure
        results = {
            'positions': top_positions,
            'total_evaluated': len(filtered_centers),
            'total_filtered': len(analyzed_positions),
            'unit_type': unit_type.name,
            'purpose': position_purpose.name,
            'filter_size': filter_size,
            'stride': stride
        }

        # Convert to tactical positions
        tactical_positions = self._convert_to_tactical_positions(results)

        # Add tactical positions to results
        results['tactical_positions'] = tactical_positions

        return results

    def _get_unit_dimensions(self, unit_type: UnitType) -> int:
        """Get spatial filter dimensions based on unit type."""
        if unit_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM]:
            return 7  # 7x7 cell filter for team (70m x 70m)
        else:
            return 12  # 12x12 cell filter for squad (120m x 120m)

    def _get_range_based_positions(self,
                                   objective: Tuple[int, int],
                                   min_range: int,
                                   max_range: int) -> List[Tuple[int, int]]:
        """
        Get valid positions within range band from objective.

        Args:
            objective: Target position (x,y)
            min_range: Minimum range in cells
            max_range: Maximum range in cells

        Returns:
            List of (x,y) positions within range band
        """
        positions = []
        obj_x, obj_y = objective

        # Calculate bounding box for efficiency
        min_x = max(0, obj_x - max_range)
        max_x = min(self.width - 1, obj_x + max_range)
        min_y = max(0, obj_y - max_range)
        max_y = min(self.height - 1, obj_y + max_range)

        # Check positions in bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance = math.sqrt((x - obj_x) ** 2 + (y - obj_y) ** 2)
                if min_range <= distance <= max_range:
                    positions.append((x, y))

        return positions

    def _get_filter_cells(self,
                          center: Tuple[int, int],
                          filter_size: int) -> List[Tuple[int, int]]:
        """
        Get all valid cells within filter area with proper edge handling.

        Args:
            center: Center position (x, y)
            filter_size: Size of square filter (must be integer)
        """
        cells = []
        half_size = filter_size // 2  # Integer division

        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                x = center[0] + dx
                y = center[1] + dy

                # Include cell with zero values if outside map
                if 0 <= x < self.width and 0 <= y < self.height:
                    cells.append((x, y))

        return cells

    def _apply_kernel_filter(self, positions: List[Tuple[int, int]],
                             filter_size: int, stride: int = 2) -> List[Tuple[int, int]]:
        """
        Apply kernel filter to positions with proper striding and padding.

        Args:
            positions: List of potential position centers
            filter_size: Size of square filter (7 for team, 12 for squad)
            stride: Number of cells to move filter each step

        Returns:
            List of valid position centers that meet spacing requirements
        """
        # Sort positions by coordinates for efficient filtering
        sorted_positions = sorted(positions)
        valid_positions = []
        used_areas = set()

        half_size = filter_size // 2

        for pos in sorted_positions:
            x, y = pos

            # Skip if position is too close to existing position
            area_key = (x // stride, y // stride)
            if area_key in used_areas:
                continue

            # Check if position is too close to map edge
            if (x < half_size or x >= self.width - half_size or
                    y < half_size or y >= self.height - half_size):
                continue

            # Get all cells in filter area
            filter_cells = []
            valid_area = True

            for dy in range(-half_size, half_size + 1):
                for dx in range(-half_size, half_size + 1):
                    cell_x = x + dx
                    cell_y = y + dy

                    # Mark cells with zero value if outside map
                    if not (0 <= cell_x < self.width and 0 <= cell_y < self.height):
                        valid_area = False
                        break

                    filter_cells.append((cell_x, cell_y))

            if valid_area:
                valid_positions.append(pos)
                # Mark area as used based on stride
                for sy in range(y // stride - 1, y // stride + 2):
                    for sx in range(x // stride - 1, x // stride + 2):
                        used_areas.add((sx, sy))

        return valid_positions

    def _initialize_threat_analysis(self, enemy_threats: List[Dict]):
        """Initialize threat analyzer with enemy positions."""
        self.threat_analyzer = ThreatAnalysis(self.env)
        for threat in enemy_threats:
            threat_obj = Threat(
                position=threat['position'],
                unit=threat.get('unit', None),
                observation_range=threat['observation_range'],
                engagement_range=threat['engagement_range'],
                suspected_accuracy=threat.get('accuracy', 0.8)
            )
            self.threat_analyzer.add_threat(threat_obj)

    def _analyze_filter_area(self,
                             center: Tuple[int, int],
                             filter_cells: List[Tuple[int, int]],
                             objective: Tuple[int, int],
                             start_positions: List[Tuple[int, int]],
                             show_progress: bool = False) -> Dict:
        """
        Analyze metrics for all cells in filter area with proper pooling.
        """
        if show_progress:
            print(f"  Analyzing {len(filter_cells)} cells in filter area...")

        # Initialize metric arrays
        threat_values = []
        observation_values = []
        engagement_values = []
        fire_quality_values = []
        cover_values = []
        concealment_values = []
        fire_positions = []

        # Analyze each cell in filter area
        for cell in filter_cells:
            # Get threat analysis
            threat_analysis = self.threat_analyzer.analyze_position(cell)
            threat_values.append(threat_analysis['total_threat'])
            observation_values.append(threat_analysis['observation_threat'])
            engagement_values.append(threat_analysis['engagement_threat'])

            # Get direct fire analysis
            fire_analysis = self.direct_fire_analyzer.analyze_position(
                position=cell,
                capabilities=self._get_unit_capabilities(UnitType.INFANTRY_TEAM),
                targets=[objective],
                position_purpose=PositionPurpose.SUPPORT
            )

            # Track fire quality
            fire_quality = fire_analysis['quality_scores']['engagement_quality']
            fire_quality_values.append(fire_quality)
            if fire_quality > 0.5:
                fire_positions.append(cell)

            # Get terrain values
            cover = self.env.terrain_manager.get_cover(cell)
            concealment = 1 - self.env.terrain_manager.get_visibility(cell)
            cover_values.append(cover)
            concealment_values.append(concealment)

        # Calculate elevation characteristics
        elevation_cells = [cell for cell in filter_cells
                           if self.env.terrain_manager.get_elevation_type(cell) ==
                           ElevationType.ELEVATED_LEVEL]
        has_elevation = len(elevation_cells) > 0
        elevation_percentage = len(elevation_cells) / len(filter_cells)

        # Get accessibility analysis
        accessibility = self._evaluate_position_accessibility(
            center, start_positions)

        # Calculate average metrics using numpy to handle empty arrays
        fire_quality_mean = np.mean(fire_quality_values) if fire_quality_values else 0.0
        cover_mean = np.mean(cover_values) if cover_values else 0.0
        concealment_mean = np.mean(concealment_values) if concealment_values else 0.0

        avg_quality = (
                fire_quality_mean * 0.4 +
                cover_mean * 0.3 +
                concealment_mean * 0.3
        )

        if show_progress:
            print(f"  Average Quality Score: {avg_quality:.2f}")
            print(f"  Average Cover: {cover_mean:.2f}")
            print(f"  Average Concealment: {concealment_mean:.2f}")
            print(f"  Average Threat: {np.mean(threat_values) if threat_values else 0.0:.2f}")

        # Ensure all numpy calculations handle empty arrays
        return {
            'avg_threat': float(np.mean(threat_values)) if threat_values else 0.0,
            'max_threat': float(np.max(threat_values)) if threat_values else 0.0,
            'avg_observation': float(np.mean(observation_values)) if observation_values else 0.0,
            'avg_engagement': float(np.mean(engagement_values)) if engagement_values else 0.0,
            'avg_fire_quality': float(fire_quality_mean),
            'max_fire_quality': float(np.max(fire_quality_values)) if fire_quality_values else 0.0,
            'good_fire_positions': len(fire_positions),
            'avg_cover': float(cover_mean),
            'min_cover': float(np.min(cover_values)) if cover_values else 0.0,
            'avg_concealment': float(concealment_mean),
            'min_concealment': float(np.min(concealment_values)) if concealment_values else 0.0,
            'has_elevation': has_elevation,
            'elevation_percentage': float(elevation_percentage),
            'accessibility': accessibility,
            'avg_quality': float(avg_quality),
            'total_cells': len(filter_cells),
            'usable_cells': sum(1 for t in threat_values if t < 0.7),
            'center_position': center
        }

    def _calculate_position_score(self, metrics: Dict, position_purpose: PositionPurpose) -> Dict:
        """
        Calculate position score with accessibility bonus.

        Args:
            metrics: Dictionary of pooled metrics from filter area
            position_purpose: Tactical purpose of position

        Returns:
            Dictionary containing base score, bonuses, and final score
        """
        # Base scoring weights by position purpose
        if position_purpose == PositionPurpose.SUPPORT:
            weights = {
                'avg_fire_quality': 0.4,  # Emphasize fires
                'avg_cover': 0.2,  # Moderate protection
                'avg_concealment': 0.2,  # Moderate concealment
                'avg_threat': -0.2,  # Penalize threats
                'usable_cells': 0.1  # Some value for space
            }
            accessibility_importance = 0.05  # Small bonus for support

        elif position_purpose == PositionPurpose.ASSAULT:
            weights = {
                'avg_cover': 0.3,  # High protection
                'avg_concealment': 0.3,  # High concealment
                'avg_fire_quality': 0.1,  # Less emphasis on fires
                'avg_threat': -0.2,  # Consider threats
                'usable_cells': 0.2  # Value maneuverable space
            }
            accessibility_importance = 0.1  # Moderate bonus for assault

        else:  # RESERVE
            weights = {
                'avg_concealment': 0.4,  # Maximum concealment
                'avg_cover': 0.2,  # Good protection
                'avg_fire_quality': 0.1,  # Minimal fires
                'avg_threat': -0.2,  # Avoid threats
                'usable_cells': 0.2  # Value space
            }
            accessibility_importance = 0.15  # Larger bonus for reserve

        # Calculate base score
        base_score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == 'usable_cells':
                    value = value / metrics['total_cells']  # Normalize
                if metric == 'avg_threat':
                    value = 1.0 - value  # Invert threat value
                base_score += value * weight

        # Track applied bonuses
        bonuses = []
        final_score = base_score

        # Apply accessibility bonus if good
        if metrics['accessibility']['overall_accessibility'] > 0.5:
            accessibility_bonus = (metrics['accessibility']['overall_accessibility'] - 0.5) * accessibility_importance
            final_score += accessibility_bonus
            bonuses.append({
                'type': 'accessibility',
                'value': accessibility_bonus,
                'description': f"{metrics['accessibility']['access_points']} approaches, {metrics['accessibility']['exit_points']} exits"
            })

        # Apply elevation bonus if present
        if metrics['has_elevation']:
            elevation_bonus = 0.1  # 10% bonus
            final_score *= (1 + elevation_bonus)
            bonuses.append({
                'type': 'elevation',
                'value': elevation_bonus,
                'description': f"{int(metrics['elevation_percentage'] * 100)}% elevated terrain"
            })

        # Ensure final score is between 0 and 1
        final_score = max(0.0, min(1.0, final_score))

        return {
            'base_score': base_score,
            'bonuses': bonuses,
            'final_score': final_score
        }

    def _get_unit_capabilities(self, unit_type: UnitType) -> UnitFireCapabilities:
        """
        Get fire capabilities based on unit type and weapons.

        Args:
            unit_type: Type of unit to get capabilities for

        Returns:
            UnitFireCapabilities object with ranges and weapon types
        """
        if unit_type == UnitType.INFANTRY_TEAM:
            # Basic fire team capabilities
            return UnitFireCapabilities(
                observation_range=50,  # 500m
                engagement_range=40,  # 400m
                primary_weapon_type="M4",
                min_range=0
            )
        elif unit_type == UnitType.WEAPONS_TEAM:
            # Machine gun team capabilities
            return UnitFireCapabilities(
                observation_range=70,  # 700m
                engagement_range=60,  # 600m
                primary_weapon_type="M240",
                min_range=0
            )
        else:  # Squad has mix of weapons
            return UnitFireCapabilities(
                observation_range=70,  # Use highest organic capability
                engagement_range=60,  # Use highest organic capability
                primary_weapon_type="Mixed",
                min_range=0
            )

    def _get_position_ranges(self, unit_type: UnitType, position_purpose: PositionPurpose) -> Tuple[int, int]:
        """
        Get min/max ranges based on unit capabilities and purpose.

        Args:
            unit_type: Type of unit for capabilities
            position_purpose: Purpose determining range bands

        Returns:
            Tuple of (min_range, max_range) in cells
        """
        capabilities = self._get_unit_capabilities(unit_type)

        if position_purpose == PositionPurpose.ASSAULT:
            # Assault positions close to objective
            min_range = 0
            max_range = int(capabilities.engagement_range * 0.5)  # Half max range

        elif position_purpose == PositionPurpose.SUPPORT:
            # Support positions at effective range
            min_range = int(capabilities.engagement_range * 0.4)  # 40% of max
            max_range = capabilities.engagement_range  # Full range

        else:  # RESERVE
            # Reserve positions further back
            min_range = int(capabilities.engagement_range * 0.7)  # 70% of max
            max_range = capabilities.observation_range  # Use observation range

        return min_range, max_range

    def _evaluate_position_accessibility(self,
                                         pos: Tuple[int, int],
                                         start_positions: List[Tuple[int, int]]
                                         ) -> Dict:
        """
        Evaluate basic accessibility of a position for movement bonuses.

        Args:
            pos: Position to evaluate
            start_positions: Potential starting positions

        Returns:
            Dictionary containing:
            - access_points: Number of clear approaches
            - exit_points: Number of clear exits
            - overall_accessibility: Score 0-1 for position access
            - viable_approaches: List of viable approach directions
        """
        # Determine primary approach directions from start positions
        approach_directions = set()
        for start in start_positions:
            # Calculate general direction from start to position
            dx = pos[0] - start[0]
            dy = pos[1] - start[1]

            # Convert to closest cardinal/intercardinal direction
            angle = math.degrees(math.atan2(dy, dx))
            dir_angle = round(angle / 45.0) * 45
            if dir_angle < 0:
                dir_angle += 360

            approach_directions.add(dir_angle)

        # Define direction vectors for 8 cardinal/intercardinal directions
        directions = {
            0: (1, 0),  # E
            45: (1, 1),  # NE
            90: (0, 1),  # N
            135: (-1, 1),  # NW
            180: (-1, 0),  # W
            225: (-1, -1),  # SW
            270: (0, -1),  # S
            315: (1, -1)  # SE
        }

        # Check approach directions first
        access_points = 0
        viable_approaches = []

        for angle in approach_directions:
            dx, dy = directions[angle]
            # Look 5 cells out from position
            clear_path = True
            path_cells = []

            for distance in range(1, 6):
                check_x = pos[0] + (dx * distance)
                check_y = pos[1] + (dy * distance)

                # Check bounds
                if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                    clear_path = False
                    break

                # Check if cell is traversable
                terrain_type = self.env.terrain_manager.get_terrain_type((check_x, check_y))
                if terrain_type == TerrainType.STRUCTURE:
                    clear_path = False
                    break

                # Check threat level
                threat_level = self.threat_analyzer.total_threat_matrix[check_y, check_x]
                if threat_level > 0.8:  # Very high threat
                    clear_path = False
                    break

                path_cells.append((check_x, check_y))

            if clear_path:
                access_points += 1
                viable_approaches.append(angle)

        # Check exit directions (all directions except approaches)
        exit_points = 0
        for angle, (dx, dy) in directions.items():
            # Skip approach directions - already counted
            if angle in viable_approaches:
                continue

            clear_path = True
            for distance in range(1, 6):
                check_x = pos[0] + (dx * distance)
                check_y = pos[1] + (dy * distance)

                if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                    clear_path = False
                    break

                terrain_type = self.env.terrain_manager.get_terrain_type((check_x, check_y))
                if terrain_type == TerrainType.STRUCTURE:
                    clear_path = False
                    break

                # Less stringent threat check for exits
                threat_level = self.threat_analyzer.total_threat_matrix[check_y, check_x]
                if threat_level > 0.9:  # Extremely high threat
                    clear_path = False
                    break

            if clear_path:
                exit_points += 1

        # Calculate overall accessibility score
        # Weight approach directions more heavily
        approach_score = access_points / len(approach_directions) if approach_directions else 0
        exit_score = exit_points / (8 - len(approach_directions)) if (8 - len(approach_directions)) > 0 else 0

        overall_accessibility = (approach_score * 0.7) + (exit_score * 0.3)

        return {
            'access_points': access_points,
            'exit_points': exit_points,
            'overall_accessibility': overall_accessibility,
            'viable_approaches': viable_approaches
        }

    def _convert_to_tactical_positions(self, results: Dict) -> List[TacticalPosition]:
        """Convert filter results to TacticalPosition objects."""
        tactical_positions = []

        for pos_data in results['positions']:
            # Get base position and metrics
            pos = pos_data['position']
            metrics = pos_data['metrics']

            # Use stored filter cells if available, otherwise calculate them
            cells = pos_data.get('filter_cells', self._get_filter_cells(
                pos,
                results['filter_size']
            ))

            # Create TacticalPosition object
            tactical_pos = TacticalPosition(
                position=pos,
                cells=cells,
                position_type=PositionPurpose[results['purpose']],  # Now converts from name to enum
                unit_size=UnitSize.TEAM if results['unit_type'] == 'INFANTRY_TEAM' else UnitSize.SQUAD,
                coverage_arc=(0, 360),  # Default full arc
                max_range=metrics.get('engagement_range', 40),
                covered_threats=metrics.get('covered_threats', []),
                engagement_quality=metrics.get('avg_fire_quality', 0.0),
                cover_score=metrics.get('avg_cover', 0.0),
                concealment_score=metrics.get('avg_concealment', 0.0),
                movement_score=metrics.get('movement_score', 0.0),
                approach_routes=[],  # Will be calculated in route analyzer
                withdrawal_routes=[],
                has_elevation=metrics.get('has_elevation', False),
                observation_arc=(0, 360),
                observation_range=metrics.get('observation_range', 50),
                observation_quality=metrics.get('avg_observation', 0.0),
                threat_exposure=metrics.get('avg_threat', 0.0),
                primary_threats=metrics.get('primary_threats', []),
                supporting_positions=[],
                requires_support=True,
                mutual_support_positions=[],
                quality_score=pos_data['score_data']['final_score'],
                quality_breakdown=pos_data['score_data']
            )

            tactical_positions.append(tactical_pos)

        return tactical_positions


# Test code
def convert_to_plot_coordinates(y: int, height: int) -> int:
    """Convert environment y-coordinate to plot y-coordinate."""
    return height - 1 - y


def plot_terrain_base(ax: plt.Axes, env: MilitaryEnvironment) -> np.ndarray:
    """
    Plot terrain base layer with proper coordinate system.
    Returns the RGB terrain map for reuse.
    """
    # Create terrain matrix
    terrain_matrix = np.zeros((env.height, env.width))
    for y in range(env.height):
        # Convert to plot coordinates
        plot_y = convert_to_plot_coordinates(y, env.height)
        for x in range(env.width):
            terrain_type = env.terrain_manager.get_terrain_type((x, y))
            terrain_matrix[plot_y, x] = terrain_type.value

    # Convert to RGB
    terrain_colors = {
        TerrainType.BARE.value: [0.9, 0.9, 0.9],
        TerrainType.SPARSE_VEG.value: [0.8, 0.9, 0.8],
        TerrainType.DENSE_VEG.value: [0.4, 0.7, 0.4],
        TerrainType.WOODS.value: [0.2, 0.5, 0.2],
        TerrainType.STRUCTURE.value: [0.6, 0.6, 0.7]
    }

    rgb_map = np.zeros((*terrain_matrix.shape, 3))
    for value, color in terrain_colors.items():
        mask = terrain_matrix == value
        for i in range(3):
            rgb_map[:, :, i][mask] = color[i]

    # Plot with proper coordinates
    ax.imshow(rgb_map, origin='lower')
    return rgb_map


def visualize_tactical_positions(env: MilitaryEnvironment,
                                 results: Dict,
                                 objective: Tuple[int, int],
                                 enemy_threats: List[Dict],
                                 unit_type: UnitType):
    """Visualize tactical positions with terrain, ranges, and position areas."""
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot terrain base
    rgb_map = plot_terrain_base(ax, env)

    # Plot objective
    ax.plot(objective[0], objective[1], 'r*', markersize=15, label='Objective')

    # Plot enemy positions and ranges
    for threat in enemy_threats:
        pos = threat['position']
        # Observation range
        obs_circle = Circle(pos, threat['observation_range'],
                            fill=False, linestyle='--', color='red', alpha=0.3)
        ax.add_patch(obs_circle)
        # Engagement range
        eng_circle = Circle(pos, threat['engagement_range'],
                            fill=False, color='red', alpha=0.5)
        ax.add_patch(eng_circle)
        # Position marker
        ax.plot(pos[0], pos[1], 'rv', markersize=10)

    # Plot top positions
    filter_size = 7 if unit_type in [UnitType.INFANTRY_TEAM, UnitType.WEAPONS_TEAM] else 12
    half_size = filter_size // 2

    for i, pos_data in enumerate(results['positions'], 1):
        pos = pos_data['position']
        score = pos_data['score_data']['final_score']

        # Plot position area
        rect = Rectangle((pos[0] - half_size, pos[1] - half_size),
                         filter_size, filter_size,
                         fill=False, color='blue', alpha=0.5)
        ax.add_patch(rect)

        # Plot center point with score
        ax.scatter(pos[0], pos[1], c='blue', s=100)
        ax.annotate(f"Pos {i}\nScore: {score:.2f}",
                    (pos[0], pos[1]), xytext=(5, 5),
                    textcoords='offset points')

        # Plot observation and engagement ranges if available
        if 'metrics' in pos_data and 'observation_range' in pos_data['metrics']:
            obs_circle = Circle(pos, pos_data['metrics']['observation_range'],
                                fill=False, linestyle='--', color='blue', alpha=0.2)
            ax.add_patch(obs_circle)

        if 'metrics' in pos_data and 'engagement_range' in pos_data['metrics']:
            eng_circle = Circle(pos, pos_data['metrics']['engagement_range'],
                                fill=False, color='green', alpha=0.2)
            ax.add_patch(eng_circle)

    ax.set_title(f'Tactical Positions Analysis - {unit_type.name}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig(f'tactical_positions_{unit_type.name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_threat_analysis():
    """Test enhanced threat analysis with visualization."""
    print("\n=== Testing Enhanced Threat Analysis ===")

    # Initialize environment with terrain
    config = EnvironmentConfig(width=100, height=100, debug_level=1)
    env = MilitaryEnvironment(config)
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)

    # Try to load terrain from CSV if available
    try:
        csv_path = 'generated_map.csv'
        env.terrain_manager.load_from_csv(csv_path)
        print(f"Loaded terrain from {csv_path}")
    except Exception as e:
        print(f"Using default terrain: {str(e)}")

    # Initialize threat analyzer
    threat_analyzer = ThreatAnalysis(env)

    # Define test enemy positions with different capabilities
    enemy_positions = [
        {
            'position': (50, 50),  # Center
            'name': 'Main Position',
            'observation_range': 48,  # 480m
            'engagement_range': 30,  # 300m
            'accuracy': 0.8
        },
        {
            'position': (45, 48),  # Southwest of main
            'name': 'Support Position',
            'observation_range': 40,  # 400m
            'engagement_range': 25,  # 250m
            'accuracy': 0.7
        },
        {
            'position': (55, 52),  # Northeast of main
            'name': 'Security Position',
            'observation_range': 35,  # 350m
            'engagement_range': 20,  # 200m
            'accuracy': 0.6
        }
    ]

    # Add threats and analyze
    print("\nAdding enemy positions...")
    for enemy in enemy_positions:
        print(f"\nProcessing {enemy['name']}:")
        print(f"Position: {enemy['position']}")
        print(f"Observation Range: {enemy['observation_range'] * 10}m")
        print(f"Engagement Range: {enemy['engagement_range'] * 10}m")
        print(f"Accuracy: {enemy['accuracy']}")

        threat = Threat(
            position=enemy['position'],
            unit=None,
            observation_range=enemy['observation_range'],
            engagement_range=enemy['engagement_range'],
            suspected_accuracy=enemy['accuracy']
        )
        threat_analyzer.add_threat(threat)

    # Create visualization with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

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

    # Plot terrain base on all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.imshow(terrain_grid, origin='lower')

    # Plot threat analysis results
    observation_overlay = ax1.imshow(
        threat_analyzer.observation_matrix,
        alpha=0.5,
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        origin='lower'
    )
    ax1.set_title('Enemy Observation Coverage')
    plt.colorbar(observation_overlay, ax=ax1)

    engagement_overlay = ax2.imshow(
        threat_analyzer.engagement_matrix,
        alpha=0.5,
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        origin='lower'
    )
    ax2.set_title('Enemy Engagement Coverage')
    plt.colorbar(engagement_overlay, ax=ax2)

    total_threat_overlay = ax3.imshow(
        threat_analyzer.total_threat_matrix,
        alpha=0.5,
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        origin='lower'
    )
    ax3.set_title('Combined Threat Analysis')
    plt.colorbar(total_threat_overlay, ax=ax3)

    # Plot enemy positions and ranges on fourth subplot
    ax4.set_title('Enemy Positions and Ranges')
    for enemy in enemy_positions:
        pos = enemy['position']
        # Plot observation range
        obs_circle = Circle(
            pos,
            enemy['observation_range'],
            fill=False,
            linestyle='--',
            color='yellow',
            alpha=0.5
        )
        ax4.add_patch(obs_circle)

        # Plot engagement range
        eng_circle = Circle(
            pos,
            enemy['engagement_range'],
            fill=False,
            color='red',
            alpha=0.5
        )
        ax4.add_patch(eng_circle)

        # Plot position
        ax4.plot(pos[0], pos[1], 'r^', markersize=10, label=enemy['name'])

    # Test specific positions
    test_positions = [
        (30, 30),  # Southwest
        (40, 40),  # South central
        (48, 48),  # Near main position
        (60, 60)  # Northeast
    ]

    print("\nAnalyzing test positions:")
    for pos in test_positions:
        analysis = threat_analyzer.analyze_position(pos)

        print(f"\nPosition {pos}:")
        print(f"Total Threat: {analysis['total_threat']:.2f}")
        print(f"Observation Threat: {analysis['observation_threat']:.2f}")
        print(f"Engagement Threat: {analysis['engagement_threat']:.2f}")

        if analysis['threatening_enemies']:
            print("Threatening enemies:")
            for threat in analysis['threatening_enemies']:
                print(f"- Position {threat['position']}")
                threat_info = threat['threat_level']
                print(f"  * Can observe: {threat_info['can_observe']}")
                print(f"  * Can engage: {threat_info['can_engage']}")
                print(f"  * Observation quality: {threat_info['observation_quality']:.2f}")
                print(f"  * Engagement quality: {threat_info['engagement_quality']:.2f}")

        # Plot test position with threat-based color
        cmap = plt.get_cmap('Reds')  # Using 'Reds' colormap
        color = cmap(float(analysis['total_threat']))
        ax4.scatter(pos[0], pos[1], c=[color], s=100)
        ax4.annotate(
            f"{analysis['total_threat']:.2f}",
            (pos[0], pos[1]),
            xytext=(5, 5),
            textcoords='offset points'
        )

    # Add legend
    ax4.legend()

    # Set proper axis labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_threat_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nThreat analysis visualization saved as 'enhanced_threat_analysis.png'")


def test_direct_fire_analysis():
    """Test direct fire analysis component with proper coordinate system."""
    print("\n=== Testing Direct Fire Analysis Component ===")
    print("Note: Using military coordinate system (0,0 at bottom left)")

    # Initialize environment
    config = EnvironmentConfig(width=100, height=100, debug_level=1)
    env = MilitaryEnvironment(config)

    # Initialize terrain
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)
    print("Terrain initialized")

    # Load terrain data if available
    try:
        csv_path = 'generated_map.csv'
        env.terrain_manager.load_from_csv(csv_path)
        print(f"Loaded terrain from {csv_path}")
    except Exception as e:
        print(f"Using default terrain: {str(e)}")

    # Initialize direct fire analyzer
    fire_analyzer = DirectFireAnalysis(env)

    # Define test friendly positions with different capabilities
    friendly_positions = [
        {
            'position': (30, 50),  # West central
            'name': 'Support Position 1',
            'capabilities': UnitFireCapabilities(
                observation_range=70,  # 700m
                engagement_range=60,  # 600m
                primary_weapon_type="M240"
            )
        },
        {
            'position': (25, 45),  # Southwest
            'name': 'Support Position 2',
            'capabilities': UnitFireCapabilities(
                observation_range=70,
                engagement_range=60,
                primary_weapon_type="M240"
            )
        },
        {
            'position': (35, 55),  # Northwest
            'name': 'Assault Position',
            'capabilities': UnitFireCapabilities(
                observation_range=50,  # 500m
                engagement_range=40,  # 400m
                primary_weapon_type="M4"
            )
        }
    ]

    # Define enemy targets (in environment coordinates)
    enemy_targets = [
        (80, 50),  # Main position (east central)
        (75, 48),  # Support (southeast)
        (77, 52)  # Security (northeast)
    ]

    print("\nFriendly Positions (environment coordinates):")
    for friendly in friendly_positions:
        print(f"{friendly['name']}: {friendly['position']}")
        print(
            f"Plot coordinates: ({friendly['position'][0]}, {convert_to_plot_coordinates(friendly['position'][1], env.height)})")

    print("\nEnemy Targets (environment coordinates):")
    for i, target in enumerate(enemy_targets):
        print(f"Target {i + 1}: {target}")
        print(f"Plot coordinates: ({target[0]}, {convert_to_plot_coordinates(target[1], env.height)})")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))

    # Plot terrain base on all subplots
    rgb_map = plot_terrain_base(ax1, env)
    ax2.imshow(rgb_map, origin='lower')
    ax3.imshow(rgb_map, origin='lower')
    ax4.imshow(rgb_map, origin='lower')

    # Initialize observation and engagement matrices
    observation_matrix = np.zeros((env.height, env.width))
    engagement_matrix = np.zeros((env.height, env.width))

    # Process each friendly position
    print("\nAnalyzing friendly positions...")
    for friendly in friendly_positions:
        pos = friendly['position']
        print(f"\nAnalyzing {friendly['name']} at {pos}")
        plot_pos = (pos[0], convert_to_plot_coordinates(pos[1], env.height))

        # Get analysis for assault and support purposes
        assault_analysis = fire_analyzer.analyze_position(
            position=pos,
            capabilities=friendly['capabilities'],
            targets=enemy_targets,
            position_purpose=PositionPurpose.ASSAULT
        )

        support_analysis = fire_analyzer.analyze_position(
            position=pos,
            capabilities=friendly['capabilities'],
            targets=enemy_targets,
            position_purpose=PositionPurpose.SUPPORT
        )

        print("\nTerrain Analysis at Position:")
        x, y = pos
        terrain_type = env.terrain_manager.get_terrain_type(pos)
        cover = env.terrain_manager.get_cover(pos)
        visibility = env.terrain_manager.get_visibility(pos)
        elevation = env.terrain_manager.get_elevation_type(pos)

        print(f"- Terrain Type: {terrain_type.name}")
        print(f"- Cover Value: {cover:.2f}")
        print(f"- Visibility: {visibility:.2f}")
        print(f"- Elevation: {elevation.name}")

        print("\nAssault Role Analysis:")
        print(f"Meets Requirements: {assault_analysis['meets_requirements']}")
        print("Quality Scores:")
        for metric, score in assault_analysis['quality_scores'].items():
            print(f"- {metric}: {score:.2f}")

        print("Covered Targets:")
        for target in assault_analysis['covered_targets']:
            # Print target with correct field names
            position = target['position']
            hit_prob = target['hit_probability']
            damage = target['damage_potential']
            los_quality = target['los_quality']
            distance = target['distance']

            print(f"- Target at {position}:")
            print(f"  * Distance: {distance:.1f}m")
            print(f"  * Hit Probability: {hit_prob:.2f}")
            print(f"  * Damage Potential: {damage:.1f}")
            print(f"  * LOS Quality: {los_quality:.2f}")

        print("Engagement Arc Coverage:")
        total_cells = len(assault_analysis['engagement_arc'])
        effective_cells = sum(1 for effect in assault_analysis['engagement_arc'].values() if effect > 0.3)
        print(f"- Total cells in arc: {total_cells}")
        print(f"- Effective cells (>0.3): {effective_cells}")

        print("\nSupport Role Analysis:")
        print(f"Meets Requirements: {support_analysis['meets_requirements']}")
        print("Quality Scores:")
        for metric, score in support_analysis['quality_scores'].items():
            print(f"- {metric}: {score:.2f}")

        print("Covered Targets:")
        for target in assault_analysis['covered_targets']:
            # Print full target data to see structure
            print(f"- Raw target data: {target}")

            # Try to access data more safely
            position = target.get('position', None)
            if position is None:
                continue

            # Try different possible quality metrics
            quality = target.get('hit_probability',
                                 target.get('engagement_quality',
                                            target.get('los_quality', 0.0)))

            print(f"- Target at {position}: Quality {quality:.2f}")

        print("Engagement Arc Coverage:")
        if 'engagement_arc' in assault_analysis:
            total_cells = len(assault_analysis['engagement_arc'])
            effective_cells = sum(1 for effect in assault_analysis['engagement_arc'].values() if effect > 0.3)
            print(f"Total cells in arc: {total_cells}")
            print(f"Effective cells (>0.3): {effective_cells}")
        else:
            print("No engagement arc data available")

        if 'observation_arc' in assault_analysis:
            for target, quality in assault_analysis['observation_arc'].items():
                print(f"Observation to {target}: {quality:.2f}")

        # Plot ranges and arcs
        obs_circle = Circle(
            plot_pos,
            friendly['capabilities'].observation_range,
            fill=False,
            linestyle='--',
            color='blue',
            alpha=0.5
        )
        ax4.add_patch(obs_circle)

        eng_circle = Circle(
            plot_pos,
            friendly['capabilities'].engagement_range,
            fill=False,
            color='green',
            alpha=0.5
        )
        ax4.add_patch(eng_circle)

        # Plot covered targets and engagement lines
        for target in assault_analysis['covered_targets']:
            target_pos = target['position']
            plot_target = (target_pos[0], convert_to_plot_coordinates(target_pos[1], env.height))
            quality = target['hit_probability']

            # Draw line to target
            ax4.plot(
                [plot_pos[0], plot_target[0]],
                [plot_pos[1], plot_target[1]],
                'g--',
                alpha=quality * 0.5
            )

        # Update observation and engagement matrices
        # Initialize matrices as float arrays
        observation_matrix = np.zeros((env.height, env.width), dtype=float)
        engagement_matrix = np.zeros((env.height, env.width), dtype=float)

        # For observation arc
        obs_data = assault_analysis['observation_arc']
        try:
            if isinstance(obs_data, dict):
                for pos, effect in obs_data.items():
                    x, y = map(int, pos)  # Convert to integers
                    plot_y = convert_to_plot_coordinates(y, env.height)
                    curr_val = float(observation_matrix[plot_y, x])
                    new_val = float(effect)
                    observation_matrix[plot_y, x] = max(curr_val, new_val)
            else:
                # Handle as list/array
                for item in obs_data:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        pos, effect = item
                        if isinstance(pos, (list, tuple, np.ndarray)):
                            x, y = map(int, pos)
                        else:
                            x, y = map(int, [pos[0], pos[1]])
                        plot_y = convert_to_plot_coordinates(y, env.height)
                        curr_val = float(observation_matrix[plot_y, x])
                        new_val = float(effect)
                        observation_matrix[plot_y, x] = max(curr_val, new_val)
        except Exception as e:
            print(f"Error processing observation arc: {str(e)}")
            print(f"Data type: {type(obs_data)}")
            print(f"Data sample: {obs_data[:5] if hasattr(obs_data, '__getitem__') else obs_data}")

        # For engagement arc
        eng_data = assault_analysis['engagement_arc']
        try:
            if isinstance(eng_data, dict):
                for pos, effect in eng_data.items():
                    x, y = map(int, pos)  # Convert to integers
                    plot_y = convert_to_plot_coordinates(y, env.height)
                    curr_val = float(engagement_matrix[plot_y, x])
                    new_val = float(effect)
                    engagement_matrix[plot_y, x] = max(curr_val, new_val)
            else:
                # Handle as list/array
                for item in eng_data:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        pos, effect = item
                        if isinstance(pos, (list, tuple, np.ndarray)):
                            x, y = map(int, pos)
                        else:
                            x, y = map(int, [pos[0], pos[1]])
                        plot_y = convert_to_plot_coordinates(y, env.height)
                        curr_val = float(engagement_matrix[plot_y, x])
                        new_val = float(effect)
                        engagement_matrix[plot_y, x] = max(curr_val, new_val)
        except Exception as e:
            print(f"Error processing engagement arc: {str(e)}")
            print(f"Data type: {type(eng_data)}")
            print(f"Data sample: {eng_data[:5] if hasattr(eng_data, '__getitem__') else eng_data}")

        # Plot position with quality indicator
        cmap = plt.get_cmap('Blues')  # Using 'Blues' colormap
        color = cmap(float(assault_analysis['quality_scores']['overall_quality']))
        ax4.scatter(plot_pos[0], plot_pos[1], c=[color], s=100)

        # Add label
        ax4.annotate(
            f"{friendly['name']}\n{assault_analysis['quality_scores']['overall_quality']:.2f}",
            (plot_pos[0], plot_pos[1]),
            xytext=(5, 5),
            textcoords='offset points'
        )

    # Plot enemy targets
    for target in enemy_targets:
        plot_target = (target[0], convert_to_plot_coordinates(target[1], env.height))
        ax4.scatter(plot_target[0], plot_target[1], c='red', marker='^', s=100)

    # Plot observation analysis
    obs_overlay = ax1.imshow(
        observation_matrix,
        alpha=0.5,
        cmap='Blues',
        vmin=0,
        vmax=1,
        origin='lower'
    )
    ax1.set_title('Friendly Observation Coverage')
    plt.colorbar(obs_overlay, ax=ax1)

    # Plot engagement analysis
    eng_overlay = ax2.imshow(
        engagement_matrix,
        alpha=0.5,
        cmap='Greens',
        vmin=0,
        vmax=1,
        origin='lower'
    )
    ax2.set_title('Friendly Engagement Coverage')
    plt.colorbar(eng_overlay, ax=ax2)

    # Plot terrain effects analysis
    terrain_effects = np.zeros((env.height, env.width))
    for y in range(env.height):
        plot_y = convert_to_plot_coordinates(y, env.height)
        for x in range(env.width):
            cover = env.terrain_manager.get_cover((x, y))
            visibility = env.terrain_manager.get_visibility((x, y))
            terrain_effects[plot_y, x] = (cover * 0.5 + (1 - visibility) * 0.5)

    terrain_overlay = ax3.imshow(
        terrain_effects,
        alpha=0.5,
        cmap='YlOrBr',
        vmin=0,
        vmax=1,
        origin='lower'
    )
    ax3.set_title('Terrain Effects Analysis')
    plt.colorbar(terrain_overlay, ax=ax3)

    ax4.set_title('Direct Fire Analysis Overview')

    # Set proper axis labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('X (East →)')
        ax.set_ylabel('Y (North ↑)')

    plt.tight_layout()
    plt.savefig('direct_fire_analysis_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nDirect fire analysis visualization saved as 'direct_fire_analysis_test.png'")
    print("Note: Visualization uses standard plotting coordinates but all calculations")
    print("      use military environment coordinates (0,0 at bottom left)")


def test_tactical_filter():
    """Test tactical position filtering and analysis with visualization."""
    print("\nTesting Tactical Position Filter")
    print("=" * 80)

    # Initialize environment
    config = EnvironmentConfig(width=400, height=100, debug_level=0)
    env = MilitaryEnvironment(config)
    env.terrain_manager.initialize_terrain(env.state_manager.state_tensor)

    try:
        # Try to load terrain from CSV if available
        csv_path = 'generated_map.csv'  # 'generated_map.csv'
        env.terrain_manager.load_from_csv(csv_path)
        print("Loaded terrain data from CSV")
    except Exception as e:
        print(f"Using default terrain: {str(e)}")

    # Create test scenario
    objective = (85, 50)
    start_positions = [(20, 45), (20, 50), (20, 55)]
    enemy_threats = [
        {
            'position': (80, 50),
            'observation_range': 48,  # 480m
            'engagement_range': 30,  # 300m
            'accuracy': 0.8
        },
        {
            'position': (75, 48),
            'observation_range': 40,  # 400m
            'engagement_range': 25,  # 250m
            'accuracy': 0.7
        },
        {
            'position': (77, 52),
            'observation_range': 35,  # 350m
            'engagement_range': 20,  # 200m
            'accuracy': 0.6
        }
    ]

    # Initialize tactical filter
    tactical_filter = TacticalFilter(env)

    # Test different unit types and purposes
    test_cases = [
        (UnitType.INFANTRY_TEAM, PositionPurpose.ASSAULT),
        (UnitType.INFANTRY_TEAM, PositionPurpose.SUPPORT),
        (UnitType.INFANTRY_SQUAD, PositionPurpose.SUPPORT)
    ]

    for unit_type, position_purpose in test_cases:
        print("\n" + "=" * 80)
        print(f"Testing {unit_type.name} {position_purpose.value} Position Search")
        print("=" * 80)

        try:
            # Find positions
            results = tactical_filter.find_positions(
                objective=objective,
                unit_type=unit_type,
                position_purpose=position_purpose,
                start_positions=start_positions,
                enemy_threats=enemy_threats,
                max_positions=5
            )

            # Create visualization
            visualize_tactical_positions(
                env=env,
                results=results,
                objective=objective,
                enemy_threats=enemy_threats,
                unit_type=unit_type
            )

            # Results to TacticalPosition format
            print("\nTactical Position details:")
            if results['tactical_positions']:
                for i, pos in enumerate(results['tactical_positions'], 1):
                    print(f"\nPosition {i}:")
                    print(f"Position: {pos.position}")
                    print(f"Unit Size: {pos.unit_size}")
                    print(f"Quality Score: {pos.quality_score:.2f}")
                    print(f"Cover Score: {pos.cover_score:.2f}")
                    print(f"Concealment Score: {pos.concealment_score:.2f}")
                    print(f"Number of cells: {len(pos.cells)}")

        except Exception as e:
            print(f"Error during position search: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    print("\nRunning tactical position analyzer tests...")

    # test_threat_analysis()
    # test_direct_fire_analysis()
    test_tactical_filter()

    print("\nTests complete")
