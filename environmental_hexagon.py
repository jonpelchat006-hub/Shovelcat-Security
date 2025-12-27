"""
ENVIRONMENTAL HEXAGON SYSTEM
============================
Nested hexagons with observer security, environmental composition,
and phase state dynamics.

Key Concepts:
1. Nested Rings: Core → Inner → Middle → Outer (coupling strength)
2. Observer Network: Multiple observers at different angles
3. Tilt Angles: Entities can hide by tilting between observers
4. Environmental Axes: Trust (O2), Authenticity (Salt), Activity (Heat)
5. Phase States: Solid, Liquid, Gas, Plasma (top and bottom can differ)
6. Blind Spot Detection: Security vulnerability when entities hide

The Salinity-Legitimacy Connection:
- Fresh water (0 ppt) = Clean money (90°)
- Ocean water (35 ppt) = Questionable money (30°)
- Salt accumulates through transfers just like provenance degrades

Author: Jonathan Pelchat
Based on Shovelcat Theory - Saturn Vortex + Hydrology of Economics
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum, auto
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2

# Environmental constants
EARTH_O2 = 21.0  # Normal atmospheric oxygen %
OCEAN_SALT = 35.0  # Standard ocean salinity ppt
ROOM_TEMP = 293.0  # Room temperature in Kelvin


# ═══════════════════════════════════════════════════════════════════════════════
# NESTED HEXAGON RINGS
# ═══════════════════════════════════════════════════════════════════════════════

class HexRing(Enum):
    """
    Nested hexagon rings - from core outward.
    Each ring has different coupling strength and mesh friction.
    """
    CORE = ("CORE", 0, 0.00, "Internal, same entity")
    INNER = ("INNER", 1, 0.15, "Subsidiaries, owned entities")
    MIDDLE = ("MIDDLE", 2, 0.40, "Partners, key suppliers")
    OUTER = ("OUTER", 3, 0.60, "Industry peers, associations")
    BEYOND = ("BEYOND", 4, 0.80, "Strangers, arm's length")
    
    def __init__(self, name: str, level: int, base_friction: float, description: str):
        self._name = name
        self.level = level
        self.base_friction = base_friction
        self.description = description
    
    @property
    def max_friction(self) -> float:
        """Maximum friction before connection breaks."""
        return self.base_friction + 0.15
    
    @classmethod
    def from_friction(cls, friction: float) -> 'HexRing':
        """Determine ring from measured friction."""
        for ring in cls:
            if friction <= ring.max_friction:
                return ring
        return cls.BEYOND


@dataclass
class NestedHexagon:
    """
    A hexagon with multiple nested rings.
    Each ring has different properties and entities.
    """
    core_radius: float = 1.0
    ring_spacing: float = 0.5
    
    def get_ring_radius(self, ring: HexRing) -> float:
        """Get radius for a specific ring."""
        return self.core_radius + (ring.level * self.ring_spacing)
    
    def get_ring_area(self, ring: HexRing) -> float:
        """Get area of a ring (annulus)."""
        if ring == HexRing.CORE:
            return PI * self.core_radius ** 2
        
        outer_r = self.get_ring_radius(ring)
        # Find the inner ring by level
        inner_ring = None
        for r in HexRing:
            if r.level == ring.level - 1:
                inner_ring = r
                break
        inner_r = self.get_ring_radius(inner_ring) if inner_ring else self.core_radius
        return PI * (outer_r ** 2 - inner_r ** 2)
    
    def get_ring_capacity(self, ring: HexRing, density: float = 10.0) -> int:
        """How many entities can fit in this ring."""
        return int(self.get_ring_area(ring) * density)


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVER NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Observer:
    """
    An observer watching the hexagon from a specific angle.
    
    Observers are orthogonal to the domain but can see projections
    of entities based on their tilt angles.
    """
    observer_id: str
    watch_angle: float  # Angle in degrees (0-360)
    coverage_width: float = 30.0  # How wide their view is (degrees)
    authority: float = 1.0  # Weight of this observer's assessment
    
    observer_type: str = "GENERAL"  # BOARD, AUDITOR, REGULATOR, etc.
    
    def can_see_angle(self, entity_tilt: float) -> bool:
        """Can this observer see an entity tilted at given angle?"""
        # Normalize angles
        diff = abs((entity_tilt - self.watch_angle + 180) % 360 - 180)
        return diff <= self.coverage_width / 2
    
    def visibility_score(self, entity_tilt: float) -> float:
        """How well can this observer see the entity? (0-1)"""
        diff = abs((entity_tilt - self.watch_angle + 180) % 360 - 180)
        if diff > self.coverage_width / 2:
            return 0.0
        # Cosine falloff from center of view
        return math.cos(PI * diff / self.coverage_width)
    
    def get_coverage_range(self) -> Tuple[float, float]:
        """Get the angular range this observer covers."""
        half = self.coverage_width / 2
        start = (self.watch_angle - half) % 360
        end = (self.watch_angle + half) % 360
        return (start, end)


class ObserverNetwork:
    """
    Network of observers providing coverage of the hexagon.
    Detects blind spots where entities can hide.
    """
    
    def __init__(self):
        self.observers: Dict[str, Observer] = {}
        self._coverage_cache: Optional[List[float]] = None
    
    def add_observer(self, observer: Observer):
        """Add an observer to the network."""
        self.observers[observer.observer_id] = observer
        self._coverage_cache = None  # Invalidate cache
    
    def remove_observer(self, observer_id: str):
        """Remove an observer."""
        if observer_id in self.observers:
            del self.observers[observer_id]
            self._coverage_cache = None
    
    def get_coverage_map(self, resolution: int = 360) -> List[float]:
        """
        Compute coverage at each angle (0-359 degrees).
        Returns list of coverage scores.
        """
        if self._coverage_cache is not None:
            return self._coverage_cache
        
        coverage = [0.0] * resolution
        
        for angle in range(resolution):
            for obs in self.observers.values():
                coverage[angle] += obs.visibility_score(angle) * obs.authority
        
        self._coverage_cache = coverage
        return coverage
    
    def find_blind_spots(self, threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        Find angular ranges with insufficient coverage.
        Returns list of (start_angle, end_angle) tuples.
        """
        coverage = self.get_coverage_map()
        blind_spots = []
        
        in_blind_spot = False
        start_angle = 0
        
        for angle, cov in enumerate(coverage):
            if cov < threshold and not in_blind_spot:
                in_blind_spot = True
                start_angle = angle
            elif cov >= threshold and in_blind_spot:
                in_blind_spot = False
                blind_spots.append((start_angle, angle - 1))
        
        # Handle wrap-around
        if in_blind_spot:
            blind_spots.append((start_angle, 359))
        
        return blind_spots
    
    def get_total_coverage(self) -> float:
        """Get percentage of angles with sufficient coverage."""
        coverage = self.get_coverage_map()
        covered = sum(1 for c in coverage if c >= 0.3)
        return covered / len(coverage)
    
    def is_entity_visible(self, entity_tilt: float) -> Tuple[bool, float, List[str]]:
        """
        Check if an entity at given tilt is visible.
        Returns (is_visible, visibility_score, list_of_observers_who_see_it)
        """
        total_visibility = 0.0
        seeing_observers = []
        
        for obs_id, obs in self.observers.items():
            vis = obs.visibility_score(entity_tilt)
            if vis > 0.1:
                total_visibility += vis * obs.authority
                seeing_observers.append(obs_id)
        
        is_visible = total_visibility >= 0.3
        return (is_visible, total_visibility, seeing_observers)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

class AlignmentStatus(Enum):
    """Entity alignment certification status."""
    FULLY_ALIGNED = ("FULLY_ALIGNED", 0, 15, "Expected, all observers see")
    SLIGHTLY_TILTED = ("SLIGHTLY_TILTED", 15, 30, "Explain required")
    SIGNIFICANTLY_TILTED = ("SIGNIFICANTLY_TILTED", 30, 45, "Flag for review")
    MOSTLY_HIDDEN = ("MOSTLY_HIDDEN", 45, 60, "Investigate")
    FULLY_HIDDEN = ("FULLY_HIDDEN", 60, 90, "Reject/block")
    
    def __init__(self, name: str, min_tilt: float, max_tilt: float, action: str):
        self._name = name
        self.min_tilt = min_tilt
        self.max_tilt = max_tilt
        self.action = action
    
    @classmethod
    def from_tilt(cls, tilt: float) -> 'AlignmentStatus':
        """Get status from tilt angle."""
        tilt = abs(tilt) % 90
        for status in cls:
            if status.min_tilt <= tilt < status.max_tilt:
                return status
        return cls.FULLY_HIDDEN


@dataclass
class ConnectedEntity:
    """
    An entity connected to the hexagon with alignment properties.
    """
    entity_id: str
    entity_type: str  # SUBSIDIARY, PARTNER, VENDOR, SHELL, etc.
    
    # Declared vs actual tilt (mismatch = fraud indicator)
    declared_tilt: float = 0.0
    actual_tilt: float = 0.0
    
    # Connection properties
    ring: HexRing = HexRing.OUTER
    slot: int = 0  # Which of 6 hexagon vertices (0-5)
    
    # Flow properties
    inflow: float = 0.0  # Money/value flowing IN
    outflow: float = 0.0  # Money/value flowing OUT
    
    def get_alignment_status(self) -> AlignmentStatus:
        """Get declared alignment status."""
        return AlignmentStatus.from_tilt(self.declared_tilt)
    
    def get_tilt_mismatch(self) -> float:
        """Difference between declared and actual tilt."""
        return abs(self.actual_tilt - self.declared_tilt)
    
    def is_suspicious(self, threshold: float = 10.0) -> bool:
        """Is there a suspicious mismatch?"""
        return self.get_tilt_mismatch() > threshold
    
    def get_projection(self) -> float:
        """How much of this entity is visible (0-1)."""
        return math.cos(math.radians(self.actual_tilt))


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENTAL AXES
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseState(Enum):
    """Physical phase states for organizational environments."""
    VACUUM = ("VACUUM", 0.0, "No transactions possible")
    PLASMA = ("PLASMA", 0.1, "Too hot, structure melts")
    GAS = ("GAS", 0.3, "Diffuse, fast, low friction")
    LIQUID = ("LIQUID", 0.6, "Fluid, moderate, standard")
    SOLID = ("SOLID", 0.9, "Dense, slow, high friction")
    FROZEN = ("FROZEN", 1.0, "No movement, stagnant")
    
    def __init__(self, name: str, density: float, description: str):
        self._name = name
        self.density = density
        self.description = description


@dataclass
class EnvironmentalComposition:
    """
    The three environmental axes plus phase states.
    
    Trust (O2): Can transactions "breathe" here?
    Authenticity (Salt): How pure is the flow?
    Activity (Heat): How fast is the flow?
    """
    
    # Trust = Oxygen level (0-100%)
    trust_o2: float = EARTH_O2  # Default: Earth-like
    
    # Authenticity = Salinity (0-100+ ppt)
    authenticity_salt: float = 0.0  # Default: Fresh
    
    # Activity = Temperature (Kelvin)
    activity_heat: float = ROOM_TEMP  # Default: Room temp
    
    # Phase states (can differ top vs bottom)
    top_phase: PhaseState = PhaseState.GAS
    bottom_phase: PhaseState = PhaseState.SOLID
    
    def get_trust_level(self) -> str:
        """Interpret trust level."""
        if self.trust_o2 < 5:
            return "VACUUM"
        elif self.trust_o2 < 15:
            return "LOW"
        elif self.trust_o2 < 30:
            return "NORMAL"
        elif self.trust_o2 < 50:
            return "ENRICHED"
        else:
            return "DANGEROUS"  # Too much trust = fraud risk
    
    def get_authenticity_level(self) -> str:
        """Interpret authenticity level (inverse of salt)."""
        if self.authenticity_salt < 5:
            return "PRISTINE"
        elif self.authenticity_salt < 15:
            return "CLEAN"
        elif self.authenticity_salt < 35:
            return "BRACKISH"
        elif self.authenticity_salt < 70:
            return "SALTY"
        else:
            return "BRINE"  # Over-saturated
    
    def get_activity_level(self) -> str:
        """Interpret activity level."""
        if self.activity_heat < 200:
            return "FROZEN"
        elif self.activity_heat < 273:
            return "COLD"
        elif self.activity_heat < 310:
            return "WARM"
        elif self.activity_heat < 373:
            return "HOT"
        else:
            return "PLASMA"
    
    def get_legitimacy_angle(self) -> float:
        """Convert salt to legitimacy angle (inverse relationship)."""
        # 0 ppt = 90°, 100 ppt = 0°
        return max(0, 90 * (1 - self.authenticity_salt / 100))
    
    def is_stable(self) -> Tuple[bool, List[str]]:
        """Check if environment supports stable hexagon."""
        issues = []
        
        # Trust bounds
        if self.trust_o2 < 15:
            issues.append(f"Trust too low ({self.trust_o2}% O2)")
        if self.trust_o2 > 50:
            issues.append(f"Trust dangerously high ({self.trust_o2}% O2)")
        
        # Salt bounds
        if self.authenticity_salt > 70:
            issues.append(f"Over-saturated ({self.authenticity_salt} ppt)")
        
        # Temperature bounds
        if self.activity_heat < 200:
            issues.append(f"Too cold ({self.activity_heat}K)")
        if self.activity_heat > 373:
            issues.append(f"Too hot ({self.activity_heat}K)")
        
        # Phase compatibility
        if self.top_phase == PhaseState.PLASMA or self.bottom_phase == PhaseState.PLASMA:
            issues.append("Plasma phase unstable")
        if self.top_phase == PhaseState.VACUUM or self.bottom_phase == PhaseState.VACUUM:
            issues.append("Vacuum phase unsustainable")
        
        return len(issues) == 0, issues
    
    def compute_flow_rate(self, base_rate: float = 1.0) -> float:
        """
        Compute flow rate based on environment.
        
        F = F_base × (O2/O2_normal) × (1 - Salt/Salt_max) × f(Temp)
        """
        o2_factor = self.trust_o2 / EARTH_O2
        salt_factor = 1 - (self.authenticity_salt / 100)
        
        # Temperature factor peaks at optimal (~300K)
        temp_factor = math.exp(-((self.activity_heat - 300) / 100) ** 2)
        
        return base_rate * o2_factor * salt_factor * temp_factor


# ═══════════════════════════════════════════════════════════════════════════════
# SALINITY-LEGITIMACY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

class SalinityLegitimacyMapper:
    """
    Maps between salinity (ppt) and legitimacy angle (degrees).
    
    Fresh water = Clean money
    Salt water = Questionable money
    
    Just as water picks up minerals through cycles,
    money picks up provenance issues through transfers.
    """
    
    # Salinity to legitimacy mapping
    SALINITY_BANDS = [
        (0, 5, 90, 75, "PRISTINE", "Fresh aquifer"),
        (5, 15, 75, 60, "CLEAN", "River system"),
        (15, 35, 60, 45, "BRACKISH", "Estuary mixing"),
        (35, 50, 45, 30, "OCEAN", "Open ocean"),
        (50, 70, 30, 15, "SALTY", "Evaporation basin"),
        (70, 100, 15, 0, "BRINE", "Salt flat"),
    ]
    
    @classmethod
    def salt_to_legitimacy(cls, salt_ppt: float) -> float:
        """Convert salinity to legitimacy angle."""
        for min_salt, max_salt, max_angle, min_angle, _, _ in cls.SALINITY_BANDS:
            if min_salt <= salt_ppt < max_salt:
                # Linear interpolation within band
                t = (salt_ppt - min_salt) / (max_salt - min_salt)
                return max_angle - t * (max_angle - min_angle)
        return 0.0  # Beyond brine
    
    @classmethod
    def legitimacy_to_salt(cls, angle: float) -> float:
        """Convert legitimacy angle to salinity."""
        for min_salt, max_salt, max_angle, min_angle, _, _ in cls.SALINITY_BANDS:
            if min_angle <= angle <= max_angle:
                # Linear interpolation within band
                t = (max_angle - angle) / (max_angle - min_angle)
                return min_salt + t * (max_salt - min_salt)
        return 100.0  # Beyond brine
    
    @classmethod
    def get_band_name(cls, salt_ppt: float) -> str:
        """Get the band name for a salinity level."""
        for min_salt, max_salt, _, _, name, _ in cls.SALINITY_BANDS:
            if min_salt <= salt_ppt < max_salt:
                return name
        return "REJECTED"
    
    @classmethod
    def get_water_type(cls, salt_ppt: float) -> str:
        """Get the water type description."""
        for min_salt, max_salt, _, _, _, water_type in cls.SALINITY_BANDS:
            if min_salt <= salt_ppt < max_salt:
                return water_type
        return "Dead sea"


# ═══════════════════════════════════════════════════════════════════════════════
# BUSINESS ENVIRONMENT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class BusinessEnvironmentType(Enum):
    """
    Different business types based on top/bottom phase combinations.
    """
    CONSUMER_APP = ("CONSUMER_APP", PhaseState.GAS, PhaseState.SOLID,
                   "Many users, small team, stable foundation")
    PLATFORM = ("PLATFORM", PhaseState.GAS, PhaseState.GAS,
               "Light on both sides, gig economy")
    INSTITUTIONAL = ("INSTITUTIONAL", PhaseState.LIQUID, PhaseState.LIQUID,
                    "B2B, thick both sides, banks/exchanges")
    HEAVY_INDUSTRY = ("HEAVY_INDUSTRY", PhaseState.SOLID, PhaseState.SOLID,
                     "Few customers, few workers, high density")
    CONSUMER_ENTERPRISE = ("CONSUMER_ENTERPRISE", PhaseState.GAS, PhaseState.LIQUID,
                          "Many users, substantial operations")
    INVERTED = ("INVERTED", PhaseState.SOLID, PhaseState.GAS,
               "Few premium customers, massive contractor network")
    
    def __init__(self, name: str, top: PhaseState, bottom: PhaseState, description: str):
        self._name = name
        self.top_phase = top
        self.bottom_phase = bottom
        self.description = description
    
    @classmethod
    def detect_type(cls, top: PhaseState, bottom: PhaseState) -> 'BusinessEnvironmentType':
        """Detect business type from phases."""
        for env_type in cls:
            if env_type.top_phase == top and env_type.bottom_phase == bottom:
                return env_type
        return cls.PLATFORM  # Default


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE ENVIRONMENTAL HEXAGON
# ═══════════════════════════════════════════════════════════════════════════════

class EnvironmentalHexagon:
    """
    Complete hexagon system with:
    - Nested rings
    - Observer network
    - Environmental composition
    - Connected entities
    - Security analysis
    """
    
    def __init__(self, hexagon_id: str):
        self.hexagon_id = hexagon_id
        
        # Structure
        self.nested = NestedHexagon()
        
        # Observers
        self.observer_network = ObserverNetwork()
        
        # Environment
        self.environment = EnvironmentalComposition()
        
        # Connected entities
        self.entities: Dict[str, ConnectedEntity] = {}
        
        # Flow tracking
        self.total_inflow = 0.0
        self.total_outflow = 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Observer Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_observer(self, 
                    observer_id: str,
                    watch_angle: float,
                    observer_type: str = "GENERAL",
                    coverage_width: float = 30.0,
                    authority: float = 1.0):
        """Add an observer to the network."""
        obs = Observer(
            observer_id=observer_id,
            watch_angle=watch_angle,
            coverage_width=coverage_width,
            authority=authority,
            observer_type=observer_type
        )
        self.observer_network.add_observer(obs)
    
    def setup_standard_observers(self):
        """Set up a standard observer configuration."""
        # Board at top
        self.add_observer("board", 0, "BOARD", 45, 1.2)
        # Auditor at 120°
        self.add_observer("auditor", 120, "AUDITOR", 40, 1.0)
        # Regulator at 240°
        self.add_observer("regulator", 240, "REGULATOR", 40, 1.5)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Entity Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def connect_entity(self,
                      entity_id: str,
                      entity_type: str,
                      ring: HexRing,
                      slot: int = 0,
                      declared_tilt: float = 0.0,
                      actual_tilt: float = None) -> ConnectedEntity:
        """Connect an entity to the hexagon."""
        if actual_tilt is None:
            actual_tilt = declared_tilt
        
        entity = ConnectedEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            declared_tilt=declared_tilt,
            actual_tilt=actual_tilt,
            ring=ring,
            slot=slot
        )
        self.entities[entity_id] = entity
        return entity
    
    def record_flow(self, entity_id: str, inflow: float = 0.0, outflow: float = 0.0):
        """Record flow through an entity."""
        if entity_id in self.entities:
            self.entities[entity_id].inflow += inflow
            self.entities[entity_id].outflow += outflow
            self.total_inflow += inflow
            self.total_outflow += outflow
    
    # ─────────────────────────────────────────────────────────────────────────
    # Security Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def analyze_entity_visibility(self, entity_id: str) -> Dict[str, Any]:
        """Analyze visibility of a specific entity."""
        if entity_id not in self.entities:
            return {"error": "Entity not found"}
        
        entity = self.entities[entity_id]
        is_visible, score, observers = self.observer_network.is_entity_visible(
            entity.actual_tilt
        )
        
        return {
            "entity_id": entity_id,
            "declared_tilt": entity.declared_tilt,
            "actual_tilt": entity.actual_tilt,
            "tilt_mismatch": entity.get_tilt_mismatch(),
            "alignment_status": entity.get_alignment_status().name,
            "is_visible": is_visible,
            "visibility_score": score,
            "observing": observers,
            "suspicious": entity.is_suspicious(),
            "projection": entity.get_projection()
        }
    
    def find_hidden_entities(self) -> List[Dict[str, Any]]:
        """Find entities that are hiding in blind spots."""
        hidden = []
        
        for entity_id, entity in self.entities.items():
            is_visible, score, _ = self.observer_network.is_entity_visible(
                entity.actual_tilt
            )
            
            if not is_visible:
                hidden.append({
                    "entity_id": entity_id,
                    "entity_type": entity.entity_type,
                    "actual_tilt": entity.actual_tilt,
                    "visibility_score": score,
                    "flow_through": entity.inflow + entity.outflow
                })
        
        return hidden
    
    def find_mismatched_declarations(self, threshold: float = 10.0) -> List[Dict[str, Any]]:
        """Find entities with suspicious tilt mismatches."""
        mismatches = []
        
        for entity_id, entity in self.entities.items():
            mismatch = entity.get_tilt_mismatch()
            if mismatch > threshold:
                mismatches.append({
                    "entity_id": entity_id,
                    "entity_type": entity.entity_type,
                    "declared_tilt": entity.declared_tilt,
                    "actual_tilt": entity.actual_tilt,
                    "mismatch": mismatch,
                    "flow_through": entity.inflow + entity.outflow
                })
        
        return mismatches
    
    def security_audit(self) -> Dict[str, Any]:
        """Perform complete security audit."""
        blind_spots = self.observer_network.find_blind_spots()
        hidden_entities = self.find_hidden_entities()
        mismatched = self.find_mismatched_declarations()
        coverage = self.observer_network.get_total_coverage()
        env_stable, env_issues = self.environment.is_stable()
        
        # Count entities by alignment status
        alignment_counts = defaultdict(int)
        for entity in self.entities.values():
            alignment_counts[entity.get_alignment_status().name] += 1
        
        # Risk score
        risk_score = 0.0
        risk_score += len(blind_spots) * 10
        risk_score += len(hidden_entities) * 25
        risk_score += len(mismatched) * 20
        risk_score += (1 - coverage) * 30
        risk_score += len(env_issues) * 15
        
        return {
            "hexagon_id": self.hexagon_id,
            "total_entities": len(self.entities),
            "observer_coverage": coverage,
            "blind_spots": blind_spots,
            "hidden_entities": hidden_entities,
            "declaration_mismatches": mismatched,
            "alignment_distribution": dict(alignment_counts),
            "environment_stable": env_stable,
            "environment_issues": env_issues,
            "risk_score": min(100, risk_score),
            "status": "SECURE" if risk_score < 20 else 
                     "CAUTION" if risk_score < 50 else
                     "WARNING" if risk_score < 75 else "CRITICAL"
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Environmental Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_business_type(self) -> BusinessEnvironmentType:
        """Detect business type from environment."""
        return BusinessEnvironmentType.detect_type(
            self.environment.top_phase,
            self.environment.bottom_phase
        )
    
    def get_flow_capacity(self) -> float:
        """Get maximum sustainable flow rate."""
        return self.environment.compute_flow_rate(100.0)
    
    def check_flow_composition(self, salt_ppt: float) -> Dict[str, Any]:
        """Check if incoming flow matches environment."""
        env_salt = self.environment.authenticity_salt
        salt_diff = abs(salt_ppt - env_salt)
        
        # Contamination risk
        if salt_ppt > env_salt + 20:
            status = "CONTAMINATION_RISK"
            message = f"Incoming flow ({salt_ppt} ppt) saltier than environment ({env_salt} ppt)"
        elif salt_ppt < env_salt - 20:
            status = "DILUTION"
            message = f"Incoming flow ({salt_ppt} ppt) fresher than environment ({env_salt} ppt)"
        else:
            status = "COMPATIBLE"
            message = "Flow composition matches environment"
        
        return {
            "incoming_salt": salt_ppt,
            "environment_salt": env_salt,
            "difference": salt_diff,
            "incoming_legitimacy": SalinityLegitimacyMapper.salt_to_legitimacy(salt_ppt),
            "environment_legitimacy": SalinityLegitimacyMapper.salt_to_legitimacy(env_salt),
            "status": status,
            "message": message
        }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get complete state summary."""
        business_type = self.get_business_type()
        env_stable, env_issues = self.environment.is_stable()
        
        return {
            "hexagon_id": self.hexagon_id,
            "business_type": business_type.name,
            "business_description": business_type.description,
            "environment": {
                "trust_o2": self.environment.trust_o2,
                "trust_level": self.environment.get_trust_level(),
                "authenticity_salt": self.environment.authenticity_salt,
                "authenticity_level": self.environment.get_authenticity_level(),
                "activity_heat": self.environment.activity_heat,
                "activity_level": self.environment.get_activity_level(),
                "top_phase": self.environment.top_phase.name,
                "bottom_phase": self.environment.bottom_phase.name,
                "stable": env_stable,
                "issues": env_issues
            },
            "structure": {
                "total_entities": len(self.entities),
                "total_observers": len(self.observer_network.observers),
                "observer_coverage": self.observer_network.get_total_coverage(),
                "blind_spots": len(self.observer_network.find_blind_spots())
            },
            "flows": {
                "total_inflow": self.total_inflow,
                "total_outflow": self.total_outflow,
                "net_flow": self.total_inflow - self.total_outflow,
                "flow_capacity": self.get_flow_capacity()
            },
            "legitimacy_angle": self.environment.get_legitimacy_angle()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_nested_rings():
    """Demonstrate nested hexagon rings."""
    print("\n" + "=" * 80)
    print("NESTED HEXAGON RINGS")
    print("=" * 80)
    
    nested = NestedHexagon(core_radius=1.0, ring_spacing=0.5)
    
    print("""
    Looking DOWN at the hexagon (top view):
    
                    OUTER RING (Strangers)
                   ╱                      ╲
                  ╱   MIDDLE (Partners)    ╲
                 ╱   ╱                  ╲   ╲
                ╱   ╱   INNER (Owned)   ╲   ╲
               ╱   ╱   ╱            ╲   ╲   ╲
              │   │   │    CORE     │   │   │
               ╲   ╲   ╲            ╱   ╱   ╱
                ╲   ╲   ╲__________╱   ╱   ╱
                 ╲   ╲________________╱   ╱
                  ╲______________________╱
    """)
    
    print("\n┌────────────┬────────┬───────────┬──────────┬─────────────────────────┐")
    print("│ Ring       │ Level  │ Friction  │ Radius   │ Capacity                │")
    print("├────────────┼────────┼───────────┼──────────┼─────────────────────────┤")
    
    for ring in HexRing:
        radius = nested.get_ring_radius(ring)
        capacity = nested.get_ring_capacity(ring)
        print(f"│ {ring.name:10} │ {ring.level:6} │ {ring.base_friction:9.2f} │ {radius:8.2f} │ {capacity:23} │")
    
    print("└────────────┴────────┴───────────┴──────────┴─────────────────────────┘")


def demo_observer_network():
    """Demonstrate observer network and blind spots."""
    print("\n" + "=" * 80)
    print("OBSERVER NETWORK AND BLIND SPOTS")
    print("=" * 80)
    
    network = ObserverNetwork()
    
    # Add some observers
    network.add_observer(Observer("board", 0, 45, 1.2, "BOARD"))
    network.add_observer(Observer("auditor", 120, 40, 1.0, "AUDITOR"))
    network.add_observer(Observer("regulator", 240, 40, 1.5, "REGULATOR"))
    
    print("""
    Observer positions (top view of hexagon):
    
                     BOARD (0°)
                        │
                        │ coverage: ±22.5°
                        ▼
              ╲                   ╱
               ╲                 ╱
                ╲     COMPANY   ╱
                 ╲             ╱
                  ╲___________╱
                 ╱             ╲
                ╱               ╲
         REGULATOR (240°)    AUDITOR (120°)
         coverage: ±20°      coverage: ±20°
    """)
    
    blind_spots = network.find_blind_spots()
    coverage = network.get_total_coverage()
    
    print(f"\nTotal Coverage: {coverage * 100:.1f}%")
    print(f"Blind Spots Found: {len(blind_spots)}")
    
    if blind_spots:
        print("\n┌──────────────────┬────────────────────────────────────────────┐")
        print("│ Blind Spot       │ Angles (degrees)                           │")
        print("├──────────────────┼────────────────────────────────────────────┤")
        for i, (start, end) in enumerate(blind_spots):
            print(f"│ Spot {i+1:12} │ {start:3}° to {end:3}° ({end-start+1}° span)                  │")
        print("└──────────────────┴────────────────────────────────────────────┘")
    
    # Test entity visibility
    print("\n--- Entity Visibility Tests ---")
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    print("\n┌──────────┬───────────┬───────────────────┬───────────────────────┐")
    print("│ Tilt °   │ Visible?  │ Score             │ Seen By               │")
    print("├──────────┼───────────┼───────────────────┼───────────────────────┤")
    
    for angle in test_angles:
        visible, score, observers = network.is_entity_visible(angle)
        obs_str = ", ".join(observers) if observers else "NONE"
        status = "YES" if visible else "NO"
        print(f"│ {angle:8} │ {status:9} │ {score:17.2f} │ {obs_str:21} │")
    
    print("└──────────┴───────────┴───────────────────┴───────────────────────┘")


def demo_salinity_legitimacy():
    """Demonstrate salinity-legitimacy mapping."""
    print("\n" + "=" * 80)
    print("SALINITY-LEGITIMACY MAPPING")
    print("=" * 80)
    
    print("""
    Fresh water = Clean money    |    Salt water = Questionable money
    
    Water picks up minerals as it flows through the cycle.
    Money picks up provenance issues as it flows through transfers.
    
         Rain (pure)  →  River  →  Lake  →  Ocean  →  Evaporate
              0 ppt      0.5 ppt   varies    35 ppt
              90°         85°       60°       30°
    """)
    
    print("\n┌──────────────┬──────────────┬────────────────┬───────────────────┐")
    print("│ Salinity     │ Legitimacy   │ Band           │ Water Type        │")
    print("│ (ppt)        │ (degrees)    │                │                   │")
    print("├──────────────┼──────────────┼────────────────┼───────────────────┤")
    
    test_salts = [0, 3, 10, 25, 35, 50, 70, 100]
    for salt in test_salts:
        angle = SalinityLegitimacyMapper.salt_to_legitimacy(salt)
        band = SalinityLegitimacyMapper.get_band_name(salt)
        water = SalinityLegitimacyMapper.get_water_type(salt)
        print(f"│ {salt:12} │ {angle:12.1f} │ {band:14} │ {water:17} │")
    
    print("└──────────────┴──────────────┴────────────────┴───────────────────┘")


def demo_business_environments():
    """Demonstrate business environment types."""
    print("\n" + "=" * 80)
    print("BUSINESS ENVIRONMENT TYPES")
    print("=" * 80)
    
    print("""
    Different combinations of top/bottom phase states:
    
    TOP (+Z)     BOTTOM (-Z)     BUSINESS TYPE
    ════════     ═══════════     ═════════════
    Gas          Solid           Consumer App (Instagram)
    Gas          Gas             Platform (Uber)
    Liquid       Liquid          Institutional (Banks)
    Solid        Solid           Heavy Industry (Boeing)
    Gas          Liquid          Consumer Enterprise (Amazon)
    Solid        Gas             Inverted (Consulting firms)
    """)
    
    print("\n┌───────────────────────┬──────────┬──────────┬──────────────────────────────┐")
    print("│ Type                  │ Top      │ Bottom   │ Description                  │")
    print("├───────────────────────┼──────────┼──────────┼──────────────────────────────┤")
    
    for env_type in BusinessEnvironmentType:
        print(f"│ {env_type.name:21} │ {env_type.top_phase.name:8} │ {env_type.bottom_phase.name:8} │ {env_type.description:28} │")
    
    print("└───────────────────────┴──────────┴──────────┴──────────────────────────────┘")


def demo_security_audit():
    """Demonstrate security audit with hidden entities."""
    print("\n" + "=" * 80)
    print("SECURITY AUDIT DEMONSTRATION")
    print("=" * 80)
    
    # Create hexagon
    hex_system = EnvironmentalHexagon("ACME_CORP")
    
    # Set up observers
    hex_system.setup_standard_observers()
    
    # Set environment
    hex_system.environment.trust_o2 = 25.0  # Good trust
    hex_system.environment.authenticity_salt = 15.0  # Somewhat brackish
    hex_system.environment.activity_heat = 300.0  # Active
    hex_system.environment.top_phase = PhaseState.GAS
    hex_system.environment.bottom_phase = PhaseState.LIQUID
    
    # Add legitimate entities
    hex_system.connect_entity("subsidiary_1", "SUBSIDIARY", HexRing.INNER, 0, 
                              declared_tilt=5.0, actual_tilt=5.0)
    hex_system.connect_entity("partner_a", "PARTNER", HexRing.MIDDLE, 1,
                              declared_tilt=10.0, actual_tilt=12.0)
    hex_system.connect_entity("vendor_1", "VENDOR", HexRing.OUTER, 2,
                              declared_tilt=0.0, actual_tilt=0.0)
    
    # Add suspicious entity (tilted to hide)
    hex_system.connect_entity("shell_corp", "SHELL", HexRing.OUTER, 3,
                              declared_tilt=15.0, actual_tilt=67.0)  # HIDING!
    
    # Add entity with mismatch
    hex_system.connect_entity("offshore_sub", "OFFSHORE", HexRing.MIDDLE, 4,
                              declared_tilt=5.0, actual_tilt=35.0)  # Mismatch!
    
    # Record some flows
    hex_system.record_flow("subsidiary_1", inflow=100000, outflow=80000)
    hex_system.record_flow("partner_a", inflow=50000, outflow=45000)
    hex_system.record_flow("shell_corp", inflow=500000, outflow=490000)  # Large hidden flow!
    hex_system.record_flow("offshore_sub", inflow=200000, outflow=195000)
    
    # Run audit
    audit = hex_system.security_audit()
    
    print(f"\n--- Security Audit: {audit['hexagon_id']} ---")
    print(f"\nTotal Entities: {audit['total_entities']}")
    print(f"Observer Coverage: {audit['observer_coverage']*100:.1f}%")
    print(f"Blind Spots: {len(audit['blind_spots'])}")
    
    print(f"\n--- Alignment Distribution ---")
    for status, count in audit['alignment_distribution'].items():
        print(f"  {status}: {count}")
    
    if audit['hidden_entities']:
        print(f"\n⚠️  HIDDEN ENTITIES DETECTED:")
        for hidden in audit['hidden_entities']:
            print(f"  • {hidden['entity_id']} ({hidden['entity_type']})")
            print(f"    Tilt: {hidden['actual_tilt']}°, Flow: ${hidden['flow_through']:,.0f}")
    
    if audit['declaration_mismatches']:
        print(f"\n⚠️  DECLARATION MISMATCHES:")
        for mismatch in audit['declaration_mismatches']:
            print(f"  • {mismatch['entity_id']} ({mismatch['entity_type']})")
            print(f"    Declared: {mismatch['declared_tilt']}°, Actual: {mismatch['actual_tilt']}°")
            print(f"    Mismatch: {mismatch['mismatch']}°, Flow: ${mismatch['flow_through']:,.0f}")
    
    print(f"\n--- Environment Status ---")
    print(f"Stable: {audit['environment_stable']}")
    if audit['environment_issues']:
        for issue in audit['environment_issues']:
            print(f"  ⚠️  {issue}")
    
    print(f"\n{'='*40}")
    print(f"RISK SCORE: {audit['risk_score']:.0f}/100")
    print(f"STATUS: {audit['status']}")
    print(f"{'='*40}")


def demo_complete_system():
    """Demonstrate complete environmental hexagon."""
    print("\n" + "=" * 80)
    print("COMPLETE ENVIRONMENTAL HEXAGON")
    print("=" * 80)
    
    # Create a consumer app company
    company = EnvironmentalHexagon("STARTUP_INC")
    
    # Set up observers
    company.add_observer("investors", 0, "BOARD", 50, 1.0)
    company.add_observer("accountant", 90, "AUDITOR", 35, 0.8)
    company.add_observer("legal", 180, "LEGAL", 35, 0.8)
    company.add_observer("tax_authority", 270, "REGULATOR", 45, 1.2)
    
    # Configure environment (Consumer App: Gas top, Solid bottom)
    company.environment.trust_o2 = 22.0  # Normal trust
    company.environment.authenticity_salt = 5.0  # Clean money
    company.environment.activity_heat = 310.0  # Active
    company.environment.top_phase = PhaseState.GAS  # Many users
    company.environment.bottom_phase = PhaseState.SOLID  # Small team
    
    # Add entities at various rings
    company.connect_entity("founders_llc", "HOLDING", HexRing.CORE, 0, 0, 0)
    company.connect_entity("dev_contractor", "CONTRACTOR", HexRing.INNER, 1, 5, 5)
    company.connect_entity("aws", "VENDOR", HexRing.MIDDLE, 2, 0, 0)
    company.connect_entity("payment_processor", "PARTNER", HexRing.MIDDLE, 3, 0, 0)
    company.connect_entity("ad_network", "PARTNER", HexRing.OUTER, 4, 10, 10)
    
    # Get summary
    summary = company.get_state_summary()
    
    print(f"\n--- {summary['hexagon_id']} ---")
    print(f"\nBusiness Type: {summary['business_type']}")
    print(f"Description: {summary['business_description']}")
    
    print(f"\n--- Environment ---")
    env = summary['environment']
    print(f"Trust: {env['trust_o2']}% O2 ({env['trust_level']})")
    print(f"Authenticity: {env['authenticity_salt']} ppt ({env['authenticity_level']})")
    print(f"Activity: {env['activity_heat']}K ({env['activity_level']})")
    print(f"Phases: {env['top_phase']} (top) / {env['bottom_phase']} (bottom)")
    print(f"Legitimacy Angle: {summary['legitimacy_angle']:.1f}°")
    
    print(f"\n--- Structure ---")
    struct = summary['structure']
    print(f"Entities: {struct['total_entities']}")
    print(f"Observers: {struct['total_observers']}")
    print(f"Coverage: {struct['observer_coverage']*100:.1f}%")
    print(f"Blind Spots: {struct['blind_spots']}")
    
    # Check flow composition
    print(f"\n--- Flow Composition Check ---")
    # Simulate incoming payment at different salt levels
    for salt in [2, 15, 50]:
        check = company.check_flow_composition(salt)
        print(f"Incoming {salt} ppt: {check['status']} - {check['message']}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("ENVIRONMENTAL HEXAGON - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Nested rings
    print("\n--- TEST 1: Nested Rings ---")
    nested = NestedHexagon()
    assert nested.get_ring_radius(HexRing.CORE) == 1.0
    assert nested.get_ring_radius(HexRing.INNER) > nested.get_ring_radius(HexRing.CORE)
    print("✓ Rings expand outward correctly")
    
    # Test 2: Observer visibility
    print("\n--- TEST 2: Observer Visibility ---")
    obs = Observer("test", 90, 30)
    assert obs.can_see_angle(90)
    assert obs.can_see_angle(100)
    assert not obs.can_see_angle(130)
    print("✓ Observer visibility works")
    
    # Test 3: Blind spot detection
    print("\n--- TEST 3: Blind Spot Detection ---")
    network = ObserverNetwork()
    network.add_observer(Observer("a", 0, 60))
    network.add_observer(Observer("b", 180, 60))
    blind_spots = network.find_blind_spots()
    assert len(blind_spots) > 0  # Should have gaps
    print(f"✓ Found {len(blind_spots)} blind spots")
    
    # Test 4: Salinity-legitimacy mapping
    print("\n--- TEST 4: Salinity-Legitimacy Mapping ---")
    assert SalinityLegitimacyMapper.salt_to_legitimacy(0) == 90.0
    assert SalinityLegitimacyMapper.salt_to_legitimacy(100) == 0.0
    assert 40 < SalinityLegitimacyMapper.salt_to_legitimacy(35) < 50
    print("✓ Salinity maps to legitimacy correctly")
    
    # Test 5: Environmental stability
    print("\n--- TEST 5: Environmental Stability ---")
    env = EnvironmentalComposition(trust_o2=21, authenticity_salt=10, activity_heat=300)
    stable, issues = env.is_stable()
    assert stable
    assert len(issues) == 0
    print("✓ Normal environment is stable")
    
    # Test 6: Entity alignment
    print("\n--- TEST 6: Entity Alignment ---")
    entity = ConnectedEntity("test", "TEST", declared_tilt=10, actual_tilt=50)
    assert entity.is_suspicious()
    assert entity.get_tilt_mismatch() == 40
    print("✓ Alignment mismatch detected")
    
    # Test 7: Security audit
    print("\n--- TEST 7: Security Audit ---")
    hex_sys = EnvironmentalHexagon("test")
    hex_sys.setup_standard_observers()
    hex_sys.connect_entity("hidden", "SHELL", HexRing.OUTER, 0, 
                          declared_tilt=5, actual_tilt=75)
    audit = hex_sys.security_audit()
    assert len(audit['hidden_entities']) > 0 or len(audit['declaration_mismatches']) > 0
    print("✓ Security audit detects issues")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--rings":
            demo_nested_rings()
        elif sys.argv[1] == "--observers":
            demo_observer_network()
        elif sys.argv[1] == "--salinity":
            demo_salinity_legitimacy()
        elif sys.argv[1] == "--environments":
            demo_business_environments()
        elif sys.argv[1] == "--audit":
            demo_security_audit()
        elif sys.argv[1] == "--complete":
            demo_complete_system()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_nested_rings()
            demo_observer_network()
            demo_salinity_legitimacy()
            demo_business_environments()
            demo_security_audit()
            demo_complete_system()
        else:
            run_tests()
    else:
        run_tests()
