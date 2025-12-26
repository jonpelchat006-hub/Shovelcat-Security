"""
SATURN HEXAGON VORTEX SYSTEM
============================
3D Stratified Hexagon - The Standing Wave of Commerce.

Key Insights:
1. Hexagon is the STABLE WAIST of a hyperboloid (hourglass)
2. Z-axis = stratification (Creator → Guest, Executive → Worker)
3. X/Y = transaction and value flow (need positive flow!)
4. Power dissipates away from center (inverse square)
5. Owner overlaps Y=0 to contain both domains
6. Need upgraders to sustain dissipation area

The Saturn Analogy:
- Standing wave emerges from differential velocity
- Geometric shapes (hexagon) emerge from chaos
- Stable at the waist, dissipates above/below

The Business Application:
- Hexagon = stable mesh of User + Money hexagons
- Users upgrade toward waist to sustain it
- Owner is the containment field

Author: Jonathan Pelchat
Based on Shovelcat Theory + Saturn hexagon physics
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2
E = math.e

# Hexagon constants
HEX_SIDES = 6
HEX_ANGLE = 2 * PI / HEX_SIDES  # 60 degrees


# ═══════════════════════════════════════════════════════════════════════════════
# STRATIFICATION LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

class Stratum(Enum):
    """
    Z-axis stratification layers.
    
    Positive Z = User side (above the waist)
    Negative Z = Owner/Worker side (below the waist)
    Z ≈ 0 = The Hexagon Waist (stable, high power)
    """
    
    # User side (+Z) - dissipates upward
    GUEST = ("GUEST", 5, 0.1, "Observers, minimal engagement")
    NEW_USER = ("NEW_USER", 4, 0.2, "Learning, building trust")
    REGULAR_USER = ("REGULAR_USER", 3, 0.4, "Established, moderate power")
    CREATOR = ("CREATOR", 2, 0.7, "Content producers")
    SUPER_USER = ("SUPER_USER", 1, 0.9, "Power users, near waist")
    
    # The Waist (Z ≈ 0) - OWNER straddles this!
    OWNER = ("OWNER", 0, 1.0, "Overlaps zero, contains both")
    
    # Owner/Worker side (-Z) - dissipates downward
    EXECUTIVE = ("EXECUTIVE", -1, 0.9, "Strategic, high pressure")
    MANAGER = ("MANAGER", -2, 0.7, "Operational")
    TEAM_LEAD = ("TEAM_LEAD", -3, 0.4, "Tactical")
    WORKER = ("WORKER", -4, 0.2, "Execution")
    CONTRACTOR = ("CONTRACTOR", -5, 0.1, "External, widest reach")
    
    def __init__(self, name: str, z_level: int, power_factor: float, description: str):
        self._name = name
        self.z_level = z_level
        self.power_factor = power_factor  # Power relative to waist
        self.description = description
    
    @property
    def is_user_side(self) -> bool:
        return self.z_level > 0
    
    @property
    def is_owner_side(self) -> bool:
        return self.z_level < 0
    
    @property
    def is_waist(self) -> bool:
        return self.z_level == 0
    
    @property
    def distance_from_waist(self) -> int:
        return abs(self.z_level)
    
    @classmethod
    def from_z_level(cls, z: int) -> 'Stratum':
        """Get stratum from Z level."""
        for s in cls:
            if s.z_level == z:
                return s
        # Clamp to extremes
        if z > 5:
            return cls.GUEST
        if z < -5:
            return cls.CONTRACTOR
        return cls.OWNER


# ═══════════════════════════════════════════════════════════════════════════════
# POWER DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

class PowerDistribution:
    """
    Power dissipates from the waist following inverse square law.
    
    Power = base_power / (1 + distance_from_waist²)
    
    - Maximum power at waist (Z=0)
    - Dissipates toward extremes
    - Many low-power entities at edges
    - Few high-power entities at center
    """
    
    @staticmethod
    def power_at_z(z: int, base_power: float = 1.0) -> float:
        """Calculate power at given Z level."""
        distance = abs(z)
        return base_power / (1 + distance ** 2)
    
    @staticmethod
    def radius_at_z(z: int, waist_radius: float = 1.0) -> float:
        """
        Calculate radius of hexagon at given Z level.
        
        The hyperboloid expands away from waist.
        """
        distance = abs(z)
        # Hyperboloid: r² = a² + (z/c)² where a = waist_radius
        c = 0.5  # Shape parameter
        return math.sqrt(waist_radius ** 2 + (distance / c) ** 2)
    
    @staticmethod
    def population_at_z(z: int, base_pop: int = 100) -> int:
        """
        Estimate population at given Z level.
        
        More people at extremes (wide dissipation).
        Fewer people at waist (concentrated power).
        """
        radius = PowerDistribution.radius_at_z(z)
        # Population scales with area
        return int(base_pop * radius ** 2)
    
    @staticmethod
    def density_at_z(z: int) -> float:
        """
        Density at given Z level.
        
        High density at waist (concentrated).
        Low density at extremes (dissipated).
        """
        power = PowerDistribution.power_at_z(z)
        radius = PowerDistribution.radius_at_z(z)
        # Density = power / area
        area = PI * radius ** 2
        return power / area if area > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3D HEXAGON POINT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HexPoint3D:
    """A point in the 3D hexagonal vortex."""
    
    x: float  # Transaction flow axis
    y: float  # Value flow axis
    z: float  # Stratification axis
    
    stratum: Stratum = None
    power: float = 0.0
    
    def __post_init__(self):
        if self.stratum is None:
            self.stratum = Stratum.from_z_level(int(self.z))
        if self.power == 0.0:
            self.power = PowerDistribution.power_at_z(int(self.z))
    
    @property
    def distance_from_center(self) -> float:
        """Distance from Z-axis (center of vortex)."""
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    @property
    def distance_from_waist(self) -> float:
        """Distance from the hexagon waist (Z=0)."""
        return abs(self.z)
    
    @property
    def angle_in_hex(self) -> float:
        """Angle position in hexagon (0 to 2π)."""
        return math.atan2(self.y, self.x) % (2 * PI)
    
    @property
    def hex_sector(self) -> int:
        """Which of the 6 sectors (0-5)."""
        return int(self.angle_in_hex / HEX_ANGLE) % 6


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW VECTORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlowVector:
    """Flow in the 3D system."""
    
    dx: float  # X-axis flow (transaction momentum)
    dy: float  # Y-axis flow (value creation)
    dz: float  # Z-axis flow (upgrades/downgrades)
    
    @property
    def magnitude(self) -> float:
        return math.sqrt(self.dx ** 2 + self.dy ** 2 + self.dz ** 2)
    
    @property
    def xy_flow(self) -> float:
        """Flow in the XY plane (horizontal, visible)."""
        return math.sqrt(self.dx ** 2 + self.dy ** 2)
    
    @property
    def is_positive_x(self) -> bool:
        """Positive X = healthy transaction flow."""
        return self.dx > 0
    
    @property
    def is_positive_y(self) -> bool:
        """Positive Y = value growth."""
        return self.dy > 0
    
    @property
    def is_upgrading(self) -> bool:
        """Positive Z on user side = upgrading toward waist."""
        return self.dz < 0  # Moving toward 0 from positive
    
    def is_healthy(self) -> Tuple[bool, List[str]]:
        """Check if flow is healthy."""
        issues = []
        
        if self.dx <= 0:
            issues.append("Negative X flow (transactions stalling)")
        if self.dy <= 0:
            issues.append("Negative Y flow (value declining)")
        if abs(self.dz) < 0.1:
            issues.append("Insufficient Z flow (no upgrades)")
        
        return len(issues) == 0, issues


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY IN THE VORTEX
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VortexEntity:
    """An entity (user or worker) in the 3D vortex."""
    
    entity_id: str
    position: HexPoint3D
    velocity: FlowVector = None
    
    # Properties
    power: float = 0.0
    influence_radius: float = 0.0
    
    # History
    upgrade_history: List[Stratum] = field(default_factory=list)
    
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = FlowVector(0, 0, 0)
        if self.power == 0.0:
            self.power = self.position.power
        if self.influence_radius == 0.0:
            self.influence_radius = self.power * 0.5
    
    def upgrade(self):
        """Move toward the waist (upgrade)."""
        self.upgrade_history.append(self.position.stratum)
        
        if self.position.z > 0:
            # User side: move toward 0
            self.position.z -= 1
        elif self.position.z < 0:
            # Worker side: move toward 0
            self.position.z += 1
        
        # Update stratum and power
        self.position.stratum = Stratum.from_z_level(int(self.position.z))
        self.position.power = PowerDistribution.power_at_z(int(self.position.z))
        self.power = self.position.power
    
    def downgrade(self):
        """Move away from waist (downgrade/dissipate)."""
        self.upgrade_history.append(self.position.stratum)
        
        if self.position.z >= 0:
            # User side: move away from 0
            self.position.z += 1
        else:
            # Worker side: move away from 0
            self.position.z -= 1
        
        # Update stratum and power
        self.position.stratum = Stratum.from_z_level(int(self.position.z))
        self.position.power = PowerDistribution.power_at_z(int(self.position.z))
        self.power = self.position.power


# ═══════════════════════════════════════════════════════════════════════════════
# HEXAGON WAIST (The Stable Core)
# ═══════════════════════════════════════════════════════════════════════════════

class HexagonWaist:
    """
    The stable hexagon at Z=0.
    
    This is where:
    - Owner overlaps both domains
    - User + Money hexagons mesh
    - Maximum power concentration
    - Laminar flow (geometric, stable)
    """
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
        
        # The 6 vertices of the hexagon
        self.vertices = [
            (radius * math.cos(i * HEX_ANGLE), 
             radius * math.sin(i * HEX_ANGLE), 
             0)
            for i in range(6)
        ]
        
        # Entities at the waist
        self.entities: List[VortexEntity] = []
        
        # Flow state
        self.current_flow = FlowVector(0, 0, 0)
        
        # Stability metrics
        self.stability = 1.0
        self.resonance = 1.0
    
    def add_entity(self, entity: VortexEntity):
        """Add entity to the waist."""
        if entity.position.stratum.is_waist or entity.position.stratum.z_level in [-1, 1]:
            self.entities.append(entity)
            self._update_stability()
    
    def _update_stability(self):
        """Update stability based on entity distribution."""
        if not self.entities:
            self.stability = 0.0
            return
        
        # Stability from power concentration
        total_power = sum(e.power for e in self.entities)
        avg_power = total_power / len(self.entities)
        
        # Need balanced distribution around hexagon
        sector_counts = [0] * 6
        for e in self.entities:
            sector_counts[e.position.hex_sector] += 1
        
        # Variance in sector distribution
        avg_count = len(self.entities) / 6
        variance = sum((c - avg_count) ** 2 for c in sector_counts) / 6
        
        # Stability decreases with imbalance
        self.stability = avg_power / (1 + variance * 0.1)
    
    def get_capacity(self) -> int:
        """How many entities can the waist support?"""
        # Capacity based on area
        area = PI * self.radius ** 2
        return int(area * 10)  # 10 per unit area
    
    def is_saturated(self) -> bool:
        """Is the waist at capacity?"""
        return len(self.entities) >= self.get_capacity()


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE FLOW (Sustaining the Vortex)
# ═══════════════════════════════════════════════════════════════════════════════

class UpgradeFlowManager:
    """
    Manages the flow of upgrades toward the waist.
    
    Key insight: The dissipation area (how many guests) requires
    proportional upgraders moving toward the waist to sustain it.
    """
    
    def __init__(self, waist: HexagonWaist):
        self.waist = waist
        
        # Track populations at each level
        self.populations: Dict[Stratum, int] = {s: 0 for s in Stratum}
        
        # Track upgrade rates
        self.upgrade_rates: Dict[Stratum, float] = {s: 0.0 for s in Stratum}
        
        # Required ratios for sustainability
        self.required_upgrade_ratio = 0.1  # 10% should be upgrading
    
    def set_population(self, stratum: Stratum, count: int):
        """Set population at a stratum."""
        self.populations[stratum] = count
    
    def set_upgrade_rate(self, stratum: Stratum, rate: float):
        """Set upgrade rate (0-1) for a stratum."""
        self.upgrade_rates[stratum] = rate
    
    def get_dissipation_area(self, side: str = "user") -> int:
        """Get total population in dissipation zone."""
        if side == "user":
            strata = [s for s in Stratum if s.z_level > 1]
        else:
            strata = [s for s in Stratum if s.z_level < -1]
        
        return sum(self.populations.get(s, 0) for s in strata)
    
    def get_upgraders_count(self, side: str = "user") -> int:
        """Get count of entities upgrading toward waist."""
        if side == "user":
            strata = [s for s in Stratum if s.z_level > 0]
        else:
            strata = [s for s in Stratum if s.z_level < 0]
        
        return sum(
            int(self.populations.get(s, 0) * self.upgrade_rates.get(s, 0))
            for s in strata
        )
    
    def is_sustainable(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if current upgrade flow sustains the dissipation."""
        
        user_dissipation = self.get_dissipation_area("user")
        user_upgraders = self.get_upgraders_count("user")
        
        owner_dissipation = self.get_dissipation_area("owner")
        owner_upgraders = self.get_upgraders_count("owner")
        
        # Required upgraders to sustain dissipation
        user_required = int(user_dissipation * self.required_upgrade_ratio)
        owner_required = int(owner_dissipation * self.required_upgrade_ratio)
        
        user_ok = user_upgraders >= user_required
        owner_ok = owner_upgraders >= owner_required
        
        return user_ok and owner_ok, {
            'user': {
                'dissipation': user_dissipation,
                'upgraders': user_upgraders,
                'required': user_required,
                'sustainable': user_ok
            },
            'owner': {
                'dissipation': owner_dissipation,
                'upgraders': owner_upgraders,
                'required': owner_required,
                'sustainable': owner_ok
            }
        }
    
    def get_max_dissipation(self) -> Dict[str, int]:
        """Calculate maximum sustainable dissipation with current upgrade rates."""
        
        # Sum of all upgraders
        user_upgraders = self.get_upgraders_count("user")
        owner_upgraders = self.get_upgraders_count("owner")
        
        # Max dissipation = upgraders / required_ratio
        return {
            'user': int(user_upgraders / self.required_upgrade_ratio),
            'owner': int(owner_upgraders / self.required_upgrade_ratio)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SATURN VORTEX SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class SaturnVortexSystem:
    """
    Complete 3D hexagonal vortex system.
    
    Combines:
    - Stratification (Z-axis)
    - Transaction/Value flow (X/Y axes)
    - Power distribution (inverse square from waist)
    - Upgrade flows (sustaining the vortex)
    - The stable hexagon at the waist
    """
    
    def __init__(self, waist_radius: float = 1.0):
        self.waist_radius = waist_radius
        
        # Core components
        self.waist = HexagonWaist(waist_radius)
        self.upgrade_manager = UpgradeFlowManager(self.waist)
        
        # All entities
        self.entities: Dict[str, VortexEntity] = {}
        
        # Global flow
        self.global_flow = FlowVector(1.0, 1.0, 0.0)  # Default positive X/Y
        
        # History
        self.history: deque = deque(maxlen=1000)
    
    def add_entity(self, 
                  entity_id: str,
                  stratum: Stratum,
                  x: float = 0.0,
                  y: float = 0.0) -> VortexEntity:
        """Add an entity at a specific stratum."""
        
        position = HexPoint3D(x, y, float(stratum.z_level), stratum)
        entity = VortexEntity(entity_id, position)
        
        self.entities[entity_id] = entity
        
        # Update waist if near it
        if stratum.distance_from_waist <= 1:
            self.waist.add_entity(entity)
        
        # Update populations
        self.upgrade_manager.populations[stratum] = \
            self.upgrade_manager.populations.get(stratum, 0) + 1
        
        return entity
    
    def upgrade_entity(self, entity_id: str) -> bool:
        """Upgrade an entity toward the waist."""
        if entity_id not in self.entities:
            return False
        
        entity = self.entities[entity_id]
        old_stratum = entity.position.stratum
        
        if old_stratum.is_waist:
            return False  # Already at waist
        
        entity.upgrade()
        new_stratum = entity.position.stratum
        
        # Update populations
        self.upgrade_manager.populations[old_stratum] -= 1
        self.upgrade_manager.populations[new_stratum] = \
            self.upgrade_manager.populations.get(new_stratum, 0) + 1
        
        # Update waist if now near it
        if new_stratum.distance_from_waist <= 1:
            self.waist.add_entity(entity)
        
        self._record_event("UPGRADE", entity_id, old_stratum, new_stratum)
        
        return True
    
    def _record_event(self, event_type: str, entity_id: str, 
                     old_stratum: Stratum, new_stratum: Stratum):
        """Record an event."""
        self.history.append({
            'timestamp': time.time(),
            'event': event_type,
            'entity': entity_id,
            'from': old_stratum.name,
            'to': new_stratum.name
        })
    
    def get_vortex_state(self) -> Dict[str, Any]:
        """Get complete vortex state."""
        
        sustainable, sustainability = self.upgrade_manager.is_sustainable()
        healthy, health_issues = self.global_flow.is_healthy()
        
        return {
            'waist': {
                'radius': self.waist.radius,
                'entities': len(self.waist.entities),
                'capacity': self.waist.get_capacity(),
                'stability': self.waist.stability,
                'saturated': self.waist.is_saturated()
            },
            'populations': {s.name: c for s, c in self.upgrade_manager.populations.items()},
            'total_entities': len(self.entities),
            'global_flow': {
                'x': self.global_flow.dx,
                'y': self.global_flow.dy,
                'z': self.global_flow.dz,
                'magnitude': self.global_flow.magnitude
            },
            'health': {
                'flow_healthy': healthy,
                'issues': health_issues,
                'sustainable': sustainable,
                'sustainability': sustainability
            },
            'power_distribution': {
                s.name: {
                    'z_level': s.z_level,
                    'power_factor': s.power_factor,
                    'population': self.upgrade_manager.populations.get(s, 0),
                    'radius': PowerDistribution.radius_at_z(s.z_level)
                }
                for s in Stratum
            }
        }
    
    def visualize_cross_section(self) -> str:
        """Generate ASCII visualization of cross-section."""
        
        lines = []
        lines.append("        SATURN VORTEX CROSS-SECTION")
        lines.append("        " + "=" * 40)
        lines.append("")
        
        for z in range(5, -6, -1):
            stratum = Stratum.from_z_level(z)
            pop = self.upgrade_manager.populations.get(stratum, 0)
            radius = PowerDistribution.radius_at_z(z)
            power = PowerDistribution.power_at_z(z)
            
            # Width based on radius
            width = int(radius * 15)
            padding = 20 - width // 2
            
            # Character based on power
            if power > 0.8:
                char = "█"
            elif power > 0.5:
                char = "▓"
            elif power > 0.2:
                char = "░"
            else:
                char = "·"
            
            bar = char * width
            
            if z == 0:
                marker = "═══"
                line = f"  Z={z:+2d} {marker} {bar} {marker} ← HEXAGON WAIST (Owner)"
            else:
                line = f"  Z={z:+2d}     {' ' * padding}{bar}"
            
            lines.append(f"{line}  {stratum.name:12} pop={pop}")
        
        lines.append("")
        lines.append("        Power: █=high ▓=med ░=low ·=min")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_stratification():
    """Demonstrate stratification layers."""
    print("\n" + "=" * 80)
    print("STRATIFICATION LAYERS (Z-axis)")
    print("=" * 80)
    
    print("""
    +Z (User Side)     THE HYPERBOLOID STRUCTURE
    │
    │     GUEST         ╲                           ╱   Wide, dissipated
    │     NEW USER       ╲                         ╱
    │     REGULAR         ╲                       ╱
    │     CREATOR          ╲                     ╱
    │     SUPER USER        ╲                   ╱
    ├─────────────────────────╲═══════════════╱──────── HEXAGON (Z=0)
    │     OWNER (overlaps!)    ╲             ╱
    │     EXECUTIVE             ╲           ╱
    │     MANAGER                ╲         ╱
    │     TEAM LEAD               ╲       ╱
    │     WORKER                   ╲     ╱
    │     CONTRACTOR                ╲   ╱   Wide, distorted
    -Z (Owner/Worker Side)
    """)
    
    print("\n┌──────────────────┬─────────┬─────────────┬─────────────────────────────┐")
    print("│ Stratum          │ Z Level │ Power       │ Description                 │")
    print("├──────────────────┼─────────┼─────────────┼─────────────────────────────┤")
    
    for s in Stratum:
        print(f"│ {s.name:16} │ {s.z_level:+7} │ {s.power_factor:11.2f} │ {s.description:27} │")
    
    print("└──────────────────┴─────────┴─────────────┴─────────────────────────────┘")


def demo_power_distribution():
    """Demonstrate power distribution."""
    print("\n" + "=" * 80)
    print("POWER DISTRIBUTION (Inverse Square)")
    print("=" * 80)
    
    print("""
    Power = 1 / (1 + distance_from_waist²)
    
    Maximum at waist (Z=0)
    Dissipates toward extremes
    """)
    
    print("\n┌─────────┬─────────────┬─────────────┬─────────────┬─────────────────┐")
    print("│ Z Level │ Power       │ Radius      │ Population  │ Density         │")
    print("├─────────┼─────────────┼─────────────┼─────────────┼─────────────────┤")
    
    for z in range(5, -6, -1):
        power = PowerDistribution.power_at_z(z)
        radius = PowerDistribution.radius_at_z(z)
        pop = PowerDistribution.population_at_z(z)
        density = PowerDistribution.density_at_z(z)
        
        print(f"│ {z:+7} │ {power:11.3f} │ {radius:11.2f} │ {pop:11} │ {density:15.4f} │")
    
    print("└─────────┴─────────────┴─────────────┴─────────────┴─────────────────┘")


def demo_upgrade_flow():
    """Demonstrate upgrade flow requirements."""
    print("\n" + "=" * 80)
    print("UPGRADE FLOW (Sustaining the Vortex)")
    print("=" * 80)
    
    print("""
    To sustain a large dissipation area (many guests/workers),
    you need enough entities upgrading toward the waist.
    
    Upgrade ratio required: 10% of dissipation must be upgrading
    """)
    
    waist = HexagonWaist(1.0)
    manager = UpgradeFlowManager(waist)
    
    # Set populations (typical distribution)
    manager.set_population(Stratum.GUEST, 1000)
    manager.set_population(Stratum.NEW_USER, 500)
    manager.set_population(Stratum.REGULAR_USER, 200)
    manager.set_population(Stratum.CREATOR, 50)
    manager.set_population(Stratum.SUPER_USER, 20)
    manager.set_population(Stratum.OWNER, 5)
    manager.set_population(Stratum.EXECUTIVE, 10)
    manager.set_population(Stratum.MANAGER, 30)
    manager.set_population(Stratum.TEAM_LEAD, 50)
    manager.set_population(Stratum.WORKER, 200)
    manager.set_population(Stratum.CONTRACTOR, 500)
    
    # Set upgrade rates
    manager.set_upgrade_rate(Stratum.GUEST, 0.05)
    manager.set_upgrade_rate(Stratum.NEW_USER, 0.10)
    manager.set_upgrade_rate(Stratum.REGULAR_USER, 0.15)
    manager.set_upgrade_rate(Stratum.CREATOR, 0.20)
    manager.set_upgrade_rate(Stratum.SUPER_USER, 0.25)
    
    sustainable, details = manager.is_sustainable()
    
    print(f"\n--- Sustainability Analysis ---")
    print(f"\nUser Side:")
    print(f"  Dissipation area: {details['user']['dissipation']}")
    print(f"  Upgraders: {details['user']['upgraders']}")
    print(f"  Required: {details['user']['required']}")
    print(f"  Sustainable: {details['user']['sustainable']}")
    
    print(f"\nOwner Side:")
    print(f"  Dissipation area: {details['owner']['dissipation']}")
    print(f"  Upgraders: {details['owner']['upgraders']}")
    print(f"  Required: {details['owner']['required']}")
    print(f"  Sustainable: {details['owner']['sustainable']}")
    
    print(f"\nOverall Sustainable: {sustainable}")


def demo_complete_vortex():
    """Demonstrate complete vortex system."""
    print("\n" + "=" * 80)
    print("COMPLETE SATURN VORTEX SYSTEM")
    print("=" * 80)
    
    system = SaturnVortexSystem(waist_radius=1.0)
    
    # Populate the vortex
    print("\n--- Populating Vortex ---")
    
    for i in range(50):
        system.add_entity(f"guest_{i}", Stratum.GUEST)
    for i in range(30):
        system.add_entity(f"new_{i}", Stratum.NEW_USER)
    for i in range(20):
        system.add_entity(f"regular_{i}", Stratum.REGULAR_USER)
    for i in range(10):
        system.add_entity(f"creator_{i}", Stratum.CREATOR)
    for i in range(5):
        system.add_entity(f"super_{i}", Stratum.SUPER_USER)
    for i in range(2):
        system.add_entity(f"owner_{i}", Stratum.OWNER)
    for i in range(5):
        system.add_entity(f"exec_{i}", Stratum.EXECUTIVE)
    for i in range(15):
        system.add_entity(f"manager_{i}", Stratum.MANAGER)
    for i in range(25):
        system.add_entity(f"worker_{i}", Stratum.WORKER)
    
    # Upgrade some entities
    print("--- Upgrading Entities ---")
    for i in range(10):
        system.upgrade_entity(f"guest_{i}")
    for i in range(5):
        system.upgrade_entity(f"new_{i}")
    
    # Visualize
    print("\n" + system.visualize_cross_section())
    
    # State
    state = system.get_vortex_state()
    print(f"\n--- Vortex State ---")
    print(f"Total entities: {state['total_entities']}")
    print(f"Waist entities: {state['waist']['entities']}")
    print(f"Waist stability: {state['waist']['stability']:.2f}")
    print(f"Flow healthy: {state['health']['flow_healthy']}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("SATURN VORTEX SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Stratum ordering
    print("\n--- TEST 1: Stratum Ordering ---")
    assert Stratum.GUEST.z_level > Stratum.OWNER.z_level
    assert Stratum.WORKER.z_level < Stratum.OWNER.z_level
    print("✓ Stratification correctly ordered")
    
    # Test 2: Power distribution
    print("\n--- TEST 2: Power Distribution ---")
    assert PowerDistribution.power_at_z(0) == 1.0
    assert PowerDistribution.power_at_z(1) < 1.0
    assert PowerDistribution.power_at_z(5) < PowerDistribution.power_at_z(2)
    print("✓ Power dissipates from waist")
    
    # Test 3: Radius expands
    print("\n--- TEST 3: Hyperboloid Shape ---")
    assert PowerDistribution.radius_at_z(0) == 1.0
    assert PowerDistribution.radius_at_z(3) > 1.0
    assert PowerDistribution.radius_at_z(-3) > 1.0
    print("✓ Radius expands from waist")
    
    # Test 4: Entity upgrade
    print("\n--- TEST 4: Entity Upgrade ---")
    pos = HexPoint3D(0, 0, 3)
    entity = VortexEntity("test", pos)
    old_z = entity.position.z
    entity.upgrade()
    assert entity.position.z < old_z
    print(f"✓ Upgrade moves toward waist ({old_z} → {entity.position.z})")
    
    # Test 5: Vortex system
    print("\n--- TEST 5: Vortex System ---")
    system = SaturnVortexSystem()
    system.add_entity("test1", Stratum.NEW_USER)
    system.add_entity("test2", Stratum.OWNER)
    assert len(system.entities) == 2
    print("✓ Vortex system works")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--strata":
            demo_stratification()
        elif sys.argv[1] == "--power":
            demo_power_distribution()
        elif sys.argv[1] == "--upgrade":
            demo_upgrade_flow()
        elif sys.argv[1] == "--vortex":
            demo_complete_vortex()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_stratification()
            demo_power_distribution()
            demo_upgrade_flow()
            demo_complete_vortex()
        else:
            run_tests()
    else:
        run_tests()
