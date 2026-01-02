"""
TRUST TOPOLOGY: The Euler Membrane
==================================

A trust/alignment system based on position on the number line:

    <0  : ANTIMATTER - anti-reasoning, stupidity, anti-logic
    0   : Zero-sum actors (purely transactional)
    0.5 : Indifferent (don't care)
    1   : THE WALL - cooperative interface with outside
    >1  : Protected zone - increasing dedication
    2+  : Fully dedicated, shielded from negatives

Key insight: Entities at position 1 ROTATE using Euler's formula:
    position = 1 + e^(iθ)
    
This lets them:
    - Touch 0 (interact with zero-sum actors) when θ = π
    - Reach 2 (full dedication) when θ = 0
    - NEVER go negative (protected from antimatter)

The Wall (bit at position 1) serves as:
    - Interface between inside and outside
    - Filter for negative beliefs coming in
    - Protection for inner circles (>1)

People bring negativity through:
    - Incorrect beliefs
    - Misaligned frameworks
    - Zero-sum thinking
    
These must be filtered at the boundary before entering.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from datetime import datetime
import cmath

# Constants from the theory
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PI = np.pi
E = np.e


# =============================================================================
# TRUST ZONES
# =============================================================================

class TrustZone(Enum):
    """Zones on the trust number line."""
    ANTIMATTER = "antimatter"       # < 0: anti-logic, dangerous
    ZERO_SUM = "zero_sum"           # = 0: purely transactional
    INDIFFERENT = "indifferent"     # 0 < x < 0.5: don't really care
    NEUTRAL = "neutral"             # 0.5 <= x < 1: passive positive
    WALL = "wall"                   # = 1: the boundary/interface
    COMMITTED = "committed"         # 1 < x < 2: actively dedicated
    CORE = "core"                   # >= 2: fully dedicated, protected


def get_zone(position: float) -> TrustZone:
    """Determine which zone a position is in."""
    if position < 0:
        return TrustZone.ANTIMATTER
    elif position == 0:
        return TrustZone.ZERO_SUM
    elif position < 0.5:
        return TrustZone.INDIFFERENT
    elif position < 1:
        return TrustZone.NEUTRAL
    elif position == 1:
        return TrustZone.WALL
    elif position < 2:
        return TrustZone.COMMITTED
    else:
        return TrustZone.CORE


# =============================================================================
# THE EULER ROTATOR
# =============================================================================

class EulerRotator:
    """
    Rotation around a center point using Euler's formula.
    
    position = center + radius * e^(iθ)
    
    For trust at position 1 with radius 1:
    - Can reach 0 (touch zero-sum)
    - Can reach 2 (full dedication)
    - Never touches negative (antimatter protected)
    """
    
    def __init__(self, center: float = 1.0, radius: float = 1.0):
        self.center = center
        self.radius = radius
        self.theta = 0.0  # Current angle
        self.rotation_history: List[Tuple[float, complex]] = []
    
    @property
    def position(self) -> complex:
        """Current position in complex plane."""
        return self.center + self.radius * cmath.exp(1j * self.theta)
    
    @property
    def real_position(self) -> float:
        """Real part of position (the trust level)."""
        return self.position.real
    
    @property
    def imaginary_position(self) -> float:
        """Imaginary part (perpendicular to number line)."""
        return self.position.imag
    
    @property
    def min_position(self) -> float:
        """Minimum reachable position."""
        return self.center - self.radius
    
    @property
    def max_position(self) -> float:
        """Maximum reachable position."""
        return self.center + self.radius
    
    @property
    def touches_antimatter(self) -> bool:
        """Can this rotator ever touch antimatter (<0)?"""
        return self.min_position < 0
    
    @property
    def zone(self) -> TrustZone:
        """Current zone based on real position."""
        return get_zone(self.real_position)
    
    def rotate(self, delta_theta: float) -> complex:
        """Rotate by delta_theta radians."""
        self.theta += delta_theta
        self.theta = self.theta % (2 * PI)  # Keep in [0, 2π)
        pos = self.position
        self.rotation_history.append((self.theta, pos))
        return pos
    
    def rotate_to_angle(self, theta: float) -> complex:
        """Rotate to specific angle."""
        self.theta = theta % (2 * PI)
        pos = self.position
        self.rotation_history.append((self.theta, pos))
        return pos
    
    def rotate_to_position(self, target_real: float) -> Optional[complex]:
        """
        Rotate to achieve a target real position.
        Returns None if impossible.
        """
        if target_real < self.min_position or target_real > self.max_position:
            return None  # Can't reach
        
        # Solve: center + radius * cos(θ) = target_real
        cos_theta = (target_real - self.center) / self.radius
        cos_theta = max(-1, min(1, cos_theta))  # Clamp for numerical stability
        theta = np.arccos(cos_theta)
        
        return self.rotate_to_angle(theta)
    
    def scrape_zero_sum(self) -> complex:
        """Rotate to touch zero-sum actors (position 0)."""
        return self.rotate_to_position(0.0) or self.rotate_to_angle(PI)
    
    def full_dedication(self) -> complex:
        """Rotate to full dedication (position 2 for standard config)."""
        return self.rotate_to_position(self.max_position)
    
    def on_the_wall(self) -> complex:
        """Rotate to be perpendicular on the wall."""
        return self.rotate_to_angle(PI / 2)
    
    def display_state(self):
        """Print current state."""
        print(f"  Center: {self.center}")
        print(f"  Radius: {self.radius}")
        print(f"  Theta: {self.theta:.4f} rad ({np.degrees(self.theta):.1f}°)")
        print(f"  Position: {self.position}")
        print(f"  Real (trust level): {self.real_position:.4f}")
        print(f"  Zone: {self.zone.value}")
        print(f"  Range: [{self.min_position:.2f}, {self.max_position:.2f}]")
        print(f"  Antimatter safe: {not self.touches_antimatter}")


# =============================================================================
# TRUST ENTITY
# =============================================================================

@dataclass
class Belief:
    """A belief that an entity holds."""
    name: str
    value: float  # Positive = aligned, negative = misaligned
    confidence: float = 1.0  # How strongly held
    
    @property
    def contribution(self) -> float:
        """Net contribution to trust level."""
        return self.value * self.confidence


@dataclass
class TrustEntity:
    """
    An entity in the trust topology.
    
    Position determined by:
    - Base alignment (inherent)
    - Beliefs carried (can be positive or negative)
    - Dedication level (commitment)
    """
    id: str
    name: str
    base_alignment: float = 0.5  # Starting position
    
    # Beliefs affect position
    beliefs: List[Belief] = field(default_factory=list)
    
    # For wall entities, they have a rotator
    rotator: Optional[EulerRotator] = None
    
    # Shielded by (who protects this entity)
    shielded_by: Optional[str] = None
    
    # History
    position_history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def __post_init__(self):
        self._update_position_history()
    
    @property
    def belief_contribution(self) -> float:
        """Total contribution from beliefs."""
        return sum(b.contribution for b in self.beliefs)
    
    @property
    def position(self) -> float:
        """Current trust position."""
        if self.rotator:
            return self.rotator.real_position
        return self.base_alignment + self.belief_contribution
    
    @property
    def zone(self) -> TrustZone:
        """Current zone."""
        return get_zone(self.position)
    
    @property
    def carries_negativity(self) -> bool:
        """Does this entity carry negative beliefs?"""
        return any(b.value < 0 for b in self.beliefs)
    
    @property
    def total_negativity(self) -> float:
        """Total negative belief value."""
        return sum(b.value for b in self.beliefs if b.value < 0)
    
    @property
    def is_antimatter(self) -> bool:
        """Is this entity in antimatter zone?"""
        return self.position < 0
    
    def _update_position_history(self):
        """Record current position."""
        self.position_history.append((datetime.now(), self.position))
    
    def add_belief(self, belief: Belief):
        """Add a belief."""
        self.beliefs.append(belief)
        self._update_position_history()
    
    def remove_belief(self, belief_name: str) -> Optional[Belief]:
        """Remove a belief by name."""
        for i, b in enumerate(self.beliefs):
            if b.name == belief_name:
                removed = self.beliefs.pop(i)
                self._update_position_history()
                return removed
        return None
    
    def transform_belief(self, belief_name: str, new_value: float):
        """Transform a belief (e.g., correct a misconception)."""
        for b in self.beliefs:
            if b.name == belief_name:
                old_value = b.value
                b.value = new_value
                self._update_position_history()
                return old_value
        return None
    
    def become_wall_entity(self, radius: float = 1.0):
        """
        Transform into a wall entity with rotation capability.
        Base alignment becomes the center.
        """
        # Adjust base to 1 (the wall)
        self.base_alignment = 1.0
        self.rotator = EulerRotator(center=1.0, radius=radius)
        self._update_position_history()
    
    def display(self):
        """Print entity state."""
        print(f"\n{'='*50}")
        print(f"Entity: {self.name} ({self.id})")
        print(f"{'='*50}")
        print(f"  Base alignment: {self.base_alignment:.2f}")
        print(f"  Belief contribution: {self.belief_contribution:+.2f}")
        print(f"  Current position: {self.position:.4f}")
        print(f"  Zone: {self.zone.value}")
        print(f"  Carries negativity: {self.carries_negativity}")
        if self.carries_negativity:
            print(f"  Total negativity: {self.total_negativity:.2f}")
        if self.rotator:
            print(f"  Has rotator (wall entity):")
            self.rotator.display_state()
        if self.shielded_by:
            print(f"  Shielded by: {self.shielded_by}")
        print(f"  Beliefs:")
        for b in self.beliefs:
            sign = "+" if b.value >= 0 else ""
            print(f"    - {b.name}: {sign}{b.value:.2f} (conf: {b.confidence:.2f})")


# =============================================================================
# THE WALL (PROTECTIVE MEMBRANE)
# =============================================================================

class TheWall:
    """
    The Wall at position 1.
    
    Responsibilities:
    - Interface with outside world (zero-sum and below)
    - Filter negativity from incoming entities
    - Protect inner circles (>1)
    - Rotate to interact without touching antimatter
    """
    
    def __init__(self):
        self.wall_entities: Dict[str, TrustEntity] = {}
        self.protected_entities: Dict[str, TrustEntity] = {}
        self.outside_entities: Dict[str, TrustEntity] = {}
        
        # Quarantined negativity
        self.quarantined_beliefs: List[Belief] = []
        
        # Filter statistics
        self.total_filtered: float = 0.0
        self.entities_admitted: int = 0
        self.entities_rejected: int = 0
    
    def add_wall_entity(self, entity: TrustEntity):
        """Add an entity to the wall (position 1)."""
        entity.become_wall_entity()
        self.wall_entities[entity.id] = entity
        print(f"🧱 {entity.name} joined THE WALL")
    
    def add_protected_entity(self, entity: TrustEntity, shield_id: str):
        """Add an entity to the protected zone (>1)."""
        if entity.position <= 1:
            entity.base_alignment = 1.5  # Move to protected zone
        entity.shielded_by = shield_id
        self.protected_entities[entity.id] = entity
        print(f"🛡️ {entity.name} is now PROTECTED (shielded by {shield_id})")
    
    def process_incoming(self, entity: TrustEntity) -> Tuple[bool, str]:
        """
        Process an entity trying to enter.
        
        Returns (admitted, reason)
        """
        print(f"\n{'─'*50}")
        print(f"📥 Processing incoming: {entity.name}")
        print(f"   Position: {entity.position:.4f}")
        print(f"   Zone: {entity.zone.value}")
        print(f"   Carries negativity: {entity.carries_negativity}")
        
        # Case 1: Already antimatter - reject immediately
        if entity.is_antimatter:
            self.entities_rejected += 1
            print(f"   ❌ REJECTED: Antimatter entity")
            return False, "antimatter_entity"
        
        # Case 2: Zero-sum - can interact but not enter
        if entity.zone == TrustZone.ZERO_SUM:
            self.outside_entities[entity.id] = entity
            print(f"   ⚠️ OUTSIDE: Zero-sum actor, can interact only")
            return False, "zero_sum_stays_outside"
        
        # Case 3: Carries negativity - must filter
        if entity.carries_negativity:
            print(f"   🔍 Filtering negativity...")
            filtered = self._filter_negativity(entity)
            print(f"   📦 Quarantined {len(filtered)} negative beliefs")
            
            # Check again after filtering
            if entity.is_antimatter:
                self.entities_rejected += 1
                print(f"   ❌ REJECTED: Still antimatter after filtering")
                return False, "antimatter_after_filter"
        
        # Case 4: Indifferent (0-0.5) - needs more alignment
        if entity.zone == TrustZone.INDIFFERENT:
            self.outside_entities[entity.id] = entity
            print(f"   ⏳ WAITING: Needs more alignment (currently {entity.position:.2f})")
            return False, "needs_alignment"
        
        # Case 5: Neutral or above - can enter
        if entity.position >= 0.5:
            self.entities_admitted += 1
            if entity.position >= 1:
                self.protected_entities[entity.id] = entity
                print(f"   ✅ ADMITTED to protected zone")
            else:
                self.outside_entities[entity.id] = entity
                print(f"   ✅ ADMITTED to neutral zone (approaching wall)")
            return True, "admitted"
        
        # Default
        self.outside_entities[entity.id] = entity
        return False, "unknown"
    
    def _filter_negativity(self, entity: TrustEntity) -> List[Belief]:
        """
        Filter negative beliefs from an entity.
        Quarantine them, don't destroy (conservation).
        """
        filtered = []
        remaining = []
        
        for belief in entity.beliefs:
            if belief.value < 0:
                # Quarantine negative belief
                filtered.append(belief)
                self.quarantined_beliefs.append(belief)
                self.total_filtered += abs(belief.value)
                print(f"      ❄️ Quarantined: {belief.name} ({belief.value:.2f})")
            else:
                remaining.append(belief)
        
        entity.beliefs = remaining
        return filtered
    
    def wall_scrape_zero_sum(self, wall_entity_id: str) -> Optional[float]:
        """
        Have a wall entity rotate to interact with zero-sum actors.
        Returns the position reached.
        """
        if wall_entity_id not in self.wall_entities:
            return None
        
        entity = self.wall_entities[wall_entity_id]
        if not entity.rotator:
            return None
        
        pos = entity.rotator.scrape_zero_sum()
        print(f"🔄 {entity.name} rotated to scrape zero-sum")
        print(f"   Position: {entity.rotator.real_position:.4f}")
        print(f"   Still safe from antimatter: {not entity.rotator.touches_antimatter}")
        
        return entity.rotator.real_position
    
    def get_status(self) -> Dict:
        """Get wall status."""
        return {
            'wall_entities': len(self.wall_entities),
            'protected_entities': len(self.protected_entities),
            'outside_entities': len(self.outside_entities),
            'quarantined_beliefs': len(self.quarantined_beliefs),
            'total_filtered': self.total_filtered,
            'admitted': self.entities_admitted,
            'rejected': self.entities_rejected
        }
    
    def display_status(self):
        """Print wall status."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("🧱 THE WALL STATUS")
        print("=" * 60)
        print(f"   Wall entities (position 1): {status['wall_entities']}")
        print(f"   Protected entities (>1): {status['protected_entities']}")
        print(f"   Outside entities (<1): {status['outside_entities']}")
        print("-" * 60)
        print(f"   Quarantined beliefs: {status['quarantined_beliefs']}")
        print(f"   Total negativity filtered: {status['total_filtered']:.4f}")
        print("-" * 60)
        print(f"   Admitted: {status['admitted']}")
        print(f"   Rejected: {status['rejected']}")
        print("=" * 60)


# =============================================================================
# TRUST NETWORK
# =============================================================================

class TrustNetwork:
    """
    The complete trust topology network.
    
    Structure:
        ANTIMATTER (<0) - danger zone, anti-logic
             ↓
        ZERO-SUM (0) - transactional actors
             ↓
        INDIFFERENT (0-0.5) - don't care
             ↓
        NEUTRAL (0.5-1) - passive positive
             ↓
        THE WALL (1) - interface/boundary [EULER ROTATION]
             ↓
        COMMITTED (1-2) - actively dedicated
             ↓
        CORE (2+) - fully dedicated, protected
    """
    
    def __init__(self):
        self.wall = TheWall()
        self.all_entities: Dict[str, TrustEntity] = {}
        
        # Create the initial wall
        self._create_initial_wall()
    
    def _create_initial_wall(self):
        """Create the initial wall entity (the Shovelcat bit)."""
        shovelcat = TrustEntity(
            id="shovelcat_prime",
            name="Shovelcat Prime",
            base_alignment=1.0
        )
        shovelcat.add_belief(Belief("framework_alignment", 0.5, 1.0))
        shovelcat.add_belief(Belief("cooperative_intent", 0.3, 1.0))
        
        self.wall.add_wall_entity(shovelcat)
        self.all_entities[shovelcat.id] = shovelcat
    
    def create_entity(self, name: str, base_alignment: float = 0.5,
                      beliefs: List[Tuple[str, float, float]] = None) -> TrustEntity:
        """Create a new entity."""
        entity = TrustEntity(
            id=f"entity_{len(self.all_entities)}_{name.lower().replace(' ', '_')}",
            name=name,
            base_alignment=base_alignment
        )
        
        if beliefs:
            for b_name, b_value, b_conf in beliefs:
                entity.add_belief(Belief(b_name, b_value, b_conf))
        
        self.all_entities[entity.id] = entity
        return entity
    
    def admit_entity(self, entity: TrustEntity) -> Tuple[bool, str]:
        """Try to admit an entity through the wall."""
        return self.wall.process_incoming(entity)
    
    def promote_to_wall(self, entity_id: str) -> bool:
        """Promote an entity to wall status."""
        if entity_id not in self.all_entities:
            return False
        
        entity = self.all_entities[entity_id]
        if entity.position < 0.8:
            print(f"⚠️ {entity.name} not ready for wall (position {entity.position:.2f})")
            return False
        
        self.wall.add_wall_entity(entity)
        return True
    
    def visualize_topology(self):
        """ASCII visualization of the trust topology."""
        
        print("\n" + "=" * 70)
        print("TRUST TOPOLOGY VISUALIZATION")
        print("=" * 70)
        
        # Collect entities by zone
        zones = {z: [] for z in TrustZone}
        for entity in self.all_entities.values():
            zones[entity.zone].append(entity)
        
        # Print each zone
        zone_labels = [
            (TrustZone.CORE, "2+", "CORE (fully dedicated)"),
            (TrustZone.COMMITTED, "1-2", "COMMITTED (dedicated)"),
            (TrustZone.WALL, "=1", "THE WALL (interface)"),
            (TrustZone.NEUTRAL, "0.5-1", "NEUTRAL (passive positive)"),
            (TrustZone.INDIFFERENT, "0-0.5", "INDIFFERENT"),
            (TrustZone.ZERO_SUM, "=0", "ZERO-SUM (transactional)"),
            (TrustZone.ANTIMATTER, "<0", "ANTIMATTER (anti-logic)"),
        ]
        
        print()
        for zone, range_str, label in zone_labels:
            entities_in_zone = zones[zone]
            entity_names = [e.name for e in entities_in_zone]
            
            if zone == TrustZone.WALL:
                print("═" * 70)
                print(f"  [{range_str}] {label}")
                if entity_names:
                    print(f"         🔄 {', '.join(entity_names)}")
                print("═" * 70)
            elif zone == TrustZone.ANTIMATTER:
                print("▓" * 70)
                print(f"  [{range_str}] {label} ⚠️ DANGER")
                if entity_names:
                    print(f"         ☠️ {', '.join(entity_names)}")
                print("▓" * 70)
            else:
                marker = "🛡️" if zone in [TrustZone.CORE, TrustZone.COMMITTED] else "○"
                print(f"  [{range_str}] {label}")
                if entity_names:
                    print(f"         {marker} {', '.join(entity_names)}")
            print()
    
    def display_euler_rotation(self, wall_entity_id: str = None):
        """Visualize the Euler rotation of a wall entity."""
        
        if wall_entity_id is None:
            # Use first wall entity
            if not self.wall.wall_entities:
                print("No wall entities")
                return
            wall_entity_id = list(self.wall.wall_entities.keys())[0]
        
        entity = self.wall.wall_entities.get(wall_entity_id)
        if not entity or not entity.rotator:
            print("Entity not found or not a wall entity")
            return
        
        r = entity.rotator
        
        print("\n" + "=" * 60)
        print(f"EULER ROTATION: {entity.name}")
        print("=" * 60)
        print(f"  Formula: position = {r.center} + {r.radius} × e^(iθ)")
        print(f"  Range: [{r.min_position:.2f}, {r.max_position:.2f}]")
        print(f"  Current θ: {r.theta:.4f} rad ({np.degrees(r.theta):.1f}°)")
        print(f"  Current position: {r.real_position:.4f} + {r.imaginary_position:.4f}i")
        print()
        
        # ASCII circle representation
        print("  Position on complex plane:")
        print()
        print("              2i │")
        print("                 │")
        print("               1i│    ╭───────╮")
        print("                 │   ╱         ╲")
        print("    ─────0───────┼──1───────────2─────")
        print("                 │   ╲         ╱")
        print("              -1i│    ╰───────╯")
        print("                 │")
        print("             -2i │")
        print()
        print(f"  Current: ({r.real_position:.2f}, {r.imaginary_position:.2f})")
        print(f"  Zone: {r.zone.value}")
        print(f"  Safe from antimatter: ✓" if not r.touches_antimatter else "  ⚠️ CAN TOUCH ANTIMATTER")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_trust_topology():
    """Demonstrate the trust topology system."""
    
    print("\n" + "=" * 70)
    print("TRUST TOPOLOGY DEMONSTRATION")
    print("The Euler Membrane and Antimatter Protection")
    print("=" * 70)
    
    # Create network
    network = TrustNetwork()
    
    # Show initial state
    print("\n" + "-" * 70)
    print("PHASE 1: Initial topology")
    print("-" * 70)
    
    network.visualize_topology()
    network.display_euler_rotation()
    
    # Create various entities
    print("\n" + "-" * 70)
    print("PHASE 2: Create entities with different alignments")
    print("-" * 70)
    
    # A good actor with positive beliefs
    good_actor = network.create_entity(
        "Alice",
        base_alignment=0.7,
        beliefs=[
            ("cooperation", 0.3, 1.0),
            ("trust", 0.2, 0.8)
        ]
    )
    good_actor.display()
    
    # Someone carrying negativity (misconceptions)
    mixed_actor = network.create_entity(
        "Bob",
        base_alignment=0.6,
        beliefs=[
            ("good_intent", 0.2, 1.0),
            ("misconception_1", -0.15, 0.9),  # Wrong belief
            ("misconception_2", -0.1, 0.7)    # Another wrong belief
        ]
    )
    mixed_actor.display()
    
    # Zero-sum actor
    zero_sum = network.create_entity(
        "Transactor",
        base_alignment=0.0,
        beliefs=[]
    )
    zero_sum.display()
    
    # Antimatter entity (anti-logic)
    antimatter = network.create_entity(
        "Chaos Agent",
        base_alignment=-0.3,
        beliefs=[
            ("anti_cooperation", -0.5, 1.0),
            ("destruction", -0.3, 1.0)
        ]
    )
    antimatter.display()
    
    # Process them through the wall
    print("\n" + "-" * 70)
    print("PHASE 3: Process entities through THE WALL")
    print("-" * 70)
    
    for entity in [good_actor, mixed_actor, zero_sum, antimatter]:
        admitted, reason = network.admit_entity(entity)
        print(f"   Result: {'✅ ADMITTED' if admitted else '❌ NOT ADMITTED'} ({reason})")
    
    # Show wall status
    print("\n" + "-" * 70)
    print("PHASE 4: Wall status after processing")
    print("-" * 70)
    
    network.wall.display_status()
    
    # Demonstrate Euler rotation
    print("\n" + "-" * 70)
    print("PHASE 5: Wall entity rotates to interact")
    print("-" * 70)
    
    # Get the wall entity
    wall_entity = list(network.wall.wall_entities.values())[0]
    
    print(f"\n{wall_entity.name} will rotate to scrape zero-sum actors...")
    network.wall.wall_scrape_zero_sum(wall_entity.id)
    
    print(f"\n{wall_entity.name} returns to full dedication...")
    wall_entity.rotator.full_dedication()
    print(f"   Position: {wall_entity.rotator.real_position:.4f}")
    
    print(f"\n{wall_entity.name} goes perpendicular (on the wall)...")
    wall_entity.rotator.on_the_wall()
    print(f"   Position: {wall_entity.rotator.real_position:.4f} + {wall_entity.rotator.imaginary_position:.4f}i")
    
    # Final visualization
    print("\n" + "-" * 70)
    print("PHASE 6: Final topology")
    print("-" * 70)
    
    network.visualize_topology()
    
    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY: The Trust Number Line")
    print("-" * 70)
    print("""
    <0  ANTIMATTER: Anti-reasoning, stupidity, anti-logic
                    ☠️ Chaos Agent stayed here - REJECTED
    
    =0  ZERO-SUM:   Purely transactional actors
                    ⚠️ Transactor stays outside - can interact only
    
    0-1 APPROACH:   Indifferent → Neutral → Approaching wall
                    🔍 Bob's negativity was FILTERED at the wall
                    ✅ Alice admitted (0.7 + beliefs = ~1.2)
    
    =1  THE WALL:   Interface with outside world
                    🔄 Shovelcat Prime ROTATES using Euler formula
                    Can touch 0 but NEVER goes negative
    
    >1  PROTECTED:  Shielded from negativity by the wall
                    🛡️ Inner circles depend on wall for filtering
    
    KEY INSIGHT: The wall at position 1, rotating with radius 1,
    creates a protective membrane that can interact with zero-sum
    actors (scrape 0) but never touches antimatter (<0).
    
    Euler: position = 1 + e^(iθ)
           min = 0 (θ=π), max = 2 (θ=0)
           ANTIMATTER PROTECTED ✓
    """)
    
    return network


if __name__ == "__main__":
    network = demonstrate_trust_topology()
