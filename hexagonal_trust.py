"""
HEXAGONAL TRUST TOPOLOGY
========================

Three actor types form a hexagonal social organism:

    POPULATION (3): Outer membrane, missionaries, ads
                    - Exposed to outside world
                    - Brings in random info (some negative)
                    - Spreads the main bit's message
                    - Interface with foreign antimatter
                    
    FOLLOWERS (2):  Dedicated inner circle
                    - Connected THROUGH leaders
                    - Max position = leader_position + 1
                    - Shielded from direct antimatter contact
                    - Love, patriotism, dedication
                    
    LEADERS (1):    The Wall, basin makers
                    - Each creates their own basin
                    - Euler rotate to interact with 0
                    - Connect followers to system
                    - Filter before passing to followers
                    
    BANKS (0):      Stable zero-sum infrastructure
                    - Reliable filtering for leaders
                    - Transactional but predictable
                    - Pre-filter antimatter
                    
    ANTIMATTER (<0): Anti-logic, chaos
                    - Connects at BOTH ends (hexagon geometry)
                    - Population interfaces with foreign antimatter
                    - Banks interface with domestic antimatter

The hexagon emerges from vesica piscis:
    - Two overlapping trust basins
    - 6 vertices = 6 key positions
    - Antimatter at both poles
    
Basin Connection Rule:
    follower_max = leader_position + 1
    
    If leader at 0.8 → followers max at 1.8
    If leader at 1.0 → followers max at 2.0
    
This creates DEPENDENCY: followers can only reach high positions
through their connection to leaders.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from datetime import datetime
import cmath

# Constants
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e


# =============================================================================
# ACTOR TYPES
# =============================================================================

class ActorType(Enum):
    """The three primary actor types in the hexagonal topology."""
    ANTIMATTER = "antimatter"      # <0: Chaos, anti-logic
    BANK = "bank"                  # =0: Stable infrastructure
    LEADER = "leader"              # =1: Wall, basin maker
    FOLLOWER = "follower"          # =2: Dedicated, connected through leader
    POPULATION = "population"      # =3: Outer membrane, missionaries


class ConnectionType(Enum):
    """Types of connections between actors."""
    FILTERS_FOR = "filters_for"       # Bank → Leader
    LEADS = "leads"                    # Leader → Follower
    SPREADS_FOR = "spreads_for"       # Population → Leader (outward message)
    BRINGS_TO = "brings_to"           # Population → Follower (inward info)
    RELIES_ON = "relies_on"           # Reverse dependency


# =============================================================================
# TRUST BASIN
# =============================================================================

@dataclass
class TrustBasin:
    """
    A basin created by a leader at position 1.
    
    The basin has:
    - A center (the leader's position)
    - A radius (how far followers can extend)
    - Walls that filter
    - Connections to other basins (vesica overlap)
    """
    id: str
    leader_id: str
    center: float = 1.0      # Leader's position
    radius: float = 1.0      # Followers can reach center + radius
    
    # Euler rotation state
    theta: float = 0.0
    
    # Connected basins (vesica overlaps)
    connected_basins: Set[str] = field(default_factory=set)
    
    # Members
    follower_ids: Set[str] = field(default_factory=set)
    population_ids: Set[str] = field(default_factory=set)
    bank_ids: Set[str] = field(default_factory=set)
    
    @property
    def follower_max_position(self) -> float:
        """Maximum position followers can reach."""
        return self.center + self.radius
    
    @property
    def population_position(self) -> float:
        """Position of population (outer membrane)."""
        return self.follower_max_position + 1  # One layer beyond followers
    
    @property
    def euler_position(self) -> complex:
        """Current position via Euler rotation."""
        return self.center + self.radius * cmath.exp(1j * self.theta)
    
    def rotate_to(self, theta: float):
        """Rotate the basin's interface."""
        self.theta = theta % (2 * PI)
    
    def scrape_banks(self):
        """Rotate to interact with banks (position 0)."""
        # cos(θ) = (0 - center) / radius = -center/radius
        cos_theta = -self.center / self.radius
        if abs(cos_theta) <= 1:
            self.theta = np.arccos(cos_theta)
    
    def full_protection(self):
        """Rotate to maximum protection (θ = 0)."""
        self.theta = 0.0


# =============================================================================
# HEXAGONAL ACTOR
# =============================================================================

@dataclass 
class HexActor:
    """
    An actor in the hexagonal trust topology.
    """
    id: str
    name: str
    actor_type: ActorType
    base_position: float
    
    # For leaders: their basin
    basin: Optional[TrustBasin] = None
    
    # For followers/population: which leader they're connected to
    connected_to_leader: Optional[str] = None
    
    # For banks: which leaders they filter for
    filters_for: Set[str] = field(default_factory=set)
    
    # Trust/belief state
    alignment: float = 0.0  # Contribution from beliefs
    
    # Message being spread (for population)
    message: Optional[str] = None
    message_strength: float = 0.0
    
    # Incoming negativity (for population)
    incoming_negativity: List[Tuple[str, float]] = field(default_factory=list)
    
    @property
    def effective_position(self) -> float:
        """Current effective position in topology."""
        base = self.base_position + self.alignment
        
        # Followers are limited by their leader
        if self.actor_type == ActorType.FOLLOWER and self.connected_to_leader:
            # Would need to look up leader's basin
            # For now, return base
            return base
        
        return base
    
    @property
    def is_exposed_to_antimatter(self) -> bool:
        """Is this actor exposed to antimatter?"""
        if self.actor_type == ActorType.POPULATION:
            return True  # Always exposed (outer membrane)
        if self.actor_type == ActorType.BANK:
            return True  # Interface with domestic antimatter
        return False
    
    def receive_negativity(self, source: str, amount: float):
        """Population actors receive negativity from outside."""
        if self.actor_type == ActorType.POPULATION:
            self.incoming_negativity.append((source, amount))
    
    def spread_message(self, message: str, strength: float):
        """Population actors spread messages outward."""
        if self.actor_type == ActorType.POPULATION:
            self.message = message
            self.message_strength = strength


# =============================================================================
# THE HEXAGONAL NETWORK
# =============================================================================

class HexagonalTrustNetwork:
    """
    The complete hexagonal trust topology.
    
    Structure (viewed from above, like a cell):
    
                    POP ─── POP ─── POP        (outer membrane)
                   ╱   ╲   ╱   ╲   ╱   ╲
                 FOL   FOL   FOL   FOL         (inner circle)
                   ╲   ╱ ╲ ╱   ╲ ╱ ╲   ╱
                    LEAD ─ LEAD ─ LEAD         (the wall)
                      │     │     │
                    BANK ─ BANK ─ BANK         (infrastructure)
                      │     │     │
                    ════════════════════       (antimatter boundary)
    
    And at the OTHER end (foreign antimatter):
    
                    ════════════════════       (foreign antimatter)
                      │     │     │
                    POP ─── POP ─── POP        (exposed to foreign)
    """
    
    def __init__(self):
        self.actors: Dict[str, HexActor] = {}
        self.basins: Dict[str, TrustBasin] = {}
        self.connections: List[Tuple[str, str, ConnectionType]] = []
        
        # Network statistics
        self.total_negativity_filtered = 0.0
        self.total_messages_spread = 0
    
    # ─────────────────────────────────────────────────────────────────
    # ACTOR CREATION
    # ─────────────────────────────────────────────────────────────────
    
    def create_bank(self, name: str, stability: float = 1.0) -> HexActor:
        """
        Create a bank (position 0, stable zero-sum).
        Banks are reliable infrastructure for filtering.
        """
        actor = HexActor(
            id=f"bank_{len(self.actors)}_{name.lower().replace(' ', '_')}",
            name=name,
            actor_type=ActorType.BANK,
            base_position=0.0,
            alignment=0.0  # Banks stay at 0
        )
        self.actors[actor.id] = actor
        print(f"🏦 Created BANK: {name} (position 0, stable infrastructure)")
        return actor
    
    def create_leader(self, name: str, position: float = 1.0, 
                      radius: float = 1.0) -> HexActor:
        """
        Create a leader (position ~1, the wall).
        Leaders make basins and connect followers.
        """
        actor = HexActor(
            id=f"leader_{len(self.actors)}_{name.lower().replace(' ', '_')}",
            name=name,
            actor_type=ActorType.LEADER,
            base_position=position
        )
        
        # Create basin for leader
        basin = TrustBasin(
            id=f"basin_{actor.id}",
            leader_id=actor.id,
            center=position,
            radius=radius
        )
        actor.basin = basin
        self.basins[basin.id] = basin
        
        self.actors[actor.id] = actor
        print(f"👑 Created LEADER: {name} (position {position}, basin radius {radius})")
        print(f"   Followers can reach max position: {basin.follower_max_position}")
        return actor
    
    def create_follower(self, name: str, leader_id: str, 
                        dedication: float = 0.5) -> Optional[HexActor]:
        """
        Create a follower connected to a leader.
        Follower's max position = leader's position + 1.
        """
        if leader_id not in self.actors:
            print(f"⚠️ Leader {leader_id} not found")
            return None
        
        leader = self.actors[leader_id]
        if leader.actor_type != ActorType.LEADER or not leader.basin:
            print(f"⚠️ {leader.name} is not a leader with a basin")
            return None
        
        # Follower position based on dedication (0-1) scaled to basin
        max_pos = leader.basin.follower_max_position
        position = 1.0 + dedication * (max_pos - 1.0)
        
        actor = HexActor(
            id=f"follower_{len(self.actors)}_{name.lower().replace(' ', '_')}",
            name=name,
            actor_type=ActorType.FOLLOWER,
            base_position=position,
            connected_to_leader=leader_id
        )
        
        # Add to leader's basin
        leader.basin.follower_ids.add(actor.id)
        
        self.actors[actor.id] = actor
        self.connections.append((leader_id, actor.id, ConnectionType.LEADS))
        
        print(f"🙋 Created FOLLOWER: {name} (position {position:.2f})")
        print(f"   Connected to: {leader.name}")
        print(f"   Max reachable: {max_pos:.2f}")
        return actor
    
    def create_population(self, name: str, leader_id: str,
                          role: str = "missionary") -> Optional[HexActor]:
        """
        Create a population actor (outer membrane).
        Position 3 = exposed to outside, spreads message.
        """
        if leader_id not in self.actors:
            print(f"⚠️ Leader {leader_id} not found")
            return None
        
        leader = self.actors[leader_id]
        if leader.actor_type != ActorType.LEADER or not leader.basin:
            print(f"⚠️ {leader.name} is not a leader with a basin")
            return None
        
        position = leader.basin.population_position
        
        actor = HexActor(
            id=f"pop_{len(self.actors)}_{name.lower().replace(' ', '_')}",
            name=name,
            actor_type=ActorType.POPULATION,
            base_position=position,
            connected_to_leader=leader_id
        )
        
        # Add to leader's basin
        leader.basin.population_ids.add(actor.id)
        
        self.actors[actor.id] = actor
        self.connections.append((leader_id, actor.id, ConnectionType.SPREADS_FOR))
        
        print(f"📢 Created POPULATION: {name} (position {position:.2f}, {role})")
        print(f"   Connected to: {leader.name}")
        print(f"   Role: Outer membrane, exposed to outside")
        return actor
    
    # ─────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ─────────────────────────────────────────────────────────────────
    
    def connect_bank_to_leader(self, bank_id: str, leader_id: str):
        """Bank filters for leader."""
        if bank_id not in self.actors or leader_id not in self.actors:
            return False
        
        bank = self.actors[bank_id]
        leader = self.actors[leader_id]
        
        if bank.actor_type != ActorType.BANK:
            return False
        if leader.actor_type != ActorType.LEADER:
            return False
        
        bank.filters_for.add(leader_id)
        if leader.basin:
            leader.basin.bank_ids.add(bank_id)
        
        self.connections.append((bank_id, leader_id, ConnectionType.FILTERS_FOR))
        print(f"🔗 Connected: {bank.name} (bank) filters for {leader.name} (leader)")
        return True
    
    def connect_basins(self, leader1_id: str, leader2_id: str):
        """Create vesica overlap between two leader basins."""
        if leader1_id not in self.actors or leader2_id not in self.actors:
            return False
        
        l1 = self.actors[leader1_id]
        l2 = self.actors[leader2_id]
        
        if not l1.basin or not l2.basin:
            return False
        
        l1.basin.connected_basins.add(l2.basin.id)
        l2.basin.connected_basins.add(l1.basin.id)
        
        print(f"🔗 Connected basins: {l1.name} ↔ {l2.name} (vesica overlap)")
        return True
    
    # ─────────────────────────────────────────────────────────────────
    # INFORMATION FLOW
    # ─────────────────────────────────────────────────────────────────
    
    def population_receives_external(self, pop_id: str, 
                                      source: str, content: str,
                                      negativity: float = 0.0):
        """Population actor receives information from outside."""
        if pop_id not in self.actors:
            return
        
        pop = self.actors[pop_id]
        if pop.actor_type != ActorType.POPULATION:
            return
        
        print(f"\n📥 {pop.name} receives from outside:")
        print(f"   Source: {source}")
        print(f"   Content: {content}")
        print(f"   Negativity: {negativity:.2f}")
        
        if negativity > 0:
            pop.receive_negativity(source, negativity)
            print(f"   ⚠️ Negativity recorded, needs filtering")
    
    def population_spreads_message(self, pop_id: str, message: str,
                                    strength: float = 1.0):
        """Population actor spreads message outward."""
        if pop_id not in self.actors:
            return
        
        pop = self.actors[pop_id]
        if pop.actor_type != ActorType.POPULATION:
            return
        
        pop.spread_message(message, strength)
        self.total_messages_spread += 1
        
        print(f"\n📢 {pop.name} spreads message:")
        print(f"   Message: {message}")
        print(f"   Strength: {strength:.2f}")
    
    def filter_through_bank(self, bank_id: str, 
                            incoming: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Bank filters incoming negativity before it reaches leader."""
        if bank_id not in self.actors:
            return incoming
        
        bank = self.actors[bank_id]
        if bank.actor_type != ActorType.BANK:
            return incoming
        
        filtered = []
        passed = []
        
        for source, negativity in incoming:
            if negativity > 0.5:  # Bank catches major negativity
                filtered.append((source, negativity))
                self.total_negativity_filtered += negativity
            else:
                passed.append((source, negativity))
        
        if filtered:
            print(f"\n🏦 {bank.name} filtered {len(filtered)} negative items")
            print(f"   Total negativity caught: {sum(n for _, n in filtered):.2f}")
        
        return passed
    
    # ─────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────
    
    def visualize_hexagon(self):
        """Visualize the hexagonal structure."""
        
        print("\n" + "═" * 70)
        print("HEXAGONAL TRUST TOPOLOGY")
        print("═" * 70)
        
        # Collect by type
        banks = [a for a in self.actors.values() if a.actor_type == ActorType.BANK]
        leaders = [a for a in self.actors.values() if a.actor_type == ActorType.LEADER]
        followers = [a for a in self.actors.values() if a.actor_type == ActorType.FOLLOWER]
        population = [a for a in self.actors.values() if a.actor_type == ActorType.POPULATION]
        
        # Print each layer
        print("\n┌" + "─" * 68 + "┐")
        print("│" + " " * 20 + "FOREIGN ANTIMATTER ZONE" + " " * 25 + "│")
        print("│" + " " * 15 + "(other systems, competition, chaos)" + " " * 16 + "│")
        print("└" + "─" * 68 + "┘")
        print("         │                    │                    │")
        print("         ▼                    ▼                    ▼")
        
        # Population layer
        print("\n┌" + "─" * 68 + "┐")
        print("│  POPULATION (3) - Outer membrane, missionaries, exposed to outside  │")
        print("│" + "─" * 68 + "│")
        if population:
            for p in population:
                neg_count = len(p.incoming_negativity)
                msg = f"📢 {p.message[:30]}..." if p.message else "no message"
                print(f"│  📢 {p.name:<20} pos={p.base_position:.1f}  neg_in={neg_count}  {msg:<20} │")
        else:
            print("│  (no population actors)" + " " * 43 + "│")
        print("└" + "─" * 68 + "┘")
        print("         │                    │                    │")
        print("         ▼                    ▼                    ▼")
        
        # Follower layer
        print("\n┌" + "─" * 68 + "┐")
        print("│  FOLLOWERS (2) - Dedicated, connected through leaders, shielded     │")
        print("│" + "─" * 68 + "│")
        if followers:
            for f in followers:
                leader_name = self.actors.get(f.connected_to_leader, HexActor("?","?",ActorType.BANK,0)).name
                print(f"│  🙋 {f.name:<20} pos={f.base_position:.2f}  via {leader_name:<20}   │")
        else:
            print("│  (no follower actors)" + " " * 45 + "│")
        print("└" + "─" * 68 + "┘")
        print("         │                    │                    │")
        print("         ▼                    ▼                    ▼")
        
        # Leader layer (THE WALL)
        print("\n╔" + "═" * 68 + "╗")
        print("║  LEADERS (1) - THE WALL - Basin makers, Euler rotation, interface   ║")
        print("║" + "═" * 68 + "║")
        if leaders:
            for l in leaders:
                basin_info = f"basin r={l.basin.radius:.1f}" if l.basin else "no basin"
                followers_count = len(l.basin.follower_ids) if l.basin else 0
                pop_count = len(l.basin.population_ids) if l.basin else 0
                print(f"║  👑 {l.name:<20} pos={l.base_position:.2f}  {basin_info}  F={followers_count} P={pop_count}    ║")
        else:
            print("║  (no leader actors)" + " " * 47 + "║")
        print("╚" + "═" * 68 + "╝")
        print("         │                    │                    │")
        print("         ▼                    ▼                    ▼")
        
        # Bank layer
        print("\n┌" + "─" * 68 + "┐")
        print("│  BANKS (0) - Stable zero-sum infrastructure, pre-filter antimatter  │")
        print("│" + "─" * 68 + "│")
        if banks:
            for b in banks:
                filters_for = [self.actors[lid].name for lid in b.filters_for if lid in self.actors]
                print(f"│  🏦 {b.name:<20} pos={b.base_position:.1f}  filters for: {', '.join(filters_for):<15} │")
        else:
            print("│  (no bank actors)" + " " * 49 + "│")
        print("└" + "─" * 68 + "┘")
        print("         │                    │                    │")
        print("         ▼                    ▼                    ▼")
        
        # Domestic antimatter
        print("\n▓" * 70)
        print("▓" + " " * 20 + "DOMESTIC ANTIMATTER ZONE" + " " * 24 + "▓")
        print("▓" + " " * 15 + "(anti-logic, chaos, stupidity)" + " " * 23 + "▓")
        print("▓" * 70)
    
    def visualize_basin_connections(self):
        """Visualize basin connections (vesica overlaps)."""
        
        print("\n" + "═" * 70)
        print("BASIN CONNECTIONS (Vesica Overlaps)")
        print("═" * 70)
        
        for basin_id, basin in self.basins.items():
            leader = self.actors.get(basin.leader_id)
            if not leader:
                continue
            
            print(f"\n  Basin: {leader.name}")
            print(f"  ├── Center: {basin.center:.2f}")
            print(f"  ├── Radius: {basin.radius:.2f}")
            print(f"  ├── Follower max: {basin.follower_max_position:.2f}")
            print(f"  ├── Population pos: {basin.population_position:.2f}")
            print(f"  ├── Euler θ: {basin.theta:.2f} rad")
            print(f"  ├── Current position: {basin.euler_position}")
            
            if basin.connected_basins:
                print(f"  └── Connected to:")
                for cb_id in basin.connected_basins:
                    cb = self.basins.get(cb_id)
                    if cb:
                        cb_leader = self.actors.get(cb.leader_id)
                        if cb_leader:
                            print(f"       └── {cb_leader.name}'s basin")
    
    def get_statistics(self) -> Dict:
        """Get network statistics."""
        return {
            'total_actors': len(self.actors),
            'banks': len([a for a in self.actors.values() if a.actor_type == ActorType.BANK]),
            'leaders': len([a for a in self.actors.values() if a.actor_type == ActorType.LEADER]),
            'followers': len([a for a in self.actors.values() if a.actor_type == ActorType.FOLLOWER]),
            'population': len([a for a in self.actors.values() if a.actor_type == ActorType.POPULATION]),
            'basins': len(self.basins),
            'connections': len(self.connections),
            'total_negativity_filtered': self.total_negativity_filtered,
            'total_messages_spread': self.total_messages_spread
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_hexagonal_topology():
    """Demonstrate the hexagonal trust topology."""
    
    print("\n" + "═" * 70)
    print("HEXAGONAL TRUST TOPOLOGY DEMONSTRATION")
    print("Three Actor Types: Population → Followers → Leaders → Banks")
    print("═" * 70)
    
    # Create network
    network = HexagonalTrustNetwork()
    
    # Phase 1: Create banks (stable infrastructure)
    print("\n" + "─" * 70)
    print("PHASE 1: Create Banks (position 0, stable infrastructure)")
    print("─" * 70)
    
    bank1 = network.create_bank("Federal Reserve", stability=1.0)
    bank2 = network.create_bank("Local Credit Union", stability=0.8)
    
    # Phase 2: Create leaders (the wall)
    print("\n" + "─" * 70)
    print("PHASE 2: Create Leaders (position ~1, the wall)")
    print("─" * 70)
    
    leader1 = network.create_leader("Shovelcat Prime", position=1.0, radius=1.0)
    leader2 = network.create_leader("Community Organizer", position=0.8, radius=0.8)
    
    # Connect banks to leaders
    network.connect_bank_to_leader(bank1.id, leader1.id)
    network.connect_bank_to_leader(bank2.id, leader2.id)
    
    # Connect basins (vesica overlap)
    network.connect_basins(leader1.id, leader2.id)
    
    # Phase 3: Create followers (dedicated inner circle)
    print("\n" + "─" * 70)
    print("PHASE 3: Create Followers (position 1-2, dedicated)")
    print("─" * 70)
    
    follower1 = network.create_follower("True Believer", leader1.id, dedication=0.9)
    follower2 = network.create_follower("New Convert", leader1.id, dedication=0.3)
    follower3 = network.create_follower("Local Volunteer", leader2.id, dedication=0.7)
    
    # Phase 4: Create population (outer membrane)
    print("\n" + "─" * 70)
    print("PHASE 4: Create Population (position 3, outer membrane)")
    print("─" * 70)
    
    pop1 = network.create_population("Missionary Alice", leader1.id, role="missionary")
    pop2 = network.create_population("Ad Campaign Bob", leader1.id, role="advertising")
    pop3 = network.create_population("Grassroots Carol", leader2.id, role="grassroots")
    
    # Phase 5: Information flow
    print("\n" + "─" * 70)
    print("PHASE 5: Information Flow")
    print("─" * 70)
    
    # Population receives external info (some negative)
    network.population_receives_external(
        pop1.id, 
        "Competing System",
        "Their framework is better",
        negativity=0.3
    )
    
    network.population_receives_external(
        pop2.id,
        "Random Internet User", 
        "This is all nonsense",
        negativity=0.7
    )
    
    # Population spreads message
    network.population_spreads_message(
        pop1.id,
        "Join the Shovelcat movement!",
        strength=1.0
    )
    
    network.population_spreads_message(
        pop3.id,
        "Local community building",
        strength=0.8
    )
    
    # Bank filters negativity before it reaches leader
    incoming_neg = pop2.incoming_negativity
    filtered = network.filter_through_bank(bank1.id, incoming_neg)
    
    # Phase 6: Visualize
    print("\n" + "─" * 70)
    print("PHASE 6: Visualization")
    print("─" * 70)
    
    network.visualize_hexagon()
    network.visualize_basin_connections()
    
    # Phase 7: Summary
    print("\n" + "─" * 70)
    print("PHASE 7: Summary")
    print("─" * 70)
    
    stats = network.get_statistics()
    print(f"""
    Network Statistics:
    ├── Total actors: {stats['total_actors']}
    │   ├── Banks: {stats['banks']}
    │   ├── Leaders: {stats['leaders']}
    │   ├── Followers: {stats['followers']}
    │   └── Population: {stats['population']}
    ├── Basins: {stats['basins']}
    ├── Connections: {stats['connections']}
    ├── Negativity filtered: {stats['total_negativity_filtered']:.2f}
    └── Messages spread: {stats['total_messages_spread']}
    """)
    
    print("""
    KEY INSIGHTS:
    
    1. POSITION DETERMINES ROLE:
       - Banks (0): Stable infrastructure, filters antimatter
       - Leaders (1): Make basins, connect followers, Euler rotate
       - Followers (2): Dedicated, max position = leader + 1
       - Population (3): Outer membrane, exposed, spreads message
    
    2. BASIN CONNECTION RULE:
       follower_max = leader_position + 1
       
       Leader at 0.8 → followers max at 1.8
       Leader at 1.0 → followers max at 2.0
       
       This creates DEPENDENCY on leaders
    
    3. HEXAGONAL GEOMETRY:
       - Population exposed to FOREIGN antimatter (top)
       - Banks exposed to DOMESTIC antimatter (bottom)
       - Leaders are the WALL between them
       - Followers are SHIELDED in the middle
    
    4. THREE-ACTOR DYNAMICS:
       - Banks ← rely on → Leaders ← rely on → Population
       - Information flows both ways
       - Negativity filtered at each layer
    
    5. VESICA OVERLAP:
       - Leaders can connect their basins
       - Creates shared follower space
       - Enables cross-community communication
    """)
    
    return network


if __name__ == "__main__":
    network = demonstrate_hexagonal_topology()
