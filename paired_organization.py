"""
PAIRED ORGANIZATION ARCHITECTURE
================================

Every company is a 1-0 PAIR:
    - Position 1: Owner/Mission (creates, decides, explores)
    - Position 0: Finance/Budget (handles antimatter, protects owner)

The 0 actor shields the 1 actor from financial chaos:
    - Debt management
    - Market volatility  
    - Cash flow crises
    - Tax complexity
    - Legal antimatter

NEIGHBOR RELATIONSHIPS:

    Position 1 Neighbors:
        - Companies with aligned missions
        - Health Co ↔ Shovelcat (protein folder collaboration)
        - Education Co ↔ Shovelcat (teaching systems)
        - Can share followers, cross-promote
        
    Position 0 Neighbors:
        - Finance departments that coordinate
        - Shared banking relationships
        - Portfolio diversification across companies
        - Risk hedging between neighbors
        
    External (<0 interface):
        - Banks (established, stable)
        - Stock markets (volatile)
        - Credit systems
        - The 0 actors handle ALL antimatter contact

BENEFITS:
    - Owner never touches antimatter directly
    - Finance person specializes in chaos navigation
    - Clear separation of concerns
    - Neighbors can support each other at both levels
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from enum import Enum

# Constants
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e


# =============================================================================
# ACTOR TYPES
# =============================================================================

class ActorRole(Enum):
    """Roles in the paired architecture."""
    # Position 1 roles (Mission layer)
    OWNER = "owner"               # Company owner/founder
    MISSION_LEAD = "mission_lead" # Department head
    CREATIVE = "creative"         # Product/content creator
    
    # Position 0 roles (Finance layer)
    FINANCE = "finance"           # Company CFO/finance
    BUDGET = "budget"             # Department budget manager
    OPERATIONS = "operations"     # Operations/logistics
    
    # External roles
    BANK = "bank"                 # External bank
    MARKET = "market"             # Stock market
    INVESTOR = "investor"         # External investor


class AntimatterType(Enum):
    """Types of antimatter (financial chaos) that 0 actors handle."""
    DEBT = "debt"
    VOLATILITY = "volatility"
    CASH_CRISIS = "cash_crisis"
    TAX_COMPLEXITY = "tax_complexity"
    LEGAL_ISSUE = "legal_issue"
    MARKET_CRASH = "market_crash"
    CURRENCY_RISK = "currency_risk"
    INFLATION = "inflation"


# =============================================================================
# THE PAIRED ACTOR
# =============================================================================

@dataclass
class PairedActor:
    """
    An actor that exists in a 1-0 pair.
    
    Position 1 actors focus on mission/creation.
    Position 0 actors handle antimatter/finance.
    """
    id: str
    name: str
    role: ActorRole
    position: float  # 1.0 for mission, 0.0 for finance
    
    # The pair partner
    partner_id: Optional[str] = None
    
    # For position 1: mission focus
    mission: Optional[str] = None
    
    # For position 0: antimatter handling capacity
    antimatter_capacity: float = 100.0  # How much chaos can handle
    antimatter_absorbed: float = 0.0    # Current chaos load
    
    # Neighbors (same position level)
    neighbor_ids: Set[str] = field(default_factory=set)
    
    # Energy/resources
    energy: float = 1.0
    
    # External connections (for position 0)
    external_connections: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_mission_actor(self) -> bool:
        """Is this a position 1 (mission) actor?"""
        return self.position >= 1.0
    
    @property
    def is_finance_actor(self) -> bool:
        """Is this a position 0 (finance) actor?"""
        return self.position < 1.0 and self.position >= 0.0
    
    @property
    def antimatter_load(self) -> float:
        """Current antimatter load as percentage of capacity."""
        if self.antimatter_capacity == 0:
            return float('inf')
        return self.antimatter_absorbed / self.antimatter_capacity * 100
    
    @property
    def can_absorb_more(self) -> bool:
        """Can this actor absorb more antimatter?"""
        return self.antimatter_absorbed < self.antimatter_capacity
    
    def absorb_antimatter(self, amount: float, antimatter_type: AntimatterType) -> Tuple[float, float]:
        """
        Absorb antimatter (financial chaos).
        Returns (absorbed, overflow).
        """
        available_capacity = self.antimatter_capacity - self.antimatter_absorbed
        absorbed = min(amount, available_capacity)
        overflow = amount - absorbed
        
        self.antimatter_absorbed += absorbed
        
        return absorbed, overflow
    
    def process_antimatter(self, amount: float) -> float:
        """
        Process (reduce) antimatter through work.
        Returns amount processed.
        """
        processed = min(amount, self.antimatter_absorbed)
        self.antimatter_absorbed -= processed
        return processed
    
    def add_neighbor(self, neighbor_id: str):
        """Add a neighbor at the same position level."""
        self.neighbor_ids.add(neighbor_id)
    
    def connect_external(self, external_id: str, strength: float):
        """Connect to an external entity (banks, markets)."""
        self.external_connections[external_id] = strength


# =============================================================================
# THE ORGANIZATIONAL PAIR
# =============================================================================

@dataclass
class OrganizationalPair:
    """
    A 1-0 pair that forms the basic unit of organization.
    
    The pair consists of:
    - Owner (position 1): Focuses on mission, never touches antimatter
    - Finance (position 0): Handles all antimatter, shields owner
    """
    id: str
    name: str  # Company/department name
    
    # The pair
    owner: PairedActor = None
    finance: PairedActor = None
    
    # Mission
    mission: str = ""
    
    # Resources
    capital: float = 0.0
    
    # Neighbor pairs
    neighbor_pair_ids: Set[str] = field(default_factory=set)
    
    # History
    antimatter_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.owner is None:
            self.owner = PairedActor(
                id=f"{self.id}_owner",
                name=f"{self.name} Owner",
                role=ActorRole.OWNER,
                position=1.0,
                mission=self.mission
            )
        
        if self.finance is None:
            self.finance = PairedActor(
                id=f"{self.id}_finance",
                name=f"{self.name} Finance",
                role=ActorRole.FINANCE,
                position=0.0
            )
        
        # Link the pair
        self.owner.partner_id = self.finance.id
        self.finance.partner_id = self.owner.id
    
    def receive_antimatter(self, amount: float, 
                          antimatter_type: AntimatterType,
                          source: str) -> Dict:
        """
        Receive antimatter. Finance handles it, owner is shielded.
        """
        result = {
            'type': antimatter_type.value,
            'source': source,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'absorbed': 0.0,
            'overflow': 0.0,
            'owner_exposed': False
        }
        
        # Finance tries to absorb
        absorbed, overflow = self.finance.absorb_antimatter(amount, antimatter_type)
        result['absorbed'] = absorbed
        result['overflow'] = overflow
        
        # If overflow, try to spread to neighbor finances
        if overflow > 0:
            result['spread_to_neighbors'] = []
            # Would spread to neighbors here
        
        # Owner is only exposed if finance is completely overwhelmed
        if overflow > 0 and not self.finance.can_absorb_more:
            result['owner_exposed'] = True
            print(f"⚠️  WARNING: {self.owner.name} exposed to antimatter!")
        
        self.antimatter_history.append(result)
        
        return result
    
    def process_antimatter(self, work_energy: float) -> float:
        """
        Finance processes antimatter through work.
        Returns amount cleared.
        """
        return self.finance.process_antimatter(work_energy)
    
    def owner_create(self, creation: str, energy_cost: float) -> bool:
        """
        Owner creates something (mission work).
        Only possible if not exposed to antimatter.
        """
        if self.finance.antimatter_load > 90:  # Finance overwhelmed
            print(f"⚠️  {self.owner.name} can't create - finance overwhelmed")
            return False
        
        if energy_cost > self.owner.energy:
            print(f"⚠️  {self.owner.name} can't create - insufficient energy")
            return False
        
        self.owner.energy -= energy_cost
        print(f"✨ {self.owner.name} created: {creation}")
        return True
    
    def add_capital(self, amount: float, source: str):
        """Add capital to the pair."""
        self.capital += amount
        print(f"💰 {self.name} received ${amount:,.2f} from {source}")
    
    def get_status(self) -> Dict:
        """Get pair status."""
        return {
            'name': self.name,
            'mission': self.mission,
            'capital': self.capital,
            'owner_energy': self.owner.energy,
            'finance_antimatter_load': self.finance.antimatter_load,
            'finance_can_absorb': self.finance.can_absorb_more,
            'neighbor_count': len(self.neighbor_pair_ids)
        }
    
    def display(self):
        """Display pair status."""
        status = self.get_status()
        
        print(f"\n{'═'*50}")
        print(f"🏢 {self.name}")
        print(f"{'═'*50}")
        print(f"  Mission: {self.mission}")
        print(f"  Capital: ${self.capital:,.2f}")
        print(f"")
        print(f"  👑 OWNER (Position 1): {self.owner.name}")
        print(f"     Energy: {self.owner.energy:.2f}")
        print(f"     Focus: Mission (shielded from antimatter)")
        print(f"")
        print(f"  🏦 FINANCE (Position 0): {self.finance.name}")
        print(f"     Antimatter Load: {self.finance.antimatter_load:.1f}%")
        print(f"     Can Absorb More: {'✓' if self.finance.can_absorb_more else '✗'}")
        print(f"     External Connections: {len(self.finance.external_connections)}")
        print(f"")
        print(f"  Neighbors: {len(self.neighbor_pair_ids)}")


# =============================================================================
# THE PAIRED NETWORK
# =============================================================================

class PairedNetwork:
    """
    A network of organizational pairs.
    
    Structure:
        - Each company is a 1-0 pair
        - Position 1 actors neighbor other position 1 actors
        - Position 0 actors neighbor other position 0 actors
        - Position 0 actors connect to external banks/markets
    """
    
    def __init__(self, name: str = "Shovelcat Network"):
        self.name = name
        self.pairs: Dict[str, OrganizationalPair] = {}
        self.external_entities: Dict[str, PairedActor] = {}
        
        # Network statistics
        self.total_antimatter_absorbed = 0.0
        self.total_antimatter_processed = 0.0
    
    # ─────────────────────────────────────────────────────────────────
    # PAIR CREATION
    # ─────────────────────────────────────────────────────────────────
    
    def create_pair(self, name: str, mission: str, 
                   initial_capital: float = 0.0,
                   antimatter_capacity: float = 100.0) -> OrganizationalPair:
        """Create a new organizational pair."""
        pair_id = f"pair_{len(self.pairs)}_{name.lower().replace(' ', '_')}"
        
        pair = OrganizationalPair(
            id=pair_id,
            name=name,
            mission=mission,
            capital=initial_capital
        )
        
        # Set finance antimatter capacity
        pair.finance.antimatter_capacity = antimatter_capacity
        
        self.pairs[pair_id] = pair
        
        print(f"🏢 Created pair: {name}")
        print(f"   👑 Owner: {pair.owner.name} (Position 1)")
        print(f"   🏦 Finance: {pair.finance.name} (Position 0)")
        print(f"   Mission: {mission}")
        print(f"   Antimatter capacity: {antimatter_capacity}")
        
        return pair
    
    def create_external(self, name: str, role: ActorRole,
                       position: float = -0.5) -> PairedActor:
        """Create an external entity (bank, market)."""
        ext_id = f"ext_{len(self.external_entities)}_{name.lower().replace(' ', '_')}"
        
        external = PairedActor(
            id=ext_id,
            name=name,
            role=role,
            position=position,
            antimatter_capacity=float('inf')  # External can handle unlimited
        )
        
        self.external_entities[ext_id] = external
        
        print(f"🏛️  Created external: {name} ({role.value})")
        
        return external
    
    # ─────────────────────────────────────────────────────────────────
    # CONNECTIONS
    # ─────────────────────────────────────────────────────────────────
    
    def connect_pairs_as_neighbors(self, pair1_id: str, pair2_id: str):
        """Connect two pairs as neighbors (both levels)."""
        if pair1_id not in self.pairs or pair2_id not in self.pairs:
            return False
        
        pair1 = self.pairs[pair1_id]
        pair2 = self.pairs[pair2_id]
        
        # Connect at organization level
        pair1.neighbor_pair_ids.add(pair2_id)
        pair2.neighbor_pair_ids.add(pair1_id)
        
        # Connect owners (position 1 neighbors)
        pair1.owner.add_neighbor(pair2.owner.id)
        pair2.owner.add_neighbor(pair1.owner.id)
        
        # Connect finances (position 0 neighbors)
        pair1.finance.add_neighbor(pair2.finance.id)
        pair2.finance.add_neighbor(pair1.finance.id)
        
        print(f"🔗 Connected neighbors:")
        print(f"   Position 1: {pair1.owner.name} ↔ {pair2.owner.name}")
        print(f"   Position 0: {pair1.finance.name} ↔ {pair2.finance.name}")
        
        return True
    
    def connect_finance_to_external(self, pair_id: str, external_id: str,
                                    strength: float = 1.0):
        """Connect a pair's finance to an external entity."""
        if pair_id not in self.pairs or external_id not in self.external_entities:
            return False
        
        pair = self.pairs[pair_id]
        external = self.external_entities[external_id]
        
        pair.finance.connect_external(external_id, strength)
        
        print(f"🔗 Connected {pair.finance.name} → {external.name} (strength: {strength})")
        
        return True
    
    # ─────────────────────────────────────────────────────────────────
    # ANTIMATTER FLOW
    # ─────────────────────────────────────────────────────────────────
    
    def inject_antimatter(self, pair_id: str, amount: float,
                         antimatter_type: AntimatterType,
                         source: str) -> Dict:
        """
        Inject antimatter into the network.
        Finance handles it, owner is shielded.
        """
        if pair_id not in self.pairs:
            return {'error': 'pair not found'}
        
        pair = self.pairs[pair_id]
        result = pair.receive_antimatter(amount, antimatter_type, source)
        
        self.total_antimatter_absorbed += result['absorbed']
        
        print(f"\n⚡ Antimatter injection: {antimatter_type.value}")
        print(f"   Target: {pair.name}")
        print(f"   Amount: {amount:.2f}")
        print(f"   Absorbed by finance: {result['absorbed']:.2f}")
        if result['overflow'] > 0:
            print(f"   ⚠️  Overflow: {result['overflow']:.2f}")
        print(f"   Owner exposed: {'YES ⚠️' if result['owner_exposed'] else 'NO ✓'}")
        
        return result
    
    def spread_antimatter_to_neighbors(self, pair_id: str, amount: float,
                                       antimatter_type: AntimatterType) -> Dict:
        """
        Spread antimatter overflow to neighbor finances.
        This is mutual support between position 0 actors.
        """
        if pair_id not in self.pairs:
            return {'error': 'pair not found'}
        
        pair = self.pairs[pair_id]
        spread_results = []
        remaining = amount
        
        for neighbor_id in pair.neighbor_pair_ids:
            if remaining <= 0:
                break
            
            neighbor = self.pairs.get(neighbor_id)
            if neighbor and neighbor.finance.can_absorb_more:
                # Each neighbor takes a share
                share = min(remaining / len(pair.neighbor_pair_ids), 
                           neighbor.finance.antimatter_capacity - neighbor.finance.antimatter_absorbed)
                absorbed, _ = neighbor.finance.absorb_antimatter(share, antimatter_type)
                spread_results.append({
                    'neighbor': neighbor.name,
                    'absorbed': absorbed
                })
                remaining -= absorbed
        
        return {
            'spread_to': spread_results,
            'remaining': remaining
        }
    
    def external_absorbs_antimatter(self, external_id: str, amount: float,
                                    antimatter_type: AntimatterType):
        """
        External entity (bank, market) absorbs antimatter.
        This is the final sink for chaos.
        """
        if external_id not in self.external_entities:
            return
        
        external = self.external_entities[external_id]
        external.absorb_antimatter(amount, antimatter_type)
        
        print(f"🏛️  {external.name} absorbed {amount:.2f} {antimatter_type.value}")
    
    # ─────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────
    
    def visualize(self):
        """Visualize the paired network."""
        
        print("\n" + "═" * 70)
        print(f"PAIRED NETWORK: {self.name}")
        print("═" * 70)
        
        # Position 1 layer
        print("\n┌" + "─" * 68 + "┐")
        print("│  POSITION 1: OWNERS (Mission Layer)" + " " * 31 + "│")
        print("│" + "─" * 68 + "│")
        
        owners = [p.owner for p in self.pairs.values()]
        owner_line = "  ".join([f"👑 {o.name[:15]}" for o in owners])
        print(f"│  {owner_line:<66}│")
        
        # Show owner neighbors
        for pair in self.pairs.values():
            neighbors = [self.pairs[n].owner.name[:10] for n in pair.neighbor_pair_ids if n in self.pairs]
            if neighbors:
                print(f"│    └── neighbors: {', '.join(neighbors):<48}│")
        
        print("└" + "─" * 68 + "┘")
        print("         │" * len(self.pairs))
        print("         │" * len(self.pairs) + "  (paired)")
        print("         │" * len(self.pairs))
        
        # Position 0 layer
        print("┌" + "─" * 68 + "┐")
        print("│  POSITION 0: FINANCE (Shield Layer)" + " " * 31 + "│")
        print("│" + "─" * 68 + "│")
        
        finances = [p.finance for p in self.pairs.values()]
        finance_line = "  ".join([f"🏦 {f.name[:15]}" for f in finances])
        print(f"│  {finance_line:<66}│")
        
        # Show finance loads
        for pair in self.pairs.values():
            load = pair.finance.antimatter_load
            bar = "█" * int(load / 10) + "░" * (10 - int(load / 10))
            print(f"│    └── antimatter: [{bar}] {load:.1f}%{' ' * 30}│")
        
        print("└" + "─" * 68 + "┘")
        print("         │" * len(self.pairs))
        print("         ↓" * len(self.pairs))
        
        # External layer
        print("┌" + "─" * 68 + "┐")
        print("│  EXTERNAL: BANKS & MARKETS (Antimatter Sink)" + " " * 22 + "│")
        print("│" + "─" * 68 + "│")
        
        if self.external_entities:
            ext_line = "  ".join([f"🏛️ {e.name[:15]}" for e in self.external_entities.values()])
            print(f"│  {ext_line:<66}│")
        else:
            print("│  (no external entities)" + " " * 44 + "│")
        
        print("└" + "─" * 68 + "┘")
        print("         │")
        print("    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
        print("    ▓    ANTIMATTER (Market Chaos)    ▓")
        print("    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
    
    def get_network_status(self) -> Dict:
        """Get network-wide status."""
        total_capital = sum(p.capital for p in self.pairs.values())
        total_antimatter_load = sum(p.finance.antimatter_absorbed for p in self.pairs.values())
        total_capacity = sum(p.finance.antimatter_capacity for p in self.pairs.values())
        
        return {
            'pair_count': len(self.pairs),
            'external_count': len(self.external_entities),
            'total_capital': total_capital,
            'total_antimatter_load': total_antimatter_load,
            'total_capacity': total_capacity,
            'network_load_pct': total_antimatter_load / total_capacity * 100 if total_capacity > 0 else 0,
            'total_antimatter_absorbed': self.total_antimatter_absorbed,
            'total_antimatter_processed': self.total_antimatter_processed
        }
    
    def display_status(self):
        """Display network status."""
        status = self.get_network_status()
        
        print(f"\n{'═'*60}")
        print(f"📊 NETWORK STATUS: {self.name}")
        print(f"{'═'*60}")
        print(f"  Organizational pairs: {status['pair_count']}")
        print(f"  External entities: {status['external_count']}")
        print(f"  Total capital: ${status['total_capital']:,.2f}")
        print(f"")
        print(f"  ANTIMATTER MANAGEMENT:")
        print(f"    Current load: {status['total_antimatter_load']:.2f} / {status['total_capacity']:.2f}")
        print(f"    Network load: {status['network_load_pct']:.1f}%")
        print(f"    Total absorbed (all time): {status['total_antimatter_absorbed']:.2f}")
        print(f"    Total processed (all time): {status['total_antimatter_processed']:.2f}")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_paired_network():
    """Demonstrate the paired network architecture."""
    
    print("\n" + "═" * 70)
    print("PAIRED NETWORK DEMONSTRATION")
    print("Every company is a 1-0 pair (Owner + Finance)")
    print("═" * 70)
    
    # Create network
    network = PairedNetwork("Shovelcat Ecosystem")
    
    # Phase 1: Create the main pair (Shovelcat)
    print("\n" + "─" * 70)
    print("PHASE 1: Create Shovelcat (primary pair)")
    print("─" * 70)
    
    shovelcat = network.create_pair(
        name="Shovelcat",
        mission="Enable exploration through stable foundations",
        initial_capital=2000.0,
        antimatter_capacity=150.0
    )
    
    # Phase 2: Create neighbor pairs
    print("\n" + "─" * 70)
    print("PHASE 2: Create neighbor pairs")
    print("─" * 70)
    
    health_co = network.create_pair(
        name="Health Division",
        mission="Protein folding and health insights",
        initial_capital=500.0,
        antimatter_capacity=80.0
    )
    
    education_co = network.create_pair(
        name="Education Division",
        mission="Teaching and learning systems",
        initial_capital=500.0,
        antimatter_capacity=80.0
    )
    
    games_co = network.create_pair(
        name="Games Division",
        mission="Engaging exploration experiences",
        initial_capital=300.0,
        antimatter_capacity=60.0
    )
    
    # Phase 3: Connect neighbors
    print("\n" + "─" * 70)
    print("PHASE 3: Connect neighbors")
    print("─" * 70)
    
    network.connect_pairs_as_neighbors(shovelcat.id, health_co.id)
    network.connect_pairs_as_neighbors(shovelcat.id, education_co.id)
    network.connect_pairs_as_neighbors(shovelcat.id, games_co.id)
    network.connect_pairs_as_neighbors(health_co.id, education_co.id)
    
    # Phase 4: Create external entities
    print("\n" + "─" * 70)
    print("PHASE 4: Create external entities (banks, markets)")
    print("─" * 70)
    
    main_bank = network.create_external("TD Bank", ActorRole.BANK)
    market = network.create_external("Stock Market", ActorRole.MARKET)
    credit_union = network.create_external("Credit Union", ActorRole.BANK)
    
    # Connect finances to externals
    network.connect_finance_to_external(shovelcat.id, main_bank.id, 1.0)
    network.connect_finance_to_external(shovelcat.id, market.id, 0.8)
    network.connect_finance_to_external(health_co.id, credit_union.id, 0.5)
    network.connect_finance_to_external(education_co.id, credit_union.id, 0.5)
    
    # Phase 5: Test antimatter handling
    print("\n" + "─" * 70)
    print("PHASE 5: Test antimatter handling")
    print("─" * 70)
    
    # Inject various antimatter events
    network.inject_antimatter(
        shovelcat.id, 30.0,
        AntimatterType.VOLATILITY,
        "market_fluctuation"
    )
    
    network.inject_antimatter(
        shovelcat.id, 20.0,
        AntimatterType.TAX_COMPLEXITY,
        "quarterly_taxes"
    )
    
    network.inject_antimatter(
        health_co.id, 40.0,
        AntimatterType.CASH_CRISIS,
        "delayed_payment"
    )
    
    # Try to create something (owner should be shielded)
    print("\n" + "─" * 70)
    print("PHASE 6: Owners create (shielded from antimatter)")
    print("─" * 70)
    
    shovelcat.owner_create("New trading algorithm", 0.3)
    health_co.owner_create("Protein structure analysis", 0.4)
    education_co.owner_create("Learning pathway system", 0.3)
    
    # Overwhelm one finance to test exposure
    print("\n" + "─" * 70)
    print("PHASE 7: Test owner exposure when finance overwhelmed")
    print("─" * 70)
    
    network.inject_antimatter(
        games_co.id, 100.0,  # More than capacity!
        AntimatterType.MARKET_CRASH,
        "game_market_collapse"
    )
    
    # Games owner tries to create while exposed
    games_co.owner_create("New game concept", 0.2)
    
    # Phase 8: Visualize
    print("\n" + "─" * 70)
    print("PHASE 8: Visualize network")
    print("─" * 70)
    
    network.visualize()
    
    # Phase 9: Status
    print("\n" + "─" * 70)
    print("PHASE 9: Network status")
    print("─" * 70)
    
    network.display_status()
    
    # Show each pair
    for pair in network.pairs.values():
        pair.display()
    
    # Summary
    print("\n" + "─" * 70)
    print("SUMMARY: The Paired Architecture")
    print("─" * 70)
    
    print("""
    KEY INSIGHT: Every company is a 1-0 PAIR
    
    👑 OWNER (Position 1)          🏦 FINANCE (Position 0)
    ─────────────────────          ──────────────────────
    • Focuses on mission           • Handles antimatter
    • Creates value                • Shields owner
    • Makes decisions              • Manages cash flow
    • Connects to peers            • Connects to banks
    • NEVER touches antimatter     • ABSORBS all chaos
    
    THE SHIELD EFFECT:
    
        Market Crash ──→ Finance ──→ (absorbed)
                              │
                              ✗ (blocked)
                              │
                         Owner (protected)
                              │
                              ↓
                         Creation continues!
    
    NEIGHBOR SUPPORT:
    
        Position 1: Shovelcat Owner ←→ Health Owner ←→ Education Owner
                         (share ideas, collaborate on missions)
        
        Position 0: Shovelcat Finance ←→ Health Finance ←→ Education Finance
                         (share antimatter load, coordinate with banks)
    
    EXTERNAL CONNECTIONS (Finance → Banks/Markets):
    
        Shovelcat Finance ──→ TD Bank (strength 1.0)
                         └──→ Stock Market (strength 0.8)
        
        Health Finance ──→ Credit Union (strength 0.5)
        
        These are the channels for antimatter to flow OUT of the system.
    """)
    
    return network


if __name__ == "__main__":
    network = demonstrate_paired_network()
