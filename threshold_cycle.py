"""
THRESHOLD CYCLE ECONOMICS
=========================

Energy thresholds create natural economic flows:

    π (pi ≈ 3.14159): ENERGY CAP
        - Energy beyond π gets converted
        - Can become: art (up), product (down), or bank fee
        - Creates natural "tax" that funds infrastructure
    
    φ (phi ≈ 1.618): UPGRADE THRESHOLD  
        - At φ, can reach out to φ-aligned actors
        - Golden ratio split toward alignment
        - Recruitment/outreach point
    
    e (≈ 2.718): GROWTH POINT
        - Natural growth threshold
        - Exponential expansion possible

THE COMPLETE CYCLE:
    
    1. Population pushes ART out (top) → attracts new actors
    2. New actors filter DOWN through hierarchy
    3. Leader creates PRODUCT, pushes down to bank
    4. Leader PAYS bank with >π excess energy
    5. Bank pushes PRODUCT out (bottom)
    6. Product attracts actors who enter at TOP
    7. Cycle continues
    
    ART flows UP (fan-made, creative, unofficial)
    PRODUCT flows DOWN (official, sanctioned, from leader)
    ENERGY >π flows to BANK (interest/fees)
    ACTORS flow DOWN through hierarchy (filtering)
    MONEY/VALUE cycles (product → interest → infrastructure)

WHAT DOES LEADER PAY BANK?
    
    The >π excess energy!
    - Leaders can't store infinite energy
    - Energy beyond π MUST be released
    - Bank absorbs as payment for infrastructure
    - This is the INTEREST RATE
    - Creates sustainable zero-sum for bank
    - Leader gets filtering, bank gets energy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from datetime import datetime

# Mathematical constants as thresholds
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618 - upgrade/reach threshold
E = np.e                      # ≈ 2.718 - growth threshold  
PI = np.pi                    # ≈ 3.142 - energy cap

# Energy thresholds
THRESHOLD_PHI = PHI      # Can reach φ-aligned actors
THRESHOLD_E = E          # Growth threshold
THRESHOLD_PI = PI        # Energy cap - excess must be released


# =============================================================================
# FLOW TYPES
# =============================================================================

class FlowType(Enum):
    """Types of flow in the system."""
    ART_UP = "art_up"           # Fan content flowing up and out
    PRODUCT_DOWN = "product_down"  # Official product flowing down and out
    ACTOR_DOWN = "actor_down"   # New actors filtering down
    ENERGY_TO_BANK = "energy_to_bank"  # >π excess as payment
    MONEY_CYCLE = "money_cycle"  # Value cycling through system


class OutputType(Enum):
    """What excess energy becomes."""
    ART = "art"              # Creative output (flows up)
    PRODUCT = "product"      # Official output (flows down)
    BANK_FEE = "bank_fee"    # Interest/payment to bank
    MESSAGE = "message"      # Spread to other actors


# =============================================================================
# ENERGY MANAGER
# =============================================================================

@dataclass
class EnergyState:
    """Energy state of an actor."""
    current: float = 1.0
    cap: float = PI  # Default cap at π
    
    # Tracking
    total_generated: float = 0.0
    total_released_up: float = 0.0      # Art
    total_released_down: float = 0.0    # Product
    total_paid_to_bank: float = 0.0     # Fees
    
    @property
    def excess(self) -> float:
        """Energy above the cap."""
        return max(0, self.current - self.cap)
    
    @property
    def at_phi(self) -> bool:
        """Is energy at or above φ threshold?"""
        return self.current >= THRESHOLD_PHI
    
    @property
    def at_e(self) -> bool:
        """Is energy at or above e threshold?"""
        return self.current >= THRESHOLD_E
    
    @property
    def at_pi(self) -> bool:
        """Is energy at or above π cap?"""
        return self.current >= THRESHOLD_PI
    
    def add_energy(self, amount: float) -> float:
        """Add energy, return excess above cap."""
        self.current += amount
        self.total_generated += amount
        
        if self.current > self.cap:
            excess = self.current - self.cap
            self.current = self.cap
            return excess
        return 0.0
    
    def release_energy(self, amount: float, direction: str) -> float:
        """Release energy in a direction."""
        released = min(amount, self.current)
        self.current -= released
        
        if direction == "up":
            self.total_released_up += released
        elif direction == "down":
            self.total_released_down += released
        elif direction == "bank":
            self.total_paid_to_bank += released
        
        return released


# =============================================================================
# THRESHOLD ACTOR
# =============================================================================

@dataclass
class ThresholdActor:
    """
    An actor with energy thresholds and flow capabilities.
    """
    id: str
    name: str
    position: float  # 0=bank, 1=leader, 2=follower, 3=population
    
    # Energy state
    energy: EnergyState = field(default_factory=EnergyState)
    
    # Connections
    above_id: Optional[str] = None  # Who is above (toward population)
    below_id: Optional[str] = None  # Who is below (toward bank)
    bank_id: Optional[str] = None   # Connected bank
    
    # Output inventory
    art_inventory: List[Dict] = field(default_factory=list)
    product_inventory: List[Dict] = field(default_factory=list)
    
    # Flow history
    flow_history: List[Tuple[datetime, FlowType, float, str]] = field(default_factory=list)
    
    @property
    def can_reach_phi_aligned(self) -> bool:
        """Can this actor reach out to φ-aligned actors?"""
        return self.energy.at_phi
    
    @property
    def can_grow_exponentially(self) -> bool:
        """Is actor at growth threshold?"""
        return self.energy.at_e
    
    @property
    def must_release_excess(self) -> bool:
        """Is actor at π cap, must release?"""
        return self.energy.at_pi
    
    def generate_energy(self, amount: float) -> Tuple[float, Optional[OutputType]]:
        """
        Generate energy. Returns (excess, what_to_do_with_it).
        """
        excess = self.energy.add_energy(amount)
        
        if excess > 0:
            # Decide what to do with excess based on position
            if self.position >= 2:  # Population/Follower
                return excess, OutputType.ART  # Excess becomes art
            elif self.position == 1:  # Leader
                return excess, OutputType.PRODUCT  # Excess becomes product
            else:  # Bank
                return excess, OutputType.MESSAGE  # Excess spreads
        
        return 0, None
    
    def create_art(self, title: str, energy_cost: float) -> Optional[Dict]:
        """Create art (flows up and out)."""
        if self.energy.current < energy_cost:
            return None
        
        self.energy.release_energy(energy_cost, "up")
        
        art = {
            'id': f"art_{len(self.art_inventory)}_{self.id}",
            'title': title,
            'creator': self.id,
            'energy': energy_cost,
            'created_at': datetime.now()
        }
        self.art_inventory.append(art)
        
        self.flow_history.append((datetime.now(), FlowType.ART_UP, energy_cost, title))
        
        return art
    
    def create_product(self, name: str, energy_cost: float) -> Optional[Dict]:
        """Create product (flows down and out)."""
        if self.energy.current < energy_cost:
            return None
        
        self.energy.release_energy(energy_cost, "down")
        
        product = {
            'id': f"prod_{len(self.product_inventory)}_{self.id}",
            'name': name,
            'creator': self.id,
            'energy': energy_cost,
            'created_at': datetime.now()
        }
        self.product_inventory.append(product)
        
        self.flow_history.append((datetime.now(), FlowType.PRODUCT_DOWN, energy_cost, name))
        
        return product
    
    def pay_bank(self, amount: float) -> float:
        """Pay bank with energy (the >π excess fee)."""
        paid = self.energy.release_energy(amount, "bank")
        self.flow_history.append((datetime.now(), FlowType.ENERGY_TO_BANK, paid, "bank_fee"))
        return paid
    
    def display(self):
        """Display actor state."""
        print(f"\n{'─'*50}")
        print(f"Actor: {self.name} (pos={self.position})")
        print(f"{'─'*50}")
        print(f"  Energy: {self.energy.current:.3f} / {self.energy.cap:.3f}")
        print(f"  At φ ({PHI:.3f}): {self.can_reach_phi_aligned}")
        print(f"  At e ({E:.3f}): {self.can_grow_exponentially}")
        print(f"  At π ({PI:.3f}): {self.must_release_excess}")
        print(f"  Art created: {len(self.art_inventory)}")
        print(f"  Products created: {len(self.product_inventory)}")
        print(f"  Total paid to bank: {self.energy.total_paid_to_bank:.3f}")


# =============================================================================
# THE CYCLE ENGINE
# =============================================================================

class ThresholdCycleEngine:
    """
    Manages the complete threshold cycle.
    
    The cycle:
    1. Actors generate energy through activity
    2. Energy above π must be released
    3. Population releases as ART (up, attracts actors)
    4. Leaders release as PRODUCT (down, through bank)
    5. Leaders PAY bank with excess energy (the fee)
    6. Bank pushes product out, attracts actors to top
    7. New actors filter down through hierarchy
    """
    
    def __init__(self):
        self.actors: Dict[str, ThresholdActor] = {}
        self.bank: Optional[ThresholdActor] = None
        
        # Flow tracking
        self.art_released: List[Dict] = []
        self.products_released: List[Dict] = []
        self.actors_attracted: List[Dict] = []
        self.total_bank_revenue: float = 0.0
        
        # Cycle statistics
        self.cycles_completed: int = 0
    
    def create_bank(self, name: str) -> ThresholdActor:
        """Create the bank (position 0)."""
        bank = ThresholdActor(
            id=f"bank_{name.lower().replace(' ', '_')}",
            name=name,
            position=0.0
        )
        bank.energy.cap = PI * 2  # Banks can hold more
        
        self.actors[bank.id] = bank
        self.bank = bank
        
        print(f"🏦 Created BANK: {name}")
        print(f"   Energy cap: {bank.energy.cap:.3f} (2π)")
        return bank
    
    def create_leader(self, name: str, bank_id: str) -> ThresholdActor:
        """Create a leader (position 1)."""
        leader = ThresholdActor(
            id=f"leader_{name.lower().replace(' ', '_')}",
            name=name,
            position=1.0,
            bank_id=bank_id
        )
        leader.energy.cap = PI  # Leaders capped at π
        
        self.actors[leader.id] = leader
        
        print(f"👑 Created LEADER: {name}")
        print(f"   Energy cap: π ≈ {PI:.3f}")
        print(f"   Connected to bank: {bank_id}")
        return leader
    
    def create_follower(self, name: str, leader_id: str) -> ThresholdActor:
        """Create a follower (position 2)."""
        follower = ThresholdActor(
            id=f"follower_{name.lower().replace(' ', '_')}",
            name=name,
            position=2.0,
            below_id=leader_id
        )
        follower.energy.cap = PI  # Also capped at π
        
        # Link leader above
        if leader_id in self.actors:
            self.actors[leader_id].above_id = follower.id
        
        self.actors[follower.id] = follower
        
        print(f"🙋 Created FOLLOWER: {name}")
        print(f"   Energy cap: π ≈ {PI:.3f}")
        return follower
    
    def create_population(self, name: str, follower_id: str) -> ThresholdActor:
        """Create population actor (position 3)."""
        pop = ThresholdActor(
            id=f"pop_{name.lower().replace(' ', '_')}",
            name=name,
            position=3.0,
            below_id=follower_id
        )
        pop.energy.cap = PI  # All capped at π
        
        # Link follower below
        if follower_id in self.actors:
            self.actors[follower_id].above_id = pop.id
        
        self.actors[pop.id] = pop
        
        print(f"📢 Created POPULATION: {name}")
        print(f"   Energy cap: π ≈ {PI:.3f}")
        return pop
    
    # ─────────────────────────────────────────────────────────────────
    # THRESHOLD MECHANICS
    # ─────────────────────────────────────────────────────────────────
    
    def process_phi_threshold(self, actor_id: str) -> List[str]:
        """
        At φ threshold, actor can reach out to φ-aligned actors.
        Returns list of potential recruits.
        """
        if actor_id not in self.actors:
            return []
        
        actor = self.actors[actor_id]
        if not actor.can_reach_phi_aligned:
            return []
        
        # Simulate finding φ-aligned actors
        # In real system, would search for actors at φ-distance
        recruits = [
            f"phi_recruit_{i}_{actor_id}" 
            for i in range(int(actor.energy.current / PHI))
        ]
        
        print(f"✨ {actor.name} at φ threshold - can reach {len(recruits)} aligned actors")
        
        return recruits
    
    def process_e_threshold(self, actor_id: str) -> float:
        """
        At e threshold, exponential growth possible.
        Returns growth multiplier.
        """
        if actor_id not in self.actors:
            return 1.0
        
        actor = self.actors[actor_id]
        if not actor.can_grow_exponentially:
            return 1.0
        
        # Growth multiplier based on how far above e
        multiplier = actor.energy.current / E
        
        print(f"📈 {actor.name} at e threshold - growth multiplier: {multiplier:.2f}x")
        
        return multiplier
    
    def process_pi_threshold(self, actor_id: str) -> Tuple[float, OutputType]:
        """
        At π threshold, excess must be released.
        Returns (excess amount, output type).
        """
        if actor_id not in self.actors:
            return 0.0, OutputType.MESSAGE
        
        actor = self.actors[actor_id]
        excess = actor.energy.excess
        
        if excess <= 0:
            return 0.0, OutputType.MESSAGE
        
        # Determine output based on position
        if actor.position >= 2:  # Population/Follower
            output_type = OutputType.ART
        elif actor.position == 1:  # Leader
            output_type = OutputType.PRODUCT
        else:  # Bank
            output_type = OutputType.MESSAGE
        
        print(f"🔥 {actor.name} at π cap - excess {excess:.3f} → {output_type.value}")
        
        return excess, output_type
    
    # ─────────────────────────────────────────────────────────────────
    # THE COMPLETE CYCLE
    # ─────────────────────────────────────────────────────────────────
    
    def run_cycle(self, energy_input: float = 1.0) -> Dict:
        """
        Run one complete cycle of the economy.
        
        1. Generate energy at population level
        2. Process thresholds
        3. Release excess (art up, product down)
        4. Pay bank fees
        5. Attract new actors
        6. Filter through hierarchy
        """
        cycle_result = {
            'energy_input': energy_input,
            'art_created': [],
            'products_created': [],
            'bank_revenue': 0.0,
            'phi_recruits': [],
            'actors_filtered': []
        }
        
        print(f"\n{'═'*60}")
        print(f"CYCLE {self.cycles_completed + 1}")
        print(f"{'═'*60}")
        
        # Track excess per actor from generation
        excess_tracker: Dict[str, Tuple[float, OutputType]] = {}
        
        # STEP 1: Generate energy at each level
        print(f"\n📊 STEP 1: Energy generation (input={energy_input})")
        for actor in sorted(self.actors.values(), key=lambda a: -a.position):
            excess, output_type = actor.generate_energy(energy_input)
            
            if excess > 0:
                excess_tracker[actor.id] = (excess, output_type)
                print(f"   {actor.name}: excess {excess:.3f} → {output_type.value if output_type else 'none'}")
        
        # STEP 2: Process φ threshold (recruitment)
        print(f"\n✨ STEP 2: φ threshold (recruitment)")
        for actor in self.actors.values():
            if actor.can_reach_phi_aligned:
                recruits = self.process_phi_threshold(actor.id)
                cycle_result['phi_recruits'].extend(recruits)
        
        # STEP 3: Convert excess to output (using tracked excess)
        print(f"\n🔥 STEP 3: Convert >π excess to output")
        for actor_id, (excess, output_type) in excess_tracker.items():
            actor = self.actors[actor_id]
            
            if output_type == OutputType.ART:
                # Art flows UP (from population/followers)
                art = {
                    'id': f"art_{self.cycles_completed}_{actor_id}",
                    'title': f"FanArt_{self.cycles_completed}",
                    'creator': actor_id,
                    'energy': excess,
                    'created_at': datetime.now()
                }
                actor.art_inventory.append(art)
                actor.energy.total_released_up += excess
                self.art_released.append(art)
                cycle_result['art_created'].append(art)
                print(f"   🎨 {actor.name} created art ({excess:.3f} energy)")
            
            elif output_type == OutputType.PRODUCT:
                # Product flows DOWN (from leaders)
                product_energy = excess * 0.6
                bank_fee = excess * 0.4
                
                product = {
                    'id': f"prod_{self.cycles_completed}_{actor_id}",
                    'name': f"Product_{self.cycles_completed}",
                    'creator': actor_id,
                    'energy': product_energy,
                    'created_at': datetime.now()
                }
                actor.product_inventory.append(product)
                actor.energy.total_released_down += product_energy
                self.products_released.append(product)
                cycle_result['products_created'].append(product)
                print(f"   📦 {actor.name} created product ({product_energy:.3f} energy)")
                
                # Pay bank with remaining
                actor.energy.total_paid_to_bank += bank_fee
                if self.bank:
                    self.bank.energy.add_energy(bank_fee)
                self.total_bank_revenue += bank_fee
                cycle_result['bank_revenue'] += bank_fee
                print(f"   💰 {actor.name} paid bank: {bank_fee:.3f}")
        
        # STEP 4: Art attracts actors (top)
        print(f"\n📥 STEP 4: Art attracts actors (entering top)")
        for art in cycle_result['art_created']:
            attracted = {
                'id': f"attracted_by_art_{art['id']}",
                'source': 'art',
                'entry_point': 'top',
                'attracted_by': art['title']
            }
            self.actors_attracted.append(attracted)
            cycle_result['actors_filtered'].append(attracted)
            print(f"   🆕 New actor attracted by {art['title']}")
        
        # STEP 5: Product attracts actors (cycles to top)
        print(f"\n📥 STEP 5: Product attracts actors (cycling to top)")
        for product in cycle_result['products_created']:
            attracted = {
                'id': f"attracted_by_product_{product['id']}",
                'source': 'product',
                'entry_point': 'top',  # Still enter at top, filtered down
                'attracted_by': product['name']
            }
            self.actors_attracted.append(attracted)
            cycle_result['actors_filtered'].append(attracted)
            print(f"   🆕 New actor attracted by {product['name']}")
        
        self.cycles_completed += 1
        
        return cycle_result
    
    # ─────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────
    
    def visualize_cycle(self):
        """Visualize the threshold cycle."""
        
        print(f"\n{'═'*70}")
        print("THRESHOLD CYCLE VISUALIZATION")
        print(f"{'═'*70}")
        
        print(f"""
                         FOREIGN SYSTEMS
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            │    ╭───── ART (>π excess) ───────╮│
            │    │      (fan content)          ↓│
            │    │                              │
            │    │   ┌─────────────────────┐   │
            │    │   │  POPULATION (pos=3) │   │   Energy cap: π
            │    │   │  Energy: {self._get_energy('pop'):.2f}           │   │
            │    │   │  At φ: {self._at_phi('pop')}  At π: {self._at_pi('pop')} │   │
            │    │   └──────────┬──────────┘   │
            │    │              │               │
            │    │   ┌──────────┴──────────┐   │
            │    │   │  FOLLOWER (pos=2)   │   │   At φ: can reach aligned
            │    │   │  Energy: {self._get_energy('follower'):.2f}           │   │
            │    │   │  At φ: {self._at_phi('follower')}  At π: {self._at_pi('follower')} │   │
            │    │   └──────────┬──────────┘   │
            │    │              │               │
            │    │   ═══════════╪═══════════   │   THE WALL (pos=1)
            │    │   ║  LEADER (pos=1)     ║   │
            │    │   ║  Energy: {self._get_energy('leader'):.2f}          ║   │
            │    │   ║  At φ: {self._at_phi('leader')}  At π: {self._at_pi('leader')} ║   │
            │    │   ║     │               ║   │
            │    │   ║     │ ─→ pays bank ─╫───┼──→ >π excess as FEE
            │    │   ║     ↓  (40% excess) ║   │
            │    │   ╚═════════════════════╝   │
            │    │              │               │
            │    │   ┌──────────┴──────────┐   │
            │    │   │    BANK (pos=0)     │   │   Zero-sum infrastructure
            │    │   │  Energy: {self._get_energy('bank'):.2f}           │   │
            │    │   │  Revenue: {self.total_bank_revenue:.2f}         │   │
            │    │   └──────────┬──────────┘   │
            │    │              │               │
            │    ↓              ↓               │
            │    ╰───── PRODUCT (official) ────╯│
            │         (attracts to top)         │
            └─────────────────┼─────────────────┘
                              │
                         DOMESTIC CHAOS
        """)
        
        print(f"""
    THRESHOLD MEANINGS:
    
    φ ≈ {PHI:.3f} : UPGRADE - Can reach out to φ-aligned actors
                   (golden ratio split toward alignment)
    
    e ≈ {E:.3f} : GROWTH - Exponential expansion possible
                  (natural growth threshold)
    
    π ≈ {PI:.3f} : CAP - Energy must be released
                  Position 2-3: becomes ART (flows up)
                  Position 1: becomes PRODUCT (flows down) + BANK FEE
                  The fee IS the interest rate!
    
    THE CYCLE:
    
    Art ────→ attracts actors ───→ enter at TOP
      ↑                               │
      │                               ↓
    excess                      filter through hierarchy
      ↑                               │
      │                               ↓
    energy ←──── work/activity ←──── actors
      ↑                               │
      │                               ↓
    Product ←── leader creates ←── loyalty
      │
      ↓
    Bank pushes out ───→ attracts actors ───→ enter at TOP
        """)
    
    def _get_energy(self, prefix: str) -> float:
        """Helper to get energy of actor by prefix."""
        for actor_id, actor in self.actors.items():
            if actor_id.startswith(prefix):
                return actor.energy.current
        return 0.0
    
    def _at_phi(self, prefix: str) -> str:
        """Helper to check if at φ."""
        for actor_id, actor in self.actors.items():
            if actor_id.startswith(prefix):
                return "✓" if actor.can_reach_phi_aligned else "✗"
        return "?"
    
    def _at_pi(self, prefix: str) -> str:
        """Helper to check if at π."""
        for actor_id, actor in self.actors.items():
            if actor_id.startswith(prefix):
                return "✓" if actor.must_release_excess else "✗"
        return "?"
    
    def get_statistics(self) -> Dict:
        """Get cycle statistics."""
        return {
            'cycles_completed': self.cycles_completed,
            'total_art_released': len(self.art_released),
            'total_products_released': len(self.products_released),
            'total_actors_attracted': len(self.actors_attracted),
            'total_bank_revenue': self.total_bank_revenue,
            'actors_at_phi': len([a for a in self.actors.values() if a.can_reach_phi_aligned]),
            'actors_at_pi': len([a for a in self.actors.values() if a.must_release_excess])
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_threshold_cycle():
    """Demonstrate the complete threshold cycle."""
    
    print("\n" + "═" * 70)
    print("THRESHOLD CYCLE DEMONSTRATION")
    print("π = energy cap, φ = upgrade point, e = growth point")
    print("═" * 70)
    
    # Create engine
    engine = ThresholdCycleEngine()
    
    # Create hierarchy
    print("\n" + "─" * 70)
    print("PHASE 1: Create hierarchy")
    print("─" * 70)
    
    bank = engine.create_bank("Central Bank")
    leader = engine.create_leader("Shovelcat", bank.id)
    follower = engine.create_follower("True Believer", leader.id)
    pop = engine.create_population("Missionary", follower.id)
    
    # Run multiple cycles
    print("\n" + "─" * 70)
    print("PHASE 2: Run cycles (watch thresholds)")
    print("─" * 70)
    
    for i in range(5):
        result = engine.run_cycle(energy_input=1.0)
    
    # Visualize
    print("\n" + "─" * 70)
    print("PHASE 3: Visualize final state")
    print("─" * 70)
    
    engine.visualize_cycle()
    
    # Statistics
    print("\n" + "─" * 70)
    print("PHASE 4: Statistics")
    print("─" * 70)
    
    stats = engine.get_statistics()
    print(f"""
    Cycles completed: {stats['cycles_completed']}
    
    OUTPUT:
    ├── Art released: {stats['total_art_released']}
    └── Products released: {stats['total_products_released']}
    
    ATTRACTION:
    └── Actors attracted: {stats['total_actors_attracted']}
    
    BANK:
    └── Total revenue (>π fees): {stats['total_bank_revenue']:.3f}
    
    THRESHOLDS:
    ├── Actors at φ (can recruit): {stats['actors_at_phi']}
    └── Actors at π (must release): {stats['actors_at_pi']}
    """)
    
    # Show individual actors
    print("\n" + "─" * 70)
    print("PHASE 5: Individual actor states")
    print("─" * 70)
    
    for actor in sorted(engine.actors.values(), key=lambda a: -a.position):
        actor.display()
    
    print("\n" + "─" * 70)
    print("KEY INSIGHT: What does Leader pay Bank?")
    print("─" * 70)
    print(f"""
    The >π excess energy!
    
    Leader's energy cap: π ≈ {PI:.3f}
    When energy exceeds π, leader MUST release
    
    60% → becomes PRODUCT (flows down and out)
    40% → becomes BANK FEE (the interest rate!)
    
    Total bank revenue this run: {stats['total_bank_revenue']:.3f}
    
    This creates sustainable zero-sum for bank:
    - Bank provides infrastructure
    - Leader pays with overflow energy
    - Energy can't be stored infinitely
    - Natural "tax" from mathematical necessity
    """)
    
    return engine


if __name__ == "__main__":
    engine = demonstrate_threshold_cycle()
