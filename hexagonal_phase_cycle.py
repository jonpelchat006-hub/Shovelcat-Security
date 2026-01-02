"""
HEXAGONAL PHASE CYCLE WITH Z-AXIS LAYERING
==========================================

The hexagon has 4 quadrants representing phases:
    
    TOP: e (expansion) - possibilities increase
    BOTTOM: ln (collapse) - possibilities narrow
    LEFT: Quantum - internal, potential, thinking
    RIGHT: Classical - external, actual, doing

THE FOUR PHASES:

    Q-EXP (Quantum Expansion): Dreaming, imagining possibilities
        "What could I become? What do I want?"
        
    C-EXP (Classical Expansion): Browsing, seeing options
        "What's available? What can I choose from?"
        
    C-COL (Classical Collapse): Choosing, committing
        "I'll do this dungeon. I'll buy this item."
        
    Q-COL (Quantum Collapse): Deciding, narrowing needs
        "I need better gear. I want to be that guy."

THE GAMER CYCLE:

    1. Log on → C-EXP (see menu, options, world)
    2. Choose activity → C-COL (commit to dungeon)
    3. Play → dip into work/payment zone
    4. Get rewards → pulled back to start
    5. Repeat 2-4 multiple times
    6. Log off → Q-COL (realize what you need)
    7. Think/rest → Q-EXP (dream of possibilities)
    8. Log on → back to step 1

Z-AXIS LAYERING:

    Each actor level is on z-axis
    Everyone can reach ±1 from their position
    Overlap with target determines cost
    
    Position 3.0 (new user) → loops to 0.0 → close to 0.25 (gaming)
    Position 2.0 (creator) → close to 2.25 → overlaps gaming differently
    
    Less z-distance = more rotation (activity within domain)
    More z-distance = more translation (moving between domains)

MASS AND GRAVITY:

    As users engage more, they gain "mass"
    More mass = pulled closer to center of their layer
    "That guy" everyone wants to be = high mass, central position
    New users = low mass, outer orbit
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
import cmath

# Constants
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
E = np.e


# =============================================================================
# PHASE QUADRANTS
# =============================================================================

class Phase(Enum):
    """The four phases of the hexagonal cycle."""
    Q_EXP = "quantum_expansion"    # Dreaming, possibilities
    C_EXP = "classical_expansion"  # Browsing, options
    C_COL = "classical_collapse"   # Choosing, committing
    Q_COL = "quantum_collapse"     # Deciding, narrowing


class Domain(Enum):
    """Quantum vs Classical domain."""
    QUANTUM = "quantum"      # Internal, potential, thinking
    CLASSICAL = "classical"  # External, actual, doing


class Direction(Enum):
    """Expansion vs Collapse."""
    EXPANSION = "expansion"  # e - possibilities increase
    COLLAPSE = "collapse"    # ln - possibilities narrow


@dataclass
class PhaseState:
    """Current phase state of an actor."""
    phase: Phase
    domain: Domain
    direction: Direction
    
    # Position in phase (0-1, how far through this phase)
    progress: float = 0.0
    
    # Energy in this phase
    energy: float = 1.0
    
    @property
    def quadrant_angle(self) -> float:
        """Angle in the hexagonal cycle (radians)."""
        phase_angles = {
            Phase.Q_EXP: PI / 2,      # Top-left
            Phase.C_EXP: 0,            # Right
            Phase.C_COL: -PI / 2,      # Bottom-right
            Phase.Q_COL: PI,           # Left
        }
        base = phase_angles[self.phase]
        # Add progress within phase (quarter turn per phase)
        return base + self.progress * (PI / 2)


def get_phase(domain: Domain, direction: Direction) -> Phase:
    """Get phase from domain and direction."""
    if domain == Domain.QUANTUM and direction == Direction.EXPANSION:
        return Phase.Q_EXP
    elif domain == Domain.CLASSICAL and direction == Direction.EXPANSION:
        return Phase.C_EXP
    elif domain == Domain.CLASSICAL and direction == Direction.COLLAPSE:
        return Phase.C_COL
    else:  # Quantum collapse
        return Phase.Q_COL


# =============================================================================
# Z-AXIS LAYERING
# =============================================================================

@dataclass
class ZPosition:
    """
    Position on the z-axis (actor layer).
    
    Everyone can reach ±1 from their base position.
    Positions loop after π (3.14... → 0.14...)
    """
    base: float           # Base z-position (0=finance, 1=owner, 2=creator, etc.)
    current: float = 0.0  # Current z-position (can vary ±1)
    
    def __post_init__(self):
        self.current = self.base
    
    @property
    def min_reach(self) -> float:
        """Minimum reachable z-position."""
        return self.base - 1.0
    
    @property
    def max_reach(self) -> float:
        """Maximum reachable z-position."""
        return self.base + 1.0
    
    def can_reach(self, target: float) -> bool:
        """Can this actor reach the target z-position?"""
        # Handle looping (positions > π loop back)
        target_normalized = target % PI if target > PI else target
        base_normalized = self.base % PI if self.base > PI else self.base
        
        # Check direct distance
        direct_dist = abs(target_normalized - base_normalized)
        
        # Check wrapped distance (going around)
        wrapped_dist = PI - direct_dist
        
        min_dist = min(direct_dist, wrapped_dist)
        return min_dist <= 1.0
    
    def distance_to(self, target: float) -> float:
        """Calculate distance to target z-position."""
        # Handle looping
        target_normalized = target % PI if target > PI else target
        base_normalized = self.base % PI if self.base > PI else self.base
        
        direct_dist = abs(target_normalized - base_normalized)
        wrapped_dist = PI - direct_dist
        
        return min(direct_dist, wrapped_dist)
    
    def cost_to_reach(self, target: float) -> float:
        """
        Cost to interact with target z-position.
        Less distance = less cost = more rotation available
        """
        dist = self.distance_to(target)
        if dist > 1.0:
            return float('inf')  # Can't reach
        
        # Cost increases with distance
        # At distance 0: cost = 0 (all energy goes to rotation)
        # At distance 1: cost = 1 (all energy goes to translation)
        return dist
    
    def rotation_available(self, target: float) -> float:
        """
        How much rotation (activity) is available after translation cost.
        """
        cost = self.cost_to_reach(target)
        if cost == float('inf'):
            return 0.0
        return 1.0 - cost


# =============================================================================
# ACTOR WITH PHASE AND Z-POSITION
# =============================================================================

@dataclass
class CycleActor:
    """
    An actor that moves through the hexagonal phase cycle.
    """
    id: str
    name: str
    
    # Z-axis position (layer)
    z_position: ZPosition
    
    # Current phase state
    phase_state: PhaseState
    
    # Mass (engagement/importance)
    mass: float = 1.0
    
    # Orbit radius (distance from center of layer)
    orbit_radius: float = 1.0
    
    # Resources
    energy: float = 1.0
    currency: float = 0.0
    
    # History
    phase_history: List[Tuple[datetime, Phase]] = field(default_factory=list)
    
    @property
    def gravity_pull(self) -> float:
        """How strongly pulled toward center (based on mass)."""
        # More mass = stronger pull toward center
        return self.mass / (self.orbit_radius ** 2) if self.orbit_radius > 0 else float('inf')
    
    @property
    def is_central(self) -> bool:
        """Is this actor in the central/important position?"""
        return self.orbit_radius < 0.5
    
    def gain_mass(self, amount: float):
        """Gain mass (become more important/engaged)."""
        self.mass += amount
        # Mass pulls toward center
        pull = amount * 0.1
        self.orbit_radius = max(0.1, self.orbit_radius - pull)
    
    def lose_mass(self, amount: float):
        """Lose mass (become less engaged)."""
        self.mass = max(0.1, self.mass - amount)
        # Less mass = drift outward
        drift = amount * 0.1
        self.orbit_radius = min(2.0, self.orbit_radius + drift)
    
    def transition_phase(self, new_phase: Phase):
        """Transition to a new phase."""
        self.phase_history.append((datetime.now(), self.phase_state.phase))
        
        # Determine new domain and direction
        if new_phase in [Phase.Q_EXP, Phase.Q_COL]:
            domain = Domain.QUANTUM
        else:
            domain = Domain.CLASSICAL
        
        if new_phase in [Phase.Q_EXP, Phase.C_EXP]:
            direction = Direction.EXPANSION
        else:
            direction = Direction.COLLAPSE
        
        self.phase_state = PhaseState(
            phase=new_phase,
            domain=domain,
            direction=direction,
            progress=0.0
        )
    
    def advance_phase(self, amount: float = 0.1):
        """Advance progress within current phase."""
        self.phase_state.progress += amount
        
        if self.phase_state.progress >= 1.0:
            # Move to next phase
            self.phase_state.progress = 0.0
            next_phases = {
                Phase.C_EXP: Phase.C_COL,
                Phase.C_COL: Phase.Q_COL,
                Phase.Q_COL: Phase.Q_EXP,
                Phase.Q_EXP: Phase.C_EXP,
            }
            self.transition_phase(next_phases[self.phase_state.phase])


# =============================================================================
# THE GAMER CYCLE
# =============================================================================

@dataclass
class GamerCycle:
    """
    A specific implementation of the phase cycle for gamers.
    
    The cycle:
    1. Log on → C-EXP (see options)
    2. Choose → C-COL (commit)
    3. Play/work → dip negative, earn rewards
    4. Get pulled back to step 2 (repeat)
    5. Log off → Q-COL (realize needs)
    6. Think/rest → Q-EXP (dream)
    7. Log on → back to step 1
    """
    
    actor: CycleActor
    
    # Session state
    is_logged_in: bool = False
    session_loops: int = 0  # How many times repeated step 2-4
    
    # Accumulated during session
    session_rewards: float = 0.0
    session_work: float = 0.0
    
    def log_on(self) -> str:
        """Log on - enter C-EXP phase."""
        self.is_logged_in = True
        self.session_loops = 0
        self.session_rewards = 0.0
        self.session_work = 0.0
        
        self.actor.transition_phase(Phase.C_EXP)
        
        return f"{self.actor.name} logs on → C-EXP (seeing options)"
    
    def browse_options(self, options: List[str]) -> str:
        """Browse available options in C-EXP."""
        self.actor.advance_phase(0.5)
        return f"{self.actor.name} browses: {', '.join(options)}"
    
    def choose_activity(self, activity: str) -> str:
        """Choose an activity - transition to C-COL."""
        self.actor.transition_phase(Phase.C_COL)
        return f"{self.actor.name} chooses: {activity} → C-COL (committed)"
    
    def perform_work(self, work_amount: float, payment: float = 0.0) -> str:
        """
        Perform work (play, grind, etc.)
        This "dips into negative" - costs energy but earns rewards.
        """
        # Work costs energy
        self.actor.energy -= work_amount * 0.5
        self.session_work += work_amount
        
        # Payment if applicable (watching ads, subscription, etc.)
        if payment > 0:
            self.actor.currency -= payment
        
        # Advance through C-COL phase
        self.actor.advance_phase(0.3)
        
        return f"{self.actor.name} works: -{work_amount:.1f} energy, -{payment:.1f} currency"
    
    def receive_rewards(self, rewards: float) -> str:
        """
        Receive rewards - this pulls back toward start of loop.
        """
        self.actor.energy += rewards * 0.3
        self.actor.currency += rewards * 0.5
        self.session_rewards += rewards
        
        # Rewards give mass (engagement)
        self.actor.gain_mass(rewards * 0.1)
        
        # Pull back toward C-EXP to loop
        self.session_loops += 1
        self.actor.transition_phase(Phase.C_EXP)
        
        return f"{self.actor.name} rewarded: +{rewards:.1f} → pulled back to C-EXP (loop {self.session_loops})"
    
    def log_off(self) -> str:
        """Log off - transition to Q-COL."""
        self.is_logged_in = False
        self.actor.transition_phase(Phase.Q_COL)
        
        summary = f"""
{self.actor.name} logs off → Q-COL (narrowing needs)
  Session summary:
  - Loops completed: {self.session_loops}
  - Total work: {self.session_work:.1f}
  - Total rewards: {self.session_rewards:.1f}
  - New mass: {self.actor.mass:.2f}
  - Orbit radius: {self.actor.orbit_radius:.2f}
"""
        return summary
    
    def think_and_plan(self, needs: List[str]) -> str:
        """Think about needs - still in Q-COL."""
        self.actor.advance_phase(0.5)
        return f"{self.actor.name} realizes needs: {', '.join(needs)}"
    
    def dream_possibilities(self, dreams: List[str]) -> str:
        """Dream of possibilities - transition to Q-EXP."""
        self.actor.transition_phase(Phase.Q_EXP)
        return f"{self.actor.name} dreams: {', '.join(dreams)} → Q-EXP"
    
    def rest(self, recovery: float) -> str:
        """Rest and recover energy."""
        self.actor.energy += recovery
        self.actor.advance_phase(0.3)
        return f"{self.actor.name} rests: +{recovery:.1f} energy"


# =============================================================================
# DOMAIN INTERACTION (Z-AXIS OVERLAP)
# =============================================================================

class DomainInteraction:
    """
    Manages interactions between actors on different z-layers.
    
    Key insight: domains close on z-axis can convert more to rotation.
    """
    
    @staticmethod
    def calculate_interaction_cost(actor: CycleActor, target_domain: float) -> Dict:
        """
        Calculate the cost for an actor to interact with a domain.
        
        Returns breakdown of:
        - translation_cost: energy spent moving on z-axis
        - rotation_available: energy available for activity
        - path: how they reach it (direct, through bank, etc.)
        """
        z = actor.z_position
        
        result = {
            'actor_position': z.base,
            'target_domain': target_domain,
            'can_reach': z.can_reach(target_domain),
            'distance': z.distance_to(target_domain),
            'translation_cost': z.cost_to_reach(target_domain),
            'rotation_available': z.rotation_available(target_domain),
            'path': 'direct'
        }
        
        # Determine path
        if not result['can_reach']:
            # Must go through intermediate
            if target_domain < 1.0:  # Business domain
                result['path'] = 'through_bank'
                # Cost includes bank fee
                result['translation_cost'] += 0.2  # Bank fee
                result['rotation_available'] = max(0, 1.0 - result['translation_cost'])
            else:
                result['path'] = 'through_staff'
                result['translation_cost'] += 0.1  # Staff overhead
                result['rotation_available'] = max(0, 1.0 - result['translation_cost'])
        
        return result
    
    @staticmethod
    def calculate_bonus_path(actor: CycleActor, 
                            intermediary_position: float,
                            target_domain: float) -> Dict:
        """
        Calculate cost when going through an intermediary (like a content creator).
        
        Example: Subscriber at 2.5 → Creator at 2.0 → Gaming at 0.25
        """
        z = actor.z_position
        
        # Step 1: Actor to intermediary
        step1_dist = z.distance_to(intermediary_position)
        
        # Step 2: Intermediary to target (intermediary's reach)
        intermediary_z = ZPosition(base=intermediary_position)
        step2_dist = intermediary_z.distance_to(target_domain)
        
        # Total cost is sum of steps, but intermediary absorbs some
        # (This is the "bonus" - creator has relationship with domain)
        intermediary_absorption = 0.3  # Creator absorbs 30% of cost
        
        total_dist = step1_dist + step2_dist * (1 - intermediary_absorption)
        
        return {
            'actor_position': z.base,
            'intermediary_position': intermediary_position,
            'target_domain': target_domain,
            'step1_distance': step1_dist,
            'step2_distance': step2_dist,
            'intermediary_absorption': intermediary_absorption,
            'total_distance': total_dist,
            'translation_cost': min(total_dist, 1.0),
            'rotation_available': max(0, 1.0 - total_dist),
            'path': f'through_intermediary@{intermediary_position}'
        }


# =============================================================================
# THE HEXAGONAL NETWORK
# =============================================================================

class HexagonalPhaseNetwork:
    """
    The complete hexagonal network with phase cycles and z-axis layering.
    """
    
    def __init__(self, name: str = "Shovelcat Network"):
        self.name = name
        
        # Actors by z-layer
        self.actors: Dict[str, CycleActor] = {}
        
        # Domain definitions
        self.domains = {
            'finance': 0.0,
            'gaming': 0.25,
            'products': 0.4,
            'research': 0.6,
            'education': 0.9,
            'health': 0.9,
            'owner': 1.0,
            'staff': 1.3,
            'creators': 2.0,
            'power_users': 2.5,
            'new_users': PI,  # Loops to ~0.14
        }
        
        # Interaction calculator
        self.interaction = DomainInteraction()
    
    def create_actor(self, name: str, z_position: float, 
                    initial_phase: Phase = Phase.C_EXP) -> CycleActor:
        """Create an actor at a z-position."""
        actor = CycleActor(
            id=f"actor_{len(self.actors)}_{name.lower().replace(' ', '_')}",
            name=name,
            z_position=ZPosition(base=z_position),
            phase_state=PhaseState(
                phase=initial_phase,
                domain=Domain.CLASSICAL if initial_phase in [Phase.C_EXP, Phase.C_COL] else Domain.QUANTUM,
                direction=Direction.EXPANSION if initial_phase in [Phase.Q_EXP, Phase.C_EXP] else Direction.COLLAPSE
            )
        )
        self.actors[actor.id] = actor
        return actor
    
    def create_gamer(self, name: str) -> Tuple[CycleActor, GamerCycle]:
        """Create a gamer with their cycle."""
        actor = self.create_actor(name, PI)  # New users start at π
        cycle = GamerCycle(actor=actor)
        return actor, cycle
    
    def visualize_hexagon(self):
        """Visualize the hexagonal phase cycle."""
        
        print("\n" + "═" * 70)
        print("HEXAGONAL PHASE CYCLE")
        print("═" * 70)
        
        print("""
                                    e (EXPANSION)
                                         ▲
                                         │
                           ┌─────────────┼─────────────┐
                           │    Q-EXP    │    C-EXP    │
                           │             │             │
                           │   Dream     │   Browse    │
                           │   "What     │   "What's   │
                           │    could    │    avail-   │
                           │    I be?"   │    able?"   │
                           │             │             │
        QUANTUM ◀──────────┼─────────────┼─────────────┼──────────▶ CLASSICAL
                           │             │             │
                           │   Q-COL     │    C-COL    │
                           │             │             │
                           │   Decide    │   Choose    │
                           │   "What     │   "I'll do  │
                           │    do I     │    this     │
                           │    need?"   │    one."    │
                           │             │             │
                           └─────────────┼─────────────┘
                                         │
                                         ▼
                                    ln (COLLAPSE)
        
        THE CYCLE FLOW:
        
            ┌─→ C-EXP ─→ C-COL ─┐
            │   (browse)  (choose)  │
            │                       ↓
            │               [WORK/PLAY]
            │                       │
            │                    rewards
            │                       │
            └───────────────────────┘  (loop within session)
            
            Eventually:
            
            C-COL ─→ Q-COL ─→ Q-EXP ─→ C-EXP
           (log off) (reflect) (dream) (log on)
        """)
    
    def visualize_z_layers(self):
        """Visualize the z-axis layers."""
        
        print("\n" + "═" * 70)
        print("Z-AXIS LAYERS (Domain Positions)")
        print("═" * 70)
        
        print(f"""
        Z-Position │ Domain          │ Reach (±1)     │ Loops to
        ═══════════╪═════════════════╪════════════════╪══════════
        π = 3.14   │ New Users       │ 2.14 - π       │ 0.14
        2.50       │ Power Users     │ 1.50 - 3.50    │ 
        2.00       │ Creators        │ 1.00 - 3.00    │ 
        1.30       │ Staff           │ 0.30 - 2.30    │ 
        1.00       │ Owner           │ 0.00 - 2.00    │ 
        0.90       │ Education/Health│ -0.10 - 1.90   │ 
        0.60       │ Research        │ -0.40 - 1.60   │ 
        0.40       │ Products        │ -0.60 - 1.40   │ 
        0.25       │ Gaming          │ -0.75 - 1.25   │ 
        0.00       │ Finance         │ -1.00 - 1.00   │ 
        
        KEY INSIGHT: Overlap determines cost!
        
        Example 1: New User (3.14) → Gaming (0.25)
            - 3.14 loops to 0.14
            - Distance: |0.14 - 0.25| = 0.11
            - Very close! Low cost, high rotation
            - BUT must go through bank (fee)
        
        Example 2: Creator (2.0) → Gaming (0.25)
            - Direct distance: |2.0 - 0.25| = 1.75 (too far!)
            - Via loop: 2.0 → can't reach
            - Must use subscription chain
        
        Example 3: Creator (2.0) → Staff (1.3)
            - Distance: |2.0 - 1.3| = 0.7
            - Within ±1 reach!
            - Low cost, can convert to rotation
        """)
    
    def demonstrate_interaction(self, actor_pos: float, target_domain: str):
        """Demonstrate an interaction calculation."""
        
        actor = CycleActor(
            id="demo",
            name="Demo Actor",
            z_position=ZPosition(base=actor_pos),
            phase_state=PhaseState(Phase.C_EXP, Domain.CLASSICAL, Direction.EXPANSION)
        )
        
        target_pos = self.domains.get(target_domain, 0.5)
        
        result = self.interaction.calculate_interaction_cost(actor, target_pos)
        
        print(f"\n  Actor at {actor_pos:.2f} → {target_domain} ({target_pos:.2f}):")
        print(f"    Can reach: {'✓' if result['can_reach'] else '✗'}")
        print(f"    Distance: {result['distance']:.2f}")
        print(f"    Path: {result['path']}")
        print(f"    Translation cost: {result['translation_cost']:.2f}")
        print(f"    Rotation available: {result['rotation_available']:.2f}")
    
    def demonstrate_bonus_path(self, actor_pos: float, 
                               intermediary_pos: float,
                               target_domain: str):
        """Demonstrate a bonus path through intermediary."""
        
        actor = CycleActor(
            id="demo",
            name="Demo Actor",
            z_position=ZPosition(base=actor_pos),
            phase_state=PhaseState(Phase.C_EXP, Domain.CLASSICAL, Direction.EXPANSION)
        )
        
        target_pos = self.domains.get(target_domain, 0.5)
        
        result = self.interaction.calculate_bonus_path(actor, intermediary_pos, target_pos)
        
        print(f"\n  Actor at {actor_pos:.2f} → Intermediary at {intermediary_pos:.2f} → {target_domain} ({target_pos:.2f}):")
        print(f"    Step 1 (to intermediary): {result['step1_distance']:.2f}")
        print(f"    Step 2 (to target): {result['step2_distance']:.2f}")
        print(f"    Intermediary absorbs: {result['intermediary_absorption']*100:.0f}%")
        print(f"    Total distance: {result['total_distance']:.2f}")
        print(f"    Translation cost: {result['translation_cost']:.2f}")
        print(f"    Rotation available: {result['rotation_available']:.2f}")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_hexagonal_cycle():
    """Demonstrate the complete hexagonal phase cycle."""
    
    print("\n" + "═" * 70)
    print("HEXAGONAL PHASE CYCLE DEMONSTRATION")
    print("═" * 70)
    
    # Create network
    network = HexagonalPhaseNetwork()
    
    # Show the hexagon
    network.visualize_hexagon()
    
    # Show z-layers
    network.visualize_z_layers()
    
    # Create a gamer and run through their cycle
    print("\n" + "─" * 70)
    print("GAMER CYCLE DEMONSTRATION")
    print("─" * 70)
    
    actor, gamer = network.create_gamer("Player_One")
    
    print(f"\n  Initial state:")
    print(f"    Position: {actor.z_position.base:.2f} (new user)")
    print(f"    Phase: {actor.phase_state.phase.value}")
    print(f"    Mass: {actor.mass:.2f}")
    print(f"    Orbit: {actor.orbit_radius:.2f}")
    
    # Run through the cycle
    print(f"\n  === SESSION 1 ===")
    print(f"  {gamer.log_on()}")
    print(f"  {gamer.browse_options(['Dungeon A', 'Arena', 'Trading Post'])}")
    print(f"  {gamer.choose_activity('Dungeon A')}")
    print(f"  {gamer.perform_work(1.0, payment=0.5)}")
    print(f"  {gamer.receive_rewards(2.0)}")
    print(f"  {gamer.choose_activity('Arena')}")
    print(f"  {gamer.perform_work(0.8)}")
    print(f"  {gamer.receive_rewards(1.5)}")
    print(f"  {gamer.log_off()}")
    
    print(f"\n  === OFFLINE ===")
    print(f"  {gamer.think_and_plan(['Better gear', 'Join a guild'])}")
    print(f"  {gamer.rest(1.0)}")
    print(f"  {gamer.dream_possibilities(['Become top player', 'Lead a raid'])}")
    
    print(f"\n  Final state:")
    print(f"    Phase: {actor.phase_state.phase.value}")
    print(f"    Mass: {actor.mass:.2f} (increased from engagement)")
    print(f"    Orbit: {actor.orbit_radius:.2f} (pulled toward center)")
    
    # Demonstrate z-axis interactions
    print("\n" + "─" * 70)
    print("Z-AXIS INTERACTION COSTS")
    print("─" * 70)
    
    print("\n  DIRECT INTERACTIONS:")
    network.demonstrate_interaction(PI, 'gaming')    # New user → Gaming
    network.demonstrate_interaction(2.0, 'staff')    # Creator → Staff
    network.demonstrate_interaction(2.0, 'gaming')   # Creator → Gaming (too far)
    network.demonstrate_interaction(1.3, 'gaming')   # Staff → Gaming
    
    print("\n  BONUS PATH (through intermediary):")
    # Subscriber at 2.5 → Creator at 2.0 → Gaming at 0.25
    network.demonstrate_bonus_path(2.5, 2.0, 'gaming')
    # Power user at 2.5 → Staff at 1.3 → Education at 0.9
    network.demonstrate_bonus_path(2.5, 1.3, 'education')
    
    # Summary
    print("\n" + "─" * 70)
    print("SUMMARY: The Complete Model")
    print("─" * 70)
    
    print("""
    THE HEXAGONAL PHASE CYCLE:
    
        Four quadrants: Q-EXP, C-EXP, C-COL, Q-COL
        - Top/Bottom: e (expand) / ln (collapse)
        - Left/Right: Quantum / Classical
    
    THE GAMER CYCLE:
    
        Log on → Browse → Choose → Work → Rewards → (loop)
        Log off → Reflect → Dream → Rest → Log on
        
        More engagement = more mass = pulled to center
        "That guy" everyone wants to be = high mass, central orbit
    
    THE Z-AXIS LAYERS:
    
        Each domain at a z-position
        Everyone reaches ±1
        Overlap determines cost
        
        Close domains → Low translation cost → High rotation
        Far domains → Must use intermediary paths
    
    THE BONUS PATHS:
    
        Subscribers → Creator → Domain
        Creator absorbs some cost (their relationship with domain)
        
        Example: Gaming subscriber gets bonus through creator
        - Subscriber doesn't have to go through bank
        - Creator has pre-paid relationship with gaming domain
        - Bonus flows from devs → creator → subscriber
    
    THE WORKER CYCLE:
    
        Same phases apply:
        - Come to work → C-EXP (see tasks)
        - Choose task → C-COL (commit)
        - Do work → earn pay
        - Go home → Q-COL (reflect on needs)
        - Rest/plan → Q-EXP (dream of growth)
        - Return to work → cycle continues
    """)
    
    return network


if __name__ == "__main__":
    network = demonstrate_hexagonal_cycle()
