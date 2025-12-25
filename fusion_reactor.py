"""
FUSION REACTOR TRAINING SYSTEM
==============================
Thermal containment for leveling up users safely and privately.

The Problem:
- Users near upgrade threshold (θ ≈ boundary) are vulnerable
- Failures during transition are visible to competitors  
- Rushing to cross threshold leads to mistakes
- Internal processes/criteria shouldn't be public

The Solution: Fusion Reactors
- Contained spaces where heat can build safely
- Walls hide the process from outside view
- Internal secrets can be shared with candidates
- Exit only when BEC threshold is firmly crossed

Reactor Types (from Economic Hydrology):
1. Tokamak: Slow, steady, mentor-guided
2. NIF: Intense burst, multiple simultaneous validators
3. Stellarator: Complex twisted path, self-sustaining

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from datetime import datetime, timedelta
import hashlib

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2

# Temperature thresholds
BEC_THRESHOLD = 1.0        # Must reach this for fusion
ENTRY_MIN_TEMP = 0.3       # Minimum temp to enter reactor
COOLING_RATE = 0.02        # Natural cooling per hour
HEATING_RATE = 0.05        # Base heating per action

# Time limits (seconds for demo, would be days in production)
MAX_REACTOR_TIME = 86400 * 30  # 30 days max in reactor


# ═══════════════════════════════════════════════════════════════════════════════
# REACTOR TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ReactorType(Enum):
    """Types of fusion reactors for different upgrade paths."""
    
    TOKAMAK = auto()      # Slow, steady, mentor-guided
    NIF = auto()          # Intense burst, multiple validators
    STELLARATOR = auto()  # Complex path, self-sustaining
    
    @property
    def description(self) -> str:
        descs = {
            ReactorType.TOKAMAK: "Magnetic confinement - slow steady heating with mentor guidance",
            ReactorType.NIF: "Inertial confinement - intense burst from multiple validators",
            ReactorType.STELLARATOR: "Twisted confinement - complex path through multiple axes"
        }
        return descs[self]
    
    @property
    def heating_multiplier(self) -> float:
        """How fast this reactor heats up."""
        mults = {
            ReactorType.TOKAMAK: 0.8,      # Slower but safer
            ReactorType.NIF: 2.0,          # Fast but needs coordination
            ReactorType.STELLARATOR: 1.0   # Balanced
        }
        return mults[self]
    
    @property
    def cooling_multiplier(self) -> float:
        """How fast this reactor cools down."""
        mults = {
            ReactorType.TOKAMAK: 0.5,      # Holds heat well
            ReactorType.NIF: 1.5,          # Cools quickly if not maintained
            ReactorType.STELLARATOR: 0.7   # Good retention
        }
        return mults[self]
    
    @property
    def capacity(self) -> int:
        """How many candidates can be in this reactor."""
        caps = {
            ReactorType.TOKAMAK: 5,        # Small, personal
            ReactorType.NIF: 1,            # One at a time (focused)
            ReactorType.STELLARATOR: 10    # Larger cohort
        }
        return caps[self]


# ═══════════════════════════════════════════════════════════════════════════════
# THERMAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThermalState:
    """Tracks temperature and thermal history of a candidate."""
    
    user_id: str
    current_temp: float = 0.0
    peak_temp: float = 0.0
    
    # History
    temp_history: List[Tuple[float, float]] = field(default_factory=list)  # (time, temp)
    heating_events: List[Dict] = field(default_factory=list)
    cooling_events: List[Dict] = field(default_factory=list)
    
    # Timestamps
    entered_at: float = 0.0
    last_activity: float = 0.0
    
    @property
    def time_in_reactor(self) -> float:
        """Seconds since entering reactor."""
        if self.entered_at == 0:
            return 0
        return time.time() - self.entered_at
    
    @property
    def time_since_activity(self) -> float:
        """Seconds since last heating activity."""
        if self.last_activity == 0:
            return 0
        return time.time() - self.last_activity
    
    @property
    def is_at_bec(self) -> bool:
        """Has reached BEC threshold."""
        return self.current_temp >= BEC_THRESHOLD
    
    @property
    def stability(self) -> float:
        """How stable is the temperature? (0-1, higher = more stable)"""
        if len(self.temp_history) < 5:
            return 0.5
        
        recent = [t for _, t in self.temp_history[-10:]]
        if not recent:
            return 0.5
        
        variance = sum((t - sum(recent)/len(recent))**2 for t in recent) / len(recent)
        return max(0, 1 - variance * 10)
    
    def record_temp(self):
        """Record current temperature in history."""
        self.temp_history.append((time.time(), self.current_temp))
        self.peak_temp = max(self.peak_temp, self.current_temp)
    
    def heat(self, amount: float, source: str):
        """Add heat from an activity."""
        self.current_temp = min(1.5, self.current_temp + amount)  # Cap at 1.5
        self.last_activity = time.time()
        self.heating_events.append({
            'time': time.time(),
            'amount': amount,
            'source': source,
            'new_temp': self.current_temp
        })
        self.record_temp()
    
    def cool(self, amount: float, reason: str):
        """Remove heat (natural cooling or penalty)."""
        self.current_temp = max(0, self.current_temp - amount)
        self.cooling_events.append({
            'time': time.time(),
            'amount': amount,
            'reason': reason,
            'new_temp': self.current_temp
        })
        self.record_temp()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL SECRETS (Shared Inside Reactor)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReactorSecrets:
    """
    Secrets that can be shared inside the reactor walls.
    These are invisible to the outside world.
    """
    
    # Upgrade criteria (normally hidden)
    required_contributions: int = 0
    required_quality_score: float = 0.0
    required_mentor_approval: bool = False
    
    # Internal documentation
    process_docs: List[str] = field(default_factory=list)
    criteria_docs: List[str] = field(default_factory=list)
    
    # Mentor access
    assigned_mentors: List[str] = field(default_factory=list)
    mentor_notes: Dict[str, List[str]] = field(default_factory=dict)
    
    # Real feedback (candid, not public-facing)
    private_feedback: List[Dict] = field(default_factory=list)
    
    # Progress tracking (detailed, internal)
    progress_checkpoints: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# FUSION REACTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionReactor:
    """
    A contained space for upgrading users.
    
    Features:
    - Thermal containment (heat builds up safely)
    - Privacy walls (process is hidden from outside)
    - Internal secrets (shared only with candidates)
    - BEC formation detection
    """
    
    reactor_id: str
    reactor_type: ReactorType
    
    # Transition being facilitated
    source_theta: float  # Starting position (e.g., 80°)
    target_theta: float  # Target position (e.g., 90°)
    source_role: str
    target_role: str
    
    # Candidates inside
    candidates: Dict[str, ThermalState] = field(default_factory=dict)
    
    # Secrets accessible inside
    secrets: ReactorSecrets = field(default_factory=ReactorSecrets)
    
    # Wall state
    wall_integrity: float = 1.0  # 1.0 = fully opaque, 0.0 = transparent
    
    # History
    successful_fusions: List[Dict] = field(default_factory=list)
    failed_attempts: List[Dict] = field(default_factory=list)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    
    @property
    def capacity(self) -> int:
        return self.reactor_type.capacity
    
    @property
    def current_occupancy(self) -> int:
        return len(self.candidates)
    
    @property
    def has_space(self) -> bool:
        return self.current_occupancy < self.capacity
    
    @property
    def avg_temperature(self) -> float:
        """Average temperature of all candidates."""
        if not self.candidates:
            return 0.0
        return sum(c.current_temp for c in self.candidates.values()) / len(self.candidates)
    
    def can_admit(self, user_id: str, current_temp: float) -> Tuple[bool, str]:
        """Check if user can enter this reactor."""
        if user_id in self.candidates:
            return False, "Already in reactor"
        
        if not self.has_space:
            return False, f"Reactor at capacity ({self.capacity})"
        
        if current_temp < ENTRY_MIN_TEMP:
            return False, f"Temperature too low ({current_temp:.2f} < {ENTRY_MIN_TEMP})"
        
        return True, "OK"
    
    def admit(self, user_id: str, initial_temp: float) -> ThermalState:
        """Admit a candidate to the reactor."""
        state = ThermalState(
            user_id=user_id,
            current_temp=initial_temp,
            peak_temp=initial_temp,
            entered_at=time.time(),
            last_activity=time.time()
        )
        state.record_temp()
        self.candidates[user_id] = state
        return state
    
    def apply_heating(self, user_id: str, action: str, base_amount: float) -> Dict:
        """Apply heating from an action."""
        if user_id not in self.candidates:
            return {'error': 'Not in reactor'}
        
        state = self.candidates[user_id]
        amount = base_amount * self.reactor_type.heating_multiplier
        state.heat(amount, action)
        
        return {
            'heated': amount,
            'new_temp': state.current_temp,
            'is_at_bec': state.is_at_bec
        }
    
    def apply_cooling(self):
        """Apply natural cooling to all candidates."""
        for user_id, state in self.candidates.items():
            # Cooling based on time since activity
            hours_inactive = state.time_since_activity / 3600
            cooling = COOLING_RATE * hours_inactive * self.reactor_type.cooling_multiplier
            
            if cooling > 0.01:  # Meaningful cooling
                state.cool(cooling, "natural_decay")
    
    def check_bec_formation(self, user_id: str) -> Dict:
        """
        Check if candidate has achieved stable BEC.
        
        Requirements:
        1. Temperature >= BEC_THRESHOLD
        2. Stability above 0.7 (not fluctuating wildly)
        3. Sustained for some time
        """
        if user_id not in self.candidates:
            return {'error': 'Not in reactor'}
        
        state = self.candidates[user_id]
        
        result = {
            'user_id': user_id,
            'current_temp': state.current_temp,
            'threshold': BEC_THRESHOLD,
            'stability': state.stability,
            'is_at_bec': state.is_at_bec,
            'bec_stable': False
        }
        
        if state.is_at_bec and state.stability > 0.7:
            result['bec_stable'] = True
            result['ready_for_fusion'] = True
        
        return result
    
    def complete_fusion(self, user_id: str) -> Dict:
        """
        Complete the fusion process - user exits at new level.
        """
        if user_id not in self.candidates:
            return {'error': 'Not in reactor'}
        
        state = self.candidates[user_id]
        
        if not state.is_at_bec:
            return {'error': 'Not at BEC threshold', 'temp': state.current_temp}
        
        # Record successful fusion
        fusion_record = {
            'user_id': user_id,
            'completed_at': time.time(),
            'time_in_reactor': state.time_in_reactor,
            'peak_temp': state.peak_temp,
            'heating_events': len(state.heating_events),
            'source_theta': self.source_theta,
            'target_theta': self.target_theta,
            'source_role': self.source_role,
            'target_role': self.target_role
        }
        self.successful_fusions.append(fusion_record)
        
        # Remove from reactor
        del self.candidates[user_id]
        
        return {
            'success': True,
            'fusion': fusion_record,
            'new_role': self.target_role,
            'new_theta': self.target_theta
        }
    
    def abort(self, user_id: str, reason: str) -> Dict:
        """
        Abort the process - user exits at original level.
        """
        if user_id not in self.candidates:
            return {'error': 'Not in reactor'}
        
        state = self.candidates[user_id]
        
        # Record failed attempt
        failure_record = {
            'user_id': user_id,
            'aborted_at': time.time(),
            'time_in_reactor': state.time_in_reactor,
            'peak_temp': state.peak_temp,
            'final_temp': state.current_temp,
            'reason': reason
        }
        self.failed_attempts.append(failure_record)
        
        # Remove from reactor
        del self.candidates[user_id]
        
        return {
            'success': False,
            'aborted': True,
            'reason': reason,
            'original_role': self.source_role
        }
    
    def get_internal_view(self, user_id: str) -> Dict:
        """
        Get the internal view (only available inside reactor).
        Includes secrets and detailed progress.
        """
        if user_id not in self.candidates:
            return {'error': 'Not in reactor - no access to internal view'}
        
        state = self.candidates[user_id]
        
        return {
            'user_id': user_id,
            'thermal_state': {
                'current_temp': state.current_temp,
                'peak_temp': state.peak_temp,
                'stability': state.stability,
                'time_in_reactor_hours': state.time_in_reactor / 3600,
                'is_at_bec': state.is_at_bec
            },
            'secrets': {
                'required_contributions': self.secrets.required_contributions,
                'required_quality': self.secrets.required_quality_score,
                'mentors': self.secrets.assigned_mentors,
                'process_docs': len(self.secrets.process_docs),
                'private_feedback': len(self.secrets.private_feedback)
            },
            'progress': {
                'heating_events': len(state.heating_events),
                'checkpoints_passed': len(self.secrets.progress_checkpoints)
            }
        }
    
    def get_external_view(self) -> Dict:
        """
        Get the external view (what outsiders see).
        Walls hide the details!
        """
        return {
            'reactor_id': self.reactor_id,
            'reactor_type': self.reactor_type.name,
            'transition': f"{self.source_role} → {self.target_role}",
            'occupancy': f"{self.current_occupancy}/{self.capacity}",
            # NO internal details!
            'candidates': list(self.candidates.keys()),  # Just names, no temps
            'successful_fusions': len(self.successful_fusions),
            'wall_status': 'OPAQUE' if self.wall_integrity > 0.5 else 'DEGRADED'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REACTOR FACILITY (Multiple Reactors)
# ═══════════════════════════════════════════════════════════════════════════════

class ReactorFacility:
    """
    Manages multiple fusion reactors for different role transitions.
    """
    
    def __init__(self):
        self.reactors: Dict[str, FusionReactor] = {}
        self.user_positions: Dict[str, float] = {}  # Current theta
        self.user_roles: Dict[str, str] = {}
        
        # Setup default reactors
        self._setup_default_reactors()
    
    def _setup_default_reactors(self):
        """Create reactors for common transitions."""
        
        # Core User → Creator (Tokamak - slow and steady)
        self.create_reactor(
            reactor_id="core_to_creator",
            reactor_type=ReactorType.TOKAMAK,
            source_theta=math.radians(60),
            target_theta=math.radians(90),
            source_role="Core User",
            target_role="Creator"
        )
        
        # Creator → Staff (NIF - intense validation)
        self.create_reactor(
            reactor_id="creator_to_staff",
            reactor_type=ReactorType.NIF,
            source_theta=math.radians(90),
            target_theta=math.radians(120),
            source_role="Creator",
            target_role="Staff"
        )
        
        # Staff → Admin (Stellarator - complex path)
        self.create_reactor(
            reactor_id="staff_to_admin",
            reactor_type=ReactorType.STELLARATOR,
            source_theta=math.radians(120),
            target_theta=math.radians(150),
            source_role="Staff",
            target_role="Admin"
        )
    
    def create_reactor(self, 
                      reactor_id: str,
                      reactor_type: ReactorType,
                      source_theta: float,
                      target_theta: float,
                      source_role: str,
                      target_role: str) -> FusionReactor:
        """Create a new reactor."""
        reactor = FusionReactor(
            reactor_id=reactor_id,
            reactor_type=reactor_type,
            source_theta=source_theta,
            target_theta=target_theta,
            source_role=source_role,
            target_role=target_role
        )
        
        # Setup secrets for this reactor
        reactor.secrets.required_contributions = random.randint(10, 50)
        reactor.secrets.required_quality_score = random.uniform(0.7, 0.9)
        reactor.secrets.process_docs = [
            f"Step {i}: {desc}" 
            for i, desc in enumerate([
                "Initial assessment",
                "Mentor assignment", 
                "Contribution phase",
                "Review period",
                "Final validation"
            ], 1)
        ]
        
        self.reactors[reactor_id] = reactor
        return reactor
    
    def register_user(self, user_id: str, theta: float, role: str):
        """Register a user's current position."""
        self.user_positions[user_id] = theta
        self.user_roles[user_id] = role
    
    def find_reactor_for_user(self, user_id: str) -> Optional[FusionReactor]:
        """Find appropriate reactor for user's next transition."""
        theta = self.user_positions.get(user_id, 0)
        
        for reactor in self.reactors.values():
            # Check if user is at the source level for this reactor
            theta_diff = abs(theta - reactor.source_theta)
            if theta_diff < math.radians(15):  # Within 15 degrees
                if reactor.has_space:
                    return reactor
        
        return None
    
    def enter_reactor(self, user_id: str, reactor_id: str, 
                     initial_temp: float) -> Dict:
        """Attempt to enter a reactor."""
        reactor = self.reactors.get(reactor_id)
        if not reactor:
            return {'error': 'Unknown reactor'}
        
        can_enter, reason = reactor.can_admit(user_id, initial_temp)
        if not can_enter:
            return {'error': reason}
        
        state = reactor.admit(user_id, initial_temp)
        
        return {
            'success': True,
            'reactor_id': reactor_id,
            'transition': f"{reactor.source_role} → {reactor.target_role}",
            'initial_temp': initial_temp,
            'internal_view': reactor.get_internal_view(user_id)
        }
    
    def perform_action(self, user_id: str, action: str) -> Dict:
        """User performs action inside reactor."""
        # Find which reactor they're in
        for reactor_id, reactor in self.reactors.items():
            if user_id in reactor.candidates:
                # Action heats based on type
                heat_amounts = {
                    'contribution': 0.1,
                    'review': 0.05,
                    'mentoring': 0.08,
                    'documentation': 0.06,
                    'validation': 0.15
                }
                amount = heat_amounts.get(action, 0.03)
                
                result = reactor.apply_heating(user_id, action, amount)
                result['reactor_id'] = reactor_id
                result['action'] = action
                
                # Check for BEC
                if result.get('is_at_bec'):
                    bec_check = reactor.check_bec_formation(user_id)
                    result['bec_check'] = bec_check
                
                return result
        
        return {'error': 'Not in any reactor'}
    
    def check_and_complete(self, user_id: str) -> Dict:
        """Check if user can complete fusion and exit."""
        for reactor_id, reactor in self.reactors.items():
            if user_id in reactor.candidates:
                bec_check = reactor.check_bec_formation(user_id)
                
                if bec_check.get('ready_for_fusion'):
                    result = reactor.complete_fusion(user_id)
                    if result.get('success'):
                        # Update user position
                        self.user_positions[user_id] = reactor.target_theta
                        self.user_roles[user_id] = reactor.target_role
                    return result
                else:
                    return {
                        'not_ready': True,
                        'current_temp': bec_check['current_temp'],
                        'stability': bec_check['stability'],
                        'needed': 'stable BEC formation'
                    }
        
        return {'error': 'Not in any reactor'}
    
    def get_facility_status(self) -> Dict:
        """Get status of all reactors (external view only)."""
        return {
            'reactors': {
                rid: reactor.get_external_view()
                for rid, reactor in self.reactors.items()
            },
            'total_successful_fusions': sum(
                len(r.successful_fusions) for r in self.reactors.values()
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_reactor_types():
    """Demonstrate different reactor types."""
    print("\n" + "=" * 80)
    print("FUSION REACTOR TYPES")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  1. TOKAMAK (Magnetic Confinement)                                         │
    │     ╭──────────────────────╮                                               │
    │     │  ~ ~ ~ ~ ~ ~ ~ ~ ~   │  ← Magnetic field guides plasma               │
    │     │  ~   PLASMA    ~     │                                               │
    │     │  ~ ~ ~ ~ ~ ~ ~ ~ ~   │  Properties:                                  │
    │     ╰──────────────────────╯  • Slow, steady heating (0.8x)                │
    │                                • Low cooling (0.5x) - holds heat well       │
    │     Good for: Careful candidates who need mentorship                        │
    │                                                                             │
    │  2. NIF (Inertial Confinement)                                             │
    │        ↘   ↓   ↙                                                           │
    │         ╲  │  ╱                                                            │
    │     → →  ◉ ← ←  ← All validators focus simultaneously                     │
    │         ╱  │  ╲                                                            │
    │        ↗   ↑   ↖              Properties:                                  │
    │                               • Fast heating (2.0x)                         │
    │     Good for: Proven candidates needing quick validation                    │
    │     Capacity: 1 (focused attention)                                         │
    │                                                                             │
    │  3. STELLARATOR (Twisted Confinement)                                      │
    │     ╭─╮ ╭─╮ ╭─╮ ╭─╮ ╭─╮                                                   │
    │    ─╯ ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─     ← Complex twisted path                       │
    │                                                                             │
    │     Properties:                                                             │
    │     • Balanced heating (1.0x)                                              │
    │     • Good heat retention (0.7x)                                           │
    │     • Largest capacity (10)                                                │
    │     Good for: Complex transitions, cohort learning                          │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n┌───────────────┬──────────┬──────────┬──────────┬─────────────────────┐")
    print("│ Reactor Type  │ Heating  │ Cooling  │ Capacity │ Best For            │")
    print("├───────────────┼──────────┼──────────┼──────────┼─────────────────────┤")
    
    for rt in ReactorType:
        print(f"│ {rt.name:13} │ {rt.heating_multiplier:6.1f}x  │ {rt.cooling_multiplier:6.1f}x  │ {rt.capacity:8} │ {rt.description[:19]} │")
    
    print("└───────────────┴──────────┴──────────┴──────────┴─────────────────────┘")


def demo_containment_and_privacy():
    """Demonstrate how walls provide containment AND privacy."""
    print("\n" + "=" * 80)
    print("CONTAINMENT AND PRIVACY WALLS")
    print("=" * 80)
    
    facility = ReactorFacility()
    
    # Register a user ready for transition
    facility.register_user("alice", math.radians(58), "Core User")
    
    print("\n--- EXTERNAL VIEW (What competitors see) ---")
    
    # Enter reactor
    result = facility.enter_reactor("alice", "core_to_creator", initial_temp=0.5)
    print(f"Alice entered reactor: {result['success']}")
    
    # External view - minimal info
    external = facility.reactors["core_to_creator"].get_external_view()
    print(f"\nExternal view of reactor:")
    for key, value in external.items():
        print(f"  {key}: {value}")
    
    print("\n--- INTERNAL VIEW (What Alice sees) ---")
    
    internal = facility.reactors["core_to_creator"].get_internal_view("alice")
    print(f"\nInternal view (inside the walls):")
    for section, data in internal.items():
        if isinstance(data, dict):
            print(f"  {section}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {section}: {data}")
    
    print("""
    
    KEY INSIGHT:
    ════════════
    
    External view:                     Internal view:
    ┌─────────────────────┐           ┌─────────────────────────────────┐
    │ • reactor_type      │           │ • current_temp: 0.5             │
    │ • occupancy: 1/5    │           │ • peak_temp: 0.5                │
    │ • candidates: alice │           │ • stability: 0.5                │
    │ • wall: OPAQUE      │           │ • required_contributions: 47    │
    │                     │           │ • required_quality: 0.82        │
    │ That's it!          │           │ • mentors: [...]                │
    │                     │           │ • private_feedback: [...]       │
    └─────────────────────┘           │ • process_docs: 5 steps         │
                                      └─────────────────────────────────┘
    
    Competitors only see that Alice is "in training"
    They don't see:
    • Her progress (temperature)
    • The criteria she needs to meet
    • Feedback she's receiving
    • How close she is to completion
    """)


def demo_heating_journey():
    """Demonstrate a complete heating journey to BEC."""
    print("\n" + "=" * 80)
    print("HEATING JOURNEY TO BEC FUSION")
    print("=" * 80)
    
    facility = ReactorFacility()
    facility.register_user("bob", math.radians(59), "Core User")
    
    # Enter reactor
    result = facility.enter_reactor("bob", "core_to_creator", initial_temp=0.35)
    print(f"\nBob enters reactor at temp = 0.35")
    
    reactor = facility.reactors["core_to_creator"]
    
    # Simulate actions over time
    actions = [
        ("contribution", "Created tutorial"),
        ("contribution", "Fixed bug"),
        ("mentoring", "Helped new user"),
        ("documentation", "Wrote docs"),
        ("contribution", "Major feature"),
        ("review", "Reviewed PR"),
        ("contribution", "Another feature"),
        ("validation", "Passed test"),
        ("contribution", "Great work"),
        ("validation", "Final approval"),
    ]
    
    print("\n┌─────┬────────────────┬────────────────┬──────────┬─────────┐")
    print("│ Day │ Action         │ Heat Added     │ New Temp │ Status  │")
    print("├─────┼────────────────┼────────────────┼──────────┼─────────┤")
    
    for i, (action, desc) in enumerate(actions, 1):
        result = facility.perform_action("bob", action)
        
        temp = result['new_temp']
        heated = result['heated']
        status = "🔥 BEC!" if result.get('is_at_bec') else ""
        
        print(f"│ {i:3} │ {action:14} │ +{heated:.3f}          │ {temp:8.3f} │ {status:7} │")
        
        if result.get('bec_check', {}).get('ready_for_fusion'):
            print("├─────┴────────────────┴────────────────┴──────────┴─────────┤")
            print("│                    FUSION READY!                          │")
            break
    
    print("└───────────────────────────────────────────────────────────────┘")
    
    # Complete fusion
    print("\n--- Completing Fusion ---")
    result = facility.check_and_complete("bob")
    
    if result.get('success'):
        print(f"✓ Bob is now a {result['new_role']}!")
        print(f"  New position: θ = {math.degrees(result['new_theta']):.0f}°")
        print(f"  Time in reactor: {result['fusion']['time_in_reactor']:.1f}s")
    else:
        print(f"Not ready: {result}")


def demo_external_vs_internal():
    """Show the stark difference between external and internal views."""
    print("\n" + "=" * 80)
    print("WHAT OUTSIDERS SEE vs WHAT INSIDERS SEE")
    print("=" * 80)
    
    print("""
    
    COMPETITOR'S VIEW:
    ══════════════════
    
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   Alice: Core User                                       │
    │      │                                                   │
    │      ├──── entered "core_to_creator" reactor ────┐      │
    │      │                                           │      │
    │      │     ┌───────────────────────────────┐     │      │
    │      │     │                               │     │      │
    │      │     │     ?????????????????????     │     │      │
    │      │     │     ?????????????????????     │     │      │
    │      │     │     ?????????????????????     │     │      │
    │      │     │                               │     │      │
    │      │     └───────────────────────────────┘     │      │
    │      │                                           │      │
    │      └───────────────────────────────────────────┘      │
    │      │                                                   │
    │      ▼                                                   │
    │   Alice: Creator  (some time later)                     │
    │                                                          │
    │   Competitor thinks: "How did she level up so fast?"    │
    │   Competitor knows:  Nothing about the process          │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    
    
    ALICE'S VIEW (Inside Reactor):
    ══════════════════════════════
    
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   TEMPERATURE GAUGE:                                     │
    │   ░░░░░░░░░████████████████░░░░░  T = 0.72 / 1.0        │
    │                                                          │
    │   REQUIREMENTS:                                          │
    │   ✓ Contributions: 35/30 (exceeded!)                    │
    │   ◐ Quality Score: 0.78/0.82 (almost there)             │
    │   ✓ Mentor Approval: Yes                                │
    │                                                          │
    │   PRIVATE FEEDBACK:                                      │
    │   "Great progress on docs, focus more on code quality"  │
    │   "Your mentoring is excellent, keep it up"             │
    │                                                          │
    │   NEXT STEPS:                                            │
    │   1. Submit one more high-quality contribution          │
    │   2. Pass final validation review                       │
    │   3. Maintain temperature until BEC stable              │
    │                                                          │
    │   MENTOR: @sarah (available for questions)              │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    
    
    THE WALLS PROVIDE:
    ══════════════════
    
    1. CONTAINMENT - Heat builds safely without external pressure
    2. PRIVACY    - Competitors can't see progress or criteria
    3. SECRETS    - Internal docs and feedback stay internal
    4. SAFETY     - Failures inside don't affect outside reputation
    """)


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("FUSION REACTOR SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Reactor types
    print("\n--- TEST 1: Reactor Types ---")
    for rt in ReactorType:
        assert rt.heating_multiplier > 0
        assert rt.cooling_multiplier > 0
        assert rt.capacity > 0
    print("✓ All reactor types have valid properties")
    
    # Test 2: Thermal state
    print("\n--- TEST 2: Thermal State ---")
    state = ThermalState(user_id="test")
    state.heat(0.5, "test_action")
    assert state.current_temp == 0.5
    assert state.peak_temp == 0.5
    state.cool(0.2, "test_cooling")
    assert state.current_temp == 0.3
    assert state.peak_temp == 0.5  # Peak unchanged
    print("✓ Thermal state heating/cooling works")
    
    # Test 3: Reactor admission
    print("\n--- TEST 3: Reactor Admission ---")
    reactor = FusionReactor(
        reactor_id="test",
        reactor_type=ReactorType.TOKAMAK,
        source_theta=0.5,
        target_theta=1.0,
        source_role="A",
        target_role="B"
    )
    
    can, reason = reactor.can_admit("user1", 0.1)  # Too cold
    assert not can
    print(f"✓ Rejected cold user: {reason}")
    
    can, reason = reactor.can_admit("user1", 0.5)  # Warm enough
    assert can
    print("✓ Accepted warm user")
    
    # Test 4: BEC detection
    print("\n--- TEST 4: BEC Detection ---")
    state = reactor.admit("user1", 0.5)
    
    # Heat to BEC
    for _ in range(20):
        reactor.apply_heating("user1", "contribution", 0.1)
    
    bec_check = reactor.check_bec_formation("user1")
    assert bec_check['is_at_bec']
    print(f"✓ BEC detected at temp {bec_check['current_temp']:.2f}")
    
    # Test 5: Facility
    print("\n--- TEST 5: Facility ---")
    facility = ReactorFacility()
    assert len(facility.reactors) >= 3
    print(f"✓ Facility has {len(facility.reactors)} reactors")
    
    status = facility.get_facility_status()
    assert 'reactors' in status
    print("✓ Facility status accessible")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--types":
            demo_reactor_types()
        elif sys.argv[1] == "--walls":
            demo_containment_and_privacy()
        elif sys.argv[1] == "--journey":
            demo_heating_journey()
        elif sys.argv[1] == "--views":
            demo_external_vs_internal()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_reactor_types()
            demo_containment_and_privacy()
            demo_heating_journey()
            demo_external_vs_internal()
        else:
            run_tests()
    else:
        run_tests()
