"""
THERMAL HIERARCHY SYSTEM
========================
Cryo chambers and fusion reactors balanced by mathematical constants.

Key Insights:
1. Wave peaks = furnace (hot), dips = freezer (cold)
2. Base frequency oscillates everyone through thermal zones
3. Thresholds based on φ, π, e create natural boundaries
4. Regular users stay in fridge/microwave (0.618 to 1.618)
5. New hires must oscillate full range (cryo to reactor)
6. Cryo = verification zone, catches "hand in cookie jar"

Thermal Zones (based on mathematical constants):

COLD SIDE (< 1, verification):    HOT SIDE (> 1, production):
─────────────────────────────    ─────────────────────────────
CRYO:        0 to φ⁻³ (0.236)    REACTOR:    > π (3.14)
DEEP FREEZE: φ⁻³ to 1/π (0.318)  FURNACE:    φ² to π
FREEZER:     1/π to φ⁻² (0.382)  OVEN:       φ to φ² (2.618)
FRIDGE:      φ⁻² to φ⁻¹ (0.618)  MICROWAVE:  1 to φ (1.618)
COOL:        φ⁻¹ to 1            WARM:       [transition]

Security Fixes:
1. Private nonce for phase offset (phase is secret)
2. Report both cycle_phase_type AND energy_phase_type

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import cmath
import time
import secrets
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

E = math.e           # ≈ 2.71828
PI = math.pi         # ≈ 3.14159
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.61803
I = complex(0, 1)

# Powers of PHI (our natural thresholds)
PHI_3 = PHI ** 3     # ≈ 4.236
PHI_2 = PHI ** 2     # ≈ 2.618
PHI_1 = PHI          # ≈ 1.618
PHI_0 = 1.0          # = 1 (balanced)
PHI_NEG1 = 1 / PHI   # ≈ 0.618
PHI_NEG2 = 1 / PHI_2 # ≈ 0.382
PHI_NEG3 = 1 / PHI_3 # ≈ 0.236

# Pi-based thresholds
INV_PI = 1 / PI      # ≈ 0.318
PI_OVER_E = PI / E   # ≈ 1.156

# Time constants
SECONDS = 1
MINUTES = 60
HOURS = 3600
DAYS = 86400


# ═══════════════════════════════════════════════════════════════════════════════
# THERMAL ZONES
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalZone(Enum):
    """
    Thermal zones based on mathematical constants.
    
    Cold side: verification, testing, auditing
    Hot side: production, action, growth
    """
    
    # Cold side (< 1) - Verification zones
    CRYO = ("CRYO", 0, PHI_NEG3, "Total freeze, full audit")
    DEEP_FREEZE = ("DEEP_FREEZE", PHI_NEG3, INV_PI, "Deep investigation")
    FREEZER = ("FREEZER", INV_PI, PHI_NEG2, "Serious verification")
    FRIDGE = ("FRIDGE", PHI_NEG2, PHI_NEG1, "Routine cooling")
    COOL = ("COOL", PHI_NEG1, 1.0, "Slight cooling")
    
    # Balanced
    BALANCED = ("BALANCED", 1.0, 1.0, "Perfect equilibrium")
    
    # Hot side (> 1) - Production zones
    WARM = ("WARM", 1.0, PI_OVER_E, "Slight heating")
    MICROWAVE = ("MICROWAVE", PI_OVER_E, PHI_1, "Regular work")
    OVEN = ("OVEN", PHI_1, PHI_2, "Intense work")
    FURNACE = ("FURNACE", PHI_2, PI, "High pressure")
    REACTOR = ("REACTOR", PI, float('inf'), "Fusion level")
    
    def __init__(self, name: str, min_val: float, max_val: float, description: str):
        self._name = name
        self.min_val = min_val
        self.max_val = max_val
        self.description = description
    
    @property
    def is_cold(self) -> bool:
        return self.max_val <= 1.0
    
    @property
    def is_hot(self) -> bool:
        return self.min_val >= 1.0
    
    @property
    def is_extreme(self) -> bool:
        """Is this an extreme zone (cryo or reactor)?"""
        return self in [ThermalZone.CRYO, ThermalZone.DEEP_FREEZE, 
                       ThermalZone.FURNACE, ThermalZone.REACTOR]
    
    @classmethod
    def from_temperature(cls, temp: float) -> 'ThermalZone':
        """Get zone for a given temperature."""
        if temp <= 0:
            return cls.CRYO
        elif temp < PHI_NEG3:
            return cls.CRYO
        elif temp < INV_PI:
            return cls.DEEP_FREEZE
        elif temp < PHI_NEG2:
            return cls.FREEZER
        elif temp < PHI_NEG1:
            return cls.FRIDGE
        elif temp < 1.0:
            return cls.COOL
        elif temp == 1.0:
            return cls.BALANCED
        elif temp < PI_OVER_E:
            return cls.WARM
        elif temp < PHI_1:
            return cls.MICROWAVE
        elif temp < PHI_2:
            return cls.OVEN
        elif temp < PI:
            return cls.FURNACE
        else:
            return cls.REACTOR


# ═══════════════════════════════════════════════════════════════════════════════
# SECURE PHASE (with private nonce)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SecureCycle:
    """
    Secure company cycle with private nonce.
    
    FIX #1: Phase offset uses a private nonce, not just genesis_time.
    The nonce is server-side only, making phase unlearnable from public info.
    """
    
    genesis_time: float
    cycle_period: float
    
    # PRIVATE: This nonce never leaves the server
    _private_nonce: str = field(default_factory=lambda: secrets.token_hex(32))
    
    # Derived phase offset (computed from nonce + genesis)
    phase_offset: float = field(default=0.0, init=False)
    
    def __post_init__(self):
        """Compute phase offset from BOTH genesis time AND private nonce."""
        # Hash both together - without the nonce, phase is unknowable
        combined = f"{self.genesis_time}:{self._private_nonce}"
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        
        # Convert to phase offset in [0, 2π)
        hash_value = int.from_bytes(hash_bytes[:8], 'big') / (2**64)
        self.phase_offset = hash_value * 2 * PI
    
    def current_phase(self, at_time: float = None) -> float:
        """Get current phase (only knowable with the nonce)."""
        if at_time is None:
            at_time = time.time()
        
        elapsed = at_time - self.genesis_time
        raw_phase = (2 * PI * elapsed / self.cycle_period) + self.phase_offset
        return raw_phase % (2 * PI)
    
    def get_euler_factor(self, at_time: float = None) -> complex:
        """Get e^(iθ) at current time."""
        return cmath.exp(I * self.current_phase(at_time))
    
    def phase_in_range(self, phase: float, start: float, end: float) -> bool:
        """Check if phase is in a given range (handles wraparound)."""
        phase = phase % (2 * PI)
        start = start % (2 * PI)
        end = end % (2 * PI)
        
        if start <= end:
            return start <= phase <= end
        else:  # Wraps around 0
            return phase >= start or phase <= end


# ═══════════════════════════════════════════════════════════════════════════════
# THERMAL STATE (Reports BOTH phase types)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThermalState:
    """
    Complete thermal state with both phase types.
    
    FIX #2: Report cycle_phase_type AND energy_phase_type separately.
    A clever attacker might try to hide at constructive cycle phase
    while pushing energy into destructive alignment.
    """
    
    # Current temperature (determines zone)
    temperature: float
    
    # The zone based on temperature
    zone: ThermalZone
    
    # Cycle phase info
    cycle_phase: float
    cycle_phase_degrees: float
    cycle_phase_type: str  # CONSTRUCTIVE, DESTRUCTIVE, ORTHOGONAL
    
    # Energy phase info (where the energy actually IS)
    energy_phase: float
    energy_phase_degrees: float
    energy_phase_type: str  # CONSTRUCTIVE, DESTRUCTIVE, ORTHOGONAL
    
    # The complex energy
    complex_energy: complex
    magnitude: float
    
    # Mismatch detection
    phase_mismatch: bool  # cycle says one thing, energy says another
    mismatch_warning: str
    
    # 3D position
    x: float
    y: float
    z: float

    @classmethod
    def phase_type(cls, phase: float) -> str:
        """Classify a phase as constructive, destructive, or orthogonal."""
        normalized = phase % (2 * PI)
        
        if normalized < PI/4 or normalized > 7*PI/4:
            return "CONSTRUCTIVE"
        elif 3*PI/4 < normalized < 5*PI/4:
            return "DESTRUCTIVE"
        else:
            return "ORTHOGONAL"


# ═══════════════════════════════════════════════════════════════════════════════
# CRYO CHAMBER (Freeze and Examine)
# ═══════════════════════════════════════════════════════════════════════════════

class CryoVerificationResult(Enum):
    """Results of cryo verification."""
    VERIFIED = auto()      # Passed all checks
    SUSPENDED = auto()     # Needs more investigation
    REJECTED = auto()      # Caught with hand in cookie jar
    THAWING = auto()       # Verification complete, returning to normal


@dataclass
class CryoSnapshot:
    """A frozen snapshot of state for verification."""
    
    timestamp: float
    
    # What they claimed
    claimed_cycle_phase_type: str
    claimed_zone: ThermalZone
    
    # What we found
    actual_energy_phase_type: str
    actual_temperature: float
    
    # The mismatch
    has_mismatch: bool
    mismatch_details: str
    
    # Complex state frozen for examination
    frozen_energy: complex
    frozen_magnitude: float
    frozen_phase: float
    
    # Verification result
    result: CryoVerificationResult = CryoVerificationResult.SUSPENDED


class CryoChamber:
    """
    Cryo chamber for freezing and verifying states.
    
    When someone is "frozen":
    1. All actions are suspended
    2. Their state is examined in detail
    3. Mismatches between claimed and actual state are detected
    4. "Hand in cookie jar" = claimed constructive but actually destructive
    """
    
    def __init__(self):
        self.frozen_subjects: Dict[str, CryoSnapshot] = {}
        self.verification_history: List[CryoSnapshot] = []
    
    def freeze(self, 
              subject_id: str,
              thermal_state: ThermalState) -> CryoSnapshot:
        """
        Freeze a subject for verification.
        
        This captures their current state and flags any mismatches.
        """
        # Detect the classic attack: claim constructive, actually destructive
        has_mismatch = (
            thermal_state.cycle_phase_type != thermal_state.energy_phase_type
        )
        
        if has_mismatch:
            mismatch_details = (
                f"Cycle claims {thermal_state.cycle_phase_type} but "
                f"energy is {thermal_state.energy_phase_type}"
            )
        else:
            mismatch_details = "No mismatch detected"
        
        snapshot = CryoSnapshot(
            timestamp=time.time(),
            claimed_cycle_phase_type=thermal_state.cycle_phase_type,
            claimed_zone=thermal_state.zone,
            actual_energy_phase_type=thermal_state.energy_phase_type,
            actual_temperature=thermal_state.temperature,
            has_mismatch=has_mismatch,
            mismatch_details=mismatch_details,
            frozen_energy=thermal_state.complex_energy,
            frozen_magnitude=thermal_state.magnitude,
            frozen_phase=thermal_state.energy_phase
        )
        
        self.frozen_subjects[subject_id] = snapshot
        
        return snapshot
    
    def examine(self, subject_id: str) -> Dict[str, Any]:
        """
        Examine a frozen subject in detail.
        
        This is where we catch "hand in cookie jar".
        """
        snapshot = self.frozen_subjects.get(subject_id)
        if not snapshot:
            return {'error': 'Subject not frozen'}
        
        # Deep examination
        examination = {
            'subject_id': subject_id,
            'frozen_at': snapshot.timestamp,
            'duration_frozen': time.time() - snapshot.timestamp,
            
            # What they claimed vs reality
            'claimed': {
                'cycle_phase_type': snapshot.claimed_cycle_phase_type,
                'zone': snapshot.claimed_zone.name
            },
            'actual': {
                'energy_phase_type': snapshot.actual_energy_phase_type,
                'temperature': snapshot.actual_temperature,
                'zone': ThermalZone.from_temperature(snapshot.actual_temperature).name
            },
            
            # The verdict
            'mismatch_detected': snapshot.has_mismatch,
            'mismatch_details': snapshot.mismatch_details,
            
            # Complex analysis
            'frozen_energy': {
                'magnitude': snapshot.frozen_magnitude,
                'phase_degrees': math.degrees(snapshot.frozen_phase) % 360,
                'real': snapshot.frozen_energy.real,
                'imag': snapshot.frozen_energy.imag
            }
        }
        
        # Determine result
        if snapshot.has_mismatch:
            if snapshot.actual_energy_phase_type == "DESTRUCTIVE":
                # Caught! Claimed constructive but actually destructive
                snapshot.result = CryoVerificationResult.REJECTED
                examination['result'] = 'REJECTED'
                examination['reason'] = 'Hand in cookie jar - claimed constructive while destructive'
            else:
                # Just orthogonal mismatch - suspicious but not fatal
                snapshot.result = CryoVerificationResult.SUSPENDED
                examination['result'] = 'SUSPENDED'
                examination['reason'] = 'Phase mismatch requires investigation'
        else:
            # Clean
            snapshot.result = CryoVerificationResult.VERIFIED
            examination['result'] = 'VERIFIED'
            examination['reason'] = 'State matches claims'
        
        return examination
    
    def thaw(self, subject_id: str) -> CryoVerificationResult:
        """Release a subject from cryo."""
        snapshot = self.frozen_subjects.pop(subject_id, None)
        if snapshot:
            snapshot.result = CryoVerificationResult.THAWING
            self.verification_history.append(snapshot)
            return snapshot.result
        return CryoVerificationResult.THAWING


# ═══════════════════════════════════════════════════════════════════════════════
# THERMAL OSCILLATOR (Base Frequency with Hot/Cold Phases)
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalOscillator:
    """
    Oscillator that moves through thermal zones based on phase.
    
    The wave naturally creates:
    - Peak (phase 0, 2π) → HOT zones (production)
    - Trough (phase π) → COLD zones (verification)
    
    Base frequency = room temp
    Owner actions perturb this, shaking everyone up
    """
    
    def __init__(self,
                 base_temperature: float = 1.0,
                 amplitude: float = 0.5,
                 cycle: SecureCycle = None):
        
        self.base_temperature = base_temperature
        self.amplitude = amplitude
        self.cycle = cycle or SecureCycle(
            genesis_time=time.time(),
            cycle_period=DAYS
        )
        
        # Energy state
        self.complex_energy = complex(base_temperature, 0)
        
        # Perturbations from owner actions
        self.perturbation: complex = complex(0, 0)
        
        # History for analysis
        self.history: deque = deque(maxlen=1000)
    
    def get_oscillated_temperature(self, at_time: float = None) -> float:
        """
        Get current temperature including oscillation.
        
        T(t) = base + amplitude × cos(phase)
        
        - cos(0) = 1 → peak → HOT
        - cos(π) = -1 → trough → COLD
        """
        phase = self.cycle.current_phase(at_time)
        oscillation = self.amplitude * math.cos(phase)
        
        # Include perturbation magnitude
        perturbation_effect = abs(self.perturbation) * 0.1
        
        return self.base_temperature + oscillation + perturbation_effect
    
    def perturb(self, owner_action_energy: float, action_phase: float):
        """
        Owner action perturbs the system.
        
        This "shakes things up" - prevents hiding.
        """
        perturbation = owner_action_energy * cmath.exp(I * action_phase)
        self.perturbation += perturbation
        
        # Add to energy
        self.complex_energy += perturbation * 0.5
        
        # Dampen over time
        self.perturbation *= 0.9
    
    def get_thermal_state(self, at_time: float = None) -> ThermalState:
        """Get complete thermal state with both phase types."""
        if at_time is None:
            at_time = time.time()
        
        # Temperatures and zones
        temp = self.get_oscillated_temperature(at_time)
        zone = ThermalZone.from_temperature(temp)
        
        # Cycle phase
        cycle_phase = self.cycle.current_phase(at_time)
        cycle_phase_type = ThermalState.phase_type(cycle_phase)
        
        # Energy phase (where energy actually is)
        if abs(self.complex_energy) > 0.01:
            energy_phase = cmath.phase(self.complex_energy)
        else:
            energy_phase = 0.0
        energy_phase_type = ThermalState.phase_type(energy_phase)
        
        # Mismatch detection
        phase_mismatch = (cycle_phase_type != energy_phase_type)
        if phase_mismatch:
            mismatch_warning = f"ALERT: Cycle={cycle_phase_type}, Energy={energy_phase_type}"
        else:
            mismatch_warning = ""
        
        # 3D position
        magnitude = abs(self.complex_energy)
        x = magnitude * math.cos(energy_phase)
        y = magnitude * math.sin(energy_phase)
        z = temp
        
        return ThermalState(
            temperature=temp,
            zone=zone,
            cycle_phase=cycle_phase,
            cycle_phase_degrees=math.degrees(cycle_phase) % 360,
            cycle_phase_type=cycle_phase_type,
            energy_phase=energy_phase,
            energy_phase_degrees=math.degrees(energy_phase) % 360,
            energy_phase_type=energy_phase_type,
            complex_energy=self.complex_energy,
            magnitude=magnitude,
            phase_mismatch=phase_mismatch,
            mismatch_warning=mismatch_warning,
            x=x, y=y, z=z
        )


# ═══════════════════════════════════════════════════════════════════════════════
# USER THERMAL ENVELOPE
# ═══════════════════════════════════════════════════════════════════════════════

class UserType(Enum):
    """User types determine thermal envelope (allowed range)."""
    
    REGULAR = ("REGULAR", PHI_NEG1, PHI_1, "Standard user")      # 0.618 to 1.618
    TRUSTED = ("TRUSTED", PHI_NEG2, PHI_2, "Trusted user")       # 0.382 to 2.618
    NEW_HIRE = ("NEW_HIRE", 0, PI + 1, "Must oscillate full")    # 0 to π+ (cryo to reactor)
    OWNER = ("OWNER", 0, float('inf'), "Unlimited range")
    
    def __init__(self, name: str, min_temp: float, max_temp: float, description: str):
        self._name = name
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.description = description
    
    def in_envelope(self, temp: float) -> bool:
        """Check if temperature is within allowed envelope."""
        return self.min_temp <= temp <= self.max_temp
    
    def required_zones(self) -> List[ThermalZone]:
        """Zones this user type must visit (for new hires)."""
        if self == UserType.NEW_HIRE:
            return [ThermalZone.CRYO, ThermalZone.REACTOR]  # Must visit extremes
        return []


@dataclass
class UserThermalProfile:
    """A user's thermal profile tracking their movement through zones."""
    
    user_id: str
    user_type: UserType
    
    # Current state
    current_oscillator: ThermalOscillator = None
    
    # Zone visit tracking (for new hires)
    zones_visited: Dict[ThermalZone, int] = field(default_factory=dict)
    
    # Time in each zone
    time_in_zones: Dict[ThermalZone, float] = field(default_factory=dict)
    
    # Last known state
    last_state: ThermalState = None
    last_update: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.current_oscillator is None:
            self.current_oscillator = ThermalOscillator()
    
    def update(self) -> ThermalState:
        """Update thermal state and track zones."""
        now = time.time()
        state = self.current_oscillator.get_thermal_state(now)
        
        # Track zone visit
        zone = state.zone
        self.zones_visited[zone] = self.zones_visited.get(zone, 0) + 1
        
        # Track time in zone
        if self.last_state and self.last_state.zone == zone:
            elapsed = now - self.last_update
            self.time_in_zones[zone] = self.time_in_zones.get(zone, 0) + elapsed
        
        self.last_state = state
        self.last_update = now
        
        return state
    
    def has_completed_oscillation(self) -> bool:
        """For new hires: have they visited all required zones?"""
        required = self.user_type.required_zones()
        for zone in required:
            if zone not in self.zones_visited:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE THERMAL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalHierarchySystem:
    """
    Complete thermal hierarchy with cryo chambers and fusion reactors.
    
    Features:
    1. Secure phase (private nonce)
    2. Both phase types reported
    3. Cryo verification for catching mismatches
    4. User thermal envelopes
    5. New hire oscillation requirements
    """
    
    def __init__(self):
        self.users: Dict[str, UserThermalProfile] = {}
        self.cryo_chamber = CryoChamber()
        
        # Master cycle (company-wide oscillation)
        self.master_cycle = SecureCycle(
            genesis_time=time.time(),
            cycle_period=DAYS  # Daily cycle
        )
    
    def register_user(self, 
                     user_id: str, 
                     user_type: UserType) -> Dict[str, Any]:
        """Register a user with their thermal envelope."""
        
        profile = UserThermalProfile(
            user_id=user_id,
            user_type=user_type,
            current_oscillator=ThermalOscillator(
                base_temperature=1.0,
                amplitude=0.3,
                cycle=self.master_cycle  # Tied to master cycle
            )
        )
        
        self.users[user_id] = profile
        
        return {
            'registered': True,
            'user_id': user_id,
            'user_type': user_type.name,
            'envelope': {
                'min': user_type.min_temp,
                'max': user_type.max_temp
            },
            'required_zones': [z.name for z in user_type.required_zones()]
        }
    
    def get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get current thermal state for a user."""
        profile = self.users.get(user_id)
        if not profile:
            return {'error': 'Unknown user'}
        
        state = profile.update()
        
        return {
            'user_id': user_id,
            'user_type': profile.user_type.name,
            'temperature': state.temperature,
            'zone': state.zone.name,
            'zone_description': state.zone.description,
            
            # Both phase types (FIX #2)
            'cycle_phase': {
                'degrees': state.cycle_phase_degrees,
                'type': state.cycle_phase_type
            },
            'energy_phase': {
                'degrees': state.energy_phase_degrees,
                'type': state.energy_phase_type
            },
            
            # Mismatch warning
            'phase_mismatch': state.phase_mismatch,
            'mismatch_warning': state.mismatch_warning,
            
            # 3D position
            'position': {'x': state.x, 'y': state.y, 'z': state.z},
            
            # Envelope check
            'in_envelope': profile.user_type.in_envelope(state.temperature),
            
            # New hire progress
            'zones_visited': {z.name: c for z, c in profile.zones_visited.items()},
            'oscillation_complete': profile.has_completed_oscillation()
        }
    
    def freeze_for_verification(self, user_id: str) -> Dict[str, Any]:
        """Send user to cryo for verification."""
        profile = self.users.get(user_id)
        if not profile:
            return {'error': 'Unknown user'}
        
        state = profile.update()
        snapshot = self.cryo_chamber.freeze(user_id, state)
        
        return {
            'frozen': True,
            'user_id': user_id,
            'snapshot': {
                'claimed_type': snapshot.claimed_cycle_phase_type,
                'actual_type': snapshot.actual_energy_phase_type,
                'mismatch': snapshot.has_mismatch,
                'details': snapshot.mismatch_details
            }
        }
    
    def examine_frozen(self, user_id: str) -> Dict[str, Any]:
        """Examine a frozen user in detail."""
        return self.cryo_chamber.examine(user_id)
    
    def thaw(self, user_id: str) -> Dict[str, Any]:
        """Release user from cryo."""
        result = self.cryo_chamber.thaw(user_id)
        return {
            'thawed': True,
            'user_id': user_id,
            'result': result.name
        }
    
    def owner_action(self, energy: float, phase: float = None) -> Dict[str, Any]:
        """
        Owner action perturbs the whole system.
        
        This shakes everyone up, preventing hiding.
        """
        if phase is None:
            phase = self.master_cycle.current_phase()
        
        # Perturb all users
        for user_id, profile in self.users.items():
            profile.current_oscillator.perturb(energy, phase)
        
        return {
            'perturbed': True,
            'energy': energy,
            'phase_degrees': math.degrees(phase) % 360,
            'users_affected': len(self.users)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_thermal_zones():
    """Demonstrate the thermal zone hierarchy."""
    print("\n" + "=" * 80)
    print("THERMAL ZONE HIERARCHY (Based on Mathematical Constants)")
    print("=" * 80)
    
    print(f"""
    Using fundamental constants as natural thresholds:
    
    φ = {PHI:.5f}  (golden ratio)
    π = {PI:.5f}
    e = {E:.5f}
    
    Powers of φ create the hierarchy:
    
    φ³  = {PHI_3:.3f}
    φ²  = {PHI_2:.3f}
    φ¹  = {PHI_1:.3f}
    φ⁰  = {PHI_0:.3f}  (balanced)
    φ⁻¹ = {PHI_NEG1:.3f}
    φ⁻² = {PHI_NEG2:.3f}
    φ⁻³ = {PHI_NEG3:.3f}
    1/π = {INV_PI:.3f}
    """)
    
    print("\n┌────────────────┬────────────────────┬──────────────────────────────┐")
    print("│ Zone           │ Temperature Range  │ Description                  │")
    print("├────────────────┼────────────────────┼──────────────────────────────┤")
    
    for zone in ThermalZone:
        if zone.max_val == float('inf'):
            range_str = f"{zone.min_val:.3f} to ∞"
        elif zone.min_val == zone.max_val:
            range_str = f"exactly {zone.min_val:.3f}"
        else:
            range_str = f"{zone.min_val:.3f} to {zone.max_val:.3f}"
        
        print(f"│ {zone.name:14} │ {range_str:18} │ {zone.description:28} │")
    
    print("└────────────────┴────────────────────┴──────────────────────────────┘")
    
    print("""
    
    COLD SIDE (< 1):          HOT SIDE (> 1):
    ════════════════          ═══════════════
    CRYO       ← audit        REACTOR → fusion
    DEEP_FREEZE← investigate  FURNACE → high pressure  
    FREEZER    ← verify       OVEN    → intense work
    FRIDGE     ← routine      MICROWAVE → regular
    COOL       ← slight       WARM    → slight
                  ↓              ↑
                  └── 1.0 ───────┘
                    (balanced)
    """)


def demo_secure_phase():
    """Demonstrate secure phase with private nonce."""
    print("\n" + "=" * 80)
    print("SECURE PHASE (FIX #1: Private Nonce)")
    print("=" * 80)
    
    print("""
    PROBLEM: If genesis_time leaks, attacker can track phase.
    
    OLD (insecure):
        phase_offset = H(genesis_time)
        
    NEW (secure):
        phase_offset = H(genesis_time || private_nonce)
        
    The nonce never leaves the server!
    """)
    
    # Create two cycles with same genesis but different nonces
    genesis = time.time()
    
    cycle1 = SecureCycle(genesis_time=genesis, cycle_period=HOURS)
    cycle2 = SecureCycle(genesis_time=genesis, cycle_period=HOURS)
    
    print(f"\n--- Two cycles with SAME genesis_time ---\n")
    print(f"Genesis time: {genesis}")
    print(f"Cycle 1 phase: {math.degrees(cycle1.current_phase()):.1f}°")
    print(f"Cycle 2 phase: {math.degrees(cycle2.current_phase()):.1f}°")
    print(f"Different? {abs(cycle1.current_phase() - cycle2.current_phase()) > 0.1}")
    
    print("""
    
    Even with the SAME genesis_time, the phases are DIFFERENT
    because each has its own private nonce!
    
    An attacker who knows genesis_time STILL can't predict phase.
    """)


def demo_phase_mismatch():
    """Demonstrate detecting phase mismatches."""
    print("\n" + "=" * 80)
    print("PHASE MISMATCH DETECTION (FIX #2: Both Phase Types)")
    print("=" * 80)
    
    print("""
    PROBLEM: Attacker might:
    - Check at "constructive cycle phase"
    - While pushing energy into "destructive alignment"
    
    SOLUTION: Report BOTH phase types!
    """)
    
    # Create oscillator
    osc = ThermalOscillator(
        base_temperature=1.0,
        amplitude=0.5
    )
    
    # Normal state
    state = osc.get_thermal_state()
    
    print(f"\n--- Normal State ---\n")
    print(f"Cycle phase: {state.cycle_phase_degrees:.1f}° ({state.cycle_phase_type})")
    print(f"Energy phase: {state.energy_phase_degrees:.1f}° ({state.energy_phase_type})")
    print(f"Mismatch: {state.phase_mismatch}")
    
    # Simulate attack: perturb energy to opposite phase
    print(f"\n--- After Perturbation (simulating attack) ---\n")
    
    # Push energy to opposite phase
    osc.complex_energy = 10 * cmath.exp(I * PI)  # Force to π (destructive)
    
    state = osc.get_thermal_state()
    
    print(f"Cycle phase: {state.cycle_phase_degrees:.1f}° ({state.cycle_phase_type})")
    print(f"Energy phase: {state.energy_phase_degrees:.1f}° ({state.energy_phase_type})")
    print(f"Mismatch: {state.phase_mismatch}")
    print(f"Warning: {state.mismatch_warning}")
    
    print("""
    
    Now we can SEE the attack:
    - Cycle claims CONSTRUCTIVE
    - Energy is actually DESTRUCTIVE
    - MISMATCH DETECTED!
    """)


def demo_cryo_chamber():
    """Demonstrate cryo chamber verification."""
    print("\n" + "=" * 80)
    print("CRYO CHAMBER: Freeze and Examine")
    print("=" * 80)
    
    print("""
    The cryo chamber FREEZES suspects and examines them closely.
    
    If they claimed "constructive" but are actually "destructive":
    → CAUGHT WITH HAND IN COOKIE JAR!
    → REJECTED
    """)
    
    system = ThermalHierarchySystem()
    
    # Register a suspicious user
    system.register_user("alice", UserType.REGULAR)
    
    # Manipulate their energy to create mismatch
    profile = system.users["alice"]
    profile.current_oscillator.complex_energy = 10 * cmath.exp(I * PI)
    
    # Freeze them
    print("\n--- Freezing Alice ---\n")
    result = system.freeze_for_verification("alice")
    print(f"Frozen: {result['frozen']}")
    print(f"Claimed: {result['snapshot']['claimed_type']}")
    print(f"Actual: {result['snapshot']['actual_type']}")
    print(f"Mismatch: {result['snapshot']['mismatch']}")
    
    # Examine
    print("\n--- Examining Frozen Alice ---\n")
    exam = system.examine_frozen("alice")
    print(f"Result: {exam['result']}")
    print(f"Reason: {exam['reason']}")
    
    print("""
    
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  CRYO VERIFICATION RESULTS:                                     │
    │                                                                  │
    │  VERIFIED  → State matches claims, release                      │
    │  SUSPENDED → Needs more investigation                           │
    │  REJECTED  → Caught! Hand in cookie jar                        │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    """)


def demo_user_envelopes():
    """Demonstrate user thermal envelopes."""
    print("\n" + "=" * 80)
    print("USER THERMAL ENVELOPES")
    print("=" * 80)
    
    print("""
    Different user types have different allowed temperature ranges:
    """)
    
    print("\n┌─────────────┬───────────────────┬──────────────────────────────────┐")
    print("│ User Type   │ Envelope          │ Description                      │")
    print("├─────────────┼───────────────────┼──────────────────────────────────┤")
    
    for ut in UserType:
        if ut.max_temp == float('inf'):
            envelope = f"{ut.min_temp:.3f} to ∞"
        else:
            envelope = f"{ut.min_temp:.3f} to {ut.max_temp:.3f}"
        
        print(f"│ {ut.name:11} │ {envelope:17} │ {ut.description:32} │")
    
    print("└─────────────┴───────────────────┴──────────────────────────────────┘")
    
    print("""
    
    REGULAR:  Stays in fridge/microwave zone (0.618 to 1.618)
              Normal work, normal verification
    
    TRUSTED:  Can go to freezer/oven (0.382 to 2.618)
              More responsibility, more scrutiny
    
    NEW_HIRE: MUST visit cryo AND reactor (0 to π+)
              Full oscillation required before promotion!
              This catches people who can't handle the pressure.
    
    OWNER:    Unlimited range
              Can go anywhere, but movements affect everyone
    """)


def demo_complete_system():
    """Demonstrate the complete thermal hierarchy system."""
    print("\n" + "=" * 80)
    print("COMPLETE THERMAL HIERARCHY SYSTEM")
    print("=" * 80)
    
    system = ThermalHierarchySystem()
    
    # Register users
    print("\n--- Registering Users ---\n")
    
    for user_id, user_type in [
        ("alice", UserType.REGULAR),
        ("bob", UserType.NEW_HIRE),
        ("carol", UserType.OWNER)
    ]:
        result = system.register_user(user_id, user_type)
        print(f"{user_id}: {result['user_type']}, envelope={result['envelope']}")
        if result['required_zones']:
            print(f"         Must visit: {result['required_zones']}")
    
    # Check states
    print("\n--- Current States ---\n")
    
    for user_id in ["alice", "bob", "carol"]:
        state = system.get_user_state(user_id)
        print(f"{user_id}:")
        print(f"  Zone: {state['zone']} (T={state['temperature']:.3f})")
        print(f"  Cycle: {state['cycle_phase']['degrees']:.0f}° {state['cycle_phase']['type']}")
        print(f"  Energy: {state['energy_phase']['degrees']:.0f}° {state['energy_phase']['type']}")
        print(f"  In envelope: {state['in_envelope']}")
    
    # Owner action shakes everyone
    print("\n--- Owner Action (Perturbs Everyone) ---\n")
    
    result = system.owner_action(energy=5.0)
    print(f"Perturbation applied: {result['energy']} energy at {result['phase_degrees']:.0f}°")
    print(f"Users affected: {result['users_affected']}")
    
    # Check new states
    print("\n--- States After Perturbation ---\n")
    
    for user_id in ["alice", "bob"]:
        state = system.get_user_state(user_id)
        print(f"{user_id}: Zone={state['zone']}, T={state['temperature']:.3f}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("THERMAL HIERARCHY SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Thermal zones
    print("\n--- TEST 1: Thermal Zones ---")
    assert ThermalZone.from_temperature(0.1) == ThermalZone.CRYO
    assert ThermalZone.from_temperature(0.5) == ThermalZone.FRIDGE
    assert ThermalZone.from_temperature(1.0) == ThermalZone.BALANCED
    assert ThermalZone.from_temperature(2.0) == ThermalZone.OVEN
    assert ThermalZone.from_temperature(4.0) == ThermalZone.REACTOR
    print("✓ Zone classification works")
    
    # Test 2: Secure phase (different nonces)
    print("\n--- TEST 2: Secure Phase ---")
    genesis = time.time()
    c1 = SecureCycle(genesis_time=genesis, cycle_period=HOURS)
    c2 = SecureCycle(genesis_time=genesis, cycle_period=HOURS)
    assert abs(c1.phase_offset - c2.phase_offset) > 0.1
    print(f"✓ Different nonces create different phases")
    
    # Test 3: Thermal state with both phases
    print("\n--- TEST 3: Thermal State ---")
    osc = ThermalOscillator()
    state = osc.get_thermal_state()
    assert hasattr(state, 'cycle_phase_type')
    assert hasattr(state, 'energy_phase_type')
    print(f"✓ Both phase types reported")
    
    # Test 4: Cryo chamber
    print("\n--- TEST 4: Cryo Chamber ---")
    cryo = CryoChamber()
    snapshot = cryo.freeze("test", state)
    assert snapshot.timestamp > 0
    exam = cryo.examine("test")
    assert 'result' in exam
    print(f"✓ Cryo freeze and examine works")
    
    # Test 5: User profiles
    print("\n--- TEST 5: User Profiles ---")
    profile = UserThermalProfile(user_id="test", user_type=UserType.REGULAR)
    assert profile.user_type.in_envelope(1.0)
    assert not profile.user_type.in_envelope(0.1)
    print(f"✓ User envelope works")
    
    # Test 6: Complete system
    print("\n--- TEST 6: Complete System ---")
    system = ThermalHierarchySystem()
    system.register_user("alice", UserType.REGULAR)
    state = system.get_user_state("alice")
    assert 'temperature' in state
    assert 'cycle_phase' in state
    assert 'energy_phase' in state
    print(f"✓ Complete system works")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--zones":
            demo_thermal_zones()
        elif sys.argv[1] == "--secure":
            demo_secure_phase()
        elif sys.argv[1] == "--mismatch":
            demo_phase_mismatch()
        elif sys.argv[1] == "--cryo":
            demo_cryo_chamber()
        elif sys.argv[1] == "--envelope":
            demo_user_envelopes()
        elif sys.argv[1] == "--system":
            demo_complete_system()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_thermal_zones()
            demo_secure_phase()
            demo_phase_mismatch()
            demo_cryo_chamber()
            demo_user_envelopes()
            demo_complete_system()
        else:
            run_tests()
    else:
        run_tests()
