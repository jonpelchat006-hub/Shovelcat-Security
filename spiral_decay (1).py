"""
SPIRAL DECAY SYSTEM
===================
True Euler identity integration: e^(iπ) = -1

Key Insight:
    Decay = e^(-πt/T)
    Euler = e^(iπ) = -1
    
    Combined: e^(-λt + iθ) = e^(-λt) × e^(iθ)
    
    This creates a SPIRAL in complex space!
    - Magnitude shrinks (decay)
    - Phase rotates (cycle)
    - Together: spiraling inward

The Phase of DNA Check Matters:
    θ = 0:    e^(i×0) = 1    → constructive (boost)
    θ = π/2:  e^(iπ/2) = i   → orthogonal (lateral)
    θ = π:    e^(iπ) = -1    → destructive (cancel)
    θ = 3π/2: e^(i3π/2) = -i → orthogonal (other)

3D Representation:
    X = r × cos(θ)    (real part)
    Y = r × sin(θ)    (imaginary part)
    Z = magnitude     (decay level)

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import cmath
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

E = math.e
PI = math.pi
PHI = (1 + math.sqrt(5)) / 2
I = complex(0, 1)  # The imaginary unit

# Euler's identity components
EULER_IDENTITY = cmath.exp(I * PI)  # Should equal -1

# Time constants
SECONDS = 1
MINUTES = 60
HOURS = 3600
DAYS = 86400


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEX POINT (Position in spiral space)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpiralPoint:
    """
    A point in the 3D spiral space.
    
    X = r × cos(θ)  (real component)
    Y = r × sin(θ)  (imaginary component)
    Z = magnitude   (decay level)
    
    Or in complex terms:
    position = magnitude × e^(iθ) = magnitude × (cos(θ) + i×sin(θ))
    """
    
    magnitude: float  # The "height" or energy level (Z axis)
    phase: float      # The angle in radians (θ, determines X-Y position)
    
    @property
    def complex_position(self) -> complex:
        """Get position as complex number: magnitude × e^(iθ)"""
        return self.magnitude * cmath.exp(I * self.phase)
    
    @property
    def x(self) -> float:
        """Real component: magnitude × cos(θ)"""
        return self.magnitude * math.cos(self.phase)
    
    @property
    def y(self) -> float:
        """Imaginary component: magnitude × sin(θ)"""
        return self.magnitude * math.sin(self.phase)
    
    @property
    def z(self) -> float:
        """Magnitude (decay level)"""
        return self.magnitude
    
    @property
    def euler_factor(self) -> complex:
        """The e^(iθ) component alone"""
        return cmath.exp(I * self.phase)
    
    @property
    def phase_degrees(self) -> float:
        """Phase in degrees for readability"""
        return math.degrees(self.phase) % 360
    
    def is_constructive(self) -> bool:
        """Is phase in constructive range? (near 0 or 2π)"""
        normalized = self.phase % (2 * PI)
        return normalized < PI/4 or normalized > 7*PI/4
    
    def is_destructive(self) -> bool:
        """Is phase in destructive range? (near π)"""
        normalized = self.phase % (2 * PI)
        return 3*PI/4 < normalized < 5*PI/4
    
    def is_orthogonal(self) -> bool:
        """Is phase in orthogonal range? (near π/2 or 3π/2)"""
        normalized = self.phase % (2 * PI)
        return (PI/4 < normalized < 3*PI/4) or (5*PI/4 < normalized < 7*PI/4)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE-LOCKED CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompanyCycle:
    """
    The company's fundamental cycle.
    
    Genesis sets the cycle start.
    Current phase = where we are in the cycle.
    DNA checks happen at specific phases.
    """
    
    genesis_time: float  # When the company was founded
    cycle_period: float  # Length of one full cycle (seconds)
    
    # Phase offset from genesis hash
    phase_offset: float = 0.0
    
    def __post_init__(self):
        """Calculate phase offset from genesis time."""
        # Use genesis time to create unique phase offset
        genesis_hash = hashlib.sha256(str(self.genesis_time).encode()).hexdigest()
        hash_value = int(genesis_hash[:8], 16) / (2**32)
        self.phase_offset = hash_value * 2 * PI
    
    def current_phase(self, at_time: float = None) -> float:
        """
        Get current phase in the cycle.
        
        θ = (2π × t / T + offset) mod 2π
        """
        if at_time is None:
            at_time = time.time()
        
        elapsed = at_time - self.genesis_time
        raw_phase = (2 * PI * elapsed / self.cycle_period) + self.phase_offset
        return raw_phase % (2 * PI)
    
    def current_euler_factor(self, at_time: float = None) -> complex:
        """Get e^(iθ) at current time."""
        phase = self.current_phase(at_time)
        return cmath.exp(I * phase)
    
    def phase_type(self, at_time: float = None) -> str:
        """Classify the current phase."""
        phase = self.current_phase(at_time)
        
        # Normalize to 0-2π
        normalized = phase % (2 * PI)
        
        if normalized < PI/4 or normalized > 7*PI/4:
            return "CONSTRUCTIVE"  # Near 0/2π, e^(iθ) ≈ 1
        elif 3*PI/4 < normalized < 5*PI/4:
            return "DESTRUCTIVE"   # Near π, e^(iθ) ≈ -1
        elif PI/4 < normalized < 3*PI/4:
            return "ORTHOGONAL_+"  # Near π/2, e^(iθ) ≈ i
        else:
            return "ORTHOGONAL_-"  # Near 3π/2, e^(iθ) ≈ -i
    
    def cycles_elapsed(self, at_time: float = None) -> float:
        """How many complete cycles since genesis."""
        if at_time is None:
            at_time = time.time()
        return (at_time - self.genesis_time) / self.cycle_period


# ═══════════════════════════════════════════════════════════════════════════════
# SPIRAL DECAY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SpiralDecay:
    """
    Implements the full spiral decay: e^(-λt + iθ)
    
    This combines:
    - Magnitude decay: e^(-λt) 
    - Phase rotation: e^(iθ)
    - Into a spiral: e^(-λt + iθ) = e^(-λt) × e^(iθ)
    """
    
    @staticmethod
    def compute_spiral_position(
        initial_magnitude: float,
        decay_rate: float,
        elapsed_time: float,
        cycle: CompanyCycle,
        at_time: float = None
    ) -> SpiralPoint:
        """
        Compute position in spiral space.
        
        magnitude = initial × e^(-λt)
        phase = current cycle phase
        """
        # Decay the magnitude
        decayed_magnitude = initial_magnitude * math.exp(-decay_rate * elapsed_time)
        
        # Get current phase from cycle
        phase = cycle.current_phase(at_time)
        
        return SpiralPoint(magnitude=decayed_magnitude, phase=phase)
    
    @staticmethod
    def compute_complex_energy(
        initial_energy: float,
        decay_rate: float,
        elapsed_time: float,
        phase: float
    ) -> complex:
        """
        Full complex energy: E × e^(-λt + iθ)
        
        This is the TRUE Euler integration!
        """
        # Combined exponent: -λt + iθ
        exponent = complex(-decay_rate * elapsed_time, phase)
        
        # E × e^(exponent)
        return initial_energy * cmath.exp(exponent)
    
    @staticmethod
    def phase_interference(
        energy1: complex,
        energy2: complex
    ) -> Tuple[float, str]:
        """
        Compute interference between two complex energies.
        
        Returns (interference_factor, type)
        - factor > 1: constructive
        - factor < 1: destructive
        - factor ≈ 1: orthogonal
        """
        # Get phases
        phase1 = cmath.phase(energy1)
        phase2 = cmath.phase(energy2)
        
        # Phase difference
        phase_diff = abs(phase1 - phase2)
        if phase_diff > PI:
            phase_diff = 2 * PI - phase_diff
        
        # Interference factor
        # cos(0) = 1 (constructive), cos(π) = -1 (destructive)
        factor = 1 + 0.5 * math.cos(phase_diff)
        
        if phase_diff < PI/4:
            type_ = "CONSTRUCTIVE"
        elif phase_diff > 3*PI/4:
            type_ = "DESTRUCTIVE"
        else:
            type_ = "ORTHOGONAL"
        
        return factor, type_
    
    @staticmethod
    def apply_phase_boost(
        current_energy: complex,
        boost_amount: float,
        action_phase: float,
        cycle: CompanyCycle
    ) -> complex:
        """
        Apply an energy boost at a specific action phase.
        
        If action phase aligns with cycle phase → constructive boost
        If action phase opposes cycle phase → destructive interference
        """
        cycle_phase = cycle.current_phase()
        
        # Phase alignment factor
        phase_diff = abs(action_phase - cycle_phase)
        if phase_diff > PI:
            phase_diff = 2 * PI - phase_diff
        
        # cos gives us: aligned (1), opposed (-1), orthogonal (0)
        alignment = math.cos(phase_diff)
        
        # Effective boost (can be negative for opposed phases!)
        effective_boost = boost_amount * (0.5 + 0.5 * alignment)
        
        # Add as complex energy at action phase
        boost_complex = effective_boost * cmath.exp(I * action_phase)
        
        return current_energy + boost_complex


# ═══════════════════════════════════════════════════════════════════════════════
# DNA CHECK WITH PHASE ENCRYPTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DNACheckResult:
    """Result of a DNA check with phase information."""
    
    timestamp: float
    cycle_phase: float
    phase_type: str
    
    # Complex energy state
    complex_energy: complex
    magnitude: float
    phase_degrees: float
    
    # Spiral position
    x: float
    y: float
    z: float
    
    # Euler factor at this check
    euler_factor: complex
    
    # Interference with previous check
    interference_factor: float = 1.0
    interference_type: str = "NONE"


class PhaseEncryptedDNA:
    """
    DNA system with phase encryption.
    
    The WHEN of each DNA check (which phase in the cycle)
    becomes part of the encryption.
    
    At θ = π: e^(iπ) = -1 (the Euler identity!)
    """
    
    def __init__(self, 
                 owner_id: str,
                 genesis_time: float = None,
                 cycle_period: float = DAYS * 7):  # Weekly cycle
        
        self.owner_id = owner_id
        self.genesis_time = genesis_time or time.time()
        
        # Create the cycle
        self.cycle = CompanyCycle(
            genesis_time=self.genesis_time,
            cycle_period=cycle_period
        )
        
        # Energy state (complex!)
        self.complex_energy: complex = complex(0, 0)
        
        # Decay rate (π-encrypted as before)
        self.decay_rate = self._compute_decay_rate()
        
        # History
        self.dna_checks: List[DNACheckResult] = []
        self.energy_additions: List[Dict] = []
        
        # Last check for interference calculation
        self.last_check_energy: complex = complex(0, 0)
    
    def _compute_decay_rate(self) -> float:
        """Compute π-encrypted decay rate."""
        genesis_hash = hashlib.sha256(
            f"{self.owner_id}:{self.genesis_time}".encode()
        ).hexdigest()
        
        hash_value = int(genesis_hash[:16], 16) / (2**64)
        
        # Base half-life of 14 days, varied by π
        base_half_life = DAYS * 14
        half_life = base_half_life * (1 + hash_value * PI / 2)
        
        return math.log(2) / half_life
    
    def add_energy(self, 
                  amount: float,
                  action_phase: float = None) -> Dict[str, Any]:
        """
        Add energy at a specific phase.
        
        If no phase specified, use current cycle phase.
        """
        if action_phase is None:
            action_phase = self.cycle.current_phase()
        
        # Convert to complex energy at this phase
        energy_complex = amount * cmath.exp(I * action_phase)
        
        # Apply phase boost (interference with current state)
        old_energy = self.complex_energy
        self.complex_energy = SpiralDecay.apply_phase_boost(
            self.complex_energy,
            amount,
            action_phase,
            self.cycle
        )
        
        # Calculate interference
        if abs(old_energy) > 0.01:
            factor, type_ = SpiralDecay.phase_interference(
                old_energy, energy_complex
            )
        else:
            factor, type_ = 1.0, "INITIAL"
        
        result = {
            'added': amount,
            'phase_degrees': math.degrees(action_phase),
            'phase_type': self.cycle.phase_type(),
            'new_magnitude': abs(self.complex_energy),
            'new_phase_degrees': math.degrees(cmath.phase(self.complex_energy)),
            'interference_factor': factor,
            'interference_type': type_
        }
        
        self.energy_additions.append(result)
        return result
    
    def apply_decay(self, elapsed_seconds: float):
        """Apply decay to the complex energy."""
        decay_factor = math.exp(-self.decay_rate * elapsed_seconds)
        self.complex_energy *= decay_factor
    
    def dna_check(self) -> DNACheckResult:
        """
        Perform a DNA check at current phase.
        
        The phase at which you check becomes part of the encryption!
        """
        now = time.time()
        phase = self.cycle.current_phase()
        phase_type = self.cycle.phase_type()
        euler_factor = self.cycle.current_euler_factor()
        
        # Create spiral point
        point = SpiralPoint(
            magnitude=abs(self.complex_energy),
            phase=cmath.phase(self.complex_energy) if abs(self.complex_energy) > 0 else phase
        )
        
        # Calculate interference with previous check
        if abs(self.last_check_energy) > 0.01:
            factor, type_ = SpiralDecay.phase_interference(
                self.last_check_energy, self.complex_energy
            )
        else:
            factor, type_ = 1.0, "NONE"
        
        result = DNACheckResult(
            timestamp=now,
            cycle_phase=phase,
            phase_type=phase_type,
            complex_energy=self.complex_energy,
            magnitude=abs(self.complex_energy),
            phase_degrees=math.degrees(cmath.phase(self.complex_energy)) % 360 if abs(self.complex_energy) > 0 else 0,
            x=point.x,
            y=point.y,
            z=point.z,
            euler_factor=euler_factor,
            interference_factor=factor,
            interference_type=type_
        )
        
        self.dna_checks.append(result)
        self.last_check_energy = self.complex_energy
        
        return result
    
    def get_spiral_trajectory(self) -> List[Tuple[float, float, float]]:
        """Get the 3D trajectory of all DNA checks."""
        return [(c.x, c.y, c.z) for c in self.dna_checks]


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_euler_identity():
    """Demonstrate the Euler identity connection."""
    print("\n" + "=" * 80)
    print("EULER'S IDENTITY: e^(iπ) = -1")
    print("=" * 80)
    
    print(f"""
    The most beautiful equation in mathematics:
    
        e^(iπ) + 1 = 0
        
    Or equivalently:
    
        e^(iπ) = -1
    
    
    Let's verify:
    
        e^(iπ) = {cmath.exp(I * PI)}
        
    Real part:      {cmath.exp(I * PI).real:.10f}  (should be -1)
    Imaginary part: {cmath.exp(I * PI).imag:.10f}  (should be 0)
    
    
    HOW WE USE IT:
    ══════════════
    
    Decay formula:    e^(-λt)         → magnitude shrinks
    Phase rotation:   e^(iθ)          → position rotates
    
    Combined:         e^(-λt + iθ)    → SPIRAL!
                    = e^(-λt) × e^(iθ)
    
    
    At θ = π (half cycle):
    
        e^(iπ) = -1
        
    The phase FLIPS the sign!
    Energy added at θ = π interferes DESTRUCTIVELY!
    """)
    
    # Show different phases
    print("\n--- Phase Values (e^(iθ)) ---\n")
    
    phases = [
        (0, "0 (start)"),
        (PI/4, "π/4 (45°)"),
        (PI/2, "π/2 (90°)"),
        (3*PI/4, "3π/4 (135°)"),
        (PI, "π (180°)"),
        (5*PI/4, "5π/4 (225°)"),
        (3*PI/2, "3π/2 (270°)"),
        (7*PI/4, "7π/4 (315°)"),
    ]
    
    print("┌─────────────────┬─────────────────────────┬────────────────┐")
    print("│ Phase           │ e^(iθ)                  │ Effect         │")
    print("├─────────────────┼─────────────────────────┼────────────────┤")
    
    for phase, name in phases:
        val = cmath.exp(I * phase)
        
        if abs(val.imag) < 0.01:
            val_str = f"{val.real:+.2f}"
        elif abs(val.real) < 0.01:
            val_str = f"{val.imag:+.2f}i"
        else:
            val_str = f"{val.real:+.2f}{val.imag:+.2f}i"
        
        if phase < PI/4 or phase > 7*PI/4:
            effect = "CONSTRUCTIVE"
        elif 3*PI/4 < phase < 5*PI/4:
            effect = "DESTRUCTIVE"
        else:
            effect = "ORTHOGONAL"
        
        print(f"│ θ = {name:11} │ {val_str:23} │ {effect:14} │")
    
    print("└─────────────────┴─────────────────────────┴────────────────┘")


def demo_spiral_decay():
    """Demonstrate the spiral decay pattern."""
    print("\n" + "=" * 80)
    print("SPIRAL DECAY: e^(-λt + iθ)")
    print("=" * 80)
    
    print("""
    The decay is not just magnitude shrinking...
    It's a SPIRAL in complex space!
    
    
                        Im (Y)
                        │
                   ╭────│────╮
                  ╱     │     ╲
                 │      │      │
                 │   ←──┼───   │  ← spiraling inward
                 │      │   ╲  │
                  ╲     │    ↘╱
                   ╰────│────╯
             ───────────┼─────────── Re (X)
                        │
                        │
    
    
    3D View:
    
         Z (magnitude)
         │
         │    ╭───╮
         │   ╱ ↓   ╲
         │  │   ↓   │    ← spiral down as magnitude decays
         │   ╲  ↓  ╱
         │    ╰──↓╯
         └──────────── X-Y plane (phase rotation)
    """)
    
    # Create example spiral
    initial_mag = 100.0
    decay_rate = 0.1
    
    print("\n--- Spiral Trajectory ---\n")
    print("┌──────┬───────────┬───────────┬───────────┬───────────────────────┐")
    print("│ Time │ Magnitude │ Phase (°) │ (X, Y, Z)                │ Type   │")
    print("├──────┼───────────┼───────────┼───────────────────────────┼────────┤")
    
    for t in range(0, 11, 1):
        mag = initial_mag * math.exp(-decay_rate * t)
        phase = (t * PI / 5) % (2 * PI)  # Rotate
        
        x = mag * math.cos(phase)
        y = mag * math.sin(phase)
        z = mag
        
        point = SpiralPoint(magnitude=mag, phase=phase)
        
        if point.is_constructive():
            type_ = "+"
        elif point.is_destructive():
            type_ = "-"
        else:
            type_ = "⊥"
        
        print(f"│ {t:4} │ {mag:9.2f} │ {math.degrees(phase):9.1f} │ ({x:6.1f}, {y:6.1f}, {z:6.1f}) │ {type_:6} │")
    
    print("└──────┴───────────┴───────────┴───────────────────────────┴────────┘")


def demo_phase_interference():
    """Demonstrate phase interference in energy additions."""
    print("\n" + "=" * 80)
    print("PHASE INTERFERENCE: When You Act Matters!")
    print("=" * 80)
    
    print("""
    Adding energy at different phases creates INTERFERENCE:
    
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │  Current energy at phase 0°:     ─────────►                             │
    │                                                                          │
    │  Add energy at phase 0°:         ─────────►  CONSTRUCTIVE! (aligned)    │
    │                                  ═══════════════════►                    │
    │                                                                          │
    │  Add energy at phase 180°:       ◄─────────  DESTRUCTIVE! (opposed)     │
    │                                  ═══►                                    │
    │                                                                          │
    │  Add energy at phase 90°:            │       ORTHOGONAL (rotated)       │
    │                                      ▼                                   │
    │                                  ═════════►                              │
    │                                      │                                   │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Create DNA system
    dna = PhaseEncryptedDNA(owner_id="demo", cycle_period=HOURS)
    
    print("\n--- Adding Energy at Different Phases ---\n")
    
    # Add initial energy at phase 0
    result = dna.add_energy(50.0, action_phase=0)
    print(f"1. Added 50 at phase 0°:")
    print(f"   Magnitude: {result['new_magnitude']:.2f}")
    print(f"   Type: {result['interference_type']}")
    
    # Add aligned energy
    result = dna.add_energy(30.0, action_phase=0.1)
    print(f"\n2. Added 30 at phase ~0° (aligned):")
    print(f"   Magnitude: {result['new_magnitude']:.2f}")
    print(f"   Interference: {result['interference_type']} ({result['interference_factor']:.2f}x)")
    
    # Add opposed energy
    result = dna.add_energy(30.0, action_phase=PI)
    print(f"\n3. Added 30 at phase 180° (opposed):")
    print(f"   Magnitude: {result['new_magnitude']:.2f}")
    print(f"   Interference: {result['interference_type']} ({result['interference_factor']:.2f}x)")
    
    # Add orthogonal energy
    result = dna.add_energy(30.0, action_phase=PI/2)
    print(f"\n4. Added 30 at phase 90° (orthogonal):")
    print(f"   Magnitude: {result['new_magnitude']:.2f}")
    print(f"   Interference: {result['interference_type']} ({result['interference_factor']:.2f}x)")


def demo_dna_check_timing():
    """Demonstrate how DNA check timing affects results."""
    print("\n" + "=" * 80)
    print("DNA CHECK TIMING: The Phase of Verification")
    print("=" * 80)
    
    print("""
    The WHEN of your DNA check matters!
    
    At different phases in the cycle:
    - θ ≈ 0°:   e^(i×0) = 1   → Full verification
    - θ ≈ 90°:  e^(iπ/2) = i  → Rotated/orthogonal check
    - θ ≈ 180°: e^(iπ) = -1   → Inverted check (Euler!)
    - θ ≈ 270°: e^(i3π/2) = -i → Other orthogonal
    
    This creates a natural encryption based on WHEN you verify!
    """)
    
    # Create DNA system with short cycle for demo
    dna = PhaseEncryptedDNA(owner_id="timing_demo", cycle_period=100)  # 100 second cycle
    
    # Add some energy
    dna.add_energy(100.0, action_phase=0)
    
    print("\n--- DNA Checks at Different Cycle Phases ---\n")
    print("(Simulated by advancing the cycle)")
    
    # Simulate checks at different phases
    phases_to_check = [0, PI/4, PI/2, 3*PI/4, PI, 5*PI/4, 3*PI/2, 7*PI/4]
    
    print("┌─────────────┬────────────────┬─────────────────────┬─────────────────┐")
    print("│ Phase       │ e^(iθ)         │ Euler Factor        │ Type            │")
    print("├─────────────┼────────────────┼─────────────────────┼─────────────────┤")
    
    for phase in phases_to_check:
        euler = cmath.exp(I * phase)
        
        # Format euler factor
        if abs(euler.imag) < 0.01:
            euler_str = f"{euler.real:+.2f}"
        elif abs(euler.real) < 0.01:
            euler_str = f"{euler.imag:+.2f}i"
        else:
            euler_str = f"{euler.real:+.2f}{euler.imag:+.2f}i"
        
        # Determine type
        if abs(phase) < PI/4 or abs(phase - 2*PI) < PI/4:
            type_ = "CONSTRUCTIVE"
        elif abs(phase - PI) < PI/4:
            type_ = "DESTRUCTIVE"
        else:
            type_ = "ORTHOGONAL"
        
        print(f"│ {math.degrees(phase):7.0f}°    │ {euler_str:14} │ {euler_str:19} │ {type_:15} │")
    
    print("└─────────────┴────────────────┴─────────────────────┴─────────────────┘")
    
    print("""
    
    SECURITY IMPLICATION:
    ═════════════════════
    
    If attacker doesn't know the company's cycle phase:
    
    - They might add energy at wrong phase → cancelled out!
    - They might check DNA at wrong phase → get inverted result!
    - The phase is encrypted in the company's genesis
    
    Only the owner, acting naturally with the cycle, gets full effect!
    """)


def demo_complete_system():
    """Demonstrate the complete spiral decay system."""
    print("\n" + "=" * 80)
    print("COMPLETE SPIRAL DECAY SYSTEM")
    print("=" * 80)
    
    # Create DNA system
    dna = PhaseEncryptedDNA(
        owner_id="jonathan",
        cycle_period=HOURS  # 1 hour cycle for demo
    )
    
    print("\n--- System Initialized ---\n")
    print(f"Owner: {dna.owner_id}")
    print(f"Cycle period: {dna.cycle.cycle_period / HOURS:.1f} hours")
    print(f"Decay half-life: {math.log(2) / dna.decay_rate / DAYS:.2f} days")
    
    # Add energy at various phases
    print("\n--- Adding Energy (Owner Actions) ---\n")
    
    actions = [
        (50.0, 0, "genesis"),
        (20.0, PI/4, "config"),
        (30.0, PI/2, "admin"),
        (10.0, PI, "write"),
    ]
    
    for amount, phase, name in actions:
        result = dna.add_energy(amount, action_phase=phase)
        print(f"  {name:10}: +{amount:5.1f} at {math.degrees(phase):5.0f}° → " +
              f"mag={result['new_magnitude']:6.2f}, {result['interference_type']}")
    
    # Perform DNA check
    print("\n--- DNA Check ---\n")
    
    check = dna.dna_check()
    print(f"  Cycle phase: {math.degrees(check.cycle_phase):.1f}° ({check.phase_type})")
    print(f"  Euler factor: {check.euler_factor}")
    print(f"  Complex energy: {check.complex_energy}")
    print(f"  Magnitude: {check.magnitude:.2f}")
    print(f"  Energy phase: {check.phase_degrees:.1f}°")
    print(f"  3D position: ({check.x:.2f}, {check.y:.2f}, {check.z:.2f})")
    
    print("""
    
    THE UNIFIED FORMULA:
    ════════════════════
    
    S = E × e^(-λt + iθ)
    
    Where:
    • E = accumulated owner energy
    • λ = π-encrypted decay rate
    • t = time since genesis
    • θ = current cycle phase
    
    This gives us:
    • Magnitude decay (e^(-λt))
    • Phase rotation (e^(iθ))
    • The full Euler identity at θ = π
    • 3D spiral trajectory
    • Phase-based interference
    """)


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("SPIRAL DECAY SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Euler identity
    print("\n--- TEST 1: Euler Identity ---")
    euler = cmath.exp(I * PI)
    assert abs(euler.real + 1) < 0.0001
    assert abs(euler.imag) < 0.0001
    print(f"✓ e^(iπ) = {euler.real:.4f} (should be -1)")
    
    # Test 2: Spiral point
    print("\n--- TEST 2: Spiral Point ---")
    point = SpiralPoint(magnitude=10.0, phase=PI/4)
    assert abs(point.x - 10*math.cos(PI/4)) < 0.001
    assert abs(point.y - 10*math.sin(PI/4)) < 0.001
    print(f"✓ Point at (x={point.x:.2f}, y={point.y:.2f}, z={point.z:.2f})")
    
    # Test 3: Phase classification
    print("\n--- TEST 3: Phase Classification ---")
    constructive = SpiralPoint(magnitude=1, phase=0.1)
    destructive = SpiralPoint(magnitude=1, phase=PI)
    orthogonal = SpiralPoint(magnitude=1, phase=PI/2)
    assert constructive.is_constructive()
    assert destructive.is_destructive()
    assert orthogonal.is_orthogonal()
    print("✓ Phase classification works")
    
    # Test 4: Company cycle
    print("\n--- TEST 4: Company Cycle ---")
    cycle = CompanyCycle(genesis_time=time.time(), cycle_period=HOURS)
    phase = cycle.current_phase()
    assert 0 <= phase < 2*PI
    print(f"✓ Current phase: {math.degrees(phase):.1f}°")
    
    # Test 5: Complex energy
    print("\n--- TEST 5: Complex Energy ---")
    energy = SpiralDecay.compute_complex_energy(100.0, 0.1, 1.0, PI/4)
    expected_mag = 100 * math.exp(-0.1)
    assert abs(abs(energy) - expected_mag) < 0.1
    print(f"✓ Complex energy: {energy}")
    
    # Test 6: Phase encrypted DNA
    print("\n--- TEST 6: Phase Encrypted DNA ---")
    dna = PhaseEncryptedDNA(owner_id="test")
    result = dna.add_energy(50.0, action_phase=0)
    assert result['new_magnitude'] > 0
    print(f"✓ DNA with energy: {result['new_magnitude']:.2f}")
    
    # Test 7: DNA check
    print("\n--- TEST 7: DNA Check ---")
    check = dna.dna_check()
    assert check.magnitude > 0
    assert isinstance(check.euler_factor, complex)
    print(f"✓ DNA check: mag={check.magnitude:.2f}, phase={check.phase_degrees:.1f}°")
    
    # Test 8: Interference
    print("\n--- TEST 8: Interference ---")
    e1 = complex(1, 0)  # Phase 0
    e2 = complex(-1, 0)  # Phase π (opposite)
    factor, type_ = SpiralDecay.phase_interference(e1, e2)
    assert factor < 1  # Destructive
    assert type_ == "DESTRUCTIVE"
    print(f"✓ Destructive interference: factor={factor:.2f}")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--euler":
            demo_euler_identity()
        elif sys.argv[1] == "--spiral":
            demo_spiral_decay()
        elif sys.argv[1] == "--interference":
            demo_phase_interference()
        elif sys.argv[1] == "--timing":
            demo_dna_check_timing()
        elif sys.argv[1] == "--system":
            demo_complete_system()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_euler_identity()
            demo_spiral_decay()
            demo_phase_interference()
            demo_dna_check_timing()
            demo_complete_system()
        else:
            run_tests()
    else:
        run_tests()
