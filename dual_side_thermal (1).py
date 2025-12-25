"""
DUAL-SIDE THERMAL SYSTEM
========================
User and Owner sides separated by a solid wall with radiation and pipe transfer.

Key Insights:
1. Energy split using half-angle: w_user = cos²(θ/2), w_owner = sin²(θ/2)
2. Transfer via RADIATION (noisy leaks) or PIPE (intentional release)
3. Phase coupling: ρ=0 → anti-phase (rumors), ρ=1 → phase-locked (truth)
4. Two-way flow: owner→user (releases) AND user→owner (feedback)

The Euler Connection:
- At ρ=0: φ_user = φ_owner + π → e^(iπ) = -1 (speculation is opposite of reality)
- At ρ=1: φ_user = φ_owner → e^(i×0) = 1 (shared ground truth)

Real-World Mapping:
- Radiation: rumors, datamining, leaks, vibes, steam charts
- Pipe: patch notes, trailers, releases, reviews, bug reports

Author: Jonathan Pelchat
Based on Shovelcat Theory + collaborative refinement
"""

import math
import cmath
import time
import secrets
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

E = math.e
PI = math.pi
PHI = (1 + math.sqrt(5)) / 2
I = complex(0, 1)

# Time constants
SECONDS = 1
MINUTES = 60
HOURS = 3600
DAYS = 86400


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class TransferType(Enum):
    """Types of energy/information transfer through the wall."""
    
    # Radiation (noisy, leaky, speculative)
    RUMOR = ("RUMOR", 0.3, 0.8, "Unverified speculation")
    LEAK = ("LEAK", 0.5, 0.6, "Unintentional information release")
    DATAMINE = ("DATAMINE", 0.7, 0.4, "Extracted from code/assets")
    VIBE = ("VIBE", 0.1, 0.9, "General feeling/sentiment")
    
    # Pipe (intentional, high-signal)
    PATCH_NOTES = ("PATCH_NOTES", 0.9, 0.1, "Official changelog")
    TRAILER = ("TRAILER", 0.8, 0.2, "Marketing material")
    RELEASE = ("RELEASE", 1.0, 0.0, "Actual product release")
    DEV_POST = ("DEV_POST", 0.85, 0.15, "Developer communication")
    ROADMAP = ("ROADMAP", 0.7, 0.3, "Future plans")
    
    # Feedback (user → owner)
    PLAYER_COUNT = ("PLAYER_COUNT", 0.6, 0.2, "Active users metric")
    REVIEW = ("REVIEW", 0.8, 0.3, "User evaluation")
    BUG_REPORT = ("BUG_REPORT", 0.9, 0.1, "Technical feedback")
    CREATOR_VIDEO = ("CREATOR_VIDEO", 0.7, 0.4, "Content creator coverage")
    
    def __init__(self, name: str, signal: float, noise: float, description: str):
        self._name = name
        self.signal = signal  # How much truth gets through
        self.noise = noise    # How much distortion
        self.description = description
    
    @property
    def is_radiation(self) -> bool:
        """Is this a radiation (noisy) transfer?"""
        return self.noise > 0.5
    
    @property
    def is_pipe(self) -> bool:
        """Is this a pipe (intentional) transfer?"""
        return self.signal > 0.7 and self.noise < 0.5
    
    @property
    def is_feedback(self) -> bool:
        """Is this user→owner feedback?"""
        return self in [TransferType.PLAYER_COUNT, TransferType.REVIEW, 
                       TransferType.BUG_REPORT, TransferType.CREATOR_VIDEO]


# ═══════════════════════════════════════════════════════════════════════════════
# HALF-ANGLE ENERGY SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

class EnergySplit:
    """
    Split total energy between user and owner sides using half-angle formula.
    
    w_user(θ) = cos²(θ/2)
    w_owner(θ) = sin²(θ/2)
    
    These ALWAYS sum to 1 (energy conservation)!
    """
    
    @staticmethod
    def user_weight(theta: float) -> float:
        """Weight for user side: cos²(θ/2)"""
        return math.cos(theta / 2) ** 2
    
    @staticmethod
    def owner_weight(theta: float) -> float:
        """Weight for owner side: sin²(θ/2)"""
        return math.sin(theta / 2) ** 2
    
    @staticmethod
    def split(total_energy: complex, theta: float) -> Tuple[complex, complex]:
        """
        Split total energy into user and owner portions.
        
        Returns (E_user, E_owner)
        """
        magnitude = abs(total_energy)
        phase = cmath.phase(total_energy)
        
        w_user = EnergySplit.user_weight(theta)
        w_owner = EnergySplit.owner_weight(theta)
        
        # Split magnitude, preserve phase
        e_user = w_user * magnitude * cmath.exp(I * phase)
        e_owner = w_owner * magnitude * cmath.exp(I * phase)
        
        return e_user, e_owner
    
    @staticmethod
    def verify_conservation(e_user: complex, e_owner: complex, 
                           total: complex, tolerance: float = 0.001) -> bool:
        """Verify energy is conserved."""
        return abs(abs(e_user) + abs(e_owner) - abs(total)) < tolerance


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE COUPLING
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseCoupling:
    """
    Manages phase relationship between user and owner.
    
    φ_user = φ_owner + π × (1 - ρ)
    
    Where ρ is "connection strength" [0, 1]:
    - ρ ≈ 0: only rumors → ~π out of phase (pure speculation)
    - ρ ≈ 1: direct pipe → phase locked (shared reality)
    """
    
    def __init__(self, initial_rho: float = 0.1):
        self.rho = initial_rho  # Connection strength
        self.rho_history: deque = deque(maxlen=100)
    
    def phase_offset(self) -> float:
        """Get current phase offset: π × (1 - ρ)"""
        return PI * (1 - self.rho)
    
    def user_phase(self, owner_phase: float) -> float:
        """Calculate user phase from owner phase."""
        return (owner_phase + self.phase_offset()) % (2 * PI)
    
    def phase_difference(self, owner_phase: float) -> float:
        """Get phase difference in degrees."""
        return math.degrees(self.phase_offset())
    
    def euler_factor(self) -> complex:
        """Get the e^(i×offset) factor."""
        return cmath.exp(I * self.phase_offset())
    
    def adjust_connection(self, transfer_type: TransferType, amount: float):
        """
        Adjust connection strength based on transfer type.
        
        Pipe transfers INCREASE ρ (toward phase lock)
        Radiation transfers DECREASE ρ (toward anti-phase)
        """
        old_rho = self.rho
        
        if transfer_type.is_pipe:
            # Pipe increases connection (toward shared truth)
            self.rho = min(1.0, self.rho + amount * transfer_type.signal * 0.1)
        elif transfer_type.is_radiation:
            # Radiation decreases connection (toward speculation)
            self.rho = max(0.0, self.rho - amount * transfer_type.noise * 0.05)
        
        self.rho_history.append({
            'time': time.time(),
            'old_rho': old_rho,
            'new_rho': self.rho,
            'transfer': transfer_type.name,
            'amount': amount
        })
    
    def connection_status(self) -> str:
        """Get human-readable connection status."""
        if self.rho > 0.8:
            return "PHASE_LOCKED (shared reality)"
        elif self.rho > 0.5:
            return "CONNECTED (mostly aligned)"
        elif self.rho > 0.2:
            return "DECOUPLED (significant speculation)"
        else:
            return "ANTI_PHASE (pure rumor land)"


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER EVENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferEvent:
    """A single transfer of energy through the wall."""
    
    timestamp: float
    transfer_type: TransferType
    direction: str  # "owner_to_user" or "user_to_owner"
    
    # Energy transferred
    intended_amount: float
    actual_signal: float   # What got through clearly
    noise_added: float     # Distortion/speculation
    
    # Phase effects
    phase_shift: float
    
    @property
    def total_transferred(self) -> float:
        return self.actual_signal + self.noise_added
    
    @property
    def signal_to_noise(self) -> float:
        if self.noise_added < 0.001:
            return float('inf')
        return self.actual_signal / self.noise_added


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-SIDE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SideState:
    """State of one side (user or owner)."""
    
    name: str
    energy: complex = complex(0, 0)
    temperature: float = 1.0
    phase: float = 0.0
    
    @property
    def magnitude(self) -> float:
        return abs(self.energy)
    
    @property
    def phase_degrees(self) -> float:
        if self.magnitude < 0.001:
            return 0.0
        return math.degrees(cmath.phase(self.energy)) % 360


class DualSideThermal:
    """
    Complete dual-side thermal system with wall, radiation, and pipe.
    
    Two sides:
    - OWNER (furnace): where development/creation happens
    - USER (community): where consumption/feedback happens
    
    Energy flows through:
    - RADIATION: noisy, leaky (rumors, leaks, vibes)
    - PIPE: intentional, clear (releases, patch notes)
    
    Phase coupling:
    - ρ=0: anti-phase (e^(iπ) = -1, speculation opposite of reality)
    - ρ=1: phase-locked (e^(i×0) = 1, shared ground truth)
    """
    
    def __init__(self,
                 total_energy: float = 100.0,
                 initial_theta: float = PI/2,  # 50-50 split
                 initial_rho: float = 0.1):    # Start in rumor land
        
        # Theta determines energy split
        self.theta = initial_theta
        
        # Total system energy
        self.total_energy = complex(total_energy, 0)
        
        # Split into sides
        e_user, e_owner = EnergySplit.split(self.total_energy, self.theta)
        
        self.owner = SideState(name="OWNER", energy=e_owner)
        self.user = SideState(name="USER", energy=e_user)
        
        # Phase coupling
        self.coupling = PhaseCoupling(initial_rho=initial_rho)
        
        # Transfer history
        self.transfers: List[TransferEvent] = []
        
        # Radiation and pipe coefficients
        self.kappa_r = 0.1   # Radiation coefficient
        self.kappa_p = 0.3   # Pipe coefficient
        
        # Base temperatures
        self.T_base_owner = 1.5  # Furnace runs hot
        self.T_base_user = 0.8   # Community cooler
    
    def transfer(self, 
                transfer_type: TransferType,
                amount: float,
                direction: str = "owner_to_user") -> TransferEvent:
        """
        Transfer energy through the wall.
        
        direction: "owner_to_user" or "user_to_owner"
        """
        # Determine source and destination
        if direction == "owner_to_user":
            source = self.owner
            dest = self.user
        else:
            source = self.user
            dest = self.owner
        
        # Calculate actual transfer based on type
        signal = amount * transfer_type.signal
        noise = amount * transfer_type.noise * random.uniform(0.5, 1.5)
        
        # Transfer energy (conserved!)
        transfer_magnitude = signal + noise
        
        if transfer_magnitude > source.magnitude:
            transfer_magnitude = source.magnitude * 0.9
            signal = transfer_magnitude * transfer_type.signal
            noise = transfer_magnitude * transfer_type.noise
        
        # Create complex transfer with source phase
        source_phase = cmath.phase(source.energy) if abs(source.energy) > 0.01 else 0
        transfer_complex = transfer_magnitude * cmath.exp(I * source_phase)
        
        # Apply phase noise for radiation
        if transfer_type.is_radiation:
            # Radiation adds phase noise (distortion)
            phase_noise = random.uniform(-PI/4, PI/4) * transfer_type.noise
            transfer_complex *= cmath.exp(I * phase_noise)
        
        # Move energy
        source.energy -= transfer_complex
        dest.energy += transfer_complex
        
        # Calculate phase shift
        phase_shift = cmath.phase(transfer_complex) - source_phase
        
        # Update phase coupling
        self.coupling.adjust_connection(transfer_type, amount)
        
        # Update temperatures
        self._update_temperatures()
        
        # Record event
        event = TransferEvent(
            timestamp=time.time(),
            transfer_type=transfer_type,
            direction=direction,
            intended_amount=amount,
            actual_signal=signal,
            noise_added=noise,
            phase_shift=phase_shift
        )
        
        self.transfers.append(event)
        
        return event
    
    def _update_temperatures(self):
        """Update side temperatures based on energy and phase."""
        # Owner temperature (furnace)
        self.owner.temperature = self.T_base_owner + abs(self.owner.energy) * 0.01
        self.owner.phase = cmath.phase(self.owner.energy) if abs(self.owner.energy) > 0.01 else 0
        
        # User temperature (community) - affected by coupling
        coupling_factor = self.coupling.rho
        user_phase = self.coupling.user_phase(self.owner.phase)
        
        self.user.temperature = self.T_base_user + abs(self.user.energy) * 0.01
        self.user.phase = user_phase
    
    def radiate(self, amount: float = 1.0, 
               radiation_type: TransferType = TransferType.RUMOR) -> TransferEvent:
        """
        Passive radiation from owner to user (leaks, rumors).
        
        R(t) = κ_r × E_owner × η(t)
        """
        actual_amount = self.kappa_r * abs(self.owner.energy) * amount
        return self.transfer(radiation_type, actual_amount, "owner_to_user")
    
    def open_pipe(self, amount: float = 1.0,
                 pipe_type: TransferType = TransferType.PATCH_NOTES) -> TransferEvent:
        """
        Intentional transfer from owner to user (release, patch notes).
        
        P(t) = κ_p × E_owner × u(t)
        """
        actual_amount = self.kappa_p * abs(self.owner.energy) * amount
        return self.transfer(pipe_type, actual_amount, "owner_to_user")
    
    def receive_feedback(self, amount: float = 1.0,
                        feedback_type: TransferType = TransferType.REVIEW) -> TransferEvent:
        """
        Receive feedback from user to owner.
        """
        actual_amount = self.kappa_p * abs(self.user.energy) * amount
        return self.transfer(feedback_type, actual_amount, "user_to_owner")
    
    def add_owner_energy(self, amount: float):
        """Add energy to owner side (development work)."""
        self.owner.energy += amount
        self.total_energy += amount
        self._update_temperatures()
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        return {
            'total_energy': abs(self.total_energy),
            'theta_degrees': math.degrees(self.theta),
            
            'owner': {
                'energy': abs(self.owner.energy),
                'temperature': self.owner.temperature,
                'phase_degrees': self.owner.phase_degrees
            },
            
            'user': {
                'energy': abs(self.user.energy),
                'temperature': self.user.temperature,
                'phase_degrees': self.user.phase_degrees
            },
            
            'coupling': {
                'rho': self.coupling.rho,
                'phase_offset_degrees': self.coupling.phase_difference(self.owner.phase),
                'status': self.coupling.connection_status(),
                'euler_factor': self.coupling.euler_factor()
            },
            
            'transfers': len(self.transfers),
            
            'conservation_check': EnergySplit.verify_conservation(
                self.user.energy, self.owner.energy, self.total_energy
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GAME DEVELOPMENT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class GameDevSimulation:
    """
    Simulate a game development cycle using dual-side thermal model.
    
    Phases:
    1. Pre-announcement: Owner builds energy, minimal transfer
    2. Announcement: First pipe opened (trailer)
    3. Development: Radiation (leaks) + occasional pipes (dev posts)
    4. Pre-launch: Hype building, more radiation
    5. Launch: Big pipe opened (release)
    6. Post-launch: Feedback flows back, connection strengthens
    """
    
    def __init__(self):
        self.system = DualSideThermal(
            total_energy=100.0,
            initial_theta=3*PI/4,  # 75% owner-side at start
            initial_rho=0.05       # Almost pure speculation
        )
        
        self.phase = "pre_announcement"
        self.day = 0
        self.log: List[Dict] = []
    
    def simulate_day(self) -> Dict[str, Any]:
        """Simulate one day of development."""
        self.day += 1
        events = []
        
        # Owner always adds some energy (development work)
        dev_work = random.uniform(2, 8)
        self.system.add_owner_energy(dev_work)
        events.append(f"Dev work: +{dev_work:.1f} energy")
        
        # Phase-specific activities
        if self.phase == "pre_announcement":
            # Minimal leaks, building up
            if random.random() < 0.1:
                e = self.system.radiate(0.2, TransferType.RUMOR)
                events.append(f"Rumor leaked: {e.actual_signal:.2f} signal")
            
            if self.day > 30:
                self.phase = "announcement"
        
        elif self.phase == "announcement":
            # Big trailer drop!
            e = self.system.open_pipe(2.0, TransferType.TRAILER)
            events.append(f"TRAILER: {e.actual_signal:.2f} signal, ρ now {self.system.coupling.rho:.2f}")
            self.phase = "development"
        
        elif self.phase == "development":
            # Regular leaks + occasional dev posts
            if random.random() < 0.3:
                e = self.system.radiate(0.5, TransferType.DATAMINE)
                events.append(f"Datamine: {e.actual_signal:.2f} signal")
            
            if random.random() < 0.15:
                e = self.system.open_pipe(1.0, TransferType.DEV_POST)
                events.append(f"Dev post: {e.actual_signal:.2f} signal")
            
            if self.day > 90:
                self.phase = "pre_launch"
        
        elif self.phase == "pre_launch":
            # More leaks, hype building
            if random.random() < 0.5:
                e = self.system.radiate(1.0, TransferType.LEAK)
                events.append(f"Leak: {e.actual_signal:.2f} signal")
            
            e = self.system.open_pipe(1.5, TransferType.PATCH_NOTES)
            events.append(f"Patch notes: {e.actual_signal:.2f} signal")
            
            if self.day > 100:
                self.phase = "launch"
        
        elif self.phase == "launch":
            # BIG RELEASE
            e = self.system.open_pipe(5.0, TransferType.RELEASE)
            events.append(f"RELEASE: {e.actual_signal:.2f} signal, ρ now {self.system.coupling.rho:.2f}")
            self.phase = "post_launch"
        
        elif self.phase == "post_launch":
            # Feedback flows back
            if random.random() < 0.7:
                e = self.system.receive_feedback(1.0, TransferType.REVIEW)
                events.append(f"Review: {e.actual_signal:.2f} feedback")
            
            if random.random() < 0.4:
                e = self.system.receive_feedback(0.5, TransferType.BUG_REPORT)
                events.append(f"Bug report: {e.actual_signal:.2f} feedback")
            
            # Ongoing patches
            if random.random() < 0.2:
                e = self.system.open_pipe(1.0, TransferType.PATCH_NOTES)
                events.append(f"Patch: {e.actual_signal:.2f} signal")
        
        state = self.system.get_state()
        
        result = {
            'day': self.day,
            'phase': self.phase,
            'events': events,
            'state': state
        }
        
        self.log.append(result)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_energy_split():
    """Demonstrate the half-angle energy split."""
    print("\n" + "=" * 80)
    print("HALF-ANGLE ENERGY SPLIT")
    print("=" * 80)
    
    print("""
    Energy split using half-angle formulas:
    
    w_user(θ)  = cos²(θ/2)  → user side weight
    w_owner(θ) = sin²(θ/2)  → owner side weight
    
    These ALWAYS sum to 1 (energy conservation)!
    """)
    
    print("\n┌──────────────┬─────────────┬──────────────┬───────────────┐")
    print("│ θ (degrees)  │ w_user      │ w_owner      │ Sum           │")
    print("├──────────────┼─────────────┼──────────────┼───────────────┤")
    
    for theta_deg in [0, 30, 45, 60, 90, 120, 150, 180]:
        theta = math.radians(theta_deg)
        w_u = EnergySplit.user_weight(theta)
        w_o = EnergySplit.owner_weight(theta)
        
        print(f"│ {theta_deg:12} │ {w_u:11.3f} │ {w_o:12.3f} │ {w_u + w_o:13.3f} │")
    
    print("└──────────────┴─────────────┴──────────────┴───────────────┘")
    
    print("""
    
    At θ = 0°:   ALL energy on user side (100% public)
    At θ = 90°:  Equal split (50-50)
    At θ = 180°: ALL energy on owner side (100% private)
    """)


def demo_phase_coupling():
    """Demonstrate phase coupling between sides."""
    print("\n" + "=" * 80)
    print("PHASE COUPLING: From Rumors to Shared Truth")
    print("=" * 80)
    
    print("""
    φ_user = φ_owner + π × (1 - ρ)
    
    Where ρ = connection strength [0, 1]
    
    At ρ = 0: φ_user = φ_owner + π (ANTI-PHASE!)
              Community believes OPPOSITE of reality
              e^(iπ) = -1 (Euler identity!)
    
    At ρ = 1: φ_user = φ_owner (PHASE-LOCKED!)
              Community knows truth
              e^(i×0) = 1
    """)
    
    print("\n┌──────────┬────────────────┬────────────────┬─────────────────────────┐")
    print("│ ρ        │ Phase Offset   │ e^(i×offset)   │ Status                  │")
    print("├──────────┼────────────────┼────────────────┼─────────────────────────┤")
    
    for rho in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        coupling = PhaseCoupling(initial_rho=rho)
        offset = coupling.phase_offset()
        euler = coupling.euler_factor()
        status = coupling.connection_status()
        
        if abs(euler.imag) < 0.01:
            euler_str = f"{euler.real:+.2f}"
        else:
            euler_str = f"{euler.real:+.2f}{euler.imag:+.2f}i"
        
        print(f"│ {rho:8.2f} │ {math.degrees(offset):12.1f}° │ {euler_str:14} │ {status:23} │")
    
    print("└──────────┴────────────────┴────────────────┴─────────────────────────┘")


def demo_transfer_types():
    """Demonstrate different transfer types."""
    print("\n" + "=" * 80)
    print("TRANSFER TYPES: Radiation vs Pipe")
    print("=" * 80)
    
    print("""
    RADIATION (noisy, leaky):        PIPE (intentional, clear):
    ─────────────────────────        ─────────────────────────
    • Rumors                         • Patch notes
    • Leaks                          • Trailers
    • Datamines                      • Releases
    • Vibes                          • Dev posts
    
    Signal = how much truth gets through
    Noise = how much distortion/speculation
    """)
    
    print("\n┌────────────────┬─────────┬─────────┬──────────┬─────────────────────────┐")
    print("│ Type           │ Signal  │ Noise   │ Category │ Description             │")
    print("├────────────────┼─────────┼─────────┼──────────┼─────────────────────────┤")
    
    for tt in TransferType:
        if tt.is_radiation:
            cat = "RADIATION"
        elif tt.is_pipe:
            cat = "PIPE"
        elif tt.is_feedback:
            cat = "FEEDBACK"
        else:
            cat = "OTHER"
        
        print(f"│ {tt.name:14} │ {tt.signal:7.2f} │ {tt.noise:7.2f} │ {cat:8} │ {tt.description:23} │")
    
    print("└────────────────┴─────────┴─────────┴──────────┴─────────────────────────┘")


def demo_dual_side_system():
    """Demonstrate the complete dual-side system."""
    print("\n" + "=" * 80)
    print("DUAL-SIDE THERMAL SYSTEM")
    print("=" * 80)
    
    system = DualSideThermal(
        total_energy=100.0,
        initial_theta=PI/2,  # 50-50 split
        initial_rho=0.1      # Mostly rumors
    )
    
    print("\n--- Initial State ---\n")
    state = system.get_state()
    print(f"Total energy: {state['total_energy']:.2f}")
    print(f"Owner energy: {state['owner']['energy']:.2f}, T={state['owner']['temperature']:.2f}")
    print(f"User energy: {state['user']['energy']:.2f}, T={state['user']['temperature']:.2f}")
    print(f"Coupling: ρ={state['coupling']['rho']:.2f}, {state['coupling']['status']}")
    
    # Add development work
    print("\n--- Add Development Work ---\n")
    system.add_owner_energy(50)
    print(f"Added 50 energy to owner side")
    print(f"Owner energy now: {abs(system.owner.energy):.2f}")
    
    # Radiate some rumors
    print("\n--- Rumors Spread ---\n")
    for _ in range(3):
        e = system.radiate(1.0, TransferType.RUMOR)
        print(f"Rumor: signal={e.actual_signal:.2f}, noise={e.noise_added:.2f}, S/N={e.signal_to_noise:.2f}")
    
    state = system.get_state()
    print(f"\nCoupling now: ρ={state['coupling']['rho']:.2f}, {state['coupling']['status']}")
    
    # Open a pipe (release something official)
    print("\n--- Official Announcement (Trailer) ---\n")
    e = system.open_pipe(3.0, TransferType.TRAILER)
    print(f"Trailer: signal={e.actual_signal:.2f}, noise={e.noise_added:.2f}")
    
    state = system.get_state()
    print(f"Coupling now: ρ={state['coupling']['rho']:.2f}, {state['coupling']['status']}")
    print(f"Phase offset: {state['coupling']['phase_offset_degrees']:.1f}°")
    
    # Big release
    print("\n--- RELEASE DAY ---\n")
    e = system.open_pipe(5.0, TransferType.RELEASE)
    print(f"RELEASE: signal={e.actual_signal:.2f}, noise={e.noise_added:.2f}")
    
    state = system.get_state()
    print(f"Coupling now: ρ={state['coupling']['rho']:.2f}, {state['coupling']['status']}")
    print(f"Phase offset: {state['coupling']['phase_offset_degrees']:.1f}°")
    print(f"Euler factor: {state['coupling']['euler_factor']}")
    
    # Receive feedback
    print("\n--- Feedback Flows Back ---\n")
    for _ in range(5):
        e = system.receive_feedback(1.0, TransferType.REVIEW)
        print(f"Review received: signal={e.actual_signal:.2f}")
    
    e = system.receive_feedback(2.0, TransferType.BUG_REPORT)
    print(f"Bug report: signal={e.actual_signal:.2f}")
    
    state = system.get_state()
    print(f"\n--- Final State ---")
    print(f"Owner energy: {state['owner']['energy']:.2f}")
    print(f"User energy: {state['user']['energy']:.2f}")
    print(f"Coupling: ρ={state['coupling']['rho']:.2f}, {state['coupling']['status']}")
    print(f"Conservation check: {state['conservation_check']}")


def demo_game_simulation():
    """Demonstrate game development simulation."""
    print("\n" + "=" * 80)
    print("GAME DEVELOPMENT SIMULATION")
    print("=" * 80)
    
    print("""
    Simulating 120 days of game development:
    
    Day 1-30:    Pre-announcement (building in secret)
    Day 31:      Announcement (trailer drops!)
    Day 32-90:   Development (leaks + dev posts)
    Day 91-100:  Pre-launch (hype building)
    Day 101:     LAUNCH
    Day 102+:    Post-launch (feedback flows)
    """)
    
    sim = GameDevSimulation()
    
    # Run simulation
    milestones = [1, 30, 31, 60, 90, 100, 101, 110, 120]
    
    print("\n┌───────┬──────────────────┬────────────┬────────────┬────────┬──────────────────────┐")
    print("│ Day   │ Phase            │ Owner E    │ User E     │ ρ      │ Status               │")
    print("├───────┼──────────────────┼────────────┼────────────┼────────┼──────────────────────┤")
    
    for day in range(1, 121):
        result = sim.simulate_day()
        
        if day in milestones:
            s = result['state']
            print(f"│ {day:5} │ {result['phase']:16} │ {s['owner']['energy']:10.1f} │ {s['user']['energy']:10.1f} │ {s['coupling']['rho']:6.2f} │ {s['coupling']['status'][:20]:20} │")
    
    print("└───────┴──────────────────┴────────────┴────────────┴────────┴──────────────────────┘")
    
    print("\n--- Key Observations ---")
    print(f"Total transfers: {len(sim.system.transfers)}")
    print(f"Final coupling: {sim.system.coupling.connection_status()}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("DUAL-SIDE THERMAL SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Energy split conservation
    print("\n--- TEST 1: Energy Split Conservation ---")
    for theta in [0, PI/4, PI/2, 3*PI/4, PI]:
        w_u = EnergySplit.user_weight(theta)
        w_o = EnergySplit.owner_weight(theta)
        assert abs(w_u + w_o - 1.0) < 0.0001
    print("✓ Weights always sum to 1")
    
    # Test 2: Phase coupling at extremes
    print("\n--- TEST 2: Phase Coupling Extremes ---")
    coupling_0 = PhaseCoupling(initial_rho=0)
    coupling_1 = PhaseCoupling(initial_rho=1)
    assert abs(coupling_0.phase_offset() - PI) < 0.001
    assert abs(coupling_1.phase_offset()) < 0.001
    print(f"✓ ρ=0 → π offset, ρ=1 → 0 offset")
    
    # Test 3: Euler factors
    print("\n--- TEST 3: Euler Factors ---")
    euler_0 = PhaseCoupling(initial_rho=0).euler_factor()
    euler_1 = PhaseCoupling(initial_rho=1).euler_factor()
    assert abs(euler_0 + 1) < 0.001  # e^(iπ) = -1
    assert abs(euler_1 - 1) < 0.001  # e^(i×0) = 1
    print(f"✓ ρ=0 → e^(iπ)={euler_0:.2f}, ρ=1 → e^(i×0)={euler_1:.2f}")
    
    # Test 4: Transfer types
    print("\n--- TEST 4: Transfer Types ---")
    assert TransferType.RUMOR.is_radiation
    assert TransferType.RELEASE.is_pipe
    assert TransferType.REVIEW.is_feedback
    print("✓ Transfer type classification works")
    
    # Test 5: Dual-side system
    print("\n--- TEST 5: Dual-Side System ---")
    system = DualSideThermal(total_energy=100, initial_theta=PI/2)
    assert abs(abs(system.owner.energy) + abs(system.user.energy) - 100) < 1
    print("✓ Energy conserved in system")
    
    # Test 6: Transfer
    print("\n--- TEST 6: Transfer ---")
    initial_owner = abs(system.owner.energy)
    system.radiate(1.0, TransferType.RUMOR)
    assert abs(system.owner.energy) < initial_owner
    print("✓ Radiation reduces owner energy")
    
    # Test 7: Coupling adjustment
    print("\n--- TEST 7: Coupling Adjustment ---")
    system2 = DualSideThermal(total_energy=100, initial_rho=0.1)
    old_rho = system2.coupling.rho
    system2.open_pipe(2.0, TransferType.RELEASE)
    assert system2.coupling.rho > old_rho
    print(f"✓ Pipe increased ρ from {old_rho:.2f} to {system2.coupling.rho:.2f}")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--split":
            demo_energy_split()
        elif sys.argv[1] == "--coupling":
            demo_phase_coupling()
        elif sys.argv[1] == "--transfer":
            demo_transfer_types()
        elif sys.argv[1] == "--system":
            demo_dual_side_system()
        elif sys.argv[1] == "--game":
            demo_game_simulation()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_energy_split()
            demo_phase_coupling()
            demo_transfer_types()
            demo_dual_side_system()
            demo_game_simulation()
        else:
            run_tests()
    else:
        run_tests()
