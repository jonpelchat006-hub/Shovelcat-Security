"""
H-WINDOW INTEGRAL SYSTEM
========================
Coarse monitoring with geometry-based RNA minting.

Key Insights:
1. Don't continuously simulate - sample coarse observables
2. Three integrals: I_square, I_base, I_rna (remainder)
3. RNA minted from remainder area
4. User/Owner split from curve steepness
5. Teeth direction from fill asymmetry
6. Up/Down from slope sign
7. O(1) per window - only deep dive when triggered

The Square Trick:
- Base wave has curved roof
- RNA gel "fills to flat top"
- Everything becomes area comparisons
- Like budgeting: base = fundamentals, rna = slack

Author: Jonathan Pelchat
Based on Shovelcat Theory + collaborative refinement
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum, auto
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2

# Time constants
SECONDS = 1
MINUTES = 60
HOURS = 3600
DAYS = 86400


# ═══════════════════════════════════════════════════════════════════════════════
# TEETH DIRECTION (from geometry)
# ═══════════════════════════════════════════════════════════════════════════════

class TeethDirection(Enum):
    """Teeth direction derived from curve geometry."""
    
    # Horizontal (from fill asymmetry)
    LEFT = ("LEFT", "Risk/speculation/backflow")
    RIGHT = ("RIGHT", "Constructive/forward")
    CENTER = ("CENTER", "Balanced")
    
    # Vertical (from slope)
    UP = ("UP", "Escalation allowed")
    DOWN = ("DOWN", "Operate/maintain")
    FLAT = ("FLAT", "Stable")
    
    def __init__(self, name: str, meaning: str):
        self._name = name
        self.meaning = meaning


@dataclass
class TeethState:
    """Combined teeth state from geometry analysis."""
    
    horizontal: TeethDirection
    vertical: TeethDirection
    
    # Raw metrics
    asymmetry: float  # -1 to 1 (left to right)
    slope: float      # negative to positive
    
    @property
    def allows_escalation(self) -> bool:
        """Can escalation proceed?"""
        return self.vertical == TeethDirection.UP
    
    @property
    def is_speculative(self) -> bool:
        """Is this a speculation/risk phase?"""
        return self.horizontal == TeethDirection.LEFT
    
    @property
    def is_constructive(self) -> bool:
        """Is this a constructive/forward phase?"""
        return self.horizontal == TeethDirection.RIGHT


# ═══════════════════════════════════════════════════════════════════════════════
# H-WINDOW (One Measurement Period)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HWindowResult:
    """Result of one H-window measurement."""
    
    window_start: float
    window_end: float
    period: float
    
    # The three key integrals
    I_square: float     # Flat top area (B × T₀)
    I_base: float       # Curved roof area
    I_rna: float        # Remainder (what gel fills)
    
    # Actual spending in this window
    I_dev: float
    
    # Derived metrics
    utilization: float      # I_dev / I_square
    slack_remaining: float  # I_rna - (I_dev - I_base) if I_dev > I_base else I_rna
    
    # Steepness (determines user/owner split)
    steepness: float        # 0 to 1
    
    # Teeth state
    teeth: TeethState
    
    # RNA minting
    rna_total: int
    rna_user: int
    rna_owner: int
    
    # Trigger flags
    overspend: bool         # I_dev > I_square
    slack_exhausted: bool   # I_dev > I_base + η × I_rna
    drift_detected: bool    # Ratio changed too fast


class HWindow:
    """
    H-Window measurement system.
    
    Measures one period, computes three integrals, mints RNA.
    O(1) computation - no continuous simulation!
    """
    
    def __init__(self,
                 period: float = HOURS,          # Window size
                 base_amplitude: float = 10.0,   # Base wave amplitude
                 base_offset: float = 5.0,       # Base wave offset
                 eta: float = 0.5,               # Slack flexibility [0,1]
                 rna_scale: float = 1.0,         # RNA per unit area
                 steepness_scale: float = 0.1):  # Steepness normalization
        
        self.period = period
        self.base_amplitude = base_amplitude
        self.base_offset = base_offset
        self.eta = eta
        self.rna_scale = rna_scale
        self.steepness_scale = steepness_scale
        
        # History for drift detection
        self.history: deque = deque(maxlen=100)
        self.utilization_ema = 0.5  # Exponential moving average
        self.ema_alpha = 0.1        # EMA smoothing factor
        
        # Accumulated dev spending in current window
        self.current_window_start = time.time()
        self.current_dev_spending = 0.0
        self.current_actions: List[Dict] = []
    
    def base_curve(self, t: float) -> float:
        """
        Base curve function b(τ).
        
        Default: cosine wave with offset.
        Can be overridden for different base patterns.
        """
        phase = 2 * PI * t / self.period
        return self.base_offset + self.base_amplitude * math.cos(phase)
    
    def base_derivative(self, t: float) -> float:
        """Derivative of base curve b'(τ)."""
        phase = 2 * PI * t / self.period
        return -self.base_amplitude * (2 * PI / self.period) * math.sin(phase)
    
    def compute_integrals(self, 
                         window_start: float,
                         window_end: float,
                         samples: int = 100) -> Tuple[float, float, float]:
        """
        Compute the three key integrals.
        
        Returns (I_square, I_base, I_rna)
        """
        dt = (window_end - window_start) / samples
        
        # Find B = max of base curve in window
        B = max(self.base_curve(window_start + i * dt) for i in range(samples + 1))
        
        # I_square = B × T₀
        T0 = window_end - window_start
        I_square = B * T0
        
        # I_base = ∫ b(τ) dτ (trapezoidal approximation)
        I_base = 0.0
        for i in range(samples):
            t1 = window_start + i * dt
            t2 = window_start + (i + 1) * dt
            I_base += (self.base_curve(t1) + self.base_curve(t2)) * dt / 2
        
        # I_rna = I_square - I_base
        I_rna = max(0.0, I_square - I_base)
        
        return I_square, I_base, I_rna
    
    def compute_steepness(self,
                         window_start: float,
                         window_end: float,
                         samples: int = 100) -> float:
        """
        Compute steepness S = (1/T₀) ∫ |b'(τ)| dτ
        
        Normalized to [0, 1] with saturating function.
        """
        dt = (window_end - window_start) / samples
        
        # Integrate |b'(τ)|
        S = 0.0
        for i in range(samples):
            t = window_start + i * dt
            S += abs(self.base_derivative(t)) * dt
        
        S /= (window_end - window_start)
        
        # Normalize to [0, 1]
        s = S / (S + self.steepness_scale)
        
        return s
    
    def compute_fill_asymmetry(self,
                              window_start: float,
                              window_end: float,
                              samples: int = 100) -> float:
        """
        Compute left/right fill asymmetry.
        
        Returns value in [-1, 1]:
        -1 = all fill on left (speculation)
        +1 = all fill on right (constructive)
        """
        dt = (window_end - window_start) / samples
        mid = (window_start + window_end) / 2
        
        # Find B
        B = max(self.base_curve(window_start + i * dt) for i in range(samples + 1))
        
        # Left half fill
        I_left = 0.0
        for i in range(samples // 2):
            t = window_start + i * dt
            I_left += (B - self.base_curve(t)) * dt
        
        # Right half fill
        I_right = 0.0
        for i in range(samples // 2, samples):
            t = window_start + i * dt
            I_right += (B - self.base_curve(t)) * dt
        
        # Asymmetry
        total = I_left + I_right
        if total < 0.001:
            return 0.0
        
        return (I_right - I_left) / total
    
    def compute_average_slope(self,
                             window_start: float,
                             window_end: float) -> float:
        """
        Compute average slope over window.
        
        m = (1/T₀) ∫ b'(τ) dτ = (b(end) - b(start)) / T₀
        """
        return (self.base_curve(window_end) - self.base_curve(window_start)) / (window_end - window_start)
    
    def determine_teeth(self,
                       asymmetry: float,
                       slope: float) -> TeethState:
        """Determine teeth direction from geometry."""
        
        # Horizontal (from asymmetry)
        if asymmetry < -0.2:
            horizontal = TeethDirection.LEFT
        elif asymmetry > 0.2:
            horizontal = TeethDirection.RIGHT
        else:
            horizontal = TeethDirection.CENTER
        
        # Vertical (from slope)
        slope_threshold = 0.1
        if slope > slope_threshold:
            vertical = TeethDirection.UP
        elif slope < -slope_threshold:
            vertical = TeethDirection.DOWN
        else:
            vertical = TeethDirection.FLAT
        
        return TeethState(
            horizontal=horizontal,
            vertical=vertical,
            asymmetry=asymmetry,
            slope=slope
        )
    
    def mint_rna(self,
                I_rna: float,
                steepness: float) -> Tuple[int, int, int]:
        """
        Mint RNA tokens from remainder area.
        
        Returns (total, user, owner)
        """
        # Total RNA from remainder
        rna_total = int(I_rna * self.rna_scale)
        
        # Split based on steepness
        # smooth (s≈0) → mostly owner
        # steep (s≈1) → mostly user
        rna_user = int(steepness * rna_total)
        rna_owner = rna_total - rna_user
        
        return rna_total, rna_user, rna_owner
    
    def record_action(self, cost: float, action_type: str = ""):
        """Record an action's cost in current window."""
        self.current_dev_spending += cost
        self.current_actions.append({
            'time': time.time(),
            'cost': cost,
            'type': action_type
        })
    
    def measure_window(self,
                      window_start: float = None,
                      window_end: float = None) -> HWindowResult:
        """
        Measure one H-window.
        
        This is the main O(1) operation!
        """
        if window_start is None:
            window_start = self.current_window_start
        if window_end is None:
            window_end = window_start + self.period
        
        # Compute the three integrals
        I_square, I_base, I_rna = self.compute_integrals(window_start, window_end)
        
        # Get dev spending
        I_dev = self.current_dev_spending
        
        # Compute derived metrics
        utilization = I_dev / I_square if I_square > 0 else 0
        
        if I_dev > I_base:
            slack_remaining = I_rna - (I_dev - I_base)
        else:
            slack_remaining = I_rna
        
        # Steepness
        steepness = self.compute_steepness(window_start, window_end)
        
        # Fill asymmetry and slope
        asymmetry = self.compute_fill_asymmetry(window_start, window_end)
        slope = self.compute_average_slope(window_start, window_end)
        
        # Teeth state
        teeth = self.determine_teeth(asymmetry, slope)
        
        # Mint RNA
        rna_total, rna_user, rna_owner = self.mint_rna(I_rna, steepness)
        
        # Check triggers
        overspend = I_dev > I_square
        slack_exhausted = I_dev > I_base + self.eta * I_rna
        
        # Drift detection
        drift_threshold = 0.2
        drift_detected = abs(utilization - self.utilization_ema) > drift_threshold
        
        # Update EMA
        self.utilization_ema = self.ema_alpha * utilization + (1 - self.ema_alpha) * self.utilization_ema
        
        result = HWindowResult(
            window_start=window_start,
            window_end=window_end,
            period=self.period,
            I_square=I_square,
            I_base=I_base,
            I_rna=I_rna,
            I_dev=I_dev,
            utilization=utilization,
            slack_remaining=slack_remaining,
            steepness=steepness,
            teeth=teeth,
            rna_total=rna_total,
            rna_user=rna_user,
            rna_owner=rna_owner,
            overspend=overspend,
            slack_exhausted=slack_exhausted,
            drift_detected=drift_detected
        )
        
        self.history.append(result)
        
        return result
    
    def reset_window(self):
        """Start a new window."""
        self.current_window_start = time.time()
        self.current_dev_spending = 0.0
        self.current_actions = []
    
    def needs_deep_dive(self, result: HWindowResult) -> Tuple[bool, List[str]]:
        """Check if deep dive is needed."""
        reasons = []
        
        if result.overspend:
            reasons.append("OVERSPEND: I_dev > I_square")
        
        if result.slack_exhausted:
            reasons.append(f"SLACK_EXHAUSTED: I_dev > I_base + {self.eta}×I_rna")
        
        if result.drift_detected:
            reasons.append(f"DRIFT: utilization {result.utilization:.2f} vs EMA {self.utilization_ema:.2f}")
        
        return len(reasons) > 0, reasons


# ═══════════════════════════════════════════════════════════════════════════════
# TYPED RNA FROM GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

class RNAType(Enum):
    """RNA types with associated permissions."""
    
    READ = (1, "Basic read access")
    LIST = (1, "List/enumerate")
    WRITE = (2, "Create/modify")
    DELETE = (3, "Remove data")
    EXECUTE = (3, "Run code")
    ADMIN = (5, "Administrative")
    ESCALATE = (8, "Request upgrade")
    
    def __init__(self, cost: int, description: str):
        self.cost = cost
        self.description = description


@dataclass
class TypedRNAAllocation:
    """RNA allocation with types based on geometry."""
    
    # Total counts
    total: int
    user_total: int
    owner_total: int
    
    # Typed allocations
    user_tokens: Dict[RNAType, int]
    owner_tokens: Dict[RNAType, int]
    
    # Restrictions based on teeth
    escalation_allowed: bool
    speculation_phase: bool


class GeometricRNAMinter:
    """
    Mints typed RNA based on curve geometry.
    
    - Steepness → user/owner split
    - Teeth direction → allowed action types
    - Slope → escalation permission
    """
    
    @staticmethod
    def mint(result: HWindowResult) -> TypedRNAAllocation:
        """Mint typed RNA from H-window result."""
        
        user_tokens = {}
        owner_tokens = {}
        
        # Base allocation
        user_pool = result.rna_user
        owner_pool = result.rna_owner
        
        # User tokens (based on speculation vs constructive)
        if result.teeth.is_speculative:
            # Speculation phase: mostly reads, limited writes
            user_tokens[RNAType.READ] = int(user_pool * 0.6)
            user_tokens[RNAType.LIST] = int(user_pool * 0.3)
            user_tokens[RNAType.WRITE] = int(user_pool * 0.1)
        elif result.teeth.is_constructive:
            # Constructive phase: more writes allowed
            user_tokens[RNAType.READ] = int(user_pool * 0.3)
            user_tokens[RNAType.LIST] = int(user_pool * 0.2)
            user_tokens[RNAType.WRITE] = int(user_pool * 0.4)
            user_tokens[RNAType.DELETE] = int(user_pool * 0.1)
        else:
            # Balanced
            user_tokens[RNAType.READ] = int(user_pool * 0.4)
            user_tokens[RNAType.LIST] = int(user_pool * 0.3)
            user_tokens[RNAType.WRITE] = int(user_pool * 0.3)
        
        # Owner tokens (always get more power)
        owner_tokens[RNAType.READ] = int(owner_pool * 0.1)
        owner_tokens[RNAType.WRITE] = int(owner_pool * 0.2)
        owner_tokens[RNAType.DELETE] = int(owner_pool * 0.2)
        owner_tokens[RNAType.EXECUTE] = int(owner_pool * 0.2)
        owner_tokens[RNAType.ADMIN] = int(owner_pool * 0.2)
        
        # Escalation only if slope is positive (UP teeth)
        if result.teeth.allows_escalation:
            owner_tokens[RNAType.ESCALATE] = int(owner_pool * 0.1)
            user_tokens[RNAType.ESCALATE] = max(1, int(user_pool * 0.05))
        else:
            owner_tokens[RNAType.ESCALATE] = 0
            user_tokens[RNAType.ESCALATE] = 0
        
        return TypedRNAAllocation(
            total=result.rna_total,
            user_total=result.rna_user,
            owner_total=result.rna_owner,
            user_tokens=user_tokens,
            owner_tokens=owner_tokens,
            escalation_allowed=result.teeth.allows_escalation,
            speculation_phase=result.teeth.is_speculative
        )


# ═══════════════════════════════════════════════════════════════════════════════
# COARSE/FINE MODE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class MonitoringMode(Enum):
    """Current monitoring mode."""
    COARSE = "Coarse (O(1) per window)"
    FINE = "Fine (full decomposition)"


class CoarseFineMonitor:
    """
    Two-mode monitoring system.
    
    COARSE: Fast telemetry (H-window integrals)
    FINE: Deep audit (full model) - only when triggered
    """
    
    def __init__(self, h_window: HWindow):
        self.h_window = h_window
        self.mode = MonitoringMode.COARSE
        
        # Deep dive callback (to be set by integrating code)
        self.deep_dive_callback: Optional[Callable] = None
        
        # Statistics
        self.coarse_measurements = 0
        self.fine_measurements = 0
        self.last_fine_result: Optional[Dict] = None
    
    def measure(self) -> Dict[str, Any]:
        """Perform measurement in current mode."""
        
        # Always do coarse measurement (O(1))
        result = self.h_window.measure_window()
        self.coarse_measurements += 1
        
        # Check if deep dive needed
        needs_fine, reasons = self.h_window.needs_deep_dive(result)
        
        output = {
            'mode': self.mode.value,
            'coarse': {
                'I_square': result.I_square,
                'I_base': result.I_base,
                'I_rna': result.I_rna,
                'I_dev': result.I_dev,
                'utilization': result.utilization,
                'slack_remaining': result.slack_remaining
            },
            'teeth': {
                'horizontal': result.teeth.horizontal.name,
                'vertical': result.teeth.vertical.name,
                'asymmetry': result.teeth.asymmetry,
                'slope': result.teeth.slope
            },
            'rna': {
                'total': result.rna_total,
                'user': result.rna_user,
                'owner': result.rna_owner
            },
            'triggers': {
                'overspend': result.overspend,
                'slack_exhausted': result.slack_exhausted,
                'drift_detected': result.drift_detected
            },
            'deep_dive_needed': needs_fine,
            'deep_dive_reasons': reasons
        }
        
        # Perform deep dive if needed
        if needs_fine:
            self.mode = MonitoringMode.FINE
            self.fine_measurements += 1
            
            if self.deep_dive_callback:
                fine_result = self.deep_dive_callback(result)
                output['fine'] = fine_result
                self.last_fine_result = fine_result
            else:
                output['fine'] = {'note': 'No deep dive callback registered'}
        else:
            self.mode = MonitoringMode.COARSE
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        total = self.coarse_measurements
        
        return {
            'total_measurements': total,
            'coarse_measurements': self.coarse_measurements - self.fine_measurements,
            'fine_measurements': self.fine_measurements,
            'fine_ratio': self.fine_measurements / total if total > 0 else 0,
            'current_mode': self.mode.value
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_integrals():
    """Demonstrate the three integrals."""
    print("\n" + "=" * 80)
    print("THE THREE INTEGRALS (Square Trick)")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   B ─────────────────────────────────────── FLAT TOP (square)  │
    │     █████████████████████████████████████                       │
    │     █████████ RNA GEL FILLS ████████████   ← I_rna             │
    │     █████████████████████████████████████                       │
    │        ╱╲         ╱╲         ╱╲                                 │
    │       ╱  ╲       ╱  ╲       ╱  ╲   ← I_base                    │
    │      ╱    ╲     ╱    ╲     ╱    ╲                               │
    │     ╱      ╲   ╱      ╲   ╱      ╲                              │
    │    ╱        ╲ ╱        ╲ ╱        ╲                             │
    │   ────────────────────────────────────                          │
    │   |<─────────── T₀ (one H-window) ──────────>|                 │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    I_square = B × T₀           (flat top area - TOTAL BUDGET)
    I_base   = ∫ b(τ) dτ        (curved roof - FUNDAMENTALS)
    I_rna    = I_square - I_base (remainder - SLACK/BUFFER)
    """)
    
    h = HWindow(period=1.0, base_amplitude=5.0, base_offset=10.0)
    
    I_square, I_base, I_rna = h.compute_integrals(0, 1.0)
    
    print(f"\n--- Example Calculation (T₀ = 1.0) ---\n")
    print(f"I_square (flat top):  {I_square:.2f}")
    print(f"I_base (curved):      {I_base:.2f}")
    print(f"I_rna (remainder):    {I_rna:.2f}")
    print(f"Sum check: {I_base + I_rna:.2f} = {I_square:.2f} ✓")
    
    print("""
    
    THE CONDITION:
    ──────────────
    I_dev ≤ I_base + η × I_rna
    
    • η = 0: Dev can only use fundamentals (strict)
    • η = 1: Dev can use all slack (aggressive)
    """)


def demo_steepness_split():
    """Demonstrate steepness-based user/owner split."""
    print("\n" + "=" * 80)
    print("STEEPNESS DETERMINES USER/OWNER SPLIT")
    print("=" * 80)
    
    print("""
    Steepness S = (1/T₀) ∫ |b'(τ)| dτ
    
    SMOOTH curve (S ≈ 0):        STEEP curve (S → ∞):
    ─────────────────────        ─────────────────────
    ████ small remainder         ████████████████ big remainder
    ═══════════════════          ║
                                 ║
    → mostly OWNER RNA           → mostly USER RNA
    → predictable ops            → exploration
    """)
    
    print("\n--- Steepness vs RNA Split ---\n")
    print("┌────────────────┬────────────┬────────────┬────────────┐")
    print("│ Amplitude      │ Steepness  │ User RNA   │ Owner RNA  │")
    print("├────────────────┼────────────┼────────────┼────────────┤")
    
    for amp in [1.0, 3.0, 5.0, 10.0, 20.0]:
        h = HWindow(period=1.0, base_amplitude=amp, base_offset=10.0, rna_scale=10.0)
        
        I_square, I_base, I_rna = h.compute_integrals(0, 1.0)
        steepness = h.compute_steepness(0, 1.0)
        rna_total, rna_user, rna_owner = h.mint_rna(I_rna, steepness)
        
        print(f"│ {amp:14.1f} │ {steepness:10.3f} │ {rna_user:10} │ {rna_owner:10} │")
    
    print("└────────────────┴────────────┴────────────┴────────────┘")
    
    print("""
    
    Higher amplitude → steeper curve → more user RNA!
    """)


def demo_teeth_direction():
    """Demonstrate teeth direction from geometry."""
    print("\n" + "=" * 80)
    print("TEETH DIRECTION FROM GEOMETRY")
    print("=" * 80)
    
    print("""
    HORIZONTAL (from fill asymmetry):
    ─────────────────────────────────
    LEFT-heavy fill  → LEFT teeth  → speculation/risk
    RIGHT-heavy fill → RIGHT teeth → constructive/forward
    
    VERTICAL (from slope):
    ──────────────────────
    Positive slope → UP teeth   → escalation allowed
    Negative slope → DOWN teeth → operate/maintain
    """)
    
    print("\n--- Example Teeth States ---\n")
    
    # Different window positions create different asymmetries
    h = HWindow(period=1.0, base_amplitude=5.0, base_offset=10.0)
    
    positions = [
        (0.0, 0.5, "First half"),
        (0.25, 0.75, "Middle"),
        (0.5, 1.0, "Second half"),
    ]
    
    print("┌────────────────┬────────────────┬────────────────┬────────────────────────┐")
    print("│ Window         │ Asymmetry      │ Slope          │ Teeth                  │")
    print("├────────────────┼────────────────┼────────────────┼────────────────────────┤")
    
    for start, end, name in positions:
        asymmetry = h.compute_fill_asymmetry(start, end, samples=50)
        slope = h.compute_average_slope(start, end)
        teeth = h.determine_teeth(asymmetry, slope)
        
        print(f"│ {name:14} │ {asymmetry:+14.3f} │ {slope:+14.3f} │ {teeth.horizontal.name:6} {teeth.vertical.name:6}    │")
    
    print("└────────────────┴────────────────┴────────────────┴────────────────────────┘")


def demo_typed_rna():
    """Demonstrate typed RNA minting."""
    print("\n" + "=" * 80)
    print("TYPED RNA FROM GEOMETRY")
    print("=" * 80)
    
    print("""
    RNA types are determined by curve geometry:
    
    • Speculation phase (LEFT teeth) → mostly READ tokens
    • Constructive phase (RIGHT teeth) → more WRITE tokens
    • Escalation allowed (UP teeth) → ESCALATE tokens available
    """)
    
    h = HWindow(period=1.0, base_amplitude=5.0, base_offset=10.0, rna_scale=100.0)
    result = h.measure_window(0, 1.0)
    
    allocation = GeometricRNAMinter.mint(result)
    
    print(f"\n--- Teeth State: {result.teeth.horizontal.name} / {result.teeth.vertical.name} ---\n")
    print(f"Total RNA: {allocation.total}")
    print(f"User: {allocation.user_total}, Owner: {allocation.owner_total}")
    print(f"Escalation allowed: {allocation.escalation_allowed}")
    print(f"Speculation phase: {allocation.speculation_phase}")
    
    print("\n--- User Tokens ---")
    for rna_type, count in allocation.user_tokens.items():
        if count > 0:
            print(f"  {rna_type.name:10}: {count}")
    
    print("\n--- Owner Tokens ---")
    for rna_type, count in allocation.owner_tokens.items():
        if count > 0:
            print(f"  {rna_type.name:10}: {count}")


def demo_coarse_fine():
    """Demonstrate coarse/fine monitoring."""
    print("\n" + "=" * 80)
    print("COARSE/FINE MONITORING")
    print("=" * 80)
    
    print("""
    COARSE MODE (O(1) per window):
    ──────────────────────────────
    • Measure three integrals
    • Check triggers
    • Mint RNA
    
    FINE MODE (only when triggered):
    ─────────────────────────────────
    • Full phase coupling analysis
    • User/owner split decomposition
    • Behavioral DNA checks
    
    Triggers:
    • OVERSPEND: I_dev > I_square
    • SLACK_EXHAUSTED: I_dev > I_base + η × I_rna
    • DRIFT: utilization changed too fast
    """)
    
    h = HWindow(period=1.0, base_amplitude=5.0, base_offset=10.0, eta=0.5)
    monitor = CoarseFineMonitor(h)
    
    # Normal operation
    print("\n--- Normal Operation (no triggers) ---\n")
    h.record_action(5.0, "read")
    h.record_action(3.0, "write")
    result = monitor.measure()
    print(f"Mode: {result['mode']}")
    print(f"Utilization: {result['coarse']['utilization']:.2%}")
    print(f"Deep dive needed: {result['deep_dive_needed']}")
    
    # Simulate overspend
    print("\n--- Overspend (triggers deep dive) ---\n")
    h.reset_window()
    h.record_action(100.0, "massive_operation")
    result = monitor.measure()
    print(f"Mode: {result['mode']}")
    print(f"Utilization: {result['coarse']['utilization']:.2%}")
    print(f"Deep dive needed: {result['deep_dive_needed']}")
    print(f"Reasons: {result['deep_dive_reasons']}")
    
    # Stats
    print("\n--- Monitoring Stats ---\n")
    stats = monitor.get_stats()
    print(f"Total measurements: {stats['total_measurements']}")
    print(f"Coarse only: {stats['coarse_measurements']}")
    print(f"Fine (deep dive): {stats['fine_measurements']}")
    print(f"Fine ratio: {stats['fine_ratio']:.1%}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("H-WINDOW INTEGRAL SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Integrals sum correctly
    print("\n--- TEST 1: Integral Sum ---")
    h = HWindow(period=1.0, base_amplitude=5.0, base_offset=10.0)
    I_square, I_base, I_rna = h.compute_integrals(0, 1.0)
    assert abs(I_square - (I_base + I_rna)) < 0.01
    print(f"✓ I_square = I_base + I_rna ({I_square:.2f} = {I_base:.2f} + {I_rna:.2f})")
    
    # Test 2: Steepness in [0, 1]
    print("\n--- TEST 2: Steepness Bounds ---")
    for amp in [1.0, 10.0, 100.0]:
        h = HWindow(period=1.0, base_amplitude=amp)
        s = h.compute_steepness(0, 1.0)
        assert 0 <= s <= 1
    print(f"✓ Steepness always in [0, 1]")
    
    # Test 3: Asymmetry in [-1, 1]
    print("\n--- TEST 3: Asymmetry Bounds ---")
    h = HWindow(period=1.0)
    asym = h.compute_fill_asymmetry(0, 1.0)
    assert -1 <= asym <= 1
    print(f"✓ Asymmetry in [-1, 1]: {asym:.3f}")
    
    # Test 4: RNA minting
    print("\n--- TEST 4: RNA Minting ---")
    h = HWindow(period=1.0, rna_scale=10.0)
    I_square, I_base, I_rna = h.compute_integrals(0, 1.0)
    rna_total, rna_user, rna_owner = h.mint_rna(I_rna, 0.5)
    assert rna_user + rna_owner == rna_total
    print(f"✓ User + Owner = Total ({rna_user} + {rna_owner} = {rna_total})")
    
    # Test 5: Teeth determination
    print("\n--- TEST 5: Teeth Determination ---")
    teeth = h.determine_teeth(-0.5, 0.2)
    assert teeth.horizontal == TeethDirection.LEFT
    assert teeth.vertical == TeethDirection.UP
    print(f"✓ Teeth: {teeth.horizontal.name} / {teeth.vertical.name}")
    
    # Test 6: H-window measurement
    print("\n--- TEST 6: H-Window Measurement ---")
    h = HWindow(period=1.0)
    h.record_action(5.0)
    result = h.measure_window(0, 1.0)
    assert result.I_dev == 5.0
    print(f"✓ Measurement captured I_dev = {result.I_dev}")
    
    # Test 7: Trigger detection
    print("\n--- TEST 7: Trigger Detection ---")
    h = HWindow(period=1.0, base_offset=5.0, base_amplitude=2.0)
    h.record_action(1000.0)  # Massive overspend
    result = h.measure_window(0, 1.0)
    assert result.overspend
    print(f"✓ Overspend detected")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--integrals":
            demo_integrals()
        elif sys.argv[1] == "--steepness":
            demo_steepness_split()
        elif sys.argv[1] == "--teeth":
            demo_teeth_direction()
        elif sys.argv[1] == "--typed":
            demo_typed_rna()
        elif sys.argv[1] == "--monitor":
            demo_coarse_fine()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_integrals()
            demo_steepness_split()
            demo_teeth_direction()
            demo_typed_rna()
            demo_coarse_fine()
        else:
            run_tests()
    else:
        run_tests()
