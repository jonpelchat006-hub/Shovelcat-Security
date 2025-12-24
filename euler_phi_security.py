"""
EULER-PHI COMBINED SECURITY
===========================
Two orthogonal security dimensions:
1. Euler (Phase): e^(iθ) - Opposition, where are you on the circle?
2. Phi (Depth): φⁿ - Nesting, how deep is your verification?

Combined identity: S = e^(iθ) × φⁿ

Key insight:
- Euler gives PHASE (user ↔ owner opposition)
- Phi gives MAGNITUDE (verification depth)
- Hackers can fake phase but not magnitude!
- Each φ level CONTAINS all previous levels (fractal)

Golden ratio property: 1/φ + 1 = φ
- Part + Unity = Whole
- Each level = previous level + one new verification

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
PHI_INVERSE = 1 / PHI         # ≈ 0.618033988749895

# Verify the golden ratio identity
assert abs((PHI_INVERSE + 1) - PHI) < 1e-10, "1/φ + 1 should equal φ"

# Fibonacci sequence (for verification counts)
def fibonacci(n: int) -> int:
    """Get nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ═══════════════════════════════════════════════════════════════════════════════
# PHI DEPTH LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

class PhiDepth(Enum):
    """
    Verification depth levels based on φ powers.
    Each level CONTAINS all previous levels plus one new verification.
    """
    SURFACE = 0      # φ⁰ = 1.000 - Current identity only
    SHALLOW = 1      # φ¹ = 1.618 - + History
    MEDIUM = 2       # φ² = 2.618 - + Cross-domain
    DEEP = 3         # φ³ = 4.236 - + BEC formation
    PROFOUND = 4     # φ⁴ = 6.854 - + Organizational embedding
    FOUNDATIONAL = 5 # φ⁵ = 11.09 - + Structural integration
    
    @property
    def phi_power(self) -> float:
        """φⁿ for this depth."""
        return PHI ** self.value
    
    @property
    def complexity(self) -> float:
        """Verification complexity at this depth."""
        return self.phi_power
    
    @property
    def fibonacci_checks(self) -> int:
        """Number of Fibonacci-structured checks needed."""
        return fibonacci(self.value + 2) - 1
    
    @property
    def contains_previous(self) -> bool:
        """Does this level contain all previous?"""
        return self.value > 0
    
    @classmethod
    def from_magnitude(cls, magnitude: float) -> 'PhiDepth':
        """Get depth level from magnitude."""
        for depth in reversed(cls):
            if magnitude >= depth.phi_power * 0.9:  # 10% tolerance
                return depth
        return cls.SURFACE


# ═══════════════════════════════════════════════════════════════════════════════
# NESTED VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NestedVerification:
    """
    A verification at a specific depth level.
    Contains reference to inner (previous) verification.
    """
    depth: PhiDepth
    verification_type: str
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)
    inner: Optional['NestedVerification'] = None  # Previous level
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Is this verification complete including all inner levels?"""
        if not self.passed:
            return False
        if self.depth.value == 0:
            return True
        if self.inner is None:
            return False
        return self.inner.is_complete
    
    @property
    def total_verifications(self) -> int:
        """Count all verifications in the chain."""
        count = 1 if self.passed else 0
        if self.inner:
            count += self.inner.total_verifications
        return count
    
    def get_chain(self) -> List['NestedVerification']:
        """Get the full verification chain."""
        chain = [self]
        if self.inner:
            chain.extend(self.inner.get_chain())
        return chain
    
    def visualize(self) -> str:
        """ASCII visualization of nested structure."""
        lines = []
        chain = self.get_chain()
        
        for i, v in enumerate(chain):
            indent = "  " * i
            status = "✓" if v.passed else "✗"
            lines.append(f"{indent}[{status}] Depth {v.depth.value} ({v.depth.name}): {v.verification_type}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# EULER-PHI SECURITY POSITION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SecurityPosition:
    """
    Combined Euler-Phi security position.
    
    S = e^(iθ) × φⁿ
    
    - θ: Phase angle (Euler dimension)
    - n: Depth level (Phi dimension)
    """
    theta: float  # Phase in radians
    depth: PhiDepth
    user_id: str
    verification_chain: Optional[NestedVerification] = None
    
    @property
    def euler_component(self) -> complex:
        """e^(iθ) - the phase component."""
        return cmath.exp(1j * self.theta)
    
    @property
    def phi_component(self) -> float:
        """φⁿ - the depth component."""
        return self.depth.phi_power
    
    @property
    def full_position(self) -> complex:
        """S = e^(iθ) × φⁿ"""
        return self.euler_component * self.phi_component
    
    @property
    def magnitude(self) -> float:
        """Magnitude |S| = φⁿ (depth determines magnitude)."""
        return abs(self.full_position)
    
    @property
    def phase(self) -> float:
        """Phase angle of S (should equal θ)."""
        return cmath.phase(self.full_position)
    
    @property
    def real_part(self) -> float:
        """Real component = cos(θ) × φⁿ"""
        return self.full_position.real
    
    @property
    def imag_part(self) -> float:
        """Imaginary component = sin(θ) × φⁿ"""
        return self.full_position.imag
    
    @property
    def theta_degrees(self) -> float:
        """Phase in degrees."""
        return math.degrees(self.theta)
    
    def is_consistent(self) -> Tuple[bool, str]:
        """
        Check if position is internally consistent.
        Magnitude should match depth, phase should match permissions.
        """
        # Check magnitude matches claimed depth
        expected_magnitude = self.depth.phi_power
        actual_magnitude = self.magnitude
        
        if abs(actual_magnitude - expected_magnitude) > 0.1:
            return False, f"Magnitude mismatch: claimed {expected_magnitude:.2f}, actual {actual_magnitude:.2f}"
        
        # Check verification chain exists and is complete
        if self.depth.value > 0 and not self.verification_chain:
            return False, "Depth > 0 but no verification chain"
        
        if self.verification_chain and not self.verification_chain.is_complete:
            return False, "Verification chain incomplete"
        
        return True, "Position is consistent"
    
    def __str__(self):
        return f"S[{self.user_id}] = e^(i·{self.theta_degrees:.0f}°) × φ^{self.depth.value} = {self.full_position:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY POSITION VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class EulerPhiVerifier:
    """
    Verifies security positions using both Euler and Phi dimensions.
    """
    
    # Verification types for each depth level
    DEPTH_VERIFICATIONS = {
        PhiDepth.SURFACE: "current_identity",
        PhiDepth.SHALLOW: "history_verification",
        PhiDepth.MEDIUM: "cross_domain_check",
        PhiDepth.DEEP: "bec_formation",
        PhiDepth.PROFOUND: "organizational_embedding",
        PhiDepth.FOUNDATIONAL: "structural_integration",
    }
    
    def __init__(self):
        self.positions: Dict[str, SecurityPosition] = {}
    
    def create_position(self, user_id: str, theta: float = 0, 
                       depth: PhiDepth = PhiDepth.SURFACE) -> SecurityPosition:
        """Create a new security position."""
        position = SecurityPosition(
            theta=theta,
            depth=depth,
            user_id=user_id
        )
        self.positions[user_id] = position
        return position
    
    def build_verification_chain(self, user_id: str, 
                                 target_depth: PhiDepth,
                                 verifications: Dict[PhiDepth, bool]) -> NestedVerification:
        """
        Build a nested verification chain up to target depth.
        Each level must contain all previous levels.
        """
        if target_depth.value == 0:
            return NestedVerification(
                depth=PhiDepth.SURFACE,
                verification_type=self.DEPTH_VERIFICATIONS[PhiDepth.SURFACE],
                passed=verifications.get(PhiDepth.SURFACE, False)
            )
        
        # Build from inside out (recursively)
        inner_depth = PhiDepth(target_depth.value - 1)
        inner_chain = self.build_verification_chain(user_id, inner_depth, verifications)
        
        return NestedVerification(
            depth=target_depth,
            verification_type=self.DEPTH_VERIFICATIONS[target_depth],
            passed=verifications.get(target_depth, False),
            inner=inner_chain
        )
    
    def verify_position(self, user_id: str, 
                       claimed_theta: float,
                       claimed_depth: PhiDepth) -> Dict[str, Any]:
        """
        Verify a claimed security position.
        Checks both phase (Euler) and depth (Phi).
        """
        position = self.positions.get(user_id)
        
        result = {
            'user_id': user_id,
            'claimed_theta': math.degrees(claimed_theta),
            'claimed_depth': claimed_depth.name,
            'claimed_position': cmath.exp(1j * claimed_theta) * claimed_depth.phi_power,
            'verified': False,
            'checks': {}
        }
        
        if not position:
            result['error'] = "Unknown user"
            return result
        
        # Check 1: Phase consistency
        phase_diff = abs(claimed_theta - position.theta)
        phase_ok = phase_diff < 0.1  # Small tolerance
        result['checks']['phase'] = {
            'claimed': math.degrees(claimed_theta),
            'actual': math.degrees(position.theta),
            'difference': math.degrees(phase_diff),
            'passed': phase_ok
        }
        
        # Check 2: Depth consistency
        depth_ok = claimed_depth.value <= position.depth.value
        result['checks']['depth'] = {
            'claimed': claimed_depth.name,
            'actual': position.depth.name,
            'claimed_magnitude': claimed_depth.phi_power,
            'actual_magnitude': position.magnitude,
            'passed': depth_ok
        }
        
        # Check 3: Verification chain
        chain_ok = True
        if position.verification_chain:
            chain_ok = position.verification_chain.is_complete
            result['checks']['chain'] = {
                'exists': True,
                'complete': chain_ok,
                'total_verifications': position.verification_chain.total_verifications,
                'required': claimed_depth.fibonacci_checks
            }
        else:
            chain_ok = claimed_depth == PhiDepth.SURFACE
            result['checks']['chain'] = {
                'exists': False,
                'complete': chain_ok,
                'reason': "No chain (OK for surface level)"
            }
        
        # Check 4: Magnitude verification
        claimed_magnitude = claimed_depth.phi_power
        actual_magnitude = position.magnitude
        magnitude_ok = abs(claimed_magnitude - actual_magnitude) < 0.5
        result['checks']['magnitude'] = {
            'claimed': claimed_magnitude,
            'actual': actual_magnitude,
            'passed': magnitude_ok
        }
        
        # Overall result
        result['verified'] = phase_ok and depth_ok and chain_ok and magnitude_ok
        
        if not result['verified']:
            failures = [k for k, v in result['checks'].items() if not v.get('passed', True)]
            result['failure_reason'] = f"Failed checks: {failures}"
        
        return result
    
    def detect_hacker(self, user_id: str,
                     claimed_theta: float,
                     claimed_depth: PhiDepth) -> Dict[str, Any]:
        """
        Detect if someone is faking their position.
        Hackers can fake phase but struggle with magnitude/depth.
        """
        result = {
            'user_id': user_id,
            'is_hacker': False,
            'signals': []
        }
        
        position = self.positions.get(user_id)
        if not position:
            result['is_hacker'] = True
            result['signals'].append("Unknown user claiming position")
            return result
        
        # Signal 1: Claiming depth without chain
        if claimed_depth.value > 0 and not position.verification_chain:
            result['signals'].append(f"Claims depth {claimed_depth.name} but no verification chain")
            result['is_hacker'] = True
        
        # Signal 2: Magnitude mismatch
        if position.magnitude < claimed_depth.phi_power * 0.8:
            result['signals'].append(
                f"Magnitude {position.magnitude:.2f} too low for claimed depth {claimed_depth.phi_power:.2f}"
            )
            result['is_hacker'] = True
        
        # Signal 3: Incomplete chain for claimed depth
        if position.verification_chain:
            actual_chain_depth = position.verification_chain.depth.value
            if actual_chain_depth < claimed_depth.value:
                result['signals'].append(
                    f"Chain depth {actual_chain_depth} < claimed depth {claimed_depth.value}"
                )
                result['is_hacker'] = True
        
        # Signal 4: Phase jump (claiming owner phase without depth)
        if claimed_theta > PI * 0.8 and claimed_depth.value < 3:
            result['signals'].append(
                f"Claims near-owner phase {math.degrees(claimed_theta):.0f}° but shallow depth"
            )
            result['is_hacker'] = True
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON: EULER ONLY vs EULER+PHI
# ═══════════════════════════════════════════════════════════════════════════════

def compare_security_models():
    """
    Demonstrate why Euler+Phi is more secure than Euler alone.
    """
    print("\n" + "=" * 80)
    print("SECURITY MODEL COMPARISON: Euler Only vs Euler + Phi")
    print("=" * 80)
    
    print("""
    
    THE PROBLEM WITH EULER ONLY:
    ════════════════════════════
    
    Hacker claims owner position: e^(iπ) = -1
    Real owner position:          e^(iπ) = -1
    
    IDENTICAL! Can't distinguish by phase alone.
    
    
    WITH EULER + PHI:
    ═════════════════
    
    Hacker claims owner: e^(iπ) × φ⁰ = -1 × 1 = -1
                         (correct phase, but no depth)
    
    Real owner:          e^(iπ) × φ⁴ = -1 × 6.854 = -6.854
                         (correct phase AND depth)
    
    |S_hacker| = 1
    |S_owner|  = 6.854
    
    DIFFERENT MAGNITUDES! Depth reveals the fake.
    """)
    
    # Demonstrate with actual positions
    verifier = EulerPhiVerifier()
    
    # Real owner - built up verification over time
    owner = verifier.create_position("real_owner", theta=PI, depth=PhiDepth.PROFOUND)
    owner.verification_chain = verifier.build_verification_chain(
        "real_owner", 
        PhiDepth.PROFOUND,
        {d: True for d in PhiDepth if d.value <= PhiDepth.PROFOUND.value}
    )
    
    # Hacker - claims owner but only surface verification
    hacker = verifier.create_position("hacker", theta=PI, depth=PhiDepth.SURFACE)
    # No verification chain!
    
    print(f"\n--- REAL OWNER ---")
    print(f"Position: {owner}")
    print(f"Magnitude: |S| = {owner.magnitude:.3f}")
    print(f"Verification chain:")
    print(owner.verification_chain.visualize())
    
    print(f"\n--- HACKER ---")
    print(f"Position: {hacker}")
    print(f"Magnitude: |S| = {hacker.magnitude:.3f}")
    print(f"Verification chain: None")
    
    print(f"\n--- VERIFICATION ATTEMPTS ---")
    
    # Hacker tries to claim owner position
    hacker_claim = verifier.verify_position(
        "hacker",
        claimed_theta=PI,
        claimed_depth=PhiDepth.PROFOUND  # Claims deep!
    )
    
    print(f"\nHacker claiming PROFOUND depth:")
    print(f"  Verified: {hacker_claim['verified']}")
    for check, result in hacker_claim['checks'].items():
        passed = result.get('passed', 'N/A')
        print(f"  {check}: {'✓' if passed else '✗'} {result}")
    
    # Detect hacker
    detection = verifier.detect_hacker("hacker", PI, PhiDepth.PROFOUND)
    print(f"\nHacker Detection:")
    print(f"  Is Hacker: {detection['is_hacker']}")
    print(f"  Signals: {detection['signals']}")
    
    # Real owner verification
    owner_claim = verifier.verify_position(
        "real_owner",
        claimed_theta=PI,
        claimed_depth=PhiDepth.PROFOUND
    )
    
    print(f"\nReal owner claiming PROFOUND depth:")
    print(f"  Verified: {owner_claim['verified']}")
    for check, result in owner_claim['checks'].items():
        passed = result.get('passed', 'N/A')
        print(f"  {check}: {'✓' if passed else '✗'}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_phi_properties():
    """Demonstrate the golden ratio properties."""
    print("\n" + "=" * 80)
    print("GOLDEN RATIO PROPERTIES")
    print("=" * 80)
    
    print(f"""
    φ = (1 + √5) / 2 = {PHI:.10f}
    
    THE MAGICAL IDENTITY:
    
        1/φ + 1 = φ
        
        {PHI_INVERSE:.6f} + 1 = {PHI_INVERSE + 1:.6f}
        
        ✓ Verified!
    
    
    PHI POWERS (Verification Complexity):
    
        φ⁰ = {PHI**0:.3f}  (SURFACE)
        φ¹ = {PHI**1:.3f}  (SHALLOW)
        φ² = {PHI**2:.3f}  (MEDIUM)
        φ³ = {PHI**3:.3f}  (DEEP)
        φ⁴ = {PHI**4:.3f}  (PROFOUND)
        φ⁵ = {PHI**5:.3f} (FOUNDATIONAL)
    
    
    FIBONACCI PROPERTY: φⁿ = φⁿ⁻¹ + φⁿ⁻²
    
        φ² = φ¹ + φ⁰ → {PHI**2:.3f} = {PHI**1:.3f} + {PHI**0:.3f} = {PHI**1 + PHI**0:.3f} ✓
        φ³ = φ² + φ¹ → {PHI**3:.3f} = {PHI**2:.3f} + {PHI**1:.3f} = {PHI**2 + PHI**1:.3f} ✓
        φ⁴ = φ³ + φ² → {PHI**4:.3f} = {PHI**3:.3f} + {PHI**2:.3f} = {PHI**3 + PHI**2:.3f} ✓
    
    
    FIBONACCI VERIFICATION COUNTS:
    
        Depth 0: {fibonacci(2) - 1} verification  (surface)
        Depth 1: {fibonacci(3) - 1} verifications (shallow)
        Depth 2: {fibonacci(4) - 1} verifications (medium)
        Depth 3: {fibonacci(5) - 1} verifications (deep)
        Depth 4: {fibonacci(6) - 1} verifications (profound)
        Depth 5: {fibonacci(7) - 1} verifications (foundational)
    """)


def demo_nested_verification():
    """Demonstrate nested verification structure."""
    print("\n" + "=" * 80)
    print("NESTED VERIFICATION STRUCTURE")
    print("=" * 80)
    
    verifier = EulerPhiVerifier()
    
    # Build a complete chain to DEEP level
    chain = verifier.build_verification_chain(
        "test_user",
        PhiDepth.DEEP,
        {
            PhiDepth.SURFACE: True,
            PhiDepth.SHALLOW: True,
            PhiDepth.MEDIUM: True,
            PhiDepth.DEEP: True
        }
    )
    
    print("\nComplete verification chain to DEEP level:")
    print(chain.visualize())
    print(f"\nTotal verifications: {chain.total_verifications}")
    print(f"Chain complete: {chain.is_complete}")
    
    # Show what happens with incomplete chain
    incomplete_chain = verifier.build_verification_chain(
        "incomplete_user",
        PhiDepth.DEEP,
        {
            PhiDepth.SURFACE: True,
            PhiDepth.SHALLOW: True,
            PhiDepth.MEDIUM: False,  # Failed!
            PhiDepth.DEEP: True
        }
    )
    
    print("\n\nIncomplete chain (failed at MEDIUM):")
    print(incomplete_chain.visualize())
    print(f"\nChain complete: {incomplete_chain.is_complete}")
    print("Even though DEEP passed, chain is incomplete because MEDIUM failed!")


def demo_security_positions():
    """Demonstrate various security positions."""
    print("\n" + "=" * 80)
    print("SECURITY POSITIONS: S = e^(iθ) × φⁿ")
    print("=" * 80)
    
    positions = [
        ("New User", 0, PhiDepth.SURFACE),
        ("Registered", PI/12, PhiDepth.SURFACE),
        ("Paid User", PI/6, PhiDepth.SHALLOW),
        ("Long-term", PI/4, PhiDepth.MEDIUM),
        ("Core User", PI/3, PhiDepth.MEDIUM),
        ("Creator", PI/2, PhiDepth.DEEP),
        ("Staff", 3*PI/4, PhiDepth.DEEP),
        ("Admin", 5*PI/6, PhiDepth.PROFOUND),
        ("Owner", PI, PhiDepth.PROFOUND),
    ]
    
    print("\n┌──────────────┬────────┬─────────────┬───────────────────────────┬───────────┐")
    print("│ Role         │ θ      │ Depth       │ S = e^(iθ) × φⁿ           │ |S|       │")
    print("├──────────────┼────────┼─────────────┼───────────────────────────┼───────────┤")
    
    for name, theta, depth in positions:
        pos = SecurityPosition(theta=theta, depth=depth, user_id=name)
        s = pos.full_position
        print(f"│ {name:12} │ {math.degrees(theta):5.0f}° │ {depth.name:11} │ {s.real:+8.3f} {s.imag:+8.3f}i │ {pos.magnitude:9.3f} │")
    
    print("└──────────────┴────────┴─────────────┴───────────────────────────┴───────────┘")
    
    print("""
    
    OBSERVATIONS:
    ─────────────
    
    • New User:   S = 1 (pure real, positive, surface)
    • Creator:    S = 4.236i (pure imaginary! at the boundary)
    • Owner:      S = -6.854 (pure real, negative, deep)
    
    The MAGNITUDE grows with depth (φⁿ)
    The PHASE rotates with role (0 to π)
    
    Creator at θ = π/2 is PURE IMAGINARY
    - Between user world (real) and staff world (negative real)
    - The boundary is literally on the imaginary axis!
    """)


def demo_phase_circle_with_depth():
    """Visualize the combined Euler-Phi space."""
    print("\n" + "=" * 80)
    print("EULER-PHI SECURITY SPACE")
    print("=" * 80)
    
    print("""
    
    3D VIEW: Phase (horizontal) × Depth (vertical)
    ══════════════════════════════════════════════
    
    
           |S| (Magnitude = φⁿ)
            │
            │
       φ⁴ ──┼────────────────────────●───────────── OWNER (θ=π, n=4)
      6.85  │                      ╱               |S| = -6.854
            │                    ╱
            │                  ╱
       φ³ ──┼────────────────●─────────────────── STAFF (θ=3π/4, n=3)
      4.24  │              ╱│                     |S| = -3.0 + 3.0i
            │            ╱  │
            │     CREATOR   │
       φ² ──┼────●──────────┼──────────────────── LONG-TERM (θ=π/4, n=2)
      2.62  │    │ (θ=π/2)  │                     |S| = 1.85 + 1.85i
            │    │   |      │
            │    │   |      │
       φ¹ ──┼────┼───|──────┼──────────────────── PAID (θ=π/6, n=1)
      1.62  │    │   |      │                     |S| = 1.4 + 0.81i
            │    │   |      │
            │    │   |      │
       φ⁰ ──●────┼───|──────┼──────────────────── NEW USER (θ=0, n=0)
      1.00  │    │   |      │                     |S| = 1
            │    │   |      │
    ────────┴────┴───┴──────┴────────────────────────────►
          θ=0  π/4  π/2   3π/4    π               Phase (θ)
         USER        │         STAFF              
         SIDE        │         SIDE               
                     │                            
                  BOUNDARY                        
                (pure imag)                       
    
    
    HACKER POSITION:
    ────────────────
    
    Hacker claims (θ=π, n=0):
    
       φ⁴ ──┼─────────────────────────────────────  ← Real owner here
            │                        
            │                        
       φ⁰ ──┼─────────────────────────●───────────  ← Hacker stuck here!
            │                        |S| = -1
    ────────┴────────────────────────────────────►
          θ=0                        θ=π          
    
    The hacker has correct PHASE but wrong MAGNITUDE!
    """)


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("EULER-PHI SECURITY - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Golden ratio identity
    print("\n--- TEST 1: Golden Ratio Identity ---")
    assert abs((1/PHI + 1) - PHI) < 1e-10
    print(f"✓ 1/φ + 1 = φ verified ({PHI_INVERSE:.6f} + 1 = {PHI:.6f})")
    
    # Test 2: Fibonacci property
    print("\n--- TEST 2: Fibonacci Property ---")
    for n in range(2, 6):
        assert abs(PHI**n - (PHI**(n-1) + PHI**(n-2))) < 1e-10
    print("✓ φⁿ = φⁿ⁻¹ + φⁿ⁻² verified for n=2..5")
    
    # Test 3: Security position
    print("\n--- TEST 3: Security Position ---")
    pos = SecurityPosition(theta=PI/2, depth=PhiDepth.DEEP, user_id="test")
    assert abs(pos.magnitude - PHI**3) < 0.01
    assert abs(pos.real_part) < 0.01  # Should be ~0 at π/2
    print(f"✓ Creator position at π/2: S = {pos.full_position:.3f}")
    print(f"  Pure imaginary as expected (real ≈ 0)")
    
    # Test 4: Nested verification
    print("\n--- TEST 4: Nested Verification ---")
    verifier = EulerPhiVerifier()
    chain = verifier.build_verification_chain(
        "test", PhiDepth.MEDIUM, 
        {PhiDepth.SURFACE: True, PhiDepth.SHALLOW: True, PhiDepth.MEDIUM: True}
    )
    assert chain.is_complete
    assert chain.total_verifications == 3
    print("✓ Complete chain builds correctly")
    
    incomplete = verifier.build_verification_chain(
        "test", PhiDepth.MEDIUM,
        {PhiDepth.SURFACE: True, PhiDepth.SHALLOW: False, PhiDepth.MEDIUM: True}
    )
    assert not incomplete.is_complete
    print("✓ Incomplete chain detected correctly")
    
    # Test 5: Hacker detection
    print("\n--- TEST 5: Hacker Detection ---")
    verifier2 = EulerPhiVerifier()
    hacker = verifier2.create_position("hacker", theta=PI, depth=PhiDepth.SURFACE)
    detection = verifier2.detect_hacker("hacker", PI, PhiDepth.PROFOUND)
    assert detection['is_hacker']
    print(f"✓ Hacker detected: {detection['signals']}")
    
    # Test 6: Legitimate verification
    print("\n--- TEST 6: Legitimate Verification ---")
    owner = verifier2.create_position("owner", theta=PI, depth=PhiDepth.PROFOUND)
    owner.verification_chain = verifier2.build_verification_chain(
        "owner", PhiDepth.PROFOUND,
        {d: True for d in PhiDepth if d.value <= PhiDepth.PROFOUND.value}
    )
    result = verifier2.verify_position("owner", PI, PhiDepth.PROFOUND)
    assert result['verified']
    print("✓ Legitimate owner verified successfully")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--phi":
            demo_phi_properties()
        elif sys.argv[1] == "--nested":
            demo_nested_verification()
        elif sys.argv[1] == "--positions":
            demo_security_positions()
        elif sys.argv[1] == "--space":
            demo_phase_circle_with_depth()
        elif sys.argv[1] == "--compare":
            compare_security_models()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_phi_properties()
            demo_nested_verification()
            demo_security_positions()
            demo_phase_circle_with_depth()
            compare_security_models()
        else:
            run_tests()
    else:
        run_tests()
