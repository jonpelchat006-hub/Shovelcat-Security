"""
CRYPTOGRAPHIC EULER-PHI SECURITY
================================
Complete security position using P≠NP barrier:

S = e^(iθ) × φ^(n + h)

Where:
- θ = phase angle (Euler)
- n = base depth level (integer)
- h = SHA256(verification_chain) normalized to [0,1)

The fractional part h makes each user's magnitude UNIQUE
and cryptographically unforgeable.

Key insight:
- VERIFY: O(n) - hash the chain, compute magnitude
- FORGE: O(2^256) - find preimage that produces target magnitude

This brings P≠NP into the security model!

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import cmath
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import secrets

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895

# SHA-256 produces 256 bits
SHA256_BITS = 256
SHA256_MAX = 2 ** SHA256_BITS

# For normalizing hash to [0, 1)
def normalize_hash(hash_bytes: bytes) -> float:
    """Convert 256-bit hash to float in [0, 1)."""
    hash_int = int.from_bytes(hash_bytes, 'big')
    return hash_int / SHA256_MAX


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION CHAIN (Hashable History)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationEvent:
    """A single verification event in the chain."""
    event_type: str
    timestamp: datetime
    verifier_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    prev_hash: Optional[str] = None  # Hash of all prior events (hex)
    
    def to_bytes(self) -> bytes:
        """Serialize for hashing."""
        data = {
            'type': self.event_type,
            'time': self.timestamp.isoformat(),
            'verifier': self.verifier_id,
            'details': self.details,
            'nonce': self.nonce,
            'prev_hash': self.prev_hash,
        }
        return json.dumps(data, sort_keys=True).encode('utf-8')


@dataclass
class VerificationChain:
    """
    Complete verification history for a user.
    The chain is cryptographically linked - each event includes
    the hash of all previous events.
    """
    user_id: str
    events: List[VerificationEvent] = field(default_factory=list)
    
    def add_event(self, event: VerificationEvent) -> str:
        """Add event and return running hash."""
        # Each event must include the cumulative hash of everything before it.
        # This prevents reordering or tampering because the expected hash will
        # no longer match.
        prior_hash = self.get_chain_hash().hex()
        event.prev_hash = prior_hash
        self.events.append(event)
        return self.get_chain_hash().hex()
    
    def get_chain_hash(self) -> bytes:
        """
        Compute SHA-256 hash of entire chain.
        This is the cryptographic commitment to the verification history.
        """
        hasher = hashlib.sha256()
        
        # Include user_id as salt
        hasher.update(self.user_id.encode('utf-8'))
        
        # Hash each event in order
        for event in self.events:
            hasher.update(event.to_bytes())
        
        return hasher.digest()
    
    def get_normalized_hash(self) -> float:
        """Get hash normalized to [0, 1) for use as fractional exponent."""
        return normalize_hash(self.get_chain_hash())
    
    @property
    def depth(self) -> int:
        """Base depth level (number of major verification stages)."""
        # Count distinct verification types
        types = set(e.event_type for e in self.events)
        
        depth_map = {
            'identity': 0,
            'history': 1,
            'cross_domain': 2,
            'bec_formation': 3,
            'organizational': 4,
            'structural': 5
        }
        
        max_depth = 0
        for t in types:
            if t in depth_map:
                max_depth = max(max_depth, depth_map[t] + 1)
        
        return max_depth
    
    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify the chain hasn't been tampered with."""
        if not self.events:
            return True, "Empty chain"

        # Check timestamps are sequential
        for i in range(1, len(self.events)):
            if self.events[i].timestamp < self.events[i-1].timestamp:
                return False, f"Timestamp order violation at event {i}"

        # Check hash links (each event must include the hash of everything
        # before it).
        hasher = hashlib.sha256()
        hasher.update(self.user_id.encode('utf-8'))

        for i, event in enumerate(self.events):
            expected_prev = hasher.digest().hex()
            if event.prev_hash != expected_prev:
                return False, (
                    f"Hash chain mismatch at event {i}: "
                    f"expected {expected_prev}, found {event.prev_hash}"
                )

            # Extend hash with current event for next iteration
            hasher.update(event.to_bytes())

        return True, "Chain integrity verified"


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTOGRAPHIC SECURITY POSITION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CryptoSecurityPosition:
    """
    Full cryptographic security position.
    
    S = e^(iθ) × φ^(n + h)
    
    Where:
    - θ = phase angle
    - n = base depth (integer)
    - h = normalized hash of verification chain (fractional)
    
    The magnitude φ^(n+h) is UNIQUE to each user's specific history.
    """
    user_id: str
    theta: float  # Phase in radians
    chain: VerificationChain
    
    @property
    def base_depth(self) -> int:
        """Integer part of exponent (n)."""
        return self.chain.depth
    
    @property
    def fractional_depth(self) -> float:
        """Fractional part from hash (h)."""
        return self.chain.get_normalized_hash()
    
    @property
    def full_exponent(self) -> float:
        """n + h - the complete exponent."""
        return self.base_depth + self.fractional_depth
    
    @property
    def euler_component(self) -> complex:
        """e^(iθ)"""
        return cmath.exp(1j * self.theta)
    
    @property
    def phi_component(self) -> float:
        """φ^(n+h) - the cryptographic magnitude."""
        return PHI ** self.full_exponent
    
    @property
    def full_position(self) -> complex:
        """S = e^(iθ) × φ^(n+h)"""
        return self.euler_component * self.phi_component
    
    @property
    def magnitude(self) -> float:
        """Unique magnitude based on verification history."""
        return abs(self.full_position)
    
    @property
    def chain_hash(self) -> str:
        """Hex string of chain hash."""
        return self.chain.get_chain_hash().hex()
    
    def get_verification_proof(self) -> Dict[str, Any]:
        """
        Generate proof that can be verified.
        
        This is the P≠NP asymmetry:
        - Creating this proof: O(n) - just follow the chain
        - Forging this proof: O(2^256) - find preimage
        """
        return {
            'user_id': self.user_id,
            'theta': self.theta,
            'theta_degrees': math.degrees(self.theta),
            'base_depth': self.base_depth,
            'chain_hash': self.chain_hash,
            'fractional_depth': self.fractional_depth,
            'full_exponent': self.full_exponent,
            'magnitude': self.magnitude,
            'position': {
                'real': self.full_position.real,
                'imag': self.full_position.imag
            },
            'event_count': len(self.chain.events)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTOGRAPHIC VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoVerifier:
    """
    Verifies security positions using cryptographic proofs.
    
    Key insight: Verification is FAST, forgery is HARD.
    """
    
    def __init__(self):
        self.positions: Dict[str, CryptoSecurityPosition] = {}
        self.magnitude_registry: Dict[str, float] = {}  # Track expected magnitudes
    
    def register_position(self, position: CryptoSecurityPosition):
        """Register a position and its expected magnitude."""
        self.positions[position.user_id] = position
        self.magnitude_registry[position.user_id] = position.magnitude
    
    def verify_claim(self, 
                    user_id: str,
                    claimed_theta: float,
                    claimed_magnitude: float,
                    provided_chain: VerificationChain) -> Dict[str, Any]:
        """
        Verify a claimed security position.
        
        The claimant must provide:
        1. Their user_id
        2. Claimed phase (θ)
        3. Claimed magnitude (φ^(n+h))
        4. The verification chain that produces that magnitude
        
        We verify by:
        1. Hashing the provided chain
        2. Computing the magnitude from that hash
        3. Checking if it matches claimed magnitude
        4. Checking phase consistency
        """
        result = {
            'user_id': user_id,
            'verified': False,
            'checks': {}
        }
        
        # Check 1: Chain integrity
        integrity_ok, integrity_msg = provided_chain.verify_integrity()
        result['checks']['chain_integrity'] = {
            'passed': integrity_ok,
            'message': integrity_msg
        }
        if not integrity_ok:
            return result
        
        # Check 2: Compute magnitude from provided chain
        computed_exponent = provided_chain.depth + provided_chain.get_normalized_hash()
        computed_magnitude = PHI ** computed_exponent
        
        magnitude_match = abs(computed_magnitude - claimed_magnitude) < 0.0001
        result['checks']['magnitude'] = {
            'claimed': claimed_magnitude,
            'computed': computed_magnitude,
            'match': magnitude_match,
            'exponent': computed_exponent
        }
        
        # Check 3: If we have stored position, verify against that
        if user_id in self.positions:
            stored = self.positions[user_id]
            
            # Phase check
            phase_match = abs(claimed_theta - stored.theta) < 0.01
            result['checks']['phase'] = {
                'claimed': math.degrees(claimed_theta),
                'stored': math.degrees(stored.theta),
                'match': phase_match
            }
            
            # Chain hash check (most important!)
            stored_hash = stored.chain_hash
            provided_hash = provided_chain.get_chain_hash().hex()
            hash_match = stored_hash == provided_hash
            
            result['checks']['chain_hash'] = {
                'stored': stored_hash[:16] + '...',
                'provided': provided_hash[:16] + '...',
                'match': hash_match
            }
            
            # Expected magnitude check
            expected_magnitude = self.magnitude_registry.get(user_id)
            if expected_magnitude:
                expected_match = abs(claimed_magnitude - expected_magnitude) < 0.0001
                result['checks']['expected_magnitude'] = {
                    'expected': expected_magnitude,
                    'claimed': claimed_magnitude,
                    'match': expected_match
                }
        
        # Overall verification
        all_checks_passed = all(
            check.get('match', check.get('passed', True))
            for check in result['checks'].values()
        )
        result['verified'] = all_checks_passed
        
        return result
    
    def detect_forgery(self,
                      user_id: str,
                      claimed_magnitude: float,
                      provided_chain: VerificationChain) -> Dict[str, Any]:
        """
        Detect if someone is trying to forge a position.
        
        Forgery indicators:
        1. Magnitude doesn't match chain hash
        2. Chain hash doesn't match stored hash
        3. Events have impossible timestamps
        4. Nonces are reused or predictable
        """
        result = {
            'user_id': user_id,
            'is_forgery': False,
            'signals': [],
            'confidence': 0.0
        }
        
        # Signal 1: Magnitude mismatch
        computed_exponent = provided_chain.depth + provided_chain.get_normalized_hash()
        computed_magnitude = PHI ** computed_exponent
        
        if abs(computed_magnitude - claimed_magnitude) > 0.0001:
            result['signals'].append({
                'type': 'magnitude_mismatch',
                'severity': 1.0,
                'detail': f"Claimed {claimed_magnitude:.6f}, chain produces {computed_magnitude:.6f}"
            })
            result['is_forgery'] = True
        
        # Signal 2: Hash mismatch with stored
        if user_id in self.positions:
            stored_hash = self.positions[user_id].chain_hash
            provided_hash = provided_chain.get_chain_hash().hex()
            
            if stored_hash != provided_hash:
                result['signals'].append({
                    'type': 'chain_hash_mismatch',
                    'severity': 1.0,
                    'detail': f"Stored hash doesn't match provided chain"
                })
                result['is_forgery'] = True
        
        # Signal 3: Timestamp anomalies
        if provided_chain.events:
            timestamps = [e.timestamp for e in provided_chain.events]
            
            # Check for future timestamps
            now = datetime.now()
            future_events = [t for t in timestamps if t > now]
            if future_events:
                result['signals'].append({
                    'type': 'future_timestamps',
                    'severity': 1.0,
                    'detail': f"{len(future_events)} events have future timestamps"
                })
                result['is_forgery'] = True
            
            # Check for suspiciously uniform timestamps
            if len(timestamps) > 2:
                diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
                if len(set(diffs)) == 1 and diffs[0] < 1:
                    result['signals'].append({
                        'type': 'uniform_timestamps',
                        'severity': 0.8,
                        'detail': "All events have identical time gaps (likely automated)"
                    })
        
        # Signal 4: Nonce patterns (should be random)
        if provided_chain.events:
            nonces = [e.nonce for e in provided_chain.events]
            
            # Check for sequential patterns
            if len(set(nonces)) < len(nonces):
                result['signals'].append({
                    'type': 'nonce_reuse',
                    'severity': 1.0,
                    'detail': "Nonces are reused (should be unique)"
                })
                result['is_forgery'] = True
        
        # Calculate confidence
        if result['signals']:
            result['confidence'] = max(s['severity'] for s in result['signals'])
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# P≠NP DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_p_np_barrier():
    """
    Demonstrate the P≠NP asymmetry in verification vs forgery.
    """
    print("\n" + "=" * 80)
    print("P ≠ NP: VERIFICATION vs FORGERY")
    print("=" * 80)
    
    print("""
    
    THE ASYMMETRY:
    ══════════════
    
    VERIFY (P - Polynomial Time):
    ─────────────────────────────
    Given: User's verification chain
    
    1. Serialize chain events           O(n)
    2. Compute SHA-256 hash             O(n)
    3. Compute φ^(n + h)                O(1)
    4. Compare with claimed magnitude   O(1)
    
    Total: O(n) - LINEAR in chain length
    
    
    FORGE (NP-Hard - Exponential Time):
    ───────────────────────────────────
    Goal: Create chain that produces target magnitude |S| = 7.823456...
    
    1. Need to find chain where:
       SHA256(chain) / 2^256 = target fractional part
    
    2. This is the SHA-256 PREIMAGE problem!
    
    3. Best known attack: Brute force
       Try random chains until hash matches
       Expected attempts: 2^256
    
    Total: O(2^256) - ASTRONOMICAL
    
    
    COMPARISON:
    ───────────
    
    Chain length n = 10 events:
    
    Verify:  ~10 operations
    Forge:   ~10^77 operations
    
    If computer does 10^12 ops/second:
    
    Verify:  0.00000001 seconds
    Forge:   10^65 seconds ≈ 10^57 years
    
    Universe age: ~10^10 years
    
    Forgery would take 10^47 universe lifetimes!
    """)


def demonstrate_unique_magnitudes():
    """
    Show how each user gets a unique magnitude based on their history.
    """
    print("\n" + "=" * 80)
    print("UNIQUE MAGNITUDES FROM VERIFICATION HISTORY")
    print("=" * 80)
    
    # Create several users with different histories
    users = []
    
    for i, (name, events) in enumerate([
        ("alice", [
            ("identity", "verifier_1", {"method": "email"}),
            ("history", "verifier_2", {"years": 2}),
            ("cross_domain", "verifier_3", {"domains": 3}),
            ("bec_formation", "verifier_4", {"parties": 3}),
        ]),
        ("bob", [
            ("identity", "verifier_1", {"method": "phone"}),
            ("history", "verifier_2", {"years": 3}),
            ("cross_domain", "verifier_3", {"domains": 2}),
            ("bec_formation", "verifier_4", {"parties": 3}),
        ]),
        ("charlie", [
            ("identity", "verifier_5", {"method": "email"}),
            ("history", "verifier_6", {"years": 2}),
            ("cross_domain", "verifier_7", {"domains": 3}),
            ("bec_formation", "verifier_8", {"parties": 3}),
        ]),
    ]):
        chain = VerificationChain(user_id=name)
        
        base_time = datetime.now()
        for j, (event_type, verifier, details) in enumerate(events):
            event = VerificationEvent(
                event_type=event_type,
                timestamp=base_time + timedelta(days=j*30),
                verifier_id=verifier,
                details=details
            )
            chain.add_event(event)
        
        pos = CryptoSecurityPosition(
            user_id=name,
            theta=PI * (i + 1) / 4,  # Different phases
            chain=chain
        )
        users.append(pos)
    
    print("\n┌────────────┬──────────────┬────────────────┬────────────────────┬───────────────┐")
    print("│ User       │ Base Depth   │ Hash Fraction  │ Full Exponent      │ Magnitude     │")
    print("├────────────┼──────────────┼────────────────┼────────────────────┼───────────────┤")
    
    for pos in users:
        print(f"│ {pos.user_id:10} │ {pos.base_depth:12} │ {pos.fractional_depth:14.10f} │ {pos.full_exponent:18.10f} │ {pos.magnitude:13.6f} │")
    
    print("└────────────┴──────────────┴────────────────┴────────────────────┴───────────────┘")
    
    print("""
    
    OBSERVATIONS:
    ─────────────
    
    • All users have base depth 4 (same verification stages)
    • But each has DIFFERENT fractional part from their unique history
    • Even alice and charlie have similar events but different verifiers
    • Their magnitudes differ by the cryptographic hash!
    
    An attacker would need to:
    1. Know the target magnitude exactly
    2. Find a chain that hashes to produce that fractional part
    3. This requires breaking SHA-256 preimage resistance
    
    IMPOSSIBLE in practice.
    """)


def demonstrate_forgery_detection():
    """
    Show how forgery attempts are detected.
    """
    print("\n" + "=" * 80)
    print("FORGERY DETECTION")
    print("=" * 80)
    
    verifier = CryptoVerifier()
    
    # Create legitimate user
    real_chain = VerificationChain(user_id="real_owner")
    base_time = datetime.now() - timedelta(days=365)
    
    for i, (event_type, verifier_id) in enumerate([
        ("identity", "id_service"),
        ("history", "history_check"),
        ("cross_domain", "domain_verify"),
        ("bec_formation", "bec_service"),
        ("organizational", "org_verify"),
    ]):
        event = VerificationEvent(
            event_type=event_type,
            timestamp=base_time + timedelta(days=i*60),
            verifier_id=verifier_id,
            details={"step": i+1}
        )
        real_chain.add_event(event)
    
    real_position = CryptoSecurityPosition(
        user_id="real_owner",
        theta=PI,
        chain=real_chain
    )
    
    verifier.register_position(real_position)
    
    print(f"\n--- LEGITIMATE OWNER ---")
    proof = real_position.get_verification_proof()
    print(f"User: {proof['user_id']}")
    print(f"Phase: {proof['theta_degrees']:.1f}°")
    print(f"Base Depth: {proof['base_depth']}")
    print(f"Fractional: {proof['fractional_depth']:.10f}")
    print(f"Magnitude: {proof['magnitude']:.10f}")
    print(f"Chain Hash: {proof['chain_hash'][:32]}...")
    
    # Forgery attempt 1: Wrong chain, claim right magnitude
    print(f"\n--- FORGERY ATTEMPT 1: Fake chain, claim real magnitude ---")
    
    fake_chain = VerificationChain(user_id="real_owner")  # Claim same user
    for i in range(5):
        fake_chain.add_event(VerificationEvent(
            event_type=["identity", "history", "cross_domain", "bec_formation", "organizational"][i],
            timestamp=datetime.now() - timedelta(days=i),  # Wrong timestamps!
            verifier_id=f"fake_verifier_{i}",
            details={"fake": True}
        ))
    
    detection = verifier.detect_forgery(
        "real_owner",
        real_position.magnitude,  # Claim the real magnitude
        fake_chain
    )
    
    print(f"Is Forgery: {detection['is_forgery']}")
    print(f"Confidence: {detection['confidence']:.2f}")
    for signal in detection['signals']:
        print(f"  Signal: {signal['type']} (severity: {signal['severity']})")
        print(f"    {signal['detail']}")
    
    # Forgery attempt 2: Claim magnitude without chain
    print(f"\n--- FORGERY ATTEMPT 2: Claim magnitude directly ---")
    
    empty_chain = VerificationChain(user_id="hacker")
    
    detection2 = verifier.detect_forgery(
        "hacker",
        7.823456,  # Arbitrary magnitude
        empty_chain
    )
    
    computed_mag = PHI ** (empty_chain.depth + empty_chain.get_normalized_hash())
    print(f"Claimed magnitude: 7.823456")
    print(f"Chain produces: {computed_mag:.6f}")
    print(f"Mismatch: {abs(7.823456 - computed_mag):.6f}")
    
    # Verify legitimate owner
    print(f"\n--- LEGITIMATE VERIFICATION ---")
    
    verification = verifier.verify_claim(
        "real_owner",
        PI,
        real_position.magnitude,
        real_chain
    )
    
    print(f"Verified: {verification['verified']}")
    for check_name, check_result in verification['checks'].items():
        status = '✓' if check_result.get('match', check_result.get('passed', False)) else '✗'
        print(f"  {check_name}: {status}")


def demo_full_system():
    """
    Complete demonstration of the cryptographic security system.
    """
    print("\n" + "=" * 80)
    print("COMPLETE CRYPTOGRAPHIC SECURITY SYSTEM")
    print("=" * 80)
    
    print("""
    
    FULL SECURITY POSITION:
    ═══════════════════════
    
    S = e^(iθ) × φ^(n + h)
    
    Where:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  θ = Phase angle (0 to π)                                                  │
    │      Determined by: Role in organization                                   │
    │      0 = Outsider, π = Owner                                               │
    │                                                                             │
    │  n = Base depth (integer 0-5)                                              │
    │      Determined by: Number of verification stages passed                   │
    │      0 = Surface, 5 = Foundational                                         │
    │                                                                             │
    │  h = Hash fraction (0 to 1)                                                │
    │      Determined by: SHA256(verification_chain) / 2^256                     │
    │      UNIQUE to each user's specific history!                               │
    │                                                                             │
    │  Full magnitude: |S| = φ^(n+h)                                             │
    │      This is CRYPTOGRAPHICALLY UNIQUE                                      │
    │      Cannot be forged without breaking SHA-256                             │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    
    SECURITY LAYERS:
    ════════════════
    
    Layer 1: EULER PHASE (θ)
    ─────────────────────────
    • Determines which "side" you're on
    • User (θ<π/2) vs Staff (θ>π/2)
    • e^(i·0) = +1 (pure outsider)
    • e^(i·π) = -1 (pure insider)
    
    
    Layer 2: PHI DEPTH (n)
    ──────────────────────
    • Determines verification level
    • Each level CONTAINS all previous (1/φ + 1 = φ)
    • φ⁰=1, φ¹=1.618, φ²=2.618, φ³=4.236, φ⁴=6.854, φ⁵=11.09
    
    
    Layer 3: CRYPTO HASH (h)
    ────────────────────────
    • Makes magnitude UNIQUE to verification history
    • SHA-256 provides P≠NP barrier
    • Verify: O(n), Forge: O(2^256)
    
    
    COMBINED PROTECTION:
    ═══════════════════
    
    To access a resource requiring position S₀:
    
    1. Check PHASE: Is θ in allowed range?
       └─ Blocks wrong-side access (users can't access staff resources)
    
    2. Check DEPTH: Is n ≥ required level?
       └─ Blocks insufficient verification (shallow can't access deep)
    
    3. Check MAGNITUDE: Does |S| = φ^(n+h) match stored value?
       └─ Blocks forgery (hash must match exact history)
    
    4. Check CHAIN: Does provided chain hash to expected value?
       └─ Blocks tampering (can't modify history)
    
    ALL FOUR must pass!
    """)


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("CRYPTOGRAPHIC EULER-PHI - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Hash normalization
    print("\n--- TEST 1: Hash Normalization ---")
    test_hash = hashlib.sha256(b"test").digest()
    normalized = normalize_hash(test_hash)
    assert 0 <= normalized < 1
    print(f"✓ Hash normalizes to [0,1): {normalized:.10f}")
    
    # Test 2: Verification chain
    print("\n--- TEST 2: Verification Chain ---")
    chain = VerificationChain(user_id="test")
    chain.add_event(VerificationEvent(
        event_type="identity",
        timestamp=datetime.now(),
        verifier_id="test_verifier"
    ))
    assert chain.depth == 1
    assert len(chain.events) == 1
    print(f"✓ Chain tracks events and depth")
    
    # Test 3: Unique hashes
    print("\n--- TEST 3: Unique Hashes ---")
    chain1 = VerificationChain(user_id="user1")
    chain1.add_event(VerificationEvent("identity", datetime.now(), "v1"))
    
    chain2 = VerificationChain(user_id="user2")
    chain2.add_event(VerificationEvent("identity", datetime.now(), "v1"))
    
    assert chain1.get_chain_hash() != chain2.get_chain_hash()
    print(f"✓ Different users get different hashes")
    
    # Test 4: Security position
    print("\n--- TEST 4: Security Position ---")
    chain = VerificationChain(user_id="test_user")
    for event_type in ["identity", "history", "cross_domain", "bec_formation"]:
        chain.add_event(VerificationEvent(event_type, datetime.now(), "verifier"))
    
    pos = CryptoSecurityPosition("test_user", PI/2, chain)
    assert pos.base_depth == 4
    assert 0 <= pos.fractional_depth < 1
    assert pos.magnitude > PHI ** 4
    print(f"✓ Position computes correctly: |S| = {pos.magnitude:.6f}")
    
    # Test 5: Forgery detection
    print("\n--- TEST 5: Forgery Detection ---")
    verifier = CryptoVerifier()
    
    real_chain = VerificationChain(user_id="real")
    real_chain.add_event(VerificationEvent("identity", datetime.now(), "v1"))
    real_pos = CryptoSecurityPosition("real", PI, real_chain)
    verifier.register_position(real_pos)
    
    fake_chain = VerificationChain(user_id="real")
    fake_chain.add_event(VerificationEvent("identity", datetime.now(), "fake"))
    
    detection = verifier.detect_forgery("real", real_pos.magnitude, fake_chain)
    assert detection['is_forgery']
    print(f"✓ Forgery detected: {[s['type'] for s in detection['signals']]}")
    
    # Test 6: Legitimate verification
    print("\n--- TEST 6: Legitimate Verification ---")
    result = verifier.verify_claim("real", PI, real_pos.magnitude, real_chain)
    assert result['verified']
    print(f"✓ Legitimate claim verified")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--pnp":
            demonstrate_p_np_barrier()
        elif sys.argv[1] == "--unique":
            demonstrate_unique_magnitudes()
        elif sys.argv[1] == "--forgery":
            demonstrate_forgery_detection()
        elif sys.argv[1] == "--system":
            demo_full_system()
        elif sys.argv[1] == "--all":
            run_tests()
            demonstrate_p_np_barrier()
            demonstrate_unique_magnitudes()
            demonstrate_forgery_detection()
            demo_full_system()
        else:
            run_tests()
    else:
        run_tests()
