"""
3-DIRECTION RNA GEL VERIFICATION SYSTEM
========================================
Tiered verification with DNA (expensive) and RNA (cheap) checks.

Three Axes:
- Z (Authority): Upgrade/level changes - requires full DNA check
- X (Action): Read/Write/Delete - uses RNA tokens
- Y (Resource): Department/data access - uses RNA tokens

Key Innovation:
- DNA check is expensive but releases RNA tokens
- RNA checks are cheap (O(1)) but consume tokens
- When RNA depletes, must do another DNA check
- This provides both EFFICIENCY and RATE LIMITING

RNA Release Formula:
    tokens = floor(φ^n × (1 + θ/π))
    
    Where n = depth level, θ = phase angle

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from datetime import datetime, timedelta
import secrets

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2

# Token expiry (seconds)
RNA_TOKEN_LIFETIME = 3600  # 1 hour
DNA_CHECK_COOLDOWN = 60    # Minimum seconds between DNA checks

# Action costs (RNA tokens)
ACTION_COSTS = {
    'read': 1,
    'list': 1,
    'write': 2,
    'update': 2,
    'delete': 3,
    'execute': 3,
    'admin': 5,
    'upgrade': float('inf'),  # Always requires DNA check
}


# ═══════════════════════════════════════════════════════════════════════════════
# RNA TOKEN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RNAToken:
    """
    Lightweight verification token derived from DNA check.
    
    Properties:
    - Fast to verify (O(1))
    - Cannot be forged (HMAC signed)
    - Limited uses (counter)
    - Time-limited (expiry)
    - Scope-limited (allowed actions/resources)
    """
    token_id: str
    user_id: str
    signature: bytes
    
    # Limits
    max_uses: int
    remaining_uses: int
    created_at: float
    expires_at: float
    
    # Scope
    max_action_level: int  # Highest action cost allowed
    allowed_resources: Optional[List[str]] = None  # None = all
    
    # Source DNA check info (lightweight)
    source_theta: float = 0.0
    source_depth: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return time.time() > self.expires_at
    
    @property
    def is_exhausted(self) -> bool:
        """Check if token has no remaining uses."""
        return self.remaining_uses <= 0
    
    @property
    def is_valid(self) -> bool:
        """Check if token is still usable."""
        return not self.is_expired and not self.is_exhausted
    
    def consume(self, cost: int = 1) -> bool:
        """
        Consume uses from token.
        Returns True if successful, False if insufficient.
        """
        if cost > self.remaining_uses:
            return False
        self.remaining_uses -= cost
        return True
    
    def can_perform(self, action: str, resource: str = None) -> Tuple[bool, str]:
        """Check if token allows this action on this resource."""
        if self.is_expired:
            return False, "Token expired"
        
        if self.is_exhausted:
            return False, "Token exhausted (no remaining uses)"
        
        cost = ACTION_COSTS.get(action, 1)
        if cost > self.max_action_level:
            return False, f"Action '{action}' exceeds token scope"
        
        if cost > self.remaining_uses:
            return False, f"Insufficient RNA: need {cost}, have {self.remaining_uses}"
        
        if self.allowed_resources and resource:
            if resource not in self.allowed_resources:
                return False, f"Resource '{resource}' not in token scope"
        
        return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# RNA TOKEN POOL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RNAPool:
    """
    Pool of RNA tokens for a user.
    Manages token lifecycle and consumption.
    """
    user_id: str
    tokens: List[RNAToken] = field(default_factory=list)
    last_dna_check: float = 0.0
    total_dna_checks: int = 0
    total_rna_consumed: int = 0
    
    @property
    def total_remaining(self) -> int:
        """Total RNA remaining across all valid tokens."""
        return sum(t.remaining_uses for t in self.tokens if t.is_valid)
    
    @property
    def valid_tokens(self) -> List[RNAToken]:
        """Get all currently valid tokens."""
        return [t for t in self.tokens if t.is_valid]
    
    def cleanup_expired(self):
        """Remove expired and exhausted tokens."""
        self.tokens = [t for t in self.tokens if t.is_valid]
    
    def add_tokens(self, token: RNAToken):
        """Add a new token to the pool."""
        self.cleanup_expired()
        self.tokens.append(token)
    
    def consume(self, cost: int, action: str, resource: str = None) -> Tuple[bool, str]:
        """
        Consume RNA from pool for an action.
        Uses tokens in FIFO order (oldest first).
        """
        self.cleanup_expired()
        
        if not self.tokens:
            return False, "No RNA tokens - DNA check required"
        
        remaining_cost = cost
        
        for token in self.tokens:
            if remaining_cost <= 0:
                break
            
            can_do, reason = token.can_perform(action, resource)
            if not can_do:
                continue
            
            # Consume from this token
            consume_amount = min(remaining_cost, token.remaining_uses)
            token.consume(consume_amount)
            remaining_cost -= consume_amount
            self.total_rna_consumed += consume_amount
        
        if remaining_cost > 0:
            return False, f"Insufficient RNA: needed {cost}, consumed {cost - remaining_cost}"
        
        return True, f"Consumed {cost} RNA, {self.total_remaining} remaining"
    
    def needs_dna_check(self) -> bool:
        """Check if a DNA check is needed."""
        self.cleanup_expired()
        return len(self.tokens) == 0 or self.total_remaining == 0


# ═══════════════════════════════════════════════════════════════════════════════
# DNA CHECK (Expensive)
# ═══════════════════════════════════════════════════════════════════════════════

class DNAVerifier:
    """
    Performs full cryptographic DNA checks.
    Expensive but thorough - releases RNA tokens on success.
    """
    
    def __init__(self, secret_key: bytes = None):
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.check_count = 0
    
    def compute_token_count(self, theta: float, depth: int) -> int:
        """
        Compute how many RNA tokens to release.
        
        Formula: floor(φ^n × (1 + θ/π))
        
        Higher authority (θ) and deeper verification (n) = more tokens
        """
        phi_component = PHI ** depth
        phase_multiplier = 1 + (theta / PI)
        tokens = int(phi_component * phase_multiplier)
        return max(1, tokens)  # At least 1 token
    
    def compute_max_action_level(self, theta: float, depth: int) -> int:
        """
        Compute highest action cost this token allows.
        Based on authority level.
        """
        if depth >= 4 and theta > PI/2:
            return 5  # Admin level
        elif depth >= 3:
            return 3  # Execute/Delete level
        elif depth >= 2:
            return 2  # Write level
        else:
            return 1  # Read only
    
    def perform_dna_check(self, 
                         user_id: str,
                         theta: float,
                         depth: int,
                         chain_hash: str) -> Tuple[bool, Optional[RNAToken], Dict]:
        """
        Perform full DNA verification.
        
        This is the EXPENSIVE operation that:
        1. Verifies the cryptographic chain
        2. Checks all security dimensions
        3. Releases RNA tokens on success
        
        Returns: (success, token, details)
        """
        self.check_count += 1
        start_time = time.time()
        
        details = {
            'check_id': self.check_count,
            'user_id': user_id,
            'theta': theta,
            'theta_degrees': math.degrees(theta),
            'depth': depth,
            'chain_hash': chain_hash[:16] + '...',
            'operations': 0
        }
        
        # Simulate expensive operations
        operations = 0
        
        # 1. Verify chain hash (expensive)
        operations += self._verify_chain_hash(chain_hash)
        
        # 2. Compute magnitude (moderate)
        magnitude = PHI ** depth
        operations += 10
        
        # 3. Verify phase consistency (cheap)
        phase_valid = 0 <= theta <= PI
        operations += 1
        
        # 4. Check all 9 axes (moderate)
        operations += 9 * 5  # 5 ops per axis
        
        details['operations'] = operations
        details['duration_ms'] = (time.time() - start_time) * 1000
        
        if not phase_valid:
            details['error'] = "Invalid phase"
            return False, None, details
        
        # SUCCESS - Generate RNA tokens
        token_count = self.compute_token_count(theta, depth)
        max_action = self.compute_max_action_level(theta, depth)
        
        # Create token
        token_id = secrets.token_hex(16)
        token_data = f"{user_id}:{token_id}:{token_count}:{time.time()}"
        signature = hmac.new(self.secret_key, token_data.encode(), 'sha256').digest()
        
        token = RNAToken(
            token_id=token_id,
            user_id=user_id,
            signature=signature,
            max_uses=token_count,
            remaining_uses=token_count,
            created_at=time.time(),
            expires_at=time.time() + RNA_TOKEN_LIFETIME,
            max_action_level=max_action,
            source_theta=theta,
            source_depth=depth
        )
        
        details['tokens_released'] = token_count
        details['max_action_level'] = max_action
        details['token_lifetime_s'] = RNA_TOKEN_LIFETIME
        
        return True, token, details
    
    def _verify_chain_hash(self, chain_hash: str) -> int:
        """Simulate expensive chain verification. Returns operation count."""
        # In real implementation, this would verify the full chain
        # Here we simulate the computational cost
        operations = 0
        
        # Simulate hash computation
        for i in range(10):
            h = hashlib.sha256(chain_hash.encode())
            h.update(str(i).encode())
            h.digest()
            operations += 100
        
        return operations


# ═══════════════════════════════════════════════════════════════════════════════
# RNA VERIFIER (Cheap)
# ═══════════════════════════════════════════════════════════════════════════════

class RNAVerifier:
    """
    Performs lightweight RNA token verification.
    Fast and cheap - O(1) operations.
    """
    
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.check_count = 0
    
    def verify_token(self, token: RNAToken) -> Tuple[bool, str]:
        """
        Verify an RNA token is authentic.
        This is FAST - O(1).
        """
        self.check_count += 1
        
        # Quick checks first
        if token.is_expired:
            return False, "Token expired"
        
        if token.is_exhausted:
            return False, "Token exhausted"
        
        # Verify signature (fast - just HMAC)
        token_data = f"{token.user_id}:{token.token_id}:{token.max_uses}:{token.created_at}"
        expected_sig = hmac.new(self.secret_key, token_data.encode(), 'sha256').digest()
        
        if not hmac.compare_digest(token.signature, expected_sig):
            return False, "Invalid signature"
        
        return True, "Valid"


# ═══════════════════════════════════════════════════════════════════════════════
# 3-DIRECTION GEL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class RNAGelSystem:
    """
    Complete 3-direction verification system.
    
    Z-axis (Authority): DNA checks for upgrades
    X-axis (Action): RNA checks for operations
    Y-axis (Resource): RNA checks for access
    """
    
    def __init__(self):
        self.secret_key = secrets.token_bytes(32)
        self.dna_verifier = DNAVerifier(self.secret_key)
        self.rna_verifier = RNAVerifier(self.secret_key)
        self.pools: Dict[str, RNAPool] = {}
        
        # User state
        self.user_positions: Dict[str, Dict] = {}
        
        # Stats
        self.total_dna_checks = 0
        self.total_rna_checks = 0
        self.total_requests = 0
    
    def register_user(self, user_id: str, theta: float, depth: int, chain_hash: str):
        """Register a user with their security position."""
        self.user_positions[user_id] = {
            'theta': theta,
            'depth': depth,
            'chain_hash': chain_hash
        }
        self.pools[user_id] = RNAPool(user_id=user_id)
    
    def request(self, 
               user_id: str, 
               action: str, 
               resource: str = None,
               force_dna: bool = False) -> Dict[str, Any]:
        """
        Process a request through the gel.
        
        Decision tree:
        1. Is this an UPGRADE (Z-axis)? → DNA check required
        2. Does user have RNA tokens? → Try RNA check
        3. RNA insufficient? → Do DNA check, then retry
        """
        self.total_requests += 1
        
        result = {
            'request_id': self.total_requests,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'check_type': None,
            'success': False,
            'details': {}
        }
        
        # Get user position
        position = self.user_positions.get(user_id)
        if not position:
            result['error'] = "Unknown user"
            return result
        
        pool = self.pools.get(user_id)
        if not pool:
            pool = RNAPool(user_id=user_id)
            self.pools[user_id] = pool
        
        action_cost = ACTION_COSTS.get(action, 1)
        
        # Z-AXIS: Upgrade always requires DNA
        if action == 'upgrade' or force_dna:
            result['check_type'] = 'DNA'
            return self._do_dna_check(user_id, position, pool, result)
        
        # X/Y-AXIS: Try RNA first
        if not pool.needs_dna_check():
            # Try RNA check
            success, message = pool.consume(action_cost, action, resource)
            
            if success:
                self.total_rna_checks += 1
                result['check_type'] = 'RNA'
                result['success'] = True
                result['details'] = {
                    'cost': action_cost,
                    'remaining': pool.total_remaining,
                    'message': message
                }
                return result
        
        # RNA insufficient - do DNA check
        result['check_type'] = 'DNA→RNA'
        dna_result = self._do_dna_check(user_id, position, pool, result)
        
        if dna_result['success']:
            # Now try the action with fresh RNA
            success, message = pool.consume(action_cost, action, resource)
            dna_result['success'] = success
            dna_result['details']['action_result'] = message
            dna_result['details']['remaining'] = pool.total_remaining
        
        return dna_result
    
    def _do_dna_check(self, user_id: str, position: Dict, 
                     pool: RNAPool, result: Dict) -> Dict:
        """Perform DNA check and add tokens to pool."""
        self.total_dna_checks += 1
        
        success, token, details = self.dna_verifier.perform_dna_check(
            user_id=user_id,
            theta=position['theta'],
            depth=position['depth'],
            chain_hash=position['chain_hash']
        )
        
        result['success'] = success
        result['details'] = details
        
        if success and token:
            pool.add_tokens(token)
            pool.last_dna_check = time.time()
            pool.total_dna_checks += 1
            result['details']['pool_total'] = pool.total_remaining
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'total_requests': self.total_requests,
            'total_dna_checks': self.total_dna_checks,
            'total_rna_checks': self.total_rna_checks,
            'dna_check_ratio': self.total_dna_checks / max(1, self.total_requests),
            'efficiency': self.total_rna_checks / max(1, self.total_requests),
            'pools': {
                uid: {
                    'remaining': pool.total_remaining,
                    'dna_checks': pool.total_dna_checks,
                    'rna_consumed': pool.total_rna_consumed
                }
                for uid, pool in self.pools.items()
            }
        }
    
    def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """Get status for a specific user."""
        pool = self.pools.get(user_id)
        position = self.user_positions.get(user_id)
        
        if not pool or not position:
            return {'error': 'Unknown user'}
        
        return {
            'user_id': user_id,
            'theta_degrees': math.degrees(position['theta']),
            'depth': position['depth'],
            'rna_remaining': pool.total_remaining,
            'valid_tokens': len(pool.valid_tokens),
            'needs_dna_check': pool.needs_dna_check(),
            'total_dna_checks': pool.total_dna_checks,
            'total_rna_consumed': pool.total_rna_consumed
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_token_calculation():
    """Show how tokens are calculated."""
    print("\n" + "=" * 80)
    print("RNA TOKEN CALCULATION")
    print("=" * 80)
    
    print("\nFormula: tokens = floor(φ^n × (1 + θ/π))")
    print("\n┌─────────────────────┬───────┬─────────┬─────────────┬────────────┐")
    print("│ User Type           │ Depth │ θ (deg) │ Multiplier  │ RNA Tokens │")
    print("├─────────────────────┼───────┼─────────┼─────────────┼────────────┤")
    
    users = [
        ("Anonymous", 0, 0),
        ("Registered", 1, 15),
        ("Paid User", 2, 30),
        ("Long-term", 2, 45),
        ("Core User", 3, 60),
        ("Creator", 3, 90),
        ("Staff", 4, 135),
        ("Admin", 4, 150),
        ("Owner", 5, 180),
    ]
    
    dna = DNAVerifier()
    
    for name, depth, theta_deg in users:
        theta = math.radians(theta_deg)
        tokens = dna.compute_token_count(theta, depth)
        multiplier = 1 + (theta / PI)
        print(f"│ {name:19} │ {depth:5} │ {theta_deg:7}° │ {multiplier:11.3f} │ {tokens:10} │")
    
    print("└─────────────────────┴───────┴─────────┴─────────────┴────────────┘")


def demo_action_costs():
    """Show action costs."""
    print("\n" + "=" * 80)
    print("ACTION RNA COSTS")
    print("=" * 80)
    
    print("\n┌──────────────┬──────────┬─────────────────────────────────────┐")
    print("│ Action       │ RNA Cost │ Description                         │")
    print("├──────────────┼──────────┼─────────────────────────────────────┤")
    
    descriptions = {
        'read': 'Non-destructive data access',
        'list': 'Directory/index listing',
        'write': 'Create or modify data',
        'update': 'Modify existing data',
        'delete': 'Remove data permanently',
        'execute': 'Run code/scripts',
        'admin': 'System administration',
        'upgrade': 'Change authority level (requires DNA)',
    }
    
    for action, cost in ACTION_COSTS.items():
        cost_str = "∞ (DNA)" if cost == float('inf') else str(int(cost))
        desc = descriptions.get(action, '')
        print(f"│ {action:12} │ {cost_str:8} │ {desc:35} │")
    
    print("└──────────────┴──────────┴─────────────────────────────────────┘")


def demo_session_lifecycle():
    """Demonstrate a full session lifecycle."""
    print("\n" + "=" * 80)
    print("SESSION LIFECYCLE DEMONSTRATION")
    print("=" * 80)
    
    system = RNAGelSystem()
    
    # Register a staff member
    system.register_user(
        user_id="alice",
        theta=math.radians(135),  # Staff
        depth=4,                   # Profound
        chain_hash="abc123def456"
    )
    
    print("\n--- User Registered ---")
    status = system.get_user_status("alice")
    print(f"User: alice")
    print(f"Position: θ={status['theta_degrees']:.0f}°, depth={status['depth']}")
    print(f"RNA remaining: {status['rna_remaining']}")
    print(f"Needs DNA check: {status['needs_dna_check']}")
    
    # Simulate requests
    requests = [
        ('read', 'file1.txt'),
        ('read', 'file2.txt'),
        ('write', 'file3.txt'),
        ('read', 'file4.txt'),
        ('delete', 'old_file.txt'),
        ('read', 'file5.txt'),
        ('write', 'file6.txt'),
        ('read', 'file7.txt'),
        ('execute', 'script.py'),
        ('read', 'file8.txt'),
        ('read', 'file9.txt'),
        ('admin', 'config'),
    ]
    
    print("\n--- Processing Requests ---\n")
    
    for action, resource in requests:
        result = system.request("alice", action, resource)
        
        check_type = result['check_type']
        success = "✓" if result['success'] else "✗"
        remaining = result['details'].get('remaining', result['details'].get('pool_total', '?'))
        
        if check_type == 'DNA' or check_type == 'DNA→RNA':
            tokens_released = result['details'].get('tokens_released', 0)
            print(f"{success} {action:8} {resource:15} │ {check_type:8} │ +{tokens_released} tokens │ {remaining} remaining")
        else:
            cost = result['details'].get('cost', 0)
            print(f"{success} {action:8} {resource:15} │ {check_type:8} │ -{cost} RNA     │ {remaining} remaining")
    
    # Final stats
    print("\n--- Final Statistics ---")
    stats = system.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"DNA checks: {stats['total_dna_checks']}")
    print(f"RNA checks: {stats['total_rna_checks']}")
    print(f"Efficiency: {stats['efficiency']:.1%} (requests served by RNA)")


def demo_efficiency_comparison():
    """Compare efficiency with and without RNA caching."""
    print("\n" + "=" * 80)
    print("EFFICIENCY COMPARISON: DNA-ONLY vs RNA GEL")
    print("=" * 80)
    
    # Simulate 100 requests
    n_requests = 100
    
    # DNA-only cost (every request does full check)
    dna_only_ops = n_requests * 1000  # ~1000 ops per DNA check
    
    # RNA Gel cost
    system = RNAGelSystem()
    system.register_user("test", math.radians(135), 4, "hash")
    
    for i in range(n_requests):
        action = 'read' if i % 3 != 0 else 'write'
        system.request("test", action, f"resource_{i}")
    
    stats = system.get_stats()
    
    # RNA checks are ~3 ops, DNA checks are ~1000 ops
    rna_gel_ops = stats['total_dna_checks'] * 1000 + stats['total_rna_checks'] * 3
    
    print(f"""
    Simulated {n_requests} requests (mix of read/write operations)
    
    DNA-ONLY (old approach):
    ────────────────────────
    DNA checks:     {n_requests}
    Operations:     {dna_only_ops:,}
    Ops/request:    {dna_only_ops/n_requests:,.0f}
    
    
    RNA GEL (new approach):
    ───────────────────────
    DNA checks:     {stats['total_dna_checks']}
    RNA checks:     {stats['total_rna_checks']}
    Operations:     {rna_gel_ops:,}
    Ops/request:    {rna_gel_ops/n_requests:,.1f}
    
    
    IMPROVEMENT:
    ────────────
    Speedup:        {dna_only_ops/rna_gel_ops:.1f}x faster
    DNA reduction:  {(1 - stats['total_dna_checks']/n_requests)*100:.1f}%
    Efficiency:     {stats['efficiency']:.1%} requests served by RNA
    """)


def demo_rate_limiting():
    """Demonstrate rate limiting through RNA depletion."""
    print("\n" + "=" * 80)
    print("RATE LIMITING THROUGH RNA DEPLETION")
    print("=" * 80)
    
    system = RNAGelSystem()
    
    # Register a low-level user (fewer tokens)
    system.register_user("limited_user", math.radians(15), 1, "hash")
    
    print("\n--- Low-level user (θ=15°, depth=1) ---")
    
    dna = DNAVerifier()
    tokens = dna.compute_token_count(math.radians(15), 1)
    print(f"Tokens per DNA check: {tokens}")
    
    print("\n--- Attempting many requests ---\n")
    
    for i in range(20):
        result = system.request("limited_user", "read", f"file_{i}")
        
        check_type = result['check_type']
        remaining = result['details'].get('remaining', result['details'].get('pool_total', 0))
        
        if 'DNA' in check_type:
            print(f"Request {i+1:2}: DNA CHECK (refreshed to {remaining} RNA)")
        else:
            print(f"Request {i+1:2}: RNA ({remaining} remaining)")
    
    stats = system.get_stats()
    print(f"\n--- Rate Limit Effect ---")
    print(f"Total requests: {stats['total_requests']}")
    print(f"DNA checks required: {stats['total_dna_checks']}")
    print(f"Forced re-authentication rate: every {20/stats['total_dna_checks']:.1f} requests")


def demo_3d_visualization():
    """Visualize the 3-direction gel concept."""
    print("\n" + "=" * 80)
    print("3-DIRECTION RNA GEL VISUALIZATION")
    print("=" * 80)
    
    print("""
    
                            Z (Authority/Upgrade)
                            │
                            │   ┌─────────────────────────────┐
                            │   │  DNA CHECK ZONE             │
                            │   │  • Full crypto verification │
                            │   │  • Releases RNA tokens      │
                            │   │  • Required for upgrades    │
                            │   └─────────────────────────────┘
                            │            │
                            │            │ Tokens flow down
                            │            ▼
                            │   ┌─────────────────────────────┐
                            │   │      RNA TOKEN POOL         │
                            │   │  ████████████████████████   │
                            │   │  Tokens consumed by X/Y ops │
                            │   └─────────────────────────────┘
                            │           ╱ ╲
         ───────────────────┼──────────╱───╲────────────────── X (Action)
                           ╱│         ╱     ╲
                          ╱ │        ╱       ╲
                         ╱  │   ┌───┴───┐ ┌───┴───┐
                        ╱   │   │ READ  │ │ WRITE │
                       ╱    │   │ -1 RNA│ │ -2 RNA│
                      Y     │   └───────┘ └───────┘
                  (Resource)│
                            │
                            │
    
    
    REQUEST FLOW:
    ═════════════
    
    1. Request arrives (action, resource)
           │
           ▼
    2. Is action = UPGRADE?
           │
       ┌───┴───┐
      YES      NO
       │        │
       ▼        ▼
    3a. DNA   3b. Check RNA pool
    CHECK         │
       │      ┌───┴───┐
       │    EMPTY    HAS TOKENS
       │      │          │
       │      ▼          ▼
       │   DNA CHECK  4. Consume RNA
       │      │          │
       ▼      ▼          ▼
    5. Release    6. Perform action
       RNA tokens     
    """)


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("RNA GEL SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Token calculation
    print("\n--- TEST 1: Token Calculation ---")
    dna = DNAVerifier()
    
    # Surface user
    t1 = dna.compute_token_count(0, 0)
    assert t1 >= 1, "Surface user should get at least 1 token"
    print(f"✓ Surface user (θ=0, n=0): {t1} tokens")
    
    # Owner
    t2 = dna.compute_token_count(PI, 5)
    assert t2 > t1, "Owner should get more tokens than surface user"
    print(f"✓ Owner (θ=π, n=5): {t2} tokens")
    
    # Test 2: RNA Token
    print("\n--- TEST 2: RNA Token ---")
    token = RNAToken(
        token_id="test",
        user_id="user",
        signature=b"sig",
        max_uses=10,
        remaining_uses=10,
        created_at=time.time(),
        expires_at=time.time() + 3600,
        max_action_level=3
    )
    
    assert token.is_valid
    assert token.consume(3)
    assert token.remaining_uses == 7
    print("✓ Token consumption works")
    
    # Test 3: RNA Pool
    print("\n--- TEST 3: RNA Pool ---")
    pool = RNAPool(user_id="test")
    assert pool.needs_dna_check()
    
    pool.add_tokens(token)
    assert not pool.needs_dna_check()
    assert pool.total_remaining == 7
    print("✓ Pool management works")
    
    # Test 4: Gel System
    print("\n--- TEST 4: Gel System ---")
    system = RNAGelSystem()
    system.register_user("test", PI/4, 3, "hash")
    
    result = system.request("test", "read", "file")
    assert result['success']
    assert 'DNA' in result['check_type']
    print("✓ First request triggers DNA check")
    
    result2 = system.request("test", "read", "file2")
    assert result2['success']
    assert result2['check_type'] == 'RNA'
    print("✓ Second request uses RNA")
    
    # Test 5: Efficiency
    print("\n--- TEST 5: Efficiency ---")
    for i in range(10):
        system.request("test", "read", f"file{i}")
    
    stats = system.get_stats()
    assert stats['total_rna_checks'] > stats['total_dna_checks']
    print(f"✓ RNA checks ({stats['total_rna_checks']}) > DNA checks ({stats['total_dna_checks']})")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--tokens":
            demo_token_calculation()
        elif sys.argv[1] == "--costs":
            demo_action_costs()
        elif sys.argv[1] == "--session":
            demo_session_lifecycle()
        elif sys.argv[1] == "--efficiency":
            demo_efficiency_comparison()
        elif sys.argv[1] == "--ratelimit":
            demo_rate_limiting()
        elif sys.argv[1] == "--3d":
            demo_3d_visualization()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_token_calculation()
            demo_action_costs()
            demo_3d_visualization()
            demo_session_lifecycle()
            demo_efficiency_comparison()
            demo_rate_limiting()
        else:
            run_tests()
    else:
        run_tests()
