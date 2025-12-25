"""
BEHAVIORAL DNA SYSTEM
=====================
RNA usage patterns become the user's behavioral DNA.

Key Innovations:
1. Track RNA consumption patterns (not just count)
2. Euler split: User RNA vs Owner RNA based on e^(iθ)
3. Accumulate patterns into behavioral DNA profile
4. Compare current RNA to DNA for anomaly detection

The insight: Your behavior IS your identity.
- Humans have irregular timing, varied actions
- Bots have regular timing, repeated actions
- Hackers have probing failures, escalation attempts

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import hashlib
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
import statistics
import secrets

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2

# Pattern analysis windows
RNA_WINDOW_SIZE = 100      # Events to analyze for current behavior
DNA_MIN_EVENTS = 500       # Minimum events before DNA is "stable"
ANOMALY_THRESHOLD = 0.7    # Similarity below this triggers alert


# ═══════════════════════════════════════════════════════════════════════════════
# RNA EVENT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RNAEvent:
    """A single RNA consumption event."""
    timestamp: float
    action: str
    resource: str
    rna_type: str  # 'user' or 'owner'
    cost: int
    success: bool
    time_since_last: float  # Seconds since previous event
    
    # Optional metadata
    ip_hash: str = ""
    device_hash: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'ts': self.timestamp,
            'action': self.action,
            'resource': self.resource[:20],  # Truncate for privacy
            'type': self.rna_type,
            'cost': self.cost,
            'success': self.success,
            'delta': self.time_since_last
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EULER RNA SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

class EulerRNASplitter:
    """
    Splits RNA tokens into User and Owner types based on phase angle.
    
    User RNA: For operations on YOUR resources
    Owner RNA: For operations on SYSTEM resources
    
    Split formula:
        user_fraction = (1 + cos(θ)) / 2    (ranges 1.0 to 0.0)
        owner_fraction = sin(θ)              (ranges 0.0 to 1.0)
    
    Normalized so total = original count (with rounding)
    """
    
    @staticmethod
    def split(total_tokens: int, theta: float) -> Tuple[int, int]:
        """
        Split tokens into (user_rna, owner_rna).
        
        θ = 0°:   100% user, 0% owner (anonymous - only personal access)
        θ = 90°:  50% user, 100% owner (creator - balanced)
        θ = 180°: 0% user, 0% owner wait no...
        
        Better formula that sums to total:
        user = total × cos²(θ/2)
        owner = total × sin²(θ/2)
        
        cos²(x) + sin²(x) = 1 ✓
        """
        half_theta = theta / 2
        
        user_fraction = math.cos(half_theta) ** 2
        owner_fraction = math.sin(half_theta) ** 2
        
        user_tokens = round(total_tokens * user_fraction)
        owner_tokens = round(total_tokens * owner_fraction)
        
        # Ensure total is preserved (rounding adjustment)
        diff = total_tokens - (user_tokens + owner_tokens)
        if diff > 0:
            user_tokens += diff
        elif diff < 0:
            if user_tokens > 0:
                user_tokens += diff
            else:
                owner_tokens += diff
        
        return max(0, user_tokens), max(0, owner_tokens)
    
    @staticmethod
    def get_fractions(theta: float) -> Tuple[float, float]:
        """Get the fractional split for display."""
        half_theta = theta / 2
        return math.cos(half_theta) ** 2, math.sin(half_theta) ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# RNA PATTERN ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BehaviorPattern:
    """Statistical pattern extracted from RNA events."""
    
    # Action distribution
    action_counts: Dict[str, int] = field(default_factory=dict)
    
    # Timing statistics
    timing_mean: float = 0.0
    timing_std: float = 0.0
    timing_min: float = 0.0
    timing_max: float = 0.0
    
    # Resource patterns
    unique_resources: int = 0
    resource_entropy: float = 0.0  # Higher = more varied
    
    # Success patterns
    failure_rate: float = 0.0
    retry_rate: float = 0.0  # Rapid repeated failures
    
    # RNA type usage
    user_rna_fraction: float = 0.5
    owner_rna_fraction: float = 0.5
    
    # Volume
    total_events: int = 0
    events_per_minute: float = 0.0
    
    @property
    def action_distribution(self) -> Dict[str, float]:
        """Get normalized action distribution."""
        total = sum(self.action_counts.values())
        if total == 0:
            return {}
        return {k: v/total for k, v in self.action_counts.items()}
    
    def similarity_to(self, other: 'BehaviorPattern') -> float:
        """
        Compute similarity score to another pattern.
        Returns value in [0, 1] where 1 = identical.
        """
        scores = []
        
        # Action distribution similarity (cosine-like)
        all_actions = set(self.action_counts.keys()) | set(other.action_counts.keys())
        if all_actions:
            my_dist = self.action_distribution
            other_dist = other.action_distribution
            
            dot_product = sum(
                my_dist.get(a, 0) * other_dist.get(a, 0)
                for a in all_actions
            )
            scores.append(dot_product)
        
        # Timing similarity (overlap of distributions)
        if self.timing_std > 0 and other.timing_std > 0:
            mean_diff = abs(self.timing_mean - other.timing_mean)
            avg_std = (self.timing_std + other.timing_std) / 2
            timing_sim = max(0, 1 - mean_diff / (3 * avg_std))
            scores.append(timing_sim)
        
        # Failure rate similarity
        failure_diff = abs(self.failure_rate - other.failure_rate)
        scores.append(max(0, 1 - failure_diff * 5))
        
        # Resource variety similarity
        if self.unique_resources > 0 and other.unique_resources > 0:
            ratio = min(self.unique_resources, other.unique_resources) / \
                   max(self.unique_resources, other.unique_resources)
            scores.append(ratio)
        
        # RNA type usage similarity
        user_diff = abs(self.user_rna_fraction - other.user_rna_fraction)
        scores.append(max(0, 1 - user_diff * 2))
        
        # Weighted average
        if not scores:
            return 0.5
        
        return sum(scores) / len(scores)


class RNAPatternAnalyzer:
    """Analyzes RNA events to extract behavioral patterns."""
    
    @staticmethod
    def analyze(events: List[RNAEvent]) -> BehaviorPattern:
        """Extract pattern from a list of RNA events."""
        pattern = BehaviorPattern()
        
        if not events:
            return pattern
        
        pattern.total_events = len(events)
        
        # Action counts
        for event in events:
            pattern.action_counts[event.action] = \
                pattern.action_counts.get(event.action, 0) + 1
        
        # Timing statistics
        deltas = [e.time_since_last for e in events if e.time_since_last > 0]
        if deltas:
            pattern.timing_mean = statistics.mean(deltas)
            pattern.timing_std = statistics.stdev(deltas) if len(deltas) > 1 else 0
            pattern.timing_min = min(deltas)
            pattern.timing_max = max(deltas)
        
        # Resource patterns
        resources = [e.resource for e in events]
        pattern.unique_resources = len(set(resources))
        
        # Resource entropy (measure of variety)
        if resources:
            from collections import Counter
            counts = Counter(resources)
            total = len(resources)
            entropy = -sum(
                (c/total) * math.log2(c/total) 
                for c in counts.values() if c > 0
            )
            max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
            pattern.resource_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Success patterns
        failures = [e for e in events if not e.success]
        pattern.failure_rate = len(failures) / len(events)
        
        # Retry detection (failures within 0.5s of previous failure)
        retries = 0
        for i, event in enumerate(events[1:], 1):
            if not event.success and not events[i-1].success:
                if event.time_since_last < 0.5:
                    retries += 1
        pattern.retry_rate = retries / max(1, len(failures))
        
        # RNA type usage
        user_events = [e for e in events if e.rna_type == 'user']
        pattern.user_rna_fraction = len(user_events) / len(events)
        pattern.owner_rna_fraction = 1 - pattern.user_rna_fraction
        
        # Events per minute
        if len(events) > 1:
            duration = events[-1].timestamp - events[0].timestamp
            if duration > 0:
                pattern.events_per_minute = len(events) / (duration / 60)
        
        return pattern


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL DNA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BehavioralDNA:
    """
    User's behavioral DNA - accumulated from all RNA usage.
    
    This is the "ground truth" of who the user is based on
    how they actually behave over time.
    """
    user_id: str
    
    # Accumulated pattern (running statistics)
    pattern: BehaviorPattern = field(default_factory=BehaviorPattern)
    
    # History of pattern snapshots (for drift detection)
    pattern_history: List[Tuple[float, BehaviorPattern]] = field(default_factory=list)
    
    # Hash of behavioral signature
    signature_hash: str = ""
    
    # Metadata
    first_event: float = 0.0
    last_event: float = 0.0
    total_events_ever: int = 0
    
    @property
    def is_stable(self) -> bool:
        """Has enough data accumulated for reliable comparison?"""
        return self.total_events_ever >= DNA_MIN_EVENTS
    
    @property
    def age_days(self) -> float:
        """How old is this DNA profile?"""
        if self.first_event == 0:
            return 0
        return (time.time() - self.first_event) / 86400
    
    def update_signature(self):
        """Update the behavioral signature hash."""
        data = {
            'user': self.user_id,
            'actions': self.pattern.action_counts,
            'timing_mean': round(self.pattern.timing_mean, 2),
            'failure_rate': round(self.pattern.failure_rate, 3),
            'events': self.total_events_ever
        }
        self.signature_hash = hashlib.sha256(
            str(data).encode()
        ).hexdigest()[:32]


# ═══════════════════════════════════════════════════════════════════════════════
# DNA TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class BehavioralDNATracker:
    """
    Tracks RNA usage and builds behavioral DNA profiles.
    Detects anomalies by comparing current behavior to DNA.
    """
    
    def __init__(self):
        # Per-user DNA profiles
        self.dna_profiles: Dict[str, BehavioralDNA] = {}
        
        # Current RNA windows (sliding window of recent events)
        self.rna_windows: Dict[str, Deque[RNAEvent]] = {}
        
        # Anomaly history
        self.anomalies: Dict[str, List[Dict]] = {}
        
        # Stats
        self.total_events = 0
        self.total_anomalies = 0
    
    def get_or_create_profile(self, user_id: str) -> BehavioralDNA:
        """Get or create DNA profile for user."""
        if user_id not in self.dna_profiles:
            self.dna_profiles[user_id] = BehavioralDNA(user_id=user_id)
            self.rna_windows[user_id] = deque(maxlen=RNA_WINDOW_SIZE)
            self.anomalies[user_id] = []
        return self.dna_profiles[user_id]
    
    def record_event(self, 
                    user_id: str,
                    action: str,
                    resource: str,
                    rna_type: str,
                    cost: int,
                    success: bool) -> Dict[str, Any]:
        """
        Record an RNA consumption event.
        Returns analysis including any anomaly detection.
        """
        self.total_events += 1
        
        profile = self.get_or_create_profile(user_id)
        window = self.rna_windows[user_id]
        
        # Calculate time since last event
        time_since_last = 0.0
        if window:
            time_since_last = time.time() - window[-1].timestamp
        
        # Create event
        event = RNAEvent(
            timestamp=time.time(),
            action=action,
            resource=resource,
            rna_type=rna_type,
            cost=cost,
            success=success,
            time_since_last=time_since_last
        )
        
        # Add to window
        window.append(event)
        
        # Update DNA profile
        if profile.first_event == 0:
            profile.first_event = event.timestamp
        profile.last_event = event.timestamp
        profile.total_events_ever += 1
        
        # Update action counts in DNA
        profile.pattern.action_counts[action] = \
            profile.pattern.action_counts.get(action, 0) + 1
        profile.pattern.total_events = profile.total_events_ever
        
        # Periodically update full DNA pattern (every 100 events)
        if profile.total_events_ever % 100 == 0:
            self._update_dna_pattern(user_id, profile, list(window))
        
        # Check for anomalies (if DNA is stable)
        anomaly_result = None
        if profile.is_stable and len(window) >= 20:
            anomaly_result = self._check_anomaly(user_id, profile, list(window))
        
        return {
            'event_recorded': True,
            'total_events': profile.total_events_ever,
            'dna_stable': profile.is_stable,
            'anomaly': anomaly_result
        }
    
    def _update_dna_pattern(self, user_id: str, profile: BehavioralDNA, 
                           recent_events: List[RNAEvent]):
        """Update the DNA pattern with recent events."""
        if not recent_events:
            return
        
        recent_pattern = RNAPatternAnalyzer.analyze(recent_events)
        
        # Exponential moving average for timing stats
        alpha = 0.1  # Weight for new data
        
        if profile.pattern.timing_mean > 0:
            profile.pattern.timing_mean = (
                (1 - alpha) * profile.pattern.timing_mean +
                alpha * recent_pattern.timing_mean
            )
            profile.pattern.timing_std = (
                (1 - alpha) * profile.pattern.timing_std +
                alpha * recent_pattern.timing_std
            )
        else:
            profile.pattern.timing_mean = recent_pattern.timing_mean
            profile.pattern.timing_std = recent_pattern.timing_std
        
        # Update failure rate (EMA)
        profile.pattern.failure_rate = (
            (1 - alpha) * profile.pattern.failure_rate +
            alpha * recent_pattern.failure_rate
        )
        
        # Save snapshot
        profile.pattern_history.append((time.time(), recent_pattern))
        
        # Keep only last 100 snapshots
        if len(profile.pattern_history) > 100:
            profile.pattern_history = profile.pattern_history[-100:]
        
        # Update signature
        profile.update_signature()
    
    def _check_anomaly(self, user_id: str, profile: BehavioralDNA,
                      recent_events: List[RNAEvent]) -> Optional[Dict]:
        """Compare current behavior to DNA profile."""
        current_pattern = RNAPatternAnalyzer.analyze(recent_events)
        similarity = current_pattern.similarity_to(profile.pattern)
        
        result = {
            'similarity': similarity,
            'threshold': ANOMALY_THRESHOLD,
            'is_anomaly': similarity < ANOMALY_THRESHOLD
        }
        
        if result['is_anomaly']:
            self.total_anomalies += 1
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly(current_pattern, profile.pattern)
            
            anomaly_record = {
                'timestamp': time.time(),
                'similarity': similarity,
                'type': anomaly_type,
                'details': {
                    'current_timing': current_pattern.timing_mean,
                    'expected_timing': profile.pattern.timing_mean,
                    'current_failure_rate': current_pattern.failure_rate,
                    'expected_failure_rate': profile.pattern.failure_rate
                }
            }
            
            self.anomalies[user_id].append(anomaly_record)
            result['anomaly_type'] = anomaly_type
            result['details'] = anomaly_record['details']
        
        return result
    
    def _classify_anomaly(self, current: BehaviorPattern, 
                         expected: BehaviorPattern) -> str:
        """Classify the type of anomaly."""
        
        # Bot detection: Very regular timing
        if current.timing_std < expected.timing_std * 0.3:
            return "BOT_TIMING"
        
        # Brute force: High failure rate with retries
        if current.failure_rate > expected.failure_rate * 3 and current.retry_rate > 0.5:
            return "BRUTE_FORCE"
        
        # Account takeover: Different action distribution
        action_overlap = sum(
            min(current.action_distribution.get(a, 0), 
                expected.action_distribution.get(a, 0))
            for a in set(current.action_counts.keys()) | set(expected.action_counts.keys())
        )
        if action_overlap < 0.5:
            return "ACCOUNT_TAKEOVER"
        
        # Unusual resources
        if current.resource_entropy < expected.resource_entropy * 0.5:
            return "FOCUSED_ATTACK"
        
        # High velocity
        if current.events_per_minute > expected.events_per_minute * 3:
            return "HIGH_VELOCITY"
        
        return "BEHAVIORAL_DRIFT"
    
    def get_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's behavioral DNA."""
        profile = self.dna_profiles.get(user_id)
        if not profile:
            return {'error': 'Unknown user'}
        
        return {
            'user_id': user_id,
            'dna_stable': profile.is_stable,
            'total_events': profile.total_events_ever,
            'age_days': round(profile.age_days, 1),
            'signature': profile.signature_hash,
            'pattern': {
                'action_distribution': profile.pattern.action_distribution,
                'timing_mean': round(profile.pattern.timing_mean, 2),
                'timing_std': round(profile.pattern.timing_std, 2),
                'failure_rate': round(profile.pattern.failure_rate, 3),
                'events_per_minute': round(profile.pattern.events_per_minute, 2)
            },
            'anomaly_count': len(self.anomalies.get(user_id, []))
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE BEHAVIORAL RNA SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DualRNAPool:
    """Pool with separate User and Owner RNA."""
    user_id: str
    user_rna: int = 0
    owner_rna: int = 0
    last_refill: float = 0.0
    
    @property
    def total(self) -> int:
        return self.user_rna + self.owner_rna
    
    def refill(self, user_tokens: int, owner_tokens: int):
        """Add tokens from DNA check."""
        self.user_rna += user_tokens
        self.owner_rna += owner_tokens
        self.last_refill = time.time()
    
    def consume(self, cost: int, rna_type: str) -> Tuple[bool, str]:
        """Consume tokens of specified type."""
        if rna_type == 'user':
            if self.user_rna >= cost:
                self.user_rna -= cost
                return True, f"Consumed {cost} User RNA, {self.user_rna} remaining"
            return False, f"Insufficient User RNA: need {cost}, have {self.user_rna}"
        else:
            if self.owner_rna >= cost:
                self.owner_rna -= cost
                return True, f"Consumed {cost} Owner RNA, {self.owner_rna} remaining"
            return False, f"Insufficient Owner RNA: need {cost}, have {self.owner_rna}"


class BehavioralRNASystem:
    """
    Complete system with:
    - Euler-split RNA (User vs Owner)
    - Behavioral DNA tracking
    - Anomaly detection
    """
    
    # Resource classification
    USER_RESOURCES = {'my_', 'personal_', 'profile_', 'settings_'}
    OWNER_RESOURCES = {'admin_', 'system_', 'user_', 'config_', 'audit_'}
    
    def __init__(self):
        self.pools: Dict[str, DualRNAPool] = {}
        self.user_positions: Dict[str, Dict] = {}
        self.dna_tracker = BehavioralDNATracker()
        self.splitter = EulerRNASplitter()
        
        self.total_requests = 0
        self.dna_checks = 0
    
    def register_user(self, user_id: str, theta: float, depth: int):
        """Register user with their security position."""
        self.user_positions[user_id] = {'theta': theta, 'depth': depth}
        self.pools[user_id] = DualRNAPool(user_id=user_id)
    
    def classify_resource(self, resource: str) -> str:
        """Determine if resource requires User or Owner RNA."""
        for prefix in self.OWNER_RESOURCES:
            if resource.startswith(prefix):
                return 'owner'
        return 'user'
    
    def do_dna_check(self, user_id: str) -> Dict[str, Any]:
        """Perform DNA check and refill RNA pool."""
        self.dna_checks += 1
        
        position = self.user_positions.get(user_id)
        if not position:
            return {'error': 'Unknown user'}
        
        theta = position['theta']
        depth = position['depth']
        
        # Calculate total tokens (φ^n formula)
        total = int(PHI ** depth * (1 + theta / PI))
        
        # Split into User and Owner RNA
        user_tokens, owner_tokens = self.splitter.split(total, theta)
        
        # Refill pool
        pool = self.pools[user_id]
        pool.refill(user_tokens, owner_tokens)
        
        return {
            'dna_check': True,
            'total_tokens': total,
            'user_rna': user_tokens,
            'owner_rna': owner_tokens,
            'theta_degrees': math.degrees(theta),
            'user_fraction': user_tokens / max(1, total),
            'owner_fraction': owner_tokens / max(1, total)
        }
    
    def request(self, user_id: str, action: str, resource: str) -> Dict[str, Any]:
        """Process a request with behavioral tracking."""
        self.total_requests += 1
        
        pool = self.pools.get(user_id)
        if not pool:
            return {'error': 'Unknown user'}
        
        # Determine RNA type needed
        rna_type = self.classify_resource(resource)
        
        # Action costs
        costs = {'read': 1, 'write': 2, 'delete': 3, 'execute': 3, 'admin': 5}
        cost = costs.get(action, 1)
        
        result = {
            'request_id': self.total_requests,
            'action': action,
            'resource': resource,
            'rna_type': rna_type,
            'cost': cost
        }
        
        # Check if pool has enough
        if rna_type == 'user' and pool.user_rna < cost:
            # Need DNA check
            dna_result = self.do_dna_check(user_id)
            result['dna_check'] = dna_result
        elif rna_type == 'owner' and pool.owner_rna < cost:
            # Need DNA check
            dna_result = self.do_dna_check(user_id)
            result['dna_check'] = dna_result
        
        # Consume RNA
        success, message = pool.consume(cost, rna_type)
        result['success'] = success
        result['message'] = message
        result['pool'] = {'user': pool.user_rna, 'owner': pool.owner_rna}
        
        # Record in DNA tracker
        tracking = self.dna_tracker.record_event(
            user_id=user_id,
            action=action,
            resource=resource,
            rna_type=rna_type,
            cost=cost,
            success=success
        )
        result['tracking'] = tracking
        
        return result
    
    def get_status(self, user_id: str) -> Dict[str, Any]:
        """Get complete status for user."""
        pool = self.pools.get(user_id)
        position = self.user_positions.get(user_id)
        
        if not pool or not position:
            return {'error': 'Unknown user'}
        
        dna_summary = self.dna_tracker.get_profile_summary(user_id)
        
        return {
            'user_id': user_id,
            'theta_degrees': math.degrees(position['theta']),
            'depth': position['depth'],
            'pool': {
                'user_rna': pool.user_rna,
                'owner_rna': pool.owner_rna,
                'total': pool.total
            },
            'dna': dna_summary
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_euler_split():
    """Demonstrate Euler split of RNA tokens."""
    print("\n" + "=" * 80)
    print("EULER SPLIT: USER vs OWNER RNA")
    print("=" * 80)
    
    print("""
    Formula: e^(iθ) = cos(θ) + i·sin(θ)
    
    Split using cos²(θ/2) and sin²(θ/2) to ensure sum = total
    
    User RNA:  For YOUR resources (personal data, your projects)
    Owner RNA: For SYSTEM resources (admin, other users, config)
    """)
    
    splitter = EulerRNASplitter()
    
    print("┌─────────────────────┬───────┬────────────┬────────────┬─────────────┐")
    print("│ User Type           │ θ     │ User RNA   │ Owner RNA  │ Split       │")
    print("├─────────────────────┼───────┼────────────┼────────────┼─────────────┤")
    
    users = [
        ("Anonymous", 0),
        ("Registered", 15),
        ("Paid User", 30),
        ("Long-term", 45),
        ("Core User", 60),
        ("Creator", 90),
        ("Staff", 120),
        ("Admin", 150),
        ("Owner", 180),
    ]
    
    total = 20  # Example with 20 tokens
    
    for name, theta_deg in users:
        theta = math.radians(theta_deg)
        user_rna, owner_rna = splitter.split(total, theta)
        user_frac, owner_frac = splitter.get_fractions(theta)
        
        bar_user = "█" * user_rna
        bar_owner = "▒" * owner_rna
        
        print(f"│ {name:19} │ {theta_deg:4}° │ {user_rna:5} ({user_frac*100:4.0f}%) │ {owner_rna:5} ({owner_frac*100:4.0f}%) │ {bar_user}{bar_owner:>11} │")
    
    print("└─────────────────────┴───────┴────────────┴────────────┴─────────────┘")
    print("█ = User RNA,  ▒ = Owner RNA")


def demo_behavioral_patterns():
    """Demonstrate behavioral pattern detection."""
    print("\n" + "=" * 80)
    print("BEHAVIORAL PATTERNS: HUMAN vs BOT vs HACKER")
    print("=" * 80)
    
    # Generate synthetic events for different user types
    def generate_human_events(n=100):
        events = []
        actions = ['read'] * 60 + ['write'] * 30 + ['delete'] * 10
        t = time.time()
        for i in range(n):
            events.append(RNAEvent(
                timestamp=t + i * random.gauss(2.5, 1.5),
                action=random.choice(actions),
                resource=f"file_{random.randint(1, 30)}.txt",
                rna_type='user',
                cost=1,
                success=random.random() > 0.05,  # 5% failure
                time_since_last=random.gauss(2.5, 1.5) if i > 0 else 0
            ))
        return events
    
    def generate_bot_events(n=100):
        events = []
        t = time.time()
        for i in range(n):
            events.append(RNAEvent(
                timestamp=t + i * 0.5,  # Very regular
                action='read',  # Always same action
                resource=f"file_{i % 5}.txt",  # Limited resources
                rna_type='user',
                cost=1,
                success=True,  # Always succeeds
                time_since_last=0.5 if i > 0 else 0  # Constant timing
            ))
        return events
    
    def generate_hacker_events(n=100):
        events = []
        t = time.time()
        for i in range(n):
            events.append(RNAEvent(
                timestamp=t + i * 0.1,  # Very fast
                action=random.choice(['delete', 'admin', 'write']),  # Destructive
                resource=f"admin_config_{i}.txt",  # System resources
                rna_type='owner',
                cost=3,
                success=random.random() > 0.7,  # Many failures
                time_since_last=0.1 if i > 0 else 0
            ))
        return events
    
    human_events = generate_human_events()
    bot_events = generate_bot_events()
    hacker_events = generate_hacker_events()
    
    human_pattern = RNAPatternAnalyzer.analyze(human_events)
    bot_pattern = RNAPatternAnalyzer.analyze(bot_events)
    hacker_pattern = RNAPatternAnalyzer.analyze(hacker_events)
    
    print("\n--- HUMAN PATTERN ---")
    print(f"Actions: {human_pattern.action_distribution}")
    print(f"Timing: μ={human_pattern.timing_mean:.2f}s, σ={human_pattern.timing_std:.2f}s")
    print(f"Failure rate: {human_pattern.failure_rate:.1%}")
    print(f"Unique resources: {human_pattern.unique_resources}")
    
    print("\n--- BOT PATTERN ---")
    print(f"Actions: {bot_pattern.action_distribution}")
    print(f"Timing: μ={bot_pattern.timing_mean:.2f}s, σ={bot_pattern.timing_std:.2f}s")
    print(f"Failure rate: {bot_pattern.failure_rate:.1%}")
    print(f"Unique resources: {bot_pattern.unique_resources}")
    
    print("\n--- HACKER PATTERN ---")
    print(f"Actions: {hacker_pattern.action_distribution}")
    print(f"Timing: μ={hacker_pattern.timing_mean:.2f}s, σ={hacker_pattern.timing_std:.2f}s")
    print(f"Failure rate: {hacker_pattern.failure_rate:.1%}")
    print(f"Unique resources: {hacker_pattern.unique_resources}")
    
    print("\n--- SIMILARITY MATRIX ---")
    print("(Higher = more similar)")
    print(f"Human vs Bot:    {human_pattern.similarity_to(bot_pattern):.2%}")
    print(f"Human vs Hacker: {human_pattern.similarity_to(hacker_pattern):.2%}")
    print(f"Bot vs Hacker:   {bot_pattern.similarity_to(hacker_pattern):.2%}")


def demo_anomaly_detection():
    """Demonstrate anomaly detection with behavioral DNA."""
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION: DNA vs CURRENT RNA")
    print("=" * 80)
    
    tracker = BehavioralDNATracker()
    
    # Build up DNA profile with normal behavior
    print("\n--- Building DNA profile with normal behavior ---")
    
    actions = ['read'] * 60 + ['write'] * 30 + ['delete'] * 10
    
    for i in range(600):  # 600 events to get stable DNA
        action = random.choice(actions)
        tracker.record_event(
            user_id="alice",
            action=action,
            resource=f"file_{random.randint(1, 50)}.txt",
            rna_type='user',
            cost=1,
            success=random.random() > 0.05
        )
        time.sleep(0.001)  # Small delay for timestamp variation
    
    profile = tracker.get_profile_summary("alice")
    print(f"DNA stable: {profile['dna_stable']}")
    print(f"Total events: {profile['total_events']}")
    print(f"Signature: {profile['signature']}")
    print(f"Pattern: {profile['pattern']}")
    
    # Now simulate account takeover (different behavior)
    print("\n--- Simulating account takeover ---")
    
    anomalies_detected = 0
    
    for i in range(50):
        # Hacker behavior: lots of admin actions, fast, many failures
        result = tracker.record_event(
            user_id="alice",
            action=random.choice(['admin', 'delete', 'execute']),
            resource=f"system_config_{i}.txt",
            rna_type='owner',
            cost=3,
            success=random.random() > 0.5
        )
        
        if result.get('anomaly') and result['anomaly'].get('is_anomaly'):
            anomalies_detected += 1
            print(f"  ANOMALY at event {i+601}: {result['anomaly'].get('anomaly_type')}")
            print(f"    Similarity: {result['anomaly']['similarity']:.2%}")
    
    print(f"\nTotal anomalies detected: {anomalies_detected}")


def demo_complete_system():
    """Demonstrate complete behavioral RNA system."""
    print("\n" + "=" * 80)
    print("COMPLETE BEHAVIORAL RNA SYSTEM")
    print("=" * 80)
    
    system = BehavioralRNASystem()
    
    # Register users at different levels
    system.register_user("anonymous", 0, 1)
    system.register_user("staff", math.radians(135), 4)
    
    print("\n--- Initial DNA Check ---")
    
    for uid in ["anonymous", "staff"]:
        result = system.do_dna_check(uid)
        status = system.get_status(uid)
        print(f"\n{uid}:")
        print(f"  θ = {result['theta_degrees']:.0f}°")
        print(f"  Total tokens: {result['total_tokens']}")
        print(f"  User RNA: {result['user_rna']} ({result['user_fraction']:.0%})")
        print(f"  Owner RNA: {result['owner_rna']} ({result['owner_fraction']:.0%})")
    
    print("\n--- Anonymous User Operations ---")
    
    for i in range(5):
        # Anonymous can only do user operations
        result = system.request("anonymous", "read", f"my_file_{i}.txt")
        print(f"read my_file_{i}.txt: {result['message']}")
    
    # Try owner operation (should fail or need DNA check)
    result = system.request("anonymous", "read", "admin_users.txt")
    print(f"read admin_users.txt: {result['message']}")
    
    print("\n--- Staff User Operations ---")
    
    for i in range(3):
        result = system.request("staff", "read", f"my_file_{i}.txt")
        print(f"read my_file_{i}: User RNA = {result['pool']['user']}")
    
    for i in range(3):
        result = system.request("staff", "admin", f"system_config_{i}.txt")
        print(f"admin system_config_{i}: Owner RNA = {result['pool']['owner']}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("BEHAVIORAL DNA SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Euler split
    print("\n--- TEST 1: Euler Split ---")
    splitter = EulerRNASplitter()
    
    # At θ=0, should be all user
    u, o = splitter.split(10, 0)
    assert u == 10 and o == 0, f"Expected (10,0), got ({u},{o})"
    print(f"✓ θ=0°: {u} user, {o} owner")
    
    # At θ=π, should be all owner
    u, o = splitter.split(10, PI)
    assert u == 0 and o == 10, f"Expected (0,10), got ({u},{o})"
    print(f"✓ θ=180°: {u} user, {o} owner")
    
    # At θ=π/2, should be 50/50
    u, o = splitter.split(10, PI/2)
    assert u == 5 and o == 5, f"Expected (5,5), got ({u},{o})"
    print(f"✓ θ=90°: {u} user, {o} owner")
    
    # Test 2: Pattern analyzer
    print("\n--- TEST 2: Pattern Analyzer ---")
    events = [
        RNAEvent(time.time(), 'read', 'file1', 'user', 1, True, 0),
        RNAEvent(time.time() + 1, 'read', 'file2', 'user', 1, True, 1.0),
        RNAEvent(time.time() + 2, 'write', 'file1', 'user', 2, True, 1.0),
    ]
    pattern = RNAPatternAnalyzer.analyze(events)
    assert 'read' in pattern.action_counts
    assert pattern.total_events == 3
    print(f"✓ Pattern extracted: {pattern.action_distribution}")
    
    # Test 3: DNA tracker
    print("\n--- TEST 3: DNA Tracker ---")
    tracker = BehavioralDNATracker()
    
    for i in range(10):
        tracker.record_event("test", "read", f"file_{i}", "user", 1, True)
    
    profile = tracker.get_profile_summary("test")
    assert profile['total_events'] == 10
    print(f"✓ DNA tracker recorded {profile['total_events']} events")
    
    # Test 4: Dual RNA pool
    print("\n--- TEST 4: Dual RNA Pool ---")
    pool = DualRNAPool("test")
    pool.refill(10, 5)
    
    success, _ = pool.consume(3, 'user')
    assert success and pool.user_rna == 7
    print(f"✓ User RNA consumed: {pool.user_rna} remaining")
    
    success, _ = pool.consume(2, 'owner')
    assert success and pool.owner_rna == 3
    print(f"✓ Owner RNA consumed: {pool.owner_rna} remaining")
    
    # Test 5: Complete system
    print("\n--- TEST 5: Complete System ---")
    system = BehavioralRNASystem()
    system.register_user("test", PI/4, 3)
    system.do_dna_check("test")
    
    result = system.request("test", "read", "my_file.txt")
    assert result['success']
    print(f"✓ Request processed: {result['message']}")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--split":
            demo_euler_split()
        elif sys.argv[1] == "--patterns":
            demo_behavioral_patterns()
        elif sys.argv[1] == "--anomaly":
            demo_anomaly_detection()
        elif sys.argv[1] == "--system":
            demo_complete_system()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_euler_split()
            demo_behavioral_patterns()
            demo_anomaly_detection()
            demo_complete_system()
        else:
            run_tests()
    else:
        run_tests()
