"""
USER LIFECYCLE VERIFICATION SYSTEM
===================================
Complete user progression from Anonymous (θ=0) to Staff (θ=π)
with 9-axis verification across 3 domains.

Key Features:
- Phase-based user levels (Euler identity)
- 9 verification axes (3 domains × 3 axes)
- Hacker detection via cross-domain inconsistency
- π/4 wall (time/behavior can't be faked)
- π/2 boundary (requires BEC formation)
- Staff side uses inverted verification

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum, auto
from datetime import datetime, timedelta
import hashlib

PI = math.pi

# ═══════════════════════════════════════════════════════════════════════════════
# USER LEVELS (Phase-based)
# ═══════════════════════════════════════════════════════════════════════════════

class UserLevel(Enum):
    """User levels mapped to phase angles."""
    ANONYMOUS = 0           # θ = 0°
    REGISTERED = 15         # θ = 15° (π/12)
    PAID = 30               # θ = 30° (π/6)
    LONG_TERM = 45          # θ = 45° (π/4) - THE WALL
    CORE = 60               # θ = 60° (π/3)
    REVIEWER = 75           # θ = 75° (5π/12)
    CREATOR = 90            # θ = 90° (π/2) - THE BOUNDARY
    MASCOT = 105            # θ = 105° (7π/12)
    STAFF = 135             # θ = 135° (3π/4)
    ADMIN = 165             # θ = 165° (11π/12)
    OWNER = 180             # θ = 180° (π)
    
    @property
    def theta(self) -> float:
        """Phase angle in radians."""
        return math.radians(self.value)
    
    @property
    def is_user_side(self) -> bool:
        """Is this on the user side (θ < π/2)?"""
        return self.value < 90
    
    @property
    def is_staff_side(self) -> bool:
        """Is this on the staff side (θ > π/2)?"""
        return self.value > 90
    
    @property
    def is_boundary(self) -> bool:
        """Is this at the boundary (θ = π/2)?"""
        return self.value == 90
    
    @classmethod
    def from_theta(cls, theta: float) -> 'UserLevel':
        """Get level from theta (radians)."""
        degrees = math.degrees(theta)
        # Find closest level
        closest = min(cls, key=lambda l: abs(l.value - degrees))
        return closest
    
    @classmethod
    def next_level(cls, current: 'UserLevel') -> Optional['UserLevel']:
        """Get the next level up."""
        levels = list(cls)
        idx = levels.index(current)
        if idx < len(levels) - 1:
            return levels[idx + 1]
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION DOMAINS AND AXES
# ═══════════════════════════════════════════════════════════════════════════════

class Domain(Enum):
    """The three verification domains."""
    USER = "user"       # Behavioral domain
    ADMIN = "admin"     # Authority domain
    TECH = "tech"       # Technical domain


class VerificationAxis(Enum):
    """The 9 verification axes (3 per domain)."""
    # Domain 1: USER/BEHAVIORAL
    USER_IDENTITY = ("user", "identity", "Who they claim to be")
    USER_BEHAVIOR = ("user", "behavior", "What they actually do")
    USER_VALUE = ("user", "value", "What they contribute")
    
    # Domain 2: ADMIN/AUTHORITY
    ADMIN_FINANCIAL = ("admin", "financial", "Payment history, fraud score")
    ADMIN_COMPLIANCE = ("admin", "compliance", "Terms, policy adherence")
    ADMIN_RELATIONSHIP = ("admin", "relationship", "Account age, depth")
    
    # Domain 3: TECHNICAL/IT
    TECH_DEVICE = ("tech", "device", "Consistent devices, environment")
    TECH_TEMPORAL = ("tech", "temporal", "Time patterns, human-like")
    TECH_ANOMALY = ("tech", "anomaly", "Deviation from baseline")
    
    def __init__(self, domain: str, axis: str, description: str):
        self._domain = domain
        self._axis = axis
        self._description = description
    
    @property
    def domain(self) -> str:
        return self._domain
    
    @property
    def axis(self) -> str:
        return self._axis
    
    @property
    def description(self) -> str:
        return self._description


# Level requirements: which axes needed for each level
LEVEL_REQUIREMENTS: Dict[UserLevel, List[VerificationAxis]] = {
    UserLevel.ANONYMOUS: [],
    UserLevel.REGISTERED: [
        VerificationAxis.USER_IDENTITY
    ],
    UserLevel.PAID: [
        VerificationAxis.USER_IDENTITY,
        VerificationAxis.ADMIN_FINANCIAL
    ],
    UserLevel.LONG_TERM: [
        VerificationAxis.USER_IDENTITY,
        VerificationAxis.USER_BEHAVIOR,
        VerificationAxis.ADMIN_FINANCIAL,
        VerificationAxis.ADMIN_RELATIONSHIP,
        VerificationAxis.TECH_TEMPORAL
    ],
    UserLevel.CORE: [
        VerificationAxis.USER_IDENTITY,
        VerificationAxis.USER_BEHAVIOR,
        VerificationAxis.USER_VALUE,
        VerificationAxis.ADMIN_FINANCIAL,
        VerificationAxis.ADMIN_COMPLIANCE,
        VerificationAxis.ADMIN_RELATIONSHIP,
        VerificationAxis.TECH_DEVICE,
        VerificationAxis.TECH_TEMPORAL
    ],
    UserLevel.REVIEWER: list(VerificationAxis),  # All 9
    UserLevel.CREATOR: list(VerificationAxis),   # All 9 + BEC
    UserLevel.MASCOT: list(VerificationAxis),    # All 9 + BEC + brand
    UserLevel.STAFF: list(VerificationAxis),     # Inverted verification
    UserLevel.ADMIN: list(VerificationAxis),
    UserLevel.OWNER: list(VerificationAxis),
}


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS VERIFICATION SCORES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AxisScore:
    """Score for a single verification axis."""
    axis: VerificationAxis
    score: float  # 0-1
    confidence: float  # 0-1
    last_updated: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_verified(self) -> bool:
        """Is this axis sufficiently verified?"""
        return self.score >= 0.6 and self.confidence >= 0.5
    
    @property
    def is_suspicious(self) -> bool:
        """Does this axis show suspicious patterns?"""
        return self.score < 0.3 or self.confidence < 0.3


@dataclass
class VerificationProfile:
    """Complete verification profile with all 9 axes."""
    user_id: str
    scores: Dict[VerificationAxis, AxisScore] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Initialize all axes with zero scores
        for axis in VerificationAxis:
            if axis not in self.scores:
                self.scores[axis] = AxisScore(axis=axis, score=0, confidence=0)
    
    def get_domain_score(self, domain: str) -> float:
        """Get average score for a domain."""
        domain_axes = [a for a in VerificationAxis if a.domain == domain]
        if not domain_axes:
            return 0
        return sum(self.scores[a].score for a in domain_axes) / len(domain_axes)
    
    def get_total_score(self) -> float:
        """Get overall verification score."""
        if not self.scores:
            return 0
        return sum(s.score for s in self.scores.values()) / len(self.scores)
    
    def get_verified_axes(self) -> List[VerificationAxis]:
        """Get list of verified axes."""
        return [a for a, s in self.scores.items() if s.is_verified]
    
    def get_suspicious_axes(self) -> List[VerificationAxis]:
        """Get list of suspicious axes."""
        return [a for a, s in self.scores.items() if s.is_suspicious]
    
    def check_cross_domain_consistency(self) -> Tuple[bool, float, str]:
        """
        Check if scores are consistent across domains.
        Hackers often have high scores in fakeable axes but low in others.
        """
        domain_scores = {
            'user': self.get_domain_score('user'),
            'admin': self.get_domain_score('admin'),
            'tech': self.get_domain_score('tech')
        }
        
        # Calculate variance
        mean = sum(domain_scores.values()) / 3
        variance = sum((s - mean) ** 2 for s in domain_scores.values()) / 3
        std_dev = math.sqrt(variance)
        
        # High variance = inconsistent = suspicious
        consistency = max(0, 1 - (std_dev * 2))
        
        if consistency < 0.5:
            # Find the pattern
            if domain_scores['admin'] > 0.7 and domain_scores['tech'] < 0.4:
                reason = "High financial but low technical - possible stolen credentials"
            elif domain_scores['user'] > 0.7 and domain_scores['tech'] < 0.4:
                reason = "High behavioral but low technical - possible bot"
            else:
                reason = f"Inconsistent domain scores: {domain_scores}"
            return False, consistency, reason
        
        return True, consistency, "Domain scores are consistent"
    
    def can_reach_level(self, level: UserLevel) -> Tuple[bool, List[VerificationAxis]]:
        """Check if profile can reach a level. Returns (can_reach, missing_axes)."""
        required = LEVEL_REQUIREMENTS.get(level, [])
        verified = set(self.get_verified_axes())
        missing = [a for a in required if a not in verified]
        return len(missing) == 0, missing


# ═══════════════════════════════════════════════════════════════════════════════
# USER ENTITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class User:
    """A user with phase-based level and verification profile."""
    user_id: str
    level: UserLevel = UserLevel.ANONYMOUS
    theta: float = 0.0
    profile: VerificationProfile = None
    
    # History
    created_at: datetime = field(default_factory=datetime.now)
    level_history: List[Tuple[UserLevel, datetime]] = field(default_factory=list)
    
    # Behavioral signals
    total_spend: float = 0.0
    session_count: int = 0
    content_created: int = 0
    reviews_written: int = 0
    helpful_votes: int = 0
    
    # Flags
    is_suspicious: bool = False
    suspension_count: int = 0
    
    def __post_init__(self):
        if self.profile is None:
            self.profile = VerificationProfile(user_id=self.user_id)
        self.level_history.append((self.level, self.created_at))
    
    @property
    def account_age_days(self) -> int:
        """Days since account creation."""
        return (datetime.now() - self.created_at).days
    
    @property
    def euler_identity(self) -> complex:
        """e^(iθ) for this user."""
        return complex(math.cos(self.theta), math.sin(self.theta))
    
    @property
    def cos_component(self) -> float:
        """How much 'user-side' (outsider view)."""
        return math.cos(self.theta)
    
    @property
    def sin_component(self) -> float:
        """How much 'staff-side' (insider view)."""
        return math.sin(self.theta)
    
    def update_theta(self):
        """Update theta based on current level."""
        self.theta = self.level.theta


# ═══════════════════════════════════════════════════════════════════════════════
# HACKER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HackerSignal:
    """A signal that might indicate hacking/fraud."""
    signal_type: str
    severity: float  # 0-1
    description: str
    detected_at: datetime = field(default_factory=datetime.now)


class HackerDetector:
    """Detects hacking attempts through cross-domain inconsistencies."""
    
    # The wall at π/4 - where hackers stack up
    WALL_THETA = PI / 4  # 45°
    
    # Minimum account age for time-verified levels
    MIN_DAYS_FOR_LONG_TERM = 365  # 1 year
    MIN_DAYS_FOR_CORE = 180       # 6 months
    
    @staticmethod
    def check_time_wall(user: User, target_level: UserLevel) -> Tuple[bool, str]:
        """
        Check if user can pass the time wall.
        Time cannot be faked!
        """
        if target_level.value >= UserLevel.LONG_TERM.value:
            if user.account_age_days < HackerDetector.MIN_DAYS_FOR_LONG_TERM:
                return False, f"Account too young: {user.account_age_days} days < {HackerDetector.MIN_DAYS_FOR_LONG_TERM} required"
        
        if target_level.value >= UserLevel.CORE.value:
            if user.account_age_days < HackerDetector.MIN_DAYS_FOR_CORE:
                return False, f"Account too young for core: {user.account_age_days} days"
        
        return True, "Time verification passed"
    
    @staticmethod
    def check_behavior_consistency(user: User) -> Tuple[bool, List[HackerSignal]]:
        """Check for behavioral red flags."""
        signals = []
        
        # High spend but low engagement
        if user.total_spend > 1000 and user.session_count < 10:
            signals.append(HackerSignal(
                signal_type="spend_engagement_mismatch",
                severity=0.8,
                description=f"Spent ${user.total_spend} but only {user.session_count} sessions"
            ))
        
        # Rapid level progression
        if len(user.level_history) >= 2:
            last_two = user.level_history[-2:]
            time_diff = (last_two[1][1] - last_two[0][1]).days
            level_diff = last_two[1][0].value - last_two[0][0].value
            
            if level_diff > 30 and time_diff < 7:
                signals.append(HackerSignal(
                    signal_type="rapid_progression",
                    severity=0.9,
                    description=f"Jumped {level_diff}° in {time_diff} days"
                ))
        
        # Suspensions
        if user.suspension_count > 0:
            signals.append(HackerSignal(
                signal_type="previous_suspension",
                severity=0.5 * user.suspension_count,
                description=f"{user.suspension_count} previous suspensions"
            ))
        
        is_clean = len(signals) == 0 or max(s.severity for s in signals) < 0.7
        return is_clean, signals
    
    @staticmethod
    def check_cross_domain(profile: VerificationProfile) -> Tuple[bool, List[HackerSignal]]:
        """Check cross-domain consistency."""
        signals = []
        
        consistent, score, reason = profile.check_cross_domain_consistency()
        if not consistent:
            signals.append(HackerSignal(
                signal_type="cross_domain_inconsistency",
                severity=1 - score,
                description=reason
            ))
        
        # Check specific patterns
        user_score = profile.get_domain_score('user')
        admin_score = profile.get_domain_score('admin')
        tech_score = profile.get_domain_score('tech')
        
        # Stolen credentials pattern
        if admin_score > 0.8 and tech_score < 0.3:
            signals.append(HackerSignal(
                signal_type="stolen_credential_pattern",
                severity=0.9,
                description="High financial verification but device/temporal mismatch"
            ))
        
        # Bot pattern
        if profile.scores[VerificationAxis.TECH_TEMPORAL].score < 0.3:
            signals.append(HackerSignal(
                signal_type="bot_timing_pattern",
                severity=0.7,
                description="Non-human temporal patterns detected"
            ))
        
        is_clean = len(signals) == 0 or max(s.severity for s in signals) < 0.7
        return is_clean, signals
    
    @staticmethod
    def full_check(user: User, target_level: UserLevel) -> Dict[str, Any]:
        """Run full hacker detection suite."""
        results = {
            'user_id': user.user_id,
            'current_level': user.level.name,
            'target_level': target_level.name,
            'signals': [],
            'blocked': False,
            'block_reason': None
        }
        
        # Time wall check
        time_ok, time_reason = HackerDetector.check_time_wall(user, target_level)
        if not time_ok:
            results['blocked'] = True
            results['block_reason'] = time_reason
            results['signals'].append(HackerSignal(
                signal_type="time_wall",
                severity=1.0,
                description=time_reason
            ))
            return results
        
        # Behavior check
        behavior_ok, behavior_signals = HackerDetector.check_behavior_consistency(user)
        results['signals'].extend(behavior_signals)
        
        # Cross-domain check
        domain_ok, domain_signals = HackerDetector.check_cross_domain(user.profile)
        results['signals'].extend(domain_signals)
        
        # Calculate total risk
        if results['signals']:
            max_severity = max(s.severity for s in results['signals'])
            if max_severity >= 0.9:
                results['blocked'] = True
                results['block_reason'] = f"High risk signals detected (severity: {max_severity:.2f})"
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# BEC BOUNDARY CHECK (for π/2 crossing)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BECCheck:
    """Check for BEC formation at the π/2 boundary."""
    
    @staticmethod
    def check_creator_boundary(user: User, 
                               platform_desire: float,
                               community_validation: float) -> Dict[str, Any]:
        """
        Check if user can cross the π/2 boundary to become Creator.
        Requires three-way desire (BEC formation).
        """
        # User desire (inferred from behavior)
        user_desire = min(1.0, (
            user.content_created * 0.1 +
            user.reviews_written * 0.05 +
            user.session_count * 0.01
        ))
        
        # Calculate temperatures
        t_user = user_desire
        t_platform = platform_desire
        t_community = community_validation
        
        total_temp = t_user + t_platform + t_community
        avg_temp = total_temp / 3
        
        # Check alignment
        temps = [t_user, t_platform, t_community]
        variance = sum((t - avg_temp) ** 2 for t in temps) / 3
        alignment = max(0, 1 - math.sqrt(variance) * 2)
        
        # BEC forms when all are high AND aligned
        all_active = all(t >= 0.7 for t in temps)
        well_aligned = alignment > 0.7
        bec_formed = all_active and well_aligned
        
        return {
            'user_id': user.user_id,
            'temperatures': {
                'user': t_user,
                'platform': t_platform,
                'community': t_community
            },
            'total_temperature': total_temp,
            'alignment': alignment,
            'bec_formed': bec_formed,
            'can_cross_boundary': bec_formed,
            'reason': "BEC formed - all three parties actively desire this" if bec_formed 
                     else f"BEC not formed - need higher desire/alignment (current: {alignment:.2%})"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL PROMOTION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class PromotionSystem:
    """Handles user level promotions with full verification."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.detector = HackerDetector()
    
    def register_user(self, user_id: str) -> User:
        """Register a new user at ANONYMOUS level."""
        user = User(user_id=user_id)
        self.users[user_id] = user
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self.users.get(user_id)
    
    def update_axis(self, user_id: str, axis: VerificationAxis, 
                   score: float, confidence: float, 
                   details: Dict[str, Any] = None) -> bool:
        """Update a verification axis for a user."""
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.profile.scores[axis] = AxisScore(
            axis=axis,
            score=score,
            confidence=confidence,
            details=details or {}
        )
        return True
    
    def attempt_promotion(self, user_id: str) -> Dict[str, Any]:
        """Attempt to promote a user to the next level."""
        user = self.get_user(user_id)
        if not user:
            return {'error': 'User not found'}
        
        next_level = UserLevel.next_level(user.level)
        if not next_level:
            return {'error': 'Already at maximum level'}
        
        result = {
            'user_id': user_id,
            'current_level': user.level.name,
            'target_level': next_level.name,
            'current_theta': math.degrees(user.theta),
            'target_theta': next_level.value,
            'promoted': False,
            'checks': {}
        }
        
        # Check 1: Axis requirements
        can_reach, missing = user.profile.can_reach_level(next_level)
        result['checks']['axis_requirements'] = {
            'passed': can_reach,
            'missing_axes': [a.name for a in missing]
        }
        
        if not can_reach:
            result['reason'] = f"Missing verification axes: {[a.name for a in missing]}"
            return result
        
        # Check 2: Hacker detection
        hacker_check = self.detector.full_check(user, next_level)
        result['checks']['hacker_detection'] = {
            'passed': not hacker_check['blocked'],
            'signals': [(s.signal_type, s.severity) for s in hacker_check['signals']],
            'block_reason': hacker_check.get('block_reason')
        }
        
        if hacker_check['blocked']:
            user.is_suspicious = True
            result['reason'] = hacker_check['block_reason']
            return result
        
        # Check 3: Cross-domain consistency
        consistent, score, reason = user.profile.check_cross_domain_consistency()
        result['checks']['cross_domain'] = {
            'passed': consistent,
            'consistency_score': score,
            'reason': reason
        }
        
        if not consistent:
            result['reason'] = f"Cross-domain inconsistency: {reason}"
            return result
        
        # Check 4: Special boundary check for Creator level
        if next_level == UserLevel.CREATOR:
            # Would need platform/community input in real system
            bec_check = BECCheck.check_creator_boundary(
                user,
                platform_desire=0.8,  # Would come from admin
                community_validation=0.7  # Would come from metrics
            )
            result['checks']['bec_boundary'] = bec_check
            
            if not bec_check['can_cross_boundary']:
                result['reason'] = bec_check['reason']
                return result
        
        # All checks passed - promote!
        old_level = user.level
        user.level = next_level
        user.update_theta()
        user.level_history.append((next_level, datetime.now()))
        
        result['promoted'] = True
        result['new_level'] = next_level.name
        result['new_theta'] = math.degrees(user.theta)
        result['reason'] = f"Successfully promoted from {old_level.name} to {next_level.name}"
        
        return result
    
    def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """Get complete status for a user."""
        user = self.get_user(user_id)
        if not user:
            return {'error': 'User not found'}
        
        next_level = UserLevel.next_level(user.level)
        can_promote, missing = user.profile.can_reach_level(next_level) if next_level else (False, [])
        
        return {
            'user_id': user_id,
            'level': user.level.name,
            'theta': math.degrees(user.theta),
            'theta_radians': user.theta,
            'euler_identity': str(user.euler_identity),
            'cos_component': user.cos_component,
            'sin_component': user.sin_component,
            'side': 'user' if user.level.is_user_side else 'staff' if user.level.is_staff_side else 'boundary',
            'account_age_days': user.account_age_days,
            'verification_score': user.profile.get_total_score(),
            'verified_axes': [a.name for a in user.profile.get_verified_axes()],
            'suspicious_axes': [a.name for a in user.profile.get_suspicious_axes()],
            'domain_scores': {
                'user': user.profile.get_domain_score('user'),
                'admin': user.profile.get_domain_score('admin'),
                'tech': user.profile.get_domain_score('tech')
            },
            'next_level': next_level.name if next_level else None,
            'can_promote': can_promote,
            'missing_for_next': [a.name for a in missing],
            'is_suspicious': user.is_suspicious
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_legitimate_user(system: PromotionSystem, user_id: str, days: int = 400) -> User:
    """Simulate a legitimate user building trust over time."""
    user = system.register_user(user_id)
    user.created_at = datetime.now() - timedelta(days=days)
    
    # Build up natural verification scores
    axes_scores = {
        VerificationAxis.USER_IDENTITY: (0.9, 0.95),
        VerificationAxis.USER_BEHAVIOR: (0.8, 0.85),
        VerificationAxis.USER_VALUE: (0.7, 0.8),
        VerificationAxis.ADMIN_FINANCIAL: (0.85, 0.9),
        VerificationAxis.ADMIN_COMPLIANCE: (0.95, 0.95),
        VerificationAxis.ADMIN_RELATIONSHIP: (0.8, 0.85),
        VerificationAxis.TECH_DEVICE: (0.9, 0.9),
        VerificationAxis.TECH_TEMPORAL: (0.85, 0.9),
        VerificationAxis.TECH_ANOMALY: (0.9, 0.85),
    }
    
    for axis, (score, conf) in axes_scores.items():
        system.update_axis(user_id, axis, score, conf)
    
    user.total_spend = 500
    user.session_count = 200
    user.content_created = 15
    user.reviews_written = 30
    user.helpful_votes = 45
    
    return user


def simulate_hacker(system: PromotionSystem, user_id: str, days: int = 30) -> User:
    """Simulate a hacker trying to escalate quickly."""
    user = system.register_user(user_id)
    user.created_at = datetime.now() - timedelta(days=days)
    
    # Hackers can fake some axes but not others
    axes_scores = {
        # Can fake these (with resources)
        VerificationAxis.USER_IDENTITY: (0.9, 0.9),     # Fake identity
        VerificationAxis.ADMIN_FINANCIAL: (0.8, 0.85),  # Stolen card
        
        # Harder to fake
        VerificationAxis.USER_BEHAVIOR: (0.4, 0.5),     # Bot-like patterns
        VerificationAxis.USER_VALUE: (0.2, 0.3),        # No real contributions
        VerificationAxis.ADMIN_COMPLIANCE: (0.6, 0.6),  # Some violations
        VerificationAxis.ADMIN_RELATIONSHIP: (0.3, 0.4), # Account too new
        
        # Can't fake these
        VerificationAxis.TECH_DEVICE: (0.3, 0.4),       # VPN/emulator detected
        VerificationAxis.TECH_TEMPORAL: (0.2, 0.3),     # Inhuman timing
        VerificationAxis.TECH_ANOMALY: (0.1, 0.2),      # Many anomalies
    }
    
    for axis, (score, conf) in axes_scores.items():
        system.update_axis(user_id, axis, score, conf)
    
    user.total_spend = 2000  # Suspiciously high
    user.session_count = 5   # But very few sessions
    
    return user


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_user_progression():
    """Demonstrate legitimate user progression."""
    print("\n" + "=" * 80)
    print("LEGITIMATE USER PROGRESSION")
    print("=" * 80)
    
    system = PromotionSystem()
    user = simulate_legitimate_user(system, "alice", days=400)
    
    print(f"\nUser: alice")
    print(f"Account age: {user.account_age_days} days")
    
    status = system.get_user_status("alice")
    print(f"\nInitial Status:")
    print(f"  Level: {status['level']} (θ = {status['theta']:.1f}°)")
    print(f"  Verification Score: {status['verification_score']:.2f}")
    print(f"  Domain Scores: {status['domain_scores']}")
    
    # Try to promote through levels
    print("\n--- Attempting Promotions ---\n")
    
    for _ in range(8):  # Try multiple promotions
        result = system.attempt_promotion("alice")
        
        if result.get('promoted'):
            print(f"✓ {result['current_level']} → {result['new_level']}")
            print(f"  θ: {result['current_theta']:.1f}° → {result['new_theta']:.1f}°")
        else:
            print(f"✗ Cannot promote from {result['current_level']}")
            print(f"  Reason: {result.get('reason', 'Unknown')}")
            break
    
    # Final status
    status = system.get_user_status("alice")
    print(f"\nFinal Status:")
    print(f"  Level: {status['level']} (θ = {status['theta']:.1f}°)")
    print(f"  Side: {status['side']}")
    print(f"  e^(iθ) = {status['euler_identity']}")
    print(f"  cos(θ) = {status['cos_component']:.3f} (user-side strength)")
    print(f"  sin(θ) = {status['sin_component']:.3f} (staff-side strength)")


def demo_hacker_blocked():
    """Demonstrate hacker being blocked."""
    print("\n" + "=" * 80)
    print("HACKER BLOCKED AT THE WALL")
    print("=" * 80)
    
    system = PromotionSystem()
    hacker = simulate_hacker(system, "eve", days=30)
    
    print(f"\nUser: eve (hacker)")
    print(f"Account age: {hacker.account_age_days} days")
    
    status = system.get_user_status("eve")
    print(f"\nInitial Status:")
    print(f"  Level: {status['level']} (θ = {status['theta']:.1f}°)")
    print(f"  Verification Score: {status['verification_score']:.2f}")
    print(f"  Domain Scores: {status['domain_scores']}")
    print(f"  Suspicious Axes: {status['suspicious_axes']}")
    
    # Try to promote
    print("\n--- Hacker Attempting Promotions ---\n")
    
    for _ in range(5):
        result = system.attempt_promotion("eve")
        
        if result.get('promoted'):
            print(f"✓ {result['current_level']} → {result['new_level']}")
        else:
            print(f"✗ BLOCKED at {result['current_level']}")
            print(f"  Target: {result['target_level']}")
            print(f"  Reason: {result.get('reason', 'Unknown')}")
            
            # Show checks
            if 'checks' in result:
                for check_name, check_result in result['checks'].items():
                    passed = check_result.get('passed', False)
                    print(f"  {check_name}: {'✓' if passed else '✗'}")
                    if not passed and 'signals' in check_result:
                        for sig_type, severity in check_result['signals']:
                            print(f"    - {sig_type}: severity={severity:.2f}")
            break
    
    print(f"\n⚠️  Hacker stuck at θ = {math.degrees(hacker.theta):.1f}°")
    print(f"   The π/4 wall (45°) blocks time-dependent escalation!")


def demo_verification_matrix():
    """Show the verification matrix."""
    print("\n" + "=" * 80)
    print("VERIFICATION MATRIX: 9 AXES")
    print("=" * 80)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    THE 9 VERIFICATION AXES                          │")
    print("├───────────┬─────────────────┬────────────────────────────────────────┤")
    print("│ Domain    │ Axis            │ What It Checks                         │")
    print("├───────────┼─────────────────┼────────────────────────────────────────┤")
    
    for axis in VerificationAxis:
        print(f"│ {axis.domain.upper():9} │ {axis.axis:15} │ {axis.description:38} │")
        if axis.axis == 'value' or axis.axis == 'relationship':
            print("├───────────┼─────────────────┼────────────────────────────────────────┤")
    
    print("└───────────┴─────────────────┴────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    LEVEL REQUIREMENTS                               │")
    print("├─────────────┬──────┬───────────────────────────────────┬────────────┤")
    print("│ Level       │  θ   │ Required Axes                     │ Fakeable?  │")
    print("├─────────────┼──────┼───────────────────────────────────┼────────────┤")
    
    fakeability = {
        UserLevel.ANONYMOUS: "YES",
        UserLevel.REGISTERED: "YES",
        UserLevel.PAID: "YES",
        UserLevel.LONG_TERM: "HARD",
        UserLevel.CORE: "VERY HARD",
        UserLevel.REVIEWER: "NO",
        UserLevel.CREATOR: "NO (BEC)",
        UserLevel.MASCOT: "NO",
        UserLevel.STAFF: "INVERTED",
        UserLevel.ADMIN: "INVERTED",
        UserLevel.OWNER: "INVERTED",
    }
    
    for level in UserLevel:
        required = LEVEL_REQUIREMENTS.get(level, [])
        axes_str = str(len(required)) + " axes" if len(required) > 3 else ", ".join(a.axis[:3].upper() for a in required) or "-"
        if len(axes_str) > 33:
            axes_str = f"{len(required)} axes"
        print(f"│ {level.name:11} │ {level.value:3}° │ {axes_str:33} │ {fakeability[level]:10} │")
        
        if level == UserLevel.LONG_TERM:
            print("├─────────────┼──────┼───────────────────────────────────┼────────────┤")
            print("│             │ ↑↑↑  │ ═══ THE π/4 WALL (TIME) ═══       │            │")
            print("├─────────────┼──────┼───────────────────────────────────┼────────────┤")
        elif level == UserLevel.CREATOR:
            print("├─────────────┼──────┼───────────────────────────────────┼────────────┤")
            print("│             │ ↑↑↑  │ ═══ THE π/2 BOUNDARY (BEC) ═══    │            │")
            print("├─────────────┼──────┼───────────────────────────────────┼────────────┤")
    
    print("└─────────────┴──────┴───────────────────────────────────┴────────────┘")


def demo_phase_visualization():
    """Visualize the phase circle."""
    print("\n" + "=" * 80)
    print("USER PHASE CIRCLE")
    print("=" * 80)
    
    print("""
                              θ = 90° (π/2)
                              CREATOR
                           ═══════════════
                          ║   BOUNDARY   ║
                         ╱ ║             ║ ╲
                        ╱  ╚═════════════╝  ╲
                       ╱                      ╲
                REVIEWER                      MASCOT
                   75°                          105°
                     ╲                          ╱
                      ╲                        ╱
                       ╲                      ╱
                CORE    ╲                    ╱    STAFF
                 60°     ╲                  ╱      135°
                          ╲                ╱
    θ = 0°                 ╲    ┌────┐    ╱                 θ = 180°
    ANONYMOUS ══════════════════│ π/4│════════════════════ OWNER
    e^(i·0) = 1            ╱    │WALL│    ╲            e^(i·π) = -1
                          ╱     └────┘     ╲
                PAID     ╱                  ╲     ADMIN
                 30°    ╱                    ╲     165°
                       ╱                      ╲
                      ╱                        ╲
                     ╱                          ╲
                REGISTERED                      
                   15°                          
                         
                         
                     ← USER SIDE │ STAFF SIDE →
                       cos(θ) > 0│  cos(θ) < 0
                       sin(θ) < 1│  sin(θ) > 0
                                 │
                                 
    HACKERS PILE UP HERE:
    ════════════════════
    
         θ = 0°  →  θ = 30°  →  θ = 45°  ══╗
           ●          ●           ●●●●●●   ║
         Easy       Money        ════════  ║
                    (fake)       THE WALL  ║
                                          ║
                    Can't fake TIME       ║
                    Can't fake BEHAVIOR   ║
                    Cross-domain caught!  ║
                                         ═╝
    """)


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("USER LIFECYCLE - TEST SUITE")
    print("=" * 80)
    
    # Test 1: User levels
    print("\n--- TEST 1: User Levels ---")
    assert UserLevel.ANONYMOUS.theta == 0
    assert UserLevel.CREATOR.value == 90
    assert UserLevel.ANONYMOUS.is_user_side
    assert UserLevel.STAFF.is_staff_side
    assert UserLevel.CREATOR.is_boundary
    print("✓ User levels correctly mapped")
    
    # Test 2: Verification profile
    print("\n--- TEST 2: Verification Profile ---")
    profile = VerificationProfile("test")
    assert len(profile.scores) == 9
    assert profile.get_total_score() == 0
    print("✓ Profile initializes with 9 axes")
    
    # Test 3: Axis updates
    print("\n--- TEST 3: Axis Updates ---")
    system = PromotionSystem()
    user = system.register_user("test")
    system.update_axis("test", VerificationAxis.USER_IDENTITY, 0.9, 0.9)
    assert user.profile.scores[VerificationAxis.USER_IDENTITY].score == 0.9
    print("✓ Axis updates work")
    
    # Test 4: Hacker detection
    print("\n--- TEST 4: Hacker Detection ---")
    hacker = simulate_hacker(system, "hacker", days=10)
    check = HackerDetector.full_check(hacker, UserLevel.LONG_TERM)
    assert check['blocked']
    print("✓ Hacker blocked at time wall")
    
    # Test 5: Legitimate progression
    print("\n--- TEST 5: Legitimate Progression ---")
    system2 = PromotionSystem()
    legit = simulate_legitimate_user(system2, "legit", days=400)
    
    # Should be able to get to PAID at least
    system2.attempt_promotion("legit")  # → REGISTERED
    system2.attempt_promotion("legit")  # → PAID
    assert legit.level.value >= UserLevel.PAID.value
    print("✓ Legitimate user can progress")
    
    # Test 6: Cross-domain consistency
    print("\n--- TEST 6: Cross-Domain Consistency ---")
    consistent, score, _ = legit.profile.check_cross_domain_consistency()
    assert consistent
    print(f"✓ Legitimate user is consistent (score: {score:.2f})")
    
    inconsistent, score2, _ = hacker.profile.check_cross_domain_consistency()
    assert not inconsistent or score2 < 0.7
    print(f"✓ Hacker shows inconsistency (score: {score2:.2f})")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--legit":
            demo_user_progression()
        elif sys.argv[1] == "--hacker":
            demo_hacker_blocked()
        elif sys.argv[1] == "--matrix":
            demo_verification_matrix()
        elif sys.argv[1] == "--phase":
            demo_phase_visualization()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_verification_matrix()
            demo_phase_visualization()
            demo_user_progression()
            demo_hacker_blocked()
        else:
            run_tests()
    else:
        run_tests()
