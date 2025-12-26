"""
MONEY HEXAGON SYSTEM
====================
The antimatter counterpart to the User Hexagon.

Key Insights:
1. Money has DNA (provenance - where did it come from?)
2. Money has RNA (spending authorization tokens)
3. Money has TEETH ANGLE (legitimacy - how "real" is it?)
4. Domains see the ANGLE, not the details (privacy + fraud prevention)
5. Fake money = discontinuity in provenance chain

The Dual Hexagons:
- TOP: User/Owner Hexagon (capability, verification)
- BOTTOM: Money/Debt Hexagon (legitimacy, provenance)

They MESH at Y=0. Angles must be compatible!

Author: Jonathan Pelchat
Based on Shovelcat Theory + collaborative refinement
"""

import math
import time
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# MONEY SOURCE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class MoneySourceType(Enum):
    """Types of money sources with inherent legitimacy."""
    
    # High legitimacy (teeth angle 75-90°)
    SALARY = ("SALARY", 90, "Direct employment income")
    GOVERNMENT_PAYMENT = ("GOVERNMENT_PAYMENT", 90, "Government benefits/refunds")
    BANK_TRANSFER = ("BANK_TRANSFER", 85, "Direct bank transfer")
    
    # Good legitimacy (teeth angle 60-74°)
    BUSINESS_REVENUE = ("BUSINESS_REVENUE", 75, "Business income")
    INVESTMENT_DIVIDEND = ("INVESTMENT_DIVIDEND", 70, "Investment dividends")
    STOCK_SALE = ("STOCK_SALE", 65, "Stock market sale")
    REAL_ESTATE = ("REAL_ESTATE", 70, "Real estate transaction")
    
    # Moderate legitimacy (teeth angle 45-59°)
    GIFT = ("GIFT", 55, "Gift from known party")
    INHERITANCE = ("INHERITANCE", 60, "Inheritance")
    FREELANCE = ("FREELANCE", 55, "Freelance/gig work")
    CRYPTO_SALE = ("CRYPTO_SALE", 50, "Cryptocurrency sale")
    
    # Questionable legitimacy (teeth angle 20-44°)
    COMPLEX_DERIVATIVE = ("COMPLEX_DERIVATIVE", 40, "Complex financial instrument")
    OFFSHORE_TRANSFER = ("OFFSHORE_TRANSFER", 30, "Offshore account transfer")
    SHELL_COMPANY = ("SHELL_COMPANY", 25, "Shell company profit")
    CASH_DEPOSIT = ("CASH_DEPOSIT", 35, "Large cash deposit")
    
    # Suspicious (teeth angle 0-19°)
    UNKNOWN = ("UNKNOWN", 10, "Unknown source")
    CIRCULAR = ("CIRCULAR", 5, "Circular transaction")
    NO_RECORD = ("NO_RECORD", 0, "No documentation")
    
    def __init__(self, name: str, base_angle: int, description: str):
        self._name = name
        self.base_angle = base_angle  # Base legitimacy angle
        self.description = description
    
    @property
    def is_high_legitimacy(self) -> bool:
        return self.base_angle >= 75
    
    @property
    def is_suspicious(self) -> bool:
        return self.base_angle < 20


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY RNA TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

class MoneyRNAType(Enum):
    """RNA tokens for money - what money CAN be used for."""
    
    # Basic spending
    SPEND_SMALL = (1, 0, "Small purchases")
    SPEND_MEDIUM = (2, 30, "Medium purchases")
    SPEND_LARGE = (3, 50, "Large purchases")
    
    # Transfers
    TRANSFER_INTERNAL = (1, 20, "Internal transfer")
    TRANSFER_EXTERNAL = (2, 40, "External transfer")
    TRANSFER_INTERNATIONAL = (3, 60, "International transfer")
    
    # Investment
    INVEST_LOW_RISK = (2, 30, "Low-risk investment")
    INVEST_MEDIUM_RISK = (3, 45, "Medium-risk investment")
    INVEST_HIGH_RISK = (4, 60, "High-risk investment")
    
    # Withdrawal
    WITHDRAW_SMALL = (1, 20, "Small withdrawal")
    WITHDRAW_LARGE = (3, 50, "Large withdrawal")
    
    # Special
    COLLATERAL = (4, 70, "Use as collateral")
    PROOF_OF_FUNDS = (3, 60, "Proof of funds")
    
    def __init__(self, cost: int, min_angle: int, description: str):
        self.cost = cost
        self.min_angle = min_angle  # Minimum legitimacy angle required
        self.description = description


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY DNA (Provenance)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProvenanceCheckpoint:
    """A single checkpoint in money's provenance chain."""
    
    timestamp: float
    source_type: MoneySourceType
    amount: float
    from_entity: str
    to_entity: str
    verification_status: str  # "verified", "pending", "failed"
    documentation: bool  # Has documentation?
    
    def to_hash(self) -> str:
        """Create hash of this checkpoint."""
        data = f"{self.timestamp}:{self.source_type.name}:{self.amount}:{self.from_entity}:{self.to_entity}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class MoneyDNA:
    """
    DNA for money - complete provenance chain.
    
    Like genetic history, shows where money came from.
    Each checkpoint is verified and linked.
    """
    
    money_id: str
    amount: float
    created_at: float
    
    # Provenance chain
    checkpoints: List[ProvenanceCheckpoint] = field(default_factory=list)
    
    # Computed legitimacy
    _cached_angle: Optional[float] = None
    
    @property
    def origin(self) -> Optional[ProvenanceCheckpoint]:
        """Get origin checkpoint."""
        return self.checkpoints[0] if self.checkpoints else None
    
    @property
    def current(self) -> Optional[ProvenanceCheckpoint]:
        """Get most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
    
    @property
    def chain_length(self) -> int:
        """Length of provenance chain."""
        return len(self.checkpoints)
    
    @property
    def chain_hash(self) -> str:
        """Hash of entire chain."""
        if not self.checkpoints:
            return "NO_CHAIN"
        chain_data = ":".join(cp.to_hash() for cp in self.checkpoints)
        return hashlib.sha256(chain_data.encode()).hexdigest()[:32]
    
    def add_checkpoint(self, checkpoint: ProvenanceCheckpoint):
        """Add a checkpoint to the chain."""
        self.checkpoints.append(checkpoint)
        self._cached_angle = None  # Invalidate cache
    
    def compute_legitimacy_angle(self) -> float:
        """
        Compute overall legitimacy angle from provenance.
        
        Factors:
        - Source type base angle
        - Verification status of each checkpoint
        - Documentation completeness
        - Chain continuity
        """
        if self._cached_angle is not None:
            return self._cached_angle
        
        if not self.checkpoints:
            self._cached_angle = 0.0
            return 0.0
        
        # Start with origin's base angle
        base = self.origin.source_type.base_angle
        
        # Adjust for verification
        verified_count = sum(1 for cp in self.checkpoints if cp.verification_status == "verified")
        verification_ratio = verified_count / len(self.checkpoints)
        
        # Adjust for documentation
        documented_count = sum(1 for cp in self.checkpoints if cp.documentation)
        documentation_ratio = documented_count / len(self.checkpoints)
        
        # Check for discontinuities
        discontinuity_penalty = 0
        for i in range(1, len(self.checkpoints)):
            prev = self.checkpoints[i-1]
            curr = self.checkpoints[i]
            # Gap in entities?
            if prev.to_entity != curr.from_entity:
                discontinuity_penalty += 10
            # Time gap too large?
            if curr.timestamp - prev.timestamp > 365 * 24 * 3600:  # 1 year
                discontinuity_penalty += 5
        
        # Compute final angle
        angle = base * verification_ratio * (0.5 + 0.5 * documentation_ratio)
        angle = max(0, angle - discontinuity_penalty)
        
        self._cached_angle = min(90, angle)
        return self._cached_angle
    
    def has_discontinuity(self) -> bool:
        """Check if chain has discontinuities (fake money indicator)."""
        if len(self.checkpoints) < 2:
            return self.chain_length == 0  # No checkpoints = discontinuity
        
        for i in range(1, len(self.checkpoints)):
            prev = self.checkpoints[i-1]
            curr = self.checkpoints[i]
            if prev.to_entity != curr.from_entity:
                return True
        
        return False
    
    def to_summary(self) -> Dict[str, Any]:
        """Get summary without exposing details."""
        return {
            'money_id': self.money_id,
            'amount': self.amount,
            'legitimacy_angle': self.compute_legitimacy_angle(),
            'chain_length': self.chain_length,
            'has_discontinuity': self.has_discontinuity(),
            'chain_hash': self.chain_hash
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY RNA (Spending Authorization)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoneyRNA:
    """
    RNA tokens for money - spending authorizations.
    
    Based on legitimacy angle, different actions are allowed.
    """
    
    money_id: str
    legitimacy_angle: float
    
    # Token pools
    tokens: Dict[MoneyRNAType, int] = field(default_factory=dict)
    
    def mint_tokens(self):
        """Mint tokens based on legitimacy angle."""
        self.tokens = {}
        
        for rna_type in MoneyRNAType:
            if self.legitimacy_angle >= rna_type.min_angle:
                # More tokens for higher legitimacy
                multiplier = (self.legitimacy_angle - rna_type.min_angle) / 30 + 1
                self.tokens[rna_type] = int(10 * multiplier)
            else:
                self.tokens[rna_type] = 0
    
    def can_use(self, rna_type: MoneyRNAType) -> Tuple[bool, str]:
        """Check if this money can be used for an action."""
        if self.legitimacy_angle < rna_type.min_angle:
            return False, f"Legitimacy {self.legitimacy_angle:.1f}° < required {rna_type.min_angle}°"
        
        if self.tokens.get(rna_type, 0) <= 0:
            return False, f"No {rna_type.name} tokens available"
        
        return True, "Authorized"
    
    def use_token(self, rna_type: MoneyRNAType) -> bool:
        """Use a token (consume it)."""
        if self.tokens.get(rna_type, 0) > 0:
            self.tokens[rna_type] -= 1
            return True
        return False
    
    def get_available_actions(self) -> List[MoneyRNAType]:
        """Get list of available actions."""
        return [rt for rt in MoneyRNAType if self.tokens.get(rt, 0) > 0]


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY TEETH (Legitimacy Angle)
# ═══════════════════════════════════════════════════════════════════════════════

class LegitimacyBand(Enum):
    """Bands of legitimacy based on teeth angle."""
    
    PRISTINE = (75, 90, "Pristine - fully documented, verified")
    CLEAN = (60, 74, "Clean - good documentation")
    ACCEPTABLE = (45, 59, "Acceptable - adequate provenance")
    QUESTIONABLE = (20, 44, "Questionable - limited documentation")
    SUSPICIOUS = (1, 19, "Suspicious - needs investigation")
    REJECTED = (0, 0, "Rejected - no provenance")
    
    def __init__(self, min_angle: int, max_angle: int, description: str):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.description = description
    
    @classmethod
    def from_angle(cls, angle: float) -> 'LegitimacyBand':
        """Get band from angle."""
        for band in cls:
            if band.min_angle <= angle <= band.max_angle:
                return band
        return cls.REJECTED


@dataclass
class MoneyTeeth:
    """
    Teeth for money - the legitimacy angle.
    
    This is what domains SEE without seeing the details.
    Preserves privacy while preventing fraud.
    """
    
    angle: float  # 0-90 degrees
    band: LegitimacyBand
    
    # Flags
    has_discontinuity: bool = False
    has_suspicious_pattern: bool = False
    flagged_for_review: bool = False
    
    @classmethod
    def from_dna(cls, dna: MoneyDNA) -> 'MoneyTeeth':
        """Create teeth from DNA."""
        angle = dna.compute_legitimacy_angle()
        band = LegitimacyBand.from_angle(angle)
        
        return cls(
            angle=angle,
            band=band,
            has_discontinuity=dna.has_discontinuity(),
            has_suspicious_pattern=angle < 20,
            flagged_for_review=dna.has_discontinuity() or angle < 30
        )
    
    def compatible_with(self, required_angle: float) -> bool:
        """Check if teeth angle is compatible with requirement."""
        return self.angle >= required_angle
    
    def mesh_friction(self, other_angle: float) -> float:
        """Calculate friction when meshing with another angle."""
        diff = abs(self.angle - other_angle)
        return diff / 90  # 0 = perfect mesh, 1 = maximum friction


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY UNIT (Complete)
# ═══════════════════════════════════════════════════════════════════════════════

class MoneyUnit:
    """
    A complete money unit with DNA, RNA, and teeth.
    
    The antimatter counterpart to a User.
    """
    
    def __init__(self, 
                 amount: float,
                 source_type: MoneySourceType,
                 from_entity: str,
                 to_entity: str):
        
        self.money_id = secrets.token_hex(8)
        self.amount = amount
        
        # Create DNA with initial checkpoint
        self.dna = MoneyDNA(
            money_id=self.money_id,
            amount=amount,
            created_at=time.time()
        )
        
        initial_checkpoint = ProvenanceCheckpoint(
            timestamp=time.time(),
            source_type=source_type,
            amount=amount,
            from_entity=from_entity,
            to_entity=to_entity,
            verification_status="verified",
            documentation=True
        )
        self.dna.add_checkpoint(initial_checkpoint)
        
        # Create teeth
        self.teeth = MoneyTeeth.from_dna(self.dna)
        
        # Create RNA
        self.rna = MoneyRNA(
            money_id=self.money_id,
            legitimacy_angle=self.teeth.angle
        )
        self.rna.mint_tokens()
    
    def transfer(self, 
                to_entity: str,
                verification_status: str = "verified",
                documentation: bool = True):
        """Transfer money to another entity."""
        
        current = self.dna.current
        
        checkpoint = ProvenanceCheckpoint(
            timestamp=time.time(),
            source_type=current.source_type,  # Inherits source type
            amount=self.amount,
            from_entity=current.to_entity,
            to_entity=to_entity,
            verification_status=verification_status,
            documentation=documentation
        )
        
        self.dna.add_checkpoint(checkpoint)
        
        # Update teeth and RNA
        self.teeth = MoneyTeeth.from_dna(self.dna)
        self.rna.legitimacy_angle = self.teeth.angle
        self.rna.mint_tokens()
    
    def can_spend_on(self, action: MoneyRNAType) -> Tuple[bool, str]:
        """Check if money can be spent on action."""
        return self.rna.can_use(action)
    
    def spend(self, action: MoneyRNAType) -> bool:
        """Spend on an action (consumes RNA token)."""
        can, _ = self.can_spend_on(action)
        if can:
            return self.rna.use_token(action)
        return False
    
    def get_public_info(self) -> Dict[str, Any]:
        """Get public info (what domains can see)."""
        return {
            'money_id': self.money_id,
            'amount': self.amount,
            'legitimacy_angle': self.teeth.angle,
            'legitimacy_band': self.teeth.band.name,
            'flagged': self.teeth.flagged_for_review,
            'available_actions': [a.name for a in self.rna.get_available_actions()]
        }
    
    def get_private_info(self) -> Dict[str, Any]:
        """Get private info (full provenance - only for authorized)."""
        return {
            **self.get_public_info(),
            'dna': {
                'chain_length': self.dna.chain_length,
                'chain_hash': self.dna.chain_hash,
                'origin_type': self.dna.origin.source_type.name if self.dna.origin else None,
                'has_discontinuity': self.dna.has_discontinuity(),
                'checkpoints': len(self.dna.checkpoints)
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FAKE MONEY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class FakeMoneyDetector:
    """
    Detects fake or suspicious money.
    
    Checks for:
    - Discontinuities in provenance
    - Mismatched claimed vs actual source
    - Circular transactions
    - Missing documentation
    """
    
    @staticmethod
    def check(money: MoneyUnit) -> Tuple[bool, List[str]]:
        """
        Check money for fraud indicators.
        
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for discontinuity
        if money.dna.has_discontinuity():
            issues.append("DISCONTINUITY: Gap in provenance chain")
        
        # Check legitimacy
        if money.teeth.angle < 20:
            issues.append(f"LOW_LEGITIMACY: Angle {money.teeth.angle:.1f}° is suspicious")
        
        # Check for no origin
        if money.dna.chain_length == 0:
            issues.append("NO_ORIGIN: Money has no provenance")
        
        # Check for missing verification
        unverified = [cp for cp in money.dna.checkpoints 
                     if cp.verification_status != "verified"]
        if len(unverified) > len(money.dna.checkpoints) * 0.3:
            issues.append(f"UNVERIFIED: {len(unverified)} of {len(money.dna.checkpoints)} checkpoints unverified")
        
        # Check for missing documentation
        undocumented = [cp for cp in money.dna.checkpoints if not cp.documentation]
        if len(undocumented) > len(money.dna.checkpoints) * 0.3:
            issues.append(f"UNDOCUMENTED: {len(undocumented)} checkpoints lack documentation")
        
        # Check for suspicious source type
        if money.dna.origin and money.dna.origin.source_type.is_suspicious:
            issues.append(f"SUSPICIOUS_SOURCE: Origin is {money.dna.origin.source_type.name}")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    @staticmethod
    def detect_mismatch(claimed_source: MoneySourceType, 
                       actual_dna: MoneyDNA) -> Tuple[bool, str]:
        """
        Detect if claimed source doesn't match actual DNA.
        
        Like "hand in cookie jar" for money.
        """
        if not actual_dna.origin:
            return True, "NO_ORIGIN"
        
        actual_source = actual_dna.origin.source_type
        
        if claimed_source != actual_source:
            claimed_angle = claimed_source.base_angle
            actual_angle = actual_source.base_angle
            
            if claimed_angle > actual_angle + 20:
                return True, f"MISMATCH: Claimed {claimed_source.name} ({claimed_angle}°) but actual {actual_source.name} ({actual_angle}°)"
        
        return False, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN MONEY POLICY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DomainMoneyPolicy:
    """
    A domain's policy for accepting money.
    
    Domains don't see details, but they set angle requirements.
    """
    
    domain_id: str
    
    # Minimum angles for different actions
    min_angle_small_transaction: float = 30.0
    min_angle_large_transaction: float = 60.0
    min_angle_investment: float = 70.0
    min_angle_collateral: float = 80.0
    
    # Whether to accept flagged money
    accept_flagged: bool = False
    
    # Whether to require full provenance
    require_full_provenance: bool = False
    
    def can_accept(self, money: MoneyUnit, action: MoneyRNAType) -> Tuple[bool, str]:
        """Check if domain can accept money for action."""
        
        # Check if flagged
        if money.teeth.flagged_for_review and not self.accept_flagged:
            return False, "FLAGGED_REJECTED: Domain doesn't accept flagged money"
        
        # Get required angle based on action
        if action in [MoneyRNAType.SPEND_SMALL, MoneyRNAType.TRANSFER_INTERNAL]:
            required = self.min_angle_small_transaction
        elif action in [MoneyRNAType.SPEND_LARGE, MoneyRNAType.TRANSFER_EXTERNAL, 
                       MoneyRNAType.TRANSFER_INTERNATIONAL]:
            required = self.min_angle_large_transaction
        elif action in [MoneyRNAType.INVEST_LOW_RISK, MoneyRNAType.INVEST_MEDIUM_RISK,
                       MoneyRNAType.INVEST_HIGH_RISK]:
            required = self.min_angle_investment
        elif action in [MoneyRNAType.COLLATERAL, MoneyRNAType.PROOF_OF_FUNDS]:
            required = self.min_angle_collateral
        else:
            required = self.min_angle_small_transaction
        
        # Check angle
        if money.teeth.angle < required:
            return False, f"ANGLE_INSUFFICIENT: {money.teeth.angle:.1f}° < required {required:.1f}°"
        
        # Check if money can do the action
        can, reason = money.can_spend_on(action)
        if not can:
            return False, reason
        
        return True, "ACCEPTED"


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_money_dna():
    """Demonstrate money DNA (provenance)."""
    print("\n" + "=" * 80)
    print("MONEY DNA: Provenance Chain")
    print("=" * 80)
    
    print("""
    Every dollar has a history (DNA):
    
    [ORIGIN] → [TRANSFER] → [TRANSFER] → [CURRENT]
       ↓           ↓            ↓            ↓
    verified    verified     verified     verified
    
    Legitimacy angle computed from:
    - Source type
    - Verification status
    - Documentation
    - Chain continuity
    """)
    
    # Create money with good provenance
    money = MoneyUnit(
        amount=1000,
        source_type=MoneySourceType.SALARY,
        from_entity="employer_corp",
        to_entity="alice_account"
    )
    
    print(f"\n--- Money with Good Provenance ---")
    print(f"Source: SALARY")
    print(f"Legitimacy angle: {money.teeth.angle:.1f}°")
    print(f"Band: {money.teeth.band.name}")
    
    # Transfer it
    money.transfer("bob_account", "verified", True)
    money.transfer("charlie_account", "verified", True)
    
    print(f"\nAfter 2 verified transfers:")
    print(f"Chain length: {money.dna.chain_length}")
    print(f"Legitimacy angle: {money.teeth.angle:.1f}°")
    
    # Create money with bad provenance
    bad_money = MoneyUnit(
        amount=1000,
        source_type=MoneySourceType.UNKNOWN,
        from_entity="???",
        to_entity="suspicious_account"
    )
    
    print(f"\n--- Money with Bad Provenance ---")
    print(f"Source: UNKNOWN")
    print(f"Legitimacy angle: {bad_money.teeth.angle:.1f}°")
    print(f"Band: {bad_money.teeth.band.name}")
    print(f"Flagged: {bad_money.teeth.flagged_for_review}")


def demo_money_rna():
    """Demonstrate money RNA (spending tokens)."""
    print("\n" + "=" * 80)
    print("MONEY RNA: Spending Authorization")
    print("=" * 80)
    
    print("""
    Money RNA tokens = what money CAN be used for
    
    Higher legitimacy → more actions allowed
    Lower legitimacy → restricted actions
    """)
    
    # High legitimacy money
    good_money = MoneyUnit(
        amount=1000,
        source_type=MoneySourceType.SALARY,
        from_entity="employer",
        to_entity="alice"
    )
    
    print(f"\n--- High Legitimacy Money ({good_money.teeth.angle:.1f}°) ---")
    print("Available actions:")
    for action in good_money.rna.get_available_actions():
        tokens = good_money.rna.tokens[action]
        print(f"  {action.name}: {tokens} tokens (requires {action.min_angle}°)")
    
    # Low legitimacy money
    bad_money = MoneyUnit(
        amount=1000,
        source_type=MoneySourceType.CASH_DEPOSIT,
        from_entity="unknown",
        to_entity="bob"
    )
    
    print(f"\n--- Low Legitimacy Money ({bad_money.teeth.angle:.1f}°) ---")
    print("Available actions:")
    available = bad_money.rna.get_available_actions()
    if available:
        for action in available:
            tokens = bad_money.rna.tokens[action]
            print(f"  {action.name}: {tokens} tokens")
    else:
        print("  (none available)")
    
    # Check specific action
    print(f"\nCan use for COLLATERAL?")
    can, reason = good_money.can_spend_on(MoneyRNAType.COLLATERAL)
    print(f"  Good money: {can} - {reason}")
    can, reason = bad_money.can_spend_on(MoneyRNAType.COLLATERAL)
    print(f"  Bad money: {can} - {reason}")


def demo_legitimacy_bands():
    """Demonstrate legitimacy bands."""
    print("\n" + "=" * 80)
    print("LEGITIMACY BANDS: Teeth Angles")
    print("=" * 80)
    
    print("""
    Domains see the ANGLE, not the details.
    Privacy preserved, fraud prevented.
    """)
    
    print("\n┌────────────────────┬─────────────────┬────────────────────────────────────┐")
    print("│ Band               │ Angle Range     │ Description                        │")
    print("├────────────────────┼─────────────────┼────────────────────────────────────┤")
    
    for band in LegitimacyBand:
        print(f"│ {band.name:18} │ {band.min_angle:3}° - {band.max_angle:3}°     │ {band.description:34} │")
    
    print("└────────────────────┴─────────────────┴────────────────────────────────────┘")
    
    print("\n--- Source Type Base Angles ---\n")
    print("┌────────────────────────┬─────────┬────────────────────────────────────┐")
    print("│ Source Type            │ Angle   │ Description                        │")
    print("├────────────────────────┼─────────┼────────────────────────────────────┤")
    
    for st in MoneySourceType:
        print(f"│ {st.name:22} │ {st.base_angle:5}°  │ {st.description:34} │")
    
    print("└────────────────────────┴─────────┴────────────────────────────────────┘")


def demo_fake_money_detection():
    """Demonstrate fake money detection."""
    print("\n" + "=" * 80)
    print("FAKE MONEY DETECTION")
    print("=" * 80)
    
    print("""
    Detects:
    - Discontinuities in provenance
    - Mismatched claimed vs actual
    - Missing documentation
    - Suspicious patterns
    """)
    
    # Good money
    good_money = MoneyUnit(
        amount=1000,
        source_type=MoneySourceType.SALARY,
        from_entity="employer",
        to_entity="alice"
    )
    
    print(f"\n--- Checking Good Money ---")
    is_valid, issues = FakeMoneyDetector.check(good_money)
    print(f"Valid: {is_valid}")
    if issues:
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("  (no issues)")
    
    # Suspicious money
    bad_money = MoneyUnit(
        amount=1000,
        source_type=MoneySourceType.UNKNOWN,
        from_entity="???",
        to_entity="suspicious"
    )
    
    print(f"\n--- Checking Suspicious Money ---")
    is_valid, issues = FakeMoneyDetector.check(bad_money)
    print(f"Valid: {is_valid}")
    for issue in issues:
        print(f"  ! {issue}")
    
    # Mismatch detection
    print(f"\n--- Mismatch Detection (Hand in Cookie Jar) ---")
    mismatch, reason = FakeMoneyDetector.detect_mismatch(
        claimed_source=MoneySourceType.SALARY,  # Claims to be salary
        actual_dna=bad_money.dna  # But DNA shows UNKNOWN
    )
    print(f"Mismatch detected: {mismatch}")
    print(f"Reason: {reason}")


def demo_domain_policy():
    """Demonstrate domain money policy."""
    print("\n" + "=" * 80)
    print("DOMAIN MONEY POLICY")
    print("=" * 80)
    
    print("""
    Domains set their own acceptance criteria.
    They see ANGLE, not details.
    """)
    
    # Strict domain
    strict_domain = DomainMoneyPolicy(
        domain_id="bank",
        min_angle_small_transaction=50,
        min_angle_large_transaction=70,
        min_angle_investment=80,
        accept_flagged=False
    )
    
    # Lenient domain
    lenient_domain = DomainMoneyPolicy(
        domain_id="marketplace",
        min_angle_small_transaction=20,
        min_angle_large_transaction=40,
        accept_flagged=True
    )
    
    # Test money
    good_money = MoneyUnit(1000, MoneySourceType.SALARY, "employer", "alice")
    medium_money = MoneyUnit(1000, MoneySourceType.GIFT, "friend", "bob")
    bad_money = MoneyUnit(1000, MoneySourceType.SHELL_COMPANY, "shell", "charlie")
    
    print(f"\n--- Testing Different Money ---")
    print(f"Good money: {good_money.teeth.angle:.1f}°")
    print(f"Medium money: {medium_money.teeth.angle:.1f}°")
    print(f"Bad money: {bad_money.teeth.angle:.1f}°")
    
    print(f"\n--- Strict Domain (Bank) ---")
    for money, name in [(good_money, "Good"), (medium_money, "Medium"), (bad_money, "Bad")]:
        can, reason = strict_domain.can_accept(money, MoneyRNAType.SPEND_LARGE)
        print(f"{name} money for LARGE_SPEND: {can} - {reason}")
    
    print(f"\n--- Lenient Domain (Marketplace) ---")
    for money, name in [(good_money, "Good"), (medium_money, "Medium"), (bad_money, "Bad")]:
        can, reason = lenient_domain.can_accept(money, MoneyRNAType.SPEND_LARGE)
        print(f"{name} money for LARGE_SPEND: {can} - {reason}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("MONEY HEXAGON SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Money creation
    print("\n--- TEST 1: Money Creation ---")
    money = MoneyUnit(1000, MoneySourceType.SALARY, "employer", "alice")
    assert money.amount == 1000
    assert money.teeth.angle > 0
    print(f"✓ Money created with angle {money.teeth.angle:.1f}°")
    
    # Test 2: Provenance chain
    print("\n--- TEST 2: Provenance Chain ---")
    money.transfer("bob", "verified", True)
    assert money.dna.chain_length == 2
    print(f"✓ Chain length = {money.dna.chain_length}")
    
    # Test 3: RNA tokens
    print("\n--- TEST 3: RNA Tokens ---")
    available = money.rna.get_available_actions()
    assert len(available) > 0
    print(f"✓ {len(available)} action types available")
    
    # Test 4: Legitimacy bands
    print("\n--- TEST 4: Legitimacy Bands ---")
    band = LegitimacyBand.from_angle(85)
    assert band == LegitimacyBand.PRISTINE
    print(f"✓ 85° → {band.name}")
    
    # Test 5: Fake detection
    print("\n--- TEST 5: Fake Detection ---")
    bad_money = MoneyUnit(1000, MoneySourceType.NO_RECORD, "???", "suspicious")
    is_valid, issues = FakeMoneyDetector.check(bad_money)
    assert not is_valid
    print(f"✓ Fake money detected with {len(issues)} issues")
    
    # Test 6: Domain policy
    print("\n--- TEST 6: Domain Policy ---")
    policy = DomainMoneyPolicy("test", min_angle_large_transaction=70)
    can, _ = policy.can_accept(money, MoneyRNAType.SPEND_LARGE)
    assert can
    print(f"✓ Good money accepted by strict policy")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--dna":
            demo_money_dna()
        elif sys.argv[1] == "--rna":
            demo_money_rna()
        elif sys.argv[1] == "--bands":
            demo_legitimacy_bands()
        elif sys.argv[1] == "--fake":
            demo_fake_money_detection()
        elif sys.argv[1] == "--policy":
            demo_domain_policy()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_money_dna()
            demo_money_rna()
            demo_legitimacy_bands()
            demo_fake_money_detection()
            demo_domain_policy()
        else:
            run_tests()
    else:
        run_tests()
