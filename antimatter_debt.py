"""
ANTIMATTER DEBT SYSTEM
======================
Part 3: Debt as enforcement at the edges.

Key Insights:
1. Negative roof = antimatter budget (debt capacity)
2. Two types of debt:
   - Transactional (θ left): from exchange, user-side
   - Maintenance (θ right): from structure, owner-side
3. Signed integrals: I_base+, I_base-, I_rna+, I_rna-
4. Debt reduces permissible action budget
5. Antimatter cancels matter - self-enforcing!

The Three Parts:
- Part 1: Identity/Phase (Euler θ, continuity, role)
- Part 2: Cheap tokens from expensive checks (DNA → RNA)
- Part 3: Antimatter/Debt enforcement (THIS FILE)

Debt Tokens:
- MAINTAIN: must spend on upkeep
- PAY: transaction obligations  
- AUDIT: compliance costs
- REPAIR: incident response

Author: Jonathan Pelchat
Based on Shovelcat Theory + collaborative refinement
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
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
# TOKEN TYPES (Capability + Debt)
# ═══════════════════════════════════════════════════════════════════════════════

class CapabilityToken(Enum):
    """Positive RNA tokens - what you CAN do."""
    
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


class DebtToken(Enum):
    """Negative RNA tokens - what you MUST do."""
    
    # Maintenance debt (owner-side, from structure)
    MAINTAIN = (2, "Infrastructure upkeep", "maintenance")
    UPGRADE = (3, "System upgrades", "maintenance")
    BACKUP = (2, "Data backup", "maintenance")
    MONITOR = (1, "Health monitoring", "maintenance")
    
    # Transactional debt (user-side, from exchange)
    PAY = (3, "Payment obligations", "transactional")
    SUBSCRIBE = (2, "Subscription renewal", "transactional")
    SETTLE = (4, "Settlement required", "transactional")
    REFUND = (3, "Refund processing", "transactional")
    
    # Compliance debt (edge enforcement)
    AUDIT = (4, "Compliance audit", "compliance")
    REPORT = (2, "Regulatory reporting", "compliance")
    CERTIFY = (5, "Certification renewal", "compliance")
    
    # Incident debt (repair)
    REPAIR = (5, "Incident response", "incident")
    RECOVER = (4, "Recovery actions", "incident")
    INVESTIGATE = (3, "Investigation required", "incident")
    
    def __init__(self, cost: int, description: str, category: str):
        self.cost = cost
        self.description = description
        self.category = category
    
    @property
    def is_maintenance(self) -> bool:
        return self.category == "maintenance"
    
    @property
    def is_transactional(self) -> bool:
        return self.category == "transactional"
    
    @property
    def is_compliance(self) -> bool:
        return self.category == "compliance"
    
    @property
    def is_incident(self) -> bool:
        return self.category == "incident"


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNED INTEGRALS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignedIntegrals:
    """
    Split integrals into positive (matter) and negative (antimatter) parts.
    
    I_base+ = ∫ max(b(τ), 0) dτ  (value capacity)
    I_base- = ∫ min(b(τ), 0) dτ  (debt capacity, ≤ 0)
    
    I_square+ = B+ × T₀  (positive cap)
    I_square- = B- × T₀  (negative cap, ≤ 0)
    
    I_rna+ = I_square+ - I_base+  (capability slack)
    I_rna- = I_square- - I_base-  (debt headroom)
    """
    
    # Positive (matter/value)
    B_plus: float           # max(b(τ)) in window
    I_square_plus: float    # B+ × T₀
    I_base_plus: float      # ∫ max(b(τ), 0) dτ
    I_rna_plus: float       # capability slack
    
    # Negative (antimatter/debt)
    B_minus: float          # min(b(τ)) in window (≤ 0)
    I_square_minus: float   # B- × T₀ (≤ 0)
    I_base_minus: float     # ∫ min(b(τ), 0) dτ (≤ 0)
    I_rna_minus: float      # debt headroom (≤ 0)
    
    # Derived
    @property
    def total_value_capacity(self) -> float:
        """Total positive capacity."""
        return self.I_square_plus
    
    @property
    def total_debt_capacity(self) -> float:
        """Total negative capacity (absolute value)."""
        return abs(self.I_square_minus)
    
    @property
    def net_capacity(self) -> float:
        """Net = value - debt."""
        return self.I_square_plus + self.I_square_minus
    
    @property
    def matter_antimatter_ratio(self) -> float:
        """Ratio of matter to antimatter."""
        if abs(self.I_square_minus) < 0.001:
            return float('inf')
        return self.I_square_plus / abs(self.I_square_minus)
    
    @property
    def debt_headroom_remaining(self) -> float:
        """How much more debt before collapse."""
        return abs(self.I_rna_minus)


# ═══════════════════════════════════════════════════════════════════════════════
# DEBT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass 
class DebtObligation:
    """A single debt obligation."""
    
    debt_type: DebtToken
    amount: float
    created_at: float
    due_by: float = None  # Optional deadline
    paid: float = 0.0
    
    @property
    def remaining(self) -> float:
        return max(0, self.amount - self.paid)
    
    @property
    def is_paid(self) -> bool:
        return self.remaining < 0.001
    
    @property
    def is_overdue(self) -> bool:
        if self.due_by is None:
            return False
        return time.time() > self.due_by and not self.is_paid


class DebtLedger:
    """
    Tracks all debt obligations.
    
    Two categories:
    - Transactional: from exchange with others
    - Maintenance: from structure you built
    """
    
    def __init__(self):
        self.obligations: List[DebtObligation] = []
        self.history: List[Dict] = []
    
    def incur_debt(self, 
                  debt_type: DebtToken, 
                  amount: float,
                  due_in: float = None) -> DebtObligation:
        """Incur a new debt obligation."""
        
        now = time.time()
        due_by = now + due_in if due_in else None
        
        obligation = DebtObligation(
            debt_type=debt_type,
            amount=amount,
            created_at=now,
            due_by=due_by
        )
        
        self.obligations.append(obligation)
        self.history.append({
            'action': 'incur',
            'type': debt_type.name,
            'amount': amount,
            'time': now
        })
        
        return obligation
    
    def pay_debt(self, obligation: DebtObligation, amount: float) -> float:
        """Pay down a debt. Returns amount actually paid."""
        
        actual = min(amount, obligation.remaining)
        obligation.paid += actual
        
        self.history.append({
            'action': 'pay',
            'type': obligation.debt_type.name,
            'amount': actual,
            'time': time.time()
        })
        
        return actual
    
    def get_total_debt(self) -> float:
        """Total outstanding debt."""
        return sum(o.remaining for o in self.obligations)
    
    def get_transactional_debt(self) -> float:
        """Debt from transactions."""
        return sum(o.remaining for o in self.obligations 
                  if o.debt_type.is_transactional)
    
    def get_maintenance_debt(self) -> float:
        """Debt from maintenance."""
        return sum(o.remaining for o in self.obligations 
                  if o.debt_type.is_maintenance)
    
    def get_compliance_debt(self) -> float:
        """Debt from compliance."""
        return sum(o.remaining for o in self.obligations 
                  if o.debt_type.is_compliance)
    
    def get_incident_debt(self) -> float:
        """Debt from incidents."""
        return sum(o.remaining for o in self.obligations 
                  if o.debt_type.is_incident)
    
    def get_overdue(self) -> List[DebtObligation]:
        """Get all overdue obligations."""
        return [o for o in self.obligations if o.is_overdue]
    
    def cleanup_paid(self):
        """Remove fully paid obligations."""
        self.obligations = [o for o in self.obligations if not o.is_paid]


# ═══════════════════════════════════════════════════════════════════════════════
# ANTIMATTER H-WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AntimatterWindowResult:
    """Result of antimatter-aware H-window measurement."""
    
    window_start: float
    window_end: float
    period: float
    
    # Signed integrals
    integrals: SignedIntegrals
    
    # Dev spending
    I_dev: float
    
    # Debt state
    D_transactional: float  # θ left debt
    D_maintenance: float    # θ right debt
    D_compliance: float
    D_incident: float
    D_total: float
    
    # Effective budget (after debt reduction)
    effective_budget: float
    budget_consumed: float
    budget_remaining: float
    
    # Capability tokens
    capability_tokens: int
    capability_by_type: Dict[CapabilityToken, int]
    
    # Debt tokens (what must be spent)
    debt_tokens: int
    debt_by_type: Dict[DebtToken, int]
    
    # Status
    in_debt_crisis: bool    # debt exceeds capacity
    antimatter_dominant: bool  # more debt than value
    overdue_count: int


class AntimatterWindow:
    """
    H-Window with antimatter/debt tracking.
    
    The key enforcement equation:
    
    I_dev ≤ I_base+ + η×I_rna+ - λ×D_maint - μ×D_txn
    
    Debt REDUCES your action budget!
    """
    
    def __init__(self,
                 period: float = HOURS,
                 base_amplitude: float = 10.0,
                 base_offset: float = 5.0,     # Can go negative!
                 eta: float = 0.5,             # Slack flexibility
                 lambda_maint: float = 1.0,    # Maintenance debt weight
                 mu_txn: float = 1.0,          # Transactional debt weight
                 rna_scale: float = 1.0,
                 steepness_scale: float = 0.1):
        
        self.period = period
        self.base_amplitude = base_amplitude
        self.base_offset = base_offset
        self.eta = eta
        self.lambda_maint = lambda_maint
        self.mu_txn = mu_txn
        self.rna_scale = rna_scale
        self.steepness_scale = steepness_scale
        
        # Debt ledger
        self.debt_ledger = DebtLedger()
        
        # Current window tracking
        self.current_window_start = time.time()
        self.current_dev_spending = 0.0
        
        # History
        self.history: deque = deque(maxlen=100)
    
    def base_curve(self, t: float) -> float:
        """Base curve - CAN go negative (antimatter region)."""
        phase = 2 * PI * t / self.period
        return self.base_offset + self.base_amplitude * math.cos(phase)
    
    def base_derivative(self, t: float) -> float:
        """Derivative of base curve."""
        phase = 2 * PI * t / self.period
        return -self.base_amplitude * (2 * PI / self.period) * math.sin(phase)
    
    def compute_signed_integrals(self,
                                 window_start: float,
                                 window_end: float,
                                 samples: int = 100) -> SignedIntegrals:
        """
        Compute signed integrals (matter + antimatter).
        """
        dt = (window_end - window_start) / samples
        T0 = window_end - window_start
        
        # Sample the curve
        values = [self.base_curve(window_start + i * dt) for i in range(samples + 1)]
        
        # Find caps
        B_plus = max(max(values), 0)  # At least 0
        B_minus = min(min(values), 0)  # At most 0
        
        # Square areas
        I_square_plus = B_plus * T0
        I_square_minus = B_minus * T0  # ≤ 0
        
        # Base integrals (split by sign)
        I_base_plus = 0.0
        I_base_minus = 0.0
        
        for i in range(samples):
            t1 = window_start + i * dt
            t2 = window_start + (i + 1) * dt
            v1 = self.base_curve(t1)
            v2 = self.base_curve(t2)
            avg = (v1 + v2) / 2
            
            if avg > 0:
                I_base_plus += avg * dt
            else:
                I_base_minus += avg * dt  # Negative contribution
        
        # RNA (remainder)
        I_rna_plus = max(0, I_square_plus - I_base_plus)
        I_rna_minus = I_square_minus - I_base_minus  # ≤ 0 (debt headroom)
        
        return SignedIntegrals(
            B_plus=B_plus,
            I_square_plus=I_square_plus,
            I_base_plus=I_base_plus,
            I_rna_plus=I_rna_plus,
            B_minus=B_minus,
            I_square_minus=I_square_minus,
            I_base_minus=I_base_minus,
            I_rna_minus=I_rna_minus
        )
    
    def compute_steepness(self,
                         window_start: float,
                         window_end: float,
                         samples: int = 100) -> float:
        """Compute steepness (affects maintenance debt)."""
        dt = (window_end - window_start) / samples
        
        S = 0.0
        for i in range(samples):
            t = window_start + i * dt
            S += abs(self.base_derivative(t)) * dt
        
        S /= (window_end - window_start)
        return S / (S + self.steepness_scale)
    
    def record_action(self, cost: float):
        """Record dev spending."""
        self.current_dev_spending += cost
    
    def incur_debt(self, debt_type: DebtToken, amount: float, due_in: float = None):
        """Incur a debt obligation."""
        return self.debt_ledger.incur_debt(debt_type, amount, due_in)
    
    def pay_debt(self, obligation: DebtObligation, amount: float):
        """Pay down a debt."""
        return self.debt_ledger.pay_debt(obligation, amount)
    
    def mint_capability_tokens(self, 
                              I_rna_plus: float,
                              steepness: float) -> Tuple[int, Dict[CapabilityToken, int]]:
        """Mint capability (positive) tokens."""
        
        total = int(I_rna_plus * self.rna_scale)
        
        tokens = {}
        
        # Steepness shifts distribution
        if steepness > 0.7:
            # Steep = more user exploration
            tokens[CapabilityToken.READ] = int(total * 0.3)
            tokens[CapabilityToken.LIST] = int(total * 0.2)
            tokens[CapabilityToken.WRITE] = int(total * 0.3)
            tokens[CapabilityToken.DELETE] = int(total * 0.1)
            tokens[CapabilityToken.EXECUTE] = int(total * 0.1)
        else:
            # Smooth = more controlled ops
            tokens[CapabilityToken.READ] = int(total * 0.4)
            tokens[CapabilityToken.LIST] = int(total * 0.3)
            tokens[CapabilityToken.WRITE] = int(total * 0.2)
            tokens[CapabilityToken.ADMIN] = int(total * 0.1)
        
        return total, tokens
    
    def mint_debt_tokens(self,
                        I_rna_minus: float,
                        D_maintenance: float,
                        D_transactional: float,
                        steepness: float) -> Tuple[int, Dict[DebtToken, int]]:
        """
        Mint debt (negative) tokens - obligations that MUST be fulfilled.
        
        Steep curve → more maintenance debt
        Smooth curve → more transactional debt
        """
        
        # Total debt tokens from negative remainder
        total = int(abs(I_rna_minus) * self.rna_scale)
        
        tokens = {}
        
        if total == 0:
            return 0, tokens
        
        # Split based on steepness
        maint_ratio = steepness
        txn_ratio = 1 - steepness
        
        # Maintenance tokens (owner-side, from structure)
        maint_pool = int(total * maint_ratio)
        tokens[DebtToken.MAINTAIN] = int(maint_pool * 0.4)
        tokens[DebtToken.UPGRADE] = int(maint_pool * 0.3)
        tokens[DebtToken.BACKUP] = int(maint_pool * 0.2)
        tokens[DebtToken.MONITOR] = int(maint_pool * 0.1)
        
        # Transactional tokens (user-side, from exchange)
        txn_pool = int(total * txn_ratio)
        tokens[DebtToken.PAY] = int(txn_pool * 0.4)
        tokens[DebtToken.SUBSCRIBE] = int(txn_pool * 0.3)
        tokens[DebtToken.SETTLE] = int(txn_pool * 0.2)
        tokens[DebtToken.REFUND] = int(txn_pool * 0.1)
        
        # Compliance tokens (always some)
        compliance = max(1, int(total * 0.1))
        tokens[DebtToken.AUDIT] = compliance // 2
        tokens[DebtToken.REPORT] = compliance - compliance // 2
        
        return total, tokens
    
    def compute_effective_budget(self,
                                integrals: SignedIntegrals,
                                D_maint: float,
                                D_txn: float) -> float:
        """
        Compute effective action budget after debt reduction.
        
        Budget = I_base+ + η×I_rna+ - λ×D_maint - μ×D_txn
        
        Debt LITERALLY reduces what you can do!
        """
        
        base_budget = integrals.I_base_plus + self.eta * integrals.I_rna_plus
        debt_reduction = self.lambda_maint * D_maint + self.mu_txn * D_txn
        
        return max(0, base_budget - debt_reduction)
    
    def measure_window(self,
                      window_start: float = None,
                      window_end: float = None) -> AntimatterWindowResult:
        """Measure one H-window with antimatter tracking."""
        
        if window_start is None:
            window_start = self.current_window_start
        if window_end is None:
            window_end = window_start + self.period
        
        # Compute signed integrals
        integrals = self.compute_signed_integrals(window_start, window_end)
        
        # Compute steepness
        steepness = self.compute_steepness(window_start, window_end)
        
        # Get debt state
        D_txn = self.debt_ledger.get_transactional_debt()
        D_maint = self.debt_ledger.get_maintenance_debt()
        D_compliance = self.debt_ledger.get_compliance_debt()
        D_incident = self.debt_ledger.get_incident_debt()
        D_total = self.debt_ledger.get_total_debt()
        
        # Compute effective budget
        effective_budget = self.compute_effective_budget(integrals, D_maint, D_txn)
        
        # Mint capability tokens
        cap_total, cap_by_type = self.mint_capability_tokens(
            integrals.I_rna_plus, steepness
        )
        
        # Mint debt tokens
        debt_total, debt_by_type = self.mint_debt_tokens(
            integrals.I_rna_minus, D_maint, D_txn, steepness
        )
        
        # Status checks
        in_debt_crisis = D_total > integrals.I_base_plus
        antimatter_dominant = abs(integrals.I_square_minus) > integrals.I_square_plus
        overdue = self.debt_ledger.get_overdue()
        
        result = AntimatterWindowResult(
            window_start=window_start,
            window_end=window_end,
            period=self.period,
            integrals=integrals,
            I_dev=self.current_dev_spending,
            D_transactional=D_txn,
            D_maintenance=D_maint,
            D_compliance=D_compliance,
            D_incident=D_incident,
            D_total=D_total,
            effective_budget=effective_budget,
            budget_consumed=self.current_dev_spending,
            budget_remaining=max(0, effective_budget - self.current_dev_spending),
            capability_tokens=cap_total,
            capability_by_type=cap_by_type,
            debt_tokens=debt_total,
            debt_by_type=debt_by_type,
            in_debt_crisis=in_debt_crisis,
            antimatter_dominant=antimatter_dominant,
            overdue_count=len(overdue)
        )
        
        self.history.append(result)
        
        return result
    
    def reset_window(self):
        """Start a new window."""
        self.current_window_start = time.time()
        self.current_dev_spending = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ANTIMATTER ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class AntimatterEnforcer:
    """
    Enforces debt obligations and prevents antimatter crisis.
    
    When debt exceeds capacity:
    - Capability tokens are cancelled
    - Actions are blocked
    - Recovery mode activates
    """
    
    def __init__(self, window: AntimatterWindow):
        self.window = window
        self.blocked_actions: List[str] = []
        self.recovery_mode = False
    
    def can_perform_action(self, 
                          action: CapabilityToken,
                          result: AntimatterWindowResult) -> Tuple[bool, str]:
        """Check if an action can be performed given debt state."""
        
        # In recovery mode, only MAINTAIN/PAY actions allowed
        if self.recovery_mode:
            return False, "RECOVERY_MODE: Only debt repayment allowed"
        
        # Antimatter dominant = no new capabilities
        if result.antimatter_dominant:
            return False, "ANTIMATTER_DOMINANT: Debt exceeds value capacity"
        
        # Debt crisis = limited actions
        if result.in_debt_crisis:
            if action.cost > 2:
                return False, "DEBT_CRISIS: Only low-cost actions allowed"
        
        # Check if we have tokens for this action
        available = result.capability_by_type.get(action, 0)
        if available <= 0:
            return False, f"NO_TOKENS: No {action.name} tokens available"
        
        # Check budget
        if action.cost > result.budget_remaining:
            return False, f"BUDGET_EXCEEDED: Cost {action.cost} > remaining {result.budget_remaining:.2f}"
        
        return True, "ALLOWED"
    
    def must_perform_debt_action(self, result: AntimatterWindowResult) -> List[DebtToken]:
        """Get mandatory debt actions that must be performed."""
        
        mandatory = []
        
        # If overdue obligations exist
        if result.overdue_count > 0:
            mandatory.append(DebtToken.PAY)
            mandatory.append(DebtToken.SETTLE)
        
        # If in debt crisis, maintenance is mandatory
        if result.in_debt_crisis:
            mandatory.append(DebtToken.MAINTAIN)
        
        # If antimatter dominant, recovery actions
        if result.antimatter_dominant:
            mandatory.append(DebtToken.REPAIR)
            mandatory.append(DebtToken.RECOVER)
        
        return mandatory
    
    def enter_recovery_mode(self):
        """Enter recovery mode - only debt repayment allowed."""
        self.recovery_mode = True
    
    def exit_recovery_mode(self, result: AntimatterWindowResult) -> bool:
        """Try to exit recovery mode."""
        if not result.antimatter_dominant and not result.in_debt_crisis:
            self.recovery_mode = False
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_signed_integrals():
    """Demonstrate signed integrals (matter + antimatter)."""
    print("\n" + "=" * 80)
    print("SIGNED INTEGRALS: Matter + Antimatter")
    print("=" * 80)
    
    print("""
    When base curve goes NEGATIVE, we get antimatter region:
    
    B+ ─────────────────────────────── POSITIVE CAP
       █████████████████████████████
       █████████ RNA+ █████████████   ← capability slack
          ╱╲         ╱╲         ╱╲
         ╱  ╲       ╱  ╲       ╱  ╲    ← I_base+ (value)
    ════════════════════════════════════ ZERO ═════════════
         ╲  ╱       ╲  ╱       ╲  ╱    ← I_base- (debt)
          ╲╱         ╲╱         ╲╱
       ░░░░░░░░░ RNA- ░░░░░░░░░░░   ← debt headroom
    B- ─────────────────────────────── NEGATIVE CAP
    """)
    
    # Different offset scenarios
    print("\n--- Different Offset Scenarios ---\n")
    print("┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
    print("│ Offset     │ I_base+    │ I_base-    │ I_rna+     │ I_rna-     │ Net        │")
    print("├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")
    
    for offset in [15.0, 10.0, 5.0, 0.0, -5.0]:
        w = AntimatterWindow(period=1.0, base_amplitude=10.0, base_offset=offset)
        integrals = w.compute_signed_integrals(0, 1.0)
        
        net = integrals.I_base_plus + integrals.I_base_minus
        
        print(f"│ {offset:10.1f} │ {integrals.I_base_plus:10.2f} │ {integrals.I_base_minus:10.2f} │ {integrals.I_rna_plus:10.2f} │ {integrals.I_rna_minus:10.2f} │ {net:10.2f} │")
    
    print("└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")
    
    print("""
    
    Offset = 15:  All positive (pure matter)
    Offset = 10:  Mostly positive
    Offset = 5:   Some negative (enters antimatter)
    Offset = 0:   Half positive, half negative
    Offset = -5:  Mostly negative (antimatter dominant!)
    """)


def demo_two_debts():
    """Demonstrate two types of debt."""
    print("\n" + "=" * 80)
    print("TWO TYPES OF DEBT")
    print("=" * 80)
    
    print("""
    TRANSACTIONAL DEBT (θ left):         MAINTENANCE DEBT (θ right):
    ────────────────────────────         ────────────────────────────
    
    From EXCHANGE with others            From STRUCTURE you built
    
    • PAY: Payment obligations           • MAINTAIN: Infrastructure
    • SUBSCRIBE: Renewals                • UPGRADE: System updates
    • SETTLE: Settlements                • BACKUP: Data protection
    • REFUND: Processing                 • MONITOR: Health checks
    
    → USER-SIDE antimatter               → OWNER-SIDE antimatter
    """)
    
    print("\n--- Debt Token Categories ---\n")
    print("┌────────────────┬──────────┬──────────────────────────────────┐")
    print("│ Token          │ Cost     │ Category                         │")
    print("├────────────────┼──────────┼──────────────────────────────────┤")
    
    for dt in DebtToken:
        print(f"│ {dt.name:14} │ {dt.cost:8} │ {dt.category:32} │")
    
    print("└────────────────┴──────────┴──────────────────────────────────┘")


def demo_debt_reduces_budget():
    """Demonstrate how debt reduces action budget."""
    print("\n" + "=" * 80)
    print("DEBT REDUCES ACTION BUDGET")
    print("=" * 80)
    
    print("""
    The enforcement equation:
    
    Budget = I_base+ + η×I_rna+ - λ×D_maint - μ×D_txn
             ─────────────────   ─────────────────────
                capability          debt reduction
    
    More debt = LESS you can do!
    Antimatter cancels matter!
    """)
    
    w = AntimatterWindow(period=1.0, base_amplitude=5.0, base_offset=10.0)
    
    # No debt
    result = w.measure_window(0, 1.0)
    print(f"\n--- No Debt ---")
    print(f"I_base+: {result.integrals.I_base_plus:.2f}")
    print(f"I_rna+: {result.integrals.I_rna_plus:.2f}")
    print(f"Effective budget: {result.effective_budget:.2f}")
    
    # Add transactional debt
    w.incur_debt(DebtToken.PAY, 3.0)
    w.incur_debt(DebtToken.SUBSCRIBE, 2.0)
    result = w.measure_window(0, 1.0)
    print(f"\n--- With Transactional Debt (5.0) ---")
    print(f"D_transactional: {result.D_transactional:.2f}")
    print(f"Effective budget: {result.effective_budget:.2f}")
    print(f"Budget reduced by: {result.integrals.I_base_plus + 0.5 * result.integrals.I_rna_plus - result.effective_budget:.2f}")
    
    # Add maintenance debt
    w.incur_debt(DebtToken.MAINTAIN, 4.0)
    w.incur_debt(DebtToken.UPGRADE, 3.0)
    result = w.measure_window(0, 1.0)
    print(f"\n--- With Both Debt Types ---")
    print(f"D_transactional: {result.D_transactional:.2f}")
    print(f"D_maintenance: {result.D_maintenance:.2f}")
    print(f"D_total: {result.D_total:.2f}")
    print(f"Effective budget: {result.effective_budget:.2f}")
    print(f"In debt crisis: {result.in_debt_crisis}")


def demo_antimatter_dominant():
    """Demonstrate antimatter-dominant state."""
    print("\n" + "=" * 80)
    print("ANTIMATTER DOMINANT STATE")
    print("=" * 80)
    
    print("""
    When negative capacity exceeds positive:
    
    • Antimatter dominant = TRUE
    • Capability tokens cancelled
    • Only debt repayment allowed
    • RECOVERY MODE activates
    """)
    
    # Create window that goes negative
    w = AntimatterWindow(period=1.0, base_amplitude=15.0, base_offset=0.0)
    result = w.measure_window(0, 1.0)
    
    print(f"\n--- Antimatter Window (offset=0, amp=15) ---")
    print(f"I_square+: {result.integrals.I_square_plus:.2f}")
    print(f"I_square-: {result.integrals.I_square_minus:.2f}")
    print(f"Net capacity: {result.integrals.net_capacity:.2f}")
    print(f"Matter/Antimatter ratio: {result.integrals.matter_antimatter_ratio:.2f}")
    print(f"Antimatter dominant: {result.antimatter_dominant}")
    
    # More extreme
    w2 = AntimatterWindow(period=1.0, base_amplitude=15.0, base_offset=-5.0)
    result2 = w2.measure_window(0, 1.0)
    
    print(f"\n--- Deep Antimatter (offset=-5, amp=15) ---")
    print(f"I_square+: {result2.integrals.I_square_plus:.2f}")
    print(f"I_square-: {result2.integrals.I_square_minus:.2f}")
    print(f"Net capacity: {result2.integrals.net_capacity:.2f}")
    print(f"Matter/Antimatter ratio: {result2.integrals.matter_antimatter_ratio:.2f}")
    print(f"Antimatter dominant: {result2.antimatter_dominant}")


def demo_token_minting():
    """Demonstrate capability + debt token minting."""
    print("\n" + "=" * 80)
    print("TOKEN MINTING: Capability + Debt")
    print("=" * 80)
    
    print("""
    From POSITIVE remainder → CAPABILITY tokens (what you CAN do)
    From NEGATIVE remainder → DEBT tokens (what you MUST do)
    """)
    
    w = AntimatterWindow(
        period=1.0, 
        base_amplitude=10.0, 
        base_offset=5.0,  # Some negative region
        rna_scale=10.0
    )
    
    result = w.measure_window(0, 1.0)
    
    print(f"\n--- Token Minting ---")
    print(f"\nCapability tokens: {result.capability_tokens}")
    for token, count in result.capability_by_type.items():
        if count > 0:
            print(f"  {token.name:10}: {count}")
    
    print(f"\nDebt tokens: {result.debt_tokens}")
    for token, count in result.debt_by_type.items():
        if count > 0:
            print(f"  {token.name:10}: {count} ({token.category})")


def demo_enforcement():
    """Demonstrate antimatter enforcement."""
    print("\n" + "=" * 80)
    print("ANTIMATTER ENFORCEMENT")
    print("=" * 80)
    
    print("""
    The enforcer checks:
    1. Are we in debt crisis?
    2. Is antimatter dominant?
    3. Do we have tokens for this action?
    4. Is budget sufficient?
    """)
    
    w = AntimatterWindow(period=1.0, base_amplitude=5.0, base_offset=10.0, rna_scale=10.0)
    enforcer = AntimatterEnforcer(w)
    
    result = w.measure_window(0, 1.0)
    
    print(f"\n--- Normal State ---")
    print(f"Effective budget: {result.effective_budget:.2f}")
    print(f"In debt crisis: {result.in_debt_crisis}")
    
    for action in [CapabilityToken.READ, CapabilityToken.WRITE, CapabilityToken.ADMIN]:
        can, reason = enforcer.can_perform_action(action, result)
        print(f"  {action.name}: {can} - {reason}")
    
    # Add massive debt
    for _ in range(5):
        w.incur_debt(DebtToken.PAY, 5.0)
        w.incur_debt(DebtToken.MAINTAIN, 5.0)
    
    result = w.measure_window(0, 1.0)
    
    print(f"\n--- Debt Crisis State ---")
    print(f"Total debt: {result.D_total:.2f}")
    print(f"Effective budget: {result.effective_budget:.2f}")
    print(f"In debt crisis: {result.in_debt_crisis}")
    
    for action in [CapabilityToken.READ, CapabilityToken.WRITE, CapabilityToken.ADMIN]:
        can, reason = enforcer.can_perform_action(action, result)
        print(f"  {action.name}: {can} - {reason}")
    
    print(f"\nMandatory actions: {[d.name for d in enforcer.must_perform_debt_action(result)]}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("ANTIMATTER DEBT SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Signed integrals
    print("\n--- TEST 1: Signed Integrals ---")
    w = AntimatterWindow(period=1.0, base_amplitude=10.0, base_offset=5.0)
    integrals = w.compute_signed_integrals(0, 1.0)
    assert integrals.I_base_plus >= 0
    assert integrals.I_base_minus <= 0
    print(f"✓ I_base+ ≥ 0 ({integrals.I_base_plus:.2f}), I_base- ≤ 0 ({integrals.I_base_minus:.2f})")
    
    # Test 2: Debt ledger
    print("\n--- TEST 2: Debt Ledger ---")
    ledger = DebtLedger()
    ob = ledger.incur_debt(DebtToken.PAY, 10.0)
    assert ledger.get_total_debt() == 10.0
    ledger.pay_debt(ob, 4.0)
    assert ledger.get_total_debt() == 6.0
    print(f"✓ Debt tracking works")
    
    # Test 3: Budget reduction
    print("\n--- TEST 3: Budget Reduction ---")
    w = AntimatterWindow(period=1.0, base_offset=10.0)
    result1 = w.measure_window(0, 1.0)
    budget1 = result1.effective_budget
    
    w.incur_debt(DebtToken.MAINTAIN, 5.0)
    result2 = w.measure_window(0, 1.0)
    budget2 = result2.effective_budget
    
    assert budget2 < budget1
    print(f"✓ Debt reduces budget ({budget1:.2f} → {budget2:.2f})")
    
    # Test 4: Antimatter dominant
    print("\n--- TEST 4: Antimatter Dominant ---")
    w = AntimatterWindow(period=1.0, base_amplitude=15.0, base_offset=-5.0)
    result = w.measure_window(0, 1.0)
    assert result.antimatter_dominant
    print(f"✓ Antimatter dominant detected")
    
    # Test 5: Token minting
    print("\n--- TEST 5: Token Minting ---")
    w = AntimatterWindow(period=1.0, base_offset=10.0, rna_scale=10.0)
    result = w.measure_window(0, 1.0)
    assert result.capability_tokens > 0
    print(f"✓ Capability tokens minted: {result.capability_tokens}")
    
    # Test 6: Enforcement
    print("\n--- TEST 6: Enforcement ---")
    w = AntimatterWindow(period=1.0, rna_scale=10.0)
    enforcer = AntimatterEnforcer(w)
    result = w.measure_window(0, 1.0)
    can, _ = enforcer.can_perform_action(CapabilityToken.READ, result)
    assert can
    print(f"✓ Enforcement allows valid actions")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--signed":
            demo_signed_integrals()
        elif sys.argv[1] == "--debts":
            demo_two_debts()
        elif sys.argv[1] == "--budget":
            demo_debt_reduces_budget()
        elif sys.argv[1] == "--antimatter":
            demo_antimatter_dominant()
        elif sys.argv[1] == "--tokens":
            demo_token_minting()
        elif sys.argv[1] == "--enforce":
            demo_enforcement()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_signed_integrals()
            demo_two_debts()
            demo_debt_reduces_budget()
            demo_antimatter_dominant()
            demo_token_minting()
            demo_enforcement()
        else:
            run_tests()
    else:
        run_tests()
