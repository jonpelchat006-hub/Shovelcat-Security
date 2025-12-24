"""
EULER IDENTITY ACCESS CONTROL
=============================
Using e^(iπ) = -1 to model user ↔ owner duality
and complex phase for trust/access levels.

Core Insights:
- Owner = +1, User = -1 (sum to zero, complementary)
- Trust evolves as phase rotation (θ: 0 → π)
- Departments are phase windows (hexagon alignment)
- Data visibility = Re(Data · e^(-iθ))
- H-windows define organizational resolution

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import cmath
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
E = math.e
I = complex(0, 1)  # Imaginary unit

# Euler's identity components
OWNER_IDENTITY = 1      # +1
USER_IDENTITY = -1      # -1 = e^(iπ)

# Trust evolution constants
MAX_TRUST_RATE = PI / 365  # Max phase gain per day (1 year to full trust)
TRUST_THRESHOLD_PARTNER = PI / 2  # 90° = trusted partner
TRUST_THRESHOLD_STAFF = 3 * PI / 4  # 135° = staff level


# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT WINDOWS (H-Resolution)
# ═══════════════════════════════════════════════════════════════════════════════

class Department(Enum):
    """
    Departments as phase windows, hexagonally arranged.
    Each department has a base phase angle.
    """
    SALES = 0           # 0° - Customer-facing
    MARKETING = 60      # 60° - Brand/communication
    TECHNOLOGY = 120    # 120° - Technical/IT
    FINANCE = 180       # 180° - Money/accounting (opposite of sales!)
    HR = 240            # 240° - People/internal
    LEGAL = 300         # 300° - Compliance/risk
    
    @property
    def phase(self) -> float:
        """Phase angle in radians."""
        return math.radians(self.value)
    
    @property
    def opposite(self) -> 'Department':
        """The opposite department (180° away)."""
        opposite_angle = (self.value + 180) % 360
        for dept in Department:
            if dept.value == opposite_angle:
                return dept
        return self  # Fallback


@dataclass
class HWindow:
    """
    An H-window defining organizational resolution.
    Like Planck's constant - defines minimum observable unit.
    """
    name: str
    h_value: float  # Smaller = finer resolution
    max_depth: int  # How many sub-levels visible
    departments: List[Department] = field(default_factory=list)
    
    @property
    def visible_states(self) -> int:
        """Number of visible organizational states."""
        return int(1 / self.h_value) if self.h_value > 0 else 1000
    
    def can_resolve(self, depth: int) -> bool:
        """Can this window resolve detail at given depth?"""
        return depth <= self.max_depth


# Pre-defined organizational sizes
ORG_SIZES = {
    'startup': HWindow(
        name="Startup",
        h_value=0.33,  # ~3 visible states
        max_depth=1,
        departments=[Department.SALES, Department.TECHNOLOGY]
    ),
    'small': HWindow(
        name="Small Company",
        h_value=0.15,  # ~7 visible states
        max_depth=2,
        departments=[Department.SALES, Department.MARKETING, 
                    Department.TECHNOLOGY, Department.FINANCE]
    ),
    'medium': HWindow(
        name="Medium Company",
        h_value=0.05,  # ~20 visible states
        max_depth=3,
        departments=list(Department)
    ),
    'enterprise': HWindow(
        name="Enterprise",
        h_value=0.01,  # ~100 visible states
        max_depth=5,
        departments=list(Department)
    ),
    'megacorp': HWindow(
        name="Megacorp",
        h_value=0.001,  # ~1000 visible states
        max_depth=7,
        departments=list(Department)
    )
}


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTITY PHASE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IdentityPhase:
    """
    A user's identity as a complex phase.
    
    e^(iθ) where:
    - θ = 0: New user (pure customer view)
    - θ = π/2: Trusted partner (equal customer/staff)
    - θ = π: Owner (pure admin view)
    """
    user_id: str
    theta: float = 0.0  # Phase angle in radians
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Trust components
    time_investment: float = 0.0
    monetary_investment: float = 0.0
    trust_score: float = 0.0
    violations: float = 0.0
    
    # Department affiliation (if any)
    department: Optional[Department] = None
    is_staff: bool = False
    
    def __post_init__(self):
        """Initialize signature from user_id."""
        self._signature = self._compute_signature()
    
    def _compute_signature(self) -> float:
        """Compute unique phase signature from user_id."""
        hash_bytes = hashlib.sha256(self.user_id.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], 'big')
        # Tiny offset unique to this user
        return (hash_int / (2**64)) * 0.001
    
    @property
    def identity(self) -> complex:
        """The complex identity e^(iθ)."""
        return cmath.exp(I * self.theta)
    
    @property
    def cos_component(self) -> float:
        """Real component - customer view strength."""
        return math.cos(self.theta)
    
    @property
    def sin_component(self) -> float:
        """Imaginary component - staff view strength."""
        return math.sin(self.theta)
    
    @property
    def total_phase(self) -> float:
        """Total phase including department offset and signature."""
        base = self.theta
        if self.department:
            base += self.department.phase
        return base + self._signature
    
    @property
    def trust_level(self) -> str:
        """Human-readable trust level."""
        if self.theta >= PI - 0.1:
            return "owner"
        elif self.theta >= TRUST_THRESHOLD_STAFF:
            return "staff"
        elif self.theta >= TRUST_THRESHOLD_PARTNER:
            return "trusted_partner"
        elif self.theta >= PI / 4:
            return "invested_user"
        elif self.theta >= PI / 8:
            return "regular_user"
        else:
            return "new_user"
    
    def evolve_trust(self, days: float = 1, investment: float = 0, 
                     trust_delta: float = 0, violation: bool = False):
        """
        Evolve the trust phase over time.
        """
        # Time-based evolution (slow, steady)
        time_gain = min(days * (PI / 730), MAX_TRUST_RATE)  # 2 years for time alone
        
        # Investment-based evolution
        investment_gain = math.log1p(investment / 1000) * 0.1  # Diminishing returns
        
        # Direct trust adjustments
        trust_gain = trust_delta
        
        # Violations decrease trust
        violation_loss = 0.1 if violation else 0
        
        # Update components
        self.time_investment += time_gain
        self.monetary_investment += investment_gain
        self.trust_score += trust_gain
        self.violations += violation_loss
        
        # Compute new theta
        self.theta = min(PI, max(0, 
            self.time_investment + 
            self.monetary_investment + 
            self.trust_score - 
            self.violations
        ))
        
        self.last_updated = datetime.now()
    
    def verify_phase_continuity(self, claimed_theta: float) -> Tuple[bool, str]:
        """
        Verify that a claimed phase is consistent with history.
        Detects phase jumping attacks.
        """
        # Check if claimed phase is reasonable given account age
        days_old = (datetime.now() - self.created_at).days + 1
        max_reasonable_theta = min(PI, days_old * MAX_TRUST_RATE * 3)  # 3x buffer
        
        if claimed_theta > max_reasonable_theta:
            return False, f"Phase {claimed_theta:.2f} exceeds max {max_reasonable_theta:.2f} for {days_old} day old account"
        
        # Check for sudden jumps
        if abs(claimed_theta - self.theta) > PI / 4:  # >45° jump
            return False, f"Phase jump of {abs(claimed_theta - self.theta):.2f} exceeds allowed 0.79"
        
        return True, "Phase verified"
    
    def __str__(self):
        return f"Identity[{self.user_id}]: θ={math.degrees(self.theta):.1f}° ({self.trust_level})"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEX DATA VISIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComplexData:
    """
    Data with public (real) and private (imaginary) components.
    
    D = D_public + i·D_private
    
    What you see depends on your phase!
    """
    name: str
    public_value: Any  # Real component (customer sees)
    private_value: Any  # Imaginary component (staff sees)
    required_phase: float = 0.0  # Minimum phase to see private
    department: Optional[Department] = None  # Department this belongs to
    
    def visible_to(self, identity: IdentityPhase) -> Dict[str, Any]:
        """
        Calculate what's visible to an identity at given phase.
        """
        # Base visibility calculation
        theta = identity.theta
        
        # Department alignment affects visibility
        dept_alignment = 1.0
        if self.department and identity.department:
            angle_diff = abs(self.department.phase - identity.department.phase)
            if angle_diff > PI:
                angle_diff = 2 * PI - angle_diff
            dept_alignment = math.cos(angle_diff)  # 1.0 if same dept, 0 if perpendicular
        
        # Calculate visibility components
        public_visibility = abs(math.cos(theta))  # How much public they see
        private_visibility = abs(math.sin(theta)) * dept_alignment  # How much private
        
        # Check if they meet the threshold for private data
        can_see_private = theta >= self.required_phase
        
        result = {
            'name': self.name,
            'identity': str(identity),
            'public_visible': True,
            'public_value': self.public_value,
            'public_strength': public_visibility,
            'private_visible': can_see_private and private_visibility > 0.3,
            'private_value': self.private_value if can_see_private else "[HIDDEN]",
            'private_strength': private_visibility if can_see_private else 0,
            'department_alignment': dept_alignment
        }
        
        return result
    
    def blend_value(self, identity: IdentityPhase) -> Any:
        """
        Get a blended view based on phase.
        For numeric data, returns weighted average.
        """
        theta = identity.theta
        
        if isinstance(self.public_value, (int, float)) and isinstance(self.private_value, (int, float)):
            # Numeric blending
            public_weight = abs(math.cos(theta))
            private_weight = abs(math.sin(theta)) if theta >= self.required_phase else 0
            
            total_weight = public_weight + private_weight
            if total_weight == 0:
                return self.public_value
            
            return (self.public_value * public_weight + 
                   self.private_value * private_weight) / total_weight
        else:
            # Non-numeric: return appropriate view
            if theta >= self.required_phase and theta > PI / 4:
                return self.private_value
            return self.public_value


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transaction:
    """
    A transaction with customer and seller views.
    Same data, different phases reveal different information.
    """
    transaction_id: str
    
    # Customer view (real component)
    customer_data: Dict[str, Any] = field(default_factory=dict)
    
    # Seller view (imaginary component)  
    seller_data: Dict[str, Any] = field(default_factory=dict)
    
    # Shared data (visible to both)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    
    def view_as(self, identity: IdentityPhase) -> Dict[str, Any]:
        """Get transaction view based on identity phase."""
        theta = identity.theta
        
        view = {
            'transaction_id': self.transaction_id,
            'viewer': str(identity),
            'view_type': 'customer' if theta < PI/4 else 'blended' if theta < 3*PI/4 else 'seller'
        }
        
        # Always show shared data
        view['shared'] = self.shared_data.copy()
        
        # Customer data visibility (high when θ near 0)
        customer_strength = abs(math.cos(theta))
        if customer_strength > 0.3:
            view['customer_visible'] = True
            view['customer_data'] = self.customer_data.copy()
            view['customer_strength'] = customer_strength
        else:
            view['customer_visible'] = False
            view['customer_data'] = {}
            view['customer_strength'] = 0
        
        # Seller data visibility (high when θ near π/2 or beyond)
        seller_strength = abs(math.sin(theta))
        if seller_strength > 0.3:
            view['seller_visible'] = True
            view['seller_data'] = self.seller_data.copy()
            view['seller_strength'] = seller_strength
        else:
            view['seller_visible'] = False
            view['seller_data'] = {}
            view['seller_strength'] = 0
        
        return view


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE-BASED ACCESS CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

class EulerAccessControl:
    """
    Complete access control system using Euler identity phases.
    """
    
    def __init__(self, org_size: str = 'medium'):
        self.h_window = ORG_SIZES.get(org_size, ORG_SIZES['medium'])
        self.identities: Dict[str, IdentityPhase] = {}
        self.data_items: Dict[str, ComplexData] = {}
        self.transactions: Dict[str, Transaction] = {}
    
    def create_identity(self, user_id: str, 
                       department: Optional[Department] = None,
                       is_staff: bool = False,
                       initial_theta: float = 0.0) -> IdentityPhase:
        """Create a new identity."""
        identity = IdentityPhase(
            user_id=user_id,
            theta=initial_theta,
            department=department,
            is_staff=is_staff
        )
        
        # Staff start at higher phase
        if is_staff:
            identity.theta = TRUST_THRESHOLD_STAFF
        
        self.identities[user_id] = identity
        return identity
    
    def get_identity(self, user_id: str) -> Optional[IdentityPhase]:
        """Retrieve an identity."""
        return self.identities.get(user_id)
    
    def verify_access(self, user_id: str, claimed_theta: float) -> Tuple[bool, str]:
        """
        Verify that a user's claimed phase is legitimate.
        Blocks phase-jumping attacks.
        """
        identity = self.get_identity(user_id)
        if not identity:
            return False, "Unknown user"
        
        return identity.verify_phase_continuity(claimed_theta)
    
    def register_data(self, name: str, public_value: Any, private_value: Any,
                     required_phase: float = PI/4,
                     department: Optional[Department] = None) -> ComplexData:
        """Register a data item with public/private components."""
        data = ComplexData(
            name=name,
            public_value=public_value,
            private_value=private_value,
            required_phase=required_phase,
            department=department
        )
        self.data_items[name] = data
        return data
    
    def create_transaction(self, txn_id: str,
                          customer_data: Dict[str, Any],
                          seller_data: Dict[str, Any],
                          shared_data: Dict[str, Any]) -> Transaction:
        """Create a transaction with customer/seller views."""
        txn = Transaction(
            transaction_id=txn_id,
            customer_data=customer_data,
            seller_data=seller_data,
            shared_data=shared_data
        )
        self.transactions[txn_id] = txn
        return txn
    
    def view_data(self, user_id: str, data_name: str) -> Dict[str, Any]:
        """Get a user's view of specific data."""
        identity = self.get_identity(user_id)
        if not identity:
            return {'error': 'Unknown user'}
        
        data = self.data_items.get(data_name)
        if not data:
            return {'error': 'Unknown data item'}
        
        return data.visible_to(identity)
    
    def view_transaction(self, user_id: str, txn_id: str) -> Dict[str, Any]:
        """Get a user's view of a transaction."""
        identity = self.get_identity(user_id)
        if not identity:
            return {'error': 'Unknown user'}
        
        txn = self.transactions.get(txn_id)
        if not txn:
            return {'error': 'Unknown transaction'}
        
        return txn.view_as(identity)
    
    def detect_outsider(self, user_id: str, action_requires: float) -> Tuple[bool, str]:
        """
        Detect if someone is trying to access beyond their phase.
        """
        identity = self.get_identity(user_id)
        if not identity:
            return True, "Unknown user - outsider"
        
        if identity.theta < action_requires:
            gap = action_requires - identity.theta
            return True, f"Insufficient phase: has {math.degrees(identity.theta):.1f}°, needs {math.degrees(action_requires):.1f}° (gap: {math.degrees(gap):.1f}°)"
        
        return False, "Access permitted"


# ═══════════════════════════════════════════════════════════════════════════════
# CAR SALE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def car_sale_demo():
    """Demonstrate customer vs seller views of a car sale."""
    
    print("\n" + "=" * 80)
    print("CAR SALE DEMONSTRATION: e^(iθ) Access Control")
    print("=" * 80)
    
    system = EulerAccessControl(org_size='medium')
    
    # Create identities
    customer = system.create_identity("customer_jane", is_staff=False)
    salesperson = system.create_identity(
        "sales_bob", 
        department=Department.SALES, 
        is_staff=True
    )
    finance_mgr = system.create_identity(
        "finance_mary",
        department=Department.FINANCE,
        is_staff=True
    )
    
    # Evolve customer trust a bit
    customer.evolve_trust(days=30, investment=500)  # 30 days, $500 spent
    
    print(f"\n--- IDENTITIES ---")
    print(f"Customer: {customer}")
    print(f"  e^(iθ) = {customer.identity:.3f}")
    print(f"  cos(θ) = {customer.cos_component:.3f} (customer view)")
    print(f"  sin(θ) = {customer.sin_component:.3f} (staff view)")
    
    print(f"\nSalesperson: {salesperson}")
    print(f"  e^(iθ) = {salesperson.identity:.3f}")
    print(f"  cos(θ) = {salesperson.cos_component:.3f} (customer view)")
    print(f"  sin(θ) = {salesperson.sin_component:.3f} (staff view)")
    
    # Create data items
    car_price = system.register_data(
        name="car_price",
        public_value=45000,      # Customer sees retail price
        private_value=32000,     # Staff sees dealer cost
        required_phase=PI/2,
        department=Department.SALES
    )
    
    credit_score = system.register_data(
        name="credit_score",
        public_value="Check your credit report",  # Customer sees generic
        private_value=750,                         # Staff sees actual score
        required_phase=PI/2,
        department=Department.FINANCE
    )
    
    commission = system.register_data(
        name="commission",
        public_value="N/A",       # Customer can't see
        private_value=2500,       # Staff sees commission
        required_phase=3*PI/4,    # High threshold
        department=Department.SALES
    )
    
    # Create the car sale transaction
    car_sale = system.create_transaction(
        txn_id="SALE-001",
        customer_data={
            "vehicle": "2024 Honda Accord",
            "price": 45000,
            "warranty": "3 years / 36,000 miles",
            "delivery": "2 weeks",
            "your_payments": "$650/month for 72 months"
        },
        seller_data={
            "dealer_cost": 32000,
            "holdback": 1500,
            "customer_credit_score": 750,
            "approval_status": "Auto-approved",
            "commission_rate": "5%",
            "commission_amount": 2500,
            "inventory_days": 45,
            "floor_plan_cost": 180
        },
        shared_data={
            "transaction_date": "2024-12-23",
            "vin": "1HGCV1F34MA123456"
        }
    )
    
    # Show different views
    print(f"\n{'='*80}")
    print("DATA ITEM VIEWS")
    print("="*80)
    
    print(f"\n--- CAR PRICE ---")
    print(f"Customer sees: {system.view_data('customer_jane', 'car_price')}")
    print(f"Salesperson sees: {system.view_data('sales_bob', 'car_price')}")
    print(f"Blended (salesperson): ${car_price.blend_value(salesperson):,.0f}")
    
    print(f"\n--- CREDIT SCORE ---")
    print(f"Customer sees: {system.view_data('customer_jane', 'credit_score')}")
    print(f"Finance sees: {system.view_data('finance_mary', 'credit_score')}")
    
    print(f"\n--- COMMISSION ---")
    print(f"Customer sees: {system.view_data('customer_jane', 'commission')}")
    print(f"Salesperson sees: {system.view_data('sales_bob', 'commission')}")
    
    print(f"\n{'='*80}")
    print("TRANSACTION VIEWS")
    print("="*80)
    
    print(f"\n--- CUSTOMER VIEW ---")
    customer_view = system.view_transaction('customer_jane', 'SALE-001')
    for key, value in customer_view.items():
        print(f"  {key}: {value}")
    
    print(f"\n--- SALESPERSON VIEW ---")
    seller_view = system.view_transaction('sales_bob', 'SALE-001')
    for key, value in seller_view.items():
        print(f"  {key}: {value}")
    
    # Outsider detection
    print(f"\n{'='*80}")
    print("OUTSIDER DETECTION")
    print("="*80)
    
    # Customer trying to see seller data
    is_outsider, reason = system.detect_outsider('customer_jane', PI/2)
    print(f"\nCustomer trying to access staff data (π/2):")
    print(f"  Blocked: {is_outsider}, Reason: {reason}")
    
    # Salesperson accessing their level
    is_outsider, reason = system.detect_outsider('sales_bob', PI/2)
    print(f"\nSalesperson accessing staff data (π/2):")
    print(f"  Blocked: {is_outsider}, Reason: {reason}")
    
    # Phase attack detection
    print(f"\n{'='*80}")
    print("PHASE ATTACK DETECTION")
    print("="*80)
    
    # Someone claiming high phase on new account
    attacker = system.create_identity("attacker_evil")
    valid, reason = attacker.verify_phase_continuity(PI)  # Claiming owner!
    print(f"\nNew account claiming owner phase (π):")
    print(f"  Valid: {valid}, Reason: {reason}")
    
    # Legitimate phase evolution
    valid, reason = customer.verify_phase_continuity(customer.theta + 0.1)
    print(f"\nCustomer claiming slightly higher phase:")
    print(f"  Valid: {valid}, Reason: {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT ISOLATION DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def department_demo():
    """Demonstrate department isolation via phase windows."""
    
    print("\n" + "=" * 80)
    print("DEPARTMENT ISOLATION: Phase Windows")
    print("=" * 80)
    
    system = EulerAccessControl(org_size='enterprise')
    
    # Create staff in different departments
    sales_staff = system.create_identity(
        "alice_sales",
        department=Department.SALES,
        is_staff=True
    )
    
    it_staff = system.create_identity(
        "bob_it",
        department=Department.TECHNOLOGY,
        is_staff=True
    )
    
    finance_staff = system.create_identity(
        "carol_finance",
        department=Department.FINANCE,
        is_staff=True
    )
    
    print("\n--- STAFF PHASES ---")
    for staff in [sales_staff, it_staff, finance_staff]:
        print(f"{staff.user_id}:")
        print(f"  Base θ: {math.degrees(staff.theta):.1f}°")
        print(f"  Dept phase: {math.degrees(staff.department.phase):.1f}°")
        print(f"  Total phase: {math.degrees(staff.total_phase):.1f}°")
    
    # Create department-specific data
    sales_data = system.register_data(
        "customer_list",
        public_value="Contact sales@company.com",
        private_value=["Customer A: $50k", "Customer B: $30k", "Customer C: $80k"],
        department=Department.SALES,
        required_phase=PI/2
    )
    
    it_data = system.register_data(
        "server_passwords",
        public_value="Contact IT support",
        private_value={"prod": "Pr0d$ecure!", "dev": "D3v@ccess"},
        department=Department.TECHNOLOGY,
        required_phase=3*PI/4
    )
    
    finance_data = system.register_data(
        "payroll",
        public_value="See HR for salary info",
        private_value={"alice": 85000, "bob": 95000, "carol": 105000},
        department=Department.FINANCE,
        required_phase=3*PI/4
    )
    
    print(f"\n{'='*80}")
    print("CROSS-DEPARTMENT ACCESS")
    print("="*80)
    
    # Sales trying to access IT data
    print("\n--- Sales staff viewing IT passwords ---")
    view = it_data.visible_to(sales_staff)
    print(f"  Department alignment: {view['department_alignment']:.3f}")
    print(f"  Private visible: {view['private_visible']}")
    print(f"  Private value: {view['private_value']}")
    
    # IT accessing IT data
    print("\n--- IT staff viewing IT passwords ---")
    view = it_data.visible_to(it_staff)
    print(f"  Department alignment: {view['department_alignment']:.3f}")
    print(f"  Private visible: {view['private_visible']}")
    print(f"  Private value: {view['private_value']}")
    
    # Finance accessing finance data
    print("\n--- Finance staff viewing payroll ---")
    view = finance_data.visible_to(finance_staff)
    print(f"  Department alignment: {view['department_alignment']:.3f}")
    print(f"  Private visible: {view['private_visible']}")
    print(f"  Private value: {view['private_value']}")
    
    # Sales accessing finance data (opposite departments!)
    print("\n--- Sales staff viewing payroll (opposite dept!) ---")
    view = finance_data.visible_to(sales_staff)
    print(f"  Department alignment: {view['department_alignment']:.3f}")
    print(f"  Note: Sales (0°) and Finance (180°) are OPPOSITE!")
    print(f"  Private visible: {view['private_visible']}")
    print(f"  Private value: {view['private_value']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests():
    """Run comprehensive tests."""
    
    print("\n" + "=" * 80)
    print("EULER ACCESS CONTROL - TEST SUITE")
    print("=" * 80)
    
    system = EulerAccessControl()
    
    # Test 1: Identity creation
    print("\n--- TEST 1: Identity Creation ---")
    new_user = system.create_identity("new_user_123")
    assert new_user.theta == 0, "New user should have θ=0"
    assert new_user.trust_level == "new_user"
    print(f"✓ New user created with θ=0, trust_level={new_user.trust_level}")
    
    owner = system.create_identity("owner_admin", is_staff=True)
    owner.theta = PI  # Set to owner
    assert owner.trust_level == "owner"
    print(f"✓ Owner created with θ=π, trust_level={owner.trust_level}")
    
    # Test 2: Euler identity math
    print("\n--- TEST 2: Euler Identity Math ---")
    # e^(i·0) = 1
    assert abs(new_user.identity - 1) < 0.001, "e^(i·0) should equal 1"
    print(f"✓ e^(i·0) = {new_user.identity:.3f} ≈ 1")
    
    # e^(i·π) = -1
    assert abs(owner.identity - (-1)) < 0.001, "e^(i·π) should equal -1"
    print(f"✓ e^(i·π) = {owner.identity:.3f} ≈ -1")
    
    # Owner + User = 0
    sum_identity = new_user.identity + owner.identity
    assert abs(sum_identity) < 0.001, "Owner + User should = 0"
    print(f"✓ Owner + User = {sum_identity:.3f} ≈ 0 (complementary!)")
    
    # Test 3: Trust evolution
    print("\n--- TEST 3: Trust Evolution ---")
    evolving_user = system.create_identity("evolving_user")
    initial_theta = evolving_user.theta
    evolving_user.evolve_trust(days=100, investment=1000, trust_delta=0.2)
    assert evolving_user.theta > initial_theta, "Trust should increase"
    print(f"✓ Trust evolved: {math.degrees(initial_theta):.1f}° → {math.degrees(evolving_user.theta):.1f}°")
    print(f"  Trust level: {evolving_user.trust_level}")
    
    # Test 4: Phase attack detection
    print("\n--- TEST 4: Phase Attack Detection ---")
    attacker = system.create_identity("attacker")
    valid, _ = attacker.verify_phase_continuity(PI)  # Claim owner
    assert not valid, "Should detect phase attack"
    print(f"✓ Phase attack detected: new account claiming θ=π")
    
    valid, _ = attacker.verify_phase_continuity(0.01)  # Small increment
    assert valid, "Small increment should be valid"
    print(f"✓ Small increment allowed: θ=0 → θ=0.01")
    
    # Test 5: Data visibility
    print("\n--- TEST 5: Data Visibility ---")
    data = system.register_data(
        "secret_data",
        public_value="PUBLIC",
        private_value="PRIVATE",
        required_phase=PI/2
    )
    
    customer = system.create_identity("customer")
    staff = system.create_identity("staff", is_staff=True)
    
    customer_view = data.visible_to(customer)
    assert not customer_view['private_visible'], "Customer shouldn't see private"
    print(f"✓ Customer cannot see private data")
    
    staff_view = data.visible_to(staff)
    assert staff_view['private_visible'], "Staff should see private"
    print(f"✓ Staff can see private data")
    
    # Test 6: Transaction views
    print("\n--- TEST 6: Transaction Views ---")
    txn = system.create_transaction(
        "TXN-001",
        customer_data={"price": 100},
        seller_data={"cost": 60, "margin": 40},
        shared_data={"date": "today"}
    )
    
    cust_view = txn.view_as(customer)
    assert cust_view['customer_visible'], "Customer should see customer data"
    assert not cust_view['seller_visible'], "Customer shouldn't see seller data"
    print(f"✓ Customer sees customer_data, not seller_data")
    
    staff_view = txn.view_as(staff)
    assert staff_view['seller_visible'], "Staff should see seller data"
    print(f"✓ Staff sees seller_data")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--car":
            car_sale_demo()
        elif sys.argv[1] == "--dept":
            department_demo()
        elif sys.argv[1] == "--all":
            run_tests()
            car_sale_demo()
            department_demo()
        else:
            run_tests()
    else:
        run_tests()
