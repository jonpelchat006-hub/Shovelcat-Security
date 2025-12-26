"""
Y-AXIS ANTIMATTER FLOW SYSTEM
=============================
Opposing flows create stationary layer. Double-sided brush spreads both ways.

Key Insights:
1. Y-axis: +Y = income/capability, -Y = debt/obligation
2. Opposing flows meet at Y=0 (stationary layer where owner operates)
3. Double-sided brush: Trust covers Profit, Product covers Debt
4. Wiggle room is CONSERVED (can't maximize both ends)
5. Flow vs cash: value is in the FLOW, not raw accumulation

The Fluid Dynamics:
- Income flows DOWN into system
- Debt flows UP from obligations
- Where they meet = DENSE, STATIONARY (owner's position)
- Moving outward = less dense, more flowy

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
E = math.e

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class FlowType(Enum):
    """Types of flow in the Y-axis system."""
    
    # Income flows (positive, downward into system)
    SALARY = ("SALARY", +1, 0.95, "Regular employment income")
    REVENUE = ("REVENUE", +1, 0.90, "Business revenue")
    INVESTMENT_RETURN = ("INVESTMENT_RETURN", +1, 0.80, "Investment gains")
    GIFT = ("GIFT", +1, 0.70, "Received gifts")
    WINDFALL = ("WINDFALL", +1, 0.60, "Unexpected gains")
    
    # Debt flows (negative, upward from obligations)
    RENT = ("RENT", -1, 0.95, "Rental obligation")
    PAYROLL = ("PAYROLL", -1, 0.95, "Employee payments")
    LOAN_PAYMENT = ("LOAN_PAYMENT", -1, 0.90, "Loan servicing")
    SUBSCRIPTION = ("SUBSCRIPTION", -1, 0.85, "Recurring subscriptions")
    TAX = ("TAX", -1, 0.95, "Tax obligations")
    MAINTENANCE = ("MAINTENANCE", -1, 0.80, "Upkeep costs")
    
    def __init__(self, name: str, direction: int, regularity: float, description: str):
        self._name = name
        self.direction = direction  # +1 = income, -1 = debt
        self.regularity = regularity  # How predictable (0-1)
        self.description = description
    
    @property
    def is_income(self) -> bool:
        return self.direction > 0
    
    @property
    def is_debt(self) -> bool:
        return self.direction < 0


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW EVENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlowEvent:
    """A single flow event (income or debt)."""
    
    flow_type: FlowType
    amount: float
    timestamp: float
    source: str = ""
    destination: str = ""
    
    @property
    def direction(self) -> int:
        return self.flow_type.direction
    
    @property
    def signed_amount(self) -> float:
        return self.amount * self.direction


# ═══════════════════════════════════════════════════════════════════════════════
# ACCUMULATED SQUARES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AccumulatedSquare:
    """
    Accumulated state (solid, dense).
    
    +Y: Savings square (raw cash, investments)
    -Y: Debt square (credit cards, loans)
    """
    
    name: str
    amount: float
    y_position: float  # +Y or -Y
    density: float = 1.0  # How solid (1.0 = fully accumulated)
    
    @property
    def is_savings(self) -> bool:
        return self.y_position > 0
    
    @property
    def is_debt(self) -> bool:
        return self.y_position < 0


# ═══════════════════════════════════════════════════════════════════════════════
# STATIONARY LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class StationaryLayer:
    """
    The layer at Y=0 where opposing flows meet.
    
    This is the DENSEST point - where the owner operates.
    Income flows down, debt flows up, they meet here.
    """
    
    def __init__(self, owner_id: str):
        self.owner_id = owner_id
        
        # Flows meeting at this layer
        self.income_flow_rate: float = 0.0
        self.debt_flow_rate: float = 0.0
        
        # Net flow (positive = forward, negative = backward)
        self.net_flow: float = 0.0
        
        # Density (how much is accumulated at Y=0)
        self.density: float = 1.0
        
        # History
        self.flow_history: deque = deque(maxlen=1000)
    
    def update_flows(self, income_rate: float, debt_rate: float):
        """Update the flow rates meeting at Y=0."""
        self.income_flow_rate = income_rate
        self.debt_flow_rate = debt_rate
        self.net_flow = income_rate - debt_rate
        
        # Density increases when flows are balanced
        balance = min(income_rate, debt_rate) / max(income_rate, debt_rate, 0.001)
        self.density = balance  # More balanced = more dense
        
        self.flow_history.append({
            'timestamp': time.time(),
            'income': income_rate,
            'debt': debt_rate,
            'net': self.net_flow,
            'density': self.density
        })
    
    def is_healthy(self) -> bool:
        """Is the layer in a healthy state?"""
        return self.net_flow >= 0
    
    def is_crisis(self) -> bool:
        """Is debt overwhelming income?"""
        return self.net_flow < 0 and abs(self.net_flow) > self.income_flow_rate * 0.5
    
    def get_status(self) -> str:
        """Get human-readable status."""
        if self.net_flow > self.income_flow_rate * 0.3:
            return "THRIVING (strong positive flow)"
        elif self.net_flow > 0:
            return "HEALTHY (positive flow)"
        elif self.net_flow > -self.income_flow_rate * 0.2:
            return "STRESSED (slight negative flow)"
        elif self.net_flow > -self.income_flow_rate * 0.5:
            return "STRUGGLING (significant negative flow)"
        else:
            return "CRISIS (debt overwhelming)"


# ═══════════════════════════════════════════════════════════════════════════════
# ROTATING BIT
# ═══════════════════════════════════════════════════════════════════════════════

class RotatingBit:
    """
    A bit rotating through the Y-plane.
    
    Center of rotation = initial investment / debt pivot
    How much it dips into -Y = debt exposure
    How much it rises into +Y = savings buffer
    """
    
    def __init__(self, 
                 center: float = 0.0,      # Y-position of rotation center
                 radius: float = 1.0,       # How far it reaches
                 initial_angle: float = 0.0):
        
        self.center = center
        self.radius = radius
        self.angle = initial_angle  # Current angle (0 = +Y, π = -Y)
        self.angular_velocity = 0.0
        
        self.history: deque = deque(maxlen=1000)
    
    def current_y(self) -> float:
        """Get current Y position."""
        return self.center + self.radius * math.cos(self.angle)
    
    def current_x(self) -> float:
        """Get current X position (for visualization)."""
        return self.radius * math.sin(self.angle)
    
    def max_y(self) -> float:
        """Maximum Y (savings peak)."""
        return self.center + self.radius
    
    def min_y(self) -> float:
        """Minimum Y (debt trough)."""
        return self.center - self.radius
    
    def debt_exposure(self) -> float:
        """How much of the rotation dips below Y=0."""
        if self.min_y() >= 0:
            return 0.0
        return abs(self.min_y())
    
    def savings_capacity(self) -> float:
        """How much of the rotation rises above Y=0."""
        if self.max_y() <= 0:
            return 0.0
        return self.max_y()
    
    def rotate(self, delta_angle: float):
        """Rotate the bit."""
        self.angle = (self.angle + delta_angle) % (2 * PI)
        
        self.history.append({
            'timestamp': time.time(),
            'angle': self.angle,
            'y': self.current_y(),
            'in_debt': self.current_y() < 0
        })
    
    def adjust_center(self, new_center: float):
        """Move the rotation center (change investment/debt pivot)."""
        self.center = new_center
    
    def adjust_radius(self, new_radius: float):
        """Change the radius (change exposure range)."""
        self.radius = new_radius
    
    def euler_position(self) -> complex:
        """Get position as complex number (Euler form)."""
        return self.center + self.radius * (math.cos(self.angle) + 1j * math.sin(self.angle))


# ═══════════════════════════════════════════════════════════════════════════════
# DOUBLE-SIDED BRUSH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoveringState:
    """State of one covering (trust or product)."""
    
    name: str
    amount: float
    required: float  # How much is needed to cover
    
    @property
    def coverage_ratio(self) -> float:
        if self.required < 0.001:
            return 1.0
        return min(1.0, self.amount / self.required)
    
    @property
    def exposed(self) -> float:
        """How much is uncovered."""
        return max(0, self.required - self.amount)
    
    @property
    def is_covered(self) -> bool:
        return self.amount >= self.required


class DoubleSidedBrush:
    """
    The owner's tool for spreading antimatter and collecting profit.
    
    TOP SIDE: Trust covers Profit
    - Take profit → need trust to cover it
    - Trust built by: past performance, transparency, delivering promises
    
    BOTTOM SIDE: Product covers Debt
    - Take debt → need product to cover it
    - Product built by: R&D, hiring, infrastructure, taking risks
    
    WIGGLE ROOM is CONSERVED:
    - Can't maximize both profit AND debt
    - Total stretch is limited
    """
    
    def __init__(self, 
                 total_wiggle_room: float = 100.0,
                 initial_trust: float = 50.0,
                 initial_product: float = 50.0):
        
        self.total_wiggle_room = total_wiggle_room
        
        # Current state
        self.profit_taken: float = 0.0
        self.debt_taken: float = 0.0
        
        # Coverings
        self.trust = CoveringState("TRUST", initial_trust, 0.0)
        self.product = CoveringState("PRODUCT", initial_product, 0.0)
        
        # Wiggle room used
        self.wiggle_used_profit: float = 0.0
        self.wiggle_used_debt: float = 0.0
        
        # History
        self.history: deque = deque(maxlen=1000)
    
    def take_profit(self, amount: float) -> Tuple[bool, str]:
        """
        Take profit. Requires trust to cover.
        
        Returns (success, message)
        """
        # Check wiggle room
        new_total = self.wiggle_used_profit + amount + self.wiggle_used_debt
        if new_total > self.total_wiggle_room:
            return False, f"WIGGLE_EXCEEDED: Would use {new_total:.2f} of {self.total_wiggle_room:.2f}"
        
        # Take the profit
        self.profit_taken += amount
        self.wiggle_used_profit += amount
        self.trust.required = self.profit_taken  # Need trust to cover all profit
        
        self._record_action("TAKE_PROFIT", amount)
        
        if not self.trust.is_covered:
            return True, f"WARNING: Profit exposed! Trust {self.trust.amount:.2f} < Profit {self.profit_taken:.2f}"
        
        return True, "Profit taken and covered by trust"
    
    def take_debt(self, amount: float) -> Tuple[bool, str]:
        """
        Take on debt. Requires product to cover.
        
        Returns (success, message)
        """
        # Check wiggle room
        new_total = self.wiggle_used_profit + self.wiggle_used_debt + amount
        if new_total > self.total_wiggle_room:
            return False, f"WIGGLE_EXCEEDED: Would use {new_total:.2f} of {self.total_wiggle_room:.2f}"
        
        # Take the debt
        self.debt_taken += amount
        self.wiggle_used_debt += amount
        self.product.required = self.debt_taken  # Need product to cover all debt
        
        self._record_action("TAKE_DEBT", amount)
        
        if not self.product.is_covered:
            return True, f"WARNING: Debt exposed! Product {self.product.amount:.2f} < Debt {self.debt_taken:.2f}"
        
        return True, "Debt taken and covered by product"
    
    def build_trust(self, amount: float):
        """Build trust (covers profit)."""
        self.trust.amount += amount
        self._record_action("BUILD_TRUST", amount)
    
    def build_product(self, amount: float):
        """Build product (covers debt)."""
        self.product.amount += amount
        self._record_action("BUILD_PRODUCT", amount)
    
    def repay_debt(self, amount: float):
        """Repay some debt."""
        actual = min(amount, self.debt_taken)
        self.debt_taken -= actual
        self.wiggle_used_debt -= actual
        self.product.required = self.debt_taken
        self._record_action("REPAY_DEBT", actual)
    
    def distribute_profit(self, amount: float):
        """Distribute profit (reduce taken profit)."""
        actual = min(amount, self.profit_taken)
        self.profit_taken -= actual
        self.wiggle_used_profit -= actual
        self.trust.required = self.profit_taken
        self._record_action("DISTRIBUTE_PROFIT", actual)
    
    def _record_action(self, action: str, amount: float):
        """Record action in history."""
        self.history.append({
            'timestamp': time.time(),
            'action': action,
            'amount': amount,
            'profit_taken': self.profit_taken,
            'debt_taken': self.debt_taken,
            'trust': self.trust.amount,
            'product': self.product.amount,
            'wiggle_remaining': self.wiggle_remaining
        })
    
    @property
    def wiggle_remaining(self) -> float:
        """Remaining wiggle room."""
        return self.total_wiggle_room - self.wiggle_used_profit - self.wiggle_used_debt
    
    @property
    def stretch_ratio(self) -> float:
        """How stretched are we? (0 = relaxed, 1 = max)"""
        return (self.wiggle_used_profit + self.wiggle_used_debt) / self.total_wiggle_room
    
    def is_balanced(self) -> bool:
        """Is profit/debt balanced?"""
        if self.wiggle_used_profit < 0.001 or self.wiggle_used_debt < 0.001:
            return True
        ratio = self.wiggle_used_profit / self.wiggle_used_debt
        return 0.3 < ratio < 3.0
    
    def is_breaking(self) -> bool:
        """Are we about to break?"""
        return self.stretch_ratio > 0.9
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete status."""
        return {
            'profit_taken': self.profit_taken,
            'debt_taken': self.debt_taken,
            'trust_covering': self.trust.amount,
            'trust_coverage_ratio': self.trust.coverage_ratio,
            'profit_exposed': self.trust.exposed,
            'product_covering': self.product.amount,
            'product_coverage_ratio': self.product.coverage_ratio,
            'debt_exposed': self.product.exposed,
            'wiggle_used_profit': self.wiggle_used_profit,
            'wiggle_used_debt': self.wiggle_used_debt,
            'wiggle_remaining': self.wiggle_remaining,
            'stretch_ratio': self.stretch_ratio,
            'is_balanced': self.is_balanced(),
            'is_breaking': self.is_breaking()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Y-AXIS FLOW SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class YAxisFlowSystem:
    """
    Complete Y-axis flow system.
    
    Integrates:
    - Opposing flows (income vs debt)
    - Stationary layer at Y=0
    - Rotating bit through Y-plane
    - Double-sided brush (trust/product covering)
    - Accumulated squares (savings/debt)
    """
    
    def __init__(self, 
                 owner_id: str,
                 initial_savings: float = 0.0,
                 initial_debt: float = 0.0,
                 total_wiggle: float = 100.0):
        
        self.owner_id = owner_id
        
        # Accumulated squares
        self.savings_square = AccumulatedSquare(
            name="SAVINGS",
            amount=initial_savings,
            y_position=1.0
        )
        
        self.debt_square = AccumulatedSquare(
            name="DEBT",
            amount=initial_debt,
            y_position=-1.0
        )
        
        # Stationary layer
        self.stationary_layer = StationaryLayer(owner_id)
        
        # Rotating bit (center based on net position)
        net_position = initial_savings - initial_debt
        self.rotating_bit = RotatingBit(
            center=net_position / 100,  # Normalized
            radius=1.0
        )
        
        # Double-sided brush
        self.brush = DoubleSidedBrush(total_wiggle_room=total_wiggle)
        
        # Flow tracking
        self.pending_income: List[FlowEvent] = []
        self.pending_debt: List[FlowEvent] = []
        
        # Daily rates (for stationary layer)
        self.daily_income_rate: float = 0.0
        self.daily_debt_rate: float = 0.0
    
    def add_income(self, flow_type: FlowType, amount: float, source: str = ""):
        """Add an income flow."""
        if not flow_type.is_income:
            raise ValueError(f"{flow_type.name} is not an income type")
        
        event = FlowEvent(
            flow_type=flow_type,
            amount=amount,
            timestamp=time.time(),
            source=source
        )
        
        self.pending_income.append(event)
        self.daily_income_rate += amount
        
        # Income can go to savings
        self.savings_square.amount += amount
        
        self._update_system()
    
    def add_debt_obligation(self, flow_type: FlowType, amount: float, destination: str = ""):
        """Add a debt obligation."""
        if not flow_type.is_debt:
            raise ValueError(f"{flow_type.name} is not a debt type")
        
        event = FlowEvent(
            flow_type=flow_type,
            amount=amount,
            timestamp=time.time(),
            destination=destination
        )
        
        self.pending_debt.append(event)
        self.daily_debt_rate += amount
        
        # Debt accumulates
        self.debt_square.amount += amount
        
        self._update_system()
    
    def pay_from_savings(self, amount: float) -> Tuple[bool, str]:
        """Pay debt from savings."""
        if amount > self.savings_square.amount:
            return False, f"Insufficient savings: {self.savings_square.amount:.2f} < {amount:.2f}"
        
        self.savings_square.amount -= amount
        self.debt_square.amount = max(0, self.debt_square.amount - amount)
        
        self._update_system()
        
        return True, f"Paid {amount:.2f} from savings"
    
    def _update_system(self):
        """Update all system components."""
        # Update stationary layer
        self.stationary_layer.update_flows(
            self.daily_income_rate,
            self.daily_debt_rate
        )
        
        # Update rotating bit center based on net position
        net = self.savings_square.amount - self.debt_square.amount
        self.rotating_bit.center = net / max(100, abs(net) + 1)
        
        # Rotate based on time
        self.rotating_bit.rotate(0.01)
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        return {
            'owner_id': self.owner_id,
            
            'savings': {
                'amount': self.savings_square.amount,
                'y_position': self.savings_square.y_position
            },
            
            'debt': {
                'amount': self.debt_square.amount,
                'y_position': self.debt_square.y_position
            },
            
            'net_position': self.savings_square.amount - self.debt_square.amount,
            
            'stationary_layer': {
                'income_rate': self.stationary_layer.income_flow_rate,
                'debt_rate': self.stationary_layer.debt_flow_rate,
                'net_flow': self.stationary_layer.net_flow,
                'density': self.stationary_layer.density,
                'status': self.stationary_layer.get_status()
            },
            
            'rotating_bit': {
                'center': self.rotating_bit.center,
                'radius': self.rotating_bit.radius,
                'current_y': self.rotating_bit.current_y(),
                'angle_degrees': math.degrees(self.rotating_bit.angle),
                'debt_exposure': self.rotating_bit.debt_exposure(),
                'savings_capacity': self.rotating_bit.savings_capacity()
            },
            
            'brush': self.brush.get_status()
        }
    
    def can_spend(self, amount: float) -> Tuple[bool, str]:
        """Check if spending is allowed."""
        # Must have savings
        if amount > self.savings_square.amount:
            return False, "Insufficient savings"
        
        # Must not cause crisis
        future_net = (self.savings_square.amount - amount) - self.debt_square.amount
        if future_net < -self.brush.total_wiggle_room:
            return False, "Would exceed wiggle room"
        
        return True, "Spending allowed"


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_opposing_flows():
    """Demonstrate opposing flows meeting at Y=0."""
    print("\n" + "=" * 80)
    print("OPPOSING FLOWS: Income vs Debt")
    print("=" * 80)
    
    print("""
    Income flows DOWN into system:  ▼ ▼ ▼
    Debt flows UP from obligations: ▲ ▲ ▲
    
    Where they meet (Y=0): STATIONARY LAYER
    
    ═══════════════════════════════════════════════════
             ▼ ▼ ▼ ▼ ▼  INCOME
    ─────────────────────────────────────────────────── Y=0
             ▲ ▲ ▲ ▲ ▲  DEBT
    ═══════════════════════════════════════════════════
    
    The more balanced the flows, the DENSER the layer.
    """)
    
    layer = StationaryLayer("owner_1")
    
    scenarios = [
        (100, 20, "High income, low debt"),
        (100, 50, "Balanced"),
        (100, 80, "High debt"),
        (100, 100, "Equal (maximum density)"),
        (50, 100, "Debt exceeds income (CRISIS)")
    ]
    
    print("\n┌────────────────────────────────┬──────────┬──────────┬──────────┬────────────────────────┐")
    print("│ Scenario                       │ Income   │ Debt     │ Net      │ Status                 │")
    print("├────────────────────────────────┼──────────┼──────────┼──────────┼────────────────────────┤")
    
    for income, debt, desc in scenarios:
        layer.update_flows(income, debt)
        status = layer.get_status()[:22]
        print(f"│ {desc:30} │ {income:8.1f} │ {debt:8.1f} │ {layer.net_flow:+8.1f} │ {status:22} │")
    
    print("└────────────────────────────────┴──────────┴──────────┴──────────┴────────────────────────┘")


def demo_rotating_bit():
    """Demonstrate rotating bit through Y-plane."""
    print("\n" + "=" * 80)
    print("ROTATING BIT: Through Y-Plane")
    print("=" * 80)
    
    print("""
    The bit rotates through +Y (savings) and -Y (debt):
    
              +Y (savings)
               │
           ────┼────  rotation
               │
              -Y (debt)
    
    Center of rotation = investment pivot
    Radius = how much exposure in each direction
    """)
    
    bit = RotatingBit(center=0.2, radius=1.0)
    
    print(f"\n--- Bit Configuration ---")
    print(f"Center: {bit.center:.2f}")
    print(f"Radius: {bit.radius:.2f}")
    print(f"Max Y (savings peak): {bit.max_y():.2f}")
    print(f"Min Y (debt trough): {bit.min_y():.2f}")
    print(f"Debt exposure: {bit.debt_exposure():.2f}")
    print(f"Savings capacity: {bit.savings_capacity():.2f}")
    
    print("\n--- Rotation Through Cycle ---\n")
    print("┌────────────┬────────────┬────────────┬─────────────┐")
    print("│ Angle      │ Y Position │ X Position │ Zone        │")
    print("├────────────┼────────────┼────────────┼─────────────┤")
    
    for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        bit.angle = math.radians(angle_deg)
        y = bit.current_y()
        x = bit.current_x()
        zone = "SAVINGS" if y > 0 else "DEBT" if y < 0 else "Y=0"
        print(f"│ {angle_deg:10}° │ {y:+10.3f} │ {x:+10.3f} │ {zone:11} │")
    
    print("└────────────┴────────────┴────────────┴─────────────┘")


def demo_double_sided_brush():
    """Demonstrate double-sided brush."""
    print("\n" + "=" * 80)
    print("DOUBLE-SIDED BRUSH: Trust/Product Covering")
    print("=" * 80)
    
    print("""
    TOP SIDE: Trust covers Profit
    BOTTOM SIDE: Product covers Debt
    
    WIGGLE ROOM IS CONSERVED!
    Can't maximize BOTH profit AND debt.
    
    ┌─────────────────────────────────────────────────────────────┐
    │ ░░░░░░ PROFIT ░░░░░░  ←──  TRUST covers                    │
    │ ████████ OWNER ████████                                    │
    │ ▓▓▓▓▓▓▓ DEBT ▓▓▓▓▓▓▓  ←──  PRODUCT covers                  │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    brush = DoubleSidedBrush(
        total_wiggle_room=100,
        initial_trust=30,
        initial_product=30
    )
    
    print(f"\n--- Initial State ---")
    print(f"Total wiggle room: {brush.total_wiggle_room}")
    print(f"Trust: {brush.trust.amount}, Product: {brush.product.amount}")
    
    # Take some profit
    print(f"\n--- Taking Profit ---")
    success, msg = brush.take_profit(20)
    print(f"Take profit 20: {msg}")
    print(f"Trust coverage: {brush.trust.coverage_ratio:.1%}")
    
    # Take some debt
    print(f"\n--- Taking Debt ---")
    success, msg = brush.take_debt(30)
    print(f"Take debt 30: {msg}")
    print(f"Product coverage: {brush.product.coverage_ratio:.1%}")
    
    # Status
    status = brush.get_status()
    print(f"\n--- Current Status ---")
    print(f"Profit taken: {status['profit_taken']:.2f}")
    print(f"Debt taken: {status['debt_taken']:.2f}")
    print(f"Wiggle remaining: {status['wiggle_remaining']:.2f}")
    print(f"Stretch ratio: {status['stretch_ratio']:.1%}")
    print(f"Is balanced: {status['is_balanced']}")
    
    # Try to overstretch
    print(f"\n--- Attempting Overstretch ---")
    success, msg = brush.take_profit(60)
    print(f"Take profit 60: {success} - {msg}")


def demo_wiggle_conservation():
    """Demonstrate wiggle room conservation."""
    print("\n" + "=" * 80)
    print("WIGGLE ROOM CONSERVATION")
    print("=" * 80)
    
    print("""
    Total wiggle = 100 units
    
    Can allocate between profit and debt:
    
    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │   profit_wiggle + debt_wiggle ≤ total_wiggle              │
    │                                                            │
    │   Can't have 80 profit AND 80 debt!                       │
    │   That would be 160 > 100 = BREAK!                        │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
    """)
    
    print("\n--- Allocation Scenarios ---\n")
    print("┌────────────────┬────────────────┬────────────────┬────────────────┬──────────┐")
    print("│ Profit         │ Debt           │ Total Used     │ Remaining      │ Status   │")
    print("├────────────────┼────────────────┼────────────────┼────────────────┼──────────┤")
    
    scenarios = [
        (20, 20),   # Conservative
        (40, 40),   # Balanced
        (60, 30),   # Profit-heavy
        (30, 60),   # Debt-heavy
        (50, 50),   # Stretched
        (80, 30),   # Near limit
    ]
    
    for profit, debt in scenarios:
        total = profit + debt
        remaining = 100 - total
        status = "OK" if total <= 100 else "BREAK!"
        if total > 90:
            status = "DANGER"
        print(f"│ {profit:14} │ {debt:14} │ {total:14} │ {remaining:14} │ {status:8} │")
    
    print("└────────────────┴────────────────┴────────────────┴────────────────┴──────────┘")


def demo_complete_system():
    """Demonstrate complete Y-axis flow system."""
    print("\n" + "=" * 80)
    print("COMPLETE Y-AXIS FLOW SYSTEM")
    print("=" * 80)
    
    system = YAxisFlowSystem(
        owner_id="test_owner",
        initial_savings=100,
        initial_debt=50,
        total_wiggle=100
    )
    
    print(f"\n--- Initial State ---")
    state = system.get_state()
    print(f"Savings: ${state['savings']['amount']:.2f}")
    print(f"Debt: ${state['debt']['amount']:.2f}")
    print(f"Net position: ${state['net_position']:.2f}")
    print(f"Stationary layer: {state['stationary_layer']['status']}")
    
    # Add income
    print(f"\n--- Adding Income ---")
    system.add_income(FlowType.SALARY, 300, "employer")
    state = system.get_state()
    print(f"After salary: Savings = ${state['savings']['amount']:.2f}")
    print(f"Income rate: ${state['stationary_layer']['income_rate']:.2f}/period")
    
    # Add debt obligations
    print(f"\n--- Adding Debt Obligations ---")
    system.add_debt_obligation(FlowType.RENT, 100, "landlord")
    system.add_debt_obligation(FlowType.SUBSCRIPTION, 50, "services")
    state = system.get_state()
    print(f"After obligations: Debt = ${state['debt']['amount']:.2f}")
    print(f"Debt rate: ${state['stationary_layer']['debt_rate']:.2f}/period")
    
    # Check stationary layer
    print(f"\n--- Stationary Layer Status ---")
    print(f"Net flow: ${state['stationary_layer']['net_flow']:.2f}/period")
    print(f"Density: {state['stationary_layer']['density']:.2f}")
    print(f"Status: {state['stationary_layer']['status']}")
    
    # Check rotating bit
    print(f"\n--- Rotating Bit ---")
    print(f"Center: {state['rotating_bit']['center']:.3f}")
    print(f"Current Y: {state['rotating_bit']['current_y']:.3f}")
    print(f"Debt exposure: {state['rotating_bit']['debt_exposure']:.3f}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("Y-AXIS ANTIMATTER FLOW SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Flow types
    print("\n--- TEST 1: Flow Types ---")
    assert FlowType.SALARY.is_income
    assert FlowType.RENT.is_debt
    print("✓ Flow types correctly categorized")
    
    # Test 2: Stationary layer
    print("\n--- TEST 2: Stationary Layer ---")
    layer = StationaryLayer("test")
    layer.update_flows(100, 50)
    assert layer.net_flow == 50
    assert layer.is_healthy()
    print(f"✓ Net flow = {layer.net_flow}")
    
    # Test 3: Rotating bit
    print("\n--- TEST 3: Rotating Bit ---")
    bit = RotatingBit(center=0, radius=1)
    assert abs(bit.current_y() - 1.0) < 0.01  # At angle 0, cos(0) = 1
    bit.rotate(PI)
    assert abs(bit.current_y() - (-1.0)) < 0.01  # At angle π, cos(π) = -1
    print("✓ Rotation works correctly")
    
    # Test 4: Double-sided brush
    print("\n--- TEST 4: Double-Sided Brush ---")
    brush = DoubleSidedBrush(total_wiggle_room=100)
    brush.take_profit(30)
    brush.take_debt(40)
    assert brush.wiggle_remaining == 30
    print(f"✓ Wiggle remaining = {brush.wiggle_remaining}")
    
    # Test 5: Wiggle conservation
    print("\n--- TEST 5: Wiggle Conservation ---")
    brush2 = DoubleSidedBrush(total_wiggle_room=100)
    brush2.take_profit(50)
    brush2.take_debt(40)
    success, _ = brush2.take_profit(20)  # Would exceed 100
    assert not success
    print("✓ Wiggle room conserved")
    
    # Test 6: Complete system
    print("\n--- TEST 6: Complete System ---")
    system = YAxisFlowSystem("test", initial_savings=100, initial_debt=50)
    system.add_income(FlowType.SALARY, 200)
    assert system.savings_square.amount == 300
    print(f"✓ System tracks flows correctly")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--flows":
            demo_opposing_flows()
        elif sys.argv[1] == "--bit":
            demo_rotating_bit()
        elif sys.argv[1] == "--brush":
            demo_double_sided_brush()
        elif sys.argv[1] == "--wiggle":
            demo_wiggle_conservation()
        elif sys.argv[1] == "--system":
            demo_complete_system()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_opposing_flows()
            demo_rotating_bit()
            demo_double_sided_brush()
            demo_wiggle_conservation()
            demo_complete_system()
        else:
            run_tests()
    else:
        run_tests()
