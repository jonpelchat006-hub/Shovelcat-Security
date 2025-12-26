"""
DUAL HEXAGON MESH SYSTEM
========================
User Hexagon (capability) meshes with Money Hexagon (legitimacy).

Key Insights:
1. User teeth = verification completeness
2. Money teeth = legitimacy/provenance
3. They MESH at Y=0 - angles must be compatible!
4. Transaction = User RNA + Money RNA mesh
5. Friction when angles don't align

The Complete Transaction:
1. User has appropriate action tokens (User RNA)
2. Money has appropriate spending tokens (Money RNA)
3. User verification angle sufficient
4. Money legitimacy angle sufficient
5. Angles are COMPATIBLE (mesh without friction)

Author: Jonathan Pelchat
Based on Shovelcat Theory + collaborative refinement
"""

import math
import time
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from collections import deque

# Import from other modules (or redefine for standalone)
PI = math.pi
PHI = (1 + math.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# USER RNA TYPES (From capability side)
# ═══════════════════════════════════════════════════════════════════════════════

class UserRNAType(Enum):
    """User RNA tokens - what user CAN do."""
    
    READ = (1, 0, "Read data")
    LIST = (1, 10, "List/enumerate")
    WRITE = (2, 30, "Create/modify")
    DELETE = (3, 50, "Remove data")
    EXECUTE = (3, 40, "Run code")
    TRANSFER = (2, 35, "Transfer assets")
    ADMIN = (5, 70, "Administrative")
    ESCALATE = (8, 80, "Request upgrade")
    
    def __init__(self, cost: int, min_verification: int, description: str):
        self.cost = cost
        self.min_verification = min_verification
        self.description = description


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY RNA TYPES (From legitimacy side)
# ═══════════════════════════════════════════════════════════════════════════════

class MoneyRNAType(Enum):
    """Money RNA tokens - what money CAN be used for."""
    
    SPEND_SMALL = (1, 0, "Small purchases")
    SPEND_MEDIUM = (2, 30, "Medium purchases")
    SPEND_LARGE = (3, 50, "Large purchases")
    TRANSFER_INTERNAL = (1, 20, "Internal transfer")
    TRANSFER_EXTERNAL = (2, 40, "External transfer")
    TRANSFER_INTERNATIONAL = (3, 60, "International transfer")
    INVEST = (3, 50, "Investment")
    WITHDRAW = (2, 30, "Withdrawal")
    COLLATERAL = (4, 70, "Use as collateral")
    
    def __init__(self, cost: int, min_legitimacy: int, description: str):
        self.cost = cost
        self.min_legitimacy = min_legitimacy
        self.description = description


# ═══════════════════════════════════════════════════════════════════════════════
# USER HEXAGON
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserTeeth:
    """User teeth - verification angle."""
    angle: float  # 0-90 degrees
    
    @classmethod
    def from_verification(cls, 
                         password: bool,
                         email: bool,
                         phone: bool,
                         id_verified: bool,
                         history_good: bool) -> 'UserTeeth':
        """Compute teeth angle from verification factors."""
        score = 0
        if password:
            score += 20
        if email:
            score += 15
        if phone:
            score += 20
        if id_verified:
            score += 25
        if history_good:
            score += 20
        
        return cls(angle=min(90, score))


@dataclass
class UserRNA:
    """User RNA - action tokens."""
    
    user_id: str
    verification_angle: float
    tokens: Dict[UserRNAType, int] = field(default_factory=dict)
    
    def mint_tokens(self):
        """Mint tokens based on verification."""
        self.tokens = {}
        for rna_type in UserRNAType:
            if self.verification_angle >= rna_type.min_verification:
                multiplier = (self.verification_angle - rna_type.min_verification) / 30 + 1
                self.tokens[rna_type] = int(10 * multiplier)
    
    def can_use(self, rna_type: UserRNAType) -> bool:
        return self.tokens.get(rna_type, 0) > 0
    
    def use_token(self, rna_type: UserRNAType) -> bool:
        if self.tokens.get(rna_type, 0) > 0:
            self.tokens[rna_type] -= 1
            return True
        return False


class UserHexagon:
    """Complete user hexagon with DNA, RNA, and teeth."""
    
    def __init__(self, user_id: str, verification_angle: float):
        self.user_id = user_id
        self.teeth = UserTeeth(angle=verification_angle)
        self.rna = UserRNA(user_id, verification_angle)
        self.rna.mint_tokens()
    
    def get_angle(self) -> float:
        return self.teeth.angle


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY HEXAGON
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoneyTeeth:
    """Money teeth - legitimacy angle."""
    angle: float  # 0-90 degrees
    flagged: bool = False


@dataclass
class MoneyRNA:
    """Money RNA - spending tokens."""
    
    money_id: str
    legitimacy_angle: float
    tokens: Dict[MoneyRNAType, int] = field(default_factory=dict)
    
    def mint_tokens(self):
        """Mint tokens based on legitimacy."""
        self.tokens = {}
        for rna_type in MoneyRNAType:
            if self.legitimacy_angle >= rna_type.min_legitimacy:
                multiplier = (self.legitimacy_angle - rna_type.min_legitimacy) / 30 + 1
                self.tokens[rna_type] = int(10 * multiplier)
    
    def can_use(self, rna_type: MoneyRNAType) -> bool:
        return self.tokens.get(rna_type, 0) > 0
    
    def use_token(self, rna_type: MoneyRNAType) -> bool:
        if self.tokens.get(rna_type, 0) > 0:
            self.tokens[rna_type] -= 1
            return True
        return False


class MoneyHexagon:
    """Complete money hexagon with DNA, RNA, and teeth."""
    
    def __init__(self, money_id: str, legitimacy_angle: float, flagged: bool = False):
        self.money_id = money_id
        self.teeth = MoneyTeeth(angle=legitimacy_angle, flagged=flagged)
        self.rna = MoneyRNA(money_id, legitimacy_angle)
        self.rna.mint_tokens()
    
    def get_angle(self) -> float:
        return self.teeth.angle


# ═══════════════════════════════════════════════════════════════════════════════
# MESH COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class MeshResult(Enum):
    """Result of mesh attempt."""
    
    PERFECT = ("PERFECT", 0.0, "Angles align perfectly")
    GOOD = ("GOOD", 0.2, "Minor friction")
    ACCEPTABLE = ("ACCEPTABLE", 0.4, "Moderate friction")
    POOR = ("POOR", 0.6, "High friction")
    INCOMPATIBLE = ("INCOMPATIBLE", 1.0, "Cannot mesh")
    
    def __init__(self, name: str, friction: float, description: str):
        self._name = name
        self.friction = friction
        self.description = description


@dataclass
class MeshCheck:
    """Result of checking if hexagons can mesh."""
    
    can_mesh: bool
    result: MeshResult
    friction: float
    
    user_angle: float
    money_angle: float
    angle_difference: float
    
    user_has_tokens: bool
    money_has_tokens: bool
    
    issues: List[str] = field(default_factory=list)


class HexagonMesher:
    """
    Meshes User and Money hexagons together.
    
    The mesh happens at Y=0.
    Both angles must be compatible.
    Both must have appropriate tokens.
    """
    
    @staticmethod
    def check_mesh(user: UserHexagon,
                   money: MoneyHexagon,
                   user_action: UserRNAType,
                   money_action: MoneyRNAType) -> MeshCheck:
        """
        Check if user and money can mesh for a transaction.
        """
        issues = []
        
        # Get angles
        user_angle = user.get_angle()
        money_angle = money.get_angle()
        angle_diff = abs(user_angle - money_angle)
        
        # Check user tokens
        user_has_tokens = user.rna.can_use(user_action)
        if not user_has_tokens:
            issues.append(f"User lacks {user_action.name} tokens")
        
        # Check money tokens
        money_has_tokens = money.rna.can_use(money_action)
        if not money_has_tokens:
            issues.append(f"Money lacks {money_action.name} tokens")
        
        # Check user verification for action
        if user_angle < user_action.min_verification:
            issues.append(f"User verification {user_angle:.1f}° < required {user_action.min_verification}°")
        
        # Check money legitimacy for action
        if money_angle < money_action.min_legitimacy:
            issues.append(f"Money legitimacy {money_angle:.1f}° < required {money_action.min_legitimacy}°")
        
        # Check if money is flagged
        if money.teeth.flagged:
            issues.append("Money is flagged for review")
        
        # Determine mesh result
        if len(issues) > 0:
            result = MeshResult.INCOMPATIBLE
            can_mesh = False
            friction = 1.0
        elif angle_diff < 10:
            result = MeshResult.PERFECT
            can_mesh = True
            friction = angle_diff / 90
        elif angle_diff < 20:
            result = MeshResult.GOOD
            can_mesh = True
            friction = angle_diff / 90
        elif angle_diff < 35:
            result = MeshResult.ACCEPTABLE
            can_mesh = True
            friction = angle_diff / 90
        elif angle_diff < 50:
            result = MeshResult.POOR
            can_mesh = True
            friction = angle_diff / 90
        else:
            result = MeshResult.INCOMPATIBLE
            can_mesh = False
            friction = 1.0
            issues.append(f"Angle difference {angle_diff:.1f}° too large")
        
        return MeshCheck(
            can_mesh=can_mesh,
            result=result,
            friction=friction,
            user_angle=user_angle,
            money_angle=money_angle,
            angle_difference=angle_diff,
            user_has_tokens=user_has_tokens,
            money_has_tokens=money_has_tokens,
            issues=issues
        )
    
    @staticmethod
    def execute_mesh(user: UserHexagon,
                    money: MoneyHexagon,
                    user_action: UserRNAType,
                    money_action: MoneyRNAType) -> Tuple[bool, MeshCheck]:
        """
        Execute a mesh (transaction).
        
        Returns (success, mesh_check)
        """
        check = HexagonMesher.check_mesh(user, money, user_action, money_action)
        
        if check.can_mesh:
            # Consume tokens
            user.rna.use_token(user_action)
            money.rna.use_token(money_action)
            return True, check
        
        return False, check


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionType:
    """Defines a type of transaction with required actions."""
    
    name: str
    description: str
    user_action: UserRNAType
    money_action: MoneyRNAType
    min_user_angle: float
    min_money_angle: float


# Common transaction types
TRANSACTIONS = {
    'small_purchase': TransactionType(
        name="Small Purchase",
        description="Buy something small",
        user_action=UserRNAType.READ,
        money_action=MoneyRNAType.SPEND_SMALL,
        min_user_angle=0,
        min_money_angle=0
    ),
    'large_purchase': TransactionType(
        name="Large Purchase", 
        description="Buy something expensive",
        user_action=UserRNAType.TRANSFER,
        money_action=MoneyRNAType.SPEND_LARGE,
        min_user_angle=35,
        min_money_angle=50
    ),
    'external_transfer': TransactionType(
        name="External Transfer",
        description="Send money externally",
        user_action=UserRNAType.TRANSFER,
        money_action=MoneyRNAType.TRANSFER_EXTERNAL,
        min_user_angle=35,
        min_money_angle=40
    ),
    'investment': TransactionType(
        name="Investment",
        description="Invest money",
        user_action=UserRNAType.EXECUTE,
        money_action=MoneyRNAType.INVEST,
        min_user_angle=40,
        min_money_angle=50
    ),
    'admin_operation': TransactionType(
        name="Admin Operation",
        description="Administrative action with funds",
        user_action=UserRNAType.ADMIN,
        money_action=MoneyRNAType.COLLATERAL,
        min_user_angle=70,
        min_money_angle=70
    )
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionResult:
    """Result of a transaction attempt."""
    
    transaction_id: str
    transaction_type: str
    timestamp: float
    
    success: bool
    mesh_check: MeshCheck
    
    user_id: str
    money_id: str
    
    friction_cost: float  # Extra cost due to friction
    
    message: str


class TransactionProcessor:
    """
    Processes transactions by meshing User and Money hexagons.
    """
    
    def __init__(self):
        self.history: List[TransactionResult] = []
        self.total_friction: float = 0.0
    
    def process(self,
               user: UserHexagon,
               money: MoneyHexagon,
               transaction_type: str,
               amount: float = 0.0) -> TransactionResult:
        """Process a transaction."""
        
        if transaction_type not in TRANSACTIONS:
            return TransactionResult(
                transaction_id=secrets.token_hex(8),
                transaction_type=transaction_type,
                timestamp=time.time(),
                success=False,
                mesh_check=None,
                user_id=user.user_id,
                money_id=money.money_id,
                friction_cost=0,
                message=f"Unknown transaction type: {transaction_type}"
            )
        
        tx_type = TRANSACTIONS[transaction_type]
        
        # Execute mesh
        success, mesh = HexagonMesher.execute_mesh(
            user, money,
            tx_type.user_action,
            tx_type.money_action
        )
        
        # Calculate friction cost
        friction_cost = mesh.friction * amount * 0.1  # 10% max friction cost
        
        if success:
            self.total_friction += friction_cost
            message = f"Transaction successful with {mesh.result.name} mesh"
        else:
            message = f"Transaction failed: {', '.join(mesh.issues)}"
        
        result = TransactionResult(
            transaction_id=secrets.token_hex(8),
            transaction_type=transaction_type,
            timestamp=time.time(),
            success=success,
            mesh_check=mesh,
            user_id=user.user_id,
            money_id=money.money_id,
            friction_cost=friction_cost,
            message=message
        )
        
        self.history.append(result)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHY LEVELS (Y-axis exposure)
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchyLevel(Enum):
    """Levels in the Y-axis hierarchy."""
    
    OWNER = ("OWNER", -100, 100, "Full Y exposure, handles antimatter")
    MANAGER = ("MANAGER", -50, 80, "Significant Y exposure")
    EMPLOYEE = ("EMPLOYEE", -20, 60, "Limited Y exposure")
    USER = ("USER", -5, 40, "Minimal Y exposure")
    GUEST = ("GUEST", 0, 20, "No negative Y exposure")
    
    def __init__(self, name: str, min_y: int, max_y: int, description: str):
        self._name = name
        self.min_y = min_y
        self.max_y = max_y
        self.description = description
    
    @property
    def y_range(self) -> int:
        return self.max_y - self.min_y
    
    @property
    def antimatter_exposure(self) -> int:
        return abs(self.min_y)


@dataclass
class HierarchyEntity:
    """An entity in the hierarchy with Y-axis exposure."""
    
    entity_id: str
    level: HierarchyLevel
    
    # User hexagon (capability)
    user_hex: UserHexagon
    
    # Money being handled
    money_pool: List[MoneyHexagon] = field(default_factory=list)
    
    # Antimatter collected (from below)
    antimatter_collected: float = 0.0
    
    # Antimatter distributed (to above)
    antimatter_distributed: float = 0.0
    
    def can_handle_antimatter(self, amount: float) -> bool:
        """Check if entity can handle this much antimatter."""
        return amount <= self.level.antimatter_exposure
    
    def collect_antimatter(self, amount: float) -> bool:
        """Collect antimatter from below."""
        if self.can_handle_antimatter(self.antimatter_collected + amount):
            self.antimatter_collected += amount
            return True
        return False
    
    def distribute_antimatter(self, amount: float) -> float:
        """Distribute antimatter upward. Returns amount distributed."""
        actual = min(amount, self.antimatter_collected)
        self.antimatter_collected -= actual
        self.antimatter_distributed += actual
        return actual


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE DUAL HEXAGON SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class DualHexagonSystem:
    """
    Complete system with User and Money hexagons.
    
    Manages:
    - User hexagons (capability/verification)
    - Money hexagons (legitimacy/provenance)
    - Meshing transactions
    - Hierarchy levels
    - Antimatter flow
    """
    
    def __init__(self):
        self.users: Dict[str, UserHexagon] = {}
        self.money_units: Dict[str, MoneyHexagon] = {}
        self.hierarchy: Dict[str, HierarchyEntity] = {}
        
        self.processor = TransactionProcessor()
    
    def register_user(self, 
                     user_id: str,
                     verification_angle: float,
                     level: HierarchyLevel = HierarchyLevel.USER) -> UserHexagon:
        """Register a user."""
        user_hex = UserHexagon(user_id, verification_angle)
        self.users[user_id] = user_hex
        
        entity = HierarchyEntity(
            entity_id=user_id,
            level=level,
            user_hex=user_hex
        )
        self.hierarchy[user_id] = entity
        
        return user_hex
    
    def create_money(self,
                    money_id: str,
                    legitimacy_angle: float,
                    flagged: bool = False) -> MoneyHexagon:
        """Create money unit."""
        money_hex = MoneyHexagon(money_id, legitimacy_angle, flagged)
        self.money_units[money_id] = money_hex
        return money_hex
    
    def assign_money(self, user_id: str, money_id: str):
        """Assign money to user's pool."""
        if user_id in self.hierarchy and money_id in self.money_units:
            self.hierarchy[user_id].money_pool.append(self.money_units[money_id])
    
    def process_transaction(self,
                           user_id: str,
                           money_id: str,
                           transaction_type: str,
                           amount: float = 0.0) -> TransactionResult:
        """Process a transaction."""
        if user_id not in self.users:
            return TransactionResult(
                transaction_id="",
                transaction_type=transaction_type,
                timestamp=time.time(),
                success=False,
                mesh_check=None,
                user_id=user_id,
                money_id=money_id,
                friction_cost=0,
                message=f"User {user_id} not found"
            )
        
        if money_id not in self.money_units:
            return TransactionResult(
                transaction_id="",
                transaction_type=transaction_type,
                timestamp=time.time(),
                success=False,
                mesh_check=None,
                user_id=user_id,
                money_id=money_id,
                friction_cost=0,
                message=f"Money {money_id} not found"
            )
        
        return self.processor.process(
            self.users[user_id],
            self.money_units[money_id],
            transaction_type,
            amount
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get system state."""
        return {
            'users': len(self.users),
            'money_units': len(self.money_units),
            'transactions': len(self.processor.history),
            'total_friction': self.processor.total_friction,
            'hierarchy': {
                uid: {
                    'level': e.level.name,
                    'verification_angle': e.user_hex.get_angle(),
                    'money_pool_size': len(e.money_pool),
                    'antimatter_collected': e.antimatter_collected
                }
                for uid, e in self.hierarchy.items()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_mesh_basics():
    """Demonstrate basic hexagon meshing."""
    print("\n" + "=" * 80)
    print("HEXAGON MESHING BASICS")
    print("=" * 80)
    
    print("""
    User Hexagon (TOP):           Money Hexagon (BOTTOM):
    ──────────────────            ────────────────────────
    Teeth = verification          Teeth = legitimacy
    RNA = action tokens           RNA = spending tokens
    
    They MESH at Y=0
    Angles must be compatible!
    """)
    
    # Create hexagons with matching angles
    user = UserHexagon("alice", 75.0)
    money = MoneyHexagon("money_1", 70.0)
    
    print(f"\n--- Compatible Angles ---")
    print(f"User angle: {user.get_angle():.1f}°")
    print(f"Money angle: {money.get_angle():.1f}°")
    
    check = HexagonMesher.check_mesh(
        user, money,
        UserRNAType.TRANSFER,
        MoneyRNAType.TRANSFER_EXTERNAL
    )
    
    print(f"Mesh result: {check.result.name}")
    print(f"Friction: {check.friction:.3f}")
    print(f"Can mesh: {check.can_mesh}")
    
    # Incompatible angles
    user2 = UserHexagon("bob", 30.0)
    money2 = MoneyHexagon("money_2", 85.0)
    
    print(f"\n--- Incompatible Angles ---")
    print(f"User angle: {user2.get_angle():.1f}°")
    print(f"Money angle: {money2.get_angle():.1f}°")
    
    check2 = HexagonMesher.check_mesh(
        user2, money2,
        UserRNAType.ADMIN,
        MoneyRNAType.COLLATERAL
    )
    
    print(f"Mesh result: {check2.result.name}")
    print(f"Can mesh: {check2.can_mesh}")
    print(f"Issues: {check2.issues}")


def demo_transaction_types():
    """Demonstrate different transaction types."""
    print("\n" + "=" * 80)
    print("TRANSACTION TYPES")
    print("=" * 80)
    
    print("\n┌────────────────────┬─────────────────────┬─────────────────────┬──────────────┬──────────────┐")
    print("│ Transaction        │ User Action         │ Money Action        │ Min User °   │ Min Money °  │")
    print("├────────────────────┼─────────────────────┼─────────────────────┼──────────────┼──────────────┤")
    
    for name, tx in TRANSACTIONS.items():
        print(f"│ {tx.name:18} │ {tx.user_action.name:19} │ {tx.money_action.name:19} │ {tx.min_user_angle:12.0f} │ {tx.min_money_angle:12.0f} │")
    
    print("└────────────────────┴─────────────────────┴─────────────────────┴──────────────┴──────────────┘")


def demo_friction():
    """Demonstrate mesh friction."""
    print("\n" + "=" * 80)
    print("MESH FRICTION")
    print("=" * 80)
    
    print("""
    Friction increases with angle difference:
    
    User >>>>>>>>  (verification)
    Money <<<<<<<<  (legitimacy)
    
    Aligned = low friction
    Misaligned = high friction
    """)
    
    print("\n--- Friction by Angle Difference ---\n")
    print("┌──────────────────┬──────────────────┬─────────────┬──────────────┐")
    print("│ User Angle       │ Money Angle      │ Difference  │ Friction     │")
    print("├──────────────────┼──────────────────┼─────────────┼──────────────┤")
    
    user_angle = 60.0
    for money_angle in [60, 55, 50, 40, 30, 20, 10]:
        user = UserHexagon("test", user_angle)
        money = MoneyHexagon("test", money_angle)
        
        check = HexagonMesher.check_mesh(
            user, money,
            UserRNAType.TRANSFER,
            MoneyRNAType.TRANSFER_EXTERNAL
        )
        
        diff = abs(user_angle - money_angle)
        print(f"│ {user_angle:16.1f} │ {money_angle:16.1f} │ {diff:11.1f} │ {check.friction:12.3f} │")
    
    print("└──────────────────┴──────────────────┴─────────────┴──────────────┘")


def demo_hierarchy():
    """Demonstrate hierarchy levels."""
    print("\n" + "=" * 80)
    print("HIERARCHY LEVELS (Y-axis Exposure)")
    print("=" * 80)
    
    print("""
    Different levels have different Y-axis exposure:
    
    +Y (capability)     ▲
                        │   OWNER reaches far up AND down
                        │   USER stays mostly in +Y
                        │   GUEST never goes negative
    ═══════════════════════════════════════ Y=0
                        │
                        │   ANTIMATTER zone
                        │   (only high levels reach here)
    -Y (debt/antimatter)▼
    """)
    
    print("\n┌────────────────┬──────────────┬──────────────┬────────────────┬────────────────────────┐")
    print("│ Level          │ Min Y        │ Max Y        │ Antimatter Exp │ Description            │")
    print("├────────────────┼──────────────┼──────────────┼────────────────┼────────────────────────┤")
    
    for level in HierarchyLevel:
        print(f"│ {level.name:14} │ {level.min_y:12} │ {level.max_y:12} │ {level.antimatter_exposure:14} │ {level.description:22} │")
    
    print("└────────────────┴──────────────┴──────────────┴────────────────┴────────────────────────┘")


def demo_complete_system():
    """Demonstrate complete dual hexagon system."""
    print("\n" + "=" * 80)
    print("COMPLETE DUAL HEXAGON SYSTEM")
    print("=" * 80)
    
    system = DualHexagonSystem()
    
    # Register users at different levels
    system.register_user("owner", 85.0, HierarchyLevel.OWNER)
    system.register_user("manager", 70.0, HierarchyLevel.MANAGER)
    system.register_user("employee", 50.0, HierarchyLevel.EMPLOYEE)
    system.register_user("guest", 20.0, HierarchyLevel.GUEST)
    
    # Create money units with different legitimacy
    system.create_money("clean_money", 80.0)
    system.create_money("medium_money", 50.0)
    system.create_money("suspicious_money", 25.0, flagged=True)
    
    print(f"\n--- System State ---")
    state = system.get_state()
    print(f"Users: {state['users']}")
    print(f"Money units: {state['money_units']}")
    
    print(f"\n--- Hierarchy ---")
    for uid, data in state['hierarchy'].items():
        print(f"  {uid}: {data['level']}, verification {data['verification_angle']:.0f}°")
    
    # Test transactions
    print(f"\n--- Processing Transactions ---\n")
    
    transactions = [
        ("owner", "clean_money", "admin_operation", 10000),
        ("manager", "clean_money", "large_purchase", 5000),
        ("employee", "medium_money", "small_purchase", 100),
        ("guest", "suspicious_money", "large_purchase", 1000),  # Should fail
    ]
    
    for user_id, money_id, tx_type, amount in transactions:
        result = system.process_transaction(user_id, money_id, tx_type, amount)
        status = "✓" if result.success else "✗"
        print(f"{status} {user_id} + {money_id} → {tx_type}: {result.message}")
        if result.success:
            print(f"    Friction cost: ${result.friction_cost:.2f}")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("DUAL HEXAGON MESH SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: User hexagon
    print("\n--- TEST 1: User Hexagon ---")
    user = UserHexagon("test", 75.0)
    assert user.get_angle() == 75.0
    print(f"✓ User angle = {user.get_angle()}")
    
    # Test 2: Money hexagon
    print("\n--- TEST 2: Money Hexagon ---")
    money = MoneyHexagon("test", 60.0)
    assert money.get_angle() == 60.0
    print(f"✓ Money angle = {money.get_angle()}")
    
    # Test 3: Mesh check
    print("\n--- TEST 3: Mesh Check ---")
    check = HexagonMesher.check_mesh(
        user, money,
        UserRNAType.TRANSFER,
        MoneyRNAType.TRANSFER_EXTERNAL
    )
    assert check.can_mesh
    print(f"✓ Mesh possible: {check.result.name}")
    
    # Test 4: Friction calculation
    print("\n--- TEST 4: Friction ---")
    assert check.friction < 1.0
    print(f"✓ Friction = {check.friction:.3f}")
    
    # Test 5: Transaction processing
    print("\n--- TEST 5: Transaction ---")
    system = DualHexagonSystem()
    system.register_user("alice", 70.0)
    system.create_money("m1", 65.0)
    result = system.process_transaction("alice", "m1", "small_purchase", 100)
    assert result.success
    print(f"✓ Transaction successful")
    
    # Test 6: Hierarchy
    print("\n--- TEST 6: Hierarchy ---")
    entity = HierarchyEntity("test", HierarchyLevel.OWNER, user)
    assert entity.can_handle_antimatter(50)
    print(f"✓ Owner can handle antimatter")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--mesh":
            demo_mesh_basics()
        elif sys.argv[1] == "--transactions":
            demo_transaction_types()
        elif sys.argv[1] == "--friction":
            demo_friction()
        elif sys.argv[1] == "--hierarchy":
            demo_hierarchy()
        elif sys.argv[1] == "--system":
            demo_complete_system()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_mesh_basics()
            demo_transaction_types()
            demo_friction()
            demo_hierarchy()
            demo_complete_system()
        else:
            run_tests()
    else:
        run_tests()
