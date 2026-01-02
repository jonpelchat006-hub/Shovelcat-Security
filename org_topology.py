"""
COMPLETE ORGANIZATIONAL TOPOLOGY
================================

Mathematical boundaries define who can be where:

    ln(1) = 0  →  Position 1 is the BOUNDARY
                  Below 1: ln gives negatives (quantum/ψ domain)
                  Above 1: ln gives positives (classical/φ domain)
    
    φ ≈ 1.618  →  STAFF CEILING
                  The ψ domain only reaches radius φ
                  Staff occupy 1.0 to φ
    
    π ≈ 3.142  →  USER CEILING
                  Energy cap, excess converts to art/antimatter
                  Users occupy φ to π

ORGANIZATIONAL ZONES:

    Position 0-1: BUSINESS SIDE (internal, quantum/ψ domain)
        0.0:  Finance (antimatter interface)
        0.25: Profit-focused (Gaming, Products)
        0.5:  Workers, Devs, Operations
        0.75: Mixed (Research, Open Source)
        0.9:  Non-profit (Education, Health)
        1.0:  Owner/Mission center

    Position 1-φ: STAFF (customer-facing, classical/φ domain)
        1.0-1.2: Internal support
        1.2-1.4: Community managers
        1.4-φ:   PR, Social Media, Sales

    Position φ-π: USERS/CREATORS (external community)
        φ-2.0:   Trusted users, beta testers
        2.0-2.5: Content creators
        2.5-π:   Power users, influencers

    Position >π: EXCESS CONVERSION
        Art/content → antimatter → return larger
        OR attract new users who enter at top

THE COMPLETE CYCLE:

    1. External money → Finance (0)
    2. Finance → Money makers (0-1, by profit focus)
    3. Product created → Owner center (1)
    4. Product → Staff (1-φ) for distribution
    5. Staff → Users (φ-π) via marketing
    6. Users create content, reach π
    7. At π: Convert to art/antimatter
       - Good art: integrated (skins, features)
       - Viral content: attracts new users (top entry)
       - User grows: returns as larger entity
    8. New users attracted, enter at position π
    9. Work down through φ → 1 → become staff/business
    10. Cycle continues
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from enum import Enum

# Mathematical constants as boundaries
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618 - Staff ceiling
PI = np.pi                   # ≈ 3.142 - User ceiling
E = np.e                     # ≈ 2.718 - Growth threshold
LN_BOUNDARY = 1.0           # ln(1) = 0, the owner center


# =============================================================================
# ORGANIZATIONAL ZONES
# =============================================================================

class OrgZone(Enum):
    """Zones in the organizational topology."""
    FINANCE = "finance"           # Position 0
    PROFIT_FOCUS = "profit"       # Position 0-0.5
    OPERATIONS = "operations"     # Position 0.5
    MIXED = "mixed"               # Position 0.5-0.9
    NONPROFIT = "nonprofit"       # Position 0.9-1.0
    OWNER = "owner"               # Position 1.0
    INTERNAL_STAFF = "internal"   # Position 1.0-1.2
    COMMUNITY_STAFF = "community" # Position 1.2-1.4
    PUBLIC_STAFF = "public"       # Position 1.4-φ
    TRUSTED_USER = "trusted"      # Position φ-2.0
    CREATOR = "creator"           # Position 2.0-2.5
    POWER_USER = "power"          # Position 2.5-π
    CONVERTING = "converting"     # Position >π


def get_zone(position: float) -> OrgZone:
    """Get zone for a position."""
    if position <= 0:
        return OrgZone.FINANCE
    elif position < 0.5:
        return OrgZone.PROFIT_FOCUS
    elif position == 0.5:
        return OrgZone.OPERATIONS
    elif position < 0.9:
        return OrgZone.MIXED
    elif position < 1.0:
        return OrgZone.NONPROFIT
    elif position == 1.0:
        return OrgZone.OWNER
    elif position < 1.2:
        return OrgZone.INTERNAL_STAFF
    elif position < 1.4:
        return OrgZone.COMMUNITY_STAFF
    elif position < PHI:
        return OrgZone.PUBLIC_STAFF
    elif position < 2.0:
        return OrgZone.TRUSTED_USER
    elif position < 2.5:
        return OrgZone.CREATOR
    elif position < PI:
        return OrgZone.POWER_USER
    else:
        return OrgZone.CONVERTING


class ActorType(Enum):
    """Types of actors based on position."""
    BUSINESS = "business"   # 0-1: Internal, money-focused
    STAFF = "staff"         # 1-φ: Customer-facing employees
    USER = "user"           # φ-π: Community members
    EXTERNAL = "external"   # <0: Banks, markets


def get_actor_type(position: float) -> ActorType:
    """Get actor type for a position."""
    if position < 0:
        return ActorType.EXTERNAL
    elif position < 1.0:
        return ActorType.BUSINESS
    elif position < PHI:
        return ActorType.STAFF
    else:
        return ActorType.USER


# =============================================================================
# DOMAIN MATHEMATICS
# =============================================================================

class DomainMath:
    """
    The mathematical relationships that create boundaries.
    
    Classical (φ) domain: Can only go from 1 to φ
        - ln(1) = 0, below gives negatives
        - Population/customers live here
        
    Quantum (ψ) domain: Can go from 0 to 1
        - The internal business side
        - Radius reaches up to φ from center
    """
    
    @staticmethod
    def ln_position(x: float) -> float:
        """Natural log - creates boundary at 1."""
        if x <= 0:
            return float('-inf')
        return np.log(x)
    
    @staticmethod
    def exp_position(x: float) -> float:
        """Exponential - inverse of ln."""
        return np.exp(x)
    
    @staticmethod
    def classical_valid(position: float) -> bool:
        """Is this position valid in classical domain?"""
        # Classical domain: 1 to π (ln gives 0 to ~1.14)
        return 1.0 <= position <= PI
    
    @staticmethod
    def quantum_valid(position: float) -> bool:
        """Is this position valid in quantum domain?"""
        # Quantum domain: 0 to φ (internal + staff)
        return 0.0 <= position <= PHI
    
    @staticmethod
    def staff_valid(position: float) -> bool:
        """Is this position valid for staff?"""
        return 1.0 <= position <= PHI
    
    @staticmethod
    def user_valid(position: float) -> bool:
        """Is this position valid for users?"""
        return PHI <= position <= PI
    
    @staticmethod
    def excess_at_pi(energy: float, base: float) -> float:
        """Calculate excess when energy hits π threshold."""
        threshold = base * PI
        if energy > threshold:
            return energy - threshold
        return 0.0


# =============================================================================
# ORGANIZATIONAL ACTOR
# =============================================================================

@dataclass
class OrgActor:
    """
    An actor in the organizational topology.
    
    Position determines:
    - What zone they're in
    - What they can do
    - Who they can interact with
    """
    id: str
    name: str
    position: float
    
    # Role details
    title: str = ""
    profit_focus: float = 0.5  # 0 = non-profit, 1 = profit-focused
    
    # Energy/resources
    energy: float = 1.0
    base_energy: float = 1.0
    
    # Content/output
    content_created: List[Dict] = field(default_factory=list)
    products_made: List[Dict] = field(default_factory=list)
    
    # Connections
    supervisor_id: Optional[str] = None  # Who is below (toward 0)
    reports: Set[str] = field(default_factory=set)  # Who is above
    
    # For users: antimatter they bring
    antimatter_brought: float = 0.0
    
    @property
    def zone(self) -> OrgZone:
        """Current zone."""
        return get_zone(self.position)
    
    @property
    def actor_type(self) -> ActorType:
        """Actor type based on position."""
        return get_actor_type(self.position)
    
    @property
    def is_business(self) -> bool:
        """Is this a business-side actor?"""
        return 0 <= self.position < 1.0
    
    @property
    def is_staff(self) -> bool:
        """Is this a staff member?"""
        return 1.0 <= self.position < PHI
    
    @property
    def is_user(self) -> bool:
        """Is this a user/creator?"""
        return self.position >= PHI
    
    @property
    def is_owner(self) -> bool:
        """Is this the owner (position 1)?"""
        return self.position == 1.0
    
    @property
    def at_phi(self) -> bool:
        """At staff ceiling?"""
        return self.position >= PHI
    
    @property
    def at_pi(self) -> bool:
        """At user ceiling (must convert)?"""
        return self.energy >= self.base_energy * PI
    
    @property
    def excess_energy(self) -> float:
        """Energy above π threshold."""
        return DomainMath.excess_at_pi(self.energy, self.base_energy)
    
    def add_energy(self, amount: float) -> float:
        """Add energy, return excess if at π."""
        self.energy += amount
        excess = self.excess_energy
        if excess > 0:
            self.energy = self.base_energy * PI
        return excess
    
    def create_content(self, title: str, energy_cost: float) -> Optional[Dict]:
        """Create content (for users/creators)."""
        if energy_cost > self.energy:
            return None
        
        self.energy -= energy_cost
        
        content = {
            'id': f"content_{len(self.content_created)}_{self.id}",
            'title': title,
            'creator': self.id,
            'energy': energy_cost,
            'position_created': self.position,
            'created_at': datetime.now()
        }
        self.content_created.append(content)
        return content
    
    def create_product(self, name: str, energy_cost: float) -> Optional[Dict]:
        """Create product (for business side)."""
        if energy_cost > self.energy:
            return None
        
        self.energy -= energy_cost
        
        product = {
            'id': f"product_{len(self.products_made)}_{self.id}",
            'name': name,
            'creator': self.id,
            'energy': energy_cost,
            'profit_focus': self.profit_focus,
            'created_at': datetime.now()
        }
        self.products_made.append(product)
        return product


# =============================================================================
# ORGANIZATIONAL STRUCTURE
# =============================================================================

@dataclass
class SubDomain:
    """
    A subdomain within the organization.
    
    Examples:
    - Gaming (position 0.25, profit-focused)
    - Education (position 0.9, non-profit)
    - Health (position 0.9, non-profit)
    """
    id: str
    name: str
    position: float  # 0-1 based on profit focus
    
    # Staff
    owner_id: Optional[str] = None  # Position 1 for this domain
    finance_id: Optional[str] = None  # Position 0 for this domain
    actors: Dict[str, OrgActor] = field(default_factory=dict)
    
    # Resources
    capital: float = 0.0
    
    # Settings
    profit_focus: float = 0.5  # Derived from position
    
    def __post_init__(self):
        # Profit focus from position (0.25 → 0.75 profit, 0.9 → 0.1 profit)
        self.profit_focus = 1.0 - self.position
    
    @property
    def zone(self) -> OrgZone:
        """Zone for this subdomain."""
        return get_zone(self.position)
    
    def add_actor(self, actor: OrgActor):
        """Add an actor to this subdomain."""
        self.actors[actor.id] = actor


class ShovelcatOrg:
    """
    The complete Shovelcat organizational structure.
    
    Structure:
        - Main Finance (0) handles all external antimatter
        - Subdomains (0-1) based on profit focus
        - Owner center (1) 
        - Staff (1-φ) customer-facing
        - Users/Creators (φ-π) community
    """
    
    def __init__(self, name: str = "Shovelcat", initial_capital: float = 2000.0):
        self.name = name
        self.capital = initial_capital
        
        # Core actors
        self.main_finance: Optional[OrgActor] = None
        self.owner: Optional[OrgActor] = None
        
        # Subdomains
        self.subdomains: Dict[str, SubDomain] = {}
        
        # All actors
        self.actors: Dict[str, OrgActor] = {}
        
        # External entities
        self.external_entities: Dict[str, OrgActor] = {}
        
        # User community
        self.users: Dict[str, OrgActor] = {}
        
        # Content pool (art that's been created)
        self.content_pool: List[Dict] = []
        self.integrated_content: List[Dict] = []  # Art that became features
        
        # Initialize core
        self._init_core()
    
    def _init_core(self):
        """Initialize core actors."""
        # Main Finance (position 0)
        self.main_finance = OrgActor(
            id="main_finance",
            name=f"{self.name} Finance",
            position=0.0,
            title="Chief Financial Officer",
            profit_focus=0.0  # Finance is neutral
        )
        self.actors[self.main_finance.id] = self.main_finance
        
        # Owner (position 1)
        self.owner = OrgActor(
            id="owner",
            name=f"{self.name} Owner",
            position=1.0,
            title="Founder/CEO",
            profit_focus=0.5  # Balanced
        )
        self.actors[self.owner.id] = self.owner
        
        print(f"🏢 Initialized {self.name}")
        print(f"   🏦 Finance at position 0")
        print(f"   👑 Owner at position 1")
    
    # ─────────────────────────────────────────────────────────────────
    # SUBDOMAIN MANAGEMENT
    # ─────────────────────────────────────────────────────────────────
    
    def create_subdomain(self, name: str, position: float, 
                        initial_capital: float = 0.0) -> SubDomain:
        """
        Create a subdomain at a specific position.
        Position determines profit focus (0.25 = profit, 0.9 = nonprofit).
        """
        if not 0 < position < 1:
            raise ValueError("Subdomain position must be between 0 and 1")
        
        subdomain_id = f"subdomain_{name.lower().replace(' ', '_')}"
        
        subdomain = SubDomain(
            id=subdomain_id,
            name=name,
            position=position,
            capital=initial_capital
        )
        
        # Create subdomain owner (at position 1 relative to subdomain)
        sub_owner = OrgActor(
            id=f"{subdomain_id}_owner",
            name=f"{name} Lead",
            position=position + 0.05,  # Just above subdomain base
            title="Division Lead",
            profit_focus=subdomain.profit_focus
        )
        subdomain.owner_id = sub_owner.id
        subdomain.add_actor(sub_owner)
        self.actors[sub_owner.id] = sub_owner
        
        self.subdomains[subdomain_id] = subdomain
        
        zone = subdomain.zone
        profit_pct = subdomain.profit_focus * 100
        
        print(f"📂 Created subdomain: {name}")
        print(f"   Position: {position:.2f}")
        print(f"   Zone: {zone.value}")
        print(f"   Profit focus: {profit_pct:.0f}%")
        print(f"   Capital: ${initial_capital:,.2f}")
        
        return subdomain
    
    def add_worker(self, subdomain_id: str, name: str, 
                  position_offset: float = 0.0) -> Optional[OrgActor]:
        """Add a worker to a subdomain."""
        if subdomain_id not in self.subdomains:
            return None
        
        subdomain = self.subdomains[subdomain_id]
        base_pos = subdomain.position
        
        # Position within subdomain (0-1 range, scaled to subdomain)
        actual_pos = base_pos + position_offset * (1 - base_pos)
        actual_pos = min(actual_pos, 0.99)  # Can't reach owner center
        
        worker = OrgActor(
            id=f"worker_{len(self.actors)}_{name.lower().replace(' ', '_')}",
            name=name,
            position=actual_pos,
            title="Worker",
            profit_focus=subdomain.profit_focus
        )
        
        subdomain.add_actor(worker)
        self.actors[worker.id] = worker
        
        print(f"👷 Added worker to {subdomain.name}: {name} at pos {actual_pos:.2f}")
        
        return worker
    
    # ─────────────────────────────────────────────────────────────────
    # STAFF MANAGEMENT (1 to φ)
    # ─────────────────────────────────────────────────────────────────
    
    def hire_staff(self, name: str, position: float, title: str) -> Optional[OrgActor]:
        """
        Hire staff member (position 1 to φ).
        """
        if not 1.0 < position < PHI:
            print(f"⚠️  Staff position must be between 1.0 and φ ({PHI:.3f})")
            return None
        
        staff = OrgActor(
            id=f"staff_{len(self.actors)}_{name.lower().replace(' ', '_')}",
            name=name,
            position=position,
            title=title
        )
        
        self.actors[staff.id] = staff
        
        zone = staff.zone
        print(f"👔 Hired staff: {name}")
        print(f"   Position: {position:.2f} ({zone.value})")
        print(f"   Title: {title}")
        
        return staff
    
    # ─────────────────────────────────────────────────────────────────
    # USER MANAGEMENT (φ to π)
    # ─────────────────────────────────────────────────────────────────
    
    def register_user(self, name: str, entry_position: float = PI,
                     antimatter: float = 0.0) -> OrgActor:
        """
        Register a new user. They enter at top (π) and work down.
        """
        # Users start at π (top) and work down
        position = min(entry_position, PI)
        position = max(position, PHI)  # Can't go below staff level
        
        user = OrgActor(
            id=f"user_{len(self.users)}_{name.lower().replace(' ', '_')}",
            name=name,
            position=position,
            title="User",
            antimatter_brought=antimatter
        )
        
        self.users[user.id] = user
        self.actors[user.id] = user
        
        zone = user.zone
        print(f"🆕 New user: {name}")
        print(f"   Entry position: {position:.2f} ({zone.value})")
        if antimatter > 0:
            print(f"   Antimatter brought: {antimatter:.2f}")
        
        return user
    
    def promote_user(self, user_id: str, new_position: float) -> bool:
        """
        Promote user to lower position (toward staff).
        Lower position = more trust/access.
        """
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        if new_position < PHI:
            # Becoming staff!
            print(f"🎉 {user.name} promoted to STAFF!")
            del self.users[user_id]
            user.position = new_position
            return True
        
        if new_position >= user.position:
            print(f"⚠️  Can only promote to LOWER position")
            return False
        
        old_zone = user.zone
        user.position = new_position
        new_zone = user.zone
        
        print(f"⬇️  {user.name} promoted: {old_zone.value} → {new_zone.value}")
        
        return True
    
    def user_creates_content(self, user_id: str, title: str, 
                            energy: float) -> Optional[Dict]:
        """User creates content. May trigger π conversion."""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        
        # Add energy from creating
        excess = user.add_energy(energy)
        
        content = user.create_content(title, energy * 0.8)  # Cost 80% of energy
        if content:
            self.content_pool.append(content)
            print(f"🎨 {user.name} created: {title}")
            
            if excess > 0:
                print(f"   ⚡ At π threshold! Excess: {excess:.2f}")
                print(f"   Converting to antimatter (attracting new users)...")
                self._convert_excess(user, excess, content)
        
        return content
    
    def _convert_excess(self, user: OrgActor, excess: float, content: Dict):
        """Convert excess energy at π threshold."""
        # Excess becomes:
        # 1. Antimatter that attracts new users
        # 2. Content that might get integrated
        # 3. User grows larger (reputation)
        
        # Attract new user
        new_user_name = f"Attracted_by_{user.name}"
        new_user = self.register_user(new_user_name, PI, antimatter=excess * 0.5)
        
        print(f"   🆕 Attracted new user: {new_user_name}")
        
        # Check if content should be integrated
        if content['energy'] > 1.5:  # High quality threshold
            self.integrated_content.append(content)
            print(f"   ⭐ Content '{content['title']}' integrated into product!")
    
    # ─────────────────────────────────────────────────────────────────
    # THE FLOW
    # ─────────────────────────────────────────────────────────────────
    
    def run_cycle(self) -> Dict:
        """
        Run one organizational cycle.
        
        1. External money → Finance (0)
        2. Finance → Subdomains (0-1)
        3. Subdomains → Products → Owner (1)
        4. Owner → Staff (1-φ) for distribution
        5. Staff → Users (φ-π) via marketing
        6. Users create, hit π, attract more users
        """
        cycle_result = {
            'revenue': 0.0,
            'products_created': [],
            'content_created': [],
            'users_attracted': [],
            'promotions': []
        }
        
        print(f"\n{'═'*60}")
        print("ORGANIZATIONAL CYCLE")
        print(f"{'═'*60}")
        
        # Step 1: Revenue to finance
        revenue = sum(s.capital * 0.1 for s in self.subdomains.values())
        self.main_finance.energy += revenue
        cycle_result['revenue'] = revenue
        print(f"\n💰 Revenue to Finance: ${revenue:,.2f}")
        
        # Step 2: Finance distributes to subdomains
        print(f"\n📊 Finance distributes to subdomains:")
        for subdomain in self.subdomains.values():
            # More profit-focused → more funding
            allocation = revenue * 0.2 * subdomain.profit_focus
            subdomain.capital += allocation
            print(f"   {subdomain.name}: ${allocation:,.2f}")
        
        # Step 3: Subdomains create products
        print(f"\n🏭 Subdomains create products:")
        for subdomain in self.subdomains.values():
            if subdomain.capital > 10:
                product = {
                    'name': f"{subdomain.name} Product",
                    'subdomain': subdomain.name,
                    'value': subdomain.capital * 0.1
                }
                cycle_result['products_created'].append(product)
                print(f"   📦 {product['name']}: ${product['value']:,.2f}")
        
        # Step 4: Staff distributes to users
        print(f"\n📢 Staff distributes to users:")
        staff_count = len([a for a in self.actors.values() if a.is_staff])
        user_count = len(self.users)
        print(f"   Staff: {staff_count}, Users: {user_count}")
        
        # Step 5: Users create content
        print(f"\n🎨 Users create content:")
        for user in list(self.users.values())[:3]:  # Limit for demo
            content = self.user_creates_content(
                user.id,
                f"Content by {user.name}",
                energy=0.5
            )
            if content:
                cycle_result['content_created'].append(content)
        
        return cycle_result
    
    # ─────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────
    
    def visualize(self):
        """Visualize the organizational topology."""
        
        print("\n" + "═" * 70)
        print(f"ORGANIZATIONAL TOPOLOGY: {self.name}")
        print("═" * 70)
        
        # Header with mathematical boundaries
        print(f"""
    MATHEMATICAL BOUNDARIES:
    ─────────────────────────────────────────────────────────────────
    π = {PI:.4f}  : User ceiling (excess → convert)
    e = {E:.4f}  : Growth threshold
    φ = {PHI:.4f}  : Staff ceiling (classical domain limit)
    1 = {LN_BOUNDARY:.4f}  : Owner center (ln boundary)
    0 = Finance  : Antimatter interface
        """)
        
        # Users (φ to π)
        users_by_zone = {}
        for user in self.users.values():
            zone = user.zone
            if zone not in users_by_zone:
                users_by_zone[zone] = []
            users_by_zone[zone].append(user)
        
        print("┌" + "─" * 68 + "┐")
        print("│  USERS / CREATORS (φ to π)                                        │")
        print("│  Mathematical: Classical domain, can work down toward φ           │")
        print("├" + "─" * 68 + "┤")
        
        for zone in [OrgZone.POWER_USER, OrgZone.CREATOR, OrgZone.TRUSTED_USER]:
            users_in_zone = users_by_zone.get(zone, [])
            if users_in_zone:
                names = ", ".join([u.name[:15] for u in users_in_zone[:3]])
                print(f"│  {zone.value:12}: {names:<53}│")
        
        if not self.users:
            print("│  (no users yet)                                                   │")
        
        print("└" + "─" * 68 + "┘")
        
        # φ boundary
        print(f"\n{'═'*70}")
        print(f"  φ = {PHI:.4f} : STAFF CEILING (classical domain ends)")
        print(f"{'═'*70}\n")
        
        # Staff (1 to φ)
        staff = [a for a in self.actors.values() if a.is_staff and not a.is_owner]
        
        print("┌" + "─" * 68 + "┐")
        print("│  STAFF (1 to φ)                                                    │")
        print("│  Customer-facing: PR, Community, Sales, Support                    │")
        print("├" + "─" * 68 + "┤")
        
        if staff:
            for s in sorted(staff, key=lambda x: -x.position):
                print(f"│  👔 {s.name:<20} pos={s.position:.2f} ({s.zone.value:<12})    │")
        else:
            print("│  (no staff yet)                                                   │")
        
        print("└" + "─" * 68 + "┘")
        
        # Owner center (1.0)
        print(f"\n{'═'*70}")
        print(f"  1.0 = OWNER CENTER (ln boundary, mission core)")
        if self.owner:
            print(f"  👑 {self.owner.name}")
        print(f"{'═'*70}\n")
        
        # Business side (0 to 1)
        print("┌" + "─" * 68 + "┐")
        print("│  BUSINESS SIDE (0 to 1)                                            │")
        print("│  Internal: Subdomains by profit focus                              │")
        print("├" + "─" * 68 + "┤")
        
        # Sort subdomains by position (high to low)
        sorted_subs = sorted(self.subdomains.values(), key=lambda x: -x.position)
        for sub in sorted_subs:
            profit_bar = "█" * int(sub.profit_focus * 10) + "░" * (10 - int(sub.profit_focus * 10))
            print(f"│  📂 {sub.name:<18} pos={sub.position:.2f} profit=[{profit_bar}]   │")
        
        print("│                                                                    │")
        print(f"│  🏦 Finance (position 0) - Antimatter interface                    │")
        
        print("└" + "─" * 68 + "┘")
        
        # External
        print("         │")
        print("         ↓")
        print("    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
        print("    ▓  EXTERNAL (Banks, Markets, Chaos)  ▓")
        print("    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓")
    
    def display_status(self):
        """Display organization status."""
        print(f"\n{'═'*60}")
        print(f"📊 STATUS: {self.name}")
        print(f"{'═'*60}")
        print(f"  Capital: ${self.capital:,.2f}")
        print(f"  Subdomains: {len(self.subdomains)}")
        print(f"  Staff: {len([a for a in self.actors.values() if a.is_staff])}")
        print(f"  Users: {len(self.users)}")
        print(f"  Content in pool: {len(self.content_pool)}")
        print(f"  Integrated content: {len(self.integrated_content)}")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_org_topology():
    """Demonstrate the complete organizational topology."""
    
    print("\n" + "═" * 70)
    print("COMPLETE ORGANIZATIONAL TOPOLOGY")
    print("Mathematical boundaries define who can be where")
    print("═" * 70)
    
    # Create organization
    org = ShovelcatOrg("Shovelcat", initial_capital=2000.0)
    
    # Create subdomains at different positions
    print("\n" + "─" * 70)
    print("PHASE 1: Create subdomains (by profit focus)")
    print("─" * 70)
    
    gaming = org.create_subdomain("Gaming", position=0.25, initial_capital=500)
    products = org.create_subdomain("Products", position=0.4, initial_capital=300)
    research = org.create_subdomain("Research", position=0.6, initial_capital=200)
    education = org.create_subdomain("Education", position=0.9, initial_capital=200)
    health = org.create_subdomain("Health", position=0.9, initial_capital=200)
    
    # Add workers
    print("\n" + "─" * 70)
    print("PHASE 2: Add workers to subdomains")
    print("─" * 70)
    
    org.add_worker(gaming.id, "Game Dev Alice", 0.3)
    org.add_worker(gaming.id, "Game Dev Bob", 0.2)
    org.add_worker(education.id, "Curriculum Carol", 0.1)
    org.add_worker(health.id, "Health Researcher Dan", 0.05)
    
    # Hire staff
    print("\n" + "─" * 70)
    print("PHASE 3: Hire staff (position 1 to φ)")
    print("─" * 70)
    
    org.hire_staff("Community Manager", 1.2, "Community Lead")
    org.hire_staff("PR Lead", 1.4, "Public Relations")
    org.hire_staff("Social Media", 1.5, "Social Media Manager")
    
    # Register users
    print("\n" + "─" * 70)
    print("PHASE 4: Users enter (position φ to π)")
    print("─" * 70)
    
    user1 = org.register_user("PowerUser_123", PI, antimatter=0.5)
    user2 = org.register_user("Creator_456", PI - 0.3, antimatter=0.2)
    user3 = org.register_user("TrustedUser_789", PHI + 0.2, antimatter=0.0)
    
    # Users create content
    print("\n" + "─" * 70)
    print("PHASE 5: Users create content")
    print("─" * 70)
    
    org.user_creates_content(user1.id, "Epic Fan Video", 2.0)
    org.user_creates_content(user2.id, "Game Mod", 1.0)
    org.user_creates_content(user3.id, "Tutorial Guide", 0.5)
    
    # Run a cycle
    print("\n" + "─" * 70)
    print("PHASE 6: Run organizational cycle")
    print("─" * 70)
    
    result = org.run_cycle()
    
    # Visualize
    print("\n" + "─" * 70)
    print("PHASE 7: Visualize topology")
    print("─" * 70)
    
    org.visualize()
    org.display_status()
    
    # Summary
    print("\n" + "─" * 70)
    print("SUMMARY: Mathematical Boundaries")
    print("─" * 70)
    
    print(f"""
    THE NATURAL BOUNDARIES:
    
    Position π ({PI:.3f}): USER CEILING
        - Excess energy converts to content/antimatter
        - Content attracts new users
        - Good content gets integrated
    
    Position e ({E:.3f}): GROWTH THRESHOLD
        - Exponential expansion possible
        - Power users live here
    
    Position φ ({PHI:.3f}): STAFF CEILING
        - Classical domain limit: ln(φ) ≈ 0.48
        - Staff cannot go beyond this
        - Users cannot go below this (unless promoted)
    
    Position 1.0: OWNER CENTER
        - ln(1) = 0 : The mathematical boundary
        - Below 1: negatives in ln (quantum/internal)
        - Above 1: positives in ln (classical/external)
    
    Position 0: FINANCE
        - Antimatter interface
        - Handles all external chaos
        - Shields entire organization
    
    THE CYCLE:
    
        External $ ─→ Finance (0)
              │
              ↓
        Subdomains (0-1) by profit focus
              │
              ↓
        Products ─→ Owner (1)
              │
              ↓
        Staff (1-φ) distributes
              │
              ↓
        Users (φ-π) create content
              │
              ↓
        At π: content → antimatter → attract users
              │
              ↓
        New users enter at π, work down
    """)
    
    return org


if __name__ == "__main__":
    org = demonstrate_org_topology()
