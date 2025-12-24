"""
THREE-STAGE UPGRADE VERIFICATION
================================
Promotion/upgrade requires three simultaneous verifications:
1. USER REQUEST (Desire) - User wants the upgrade
2. AUTHORITY APPROVAL (Boss) - Organization wants to promote
3. RISK VERIFICATION (IT) - Safe to promote

Geometry:
- Full verification = Square (all edges straight)
- Missing verification = Hexagon (elbows bent outward)
- Processing = Conveyor belt with interlocking teeth

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from datetime import datetime

# Import from euler_access_control if available
try:
    from euler_access_control import IdentityPhase, Department, PI
except ImportError:
    PI = math.pi
    
    @dataclass
    class IdentityPhase:
        user_id: str
        theta: float = 0.0
        department: Optional[str] = None
        
    class Department(Enum):
        SALES = 0
        TECHNOLOGY = 120
        FINANCE = 180


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION STAGES
# ═══════════════════════════════════════════════════════════════════════════════

class VerificationStage(Enum):
    """The three stages of upgrade verification."""
    REQUEST = auto()    # User's desire
    APPROVAL = auto()   # Authority's decision
    RISK = auto()       # Security verification


@dataclass
class StageResult:
    """Result of a single verification stage."""
    stage: VerificationStage
    passed: bool
    reason: str
    verified_by: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC STATE (Square ↔ Hexagon)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeometricState:
    """
    The geometric state of verification.
    
    - All verified: Clean SQUARE (edges straight)
    - Missing verifications: HEXAGON (elbows bent outward)
    
    Each missing verification bends an edge outward by an "elbow"
    """
    request_verified: bool = False
    approval_verified: bool = False
    risk_verified: bool = False
    
    @property
    def verification_count(self) -> int:
        """Number of stages verified."""
        return sum([self.request_verified, self.approval_verified, self.risk_verified])
    
    @property
    def is_complete(self) -> bool:
        """All three stages verified."""
        return self.request_verified and self.approval_verified and self.risk_verified
    
    @property
    def shape(self) -> str:
        """Current geometric shape."""
        if self.is_complete:
            return "SQUARE"
        elif self.verification_count == 0:
            return "HEXAGON"  # Maximum uncertainty
        else:
            return f"BENT_{3 - self.verification_count}"  # Partially bent
    
    @property
    def bend_angles(self) -> Dict[str, float]:
        """
        Calculate bend angle for each edge.
        Verified = 0° (straight)
        Unverified = 60° (bent outward to hexagon)
        """
        return {
            'request': 0 if self.request_verified else 60,
            'approval': 0 if self.approval_verified else 60,
            'risk': 0 if self.risk_verified else 60
        }
    
    @property
    def total_bend(self) -> float:
        """Total bend angle (0 = square, 180 = full hexagon)."""
        return sum(self.bend_angles.values())
    
    @property
    def uncertainty(self) -> float:
        """Uncertainty level (0 = certain, 1 = maximum uncertainty)."""
        return self.total_bend / 180  # Normalized to 0-1
    
    def ascii_shape(self) -> str:
        """Generate ASCII representation of current shape."""
        if self.is_complete:
            return self._draw_square()
        else:
            return self._draw_hexagon()
    
    def _draw_square(self) -> str:
        return """
              REQUEST ✓
                 │
         ┌───────┴───────┐
         │               │
         │               │
  RISK ✓─│       ●       │─ APPROVAL ✓
         │   VERIFIED    │
         │               │
         └───────────────┘
        """
    
    def _draw_hexagon(self) -> str:
        r = "✓" if self.request_verified else "?"
        a = "✓" if self.approval_verified else "?"
        s = "✓" if self.risk_verified else "?"
        
        return f"""
              REQUEST {r}
                 ╱╲
                ╱  ╲
       RISK {s}─●    ●─APPROVAL {a}
              ╱      ╲
             ╱   ○    ╲
            ╱ PENDING  ╲
           ●            ●
            ╲          ╱
             ╲        ╱
              ╲      ╱
       RISK {s}─●    ●─APPROVAL {a}
                ╲  ╱
                 ╲╱
              REQUEST {r}
        """


# ═══════════════════════════════════════════════════════════════════════════════
# TEETH AND BINDING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationTooth:
    """
    A single tooth in the verification mechanism.
    Teeth interlock like gears when they match.
    """
    stage: VerificationStage
    angle: float  # Base angle
    strength: float = 1.0  # 0-1
    bent: bool = False
    
    @property
    def effective_angle(self) -> float:
        """Angle after bending."""
        if self.bent:
            return self.angle + 30  # Bent outward by 30°
        return self.angle
    
    def meshes_with(self, receptor: 'VerificationTooth') -> Tuple[bool, float]:
        """
        Check if this tooth meshes with a receptor.
        Returns (meshes, strength)
        
        Same-stage teeth should mesh when both verified.
        """
        # Same stage teeth mesh when neither is bent
        if self.stage == receptor.stage:
            if not self.bent and not receptor.bent:
                return True, self.strength * receptor.strength
            else:
                # Bent teeth don't mesh well
                return False, 0.3 * self.strength * receptor.strength
        
        # Different stage teeth - check angle alignment
        angle_diff = abs(self.effective_angle - receptor.effective_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        mesh_strength = math.cos(math.radians(angle_diff))
        meshes = mesh_strength > 0.5
        
        return meshes, max(0, mesh_strength * self.strength)


@dataclass  
class TeethSet:
    """A set of verification teeth."""
    teeth: List[VerificationTooth] = field(default_factory=list)
    
    def add_stage(self, stage: VerificationStage, verified: bool):
        """Add a tooth for a verification stage."""
        base_angles = {
            VerificationStage.REQUEST: 0,
            VerificationStage.APPROVAL: 120,
            VerificationStage.RISK: 240
        }
        
        tooth = VerificationTooth(
            stage=stage,
            angle=base_angles[stage],
            strength=1.0 if verified else 0.5,
            bent=not verified
        )
        self.teeth.append(tooth)
    
    def mesh_strength(self, template: 'TeethSet') -> Dict[str, Any]:
        """Calculate how well this set meshes with a template."""
        result = {
            'total_strength': 0,
            'per_stage': {},
            'all_mesh': True
        }
        
        for my_tooth in self.teeth:
            # Find matching template tooth
            for template_tooth in template.teeth:
                if template_tooth.stage == my_tooth.stage:
                    meshes, strength = my_tooth.meshes_with(template_tooth)
                    result['per_stage'][my_tooth.stage.name] = {
                        'meshes': meshes,
                        'strength': strength
                    }
                    result['total_strength'] += strength
                    if not meshes:
                        result['all_mesh'] = False
                    break
        
        if self.teeth:
            result['total_strength'] /= len(self.teeth)
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONVEYOR BELT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConveyorInput:
    """An input moving along the conveyor belt."""
    id: str
    user_id: str
    requested_level: float  # Target theta
    current_level: float    # Current theta
    teeth: TeethSet = field(default_factory=TeethSet)
    position: int = 0       # Position on belt
    status: str = "pending"


class ConveyorBelt:
    """
    Processes upgrade requests through verification stages.
    Like a factory conveyor belt with alignment stations.
    """
    
    def __init__(self):
        self.inputs: List[ConveyorInput] = []
        self.stations: List[VerificationStage] = [
            VerificationStage.REQUEST,
            VerificationStage.APPROVAL,
            VerificationStage.RISK
        ]
        self.template = self._create_template()
    
    def _create_template(self) -> TeethSet:
        """Create the verification template (what inputs must match)."""
        template = TeethSet()
        for stage in self.stations:
            template.add_stage(stage, verified=True)  # Template is fully verified
        return template
    
    def add_input(self, user_id: str, requested_level: float, 
                  current_level: float) -> ConveyorInput:
        """Add a new upgrade request to the belt."""
        input_item = ConveyorInput(
            id=f"UPG-{len(self.inputs):04d}",
            user_id=user_id,
            requested_level=requested_level,
            current_level=current_level
        )
        self.inputs.append(input_item)
        return input_item
    
    def process_stage(self, input_item: ConveyorInput, 
                      stage: VerificationStage,
                      result: StageResult) -> bool:
        """
        Process a single verification stage.
        Returns True if teeth mesh at this stage.
        """
        # Add tooth based on result
        input_item.teeth.add_stage(stage, result.passed)
        
        # Check mesh
        mesh_result = input_item.teeth.mesh_strength(self.template)
        stage_mesh = mesh_result['per_stage'].get(stage.name, {})
        
        return stage_mesh.get('meshes', False)
    
    def advance(self, input_item: ConveyorInput) -> str:
        """
        Advance the input along the belt.
        Returns current status.
        """
        if input_item.position >= len(self.stations):
            # All stations passed
            mesh = input_item.teeth.mesh_strength(self.template)
            if mesh['all_mesh']:
                input_item.status = "APPROVED"
            else:
                input_item.status = "REJECTED"
        else:
            input_item.status = f"STATION_{input_item.position + 1}"
            input_item.position += 1
        
        return input_item.status
    
    def visualize(self, input_item: ConveyorInput) -> str:
        """Generate ASCII visualization of the conveyor."""
        lines = []
        lines.append("=" * 70)
        lines.append("CONVEYOR BELT VISUALIZATION")
        lines.append("=" * 70)
        
        # Belt
        belt = "═" * 60
        lines.append(f"\n    {belt}►")
        
        # Stations
        station_line = "    "
        for i, stage in enumerate(self.stations):
            marker = "●" if input_item.position > i else "○"
            name = stage.name[:3]
            station_line += f"   [{marker} {name}]        "
        lines.append(station_line)
        
        # Input position
        pos_line = "    "
        pos = input_item.position
        pos_line += "          " * pos + "▲"
        lines.append(pos_line)
        lines.append("    " + "          " * pos + f"│ {input_item.id}")
        
        lines.append(f"\n    {belt}►")
        lines.append(f"\n    Status: {input_item.status}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE REQUEST HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class UpgradeVerificationSystem:
    """
    Complete system for handling upgrade requests.
    Combines three-stage verification with conveyor processing.
    """
    
    def __init__(self):
        self.conveyor = ConveyorBelt()
        self.pending_requests: Dict[str, GeometricState] = {}
        self.verifications: Dict[str, List[StageResult]] = {}
    
    def request_upgrade(self, user_id: str, 
                       current_theta: float,
                       requested_theta: float) -> Dict[str, Any]:
        """
        Initiate an upgrade request.
        This is STAGE 1 (User Request).
        """
        # Create geometric state
        state = GeometricState()
        self.pending_requests[user_id] = state
        self.verifications[user_id] = []
        
        # Create conveyor input
        input_item = self.conveyor.add_input(
            user_id=user_id,
            requested_level=requested_theta,
            current_level=current_theta
        )
        
        # User requesting = Stage 1 verified
        stage_result = StageResult(
            stage=VerificationStage.REQUEST,
            passed=True,
            reason="User explicitly requested upgrade",
            verified_by=user_id,
            details={
                'current_level': current_theta,
                'requested_level': requested_theta,
                'delta': requested_theta - current_theta
            }
        )
        
        self.verifications[user_id].append(stage_result)
        state.request_verified = True
        
        # Process through first station
        self.conveyor.process_stage(input_item, VerificationStage.REQUEST, stage_result)
        self.conveyor.advance(input_item)
        
        return {
            'request_id': input_item.id,
            'user_id': user_id,
            'stage': 'REQUEST',
            'passed': True,
            'shape': state.shape,
            'uncertainty': state.uncertainty,
            'next_required': 'APPROVAL',
            'visualization': state.ascii_shape()
        }
    
    def approve_upgrade(self, user_id: str, 
                       approver_id: str,
                       approved: bool,
                       reason: str = "") -> Dict[str, Any]:
        """
        Authority approval for upgrade.
        This is STAGE 2 (Boss Approval).
        """
        state = self.pending_requests.get(user_id)
        if not state:
            return {'error': 'No pending request for user'}
        
        if not state.request_verified:
            return {'error': 'User must request first'}
        
        # Find the input on conveyor
        input_item = None
        for item in self.conveyor.inputs:
            if item.user_id == user_id and item.status.startswith("STATION"):
                input_item = item
                break
        
        if not input_item:
            return {'error': 'Request not found on conveyor'}
        
        # Record approval decision
        stage_result = StageResult(
            stage=VerificationStage.APPROVAL,
            passed=approved,
            reason=reason or ("Approved" if approved else "Denied"),
            verified_by=approver_id,
            details={'approver_role': 'authority'}
        )
        
        self.verifications[user_id].append(stage_result)
        state.approval_verified = approved
        
        # Process through station
        self.conveyor.process_stage(input_item, VerificationStage.APPROVAL, stage_result)
        
        if not approved:
            input_item.status = "REJECTED_APPROVAL"
            return {
                'request_id': input_item.id,
                'user_id': user_id,
                'stage': 'APPROVAL',
                'passed': False,
                'reason': reason,
                'shape': state.shape,
                'final': True,
                'visualization': state.ascii_shape()
            }
        
        self.conveyor.advance(input_item)
        
        return {
            'request_id': input_item.id,
            'user_id': user_id,
            'stage': 'APPROVAL',
            'passed': True,
            'shape': state.shape,
            'uncertainty': state.uncertainty,
            'next_required': 'RISK',
            'visualization': state.ascii_shape()
        }
    
    def verify_risk(self, user_id: str,
                   verifier_id: str,
                   safe: bool,
                   reason: str = "",
                   risk_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Risk/security verification.
        This is STAGE 3 (IT/Security Check).
        """
        state = self.pending_requests.get(user_id)
        if not state:
            return {'error': 'No pending request for user'}
        
        if not state.approval_verified:
            return {'error': 'Authority must approve first'}
        
        # Find the input on conveyor
        input_item = None
        for item in self.conveyor.inputs:
            if item.user_id == user_id and item.status.startswith("STATION"):
                input_item = item
                break
        
        if not input_item:
            return {'error': 'Request not found on conveyor'}
        
        # Record risk decision
        stage_result = StageResult(
            stage=VerificationStage.RISK,
            passed=safe,
            reason=reason or ("Safe" if safe else "Risk detected"),
            verified_by=verifier_id,
            details=risk_details or {}
        )
        
        self.verifications[user_id].append(stage_result)
        state.risk_verified = safe
        
        # Process through station
        self.conveyor.process_stage(input_item, VerificationStage.RISK, stage_result)
        
        if not safe:
            input_item.status = "REJECTED_RISK"
            return {
                'request_id': input_item.id,
                'user_id': user_id,
                'stage': 'RISK',
                'passed': False,
                'reason': reason,
                'shape': state.shape,
                'final': True,
                'visualization': state.ascii_shape()
            }
        
        self.conveyor.advance(input_item)
        
        # All stages passed!
        return {
            'request_id': input_item.id,
            'user_id': user_id,
            'stage': 'RISK',
            'passed': True,
            'shape': state.shape,
            'uncertainty': state.uncertainty,
            'final': True,
            'approved': True,
            'new_theta': input_item.requested_level,
            'visualization': state.ascii_shape(),
            'conveyor': self.conveyor.visualize(input_item)
        }
    
    def get_status(self, user_id: str) -> Dict[str, Any]:
        """Get current status of an upgrade request."""
        state = self.pending_requests.get(user_id)
        if not state:
            return {'error': 'No pending request'}
        
        return {
            'user_id': user_id,
            'request_verified': state.request_verified,
            'approval_verified': state.approval_verified,
            'risk_verified': state.risk_verified,
            'shape': state.shape,
            'uncertainty': state.uncertainty,
            'bend_angles': state.bend_angles,
            'complete': state.is_complete,
            'visualization': state.ascii_shape()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_upgrade_flow():
    """Demonstrate the complete upgrade flow."""
    
    print("\n" + "=" * 80)
    print("THREE-STAGE UPGRADE VERIFICATION DEMO")
    print("=" * 80)
    
    system = UpgradeVerificationSystem()
    
    # Scenario: Regular user wants to become trusted partner
    print("\n" + "─" * 40)
    print("SCENARIO: Regular User → Trusted Partner")
    print("─" * 40)
    
    # Stage 1: User requests
    print("\n>>> STAGE 1: User Request")
    print("User 'alice' requests upgrade from θ=0.3 to θ=1.57 (trusted partner)")
    
    result = system.request_upgrade(
        user_id="alice",
        current_theta=0.3,
        requested_theta=PI/2
    )
    
    print(f"\nResult: {result['stage']} - {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Shape: {result['shape']}")
    print(f"Uncertainty: {result['uncertainty']:.2f}")
    print(result['visualization'])
    
    # Stage 2: Boss approves
    print("\n>>> STAGE 2: Authority Approval")
    print("Manager 'bob' reviews and approves alice's upgrade request")
    
    result = system.approve_upgrade(
        user_id="alice",
        approver_id="bob",
        approved=True,
        reason="Alice has demonstrated value over 6 months"
    )
    
    print(f"\nResult: {result['stage']} - {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Shape: {result['shape']}")
    print(f"Uncertainty: {result.get('uncertainty', 'N/A')}")
    print(result['visualization'])
    
    # Stage 3: IT verifies safe
    print("\n>>> STAGE 3: Risk Verification")
    print("Security team 'charlie' runs background check")
    
    result = system.verify_risk(
        user_id="alice",
        verifier_id="charlie",
        safe=True,
        reason="No security concerns, clean history",
        risk_details={
            'background_check': 'PASS',
            'behavior_analysis': 'NORMAL',
            'access_pattern': 'CONSISTENT'
        }
    )
    
    print(f"\nResult: {result['stage']} - {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Shape: {result['shape']}")
    print(f"FINAL: {'APPROVED' if result.get('approved') else 'REJECTED'}")
    if result.get('approved'):
        print(f"New θ: {result['new_theta']:.2f} ({math.degrees(result['new_theta']):.1f}°)")
    print(result['visualization'])
    print(result.get('conveyor', ''))


def demo_rejected_upgrade():
    """Demonstrate a rejected upgrade."""
    
    print("\n" + "=" * 80)
    print("REJECTED UPGRADE DEMO")
    print("=" * 80)
    
    system = UpgradeVerificationSystem()
    
    # Stage 1: User requests
    print("\n>>> STAGE 1: Suspicious user 'eve' requests upgrade")
    
    result = system.request_upgrade(
        user_id="eve",
        current_theta=0.1,
        requested_theta=PI  # Wants owner access!
    )
    
    print(f"Shape after request: {result['shape']}")
    
    # Stage 2: Boss approves (maybe they're new)
    print("\n>>> STAGE 2: New manager approves without due diligence")
    
    result = system.approve_upgrade(
        user_id="eve",
        approver_id="new_manager",
        approved=True,
        reason="Seems fine"
    )
    
    print(f"Shape after approval: {result['shape']}")
    
    # Stage 3: IT catches the risk
    print("\n>>> STAGE 3: Security team catches suspicious pattern")
    
    result = system.verify_risk(
        user_id="eve",
        verifier_id="security_team",
        safe=False,
        reason="ALERT: Suspicious access patterns detected",
        risk_details={
            'background_check': 'FAILED',
            'behavior_analysis': 'ANOMALOUS',
            'access_pattern': 'PRIVILEGE_ESCALATION_ATTEMPT',
            'risk_score': 0.95
        }
    )
    
    print(f"\nResult: {result['stage']} - {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Shape: {result['shape']}")
    print(f"Reason: {result['reason']}")
    print(result['visualization'])
    print("\n⚠️  UPGRADE BLOCKED - TEETH DON'T MESH")


def demo_geometry():
    """Demonstrate the geometric transformation."""
    
    print("\n" + "=" * 80)
    print("GEOMETRY DEMONSTRATION: Square ↔ Hexagon")
    print("=" * 80)
    
    # All stages pending
    print("\n>>> ALL PENDING (Maximum Hexagon)")
    state = GeometricState()
    print(f"Shape: {state.shape}")
    print(f"Total bend: {state.total_bend}°")
    print(f"Uncertainty: {state.uncertainty:.2%}")
    print(state.ascii_shape())
    
    # One stage done
    print("\n>>> REQUEST VERIFIED (Partial)")
    state.request_verified = True
    print(f"Shape: {state.shape}")
    print(f"Total bend: {state.total_bend}°")
    print(f"Uncertainty: {state.uncertainty:.2%}")
    print(state.ascii_shape())
    
    # Two stages done
    print("\n>>> REQUEST + APPROVAL VERIFIED")
    state.approval_verified = True
    print(f"Shape: {state.shape}")
    print(f"Total bend: {state.total_bend}°")
    print(f"Uncertainty: {state.uncertainty:.2%}")
    print(state.ascii_shape())
    
    # All stages done
    print("\n>>> ALL VERIFIED (Clean Square)")
    state.risk_verified = True
    print(f"Shape: {state.shape}")
    print(f"Total bend: {state.total_bend}°")
    print(f"Uncertainty: {state.uncertainty:.2%}")
    print(state.ascii_shape())


def run_tests():
    """Run test suite."""
    
    print("\n" + "=" * 80)
    print("THREE-STAGE VERIFICATION - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Geometric state
    print("\n--- TEST 1: Geometric State ---")
    state = GeometricState()
    assert state.shape == "HEXAGON", "Empty state should be hexagon"
    assert state.uncertainty == 1.0, "Empty state should have max uncertainty"
    print("✓ Empty state is hexagon with max uncertainty")
    
    state.request_verified = True
    state.approval_verified = True
    state.risk_verified = True
    assert state.shape == "SQUARE", "Full state should be square"
    assert state.uncertainty == 0.0, "Full state should have no uncertainty"
    print("✓ Full state is square with no uncertainty")
    
    # Test 2: Teeth meshing
    print("\n--- TEST 2: Teeth Meshing ---")
    template = TeethSet()
    for stage in VerificationStage:
        template.add_stage(stage, True)
    
    verified_teeth = TeethSet()
    for stage in VerificationStage:
        verified_teeth.add_stage(stage, True)
    
    mesh = verified_teeth.mesh_strength(template)
    assert mesh['all_mesh'], "Verified teeth should mesh"
    print("✓ Verified teeth mesh with template")
    
    bent_teeth = TeethSet()
    for stage in VerificationStage:
        bent_teeth.add_stage(stage, False)  # All bent
    
    mesh = bent_teeth.mesh_strength(template)
    assert not mesh['all_mesh'], "Bent teeth should not mesh"
    print("✓ Bent teeth do not mesh with template")
    
    # Test 3: Upgrade flow
    print("\n--- TEST 3: Upgrade Flow ---")
    system = UpgradeVerificationSystem()
    
    result = system.request_upgrade("test_user", 0.1, 1.0)
    assert result['passed'], "Request should pass"
    assert result['shape'] != "SQUARE", "Should not be square yet"
    print("✓ Request stage works")
    
    result = system.approve_upgrade("test_user", "manager", True, "Approved")
    assert result['passed'], "Approval should pass"
    print("✓ Approval stage works")
    
    result = system.verify_risk("test_user", "security", True, "Safe")
    assert result['passed'], "Risk should pass"
    assert result['approved'], "Should be approved"
    print("✓ Risk stage works")
    
    # Test 4: Rejection works
    print("\n--- TEST 4: Rejection at Risk Stage ---")
    system2 = UpgradeVerificationSystem()
    system2.request_upgrade("risky_user", 0.1, PI)
    system2.approve_upgrade("risky_user", "manager", True, "OK")
    result = system2.verify_risk("risky_user", "security", False, "Too risky")
    assert not result['passed'], "Should fail"
    assert result.get('approved') is None or not result.get('approved'), "Should not be approved"
    print("✓ Rejection at risk stage works")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_upgrade_flow()
        elif sys.argv[1] == "--rejected":
            demo_rejected_upgrade()
        elif sys.argv[1] == "--geometry":
            demo_geometry()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_geometry()
            demo_upgrade_flow()
            demo_rejected_upgrade()
        else:
            run_tests()
    else:
        run_tests()
