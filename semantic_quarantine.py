"""
SEMANTIC TEMPERATURE & QUARANTINE SYSTEM
=========================================
Conservation-preserving handling of hallucinations.

Key insight: We can't DELETE hallucinations (violates conservation).
Instead, we FREEZE them - convert kinetic energy to potential,
preventing rebinding while preserving components.

Temperature controls binding:
  - FROZEN (T=0): Cannot bind, stored in cold pool
  - COLD (T<1): Slow binding, low activity
  - ROOM (T=1): Normal operation
  - HOT (T>2): Rapid binding, risk of hallucination
  - CRITICAL (T>3): Unstable, hallucinations likely

Hallucination handling:
  1. Detect (verify path, θ, energy)
  2. Decompose (break into components)
  3. Freeze (kinetic → potential)
  4. Store (cold pool)
  5. Thaw on legitimate request (potential → kinetic)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
from enum import Enum
import hashlib
import json

# Import from our previous modules
from semantic_wave_equation import (
    SemanticPDE, SemanticWaveFunction, SemanticEigenfunction,
    ColorEncoding, SemanticPathNavigator, PathOperation,
    DIM_ORDER, DIMENSION_STATES, H_INFO, PHI, E, PI
)
from semantic_hamiltonian import (
    SemanticHamiltonian, HamiltonianWord, SemanticDisk,
    MomentumMatrix, SemanticMass, SemanticPotential
)


# =============================================================================
# TEMPERATURE CONSTANTS
# =============================================================================

class TemperaturePhase(Enum):
    """Phase states based on temperature."""
    FROZEN = "frozen"       # T = 0, cannot bind
    COLD = "cold"           # T < 0.5, very slow binding
    COOL = "cool"           # T < 1.0, slow binding
    ROOM = "room"           # T = 1.0, normal
    WARM = "warm"           # T < 2.0, fast binding
    HOT = "hot"             # T < 3.0, rapid binding
    CRITICAL = "critical"   # T >= 3.0, unstable


PHASE_THRESHOLDS = {
    TemperaturePhase.FROZEN: 0.0,
    TemperaturePhase.COLD: 0.5,
    TemperaturePhase.COOL: 1.0,
    TemperaturePhase.ROOM: 1.5,
    TemperaturePhase.WARM: 2.0,
    TemperaturePhase.HOT: 3.0,
    TemperaturePhase.CRITICAL: float('inf')
}


# =============================================================================
# SEMANTIC TEMPERATURE
# =============================================================================

@dataclass
class TemperatureEvent:
    """Record of a temperature change."""
    timestamp: datetime
    temp_before: float
    temp_after: float
    event_type: str  # 'freeze', 'thaw', 'heat', 'cool'
    source: str
    reason: str


@dataclass
class SemanticTemperature:
    """
    Temperature state for semantic components.
    Controls binding capability and energy distribution.
    """
    temperature: float = 1.0  # Start at room temperature
    
    # History
    frozen_at: Optional[datetime] = None
    freeze_reason: Optional[str] = None
    thawed_at: Optional[datetime] = None
    history: List[TemperatureEvent] = field(default_factory=list)
    
    @property
    def phase(self) -> TemperaturePhase:
        """Current phase based on temperature."""
        if self.temperature <= 0:
            return TemperaturePhase.FROZEN
        elif self.temperature < 0.5:
            return TemperaturePhase.COLD
        elif self.temperature < 1.0:
            return TemperaturePhase.COOL
        elif self.temperature < 1.5:
            return TemperaturePhase.ROOM
        elif self.temperature < 2.0:
            return TemperaturePhase.WARM
        elif self.temperature < 3.0:
            return TemperaturePhase.HOT
        else:
            return TemperaturePhase.CRITICAL
    
    @property
    def can_bind(self) -> bool:
        """Components can only bind if not frozen."""
        return self.temperature > 0
    
    @property
    def binding_rate(self) -> float:
        """How fast binding occurs (0 if frozen)."""
        if self.temperature <= 0:
            return 0.0
        # Arrhenius-like: rate increases with temperature
        return min(1.0, self.temperature / 2.0)
    
    @property
    def hallucination_risk(self) -> float:
        """Risk of spontaneous (invalid) binding."""
        if self.temperature < 2.0:
            return 0.0
        elif self.temperature < 3.0:
            return (self.temperature - 2.0) / 2.0  # 0 to 0.5
        else:
            return 0.5 + (self.temperature - 3.0) / 4.0  # > 0.5
    
    def freeze(self, reason: str = "hallucination_detected") -> TemperatureEvent:
        """
        Freeze: remove all kinetic energy.
        Component becomes inert, cannot bind.
        """
        event = TemperatureEvent(
            timestamp=datetime.now(),
            temp_before=self.temperature,
            temp_after=0.0,
            event_type='freeze',
            source='system',
            reason=reason
        )
        
        self.temperature = 0.0
        self.frozen_at = datetime.now()
        self.freeze_reason = reason
        self.history.append(event)
        
        return event
    
    def thaw(self, heat_added: float, source: str, reason: str) -> Optional[TemperatureEvent]:
        """
        Thaw: add energy back, allowing binding.
        Requires justification.
        """
        if heat_added <= 0:
            return None
        
        event = TemperatureEvent(
            timestamp=datetime.now(),
            temp_before=self.temperature,
            temp_after=self.temperature + heat_added,
            event_type='thaw',
            source=source,
            reason=reason
        )
        
        self.temperature += heat_added
        self.thawed_at = datetime.now()
        self.history.append(event)
        
        return event
    
    def heat(self, amount: float, source: str) -> TemperatureEvent:
        """Add heat (increase binding rate)."""
        event = TemperatureEvent(
            timestamp=datetime.now(),
            temp_before=self.temperature,
            temp_after=self.temperature + amount,
            event_type='heat',
            source=source,
            reason='energy_added'
        )
        self.temperature += amount
        self.history.append(event)
        return event
    
    def cool(self, amount: float, source: str) -> TemperatureEvent:
        """Remove heat (decrease binding rate, approach frozen)."""
        event = TemperatureEvent(
            timestamp=datetime.now(),
            temp_before=self.temperature,
            temp_after=max(0, self.temperature - amount),
            event_type='cool',
            source=source,
            reason='energy_removed'
        )
        self.temperature = max(0, self.temperature - amount)
        self.history.append(event)
        return event


# =============================================================================
# SEMANTIC COMPONENT (with temperature)
# =============================================================================

@dataclass
class SemanticComponent:
    """
    A single semantic component that can be frozen or active.
    Smallest unit that can be decomposed from a hallucination.
    """
    id: str
    dimension: str  # Which color dimension, or 'operation', or 'binding'
    value: Any      # The actual value
    origin: str     # Where this came from
    
    # Temperature state
    temperature: SemanticTemperature = field(default_factory=lambda: SemanticTemperature(temperature=1.0))
    
    # Energy (for conservation)
    kinetic_energy: float = 1.0
    potential_energy: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    decomposed_from: Optional[str] = None  # ID of hallucination if decomposed
    
    @property
    def total_energy(self) -> float:
        """H = T + V, conserved even when frozen."""
        return self.kinetic_energy + self.potential_energy
    
    @property
    def can_bind(self) -> bool:
        """Can this component participate in binding?"""
        return self.temperature.can_bind
    
    def freeze(self, reason: str = "quarantine"):
        """
        Freeze: convert kinetic → potential.
        Total energy conserved!
        """
        # Store kinetic energy as potential
        self.potential_energy += self.kinetic_energy
        self.kinetic_energy = 0.0
        
        # Set temperature to frozen
        self.temperature.freeze(reason)
        
        # H_before = T + V
        # H_after = 0 + (V + T_old) = same!
    
    def thaw(self, heat_added: float, source: str, reason: str) -> bool:
        """
        Thaw: external heat becomes kinetic energy.
        Returns True if successfully thawed.
        """
        if heat_added <= 0:
            return False
        
        # External heat adds to kinetic energy
        self.kinetic_energy += heat_added
        
        # Temperature update
        self.temperature.thaw(heat_added, source, reason)
        
        return self.temperature.can_bind
    
    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            'id': self.id,
            'dimension': self.dimension,
            'value': str(self.value),
            'origin': self.origin,
            'temperature': self.temperature.temperature,
            'kinetic_energy': self.kinetic_energy,
            'potential_energy': self.potential_energy,
            'created_at': self.created_at.isoformat(),
            'decomposed_from': self.decomposed_from
        }


# =============================================================================
# HALLUCINATION DETECTOR
# =============================================================================

@dataclass
class VerificationResult:
    """Result of verifying a semantic claim."""
    is_valid: bool
    checks_passed: List[str]
    checks_failed: List[str]
    confidence: float
    details: Dict[str, Any]


class HallucinationDetector:
    """
    Detect hallucinations by verifying path, θ, and energy consistency.
    
    A valid meaning has:
    - Complete path from birth to current state
    - θ signature that matches the path
    - Energy conservation throughout history
    - Birth certificate (origin hash)
    
    A hallucination is missing one or more of these.
    """
    
    def __init__(self, h_info: float = H_INFO):
        self.h_info = h_info
        self.detection_log: List[Dict] = []
    
    def verify(self, claim: Dict[str, Any]) -> VerificationResult:
        """
        Verify a semantic claim.
        Returns detailed verification result.
        """
        checks_passed = []
        checks_failed = []
        details = {}
        
        # CHECK 1: Does encoding exist and is valid?
        encoding_valid = self._check_encoding(claim)
        if encoding_valid:
            checks_passed.append('encoding_valid')
        else:
            checks_failed.append('encoding_invalid')
        details['encoding'] = encoding_valid
        
        # CHECK 2: Does path exist?
        path_exists = self._check_path_exists(claim)
        if path_exists:
            checks_passed.append('path_exists')
        else:
            checks_failed.append('path_missing')
        details['path_exists'] = path_exists
        
        # CHECK 3: Does θ signature match path?
        theta_valid = self._check_theta_consistency(claim)
        if theta_valid:
            checks_passed.append('theta_consistent')
        else:
            checks_failed.append('theta_inconsistent')
        details['theta_valid'] = theta_valid
        
        # CHECK 4: Is energy conserved?
        energy_valid = self._check_energy_conservation(claim)
        if energy_valid:
            checks_passed.append('energy_conserved')
        else:
            checks_failed.append('energy_violated')
        details['energy_valid'] = energy_valid
        
        # CHECK 5: Does birth certificate exist?
        birth_valid = self._check_birth_certificate(claim)
        if birth_valid:
            checks_passed.append('birth_verified')
        else:
            checks_failed.append('birth_missing')
        details['birth_valid'] = birth_valid
        
        # CHECK 6: Is the hash consistent?
        hash_valid = self._check_hash_consistency(claim)
        if hash_valid:
            checks_passed.append('hash_verified')
        else:
            checks_failed.append('hash_invalid')
        details['hash_valid'] = hash_valid
        
        # Overall validity
        is_valid = len(checks_failed) == 0
        confidence = len(checks_passed) / 6.0
        
        result = VerificationResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            confidence=confidence,
            details=details
        )
        
        # Log detection
        self.detection_log.append({
            'timestamp': datetime.now().isoformat(),
            'claim': str(claim.get('label', 'unknown')),
            'is_valid': is_valid,
            'confidence': confidence,
            'checks_failed': checks_failed
        })
        
        return result
    
    def _check_encoding(self, claim: Dict) -> bool:
        """Check if encoding is valid."""
        if 'encoding' not in claim:
            return False
        encoding = claim['encoding']
        if isinstance(encoding, ColorEncoding):
            arr = encoding.to_array()
        elif isinstance(encoding, (list, tuple, np.ndarray)):
            arr = np.array(encoding)
        else:
            return False
        
        # Check bounds
        for i, dim in enumerate(DIM_ORDER):
            if i >= len(arr):
                return False
            min_val = DIMENSION_STATES[dim]['min']
            max_val = DIMENSION_STATES[dim]['max']
            if not (min_val <= arr[i] <= max_val):
                return False
        return True
    
    def _check_path_exists(self, claim: Dict) -> bool:
        """Check if path history exists."""
        if 'path' not in claim and 'path_record' not in claim:
            return False
        path = claim.get('path') or claim.get('path_record', [])
        return len(path) > 0
    
    def _check_theta_consistency(self, claim: Dict) -> bool:
        """Check if θ signature matches the recorded path."""
        if 'theta_signature' not in claim:
            return False
        if 'theta_trail' not in claim and 'path' not in claim:
            return False
        
        claimed_theta = claim['theta_signature']
        
        # If we have the trail, sum it
        if 'theta_trail' in claim:
            trail = claim['theta_trail']
            computed_theta = sum(trail) if isinstance(trail, list) else trail
            
            # Allow some tolerance
            if isinstance(claimed_theta, tuple):
                computed = sum(claimed_theta)
            else:
                computed = claimed_theta
            
            return abs(computed - computed_theta) < 0.1
        
        return True  # Can't verify without trail, assume ok
    
    def _check_energy_conservation(self, claim: Dict) -> bool:
        """Check if energy is conserved through history."""
        if 'energy_history' not in claim:
            return False
        
        history = claim['energy_history']
        if len(history) < 2:
            return True  # Not enough history to check
        
        # Check that energy changes are reasonable
        for i in range(1, len(history)):
            prev_E = history[i-1].get('H', 0)
            curr_E = history[i].get('H', 0)
            delta_E = history[i].get('delta_E', curr_E - prev_E)
            
            # Energy should change by operation, not spontaneously
            if 'operation' not in history[i]:
                if abs(delta_E) > 0.01:
                    return False  # Spontaneous energy change!
        
        return True
    
    def _check_birth_certificate(self, claim: Dict) -> bool:
        """Check if birth certificate exists and is valid."""
        if 'birth' not in claim and 'birth_certificate' not in claim:
            return False
        return True
    
    def _check_hash_consistency(self, claim: Dict) -> bool:
        """Check if claimed hash matches computed hash."""
        if 'hash' not in claim:
            return False
        
        claimed_hash = claim['hash']
        
        # Recompute hash from components
        encoding_str = ''
        if 'encoding' in claim:
            enc = claim['encoding']
            if isinstance(enc, ColorEncoding):
                encoding_str = enc.to_string()
            else:
                encoding_str = str(enc)
        
        path_str = ''
        if 'path_record' in claim:
            path_str = str(claim['path_record'])
        elif 'path' in claim:
            path_str = str([str(op) for op in claim['path']])
        
        theta_str = str(claim.get('theta_signature', ''))
        
        birth_str = ''
        if 'birth' in claim:
            birth = claim['birth']
            if isinstance(birth, dict):
                birth_str = birth.get('hash', str(birth))
            else:
                birth_str = str(birth)
        
        hash_data = {
            'encoding': encoding_str,
            'path': path_str,
            'theta': theta_str,
            'birth': birth_str
        }
        
        computed_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return claimed_hash == computed_hash


# =============================================================================
# WORK AND REHABILITATION
# =============================================================================

class WorkType(Enum):
    """Types of work frozen components can do to generate heat."""
    DECOMPOSITION = "decomposition"      # Help break down other hallucinations
    PATTERN_WATCH = "pattern_watch"      # Watch for similar fraud patterns
    CONTEXT_LABEL = "context_label"      # Serve as "known wrong answer"
    BRIDGE_BUILD = "bridge_build"        # Help build bridges between domains


@dataclass
class WorkAssignment:
    """Work assigned to a frozen component."""
    component_id: str
    work_type: WorkType
    assigned_at: datetime
    heat_per_unit: float  # How much heat generated per work unit
    units_completed: int = 0
    total_heat_earned: float = 0.0
    
    def do_work(self, units: int = 1) -> float:
        """Do work and earn heat."""
        heat_earned = units * self.heat_per_unit
        self.units_completed += units
        self.total_heat_earned += heat_earned
        return heat_earned


@dataclass
class MisconceptionIdentity:
    """
    When a hallucination becomes valuable AS a known wrong answer.
    
    "Eiffel Tower is in London" is USEFUL to know about.
    It doesn't cost energy to add this context - it ADDS value.
    
    The garbage develops its own identity as being wrong.
    """
    id: str
    original_claim: Dict
    misconception_type: str  # 'common_error', 'historical_myth', 'confusion', etc.
    correct_answer: Optional[str] = None
    frequency: int = 0  # How often this misconception is encountered
    educational_value: float = 0.0  # How useful is knowing this is wrong
    
    # This is NOT frozen - it's actively useful!
    temperature: float = 1.0  # Room temperature, can participate
    
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def value(self) -> float:
        """Value of this misconception (knowing what's wrong is valuable)."""
        return self.frequency * self.educational_value
    
    def encounter(self):
        """Someone encountered this misconception again."""
        self.frequency += 1
        # Gets MORE valuable the more common it is!
        self.educational_value = np.log(self.frequency + 1)


# =============================================================================
# QUARANTINE POOL (COLD STORAGE)
# =============================================================================

@dataclass
class DecompositionRecord:
    """Record of a decomposed hallucination."""
    hallucination_id: str
    detected_at: datetime
    reason: str
    original_claim: Dict
    component_ids: List[str]
    total_energy_preserved: float


@dataclass
class ThawRequest:
    """Request to thaw a frozen component."""
    component_id: str
    requester: str
    justification: str
    heat_offered: float
    timestamp: datetime = field(default_factory=datetime.now)
    approved: bool = False
    denial_reason: Optional[str] = None


class QuarantinePool:
    """
    Cold storage for decomposed hallucinations.
    Components are preserved but cannot recombine.
    
    NEW: Frozen components can WORK to earn their heat back.
    NEW: Some hallucinations become valuable misconceptions.
    
    Conservation law: Total energy in pool is constant
    (minus approved thaw withdrawals, plus work-generated heat).
    """
    
    def __init__(self):
        self.frozen_components: Dict[str, SemanticComponent] = {}
        self.decomposition_log: List[DecompositionRecord] = []
        self.thaw_requests: List[ThawRequest] = []
        self.thaw_history: List[Dict] = []
        
        # Track total energy for conservation
        self.total_energy_deposited = 0.0
        self.total_energy_withdrawn = 0.0
        self.total_energy_from_work = 0.0  # NEW: Heat generated by work
        
        # Known hallucination patterns (to prevent reassembly)
        self.known_patterns: Set[str] = set()
        
        # NEW: Work assignments for frozen components
        self.work_assignments: Dict[str, WorkAssignment] = {}
        
        # NEW: Misconceptions that have become valuable
        self.misconceptions: Dict[str, MisconceptionIdentity] = {}
        
        # NEW: Energy cost of decomposition (paid by the fraud)
        self.decomposition_energy_cost = 0.5  # Energy units per component
    
    @property
    def total_energy_stored(self) -> float:
        """Total energy in cold storage."""
        return sum(c.total_energy for c in self.frozen_components.values())
    
    @property
    def total_work_heat_available(self) -> float:
        """Total heat earned through work, available for self-thaw."""
        return sum(w.total_heat_earned for w in self.work_assignments.values())
    
    @property
    def component_count(self) -> int:
        """Number of frozen components."""
        return len(self.frozen_components)
    
    def quarantine(self, hallucination: Dict, reason: str = "failed_verification",
                   is_misconception: bool = False, 
                   misconception_context: str = None) -> List[str]:
        """
        Decompose a hallucination and freeze its parts.
        
        NEW: If is_misconception=True, create a MisconceptionIdentity instead.
        This costs NO energy because it adds value to the system.
        
        NEW: The fraud PAYS for its own decomposition through assigned work.
        """
        # Check if this should be a misconception instead
        if is_misconception:
            return self._create_misconception(hallucination, misconception_context)
        
        # Generate hallucination ID
        h_id = f"hall_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(hallucination) % 10000}"
        
        # Decompose
        components = self._decompose(hallucination, h_id)
        
        # Calculate decomposition cost (the fraud pays this)
        decomposition_cost = len(components) * self.decomposition_energy_cost
        
        # Track energy before freezing
        total_energy = sum(c.total_energy for c in components)
        
        # The fraud pays the decomposition cost from its own energy
        energy_after_cost = total_energy - decomposition_cost
        cost_per_component = decomposition_cost / len(components)
        
        frozen_ids = []
        for comp in components:
            # Deduct decomposition cost from component's energy
            comp.kinetic_energy = max(0, comp.kinetic_energy - cost_per_component)
            
            # Freeze
            comp.freeze(reason=f"from_{h_id}")
            
            # Store
            self.frozen_components[comp.id] = comp
            frozen_ids.append(comp.id)
            
            # Assign work so they can earn heat back
            self._assign_work(comp, WorkType.PATTERN_WATCH)
        
        # Record decomposition
        record = DecompositionRecord(
            hallucination_id=h_id,
            detected_at=datetime.now(),
            reason=reason,
            original_claim=hallucination,
            component_ids=frozen_ids,
            total_energy_preserved=energy_after_cost
        )
        self.decomposition_log.append(record)
        
        # Record pattern to prevent reassembly
        pattern = self._compute_pattern(hallucination)
        self.known_patterns.add(pattern)
        
        # Track energy (after cost deduction)
        self.total_energy_deposited += energy_after_cost
        
        print(f"❄️  Quarantined hallucination {h_id}")
        print(f"   Decomposed into {len(frozen_ids)} components")
        print(f"   Decomposition cost: {decomposition_cost:.4f} (paid by fraud)")
        print(f"   Energy preserved: {energy_after_cost:.4f}")
        print(f"   Work assigned: {WorkType.PATTERN_WATCH.value}")
        
        return frozen_ids
    
    def _create_misconception(self, hallucination: Dict, context: str = None) -> List[str]:
        """
        Instead of freezing, create a valuable misconception.
        This costs NO energy - it adds value!
        """
        m_id = f"misconception_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(hallucination) % 10000}"
        
        misconception = MisconceptionIdentity(
            id=m_id,
            original_claim=hallucination,
            misconception_type=context or 'common_error',
            correct_answer=None,  # To be filled in
            frequency=1,
            educational_value=1.0
        )
        
        self.misconceptions[m_id] = misconception
        
        print(f"📚 Created misconception identity: {m_id}")
        print(f"   Type: {misconception.misconception_type}")
        print(f"   Cost: FREE (adds educational value)")
        print(f"   Status: Active (can participate in responses)")
        
        return [m_id]
    
    def add_context_to_misconception(self, m_id: str, correct_answer: str):
        """
        Add the correct answer to a misconception.
        This is FREE - it makes the misconception more valuable.
        """
        if m_id not in self.misconceptions:
            return False
        
        misconception = self.misconceptions[m_id]
        misconception.correct_answer = correct_answer
        misconception.educational_value += 1.0  # More valuable now!
        
        print(f"📝 Added context to misconception {m_id}")
        print(f"   Correct answer: {correct_answer}")
        print(f"   New educational value: {misconception.educational_value:.2f}")
        print(f"   Cost: FREE")
        
        return True
    
    def _assign_work(self, component: SemanticComponent, work_type: WorkType):
        """Assign work to a frozen component so it can earn heat."""
        # Heat earned per work unit depends on work type
        heat_rates = {
            WorkType.DECOMPOSITION: 0.3,    # Helping decompose others
            WorkType.PATTERN_WATCH: 0.1,    # Watching for patterns
            WorkType.CONTEXT_LABEL: 0.2,    # Serving as known-wrong
            WorkType.BRIDGE_BUILD: 0.4,     # Building bridges
        }
        
        assignment = WorkAssignment(
            component_id=component.id,
            work_type=work_type,
            assigned_at=datetime.now(),
            heat_per_unit=heat_rates[work_type]
        )
        
        self.work_assignments[component.id] = assignment
    
    def do_work(self, component_id: str, units: int = 1) -> float:
        """
        Frozen component does work to earn heat.
        This is how frauds pay back and earn rehabilitation.
        """
        if component_id not in self.work_assignments:
            return 0.0
        
        if component_id not in self.frozen_components:
            return 0.0
        
        assignment = self.work_assignments[component_id]
        heat_earned = assignment.do_work(units)
        
        # Track work-generated heat
        self.total_energy_from_work += heat_earned
        
        print(f"⚒️  Component {component_id} did {units} unit(s) of {assignment.work_type.value}")
        print(f"   Heat earned: {heat_earned:.4f}")
        print(f"   Total earned: {assignment.total_heat_earned:.4f}")
        
        return heat_earned
    
    def self_thaw(self, component_id: str) -> Optional[SemanticComponent]:
        """
        Component uses its OWN earned heat to thaw itself.
        This is rehabilitation - they earned their freedom.
        """
        if component_id not in self.frozen_components:
            return None
        
        if component_id not in self.work_assignments:
            print(f"⚠️  No work assignment for {component_id}")
            return None
        
        assignment = self.work_assignments[component_id]
        comp = self.frozen_components[component_id]
        
        # Need at least 1.0 heat to thaw
        if assignment.total_heat_earned < 1.0:
            print(f"⚠️  Not enough heat earned: {assignment.total_heat_earned:.4f} < 1.0")
            print(f"   Keep working! Need {1.0 - assignment.total_heat_earned:.4f} more.")
            return None
        
        # Use earned heat to thaw
        heat_to_use = min(assignment.total_heat_earned, 1.5)
        success = comp.thaw(heat_to_use, "self", "earned through work")
        
        if success:
            # Track the original stored energy being withdrawn
            original_stored = comp.potential_energy  # What was frozen
            
            # Remove from cold pool
            del self.frozen_components[component_id]
            del self.work_assignments[component_id]
            
            # The stored energy is withdrawn, work heat was external addition
            self.total_energy_withdrawn += original_stored
            
            print(f"🔥 Self-thaw successful for {component_id}")
            print(f"   Used {heat_to_use:.4f} of earned heat")
            print(f"   Status: REHABILITATED")
            
            return comp
        
        return None
    
    def _decompose(self, hallucination: Dict, h_id: str) -> List[SemanticComponent]:
        """Break hallucination into constituent components."""
        components = []
        comp_idx = 0
        
        # Decompose encoding into dimension components
        if 'encoding' in hallucination:
            encoding = hallucination['encoding']
            if isinstance(encoding, ColorEncoding):
                arr = encoding.to_array()
            else:
                arr = np.array(encoding)
            
            for i, dim in enumerate(DIM_ORDER):
                if i < len(arr):
                    comp = SemanticComponent(
                        id=f"{h_id}_dim_{dim}_{comp_idx}",
                        dimension=dim,
                        value=int(arr[i]),
                        origin='decomposed_hallucination',
                        decomposed_from=h_id
                    )
                    components.append(comp)
                    comp_idx += 1
        
        # Decompose path operations
        path = hallucination.get('path') or hallucination.get('path_record', [])
        for i, op in enumerate(path):
            comp = SemanticComponent(
                id=f"{h_id}_op_{i}_{comp_idx}",
                dimension='operation',
                value=str(op),
                origin='decomposed_hallucination',
                decomposed_from=h_id
            )
            components.append(comp)
            comp_idx += 1
        
        # Decompose any theta values
        theta = hallucination.get('theta_trail', [])
        for i, t in enumerate(theta):
            comp = SemanticComponent(
                id=f"{h_id}_theta_{i}_{comp_idx}",
                dimension='theta',
                value=float(t),
                origin='decomposed_hallucination',
                decomposed_from=h_id
            )
            components.append(comp)
            comp_idx += 1
        
        # If nothing to decompose, create a single component
        if not components:
            comp = SemanticComponent(
                id=f"{h_id}_raw_{comp_idx}",
                dimension='raw',
                value=str(hallucination),
                origin='decomposed_hallucination',
                decomposed_from=h_id
            )
            components.append(comp)
        
        return components
    
    def _compute_pattern(self, hallucination: Dict) -> str:
        """Compute a pattern hash to detect reassembly attempts."""
        pattern_data = {
            'encoding': str(hallucination.get('encoding', '')),
            'structure': str(sorted(hallucination.keys()))
        }
        return hashlib.sha256(
            json.dumps(pattern_data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def request_thaw(self, component_id: str, requester: str, 
                     justification: str, heat_offered: float) -> ThawRequest:
        """
        Request to thaw a frozen component.
        Must be approved before thaw occurs.
        """
        request = ThawRequest(
            component_id=component_id,
            requester=requester,
            justification=justification,
            heat_offered=heat_offered
        )
        
        # Automatic approval checks
        if not self._verify_thaw_request(request):
            request.approved = False
        else:
            request.approved = True
        
        self.thaw_requests.append(request)
        return request
    
    def _verify_thaw_request(self, request: ThawRequest) -> bool:
        """
        Verify that thaw request is legitimate.
        Prevents malicious reassembly of hallucinations.
        """
        # Check 1: Component exists
        if request.component_id not in self.frozen_components:
            request.denial_reason = "component_not_found"
            return False
        
        # Check 2: Justification is substantial
        if len(request.justification) < 10:
            request.denial_reason = "insufficient_justification"
            return False
        
        # Check 3: Heat offered is positive
        if request.heat_offered <= 0:
            request.denial_reason = "no_heat_offered"
            return False
        
        # Check 4: Would this reassemble a known hallucination?
        # (Check if multiple components from same hallucination are being thawed)
        comp = self.frozen_components[request.component_id]
        if comp.decomposed_from:
            # Count how many from same source are being thawed
            same_source_requests = [
                r for r in self.thaw_requests
                if r.approved and r.component_id in self.frozen_components
                and self.frozen_components[r.component_id].decomposed_from == comp.decomposed_from
            ]
            if len(same_source_requests) > 3:
                request.denial_reason = "potential_reassembly_detected"
                return False
        
        return True
    
    def execute_thaw(self, request: ThawRequest) -> Optional[SemanticComponent]:
        """
        Execute an approved thaw request.
        Returns the thawed component, or None if failed.
        """
        if not request.approved:
            print(f"⚠️  Thaw denied: {request.denial_reason}")
            return None
        
        if request.component_id not in self.frozen_components:
            print(f"⚠️  Component not found: {request.component_id}")
            return None
        
        comp = self.frozen_components[request.component_id]
        
        # Thaw the component
        success = comp.thaw(request.heat_offered, request.requester, request.justification)
        
        if success:
            # Track energy - the stored energy plus heat added
            energy_released = comp.total_energy
            
            # Remove from cold pool
            del self.frozen_components[request.component_id]
            
            # Track energy withdrawal (only the original stored energy, not heat added)
            # Heat added is external, stored energy was deposited
            original_energy = energy_released - request.heat_offered
            self.total_energy_withdrawn += max(0, original_energy)
            
            # Log
            self.thaw_history.append({
                'timestamp': datetime.now().isoformat(),
                'component_id': request.component_id,
                'requester': request.requester,
                'justification': request.justification,
                'heat_added': request.heat_offered,
                'energy_withdrawn': original_energy
            })
            
            print(f"🔥 Thawed component {request.component_id}")
            print(f"   Requester: {request.requester}")
            print(f"   Original energy: {original_energy:.4f}")
            print(f"   Heat added: {request.heat_offered:.4f}")
            
            return comp
        else:
            print(f"⚠️  Thaw failed: insufficient heat")
            return None
    
    def get_pool_status(self) -> Dict:
        """Get current status of the quarantine pool."""
        return {
            'component_count': self.component_count,
            'total_energy_stored': self.total_energy_stored,
            'total_energy_deposited': self.total_energy_deposited,
            'total_energy_withdrawn': self.total_energy_withdrawn,
            'total_energy_from_work': self.total_energy_from_work,
            'total_work_heat_available': self.total_work_heat_available,
            'energy_conservation_check': abs(
                self.total_energy_deposited - self.total_energy_withdrawn - self.total_energy_stored
            ) < 0.1,  # Allow small tolerance
            'decomposition_count': len(self.decomposition_log),
            'known_patterns': len(self.known_patterns),
            'pending_thaw_requests': len([r for r in self.thaw_requests if not r.approved]),
            'active_work_assignments': len(self.work_assignments),
            'misconceptions_created': len(self.misconceptions),
            'total_misconception_value': sum(m.value for m in self.misconceptions.values())
        }
    
    def display_status(self):
        """Print pool status."""
        status = self.get_pool_status()
        
        print("\n" + "=" * 60)
        print("❄️  QUARANTINE POOL STATUS")
        print("=" * 60)
        print(f"   Frozen components: {status['component_count']}")
        print(f"   Energy stored: {status['total_energy_stored']:.4f}")
        print(f"   Energy deposited: {status['total_energy_deposited']:.4f}")
        print(f"   Energy withdrawn: {status['total_energy_withdrawn']:.4f}")
        print(f"   Energy from work: {status['total_energy_from_work']:.4f}")
        print(f"   Conservation check: {'✓' if status['energy_conservation_check'] else '✗'}")
        print("-" * 60)
        print(f"   Work assignments: {status['active_work_assignments']}")
        print(f"   Work heat available: {status['total_work_heat_available']:.4f}")
        print("-" * 60)
        print(f"   Misconceptions: {status['misconceptions_created']}")
        print(f"   Misconception value: {status['total_misconception_value']:.4f}")
        print("-" * 60)
        print(f"   Decompositions: {status['decomposition_count']}")
        print(f"   Known patterns: {status['known_patterns']}")
        print(f"   Pending thaws: {status['pending_thaw_requests']}")
        print("=" * 60)


# =============================================================================
# VERIFIED SEMANTIC WORD (WITH TEMPERATURE)
# =============================================================================

class VerifiedWord(HamiltonianWord):
    """
    A word that includes verification and temperature.
    Can detect and quarantine hallucinations.
    """
    
    def __init__(self, word: str, pde: SemanticPDE, quarantine: QuarantinePool):
        super().__init__(word, pde)
        self.quarantine = quarantine
        self.detector = HallucinationDetector()
        
        # Temperature for the word itself
        self.temperature = SemanticTemperature(temperature=1.0)
        
        # Birth certificate
        self.birth_certificate = self._create_birth_certificate()
        
        # Hash of current state
        self.current_hash = self._compute_hash()
    
    def _create_birth_certificate(self) -> Dict:
        """Create unforgeable birth certificate."""
        birth_data = {
            'word': self.word,
            'timestamp': datetime.now().isoformat(),
            'initial_encoding': self.current_encoding.to_string(),
            'initial_frequency': self.wave_function.eigenfunctions[0].frequency,
            'h_info': H_INFO
        }
        birth_hash = hashlib.sha256(
            json.dumps(birth_data, sort_keys=True).encode()
        ).hexdigest()
        
        return {
            'data': birth_data,
            'hash': birth_hash
        }
    
    def _compute_hash(self) -> str:
        """Compute hash of current state."""
        hash_data = {
            'encoding': self.current_encoding.to_string(),
            'path': str([op.describe() for op in self.navigator.path]),
            'theta': str(self.navigator.get_total_theta()),
            'birth': self.birth_certificate['hash']
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def get_verifiable_claim(self) -> Dict:
        """Get current state as a verifiable claim."""
        path_record = [op.describe() for op in self.navigator.path]
        theta_sig = self.navigator.get_total_theta()
        
        # Ensure hash is computed with same format as verification
        return {
            'label': self.word,
            'encoding': self.current_encoding,
            'path': self.navigator.path,
            'path_record': path_record,
            'theta_signature': theta_sig,
            'theta_trail': self.navigator.theta_trail,
            'energy_history': self.hamiltonian.energy_history,
            'birth': self.birth_certificate,
            'birth_certificate': self.birth_certificate,
            'hash': self.current_hash
        }
    
    def verify_self(self) -> VerificationResult:
        """Verify this word is not a hallucination."""
        claim = self.get_verifiable_claim()
        return self.detector.verify(claim)
    
    def verify_external(self, claim: Dict) -> Tuple[bool, Optional[List[str]]]:
        """
        Verify an external claim.
        If hallucination detected, quarantine it.
        Returns (is_valid, frozen_component_ids or None)
        """
        result = self.detector.verify(claim)
        
        if not result.is_valid:
            print(f"🚨 Hallucination detected!")
            print(f"   Failed checks: {result.checks_failed}")
            print(f"   Confidence: {result.confidence:.2f}")
            
            # Quarantine
            frozen_ids = self.quarantine.quarantine(
                claim, 
                reason=f"failed: {', '.join(result.checks_failed)}"
            )
            
            return False, frozen_ids
        
        return True, None
    
    def _update_hash(self):
        """Update hash after any operation."""
        self.current_hash = self._compute_hash()
    
    # Override operations to update hash
    def expand(self, context: str = "") -> float:
        theta = super().expand(context)
        self._update_hash()
        return theta
    
    def contract(self, context: str = "") -> float:
        theta = super().contract(context)
        self._update_hash()
        return theta
    
    def add_positive(self, context: str = "") -> float:
        theta = super().add_positive(context)
        self._update_hash()
        return theta
    
    def add_negative(self, context: str = "") -> float:
        theta = super().add_negative(context)
        self._update_hash()
        return theta
    
    def level_up(self, context: str = "") -> float:
        theta = super().level_up(context)
        self._update_hash()
        return theta
    
    def level_down(self, context: str = "") -> float:
        theta = super().level_down(context)
        self._update_hash()
        return theta


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_quarantine_system():
    """Demonstrate the full quarantine system."""
    
    print("\n" + "=" * 70)
    print("SEMANTIC QUARANTINE SYSTEM DEMONSTRATION")
    print("Conservation-Preserving Hallucination Handling")
    print("With Work-Based Rehabilitation & Misconception Identity")
    print("=" * 70)
    
    # Create systems
    pde = SemanticPDE()
    quarantine = QuarantinePool()
    
    # Create a verified word
    print("\n" + "-" * 70)
    print("PHASE 1: Create verified word 'truth'")
    print("-" * 70)
    
    truth = VerifiedWord("truth", pde, quarantine)
    truth.expand("seeking clarity")
    truth.add_positive("aligned with reality")
    truth.level_up("philosophical depth")
    
    # Verify it
    result = truth.verify_self()
    print(f"\n✓ Self-verification: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"  Checks passed: {result.checks_passed}")
    print(f"  Confidence: {result.confidence:.2f}")
    
    # Create a MALICIOUS hallucination (deliberate fraud)
    print("\n" + "-" * 70)
    print("PHASE 2: Detect FRAUD and quarantine (pays decomposition cost)")
    print("-" * 70)
    
    fraud = {
        'label': 'fake_credential',
        'encoding': ColorEncoding(1, 2, 1, 3, 2, 0, 2),
        'path': [],  # NO PATH! Fraud indicator
        'path_record': [],
        'theta_signature': (1.5, 0.8, 0.3),  # Made up
        'theta_trail': [],
        'hash': 'fakehash12345678'
    }
    
    print("\nIncoming FRAUD: 'fake_credential'")
    print("   This is a deliberate attempt to bypass verification")
    
    is_valid, frozen_ids = truth.verify_external(fraud)
    
    # Create a MISCONCEPTION (common error, not fraud)
    print("\n" + "-" * 70)
    print("PHASE 3: Handle MISCONCEPTION (FREE - adds value)")
    print("-" * 70)
    
    misconception = {
        'label': 'eiffel_in_london',
        'encoding': ColorEncoding(1, 1, 0, 3, 1, 0, 1),
        'claim': 'The Eiffel Tower is in London',
        'path': [],
        'hash': 'misconception123'
    }
    
    print("\nIncoming: 'The Eiffel Tower is in London'")
    print("   This is a common misconception, not deliberate fraud")
    print("   Creating misconception identity (FREE)...")
    
    m_ids = quarantine.quarantine(
        misconception, 
        reason="common_error",
        is_misconception=True,
        misconception_context="geographic_confusion"
    )
    
    # Add context to the misconception (also FREE)
    print("\n   Adding correct answer...")
    quarantine.add_context_to_misconception(
        m_ids[0], 
        correct_answer="The Eiffel Tower is in Paris, France"
    )
    
    # Show that misconceptions are valuable
    print(f"\n   Misconception value: {quarantine.misconceptions[m_ids[0]].value:.2f}")
    print(f"   Status: ACTIVE (can help prevent future errors)")
    
    # Check quarantine status
    print("\n" + "-" * 70)
    print("PHASE 4: Quarantine pool status")
    print("-" * 70)
    
    quarantine.display_status()
    
    # Show frozen components with work assignments
    print("\nFrozen components (with work assignments):")
    for comp_id, comp in list(quarantine.frozen_components.items())[:3]:
        work = quarantine.work_assignments.get(comp_id)
        work_type = work.work_type.value if work else "none"
        heat_earned = work.total_heat_earned if work else 0
        print(f"  {comp_id[:40]}...")
        print(f"    T={comp.temperature.temperature}, E={comp.total_energy:.4f}")
        print(f"    Work: {work_type}, Heat earned: {heat_earned:.4f}")
    
    # FRAUD DOES WORK TO EARN REHABILITATION
    print("\n" + "-" * 70)
    print("PHASE 5: Fraud does WORK to earn heat back")
    print("-" * 70)
    
    if frozen_ids:
        comp_id = frozen_ids[0]
        print(f"\nComponent {comp_id[:40]}... must work for rehabilitation")
        
        # Do work multiple times
        for i in range(12):
            quarantine.do_work(comp_id, units=1)
        
        # Check if can self-thaw
        print(f"\n   Attempting self-thaw (earned heat)...")
        thawed = quarantine.self_thaw(comp_id)
        
        if thawed:
            print(f"   ✓ REHABILITATED through work!")
            print(f"   Component can now bind: {thawed.can_bind}")
        else:
            assignment = quarantine.work_assignments.get(comp_id)
            if assignment:
                print(f"   Need more work. Heat: {assignment.total_heat_earned:.2f}/1.0")
    
    # External thaw request (someone else pays)
    print("\n" + "-" * 70)
    print("PHASE 6: External thaw request (someone else provides heat)")
    print("-" * 70)
    
    if len(frozen_ids) > 1:
        comp_id = frozen_ids[1]
        print(f"\nExternal request to thaw: {comp_id[:40]}...")
        
        request = quarantine.request_thaw(
            component_id=comp_id,
            requester="legitimate_system",
            justification="Need this dimension value for valid computation",
            heat_offered=1.5
        )
        
        print(f"  Approved: {request.approved}")
        if request.approved:
            thawed = quarantine.execute_thaw(request)
            if thawed:
                print(f"  Thawed via external heat")
    
    # Final status
    print("\n" + "-" * 70)
    print("PHASE 7: Final status")
    print("-" * 70)
    
    quarantine.display_status()
    
    # Show misconceptions
    print("\n📚 Active Misconceptions (valuable knowledge):")
    for m_id, m in quarantine.misconceptions.items():
        print(f"   {m_id}:")
        print(f"      Claim: {m.original_claim.get('claim', 'unknown')}")
        print(f"      Correct: {m.correct_answer}")
        print(f"      Value: {m.value:.2f}")
        print(f"      Temperature: {m.temperature} (ACTIVE, not frozen)")
    
    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY: Two Types of Wrong Answers")
    print("-" * 70)
    print("""
    FRAUD (deliberate deception):
      → Detected and quarantined
      → PAYS decomposition cost from own energy
      → Assigned WORK to earn heat back
      → Can self-thaw after enough work (rehabilitation)
      → Components preserved but frozen
    
    MISCONCEPTION (honest error):
      → Recognized as common mistake
      → Costs NOTHING to create identity
      → ADDS VALUE as "known wrong answer"
      → Stays ACTIVE (not frozen)
      → Helps prevent future errors
      → Gets MORE valuable when encountered more
    """)
    
    return pde, quarantine, truth


if __name__ == "__main__":
    pde, quarantine, truth = demonstrate_quarantine_system()
