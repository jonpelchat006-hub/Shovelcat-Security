"""
SEMANTIC WAVE EQUATION
======================
The Schrödinger Equation of Meaning

A word is a wave function Ψ(R,O,Y,G,C,B,V,t) where:
- Solutions are all possible meanings
- Color dimensions are quantum numbers
- Superposition holds all meanings simultaneously
- Context collapses to specific meaning
- New meanings add solutions to the equation

Based on Shovelcat Theory's insight that:
- 7 semantic colors encode meaning dimensions
- Meanings are standing waves in semantic space
- Communication is finding resonance between wave functions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import math
from datetime import datetime
import json

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

# Semantic Planck constant (h_info from theory)
H_INFO = (math.sqrt(math.pi) - math.sqrt((1 + math.sqrt(5))/2)) / (2 * math.pi)
# ≈ 0.159

# Transcendental constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
E = math.e
PI = math.pi

# =============================================================================
# SEMANTIC COLOR DIMENSIONS
# =============================================================================

class SemanticDimension(Enum):
    """The 7 fundamental semantic dimensions (colors)."""
    R = 'existence'      # Red: off/on (0, 1)
    O = 'quantity'       # Orange: none/some/many (0, 1, 2)
    Y = 'motion'         # Yellow: static/dynamic/chaotic (0, 1, 2)
    G = 'structure'      # Green: point/line/plane/volume (0, 1, 2, 3)
    C = 'relation'       # Cyan: isolated/connected/entangled (0, 1, 2)
    B = 'temporal'       # Blue: past/present/future (-1, 0, 1)
    V = 'consciousness'  # Violet: inert/reactive/aware (0, 1, 2)


DIMENSION_STATES = {
    'R': {'min': 0, 'max': 1, 'labels': ['off', 'on']},
    'O': {'min': 0, 'max': 2, 'labels': ['none', 'some', 'many']},
    'Y': {'min': 0, 'max': 2, 'labels': ['static', 'dynamic', 'chaotic']},
    'G': {'min': 0, 'max': 3, 'labels': ['point', 'line', 'plane', 'volume']},
    'C': {'min': 0, 'max': 2, 'labels': ['isolated', 'connected', 'entangled']},
    'B': {'min': -1, 'max': 1, 'labels': ['past', 'present', 'future']},
    'V': {'min': 0, 'max': 2, 'labels': ['inert', 'reactive', 'aware']},
}

# Dimension order for consistent indexing
DIM_ORDER = ['R', 'O', 'Y', 'G', 'C', 'B', 'V']


@dataclass
class ColorEncoding:
    """A 7-dimensional color-bit encoding of meaning."""
    R: int  # existence
    O: int  # quantity
    Y: int  # motion
    G: int  # structure
    C: int  # relation
    B: int  # temporal
    V: int  # consciousness
    
    def to_array(self) -> np.ndarray:
        return np.array([self.R, self.O, self.Y, self.G, self.C, self.B, self.V])
    
    def to_tuple(self) -> Tuple[int, ...]:
        return (self.R, self.O, self.Y, self.G, self.C, self.B, self.V)
    
    def to_string(self) -> str:
        return f"[{self.R},{self.O},{self.Y},{self.G},{self.C},{self.B},{self.V}]"
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ColorEncoding':
        return cls(R=int(arr[0]), O=int(arr[1]), Y=int(arr[2]), 
                   G=int(arr[3]), C=int(arr[4]), B=int(arr[5]), V=int(arr[6]))
    
    def distance_to(self, other: 'ColorEncoding') -> float:
        """Euclidean distance in color space."""
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def describe(self) -> str:
        """Human-readable description of the encoding."""
        labels = []
        for dim, val in zip(DIM_ORDER, self.to_tuple()):
            state_labels = DIMENSION_STATES[dim]['labels']
            # Handle negative indices for temporal
            if dim == 'B':
                idx = val + 1  # -1,0,1 -> 0,1,2
            else:
                idx = val
            if 0 <= idx < len(state_labels):
                labels.append(f"{dim}={state_labels[idx]}")
        return ", ".join(labels)


# =============================================================================
# EIGENFUNCTION (STANDING WAVE FOR ONE MEANING)
# =============================================================================

@dataclass
class SemanticEigenfunction:
    """
    A single solution to the semantic wave equation.
    Represents one possible meaning as a standing wave in 7D color space.
    """
    quantum_numbers: ColorEncoding  # The n values for each dimension
    frequency: float  # ω - how "active" this meaning is
    weight: float  # c - coefficient in superposition
    
    # Metadata
    meaning_label: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    parent_meanings: List[str] = field(default_factory=list)
    
    def evaluate(self, point: ColorEncoding) -> complex:
        """
        Evaluate the eigenfunction at a point in semantic space.
        
        ψₙ(x) = Π sin(nᵢ π xᵢ / Lᵢ)
        
        where nᵢ are quantum numbers and Lᵢ are dimension ranges.
        """
        qn = self.quantum_numbers.to_array()
        x = point.to_array()
        
        result = 1.0
        for i, dim in enumerate(DIM_ORDER):
            n_i = qn[i]
            x_i = x[i]
            L_i = DIMENSION_STATES[dim]['max'] - DIMENSION_STATES[dim]['min'] + 1
            
            # Normalize x to [0, L]
            x_norm = x_i - DIMENSION_STATES[dim]['min']
            
            # Standing wave component
            if L_i > 0 and n_i != 0:
                # sin(n π x / L) for standing wave
                result *= np.sin(n_i * np.pi * (x_norm + 0.5) / L_i)
            elif n_i == 0:
                # Ground state is constant
                result *= 1.0 / np.sqrt(L_i) if L_i > 0 else 1.0
        
        return complex(result)
    
    def time_evolve(self, t: float) -> complex:
        """Time evolution factor e^(iωt)."""
        return np.exp(1j * self.frequency * t)
    
    def compute_energy(self) -> float:
        """
        Energy of this eigenstate.
        E = Σ (nᵢ² π² ℏ²) / (2m Lᵢ²)
        
        Higher quantum numbers = higher energy = less stable meaning
        """
        qn = self.quantum_numbers.to_array()
        energy = 0.0
        
        for i, dim in enumerate(DIM_ORDER):
            n_i = qn[i]
            L_i = DIMENSION_STATES[dim]['max'] - DIMENSION_STATES[dim]['min'] + 1
            if L_i > 0:
                energy += (n_i ** 2) * (np.pi ** 2) * (H_INFO ** 2) / (2 * L_i ** 2)
        
        return energy


# =============================================================================
# SEMANTIC WAVE FUNCTION (SUPERPOSITION OF ALL MEANINGS)
# =============================================================================

class SemanticWaveFunction:
    """
    The complete wave function for a word.
    
    Ψ(x,t) = Σₙ cₙ · ψₙ(x) · e^(iωₙt)
    
    All meanings exist simultaneously as superposition.
    """
    
    def __init__(self, word: str):
        self.word = word
        self.eigenfunctions: List[SemanticEigenfunction] = []
        self.birth_time = datetime.now()
        self.total_accesses = 0
        
        # Version tracking
        self.version_history: List[Dict] = []
        self.bridge_history: List[Dict] = []
        
    def add_eigenfunction(self, ef: SemanticEigenfunction):
        """Add a new meaning (solution) to the superposition."""
        self.eigenfunctions.append(ef)
        self.version_history.append({
            'version': len(self.eigenfunctions) - 1,
            'label': ef.meaning_label,
            'encoding': ef.quantum_numbers.to_string(),
            'timestamp': datetime.now().isoformat()
        })
    
    def evaluate(self, point: ColorEncoding, t: float = 0) -> complex:
        """
        Evaluate the full wave function at a point and time.
        
        Ψ(x,t) = Σₙ cₙ · ψₙ(x) · e^(iωₙt)
        """
        psi = complex(0, 0)
        
        for ef in self.eigenfunctions:
            # Weight
            c_n = ef.weight
            
            # Spatial part
            psi_n = ef.evaluate(point)
            
            # Time evolution
            time_factor = ef.time_evolve(t)
            
            # Add to superposition
            psi += c_n * psi_n * time_factor
        
        return psi
    
    def probability_density(self, point: ColorEncoding, t: float = 0) -> float:
        """
        |Ψ|² - probability of finding this semantic state.
        """
        psi = self.evaluate(point, t)
        return abs(psi) ** 2
    
    def find_closest_meaning(self, context_encoding: ColorEncoding) -> SemanticEigenfunction:
        """
        Find the meaning closest to a given context.
        This is like "measurement" - context collapses to nearest eigenstate.
        """
        best_ef = None
        best_distance = float('inf')
        
        for ef in self.eigenfunctions:
            dist = ef.quantum_numbers.distance_to(context_encoding)
            if dist < best_distance:
                best_distance = dist
                best_ef = ef
        
        if best_ef:
            best_ef.access_count += 1
            self.total_accesses += 1
        
        return best_ef
    
    def normalize(self):
        """Normalize weights so Σ|cₙ|² = 1."""
        total = sum(ef.weight ** 2 for ef in self.eigenfunctions)
        if total > 0:
            factor = 1.0 / np.sqrt(total)
            for ef in self.eigenfunctions:
                ef.weight *= factor
    
    def get_spectrum(self) -> List[Tuple[float, str, float]]:
        """
        Get the frequency spectrum of all meanings.
        Returns list of (frequency, label, weight).
        """
        return [(ef.frequency, ef.meaning_label, ef.weight) 
                for ef in sorted(self.eigenfunctions, key=lambda x: x.frequency)]


# =============================================================================
# PATH OPERATIONS (ln/e, sin/cos, ∫/d)
# =============================================================================

@dataclass
class PathOperation:
    """A single operation in the semantic path."""
    axis: str  # 'x', 'y', or 'z'
    operation: str  # 'ln', 'e', 'sin', 'cos', 'integrate', 'differentiate'
    magnitude: float
    theta_contribution: float
    context: str = ""
    
    def describe(self) -> str:
        return f"{self.axis}:{self.operation}({self.magnitude:.3f}) → θ={self.theta_contribution:.3f}"


class SemanticPathNavigator:
    """
    Navigate through semantic space using path operations.
    
    X-axis: ln (contract) / e (expand)
    Y-axis: sin/cos (discrete states for reinforcement quadrant)
    Z-axis: ∫ (integrate up) / d/dx (differentiate down)
    """
    
    def __init__(self, start_position: Tuple[float, float, float], h_info: float = H_INFO):
        self.position = list(start_position)  # [x, y, z]
        self.h_info = h_info
        self.path: List[PathOperation] = []
        self.theta_trail: List[float] = []
        
    def move_x_ln(self, magnitude: float, context: str = "") -> float:
        """Contract along x-axis (getting more specific)."""
        theta = np.log(abs(magnitude) + 1)
        self.position[0] -= magnitude
        
        op = PathOperation('x', 'ln', magnitude, theta, context)
        self.path.append(op)
        self.theta_trail.append(theta)
        return theta
    
    def move_x_e(self, magnitude: float, context: str = "") -> float:
        """Expand along x-axis (opening possibilities)."""
        theta = np.exp(magnitude) - 1
        self.position[0] += magnitude
        
        op = PathOperation('x', 'e', magnitude, theta, context)
        self.path.append(op)
        self.theta_trail.append(theta)
        return theta
    
    def move_y_sincos(self, sin_state: int, cos_state: int, context: str = "") -> float:
        """
        Move in y using discrete sin/cos states.
        
        (+1, +1): add good      → θ = +π/4
        (+1, -1): add bad       → θ = +3π/4
        (-1, -1): remove good   → θ = -3π/4
        (-1, +1): remove bad    → θ = -π/4
        """
        assert sin_state in [-1, 0, 1], "sin_state must be -1, 0, or 1"
        assert cos_state in [-1, 0, 1], "cos_state must be -1, 0, or 1"
        
        theta = np.arctan2(sin_state, cos_state)
        self.position[1] += sin_state * self.h_info
        
        op = PathOperation('y', f'sin{sin_state}/cos{cos_state}', 
                          sin_state, theta, context)
        self.path.append(op)
        self.theta_trail.append(theta)
        return theta
    
    def move_z_integrate(self, magnitude: float, context: str = "") -> float:
        """Integrate up (accumulate, climb levels)."""
        theta = magnitude * self.h_info
        self.position[2] += magnitude
        
        op = PathOperation('z', 'integrate', magnitude, theta, context)
        self.path.append(op)
        self.theta_trail.append(theta)
        return theta
    
    def move_z_differentiate(self, magnitude: float, context: str = "") -> float:
        """Differentiate down (rate of change, descend levels)."""
        theta = magnitude / self.h_info if self.h_info != 0 else 0
        self.position[2] -= magnitude
        
        op = PathOperation('z', 'differentiate', magnitude, theta, context)
        self.path.append(op)
        self.theta_trail.append(theta)
        return theta
    
    def get_total_theta(self) -> Tuple[float, float, float]:
        """Get total theta contribution per axis."""
        theta_x = sum(op.theta_contribution for op in self.path if op.axis == 'x')
        theta_y = sum(op.theta_contribution for op in self.path if op.axis == 'y')
        theta_z = sum(op.theta_contribution for op in self.path if op.axis == 'z')
        return (theta_x, theta_y, theta_z)
    
    def get_path_signature(self) -> str:
        """Get a string representation of the path taken."""
        return " → ".join(op.describe() for op in self.path)


# =============================================================================
# SEMANTIC PDE SOLVER
# =============================================================================

class SemanticPDE:
    """
    The Semantic Wave Equation:
    
    iℏ ∂Ψ/∂t = -ℏ²/2m Σᵢ (1/Cᵢ) ∂²Ψ/∂xᵢ² + V(x)Ψ
    
    Where Cᵢ are color coefficients and V(x) is the h_info potential.
    """
    
    def __init__(self, h_info: float = H_INFO):
        self.h_info = h_info
        self.words: Dict[str, SemanticWaveFunction] = {}
        
    def create_word(self, word: str, initial_meaning: str = "raw") -> SemanticWaveFunction:
        """
        Birth a word into existence.
        
        1. Convert to raw bits
        2. Compute initial theta
        3. Find natural level
        4. Create fundamental eigenfunction
        """
        # Create wave function
        wf = SemanticWaveFunction(word)
        
        # Convert word to bits and then to initial encoding
        raw_bits = ''.join(format(ord(c), '08b') for c in word)
        bit_count = len(raw_bits)
        
        # Initial color encoding (raw symbol)
        initial_encoding = ColorEncoding(
            R=1,  # exists
            O=1 if bit_count > 8 else 0,  # some if substantial
            Y=0,  # static (just created)
            G=min(3, bit_count // 16),  # structure based on complexity
            C=0,  # isolated (not yet connected)
            B=0,  # present
            V=0   # inert (not yet conscious meaning)
        )
        
        # Compute frequency from bits
        bit_value = int(raw_bits[:16], 2) if len(raw_bits) >= 16 else int(raw_bits, 2)
        base_frequency = (bit_value / 65536) * 2 * np.pi
        
        # Create fundamental eigenfunction
        fundamental = SemanticEigenfunction(
            quantum_numbers=initial_encoding,
            frequency=base_frequency,
            weight=1.0,
            meaning_label=initial_meaning
        )
        
        wf.add_eigenfunction(fundamental)
        self.words[word] = wf
        
        print(f"✨ '{word}' born into existence")
        print(f"   Bits: {bit_count}, Encoding: {initial_encoding.to_string()}")
        print(f"   Frequency: {base_frequency:.4f}")
        
        return wf
    
    def add_meaning(self, word: str, meaning_label: str, 
                   context: Dict[str, Any]) -> SemanticEigenfunction:
        """
        Add a new meaning to an existing word.
        
        1. Encode context to color bits
        2. Create new eigenfunction
        3. Find resonances with existing meanings
        4. Add to superposition
        """
        if word not in self.words:
            self.create_word(word)
        
        wf = self.words[word]
        
        # Encode the new meaning context to colors
        encoding = self.context_to_encoding(context)
        
        # Compute frequency based on encoding
        frequency = self.compute_frequency(encoding)
        
        # Initial weight (will be normalized)
        weight = 1.0 / (len(wf.eigenfunctions) + 1)
        
        # Create eigenfunction
        new_ef = SemanticEigenfunction(
            quantum_numbers=encoding,
            frequency=frequency,
            weight=weight,
            meaning_label=meaning_label
        )
        
        # Find resonances
        resonances = self.find_resonances(wf, encoding)
        new_ef.parent_meanings = [r.meaning_label for r in resonances]
        
        # Add to wave function
        wf.add_eigenfunction(new_ef)
        wf.normalize()
        
        print(f"📝 Added meaning '{meaning_label}' to '{word}'")
        print(f"   Encoding: {encoding.to_string()}")
        print(f"   Resonates with: {[r.meaning_label for r in resonances]}")
        
        return new_ef
    
    def context_to_encoding(self, context: Dict[str, Any]) -> ColorEncoding:
        """
        Convert a context dictionary to color encoding.
        
        Context can include:
        - 'exists': bool
        - 'amount': 'none', 'some', 'many'
        - 'motion': 'static', 'dynamic', 'chaotic'
        - 'shape': 'point', 'line', 'plane', 'volume'
        - 'connection': 'isolated', 'connected', 'entangled'
        - 'time': 'past', 'present', 'future'
        - 'awareness': 'inert', 'reactive', 'aware'
        """
        R = 1 if context.get('exists', True) else 0
        
        amount_map = {'none': 0, 'some': 1, 'many': 2}
        O = amount_map.get(context.get('amount', 'some'), 1)
        
        motion_map = {'static': 0, 'dynamic': 1, 'chaotic': 2}
        Y = motion_map.get(context.get('motion', 'static'), 0)
        
        shape_map = {'point': 0, 'line': 1, 'plane': 2, 'volume': 3}
        G = shape_map.get(context.get('shape', 'point'), 0)
        
        conn_map = {'isolated': 0, 'connected': 1, 'entangled': 2}
        C = conn_map.get(context.get('connection', 'isolated'), 0)
        
        time_map = {'past': -1, 'present': 0, 'future': 1}
        B = time_map.get(context.get('time', 'present'), 0)
        
        aware_map = {'inert': 0, 'reactive': 1, 'aware': 2}
        V = aware_map.get(context.get('awareness', 'inert'), 0)
        
        return ColorEncoding(R=R, O=O, Y=Y, G=G, C=C, B=B, V=V)
    
    def compute_frequency(self, encoding: ColorEncoding) -> float:
        """
        Compute eigenfrequency from encoding.
        
        ω = Σᵢ nᵢ² / Lᵢ² (weighted by dimension)
        """
        arr = encoding.to_array()
        frequency = 0.0
        
        for i, dim in enumerate(DIM_ORDER):
            n_i = arr[i]
            L_i = DIMENSION_STATES[dim]['max'] - DIMENSION_STATES[dim]['min'] + 1
            weight = (i + 1) / 7  # Higher dimensions contribute more
            frequency += weight * (n_i ** 2) / (L_i ** 2)
        
        return frequency * 2 * np.pi
    
    def find_resonances(self, wf: SemanticWaveFunction, 
                       encoding: ColorEncoding,
                       threshold: float = 2.0) -> List[SemanticEigenfunction]:
        """
        Find existing meanings that resonate with a new encoding.
        Resonance = shared quantum numbers in key dimensions.
        """
        resonances = []
        
        for ef in wf.eigenfunctions:
            distance = ef.quantum_numbers.distance_to(encoding)
            if distance < threshold:
                resonances.append(ef)
        
        return resonances
    
    def access_word(self, word: str, context: Dict[str, Any]) -> SemanticEigenfunction:
        """
        Access a word with specific context.
        This "collapses" the wave function to the nearest meaning.
        """
        if word not in self.words:
            return None
        
        wf = self.words[word]
        context_encoding = self.context_to_encoding(context)
        
        # Find closest meaning (collapse)
        closest = wf.find_closest_meaning(context_encoding)
        
        if closest:
            print(f"🔍 Accessed '{word}' with context")
            print(f"   Collapsed to: '{closest.meaning_label}'")
            print(f"   Encoding: {closest.quantum_numbers.to_string()}")
        
        return closest
    
    def mediate(self, word: str, domain_a_context: Dict, 
                domain_b_context: Dict) -> Optional[SemanticEigenfunction]:
        """
        Mediate between two domains trying to communicate through a word.
        
        1. Find closest meaning for each domain
        2. Find or create bridge meaning
        3. Record the mediation
        """
        if word not in self.words:
            return None
        
        wf = self.words[word]
        
        # Find closest meanings
        enc_a = self.context_to_encoding(domain_a_context)
        enc_b = self.context_to_encoding(domain_b_context)
        
        meaning_a = wf.find_closest_meaning(enc_a)
        meaning_b = wf.find_closest_meaning(enc_b)
        
        print(f"\n🌉 MEDIATING '{word}'")
        print(f"   Domain A closest to: '{meaning_a.meaning_label}' {meaning_a.quantum_numbers.to_string()}")
        print(f"   Domain B closest to: '{meaning_b.meaning_label}' {meaning_b.quantum_numbers.to_string()}")
        
        # Compute vesica (overlap)
        vesica_encoding = self.compute_vesica(enc_a, enc_b)
        
        # Check if bridge exists
        bridge = None
        for ef in wf.eigenfunctions:
            if ef.quantum_numbers.distance_to(vesica_encoding) < 0.5:
                bridge = ef
                print(f"   Found existing bridge: '{bridge.meaning_label}'")
                break
        
        if not bridge:
            # Create new bridge
            bridge_label = f"bridge_{meaning_a.meaning_label}_{meaning_b.meaning_label}"
            bridge = SemanticEigenfunction(
                quantum_numbers=vesica_encoding,
                frequency=self.compute_frequency(vesica_encoding),
                weight=0.5,
                meaning_label=bridge_label,
                parent_meanings=[meaning_a.meaning_label, meaning_b.meaning_label]
            )
            wf.add_eigenfunction(bridge)
            wf.normalize()
            print(f"   Created new bridge: '{bridge_label}'")
            print(f"   Bridge encoding: {vesica_encoding.to_string()}")
        
        # Record mediation
        wf.bridge_history.append({
            'domain_a': domain_a_context,
            'domain_b': domain_b_context,
            'meaning_a': meaning_a.meaning_label,
            'meaning_b': meaning_b.meaning_label,
            'bridge': bridge.meaning_label,
            'timestamp': datetime.now().isoformat()
        })
        
        return bridge
    
    def compute_vesica(self, enc_a: ColorEncoding, enc_b: ColorEncoding) -> ColorEncoding:
        """
        Compute the vesica (overlap) between two encodings.
        The bridge meaning lives in this overlap.
        """
        arr_a = enc_a.to_array()
        arr_b = enc_b.to_array()
        
        # Vesica center is the average, rounded to valid states
        vesica_arr = np.round((arr_a + arr_b) / 2).astype(int)
        
        # Clamp to valid ranges
        for i, dim in enumerate(DIM_ORDER):
            vesica_arr[i] = max(DIMENSION_STATES[dim]['min'], 
                               min(DIMENSION_STATES[dim]['max'], vesica_arr[i]))
        
        return ColorEncoding.from_array(vesica_arr)
    
    def get_spectrum(self, word: str) -> None:
        """Print the frequency spectrum of a word."""
        if word not in self.words:
            print(f"Word '{word}' not found")
            return
        
        wf = self.words[word]
        spectrum = wf.get_spectrum()
        
        print(f"\n📊 SPECTRUM OF '{word}'")
        print("=" * 60)
        print(f"{'Frequency':>10} | {'Weight':>8} | Meaning")
        print("-" * 60)
        
        for freq, label, weight in spectrum:
            bar = "█" * int(weight * 20)
            print(f"{freq:>10.4f} | {weight:>8.4f} | {label} {bar}")
        
        print("=" * 60)


# =============================================================================
# LIVING WORD (FULL IMPLEMENTATION WITH PATH TRACKING)
# =============================================================================

class LivingWord:
    """
    A word that actively tracks its journey through meaning space.
    Combines wave function with path navigation.
    """
    
    def __init__(self, word: str, pde: SemanticPDE):
        self.word = word
        self.pde = pde
        
        # Create in PDE
        self.wave_function = pde.create_word(word)
        
        # Start position (origin to vesica)
        self.navigator = SemanticPathNavigator((0, 0, 0))
        
        # Navigate to initial position (vesica center)
        # Based on birth encoding
        birth_ef = self.wave_function.eigenfunctions[0]
        self.current_encoding = birth_ef.quantum_numbers
        
        # Record birth path
        self._record_birth_path()
    
    def _record_birth_path(self):
        """Record the path from origin to initial position."""
        # Simple birth: just integrate up to natural level
        level = self.current_encoding.G  # Use structure as proxy for level
        if level > 0:
            self.navigator.move_z_integrate(level, "birth: reach natural level")
    
    def expand(self, context: str = "") -> float:
        """Expand meaning (e operation on x)."""
        theta = self.navigator.move_x_e(0.5, context)
        print(f"↔️  Expanded '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def contract(self, context: str = "") -> float:
        """Contract meaning (ln operation on x)."""
        theta = self.navigator.move_x_ln(0.5, context)
        print(f"↔️  Contracted '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def add_positive(self, context: str = "") -> float:
        """Add positive association (sin+, cos+)."""
        theta = self.navigator.move_y_sincos(1, 1, context)
        print(f"✅ Added good to '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def add_negative(self, context: str = "") -> float:
        """Add negative association (sin+, cos-)."""
        theta = self.navigator.move_y_sincos(1, -1, context)
        print(f"❌ Added bad to '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def remove_negative(self, context: str = "") -> float:
        """Remove negative (sin-, cos+) - feels good but different path."""
        theta = self.navigator.move_y_sincos(-1, 1, context)
        print(f"🔄 Removed bad from '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def remove_positive(self, context: str = "") -> float:
        """Remove positive (sin-, cos-) - loss."""
        theta = self.navigator.move_y_sincos(-1, -1, context)
        print(f"💔 Removed good from '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def level_up(self, context: str = "") -> float:
        """Integrate up to higher level."""
        theta = self.navigator.move_z_integrate(1, context)
        print(f"⬆️  Leveled up '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def level_down(self, context: str = "") -> float:
        """Differentiate down to lower level."""
        theta = self.navigator.move_z_differentiate(1, context)
        print(f"⬇️  Leveled down '{self.word}': {context} (θ += {theta:.3f})")
        return theta
    
    def add_meaning(self, label: str, context: Dict) -> SemanticEigenfunction:
        """Add a complete new meaning with context."""
        ef = self.pde.add_meaning(self.word, label, context)
        
        # Record path operations based on context
        if context.get('shape', 'point') in ['plane', 'volume']:
            self.level_up(f"complexity for {label}")
        if context.get('awareness', 'inert') in ['reactive', 'aware']:
            self.level_up(f"consciousness for {label}")
        
        return ef
    
    def get_journey(self) -> str:
        """Get the complete semantic journey of this word."""
        return self.navigator.get_path_signature()
    
    def get_theta_signature(self) -> Tuple[float, float, float]:
        """Get the total theta contribution per axis."""
        return self.navigator.get_total_theta()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_semantic_wave_equation():
    """Demonstrate the complete system."""
    
    print("\n" + "=" * 70)
    print("SEMANTIC WAVE EQUATION DEMONSTRATION")
    print("The Schrödinger Equation of Meaning")
    print("=" * 70)
    
    # Create PDE solver
    pde = SemanticPDE()
    
    # Create a living word: "blue"
    print("\n" + "-" * 70)
    print("PHASE 1: Birth of 'blue'")
    print("-" * 70)
    
    blue = LivingWord("blue", pde)
    
    # Add meanings to blue
    print("\n" + "-" * 70)
    print("PHASE 2: Adding meanings to 'blue'")
    print("-" * 70)
    
    # v1: shade (specification)
    blue.expand("what shade?")
    blue.add_meaning("shade", {
        'exists': True,
        'amount': 'some',
        'motion': 'static',
        'shape': 'line',  # spectrum line
        'connection': 'connected',
        'time': 'present',
        'awareness': 'inert'
    })
    
    # v2: sky blue
    blue.add_positive("sky context")
    blue.add_meaning("sky", {
        'exists': True,
        'amount': 'many',
        'motion': 'dynamic',
        'shape': 'volume',
        'connection': 'connected',
        'time': 'present',
        'awareness': 'reactive'
    })
    
    # v3: ocean
    blue.expand("depth")
    blue.add_meaning("ocean", {
        'exists': True,
        'amount': 'many',
        'motion': 'dynamic',
        'shape': 'volume',
        'connection': 'entangled',
        'time': 'present',
        'awareness': 'reactive'
    })
    
    # v4: sadness
    blue.level_up("emotional complexity")
    blue.contract("narrow to emotion")
    blue.add_meaning("sadness", {
        'exists': True,
        'amount': 'some',
        'motion': 'static',
        'shape': 'plane',
        'connection': 'connected',
        'time': 'past',
        'awareness': 'aware'
    })
    
    # Show spectrum
    pde.get_spectrum("blue")
    
    # Show journey
    print("\n" + "-" * 70)
    print("PHASE 3: Blue's semantic journey")
    print("-" * 70)
    print(f"\nPath taken:\n{blue.get_journey()}")
    print(f"\nTotal θ signature: {blue.get_theta_signature()}")
    
    # Mediation example
    print("\n" + "-" * 70)
    print("PHASE 4: Mediation between domains")
    print("-" * 70)
    
    # Artist (visual) meets Poet (emotional)
    artist_context = {
        'exists': True,
        'amount': 'many',
        'motion': 'dynamic',
        'shape': 'volume',
        'connection': 'connected',
        'time': 'present',
        'awareness': 'reactive'
    }
    
    poet_context = {
        'exists': True,
        'amount': 'some',
        'motion': 'static',
        'shape': 'plane',
        'connection': 'entangled',
        'time': 'past',
        'awareness': 'aware'
    }
    
    bridge = pde.mediate("blue", artist_context, poet_context)
    
    # Show updated spectrum
    pde.get_spectrum("blue")
    
    # Access example
    print("\n" + "-" * 70)
    print("PHASE 5: Accessing 'blue' with different contexts")
    print("-" * 70)
    
    # Someone looking for color
    pde.access_word("blue", {
        'shape': 'line',
        'awareness': 'inert'
    })
    
    # Someone feeling sad
    pde.access_word("blue", {
        'time': 'past',
        'awareness': 'aware'
    })
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: All solutions to the Semantic Wave Equation for 'blue'")
    print("=" * 70)
    
    wf = pde.words["blue"]
    for i, ef in enumerate(wf.eigenfunctions):
        print(f"\nψ_{i} = '{ef.meaning_label}'")
        print(f"   Quantum numbers: {ef.quantum_numbers.to_string()}")
        print(f"   Description: {ef.quantum_numbers.describe()}")
        print(f"   Frequency ω: {ef.frequency:.4f}")
        print(f"   Weight c: {ef.weight:.4f}")
        print(f"   Energy E: {ef.compute_energy():.4f}")
        if ef.parent_meanings:
            print(f"   Resonates with: {ef.parent_meanings}")
    
    return pde, blue


if __name__ == "__main__":
    pde, blue = demonstrate_semantic_wave_equation()
