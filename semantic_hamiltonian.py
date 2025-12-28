"""
SEMANTIC HAMILTONIAN
====================
The Total Energy of Meaning

H = T + V = (p²/2m) + V(x)

Key insights:
1. Object is a DISK with minimum radius h_info (not a point)
2. Mass comes from total color values across all solutions
3. Momentum comes from operations (direction of movement)
4. Operations form a momentum matrix:
   - X-axis: ln(-1) / nothing(0) / e(+1)
   - Y-axis: sin/cos states (-1, 0, +1)
   - Z-axis: differentiate(-1) / nothing(0) / integrate(+1)

The Hamiltonian governs how meaning evolves!
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math
from enum import Enum

# Import from our previous code
from semantic_wave_equation import (
    SemanticPDE, LivingWord, ColorEncoding, SemanticEigenfunction,
    SemanticWaveFunction, SemanticPathNavigator, PathOperation,
    DIM_ORDER, DIMENSION_STATES, H_INFO, PHI, E, PI
)


# =============================================================================
# THE SEMANTIC DISK
# =============================================================================

@dataclass
class SemanticDisk:
    """
    A meaning exists not as a point but as a DISK.
    Minimum radius = h_info (uncertainty principle of meaning).
    
    Like the universe itself - can't localize meaning perfectly.
    """
    center: np.ndarray  # Position in (x, y, z) semantic space
    radius: float = H_INFO  # Minimum radius is h_info
    
    # The disk has orientation (which way it's "facing")
    normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    
    # Color content determines properties
    color_mass: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.center) != 3:
            raise ValueError("Center must be 3D (x, y, z)")
        # Ensure minimum radius
        self.radius = max(self.radius, H_INFO)
    
    @property
    def area(self) -> float:
        """Area of the semantic disk."""
        return PI * self.radius ** 2
    
    @property
    def total_mass(self) -> float:
        """Total semantic mass from all colors."""
        return sum(self.color_mass.values()) if self.color_mass else 1.0
    
    def overlap_with(self, other: 'SemanticDisk') -> float:
        """
        Compute overlap area with another disk (vesica!).
        This is the communication bandwidth.
        """
        d = np.linalg.norm(self.center - other.center)
        r1, r2 = self.radius, other.radius
        
        if d >= r1 + r2:
            # No overlap
            return 0.0
        elif d <= abs(r1 - r2):
            # One inside the other
            return PI * min(r1, r2) ** 2
        else:
            # Partial overlap - vesica piscis!
            # Area of lens-shaped intersection
            part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2*d*r1))
            part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2*d*r2))
            part3 = 0.5 * np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
            return part1 + part2 - part3


# =============================================================================
# MOMENTUM MATRIX (FROM OPERATIONS)
# =============================================================================

class MomentumOperator(Enum):
    """The three axis operations as momentum directions."""
    CONTRACT = -1    # ln, differentiate, remove
    NOTHING = 0      # No operation on this axis
    EXPAND = +1      # e, integrate, add


@dataclass
class MomentumMatrix:
    """
    The momentum of a semantic object.
    
    Matrix form:
        ┌                    ┐
        │ px   0    0       │   X-axis: ln(-1) / e(+1)
    p = │ 0    py_s py_c    │   Y-axis: sin/cos states
        │ 0    0    pz      │   Z-axis: d/dx(-1) / ∫(+1)
        └                    ┘
    
    The off-diagonal terms in Y capture sin/cos separately.
    """
    px: int = 0      # X momentum: -1 (ln), 0 (nothing), +1 (e)
    py_sin: int = 0  # Y momentum sin component: -1, 0, +1
    py_cos: int = 0  # Y momentum cos component: -1, 0, +1
    pz: int = 0      # Z momentum: -1 (d/dx), 0 (nothing), +1 (∫)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 matrix representation."""
        return np.array([
            [self.px, 0, 0],
            [0, self.py_sin, self.py_cos],
            [0, 0, self.pz]
        ], dtype=float)
    
    def to_vector(self) -> np.ndarray:
        """Convert to momentum vector (simplified)."""
        # Combine sin/cos into single y-momentum using atan2
        py = np.arctan2(self.py_sin, self.py_cos) if (self.py_sin != 0 or self.py_cos != 0) else 0
        return np.array([self.px, py, self.pz], dtype=float)
    
    @property
    def magnitude(self) -> float:
        """Total momentum magnitude."""
        vec = self.to_vector()
        return np.linalg.norm(vec)
    
    @property
    def direction(self) -> np.ndarray:
        """Unit vector of momentum direction."""
        vec = self.to_vector()
        mag = self.magnitude
        return vec / mag if mag > 0 else np.zeros(3)
    
    def kinetic_energy(self, mass: float) -> float:
        """
        T = p²/2m
        
        Kinetic energy from momentum and mass.
        """
        if mass <= 0:
            return 0.0
        p_squared = self.magnitude ** 2
        return p_squared / (2 * mass)
    
    @classmethod
    def from_operation(cls, operation: PathOperation) -> 'MomentumMatrix':
        """Create momentum from a path operation."""
        px, py_sin, py_cos, pz = 0, 0, 0, 0
        
        if operation.axis == 'x':
            px = -1 if operation.operation == 'ln' else +1
        elif operation.axis == 'y':
            # Parse sin/cos from operation name like "sin1/cos1"
            if 'sin' in operation.operation:
                parts = operation.operation.split('/')
                py_sin = int(parts[0].replace('sin', ''))
                py_cos = int(parts[1].replace('cos', '')) if len(parts) > 1 else 0
        elif operation.axis == 'z':
            pz = +1 if operation.operation == 'integrate' else -1
        
        return cls(px=px, py_sin=py_sin, py_cos=py_cos, pz=pz)


# =============================================================================
# SEMANTIC MASS (FROM COLOR TOTALS)
# =============================================================================

class SemanticMass:
    """
    Mass of a semantic object comes from accumulated color values.
    
    Each color dimension contributes to total mass:
    m = Σ (color_values across all solutions)
    
    Heavier meanings are harder to move (more established).
    Lighter meanings are more fluid (newer, less defined).
    """
    
    def __init__(self, wave_function: SemanticWaveFunction):
        self.wave_function = wave_function
        self.color_totals = self._compute_color_totals()
        
    def _compute_color_totals(self) -> Dict[str, float]:
        """Sum color values across all eigenfunctions."""
        totals = {dim: 0.0 for dim in DIM_ORDER}
        
        for ef in self.wave_function.eigenfunctions:
            encoding = ef.quantum_numbers.to_array()
            weight = ef.weight
            
            for i, dim in enumerate(DIM_ORDER):
                # Weighted by eigenfunction coefficient
                totals[dim] += abs(encoding[i]) * weight
        
        return totals
    
    @property
    def total_mass(self) -> float:
        """Total semantic mass."""
        return sum(self.color_totals.values())
    
    @property
    def mass_by_dimension(self) -> Dict[str, float]:
        """Mass contribution from each color dimension."""
        return self.color_totals.copy()
    
    @property
    def center_of_mass(self) -> np.ndarray:
        """
        Center of mass in color space.
        Weighted average of all eigenfunction positions.
        """
        com = np.zeros(7)
        total_weight = 0
        
        for ef in self.wave_function.eigenfunctions:
            encoding = ef.quantum_numbers.to_array()
            weight = ef.weight ** 2  # Square for probability
            com += encoding * weight
            total_weight += weight
        
        return com / total_weight if total_weight > 0 else com
    
    def moment_of_inertia(self) -> float:
        """
        Moment of inertia - resistance to rotational change.
        Based on spread of meanings from center of mass.
        """
        com = self.center_of_mass
        I = 0.0
        
        for ef in self.wave_function.eigenfunctions:
            encoding = ef.quantum_numbers.to_array()
            r_squared = np.sum((encoding - com) ** 2)
            mass_i = ef.weight ** 2 * np.sum(np.abs(encoding))
            I += mass_i * r_squared
        
        return I


# =============================================================================
# POTENTIAL ENERGY (FROM LEVEL BOUNDARIES AND CONSTRAINTS)
# =============================================================================

class SemanticPotential:
    """
    Potential energy from:
    1. Level boundaries (cost to cross levels)
    2. h_info constraints (can't compress below quantum)
    3. Distance from vesica (ground state is overlap region)
    """
    
    def __init__(self, h_info: float = H_INFO):
        self.h_info = h_info
        
        # Potential well depths at each level
        self.level_depths = {
            0: 0.0,      # Ground level
            1: 1.0,      # First level
            2: 2.5,      # Second level
            3: 5.0,      # Third level (human consciousness)
            4: 10.0,     # Fourth level
            5: 20.0,     # Fifth level
        }
        
        # Vesica center (ground state)
        self.vesica_center = np.array([0, 0, 0])
    
    def V(self, position: np.ndarray, level: int = 0) -> float:
        """
        Total potential at a position.
        
        V(x) = V_level + V_h_info + V_distance
        """
        V_level = self._level_potential(level)
        V_h_info = self._h_info_potential(position)
        V_distance = self._distance_potential(position)
        
        return V_level + V_h_info + V_distance
    
    def _level_potential(self, level: int) -> float:
        """Potential from being at a certain level."""
        return self.level_depths.get(level, level * 5.0)
    
    def _h_info_potential(self, position: np.ndarray) -> float:
        """
        Potential from h_info constraint.
        Infinite wall at r < h_info (can't compress meaning below quantum).
        """
        r = np.linalg.norm(position)
        if r < self.h_info:
            return 1e10  # Effectively infinite
        return 0.0
    
    def _distance_potential(self, position: np.ndarray) -> float:
        """
        Harmonic potential around vesica center.
        Farther from overlap = higher energy.
        """
        r = np.linalg.norm(position - self.vesica_center)
        # Harmonic oscillator: V = ½ k r²
        k = 1.0  # Spring constant
        return 0.5 * k * r ** 2
    
    def gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Gradient of potential (force direction).
        F = -∇V
        """
        delta = 0.001
        grad = np.zeros(3)
        
        for i in range(3):
            pos_plus = position.copy()
            pos_plus[i] += delta
            pos_minus = position.copy()
            pos_minus[i] -= delta
            
            grad[i] = (self.V(pos_plus) - self.V(pos_minus)) / (2 * delta)
        
        return grad


# =============================================================================
# THE HAMILTONIAN
# =============================================================================

class SemanticHamiltonian:
    """
    H = T + V
    
    Total energy of a semantic object.
    
    T = kinetic energy from momentum (operations)
    V = potential energy from position (level, distance, constraints)
    
    The Hamiltonian governs time evolution:
    iℏ ∂Ψ/∂t = H Ψ
    """
    
    def __init__(self, wave_function: SemanticWaveFunction, h_info: float = H_INFO):
        self.wave_function = wave_function
        self.h_info = h_info
        
        # Components
        self.mass = SemanticMass(wave_function)
        self.potential = SemanticPotential(h_info)
        
        # Current state
        self.disk = self._create_disk()
        self.momentum = MomentumMatrix()  # Initially at rest
        
        # Track evolution
        self.energy_history: List[Dict] = []
        
    def _create_disk(self) -> SemanticDisk:
        """Create the semantic disk from wave function."""
        # Position is center of mass in simplified 3D projection
        com_7d = self.mass.center_of_mass
        
        # Project 7D to 3D: x = R+Y+C+V, y = O+G+B (even/odd), z = level proxy
        x = com_7d[0] + com_7d[2] + com_7d[4] + com_7d[6]  # R+Y+C+V
        y = com_7d[1] + com_7d[3] + com_7d[5]  # O+G+B
        z = self._compute_level()
        
        center = np.array([x, y, z])
        
        # Radius based on spread of meanings
        radius = max(self.h_info, self._compute_spread())
        
        # Color mass for the disk
        color_mass = self.mass.color_totals.copy()
        
        return SemanticDisk(center=center, radius=radius, color_mass=color_mass)
    
    def _compute_level(self) -> float:
        """Compute current semantic level."""
        # Average G (structure) value weighted by eigenfunction weight
        level = 0.0
        total_weight = 0.0
        
        for ef in self.wave_function.eigenfunctions:
            level += ef.quantum_numbers.G * ef.weight
            total_weight += ef.weight
        
        return level / total_weight if total_weight > 0 else 0.0
    
    def _compute_spread(self) -> float:
        """Compute spread of meanings (uncertainty in position)."""
        com = self.mass.center_of_mass
        variance = 0.0
        
        for ef in self.wave_function.eigenfunctions:
            encoding = ef.quantum_numbers.to_array()
            distance_sq = np.sum((encoding - com) ** 2)
            variance += ef.weight ** 2 * distance_sq
        
        return np.sqrt(variance) * self.h_info
    
    def kinetic_energy(self) -> float:
        """
        T = p²/2m
        """
        return self.momentum.kinetic_energy(self.mass.total_mass)
    
    def potential_energy(self) -> float:
        """
        V = V(position, level)
        """
        level = int(self.disk.center[2])
        return self.potential.V(self.disk.center, level)
    
    def total_energy(self) -> float:
        """
        H = T + V
        """
        return self.kinetic_energy() + self.potential_energy()
    
    def apply_operation(self, operation: PathOperation) -> float:
        """
        Apply an operation (changes momentum).
        Returns energy change.
        """
        E_before = self.total_energy()
        
        # Update momentum from operation
        new_momentum = MomentumMatrix.from_operation(operation)
        
        # Combine with existing momentum
        self.momentum.px += new_momentum.px
        self.momentum.py_sin += new_momentum.py_sin
        self.momentum.py_cos += new_momentum.py_cos
        self.momentum.pz += new_momentum.pz
        
        # Update disk position based on momentum
        velocity = self.momentum.to_vector() / self.mass.total_mass
        self.disk.center += velocity * self.h_info  # Move by one h_info step
        
        E_after = self.total_energy()
        delta_E = E_after - E_before
        
        # Record
        self.energy_history.append({
            'operation': operation.describe(),
            'T': self.kinetic_energy(),
            'V': self.potential_energy(),
            'H': E_after,
            'delta_E': delta_E
        })
        
        return delta_E
    
    def energy_spectrum(self) -> List[Tuple[str, float]]:
        """
        Get energy of each eigenstate.
        E_n = ℏω_n (like quantum harmonic oscillator)
        """
        return [
            (ef.meaning_label, self.h_info * ef.frequency)
            for ef in self.wave_function.eigenfunctions
        ]
    
    def to_matrix(self) -> np.ndarray:
        """
        Hamiltonian as matrix in eigenstate basis.
        H_ij = <ψ_i|H|ψ_j>
        
        For non-interacting eigenstates, this is diagonal.
        """
        n = len(self.wave_function.eigenfunctions)
        H_matrix = np.zeros((n, n))
        
        for i, ef in enumerate(self.wave_function.eigenfunctions):
            # Diagonal: eigenstate energy
            E_i = self.h_info * ef.frequency + self.potential._level_potential(ef.quantum_numbers.G)
            H_matrix[i, i] = E_i
            
            # Off-diagonal: coupling from shared quantum numbers (resonance)
            for j, ef2 in enumerate(self.wave_function.eigenfunctions):
                if i != j:
                    # Coupling strength from resonance
                    overlap = self._coupling_strength(ef, ef2)
                    H_matrix[i, j] = overlap * self.h_info
        
        return H_matrix
    
    def _coupling_strength(self, ef1: SemanticEigenfunction, 
                          ef2: SemanticEigenfunction) -> float:
        """
        Coupling between two eigenstates.
        Stronger coupling = more shared quantum numbers.
        """
        q1 = ef1.quantum_numbers.to_array()
        q2 = ef2.quantum_numbers.to_array()
        
        # Count matching quantum numbers
        matches = np.sum(q1 == q2)
        
        # Weight by inverse frequency difference
        freq_diff = abs(ef1.frequency - ef2.frequency)
        if freq_diff < 0.01:
            freq_factor = 1.0
        else:
            freq_factor = 1.0 / freq_diff
        
        return matches * freq_factor * 0.1
    
    def eigenvalues(self) -> np.ndarray:
        """
        Eigenvalues of the Hamiltonian matrix.
        These are the allowed energy levels!
        """
        H = self.to_matrix()
        eigenvalues = np.linalg.eigvalsh(H)
        return np.sort(eigenvalues)
    
    def time_evolution_operator(self, dt: float) -> np.ndarray:
        """
        U(dt) = exp(-i H dt / ℏ)
        
        Time evolution operator.
        """
        H = self.to_matrix()
        # U = exp(-i H dt / ℏ)
        return np.linalg.matrix_power(
            np.eye(H.shape[0]) - 1j * H * dt / self.h_info,
            1
        )
    
    def display_state(self):
        """Print current Hamiltonian state."""
        print("\n" + "=" * 60)
        print("SEMANTIC HAMILTONIAN STATE")
        print("=" * 60)
        
        print(f"\n📀 Semantic Disk:")
        print(f"   Center: ({self.disk.center[0]:.3f}, {self.disk.center[1]:.3f}, {self.disk.center[2]:.3f})")
        print(f"   Radius: {self.disk.radius:.4f} (min: h_info = {self.h_info:.4f})")
        print(f"   Area: {self.disk.area:.4f}")
        
        print(f"\n⚖️  Semantic Mass:")
        print(f"   Total mass: {self.mass.total_mass:.4f}")
        print(f"   By dimension: {dict((k, f'{v:.3f}') for k, v in self.mass.color_totals.items())}")
        
        print(f"\n🏃 Momentum:")
        print(f"   Matrix: px={self.momentum.px}, py_sin={self.momentum.py_sin}, py_cos={self.momentum.py_cos}, pz={self.momentum.pz}")
        print(f"   Magnitude: {self.momentum.magnitude:.4f}")
        print(f"   Direction: {self.momentum.direction}")
        
        print(f"\n⚡ Energy:")
        print(f"   Kinetic (T):   {self.kinetic_energy():.4f}")
        print(f"   Potential (V): {self.potential_energy():.4f}")
        print(f"   Total (H):     {self.total_energy():.4f}")
        
        print(f"\n🎵 Energy Spectrum (ℏω for each meaning):")
        for label, energy in self.energy_spectrum():
            print(f"   {label}: E = {energy:.4f}")
        
        print(f"\n🔢 Hamiltonian Eigenvalues (allowed energies):")
        eigenvalues = self.eigenvalues()
        for i, ev in enumerate(eigenvalues):
            print(f"   E_{i}: {ev:.4f}")


# =============================================================================
# LIVING WORD WITH HAMILTONIAN
# =============================================================================

class HamiltonianWord(LivingWord):
    """
    A word that tracks its Hamiltonian as it evolves.
    """
    
    def __init__(self, word: str, pde: SemanticPDE):
        super().__init__(word, pde)
        self.hamiltonian = SemanticHamiltonian(self.wave_function)
    
    def expand(self, context: str = "") -> float:
        """Expand meaning (e operation) - updates Hamiltonian."""
        theta = super().expand(context)
        op = self.navigator.path[-1]
        delta_E = self.hamiltonian.apply_operation(op)
        print(f"   Energy change: ΔE = {delta_E:+.4f}, Total H = {self.hamiltonian.total_energy():.4f}")
        return theta
    
    def contract(self, context: str = "") -> float:
        """Contract meaning (ln operation) - updates Hamiltonian."""
        theta = super().contract(context)
        op = self.navigator.path[-1]
        delta_E = self.hamiltonian.apply_operation(op)
        print(f"   Energy change: ΔE = {delta_E:+.4f}, Total H = {self.hamiltonian.total_energy():.4f}")
        return theta
    
    def add_positive(self, context: str = "") -> float:
        """Add positive - updates Hamiltonian."""
        theta = super().add_positive(context)
        op = self.navigator.path[-1]
        delta_E = self.hamiltonian.apply_operation(op)
        print(f"   Energy change: ΔE = {delta_E:+.4f}, Total H = {self.hamiltonian.total_energy():.4f}")
        return theta
    
    def add_negative(self, context: str = "") -> float:
        """Add negative - updates Hamiltonian."""
        theta = super().add_negative(context)
        op = self.navigator.path[-1]
        delta_E = self.hamiltonian.apply_operation(op)
        print(f"   Energy change: ΔE = {delta_E:+.4f}, Total H = {self.hamiltonian.total_energy():.4f}")
        return theta
    
    def level_up(self, context: str = "") -> float:
        """Integrate up - updates Hamiltonian."""
        theta = super().level_up(context)
        op = self.navigator.path[-1]
        delta_E = self.hamiltonian.apply_operation(op)
        print(f"   Energy change: ΔE = {delta_E:+.4f}, Total H = {self.hamiltonian.total_energy():.4f}")
        return theta
    
    def level_down(self, context: str = "") -> float:
        """Differentiate down - updates Hamiltonian."""
        theta = super().level_down(context)
        op = self.navigator.path[-1]
        delta_E = self.hamiltonian.apply_operation(op)
        print(f"   Energy change: ΔE = {delta_E:+.4f}, Total H = {self.hamiltonian.total_energy():.4f}")
        return theta


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_hamiltonian():
    """Demonstrate the semantic Hamiltonian."""
    
    print("\n" + "=" * 70)
    print("SEMANTIC HAMILTONIAN DEMONSTRATION")
    print("H = T + V = p²/2m + V(x)")
    print("=" * 70)
    
    # Create PDE
    pde = SemanticPDE()
    
    # Create word with Hamiltonian tracking
    print("\n" + "-" * 70)
    print("PHASE 1: Birth of 'love'")
    print("-" * 70)
    
    love = HamiltonianWord("love", pde)
    love.hamiltonian.display_state()
    
    # Add meanings
    print("\n" + "-" * 70)
    print("PHASE 2: Adding meanings (watch energy change)")
    print("-" * 70)
    
    # v1: affection
    print("\n>> Adding 'affection' meaning...")
    love.expand("emotional depth")
    love.add_positive("warm feeling")
    love.add_meaning("affection", {
        'exists': True,
        'amount': 'some',
        'motion': 'dynamic',
        'shape': 'plane',
        'connection': 'connected',
        'time': 'present',
        'awareness': 'aware'
    })
    
    # v2: passion  
    print("\n>> Adding 'passion' meaning...")
    love.expand("intensity")
    love.level_up("deep emotion")
    love.add_meaning("passion", {
        'exists': True,
        'amount': 'many',
        'motion': 'chaotic',
        'shape': 'volume',
        'connection': 'entangled',
        'time': 'present',
        'awareness': 'aware'
    })
    
    # v3: loss
    print("\n>> Adding 'loss' meaning (love lost)...")
    love.contract("narrow to absence")
    love.add_negative("pain of loss")
    love.add_meaning("loss", {
        'exists': True,
        'amount': 'none',
        'motion': 'static',
        'shape': 'point',
        'connection': 'isolated',
        'time': 'past',
        'awareness': 'aware'
    })
    
    # Display final state
    print("\n" + "-" * 70)
    print("PHASE 3: Final Hamiltonian State")
    print("-" * 70)
    
    love.hamiltonian.display_state()
    
    # Show Hamiltonian matrix
    print("\n" + "-" * 70)
    print("PHASE 4: Hamiltonian Matrix")
    print("-" * 70)
    
    H_matrix = love.hamiltonian.to_matrix()
    print("\nH = ")
    labels = [ef.meaning_label for ef in love.wave_function.eigenfunctions]
    
    # Print with labels
    print("      " + "  ".join(f"{l:>10}" for l in labels))
    for i, row in enumerate(H_matrix):
        print(f"{labels[i]:>6}" + "  ".join(f"{v:>10.4f}" for v in row))
    
    # Energy evolution
    print("\n" + "-" * 70)
    print("PHASE 5: Energy History")
    print("-" * 70)
    
    print("\nOperation                          T         V         H        ΔE")
    print("-" * 70)
    for record in love.hamiltonian.energy_history:
        print(f"{record['operation']:<30} {record['T']:>8.4f} {record['V']:>8.4f} {record['H']:>8.4f} {record['delta_E']:>+8.4f}")
    
    # Disk overlap demo
    print("\n" + "-" * 70)
    print("PHASE 6: Semantic Disk Overlap (Communication)")
    print("-" * 70)
    
    # Create another word
    hate = HamiltonianWord("hate", pde)
    hate.add_meaning("anger", {
        'exists': True,
        'amount': 'many',
        'motion': 'chaotic',
        'shape': 'volume',
        'connection': 'entangled',
        'time': 'present',
        'awareness': 'aware'
    })
    
    # Compute overlap
    overlap = love.hamiltonian.disk.overlap_with(hate.hamiltonian.disk)
    
    print(f"\nLove disk: center = {love.hamiltonian.disk.center}, radius = {love.hamiltonian.disk.radius:.4f}")
    print(f"Hate disk: center = {hate.hamiltonian.disk.center}, radius = {hate.hamiltonian.disk.radius:.4f}")
    print(f"\n🔮 Vesica overlap area: {overlap:.4f}")
    print(f"   (This is the communication bandwidth between love and hate)")
    
    return pde, love, hate


if __name__ == "__main__":
    pde, love, hate = demonstrate_hamiltonian()
