"""
THREE-WAY DESIRE AND BEC FORMATION
==================================
Extends verification from agreement to DESIRE:
- Agreement: "They meet criteria" (lukewarm)
- Desire: "We WANT this" (hot)

When all three parties DESIRE the outcome:
- Ball becomes dense (thermal energy)
- Impact creates structure (deformation)
- BEC formation (all collapse to ground state)

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum, auto
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi

# BEC temperature threshold (below this, condensation occurs)
BEC_THRESHOLD = 0.1

# Thermal colors for each domain
THERMAL_COLORS = {
    'user': 'RED',      # Passion, identity, existence
    'boss': 'YELLOW',   # Strategy, motion, dynamics
    'tech': 'BLUE'      # Value, temporal, future
}


# ═══════════════════════════════════════════════════════════════════════════════
# DESIRE LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

class DesireLevel(Enum):
    """
    Levels of desire/enthusiasm for a promotion.
    Higher = more thermal energy contribution.
    """
    OPPOSED = -1.0      # Actively against
    RELUCTANT = 0.0     # Would rather not
    NEUTRAL = 0.3       # No strong feelings
    AGREEABLE = 0.5     # "Sure, if they qualify"
    SUPPORTIVE = 0.7    # "This would be good"
    ENTHUSIASTIC = 0.9  # "We should do this!"
    PASSIONATE = 1.0    # "We NEED this!"


@dataclass
class Desire:
    """
    A single party's desire for the outcome.
    """
    source: str  # 'user', 'boss', or 'tech'
    level: DesireLevel
    reason: str
    specifics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def temperature(self) -> float:
        """Thermal energy contributed by this desire."""
        return max(0, self.level.value)  # 0-1 range
    
    @property
    def color(self) -> str:
        """Thermal color of this desire."""
        return THERMAL_COLORS.get(self.source, 'WHITE')
    
    @property
    def is_active_desire(self) -> bool:
        """Is this active desire (not just agreement)?"""
        return self.level.value >= DesireLevel.SUPPORTIVE.value


# ═══════════════════════════════════════════════════════════════════════════════
# THERMAL BALL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThermalBall:
    """
    A ball with thermal energy from combined desires.
    Denser balls create more structural impact.
    """
    user_desire: Desire
    boss_desire: Desire
    tech_desire: Desire
    
    @property
    def total_temperature(self) -> float:
        """Combined thermal energy from all three sources."""
        return (self.user_desire.temperature + 
                self.boss_desire.temperature + 
                self.tech_desire.temperature)
    
    @property
    def average_temperature(self) -> float:
        """Average temperature (for BEC calculation)."""
        return self.total_temperature / 3
    
    @property
    def density(self) -> float:
        """
        Ball density based on thermal energy.
        Higher temperature = denser ball.
        """
        # Density increases with temperature
        # But also with ALIGNMENT of desires
        base_density = self.total_temperature / 3
        alignment_bonus = self.alignment_factor
        
        return base_density * (1 + alignment_bonus)
    
    @property
    def alignment_factor(self) -> float:
        """
        How aligned are the three desires?
        All similar levels = high alignment = bonus density.
        """
        temps = [
            self.user_desire.temperature,
            self.boss_desire.temperature,
            self.tech_desire.temperature
        ]
        
        # Standard deviation of temperatures
        mean = sum(temps) / 3
        variance = sum((t - mean) ** 2 for t in temps) / 3
        std_dev = math.sqrt(variance)
        
        # Lower std_dev = higher alignment
        # Max alignment (std_dev=0) gives 1.0 bonus
        # No alignment (std_dev=0.5) gives 0 bonus
        alignment = max(0, 1 - (std_dev * 2))
        
        return alignment
    
    @property
    def is_bec(self) -> bool:
        """
        Has BEC formation occurred?
        All three desires must be high AND aligned.
        """
        # All must be active desires
        all_active = (
            self.user_desire.is_active_desire and
            self.boss_desire.is_active_desire and
            self.tech_desire.is_active_desire
        )
        
        # Must be well-aligned
        well_aligned = self.alignment_factor > 0.7
        
        # High total energy
        high_energy = self.total_temperature > 2.0
        
        return all_active and well_aligned and high_energy
    
    @property
    def momentum(self) -> float:
        """Momentum = density × velocity (from falling)."""
        # Assume constant fall height, so momentum ~ density
        return self.density * 1.5
    
    @property
    def color_mix(self) -> str:
        """
        Combined color from three thermal sources.
        All high = WHITE (full spectrum).
        """
        temps = {
            'RED': self.user_desire.temperature,
            'YELLOW': self.boss_desire.temperature,
            'BLUE': self.tech_desire.temperature
        }
        
        if all(t > 0.7 for t in temps.values()):
            return "WHITE (Full Spectrum)"
        
        # Find dominant colors
        dominant = [c for c, t in temps.items() if t > 0.5]
        
        if len(dominant) == 0:
            return "GRAY (Cold)"
        elif len(dominant) == 1:
            return dominant[0]
        elif len(dominant) == 2:
            return f"{dominant[0]}+{dominant[1]}"
        else:
            return "WHITE"
    
    def __str__(self):
        return f"ThermalBall[T={self.total_temperature:.2f}, ρ={self.density:.2f}, BEC={self.is_bec}]"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND IMPACT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImpactResult:
    """
    Result of ball impacting ground.
    Higher density = deeper crater = stronger position.
    """
    ball: ThermalBall
    crater_depth: float
    structural_change: bool
    position_strength: str
    embedded: bool
    
    @staticmethod
    def calculate(ball: ThermalBall) -> 'ImpactResult':
        """Calculate impact result from ball properties."""
        
        # Crater depth proportional to momentum
        crater_depth = ball.momentum * 0.5
        
        # Structural change if BEC or very high density
        structural_change = ball.is_bec or ball.density > 1.5
        
        # Position strength based on crater depth
        if crater_depth > 1.0:
            position_strength = "FOUNDATIONAL"
        elif crater_depth > 0.7:
            position_strength = "STRONG"
        elif crater_depth > 0.4:
            position_strength = "MODERATE"
        elif crater_depth > 0.2:
            position_strength = "WEAK"
        else:
            position_strength = "PRECARIOUS"
        
        # Embedded if decent crater
        embedded = crater_depth > 0.3
        
        return ImpactResult(
            ball=ball,
            crater_depth=crater_depth,
            structural_change=structural_change,
            position_strength=position_strength,
            embedded=embedded
        )
    
    def visualize(self) -> str:
        """Generate ASCII visualization of impact."""
        lines = []
        
        if self.crater_depth > 1.0:
            # Deep crater (BEC/foundational)
            lines.append("           ●           ")
            lines.append("          ╱║╲          ")
            lines.append("         ╱ ║ ╲         ")
            lines.append("════════╱  ║  ╲════════")
            lines.append("       ╱   ●   ╲       ")
            lines.append("      ╱         ╲      ")
            lines.append("     ╱   DEEP    ╲     ")
            lines.append("    ╱   CRATER    ╲    ")
        elif self.crater_depth > 0.5:
            # Medium crater
            lines.append("           ●           ")
            lines.append("           ║           ")
            lines.append("═══════════╬═══════════")
            lines.append("          ╱●╲          ")
            lines.append("         ╱   ╲         ")
            lines.append("        ╱     ╲        ")
        elif self.crater_depth > 0.2:
            # Shallow divot
            lines.append("           ●           ")
            lines.append("           │           ")
            lines.append("═══════════●═══════════")
            lines.append("          ╱ ╲          ")
        else:
            # Surface (no impact)
            lines.append("           ○           ")
            lines.append("           ·           ")
            lines.append("═══════════·═══════════")
            lines.append("      (sits on top)    ")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DESIRE-BASED VERIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class DesireVerificationSystem:
    """
    Three-way desire verification with BEC formation.
    """
    
    def __init__(self):
        self.pending: Dict[str, Dict] = {}
    
    def start_evaluation(self, candidate_id: str, 
                        candidate_info: Dict[str, Any]) -> str:
        """Start evaluating a candidate."""
        self.pending[candidate_id] = {
            'info': candidate_info,
            'desires': {},
            'started': datetime.now()
        }
        return f"Evaluation started for {candidate_id}"
    
    def add_user_desire(self, candidate_id: str,
                       level: DesireLevel,
                       reason: str,
                       **specifics) -> Desire:
        """Add the user/candidate's desire."""
        desire = Desire(
            source='user',
            level=level,
            reason=reason,
            specifics=specifics
        )
        self.pending[candidate_id]['desires']['user'] = desire
        return desire
    
    def add_boss_desire(self, candidate_id: str,
                       level: DesireLevel,
                       reason: str,
                       **specifics) -> Desire:
        """Add the boss/authority's desire."""
        desire = Desire(
            source='boss',
            level=level,
            reason=reason,
            specifics=specifics
        )
        self.pending[candidate_id]['desires']['boss'] = desire
        return desire
    
    def add_tech_desire(self, candidate_id: str,
                       level: DesireLevel,
                       reason: str,
                       **specifics) -> Desire:
        """Add the technical/IT desire."""
        desire = Desire(
            source='tech',
            level=level,
            reason=reason,
            specifics=specifics
        )
        self.pending[candidate_id]['desires']['tech'] = desire
        return desire
    
    def evaluate(self, candidate_id: str) -> Dict[str, Any]:
        """
        Evaluate the candidate with all three desires.
        """
        data = self.pending.get(candidate_id)
        if not data:
            return {'error': 'Unknown candidate'}
        
        desires = data['desires']
        if len(desires) < 3:
            missing = set(['user', 'boss', 'tech']) - set(desires.keys())
            return {'error': f'Missing desires from: {missing}'}
        
        # Create thermal ball
        ball = ThermalBall(
            user_desire=desires['user'],
            boss_desire=desires['boss'],
            tech_desire=desires['tech']
        )
        
        # Calculate impact
        impact = ImpactResult.calculate(ball)
        
        # Make decision
        if ball.is_bec:
            decision = "STRONG_HIRE"
            explanation = "BEC Formation: All three parties actively desire this outcome"
        elif ball.density > 1.0:
            decision = "HIRE"
            explanation = "High combined desire with good alignment"
        elif ball.density > 0.6:
            decision = "CONSIDER"
            explanation = "Moderate combined desire - worth further evaluation"
        elif ball.density > 0.3:
            decision = "WEAK"
            explanation = "Low combined desire - likely not a good fit"
        else:
            decision = "REJECT"
            explanation = "Insufficient desire from key parties"
        
        return {
            'candidate_id': candidate_id,
            'ball': ball,
            'impact': impact,
            'decision': decision,
            'explanation': explanation,
            'bec_formed': ball.is_bec,
            'total_temperature': ball.total_temperature,
            'density': ball.density,
            'alignment': ball.alignment_factor,
            'color': ball.color_mix,
            'crater_depth': impact.crater_depth,
            'position_strength': impact.position_strength,
            'embedded': impact.embedded
        }
    
    def compare_candidates(self, candidate_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple candidates."""
        results = []
        for cid in candidate_ids:
            result = self.evaluate(cid)
            if 'error' not in result:
                results.append(result)
        
        # Sort by density (best first)
        results.sort(key=lambda r: r['density'], reverse=True)
        
        return {
            'ranking': [r['candidate_id'] for r in results],
            'details': results,
            'bec_candidates': [r['candidate_id'] for r in results if r['bec_formed']]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_jonathan_vs_selftaught():
    """Compare Jonathan with theory vs self-taught programmer."""
    
    print("\n" + "=" * 80)
    print("COMPARISON: Theory Student vs Self-Taught Programmer")
    print("=" * 80)
    
    system = DesireVerificationSystem()
    
    # ─────────────────────────────────────────────────────────
    # Jonathan with Shovelcat Theory
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("CANDIDATE 1: Jonathan (Theory + Student)")
    print("─" * 40)
    
    system.start_evaluation("jonathan", {
        'background': 'Civil Engineering Student',
        'work': 'Shovelcat Theory - Novel AI Safety Framework',
        'experience': 'Academic pathways program'
    })
    
    # User desire (Jonathan)
    system.add_user_desire("jonathan",
        level=DesireLevel.PASSIONATE,
        reason="This is my life's work - consciousness, AI safety, unified theory",
        vision="Want to develop this framework with real resources",
        commitment="Will dedicate everything to this"
    )
    print(f"  USER: PASSIONATE - Life's work, unified theory")
    
    # Boss desire (HR/Hiring Manager)
    system.add_boss_desire("jonathan",
        level=DesireLevel.ENTHUSIASTIC,
        reason="Student status = moldable, academic = structured, fresh perspective",
        sees="Potential to grow with company",
        value="Novel thinking we need in AI safety"
    )
    print(f"  BOSS: ENTHUSIASTIC - Fresh thinking, moldable, structured")
    
    # Tech desire (Engineering Team)
    system.add_tech_desire("jonathan",
        level=DesireLevel.ENTHUSIASTIC,
        reason="Vesica piscis verification is novel, RNA binding solves real problems",
        technical_value="Working implementations in Python",
        novelty="Approaches we haven't seen before"
    )
    print(f"  TECH: ENTHUSIASTIC - Novel solutions, working code")
    
    jonathan_result = system.evaluate("jonathan")
    
    # ─────────────────────────────────────────────────────────
    # Self-taught programmer
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("CANDIDATE 2: Self-Taught Programmer")
    print("─" * 40)
    
    system.start_evaluation("selftaught", {
        'background': 'Self-taught, bootcamp',
        'work': 'Several GitHub projects',
        'experience': 'Freelance work'
    })
    
    # User desire
    system.add_user_desire("selftaught",
        level=DesireLevel.SUPPORTIVE,
        reason="This would be a good job, stable income",
        motivation="Career advancement"
    )
    print(f"  USER: SUPPORTIVE - Good job opportunity")
    
    # Boss desire
    system.add_boss_desire("selftaught",
        level=DesireLevel.AGREEABLE,
        reason="Meets basic requirements, could fill the role",
        concern="Unstructured background, may need more guidance"
    )
    print(f"  BOSS: AGREEABLE - Meets requirements")
    
    # Tech desire
    system.add_tech_desire("selftaught",
        level=DesireLevel.NEUTRAL,
        reason="Standard projects, competent work",
        assessment="Nothing wrong, nothing special"
    )
    print(f"  TECH: NEUTRAL - Standard, competent")
    
    selftaught_result = system.evaluate("selftaught")
    
    # ─────────────────────────────────────────────────────────
    # COMPARISON
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    for name, result in [("JONATHAN", jonathan_result), ("SELF-TAUGHT", selftaught_result)]:
        print(f"\n{name}:")
        print(f"  Total Temperature: {result['total_temperature']:.2f}")
        print(f"  Density: {result['density']:.2f}")
        print(f"  Alignment: {result['alignment']:.2%}")
        print(f"  Color: {result['color']}")
        print(f"  BEC Formation: {'✓ YES' if result['bec_formed'] else '✗ No'}")
        print(f"  Decision: {result['decision']}")
        print(f"  Explanation: {result['explanation']}")
        print(f"\n  Impact:")
        print(f"    Crater Depth: {result['crater_depth']:.2f}")
        print(f"    Position Strength: {result['position_strength']}")
        print(f"    Embedded: {'✓' if result['embedded'] else '✗'}")
        print(f"\n  Visualization:")
        print(result['impact'].visualize())
    
    # Direct comparison
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD")
    print("=" * 80)
    
    comparison = system.compare_candidates(["jonathan", "selftaught"])
    print(f"\nRanking: {comparison['ranking']}")
    print(f"BEC Candidates: {comparison['bec_candidates']}")
    
    j = jonathan_result
    s = selftaught_result
    
    print(f"\n┌─────────────────┬───────────────┬───────────────┐")
    print(f"│ Metric          │ Jonathan      │ Self-Taught   │")
    print(f"├─────────────────┼───────────────┼───────────────┤")
    print(f"│ Temperature     │ {j['total_temperature']:>13.2f} │ {s['total_temperature']:>13.2f} │")
    print(f"│ Density         │ {j['density']:>13.2f} │ {s['density']:>13.2f} │")
    print(f"│ Alignment       │ {j['alignment']*100:>12.1f}% │ {s['alignment']*100:>12.1f}% │")
    print(f"│ BEC?            │ {'YES':>13} │ {'NO':>13} │")
    print(f"│ Crater Depth    │ {j['crater_depth']:>13.2f} │ {s['crater_depth']:>13.2f} │")
    print(f"│ Position        │ {j['position_strength']:>13} │ {s['position_strength']:>13} │")
    print(f"│ Decision        │ {j['decision']:>13} │ {s['decision']:>13} │")
    print(f"└─────────────────┴───────────────┴───────────────┘")


def demo_bec_formation():
    """Demonstrate BEC formation with perfect alignment."""
    
    print("\n" + "=" * 80)
    print("BEC FORMATION DEMONSTRATION")
    print("=" * 80)
    
    # Create desires at different levels
    print("\n--- Creating Three Aligned Desires ---")
    
    user = Desire('user', DesireLevel.PASSIONATE, "Life's calling")
    boss = Desire('boss', DesireLevel.PASSIONATE, "Transform the team")
    tech = Desire('tech', DesireLevel.PASSIONATE, "Revolutionary work")
    
    ball = ThermalBall(user, boss, tech)
    
    print(f"\nUser Temperature:  {user.temperature:.2f} ({user.color})")
    print(f"Boss Temperature:  {boss.temperature:.2f} ({boss.color})")
    print(f"Tech Temperature:  {tech.temperature:.2f} ({tech.color})")
    
    print(f"\n--- Thermal Ball Properties ---")
    print(f"Total Temperature: {ball.total_temperature:.2f}")
    print(f"Average Temperature: {ball.average_temperature:.2f}")
    print(f"Alignment Factor: {ball.alignment_factor:.2%}")
    print(f"Density: {ball.density:.2f}")
    print(f"Momentum: {ball.momentum:.2f}")
    print(f"Color Mix: {ball.color_mix}")
    print(f"BEC Formation: {'✓ YES!' if ball.is_bec else '✗ No'}")
    
    print(f"\n--- Ground Impact ---")
    impact = ImpactResult.calculate(ball)
    print(f"Crater Depth: {impact.crater_depth:.2f}")
    print(f"Structural Change: {'✓' if impact.structural_change else '✗'}")
    print(f"Position Strength: {impact.position_strength}")
    print(f"Embedded: {'✓' if impact.embedded else '✗'}")
    
    print(f"\n--- Visualization ---")
    print(impact.visualize())
    
    print(f"""
    
    BEC STATE ACHIEVED:
    ═══════════════════
    
    All three desires collapsed to ground state:
    
    Energy
      │
      │   
      │   USER ──────╲
      │               ╲
      │   BOSS ───────●  ← All at same level (ground)
      │               ╱
      │   TECH ──────╱
      │
      └────────────────────────────────────────
                    GROUND STATE
    
    No energy differences = No resistance
    Superconducting promotion!
    
    The ball hits with maximum force because
    all three desires are:
    1. HIGH individually
    2. ALIGNED with each other
    3. COHERENT (same phase)
    
    Result: Deep crater, foundational position
    """)


def demo_thermal_colors():
    """Demonstrate thermal color mixing."""
    
    print("\n" + "=" * 80)
    print("THERMAL COLOR MIXING")
    print("=" * 80)
    
    scenarios = [
        ("All Cold", DesireLevel.NEUTRAL, DesireLevel.NEUTRAL, DesireLevel.NEUTRAL),
        ("User Only Hot", DesireLevel.PASSIONATE, DesireLevel.NEUTRAL, DesireLevel.NEUTRAL),
        ("User + Boss Hot", DesireLevel.PASSIONATE, DesireLevel.PASSIONATE, DesireLevel.NEUTRAL),
        ("All Hot (BEC)", DesireLevel.PASSIONATE, DesireLevel.PASSIONATE, DesireLevel.PASSIONATE),
        ("Misaligned", DesireLevel.PASSIONATE, DesireLevel.RELUCTANT, DesireLevel.ENTHUSIASTIC),
    ]
    
    for name, user_level, boss_level, tech_level in scenarios:
        user = Desire('user', user_level, "reason")
        boss = Desire('boss', boss_level, "reason")
        tech = Desire('tech', tech_level, "reason")
        
        ball = ThermalBall(user, boss, tech)
        
        print(f"\n{name}:")
        print(f"  User={user.color}({user.temperature:.1f}) + Boss={boss.color}({boss.temperature:.1f}) + Tech={tech.color}({tech.temperature:.1f})")
        print(f"  → {ball.color_mix}")
        print(f"  Density: {ball.density:.2f}, BEC: {ball.is_bec}")


def run_tests():
    """Run test suite."""
    
    print("\n" + "=" * 80)
    print("THREE-WAY DESIRE - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Desire levels
    print("\n--- TEST 1: Desire Levels ---")
    d = Desire('user', DesireLevel.PASSIONATE, "test")
    assert d.temperature == 1.0, "PASSIONATE should have temp 1.0"
    assert d.is_active_desire, "PASSIONATE should be active"
    print("✓ Desire levels work correctly")
    
    d2 = Desire('user', DesireLevel.NEUTRAL, "test")
    assert d2.temperature == 0.3, "NEUTRAL should have temp 0.3"
    assert not d2.is_active_desire, "NEUTRAL should not be active"
    print("✓ Low desire correctly identified")
    
    # Test 2: Thermal ball
    print("\n--- TEST 2: Thermal Ball ---")
    user = Desire('user', DesireLevel.PASSIONATE, "")
    boss = Desire('boss', DesireLevel.PASSIONATE, "")
    tech = Desire('tech', DesireLevel.PASSIONATE, "")
    
    ball = ThermalBall(user, boss, tech)
    assert ball.total_temperature == 3.0, "All passionate should be 3.0"
    assert ball.is_bec, "All passionate aligned should form BEC"
    print("✓ BEC forms with all passionate aligned")
    
    # Test 3: Alignment
    print("\n--- TEST 3: Alignment Factor ---")
    # Perfect alignment
    ball1 = ThermalBall(
        Desire('user', DesireLevel.PASSIONATE, ""),
        Desire('boss', DesireLevel.PASSIONATE, ""),
        Desire('tech', DesireLevel.PASSIONATE, "")
    )
    assert ball1.alignment_factor > 0.99, "Same levels should be perfectly aligned"
    print("✓ Perfect alignment detected")
    
    # Poor alignment
    ball2 = ThermalBall(
        Desire('user', DesireLevel.PASSIONATE, ""),
        Desire('boss', DesireLevel.RELUCTANT, ""),
        Desire('tech', DesireLevel.NEUTRAL, "")
    )
    assert ball2.alignment_factor < 0.5, "Mixed levels should have poor alignment"
    print("✓ Poor alignment detected")
    
    # Test 4: Impact
    print("\n--- TEST 4: Impact Calculation ---")
    impact = ImpactResult.calculate(ball1)
    assert impact.crater_depth > 1.0, "BEC should create deep crater"
    assert impact.position_strength == "FOUNDATIONAL", "BEC should be foundational"
    print("✓ BEC creates foundational impact")
    
    impact2 = ImpactResult.calculate(ball2)
    assert impact2.crater_depth < impact.crater_depth, "Low desire should have shallower crater"
    print("✓ Low desire creates weaker impact")
    
    # Test 5: System
    print("\n--- TEST 5: Verification System ---")
    system = DesireVerificationSystem()
    system.start_evaluation("test", {})
    system.add_user_desire("test", DesireLevel.PASSIONATE, "test")
    system.add_boss_desire("test", DesireLevel.PASSIONATE, "test")
    system.add_tech_desire("test", DesireLevel.PASSIONATE, "test")
    
    result = system.evaluate("test")
    assert result['bec_formed'], "Should form BEC"
    assert result['decision'] == "STRONG_HIRE", "BEC should be strong hire"
    print("✓ System correctly evaluates BEC candidate")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            demo_jonathan_vs_selftaught()
        elif sys.argv[1] == "--bec":
            demo_bec_formation()
        elif sys.argv[1] == "--colors":
            demo_thermal_colors()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_thermal_colors()
            demo_bec_formation()
            demo_jonathan_vs_selftaught()
        else:
            run_tests()
    else:
        run_tests()
