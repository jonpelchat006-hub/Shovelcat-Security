"""
GENESIS RESONANCE SYSTEM
========================
The first transaction sets everything.

Key Insights:
1. Actions are DISCRETE PULSES (syllables), not continuous waves
2. The GENESIS TRANSACTION sets the base resonance
3. Multi-timescale: seconds → minutes → hours → days → weeks → months → years
4. Base resonance is HIDDEN behind initial investment/ownership data

When a company starts, the owner provides:
- Identity (who they are)
- Initial investment (how much capital)
- Assets owned (what they bring)
- Timestamp (when they start)

From this, we derive the company's unique resonance - its expected
rhythm of operations across all timescales.

Author: Jonathan Pelchat
Based on Shovelcat Theory
"""

import math
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
import statistics

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2
E = math.e

# Timescales in seconds
SECONDS = 1
MINUTES = 60
HOURS = 3600
DAYS = 86400
WEEKS = 604800
MONTHS = 2592000   # 30 days
YEARS = 31536000   # 365 days


# ═══════════════════════════════════════════════════════════════════════════════
# TIMESCALES
# ═══════════════════════════════════════════════════════════════════════════════

class Timescale(Enum):
    """The nested timescales of business operations."""
    
    MOMENT = (SECONDS, "Individual actions")
    BURST = (MINUTES * 5, "Activity bursts")  
    SESSION = (HOURS, "Work sessions")
    SHIFT = (HOURS * 8, "Work shifts")
    DAY = (DAYS, "Business days")
    WEEK = (WEEKS, "Work weeks")
    MONTH = (MONTHS, "Monthly cycles")
    QUARTER = (MONTHS * 3, "Quarterly reviews")
    YEAR = (YEARS, "Annual cycles")
    
    def __init__(self, seconds: int, description: str):
        self.seconds = seconds
        self.description = description
    
    @property
    def period(self) -> float:
        return float(self.seconds)


# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS TRANSACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenesisTransaction:
    """
    The first transaction that sets everything.
    
    This is like the "genesis block" in blockchain - it defines
    the initial conditions from which all resonance is derived.
    """
    
    # Owner identity
    owner_id: str
    owner_name: str
    
    # Initial capital/investment
    initial_investment: float  # In base currency units
    
    # Assets owned at start
    initial_assets: Dict[str, float] = field(default_factory=dict)
    # e.g., {"real_estate": 100000, "equipment": 50000, "inventory": 25000}
    
    # Business characteristics
    business_type: str = "general"
    employee_count: int = 1
    
    # Timestamp - THE critical moment
    timestamp: float = field(default_factory=time.time)
    
    # Derived values (computed on creation)
    genesis_hash: str = field(default="", init=False)
    
    def __post_init__(self):
        """Compute the genesis hash from all inputs."""
        # Create deterministic hash from all genesis data
        data = {
            'owner': self.owner_id,
            'investment': self.initial_investment,
            'assets': sorted(self.initial_assets.items()),
            'type': self.business_type,
            'employees': self.employee_count,
            'timestamp': self.timestamp
        }
        self.genesis_hash = hashlib.sha256(
            str(data).encode()
        ).hexdigest()
    
    @property
    def total_initial_value(self) -> float:
        """Total value at genesis: investment + assets."""
        return self.initial_investment + sum(self.initial_assets.values())
    
    @property
    def asset_complexity(self) -> int:
        """Number of different asset types."""
        return len(self.initial_assets)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE RESONANCE (Derived from Genesis)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BaseResonance:
    """
    The company's fundamental resonance, derived from genesis transaction.
    
    This is HIDDEN behind the genesis data - you can't see the resonance
    directly, only the initial investment/ownership that generates it.
    """
    
    # Source
    genesis: GenesisTransaction
    
    # Derived timescale expectations (computed)
    expected_periods: Dict[Timescale, float] = field(default_factory=dict)
    expected_amplitudes: Dict[Timescale, float] = field(default_factory=dict)
    
    # Phase offsets (when in each cycle we expect activity)
    phase_offsets: Dict[Timescale, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Derive resonance from genesis."""
        self._compute_resonance()
    
    def _compute_resonance(self):
        """
        Derive expected rhythm from genesis transaction.
        
        The math here encodes business patterns:
        - Larger investment → slower, steadier rhythm
        - More assets → more complex harmonic structure
        - Business type → characteristic patterns
        """
        genesis = self.genesis
        
        # Base scaling from investment (log scale for wide range)
        investment_scale = math.log(genesis.total_initial_value + E) / 10
        
        # Complexity from asset diversity
        complexity = 1 + genesis.asset_complexity * 0.1
        
        # Employee scaling
        employee_scale = math.log(genesis.employee_count + 1)
        
        # Genesis hash provides pseudo-random but deterministic variation
        hash_bytes = bytes.fromhex(genesis.genesis_hash[:16])
        hash_values = [b / 255.0 for b in hash_bytes]
        
        # Compute expected periods for each timescale
        for i, scale in enumerate(Timescale):
            # Base period modified by business characteristics
            base_period = scale.period
            
            # Variation from hash (±20%)
            variation = 0.8 + 0.4 * hash_values[i % len(hash_values)]
            
            # Investment affects period (larger = more regular)
            period_modifier = 1.0 / (1 + investment_scale * 0.1)
            
            self.expected_periods[scale] = base_period * variation * period_modifier
            
            # Amplitude (how much activity expected at this scale)
            # More employees = more activity
            base_amplitude = employee_scale * complexity
            amp_variation = hash_values[(i + 3) % len(hash_values)]
            self.expected_amplitudes[scale] = base_amplitude * (0.5 + amp_variation)
            
            # Phase offset (when in the cycle activity peaks)
            # Derived from timestamp of genesis
            phase_seed = genesis.timestamp / scale.period
            self.phase_offsets[scale] = (phase_seed % 1.0) * 2 * PI
    
    def get_expected_phase(self, timescale: Timescale, 
                          at_time: float = None) -> float:
        """Get expected phase at a given time for a timescale."""
        if at_time is None:
            at_time = time.time()
        
        elapsed = at_time - self.genesis.timestamp
        period = self.expected_periods[timescale]
        cycles = elapsed / period
        
        return (cycles * 2 * PI + self.phase_offsets[timescale]) % (2 * PI)


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRETE PULSE (A Single Action/Syllable)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Pulse:
    """
    A single discrete action - one "syllable" of activity.
    
    Not a continuous wave, but a point event in time.
    """
    
    timestamp: float
    actor_id: str
    action_type: str
    magnitude: float = 1.0
    
    # Optional metadata
    resource: str = ""
    success: bool = True
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)


# ═══════════════════════════════════════════════════════════════════════════════
# RHYTHM TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RhythmBucket:
    """Tracks pulses within a time bucket."""
    
    start_time: float
    end_time: float
    pulses: List[Pulse] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.pulses)
    
    @property
    def total_magnitude(self) -> float:
        return sum(p.magnitude for p in self.pulses)
    
    @property
    def center_time(self) -> float:
        return (self.start_time + self.end_time) / 2


class RhythmTracker:
    """
    Tracks the rhythm of pulses across multiple timescales.
    
    Compares actual rhythm to expected resonance.
    """
    
    def __init__(self, resonance: BaseResonance):
        self.resonance = resonance
        self.pulses: List[Pulse] = []
        
        # Buckets for each timescale
        self.buckets: Dict[Timescale, List[RhythmBucket]] = {
            scale: [] for scale in Timescale
        }
        
        # Running statistics
        self.stats: Dict[Timescale, Dict] = {
            scale: {
                'expected_count': 0,
                'actual_count': 0,
                'phase_errors': [],
                'amplitude_errors': []
            }
            for scale in Timescale
        }
    
    def record_pulse(self, pulse: Pulse):
        """Record a new pulse (action/syllable)."""
        self.pulses.append(pulse)
        
        # Update buckets for each timescale
        for scale in Timescale:
            self._update_bucket(scale, pulse)
    
    def _update_bucket(self, scale: Timescale, pulse: Pulse):
        """Update the appropriate bucket for this timescale."""
        period = self.resonance.expected_periods[scale]
        
        # Find or create bucket
        bucket_start = (pulse.timestamp // period) * period
        bucket_end = bucket_start + period
        
        # Check if we need a new bucket
        if not self.buckets[scale] or \
           self.buckets[scale][-1].end_time <= pulse.timestamp:
            new_bucket = RhythmBucket(
                start_time=bucket_start,
                end_time=bucket_end
            )
            self.buckets[scale].append(new_bucket)
        
        # Add pulse to current bucket
        self.buckets[scale][-1].pulses.append(pulse)
    
    def analyze_rhythm(self, timescale: Timescale) -> Dict[str, Any]:
        """Analyze rhythm at a specific timescale."""
        buckets = self.buckets[timescale]
        
        if len(buckets) < 2:
            return {
                'timescale': timescale.name,
                'status': 'insufficient_data',
                'buckets': len(buckets)
            }
        
        # Calculate inter-bucket intervals
        intervals = []
        magnitudes = []
        
        for i, bucket in enumerate(buckets):
            if bucket.count > 0:
                magnitudes.append(bucket.total_magnitude)
                if i > 0 and buckets[i-1].count > 0:
                    interval = bucket.center_time - buckets[i-1].center_time
                    intervals.append(interval)
        
        if not intervals:
            return {
                'timescale': timescale.name,
                'status': 'no_intervals',
                'buckets': len(buckets)
            }
        
        # Compare to expected
        expected_period = self.resonance.expected_periods[timescale]
        expected_amplitude = self.resonance.expected_amplitudes[timescale]
        
        actual_period = statistics.mean(intervals)
        period_variance = statistics.stdev(intervals) if len(intervals) > 1 else 0
        
        actual_amplitude = statistics.mean(magnitudes) if magnitudes else 0
        
        # Calculate resonance match (0-1, higher = better match)
        period_error = abs(actual_period - expected_period) / expected_period
        amplitude_error = abs(actual_amplitude - expected_amplitude) / (expected_amplitude + 0.01)
        
        resonance_match = max(0, 1 - (period_error + amplitude_error) / 2)
        
        return {
            'timescale': timescale.name,
            'status': 'analyzed',
            'buckets': len(buckets),
            'expected_period': expected_period,
            'actual_period': actual_period,
            'period_variance': period_variance,
            'period_match': max(0, 1 - period_error),
            'expected_amplitude': expected_amplitude,
            'actual_amplitude': actual_amplitude,
            'amplitude_match': max(0, 1 - amplitude_error),
            'resonance_match': resonance_match
        }
    
    def get_current_phase_alignment(self, timescale: Timescale) -> Dict[str, Any]:
        """Check if recent activity aligns with expected phase."""
        expected_phase = self.resonance.get_expected_phase(timescale)
        
        # Get recent pulses within this timescale's period
        period = self.resonance.expected_periods[timescale]
        cutoff = time.time() - period
        recent = [p for p in self.pulses if p.timestamp > cutoff]
        
        if not recent:
            return {
                'timescale': timescale.name,
                'expected_phase_degrees': math.degrees(expected_phase),
                'status': 'no_recent_activity'
            }
        
        # Calculate where in the phase cycle the activity occurred
        actual_phases = []
        for pulse in recent:
            elapsed = pulse.timestamp - self.resonance.genesis.timestamp
            cycles = elapsed / period
            phase = (cycles * 2 * PI) % (2 * PI)
            actual_phases.append(phase)
        
        # Circular mean of actual phases
        sin_sum = sum(math.sin(p) for p in actual_phases)
        cos_sum = sum(math.cos(p) for p in actual_phases)
        actual_mean_phase = math.atan2(sin_sum, cos_sum) % (2 * PI)
        
        # Phase difference
        phase_diff = abs(actual_mean_phase - expected_phase)
        if phase_diff > PI:
            phase_diff = 2 * PI - phase_diff
        
        alignment = 1 - (phase_diff / PI)  # 1 = perfect, 0 = opposite
        
        return {
            'timescale': timescale.name,
            'expected_phase_degrees': math.degrees(expected_phase),
            'actual_phase_degrees': math.degrees(actual_mean_phase),
            'phase_diff_degrees': math.degrees(phase_diff),
            'alignment': alignment,
            'status': 'aligned' if alignment > 0.7 else 'drifting'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE GENESIS SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class GenesisResonanceSystem:
    """
    Complete system for genesis-based resonance tracking.
    
    The first transaction sets everything. All subsequent activity
    is compared against the expected rhythm derived from genesis.
    """
    
    def __init__(self):
        self.companies: Dict[str, Dict] = {}
    
    def create_company(self,
                      company_id: str,
                      owner_id: str,
                      owner_name: str,
                      initial_investment: float,
                      initial_assets: Dict[str, float],
                      business_type: str = "general",
                      employee_count: int = 1) -> Dict[str, Any]:
        """
        Create a new company with its genesis transaction.
        
        This is THE moment that sets everything.
        """
        # Create genesis transaction
        genesis = GenesisTransaction(
            owner_id=owner_id,
            owner_name=owner_name,
            initial_investment=initial_investment,
            initial_assets=initial_assets,
            business_type=business_type,
            employee_count=employee_count
        )
        
        # Derive resonance (hidden behind genesis data)
        resonance = BaseResonance(genesis=genesis)
        
        # Create rhythm tracker
        tracker = RhythmTracker(resonance=resonance)
        
        # Store everything
        self.companies[company_id] = {
            'genesis': genesis,
            'resonance': resonance,
            'tracker': tracker,
            'users': {owner_id: {'role': 'owner', 'registered': time.time()}}
        }
        
        return {
            'company_id': company_id,
            'genesis_hash': genesis.genesis_hash,
            'total_initial_value': genesis.total_initial_value,
            'resonance_established': True,
            # The resonance details are HIDDEN
            # Only the genesis data is visible
            'visible': {
                'investment': initial_investment,
                'assets': initial_assets,
                'type': business_type,
                'employees': employee_count
            }
        }
    
    def record_action(self, company_id: str, actor_id: str,
                     action_type: str, magnitude: float = 1.0,
                     resource: str = "") -> Dict[str, Any]:
        """Record an action (pulse/syllable) for a company."""
        company = self.companies.get(company_id)
        if not company:
            return {'error': 'Unknown company'}
        
        pulse = Pulse(
            timestamp=time.time(),
            actor_id=actor_id,
            action_type=action_type,
            magnitude=magnitude,
            resource=resource
        )
        
        company['tracker'].record_pulse(pulse)
        
        # Quick phase check at session timescale
        alignment = company['tracker'].get_current_phase_alignment(Timescale.SESSION)
        
        return {
            'recorded': True,
            'pulse_time': pulse.datetime.isoformat(),
            'total_pulses': len(company['tracker'].pulses),
            'session_alignment': alignment
        }
    
    def get_rhythm_report(self, company_id: str) -> Dict[str, Any]:
        """Get comprehensive rhythm analysis across all timescales."""
        company = self.companies.get(company_id)
        if not company:
            return {'error': 'Unknown company'}
        
        tracker = company['tracker']
        
        report = {
            'company_id': company_id,
            'genesis_hash': company['genesis'].genesis_hash[:16] + '...',
            'total_pulses': len(tracker.pulses),
            'timescales': {}
        }
        
        for scale in Timescale:
            rhythm = tracker.analyze_rhythm(scale)
            phase = tracker.get_current_phase_alignment(scale)
            
            report['timescales'][scale.name] = {
                'rhythm': rhythm,
                'phase': phase
            }
        
        return report
    
    def detect_anomalies(self, company_id: str) -> Dict[str, Any]:
        """Detect rhythm anomalies across timescales."""
        company = self.companies.get(company_id)
        if not company:
            return {'error': 'Unknown company'}
        
        tracker = company['tracker']
        anomalies = []
        
        for scale in Timescale:
            rhythm = tracker.analyze_rhythm(scale)
            phase = tracker.get_current_phase_alignment(scale)
            
            # Check for rhythm mismatch
            if rhythm.get('resonance_match', 1) < 0.5:
                anomalies.append({
                    'type': 'RHYTHM_MISMATCH',
                    'timescale': scale.name,
                    'match': rhythm.get('resonance_match'),
                    'severity': 1 - rhythm.get('resonance_match', 0)
                })
            
            # Check for phase drift
            if phase.get('alignment', 1) < 0.5:
                anomalies.append({
                    'type': 'PHASE_DRIFT',
                    'timescale': scale.name,
                    'alignment': phase.get('alignment'),
                    'drift_degrees': phase.get('phase_diff_degrees'),
                    'severity': 1 - phase.get('alignment', 0)
                })
        
        return {
            'company_id': company_id,
            'has_anomalies': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_genesis_transaction():
    """Demonstrate how genesis transaction sets everything."""
    print("\n" + "=" * 80)
    print("GENESIS TRANSACTION - The First Transaction Sets Everything")
    print("=" * 80)
    
    print("""
    When a company starts, the owner provides:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                   GENESIS TRANSACTION                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  VISIBLE DATA:                                              │
    │  ─────────────                                              │
    │  • Owner: "Alice Chen"                                      │
    │  • Initial Investment: $100,000                             │
    │  • Assets:                                                  │
    │      - Equipment: $25,000                                   │
    │      - Inventory: $15,000                                   │
    │      - Real Estate: $50,000                                 │
    │  • Business Type: "retail"                                  │
    │  • Employees: 5                                             │
    │  • Timestamp: 2024-01-15 09:00:00                          │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  HIDDEN (Derived) - THE BASE RESONANCE:                    │
    │  ──────────────────────────────────────                    │
    │  • Expected rhythm at each timescale                        │
    │  • Phase offsets (when activity peaks)                      │
    │  • Amplitude expectations (how much activity)               │
    │  • Harmonic structure (complexity of patterns)              │
    │                                                             │
    │  This is like the company's "DNA" - you can't see it       │
    │  directly, only the genesis data that generates it.         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    # Create example genesis
    genesis = GenesisTransaction(
        owner_id="alice_001",
        owner_name="Alice Chen",
        initial_investment=100000,
        initial_assets={
            "equipment": 25000,
            "inventory": 15000,
            "real_estate": 50000
        },
        business_type="retail",
        employee_count=5
    )
    
    print(f"Genesis Hash: {genesis.genesis_hash[:32]}...")
    print(f"Total Initial Value: ${genesis.total_initial_value:,.2f}")
    print(f"Asset Complexity: {genesis.asset_complexity} types")
    
    # Derive resonance
    resonance = BaseResonance(genesis=genesis)
    
    print("\n--- Derived Resonance (normally hidden) ---")
    print("\nExpected periods at each timescale:")
    for scale in list(Timescale)[:5]:  # Show first 5
        period = resonance.expected_periods[scale]
        amplitude = resonance.expected_amplitudes[scale]
        print(f"  {scale.name:10}: period={period:10.1f}s, amplitude={amplitude:.2f}")


def demo_discrete_pulses():
    """Demonstrate discrete pulses vs continuous waves."""
    print("\n" + "=" * 80)
    print("DISCRETE PULSES (Syllables) vs Continuous Waves")
    print("=" * 80)
    
    print("""
    WRONG - Continuous wave thinking:
    ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲
    
    
    RIGHT - Discrete pulses (what actually happens):
    
                    Actions throughout the day:
    
    │    │        │  │          │      │    │
    │    │        │  │          │      │    │
    │    │        │  │          │      │    │
    ▼    ▼        ▼  ▼          ▼      ▼    ▼
    ─────────────────────────────────────────────────────
    9am  10am     1pm 2pm       5pm    8am  9am
         Day 1                         Day 2
    
    
    Each pulse is a SYLLABLE:
    ┌─────────────────────────────────────────────────────┐
    │  Pulse #1                                           │
    │  ─────────                                          │
    │  Time: 9:00:23 AM                                   │
    │  Actor: alice                                       │
    │  Action: login                                      │
    │  Magnitude: 1.0                                     │
    ├─────────────────────────────────────────────────────┤
    │  Pulse #2                                           │
    │  ─────────                                          │
    │  Time: 9:05:47 AM                                   │
    │  Actor: alice                                       │
    │  Action: read                                       │
    │  Magnitude: 0.5                                     │
    ├─────────────────────────────────────────────────────┤
    │  Pulse #3                                           │
    │  ─────────                                          │
    │  Time: 9:12:03 AM                                   │
    │  Actor: bob                                         │
    │  Action: write                                      │
    │  Magnitude: 2.0                                     │
    └─────────────────────────────────────────────────────┘
    
    
    The RHYTHM emerges from the pattern of pulses:
    
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  Session 1 (9-10am):  ││ │  │││ │                  │
    │  Session 2 (10-11am): │ ││  │ │ ││                 │
    │  Session 3 (11-12pm): ││││ │││ ││││                │
    │  ...                                                │
    │                                                     │
    │  The DENSITY and TIMING of pulses creates rhythm   │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)


def demo_multi_timescale():
    """Demonstrate multi-timescale resonance."""
    print("\n" + "=" * 80)
    print("MULTI-TIMESCALE RESONANCE")
    print("=" * 80)
    
    print("""
    From MOMENTS to YEARS - all derived from genesis:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  YEARLY     ═══════════════════════════════════════════════════   │
    │             Fiscal cycles, annual reviews, tax seasons             │
    │             Period: ~365 days                                       │
    │                                                                     │
    │  QUARTERLY  ══════════════════════════                             │
    │             Q1/Q2/Q3/Q4 reporting, planning cycles                  │
    │             Period: ~90 days                                        │
    │                                                                     │
    │  MONTHLY    ══════════════════                                     │
    │             Billing, payroll, monthly reports                       │
    │             Period: ~30 days                                        │
    │                                                                     │
    │  WEEKLY     ═══════════                                            │
    │             Work weeks, weekend patterns                            │
    │             Period: ~7 days                                         │
    │                                                                     │
    │  DAILY      ══════                                                 │
    │             Business hours, daily routines                          │
    │             Period: ~24 hours                                       │
    │                                                                     │
    │  SHIFT      ════                                                   │
    │             8-hour work shifts                                      │
    │             Period: ~8 hours                                        │
    │                                                                     │
    │  SESSION    ══                                                     │
    │             Work sessions, meetings                                 │
    │             Period: ~1 hour                                         │
    │                                                                     │
    │  BURST      ═                                                      │
    │             Activity bursts                                         │
    │             Period: ~5 minutes                                      │
    │                                                                     │
    │  MOMENT     ·                                                      │
    │             Individual actions                                      │
    │             Period: ~1 second                                       │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    
    GENESIS seeds ALL these scales:
    
    Genesis → Yearly period → Quarterly → Monthly → Weekly → Daily → ...
                    ↓
            (scaled by investment, employees, assets)
    """)
    
    # Show actual computed timescales
    genesis = GenesisTransaction(
        owner_id="demo",
        owner_name="Demo Corp",
        initial_investment=500000,
        initial_assets={"office": 100000},
        employee_count=20
    )
    
    resonance = BaseResonance(genesis=genesis)
    
    print("\n--- Computed Timescales for Demo Corp ---")
    print("(Initial investment: $500,000, 20 employees)")
    print()
    print("┌────────────┬──────────────────┬────────────────┐")
    print("│ Timescale  │ Expected Period  │ Exp. Amplitude │")
    print("├────────────┼──────────────────┼────────────────┤")
    
    for scale in Timescale:
        period = resonance.expected_periods[scale]
        amplitude = resonance.expected_amplitudes[scale]
        
        # Format period nicely
        if period < 60:
            period_str = f"{period:.1f} sec"
        elif period < 3600:
            period_str = f"{period/60:.1f} min"
        elif period < 86400:
            period_str = f"{period/3600:.1f} hours"
        else:
            period_str = f"{period/86400:.1f} days"
        
        print(f"│ {scale.name:10} │ {period_str:16} │ {amplitude:14.2f} │")
    
    print("└────────────┴──────────────────┴────────────────┘")


def demo_complete_system():
    """Demonstrate the complete genesis resonance system."""
    print("\n" + "=" * 80)
    print("COMPLETE GENESIS RESONANCE SYSTEM")
    print("=" * 80)
    
    system = GenesisResonanceSystem()
    
    # Create a company
    print("\n--- Creating Company (Genesis Transaction) ---")
    
    result = system.create_company(
        company_id="shovelcat_001",
        owner_id="jonathan",
        owner_name="Jonathan Pelchat",
        initial_investment=50000,
        initial_assets={
            "servers": 10000,
            "software": 5000,
            "domain": 100
        },
        business_type="tech",
        employee_count=3
    )
    
    print(f"Company created!")
    print(f"Genesis hash: {result['genesis_hash'][:32]}...")
    print(f"Total initial value: ${result['total_initial_value']:,.2f}")
    print(f"Visible data: {result['visible']}")
    
    # Simulate some activity
    print("\n--- Recording Activity (Pulses) ---")
    
    actions = [
        ("jonathan", "login", 1.0),
        ("jonathan", "read", 0.5),
        ("alice", "write", 2.0),
        ("jonathan", "execute", 1.5),
        ("bob", "read", 0.5),
        ("alice", "delete", 3.0),
    ]
    
    for actor, action, magnitude in actions:
        result = system.record_action(
            "shovelcat_001", actor, action, magnitude
        )
        print(f"  {actor}: {action} (mag={magnitude}) → {result['total_pulses']} pulses")
    
    # Check for anomalies
    print("\n--- Anomaly Detection ---")
    
    anomalies = system.detect_anomalies("shovelcat_001")
    print(f"Has anomalies: {anomalies['has_anomalies']}")
    if anomalies['anomalies']:
        for a in anomalies['anomalies'][:3]:
            print(f"  - {a['type']} at {a['timescale']}: severity={a['severity']:.2f}")
    else:
        print("  (Insufficient data for reliable anomaly detection)")


def demo_rhythm_visualization():
    """Visualize rhythm matching."""
    print("\n" + "=" * 80)
    print("RHYTHM VISUALIZATION")
    print("=" * 80)
    
    print("""
    Expected rhythm (from genesis):
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  EXPECTED:                                                      │
    │                                                                 │
    │  Hour 1    Hour 2    Hour 3    Hour 4    Hour 5    Hour 6     │
    │  ▓▓▓░░░    ▓▓░░░░    ▓▓▓▓░░    ▓▓░░░░    ▓▓▓░░░    ▓▓░░░░    │
    │  │         │         │         │         │         │           │
    │  peaks     quiet     busy      quiet     peaks     quiet       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Actual rhythm (from pulses):
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  ACTUAL:                                                        │
    │                                                                 │
    │  Hour 1    Hour 2    Hour 3    Hour 4    Hour 5    Hour 6     │
    │  ▓▓░░░░    ▓▓░░░░    ▓▓▓▓░░    ▓░░░░░    ▓▓▓░░░    ▓▓░░░░    │
    │  │         │         │         │         │         │           │
    │  close     match!    match!    low!      match!    match!      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    
    COMPARISON:
    
    ┌────────────────┬───────────────────────────────────┐
    │ Timescale      │ Resonance Match                   │
    ├────────────────┼───────────────────────────────────┤
    │ SESSION        │ ▓▓▓▓▓▓▓▓▓░ 90% ✓                 │
    │ SHIFT          │ ▓▓▓▓▓▓▓░░░ 70% ✓                 │
    │ DAY            │ ▓▓▓▓▓▓▓▓░░ 80% ✓                 │
    │ WEEK           │ ▓▓▓▓░░░░░░ 40% ⚠ DRIFT          │
    └────────────────┴───────────────────────────────────┘
    
    
    ANOMALY DETECTED at WEEK timescale!
    Possible causes:
    • Changed work schedule
    • Account compromise  
    • Business model shift
    """)


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 80)
    print("GENESIS RESONANCE SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Genesis transaction
    print("\n--- TEST 1: Genesis Transaction ---")
    genesis = GenesisTransaction(
        owner_id="test",
        owner_name="Test Owner",
        initial_investment=100000,
        initial_assets={"asset1": 50000}
    )
    assert genesis.genesis_hash
    assert genesis.total_initial_value == 150000
    assert genesis.asset_complexity == 1
    print(f"✓ Genesis hash: {genesis.genesis_hash[:16]}...")
    
    # Test 2: Base resonance
    print("\n--- TEST 2: Base Resonance ---")
    resonance = BaseResonance(genesis=genesis)
    assert len(resonance.expected_periods) == len(Timescale)
    assert all(p > 0 for p in resonance.expected_periods.values())
    print(f"✓ Computed {len(resonance.expected_periods)} timescale periods")
    
    # Test 3: Pulse
    print("\n--- TEST 3: Pulse ---")
    pulse = Pulse(
        timestamp=time.time(),
        actor_id="alice",
        action_type="read",
        magnitude=1.5
    )
    assert pulse.magnitude == 1.5
    print(f"✓ Created pulse at {pulse.datetime}")
    
    # Test 4: Rhythm tracker
    print("\n--- TEST 4: Rhythm Tracker ---")
    tracker = RhythmTracker(resonance=resonance)
    tracker.record_pulse(pulse)
    assert len(tracker.pulses) == 1
    print(f"✓ Recorded {len(tracker.pulses)} pulse")
    
    # Test 5: Complete system
    print("\n--- TEST 5: Complete System ---")
    system = GenesisResonanceSystem()
    result = system.create_company(
        company_id="test_co",
        owner_id="owner1",
        owner_name="Owner One",
        initial_investment=50000,
        initial_assets={}
    )
    assert result['resonance_established']
    print(f"✓ Created company with genesis")
    
    # Test 6: Record actions
    print("\n--- TEST 6: Record Actions ---")
    for i in range(5):
        system.record_action("test_co", "actor1", "read", 1.0)
    report = system.get_rhythm_report("test_co")
    assert report['total_pulses'] == 5
    print(f"✓ Recorded {report['total_pulses']} actions")
    
    print(f"\n{'='*80}")
    print("ALL TESTS PASSED ✓")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--genesis":
            demo_genesis_transaction()
        elif sys.argv[1] == "--pulses":
            demo_discrete_pulses()
        elif sys.argv[1] == "--timescales":
            demo_multi_timescale()
        elif sys.argv[1] == "--system":
            demo_complete_system()
        elif sys.argv[1] == "--rhythm":
            demo_rhythm_visualization()
        elif sys.argv[1] == "--all":
            run_tests()
            demo_genesis_transaction()
            demo_discrete_pulses()
            demo_multi_timescale()
            demo_complete_system()
            demo_rhythm_visualization()
        else:
            run_tests()
    else:
        run_tests()
