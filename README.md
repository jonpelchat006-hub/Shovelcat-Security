# Shovelcat Security Framework

**A mathematically-grounded security system combining Euler's identity, the Golden Ratio, and SHA-256 cryptography.**

```
S = e^(iθ) × φ^(n + h)

Where:
  θ = Phase angle (role: user → owner)
  n = Verification depth (0-5)
  h = SHA256(verification_chain) / 2^256
```

## 🔐 Core Innovation

Traditional access control uses binary permissions. Shovelcat Security uses **complex numbers** to encode security positions in two orthogonal dimensions:

| Dimension | Symbol | What It Encodes | Forgery Difficulty |
|-----------|--------|-----------------|-------------------|
| **Phase** | e^(iθ) | Role (user ↔ owner) | Easy to claim |
| **Depth** | φⁿ | Verification level | Must be earned |
| **Hash** | φʰ | Unique history | O(2^256) - impossible |

### Why It Works

```
OLD: Hacker claims owner     →  e^(iπ) = -1
     Real owner              →  e^(iπ) = -1
     IDENTICAL ❌

NEW: Hacker claims owner     →  e^(iπ) × φ^4.0 = -6.854
     Real owner              →  e^(iπ) × φ^4.94 = -10.798
     DIFFERENT ✓ (unique hash fingerprint)
```

The fractional exponent `h` comes from SHA-256 hashing the user's complete verification history. To forge it, an attacker would need to solve the **preimage problem** - computationally infeasible.

**Verify:** O(n) - microseconds  
**Forge:** O(2^256) - longer than the universe exists

---

## 📁 Repository Structure

```
Shovelcat-Security/
├── crypto_euler_phi.py      # Complete cryptographic system (SHA-256 + Euler + Phi)
├── antimatter_debt.py       # ⚛️ Part 3: Debt as enforcement (matter/antimatter) (NEW!)
├── h_window_system.py       # 📊 O(1) coarse monitoring + geometry RNA minting
├── dual_side_thermal.py     # 🏠 User/Owner wall + radiation/pipe transfer
├── thermal_hierarchy.py     # 🧊 Cryo chambers + thermal zones + security fixes
├── spiral_decay.py          # 🌀 TRUE Euler identity: e^(-λt + iθ) spiral
├── decay_resonance.py       # 📉 π-encrypted decay + cumulative owner energy
├── genesis_resonance.py     # 🎵 Genesis transaction → multi-timescale rhythm
├── behavioral_dna.py        # 🧬 Behavioral DNA + Euler-split RNA
├── fusion_reactor.py        # 🔥 Thermal containment + privacy walls
├── rna_gel_system.py        # ⚡ 3-Direction RNA gel (7.5x speedup + rate limiting)
├── euler_phi_security.py    # Euler × Phi without cryptography
├── user_lifecycle.py        # 9-axis verification & hacker detection
├── three_way_desire.py      # BEC formation & thermal checkpointing
├── upgrade_verification.py  # Three-stage promotion system
├── euler_access_control.py  # Phase windows & department isolation
├── hexagonal_rna_safety.py  # Quantum binding & multi-party verification
├── rna_safety_system.py     # Original binary RNA system
└── README.md
```

---

## ⚛️ Antimatter Debt System (NEW!)

**Part 3: Debt as enforcement at the edges. Antimatter cancels matter!**

### The Three Parts of Security

```
Part 1: Identity/Phase (Euler θ, continuity, role)
Part 2: Cheap tokens from expensive checks (DNA → RNA)
Part 3: Antimatter/Debt enforcement ← THIS
```

### Signed Integrals (Matter + Antimatter)

```
B+ ─────────────────────────────── POSITIVE CAP
   █████████████████████████████
   █████████ RNA+ █████████████   ← capability tokens
      ╱╲         ╱╲         ╱╲
     ╱  ╲       ╱  ╲       ╱  ╲    ← I_base+ (value)
════════════════════════════════════ ZERO ═════════════
     ╲  ╱       ╲  ╱       ╲  ╱    ← I_base- (debt)
      ╲╱         ╲╱         ╲╱
   ░░░░░░░░░ RNA- ░░░░░░░░░░░   ← debt tokens
B- ─────────────────────────────── NEGATIVE CAP
```

### Two Types of Debt

```
TRANSACTIONAL (θ left):          MAINTENANCE (θ right):
From exchange                    From structure
• PAY, SUBSCRIBE, SETTLE         • MAINTAIN, UPGRADE, BACKUP
→ USER-SIDE antimatter           → OWNER-SIDE antimatter
```

### The Enforcement Equation

```
Budget = I_base+ + η×I_rna+ - λ×D_maint - μ×D_txn

More debt = LESS you can do!
Antimatter literally cancels matter!

No debt:        Budget = 12.50
+5.0 txn debt:  Budget = 7.50
+7.0 maint debt: Budget = 0.50 ← DEBT CRISIS!
```

### Debt Tokens (What You MUST Do)

```
MAINTAIN, UPGRADE, BACKUP (maintenance)
PAY, SUBSCRIBE, SETTLE (transactional)
AUDIT, REPORT (compliance)
REPAIR, RECOVER (incident)
```

---

## 🔥 Fusion Reactor System (NEW!)

Solves the threshold problem: What happens when someone is *almost* ready for upgrade?

```
THE PROBLEM:                          THE SOLUTION:
──────────────────────                ──────────────────────

User at θ=85° trying                  Enter FUSION REACTOR
to reach θ=90°                        
                                      ┌─────────────────────┐
• Visible to competitors             │ █████████████████ │
• Failures are public                │ █               █ │
• Pressure to rush                   │ █   T = 0.72    █ │ ← Temp building
• Mistakes happen                    │ █               █ │
                                      │ █████████████████ │
                                      └─────────────────────┘
                                               │
                                      Walls hide process!
                                      Heat builds safely!
                                      Exit when BEC stable!
```

### Three Reactor Types (from Economic Hydrology)

| Type | Heating | Cooling | Capacity | Best For |
|------|---------|---------|----------|----------|
| **TOKAMAK** | 0.8x (slow) | 0.5x (holds heat) | 5 | Careful candidates, mentorship |
| **NIF** | 2.0x (fast) | 1.5x (cools quick) | 1 | Proven candidates, quick validation |
| **STELLARATOR** | 1.0x | 0.7x | 10 | Complex transitions, cohort learning |

### Dual Purpose Walls

```
1. CONTAINMENT - Heat builds safely without external pressure
2. PRIVACY     - Competitors can't see progress or criteria
3. SECRETS     - Internal docs and feedback stay internal  
4. SAFETY      - Failures inside don't affect outside reputation
```

---

## 🧬 Behavioral DNA System (NEW!)

Your behavior IS your identity. RNA usage patterns accumulate into behavioral DNA:

```
┌────────────────────────────────────────────────────────────────────┐
│ EULER SPLIT: e^(iθ) → User RNA + Owner RNA                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Anonymous (θ=0°):  ████████████████████░░░░░░░░░░  100% User     │
│  Creator (θ=90°):   ██████████░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒  50/50        │
│  Owner (θ=180°):    ░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒  100% Owner   │
│                                                                    │
│  █ = User RNA (your resources)                                    │
│  ▒ = Owner RNA (system resources)                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

ANOMALY DETECTION:
  Human vs Bot similarity:    61%
  Human vs Hacker similarity: 10%
  
  Account takeover: ALL 50 events detected as anomalies!
```

### What Gets Tracked

| Pattern | Human | Bot | Hacker |
|---------|-------|-----|--------|
| Timing variance | High (σ≈1.5s) | Zero (σ=0) | Zero (σ=0) |
| Action variety | 60% read, 30% write | 100% same | Destructive |
| Failure rate | ~5% | 0% | ~70% |
| Retry pattern | Rare, delayed | None | Rapid |
| Resource spread | 15-30 unique | 5 same | 100 (scanning) |

---

## ⚡ Performance: RNA Gel System

The RNA Gel provides **7.5x speedup** through tiered verification:

```
┌─────────────────────────────────────────────────────────────────┐
│  Z-AXIS (Authority)     DNA CHECK (expensive, rare)            │
│         │               └── Full crypto verification            │
│         │               └── Releases RNA tokens                 │
│         │                        │                              │
│         │                        ▼                              │
│         │               ┌────────────────┐                      │
│         │               │  RNA TOKEN     │                      │
│         │               │  POOL ████████ │                      │
│         │               └────────────────┘                      │
│         │                      ╱ ╲                              │
│  ───────┼─────────────────────╱───╲─────── X-AXIS (Action)     │
│        ╱│                READ(-1)  WRITE(-2)                    │
│       Y │                                                       │
│  (Resource)                                                     │
└─────────────────────────────────────────────────────────────────┘

DNA-ONLY:  100,000 ops / 100 requests = 1,000 ops/request
RNA GEL:    13,261 ops / 100 requests =   133 ops/request

SPEEDUP: 7.5x faster, 87% requests served by O(1) RNA checks
```

### Token Release Formula

```python
tokens = floor(φ^n × (1 + θ/π))

# Higher authority = more tokens = more ops between re-auth
Anonymous (θ=0°, n=0):   1 token   → re-auth every request
Staff (θ=135°, n=4):    11 tokens  → re-auth every ~11 requests  
Owner (θ=180°, n=5):    22 tokens  → re-auth every ~22 requests
```

This naturally provides **rate limiting**: low-authority users must re-authenticate frequently.

---

## 🧮 Mathematical Foundation

### Euler's Identity: e^(iπ) = -1

Used for **user/owner duality**:
- User at θ=0: e^(i·0) = +1 (pure outsider)
- Owner at θ=π: e^(i·π) = -1 (pure insider)
- User + Owner = +1 + (-1) = 0 (complementary)

### Golden Ratio: 1/φ + 1 = φ

Used for **nested verification depth**:
- Each level CONTAINS all previous levels
- φ⁰=1, φ¹=1.618, φ²=2.618, φ³=4.236, φ⁴=6.854, φ⁵=11.09
- Creates fractal security - can't skip levels

### SHA-256: P ≠ NP Barrier

Used for **unique magnitude fingerprint**:
- Hash of verification chain normalized to [0,1)
- Makes each user's magnitude cryptographically unique
- Verification is fast, forgery is impossible

---

## 🛡️ Security Layers

### Layer 1: Phase (θ) - Who Are You?

```
θ = 0° ────────────────────────────────────── θ = 180°
ANONYMOUS                                      OWNER

    ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌─────┐
    │ANON│→ │REG │→ │PAID│→ │CORE│→ │CREA│→ │STAFF│
    └────┘  └────┘  └────┘  └────┘  └────┘  └─────┘
    
    ← USER SIDE (cos θ > 0) │ STAFF SIDE (cos θ < 0) →
```

### Layer 2: Depth (n) - How Verified?

```
Level 0: SURFACE      │ φ⁰ = 1.000  │ Current identity
Level 1: SHALLOW      │ φ¹ = 1.618  │ + History verification
Level 2: MEDIUM       │ φ² = 2.618  │ + Cross-domain check
Level 3: DEEP         │ φ³ = 4.236  │ + BEC formation
Level 4: PROFOUND     │ φ⁴ = 6.854  │ + Organizational embedding
Level 5: FOUNDATIONAL │ φ⁵ = 11.09  │ + Structural integration
```

### Layer 3: Hash (h) - Prove Your History

```python
h = SHA256(verification_chain) / 2^256  # Value in [0, 1)

# Each user's magnitude is UNIQUE:
alice:   |S| = φ^4.944 = 10.798
bob:     |S| = φ^4.379 = 8.227
charlie: |S| = φ^4.526 = 8.830
```

---

## 🔍 The 9 Verification Axes

Three domains × three axes = 9-dimensional verification:

```
┌───────────┬─────────────────┬────────────────────────────────────────┐
│ Domain    │ Axis            │ What It Checks                         │
├───────────┼─────────────────┼────────────────────────────────────────┤
│ USER      │ identity        │ Who they claim to be                   │
│ USER      │ behavior        │ What they actually do                  │
│ USER      │ value           │ What they contribute                   │
├───────────┼─────────────────┼────────────────────────────────────────┤
│ ADMIN     │ financial       │ Payment history, fraud score           │
│ ADMIN     │ compliance      │ Terms, policy adherence                │
│ ADMIN     │ relationship    │ Account age, depth                     │
├───────────┼─────────────────┼────────────────────────────────────────┤
│ TECH      │ device          │ Consistent devices, environment        │
│ TECH      │ temporal        │ Time patterns, human-like              │
│ TECH      │ anomaly         │ Deviation from baseline                │
└───────────┴─────────────────┴────────────────────────────────────────┘
```

**Hackers can't fake orthogonal domains simultaneously.**

---

## 🚫 Hacker Detection

### The π/4 Wall (Time Barrier)

```
θ = 0°     θ = 30°    θ = 45°
  │          │          ║
  ●──────────●──────────║
ANON       PAID        ║ THE WALL
                       ║
Hackers pile up here → ║ Can't fake TIME
                       ║ Can't fake BEHAVIOR
                       ║ Can't fake CROSS-DOMAIN
```

### What Hackers CAN vs CAN'T Fake

| CAN Fake | CAN'T Fake |
|----------|------------|
| Identity (fake email) | Time (years of history) |
| Payment (stolen card) | Behavior (human patterns) |
| Single domain scores | Cross-domain consistency |
| Phase claims | Magnitude (hash-based) |

---

## 🌡️ BEC Formation (Three-Way Desire)

For boundary crossing (user → creator), requires **Bose-Einstein Condensate** formation:

```
All three parties must DESIRE the outcome:

USER:     "I want this opportunity"     → T_user
BOSS:     "I see potential here"        → T_boss  
TECH:     "This work is valuable"       → T_tech

If all temperatures high AND aligned:
  → BEC forms
  → Ball is dense
  → Impact creates STRUCTURE (deep crater)
  → Position is FOUNDATIONAL
```

---

## 🚀 Quick Start

```python
from crypto_euler_phi import CryptoSecurityPosition, VerificationChain, VerificationEvent, CryptoVerifier

# Create verification chain
chain = VerificationChain(user_id="alice")
chain.add_event(VerificationEvent(
    event_type="identity",
    timestamp=datetime.now(),
    verifier_id="id_service"
))

# Create security position
position = CryptoSecurityPosition(
    user_id="alice",
    theta=math.pi / 4,  # 45° - Long-term user
    chain=chain
)

# Get cryptographic proof
proof = position.get_verification_proof()
print(f"Magnitude: {proof['magnitude']}")  # Unique to alice's history

# Verify claims
verifier = CryptoVerifier()
verifier.register_position(position)
result = verifier.verify_claim("alice", theta, magnitude, chain)
```

---

## 📊 Run Demonstrations

```bash
# Full test suite
python crypto_euler_phi.py

# P≠NP barrier demonstration
python crypto_euler_phi.py --pnp

# Unique magnitudes demo
python crypto_euler_phi.py --unique

# Forgery detection demo
python crypto_euler_phi.py --forgery

# All crypto demos
python crypto_euler_phi.py --all

# RNA Gel - Session lifecycle
python rna_gel_system.py --session

# RNA Gel - Efficiency comparison (7.5x speedup)
python rna_gel_system.py --efficiency

# RNA Gel - Natural rate limiting
python rna_gel_system.py --ratelimit

# RNA Gel - 3D visualization
python rna_gel_system.py --3d

# All RNA Gel demos
python rna_gel_system.py --all

# Behavioral DNA - Euler split visualization
python behavioral_dna.py --split

# Behavioral DNA - Human/Bot/Hacker patterns
python behavioral_dna.py --patterns

# Behavioral DNA - Anomaly detection demo
python behavioral_dna.py --anomaly

# All Behavioral DNA demos
python behavioral_dna.py --all

# Fusion Reactor - Types comparison
python fusion_reactor.py --types

# Fusion Reactor - Privacy walls demo
python fusion_reactor.py --walls

# Fusion Reactor - Complete heating journey
python fusion_reactor.py --journey

# Fusion Reactor - External vs Internal views
python fusion_reactor.py --views

# All Fusion Reactor demos
python fusion_reactor.py --all

# Antimatter Debt - Signed integrals
python antimatter_debt.py --signed

# Antimatter Debt - Two debt types
python antimatter_debt.py --debts

# Antimatter Debt - Budget reduction
python antimatter_debt.py --budget

# Antimatter Debt - Antimatter dominant state
python antimatter_debt.py --antimatter

# Antimatter Debt - Token minting
python antimatter_debt.py --tokens

# Antimatter Debt - Enforcement
python antimatter_debt.py --enforce

# All Antimatter Debt demos
python antimatter_debt.py --all

# H-Window System - Three integrals
python h_window_system.py --integrals

# H-Window System - Steepness split
python h_window_system.py --steepness

# H-Window System - Teeth direction
python h_window_system.py --teeth

# H-Window System - Typed RNA
python h_window_system.py --typed

# H-Window System - Coarse/Fine monitoring
python h_window_system.py --monitor

# All H-Window demos
python h_window_system.py --all

# Dual-Side Thermal - Energy split (half-angle)
python dual_side_thermal.py --split

# Dual-Side Thermal - Phase coupling (Euler!)
python dual_side_thermal.py --coupling

# Dual-Side Thermal - Transfer types
python dual_side_thermal.py --transfer

# Dual-Side Thermal - Complete system
python dual_side_thermal.py --system

# Dual-Side Thermal - Game dev simulation
python dual_side_thermal.py --game

# All Dual-Side demos
python dual_side_thermal.py --all

# Thermal Hierarchy - Zone thresholds (φ powers)
python thermal_hierarchy.py --zones

# Thermal Hierarchy - Secure phase (private nonce)
python thermal_hierarchy.py --secure

# Thermal Hierarchy - Phase mismatch detection
python thermal_hierarchy.py --mismatch

# Thermal Hierarchy - Cryo chamber verification
python thermal_hierarchy.py --cryo

# Thermal Hierarchy - User envelopes
python thermal_hierarchy.py --envelope

# All Thermal Hierarchy demos
python thermal_hierarchy.py --all

# Spiral Decay - Euler's identity e^(iπ) = -1
python spiral_decay.py --euler

# Spiral Decay - 3D spiral trajectory
python spiral_decay.py --spiral

# Spiral Decay - Phase interference
python spiral_decay.py --interference

# Spiral Decay - DNA check timing
python spiral_decay.py --timing

# All Spiral Decay demos
python spiral_decay.py --all

# Decay Resonance - Action costs and authority
python decay_resonance.py --costs

# Decay Resonance - π-encrypted decay rates
python decay_resonance.py --decay

# Decay Resonance - Cumulative owner energy
python decay_resonance.py --energy

# Decay Resonance - Authority limits demo
python decay_resonance.py --limits

# All Decay Resonance demos
python decay_resonance.py --all

# Genesis Resonance - How first transaction sets everything
python genesis_resonance.py --genesis

# Genesis Resonance - Discrete pulses vs waves
python genesis_resonance.py --pulses

# Genesis Resonance - Nine timescales
python genesis_resonance.py --timescales

# Genesis Resonance - Complete system demo
python genesis_resonance.py --system

# All Genesis Resonance demos
python genesis_resonance.py --all

# User lifecycle (9-axis verification)
python user_lifecycle.py --all

# BEC formation
python three_way_desire.py --all
```

---

## ⚡ Efficiency: DNA/RNA Caching

The full cryptographic check is expensive. The **RNA Token System** provides a 1000x speedup through hierarchical caching:

```
DNA CHECK (Expensive):          RNA CHECK (Cheap):
━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━
• Full S = e^(iθ) × φ^(n+h)    • Token exists?
• Verify chain hash             • Not expired?
• Check all 9 axes              • Authority OK?
• O(n) with crypto              • O(1) - instant!
• Releases N tokens             • Consumes 1 token
```

### 3D Security Gel

RNA tokens fill a 3D volume:
- **X (Phase θ):** Who you are → 1 to 2 multiplier
- **Y (Depth φⁿ):** Verification level → 1 to 11 multiplier  
- **Z (Authority):** Access level → 1 to 5 multiplier

**More trust = More volume = More tokens = Fewer expensive checks**

### Natural Rate Limiting

| User Type | Tokens/Check | DNA Frequency |
|-----------|--------------|---------------|
| Anonymous | 3 | Every ~3 requests |
| Paid | 13 | Every ~13 requests |
| Long-term | 31 | Every ~31 requests |
| Creator | 75 | Every ~75 requests |
| Owner | 191 | Every ~191 requests |

**No artificial limits needed - the math handles it!**

### Performance

```
Efficiency Stats:
  RNA Ratio: 95.7%      ← Most checks are cheap
  Speedup: 1049.5x      ← Over 1000x faster!
  Time Saved: 95.6%     ← Almost all computation saved
```

---

This framework is part of **Shovelcat Theory**, a unified approach connecting:

- **Consciousness** as mathematical necessity above 4D complexity
- **Meta-learning** through integration up abstraction hierarchies
- **Economic modeling** via hydrology (basin/flow dynamics)
- **Security** through geometric verification (vesica piscis)

### Key Papers & Resources

- Vesica Piscis geometry and dual-domain verification
- RNA-inspired binding mechanics for access control
- Thermal checkpointing and BEC state transitions
- Fractional calculus mapping for verification levels

---

## 🔮 Future Directions

- [ ] Hardware implementation (FPGA/photonic)
- [ ] Integration with existing IAM systems
- [ ] Formal security proofs
- [ ] Performance benchmarking vs traditional RBAC
- [ ] Multi-party computation extensions

---

## 📄 License

MIT License - See LICENSE file

---

## 👤 Author

**Jonathan Pelchat**  
University of Calgary  
Shovelcat Theory Research

---


---

*"Phase can be claimed, but magnitude must be earned."*
