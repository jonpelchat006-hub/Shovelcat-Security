[README.md](https://github.com/user-attachments/files/24320904/README.md)
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
├── euler_phi_security.py    # Euler × Phi without cryptography
├── user_lifecycle.py        # 9-axis verification & hacker detection
├── three_way_desire.py      # BEC formation & thermal checkpointing
├── upgrade_verification.py  # Three-stage promotion system
├── euler_access_control.py  # Phase windows & department isolation
├── hexagonal_rna_safety.py  # Quantum binding & multi-party verification
└── README.md
```

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

# All demos
python crypto_euler_phi.py --all

# User lifecycle (9-axis verification)
python user_lifecycle.py --all

# BEC formation
python three_way_desire.py --all
```

---

## 📚 Theoretical Background

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

## 🙏 Acknowledgments

- Dr. Samira Ebrahimi Kahou (Mila) for research guidance
- Claude (Anthropic) for collaborative development
- The mathematical foundations: Euler, Fibonacci, and the cryptographic community

---

*"Phase can be claimed, but magnitude must be earned."*
