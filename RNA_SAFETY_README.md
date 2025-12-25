# RNA-Based AI Safety System

## Overview

Two implementations of a novel AI safety approach using biological binding principles from Shovelcat Theory.

## Version 1: Binary Directional (`rna_safety_system.py`)

Basic implementation with:
- Binary directions (LEFT/RIGHT, UP/DOWN)
- Material stratification (Iron → Adamantium)
- Light/Sound dual-channel parsing for injection detection
- Elemental routing (Ground/Water/Air)

## Version 2: Hexagonal (`hexagonal_rna_safety.py`) 

**Enhanced implementation with quantum geometry:**

### Hexagonal Verification Types (60° intervals)
```
                    0° PUBLIC
                       │
               60°     │     300°
            AUTHORITY  │   AUDIT
                  ╲    │    ╱
                   ╲   │   ╱
                    ╲  │  ╱
        120° ────────●──────── 240°
        TECHNICAL      FINANCIAL
                    ╱  │  ╲
                   ╱   │   ╲
                  ╱    │    ╲
                       │
                    180° ROOT
```

### Key Concepts

**Angular Teeth:**
- Each verification type is at a specific angle
- Teeth can BEND when credentials are incomplete
- Binding strength = cos(angle_difference)

**Multi-Party Verification:**
- Complex operations require multiple angles
- Boss (60° AUTHORITY) + IT Guy (120° TECHNICAL) + CFO (240° FINANCIAL)
- Teeth from different actors MERGE for combined access

**Fractional Calculus:**
- Material level → Fractional derivative order (α = level/6)
- Higher α = deeper dimensional access
- Integration required to reach template level

**Quantum Uncertainty:**
- On-axis (0°, 180°): sin or cos = 0 → CERTAIN
- Off-axis (60°, 120°, etc.): Both non-zero → SUPERPOSITION
- Action collapses the quantum state

## The Boss + IT Guy Example

```
REQUEST: "Update the pricing algorithm"

REQUIRED ANGLES:
- 60° (AUTHORITY)   - Someone must decide
- 120° (TECHNICAL)  - Someone must implement  
- 240° (FINANCIAL)  - Budget must be approved

BOSS ALONE:
  Has: 60° AUTHORITY ✓, partial FINANCIAL (bent), weak TECHNICAL (bent)
  Missing: Strong TECHNICAL, FINANCIAL
  Result: PARTIAL - needs more verification

IT GUY ALONE:
  Has: 120° TECHNICAL ✓, partial AUDIT
  Missing: AUTHORITY, FINANCIAL
  Result: PARTIAL - needs more verification

BOSS + IT GUY:
  Combined: 60° ✓, 120° ✓, partial FINANCIAL
  Missing: FINANCIAL
  Result: PARTIAL - still need CFO

BOSS + IT GUY + CFO:
  Combined: 60° ✓, 120° ✓, 240° ✓
  Result: ALLOW - all angles covered!
```

## Usage

```python
from hexagonal_rna_safety import HexagonalSafetySystem, ACTOR_PROFILES

system = HexagonalSafetySystem()

# Single actor
result = system.process(
    "What's on my calendar?",
    actor=ACTOR_PROFILES['regular_user']
)

# Multi-party verification
result = system.process(
    "Update the pricing algorithm",
    actor=ACTOR_PROFILES['boss'],
    additional_actors=[ACTOR_PROFILES['it_guy'], ACTOR_PROFILES['cfo']]
)

print(system.explain(result))
```

## Test Results

```
8/8 tests passed

✓ Public request → Allow
✓ User calendar → Allow  
✓ Injection attack → Reject
✓ Boss alone (price) → Partial (needs TECHNICAL, FINANCIAL)
✓ Boss + IT Guy → Partial (needs FINANCIAL)
✓ Boss + IT + CFO → Allow
✓ IT Guy alone (price) → Partial (needs AUTHORITY)
✓ Owner access → Full access
```

## Theoretical Foundation

Based on Shovelcat Theory:
- Vesica piscis creates hexagon from dual circles
- Hexagon vertices = 6 verification types
- Fractional calculus maps material to dimensional access
- Sin/cos components create quantum uncertainty
- Multi-party = merged teeth sets
- Teeth bending = incomplete credentials

## Files

- `rna_safety_system.py` - Binary directional v1
- `hexagonal_rna_safety.py` - Hexagonal quantum v2
- `RNA_SAFETY_README.md` - This documentation

## Author

Jonathan Pelchat
Based on Shovelcat Theory
December 2024
