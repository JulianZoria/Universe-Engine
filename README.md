# Universe Engine: The Geometric Theory of the Universe (v10)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18074126.svg)](https://doi.org/10.5281/zenodo.18074126)

**Final Release (v10.0)**
*This repository contains the reference implementation and theoretical framework for the Geometric Theory of the Universe (GTU).*

## Overview

The **Universe Engine** is a computational proof-of-concept for the Geometric Theory of the Universe (formerly Kinematic Theory of Matter). It demonstrates that fundamental physical laws—including Gravity, General Relativity, and Quantum Mechanics—are emergent properties of a discrete, deterministic 4D spacetime lattice.

We have identified the "source code" of the Universe. It is not based on differential equations, but on integer arithmetic and geometric probability.

### Key Breakthroughs in v10:

1.  **Emergent Gravity:** Gravity is not a force, but "Grid Strain" resulting from the geometric impossibility of fitting a curved 3-sphere onto a discrete lattice.
2.  **Zero Scatter Prediction:** Confirmed by SPARC galaxy data (Radial Acceleration Relation has ~0 intrinsic scatter).
3.  **The Lattice Constant ($S = 60,001$):** The exact resolution of the spacetime grid has been identified, explaining the mass spectrum of the Standard Model.
4.  **Cosmological Constants:** $\Omega_\Lambda = 0.75$ and $\Omega_m = 0.25$ are derived from first principles (geometry of a 4-sphere), not fitted to data.

## Contents

*   `universe_engine_reference_impl.py`: **The Code.** A Python script that simulates the discrete spacetime lattice. It implements the "Error Diffusion" algorithm that generates gravity.
*   `GTU_Refined_Framework.pdf`: **The Theory.** The final whitepaper detailing the mathematical derivation, the $S=60,001$ proof, and falsifiable predictions.
*   `Geometric_theory_of_the_universe_white_paper_v2.pdf`: **Extended Context.** A broader look at the theory's implications.

## How to Run

The code is written in standard Python 3. It requires no heavy external libraries for the core logic (only `matplotlib` for visualization).

```bash
pip install matplotlib numpy
python universe_engine_reference_impl.py
