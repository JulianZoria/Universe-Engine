# Universe Engine: The Geometric Theory of the Universe (GTU)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18019788.svg)](https://doi.org/10.5281/zenodo.18019788)

**A strictly deterministic, integer-based physics kernel that demonstrates how cosmological parameters and gravitational dynamics emerge from discrete 4D geometry.**

---

### 🌌 The Story: How We Accidentally Wrote the Universe's Source Code

It started three weeks ago with a simple thought before sleep: 
> *"What if the Universe is literally 4D? What if 'Matter' is just energy moving through Time, and 'Energy' is just matter moving through Space?"*

I’m not a physicist. Like many, I'm just curious. I asked an AI if this was crazy. It replied: *"Actually, the math works."*

This rabbit hole led to a 9-version battle between different AI models (Gemini, GPT, Claude). They kept arguing, rewriting formulas, and getting stuck. Why?

**I realized the problem wasn't the AI. It was the foundation of Physics itself.**

Look at the standard definitions taught in school:
*   **Force:** Causes change in momentum.
*   **Momentum:** Mass × Velocity (changed by force).
*   **Mass:** Resistance to force (Inertia).
*   **Inertia:** Resistance to change in momentum.

**It’s a tautology!** A circular logic loop that explains nothing. $X$ is defined by $Y$, and $Y$ by $X$. No wonder the AIs were stuck in infinite loops — they were trained on this circular logic.

#### So, I cheated.

I told the AI: 
> *"Forget physics. You are now a **Game Designer**. Build an imaginary universe from scratch. You are FORBIDDEN from using standard physics definitions. You can only use Geometry, Integers, and Constants ($c$, $h$, $\pi$)."*

**It worked.**

Freed from the "tautology trap," the AI wrote a **Technical Design Document** for a universe based purely on discrete geometry. It’s only 6 pages long.

Then, we wrote code to test it. We ran the simulation.
**And Gravity emerged automatically.** Not because we programmed it, but as a geometric glitch.

This repository contains that code. It’s a working prototype of a geometric universe.

**Check it out yourself. :)**
*This might be interesting for creating physics-based game engines as well as for various physical simulations.*

---

## 📂 Contents

*   **`universe_engine_reference_impl.py`**: The Python reference implementation. A constructive proof that gravity emerges from discrete geometry.
*   **`KTM_v9.pdf`**: The theoretical whitepaper (Geometric Theory of the Universe).
*   **`universe_engine_core_dynamics_certified_v6.pdf`**: The Core Specification (Low-Level Design of the universe).
*   **`Universe_Engine_Verification_Protocol_Signed.pdf`**: Protocol for third-party verification.

## 🚀 How to Run

1.  Ensure you have Python 3.x and NumPy installed:
    ```bash
    pip install numpy matplotlib
    ```
2.  Run the simulation:
    ```bash
    python universe_engine_reference_impl.py
    ```
3.  The script will generate a `galaxy_run_results` folder with:
    *   Rotation Curve graphs.
    *   Force Profile analysis.
    *   Heatmaps of the gravitational potential.

## 🔗 Citation

If you use this code or theory, please cite the Zenodo record:

> Zoria, J. (2025). Universe Engine Specification & Reference Implementation (Kinematic Theory of Matter V9). Zenodo. https://doi.org/10.5281/zenodo.18019788
