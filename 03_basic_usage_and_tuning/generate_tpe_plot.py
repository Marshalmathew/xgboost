r"""
Generate Publication-Grade Visualization for Tree-Structured Parzen Estimator (TPE)
Illustrates \ell(x) vs g(x) densities and the resulting acquisition ratio \ell(x) / g(x).
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def generate_tpe_plot(output_path: str = "03_basic_usage_and_tuning/tpe_parzen_densities.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})

    x = np.linspace(0, 10, 500)

    # Simulated densities for a hyperparameter (e.g., max_depth or learning rate in log-space)
    # \ell(x): top 15% trials concentrated around x = 4.5
    ell_x = 0.8 * np.exp(-0.5 * ((x - 4.5) / 0.8)**2) + 0.1 * np.exp(-0.5 * ((x - 7.5) / 1.0)**2)
    # g(x): remaining 85% trials spread across broader parameter space
    g_x = 0.3 * np.exp(-0.5 * ((x - 3.0) / 1.5)**2) + 0.35 * np.exp(-0.5 * ((x - 6.5) / 2.0)**2) + 0.05

    # Likelihood ratio / acquisition signal
    ratio = ell_x / np.maximum(g_x, 1e-4)

    # Color palette
    c_ell = "#0284c7"  # Cyan/Blue
    c_g = "#64748b"    # Slate Gray
    c_ratio = "#10b981" # Emerald Green

    # Panel 1: Parzen Window Densities
    ax1.plot(x, ell_x, label=r"$\ell(x) = p(x \mid y < y^*)$ : Top 15% Best Trials", color=c_ell, linewidth=2.5)
    ax1.fill_between(x, 0, ell_x, color=c_ell, alpha=0.25)

    ax1.plot(x, g_x, label=r"$g(x) = p(x \mid y \geq y^*)$ : Remaining 85% Trials", color=c_g, linewidth=2.0, linestyle="--")
    ax1.fill_between(x, 0, g_x, color=c_g, alpha=0.15)

    ax1.set_ylabel("Probability Density $p(x|y)$", fontsize=11, fontweight="bold")
    ax1.set_title("Tree-Structured Parzen Estimator (TPE): Dual-Density Decomposition", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(0, np.max(ell_x) * 1.15)

    # Annotate high density zone
    opt_x = x[np.argmax(ratio)]
    ax1.annotate(
        "Optimal Sampling Basin\nHigh concentration of best trials",
        xy=(4.5, ell_x[np.argmin(np.abs(x - 4.5))]),
        xytext=(2.2, 0.7),
        arrowprops=dict(facecolor="#0284c7", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        color="#0369a1",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e0f2fe", ec="#0284c7", lw=1)
    )

    # Panel 2: Likelihood Ratio / Acquisition Function
    ax2.plot(x, ratio, label=r"$\mathrm{EI}(x) \propto \frac{\ell(x)}{g(x)}$ (Likelihood Ratio)", color=c_ratio, linewidth=2.5)
    ax2.fill_between(x, 0, ratio, color=c_ratio, alpha=0.2)
    ax2.axvline(opt_x, color="#e11d48", linestyle=":", linewidth=2, label=f"Next Candidate $x^* = {opt_x:.2f}$")

    ax2.set_xlabel("Hyperparameter Search Range $x$ (e.g. Tree Depth or Learning Rate)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Ratio $\frac{\ell(x)}{g(x)}$", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(0, np.max(ratio) * 1.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated publication TPE diagram: {output_path}")

if __name__ == "__main__":
    generate_tpe_plot()
