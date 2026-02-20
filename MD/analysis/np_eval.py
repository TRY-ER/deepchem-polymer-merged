import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from numpy.linalg import eigvalsh


def assign_masses(u):
    """
    Assign Martini 3 conventional masses based on bead type prefix.
    T* (Tiny)    → 36 amu  (2 heavy atoms)
    S* (Small)   → 54 amu  (3 heavy atoms)
    All others   → 72 amu  (Regular, 4 heavy atoms)
    Falls back to 72 if name is ambiguous.
    """
    for atom in u.atoms:
        name = atom.name.upper()
        if name.startswith("T"):
            atom.mass = 36.0
        elif name.startswith("S"):
            atom.mass = 54.0
        else:
            atom.mass = 72.0


def analyze(coordinates_path, res_name, plot_output_path):
    u = mda.Universe(coordinates_path)

    # --- Mass assignment based on Martini 3 bead type ---
    assign_masses(u)

    beads = u.select_atoms(f"resname {res_name}")

    # Validate selection
    unique_resnames = np.unique(beads.resnames)
    print(f"    Residue names found in selection : {unique_resnames}")
    print(f"    Total beads selected             : {len(beads)}")

    # ── Thresholds ──────────────────────────────────────────────────────────
    # Adjust INTEGRITY_THRESHOLD_RG based on your chain size.
    # For 300-bead Martini chains, a coiled/compact chain should sit well
    # below 150 Å. Values above 300 Å almost certainly indicate PBC wrapping
    # or numerical explosion.
    INTEGRITY_THRESHOLD_RG = 300.0
    chain_length = 300  # beads per chain (adjust to your system)

    # ── Per-chain metrics ───────────────────────────────────────────────────
    chain_rgs = []
    chain_cogs = []
    chain_coms = []
    chain_end_to_end_distances = []
    chain_ee_vectors = []  # raw end-to-end vectors (for alignment)
    exploded_count = 0

    for i in range(0, len(beads), chain_length):
        chain = beads[i : i + chain_length]

        # Skip incomplete trailing slice (safety guard)
        if len(chain) < chain_length:
            continue

        chain_rg = chain.radius_of_gyration()
        chain_cog = chain.center_of_geometry()
        chain_com = chain.center_of_mass()

        # End-to-end vector and scalar distance
        pos_first = chain.atoms[0].position
        pos_last = chain.atoms[-1].position
        ee_vec = pos_last - pos_first
        chain_end_to_end_distance = np.linalg.norm(ee_vec)

        if chain_rg > INTEGRITY_THRESHOLD_RG:
            exploded_count += 1

        chain_rgs.append(chain_rg)
        chain_cogs.append(chain_cog)
        chain_coms.append(chain_com)
        chain_end_to_end_distances.append(chain_end_to_end_distance)
        chain_ee_vectors.append(ee_vec)

    n_chains = len(chain_rgs)

    # ── Convert to numpy ────────────────────────────────────────────────────
    chain_coms = np.array(chain_coms)  # (N, 3) Å
    chain_ee_vectors = np.array(chain_ee_vectors)  # (N, 3) Å

    # ── [1] Basic chain conformation statistics ─────────────────────────────
    mean_rg = np.mean(chain_rgs)
    mean_end_to_end_distance = np.mean(chain_end_to_end_distances)
    std_rg = np.std(chain_rgs)
    std_end_to_end_distance = np.std(chain_end_to_end_distances)

    # Ideal Gaussian chain: <Ree²> = 6 <Rg²>
    # ratio > 1 → more stretched than a random coil
    # ratio ≈ 1 → random coil behaviour
    ree_rg_ratio = mean_end_to_end_distance**2 / (6.0 * mean_rg**2)

    print(f"[1] CHAIN CONFORMATION")
    print(f"    Mean Rg            : {mean_rg:.3f} Å  ({mean_rg / 10:.3f} nm)")
    print(
        f"    Mean Ree           : {mean_end_to_end_distance:.3f} Å  ({mean_end_to_end_distance / 10:.3f} nm)"
    )
    print(f"    Std Rg             : {std_rg:.3f} Å")
    print(f"    Std Ree            : {std_end_to_end_distance:.3f} Å")
    print(
        f"    Ree²/(6·Rg²) ratio : {ree_rg_ratio:.3f}  (1 = ideal coil, >1 = stretched)"
    )
    print(f"    Exploded chains    : {exploded_count} / {n_chains}")

    # ── [2] Shape anisotropy of the COM cloud ───────────────────────────────
    system_com = np.mean(chain_coms, axis=0)
    centered_coords = chain_coms - system_com  # shift aggregate to origin

    gyration_tensor = np.dot(centered_coords.T, centered_coords) / n_chains
    eigvals = np.sort(eigvalsh(gyration_tensor))
    L1, L2, L3 = eigvals

    numerator = L1**2 + L2**2 + L3**2
    denominator = (L1 + L2 + L3) ** 2
    kappa2 = 1.5 * (numerator / denominator) - 0.5

    print(f"\n[2] SHAPE ANISOTROPY  (κ²: 0 = sphere, 1 = rod)")
    print(f"    Eigenvalues : λ1={L1:.1f}  λ2={L2:.1f}  λ3={L3:.1f}")
    print(f"    κ²          : {kappa2:.4f}")

    if kappa2 < 0.05:
        shape_label = "SPHERICAL"
    elif kappa2 < 0.25:
        shape_label = "GLOBULAR / slightly ellipsoidal"
    else:
        shape_label = "NON-SPHERICAL (rod or disk-like)"
    print(f"    Shape       : {shape_label}")

    # ── [3] Radial shell test ───────────────────────────────────────────────
    # In a real nanoparticle all chain COMs should sit at roughly the same
    # radius from the aggregate centre, forming a shell.
    # A random / cubic-grid arrangement will have a broad radial spread.
    #
    #   CV = σ(r) / μ(r)
    #   CV < 0.20  →  tight shell  (nanoparticle-like)
    #   CV 0.20-0.40 → loose / transitioning
    #   CV > 0.40  →  scattered, no shell
    radial_dists = np.linalg.norm(centered_coords, axis=1)  # Å
    mean_r = np.mean(radial_dists)
    std_r = np.std(radial_dists)
    cv_r = std_r / mean_r if mean_r > 0 else np.nan

    print(f"\n[3] RADIAL SHELL TEST  (CV = σ/μ of radial distances)")
    print(f"    Mean radial dist from centre : {mean_r:.3f} Å  ({mean_r / 10:.3f} nm)")
    print(f"    Std  radial dist             : {std_r:.3f} Å  ({std_r / 10:.3f} nm)")
    print(f"    Coefficient of Variation     : {cv_r:.3f}")

    if cv_r < 0.20:
        shell_label = "TIGHT SHELL — nanoparticle-like ✓"
        is_shell = True
    elif cv_r < 0.40:
        shell_label = "LOOSE / transitioning structure"
        is_shell = False
    else:
        shell_label = "SCATTERED — NOT a shell ✗"
        is_shell = False
    print(f"    Result                       : {shell_label}")

    # ── [4] Radial alignment score ──────────────────────────────────────────
    # For a proper spherical nanoparticle where chains are stretched outward,
    # the end-to-end vector of each chain should align with the radial
    # direction (unit vector from aggregate COM to chain COM).
    #
    #   cos θ = (Ree_vec · r̂) / |Ree_vec|
    #   |cos θ| → 1  : chain radially stretched outward
    #   |cos θ| → 0  : chain randomly oriented
    radial_unit_vecs = centered_coords / np.linalg.norm(
        centered_coords, axis=1, keepdims=True
    )

    ee_magnitudes = np.linalg.norm(chain_ee_vectors, axis=1)  # Å
    valid = ee_magnitudes > 0.1  # exclude collapsed

    ee_unit_vecs = np.where(
        valid[:, None],
        chain_ee_vectors
        / np.where(ee_magnitudes[:, None] > 0.1, ee_magnitudes[:, None], 1.0),
        0.0,
    )

    cos_angles = np.einsum("ij,ij->i", ee_unit_vecs, radial_unit_vecs)
    mean_cos = np.mean(np.abs(cos_angles[valid]))

    print(f"\n[4] RADIAL ALIGNMENT SCORE  (1 = perfect radial stretch, 0 = random)")
    print(f"    Mean |cos θ|  : {mean_cos:.3f}")
    print(f"    Valid chains  : {valid.sum()} / {n_chains}")

    if mean_cos > 0.70:
        align_label = "STRONG radial alignment ✓"
        is_radial = True
    elif mean_cos > 0.50:
        align_label = "MODERATE radial alignment"
        is_radial = False
    else:
        align_label = "WEAK / no radial alignment ✗"
        is_radial = False
    print(f"    Result        : {align_label}")

    # ── [5] Final verdict ───────────────────────────────────────────────────
    is_spherical = kappa2 < 0.05

    print(f"\n{'─' * 60}")
    print("FINAL VERDICT")
    print(f"{'─' * 60}")
    if is_spherical and is_shell and is_radial:
        verdict = "✅ TRUE NANOPARTICLE — spherical shell with radial chain stretch"
    elif is_spherical and is_shell and not is_radial:
        verdict = (
            "⚠️  SPHERICAL SHELL — but chains NOT radially aligned (globule / vesicle?)"
        )
    elif is_spherical and not is_shell:
        verdict = (
            "❌ ISOTROPIC but NOT a shell — likely random / cubic-grid distribution"
        )
    elif not is_spherical and is_shell:
        verdict = (
            "⚠️  SHELL structure present but shape is NON-SPHERICAL (cylinder / disk?)"
        )
    else:
        verdict = "❌ NOT a spherical nanoparticle"
    print(verdict)
    print(f"{'─' * 60}\n")

    # ── Plots ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        f"Polymer Assembly Analysis\n{os.path.basename(coordinates_path)}",
        fontsize=13,
        fontweight="bold",
    )

    # Rg distribution
    axes[0, 0].hist(chain_rgs, bins=30, color="steelblue", alpha=0.8, edgecolor="k")
    axes[0, 0].axvline(
        mean_rg, color="red", linestyle="--", label=f"Mean={mean_rg:.1f} Å"
    )
    axes[0, 0].set_title("Radius of Gyration (Rg)")
    axes[0, 0].set_xlabel("Rg (Å)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    # Ree distribution
    axes[0, 1].hist(
        chain_end_to_end_distances,
        bins=30,
        color="darkorange",
        alpha=0.8,
        edgecolor="k",
    )
    axes[0, 1].axvline(
        mean_end_to_end_distance,
        color="red",
        linestyle="--",
        label=f"Mean={mean_end_to_end_distance:.1f} Å",
    )
    axes[0, 1].set_title("End-to-End Distance (Ree)")
    axes[0, 1].set_xlabel("Ree (Å)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    # COG distribution (x, y, z flattened)
    axes[1, 0].hist(
        np.array(chain_cogs).flatten(),
        bins=50,
        color="seagreen",
        alpha=0.8,
        edgecolor="k",
    )
    axes[1, 0].set_title("Center of Geometry (x, y, z pooled)")
    axes[1, 0].set_xlabel("Coordinate (Å)")
    axes[1, 0].set_ylabel("Frequency")

    # COM distribution (x, y, z flattened)
    axes[1, 1].hist(
        chain_coms.flatten(), bins=50, color="firebrick", alpha=0.8, edgecolor="k"
    )
    axes[1, 1].set_title("Center of Mass (x, y, z pooled)")
    axes[1, 1].set_xlabel("Coordinate (Å)")
    axes[1, 1].set_ylabel("Frequency")

    # Radial distance distribution — KEY new plot
    axes[2, 0].hist(
        radial_dists, bins=30, color="mediumpurple", alpha=0.8, edgecolor="k"
    )
    axes[2, 0].axvline(
        mean_r, color="red", linestyle="--", label=f"Mean={mean_r:.1f} Å"
    )
    axes[2, 0].set_title(f"Radial Distance from Aggregate Centre  (CV={cv_r:.3f})")
    axes[2, 0].set_xlabel("Radial distance (Å)")
    axes[2, 0].set_ylabel("Number of chains")
    axes[2, 0].legend()

    # |cos θ| distribution — KEY new plot
    axes[2, 1].hist(
        np.abs(cos_angles[valid]), bins=20, color="teal", alpha=0.8, edgecolor="k"
    )
    axes[2, 1].axvline(
        mean_cos, color="red", linestyle="--", label=f"Mean={mean_cos:.3f}"
    )
    axes[2, 1].set_title(f"Radial Alignment  |cos θ|  (mean={mean_cos:.3f})")
    axes[2, 1].set_xlabel("|cos θ|")
    axes[2, 1].set_ylabel("Number of chains")
    axes[2, 1].legend()

    plt.tight_layout()
    out_png = os.path.join(plot_output_path, "distribution_plots.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"    Plot saved → {out_png}\n")

    # Return summary dict for optional downstream use
    return {
        "file": coordinates_path,
        "n_chains": n_chains,
        "mean_rg_A": mean_rg,
        "mean_ree_A": mean_end_to_end_distance,
        "std_rg_A": std_rg,
        "std_ree_A": std_end_to_end_distance,
        "ree_rg_ratio": ree_rg_ratio,
        "kappa2": kappa2,
        "cv_r": cv_r,
        "mean_cos": mean_cos,
        "exploded": exploded_count,
        "verdict": verdict,
    }


if __name__ == "__main__":
    coordinate_paths = [
        "../auto_workflow/1us_run_data/1us_whole_chains.gro",
        "../auto_workflow/1us_run_data/prod_50ns.gro",
        "../auto_workflow/1us_run_data/prod.gro",
        "../auto_workflow/1us_run_data/em.gro",
        "../auto_workflow/1us_run_data/solvated.gro",
        "../auto_workflow/1us_run_data/system_mol_100.gro",
    ]
    output_paths = [
        "./output/1us_whole",
        "./output/50ns",
        "./output/10ns",
        "./output/em",
        "./output/solvated",
        "./output/system_mol_100",
    ]

    RES_NAME = "PEG2 PLA2"
    all_results = []

    for coord_path, out_path in zip(coordinate_paths, output_paths):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print(f"\n{'=' * 60}")
        print(f"Analysing >> {coord_path}")
        print(f"{'=' * 60}")
        result = analyze(coord_path, RES_NAME, out_path)
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = (
        f"{'File':<22} {'Rg(Å)':>8} {'Ree(Å)':>8} "
        f"{'κ²i':>7} {'CV_r':>7} {'|cosθ|':>7}  Verdict"
    )
    print(header)
    print("-" * 90)
    for r in all_results:
        fname = os.path.basename(r["file"])
        print(
            f"{fname:<22} {r['mean_rg_A']:>8.2f} {r['mean_ree_A']:>8.2f} "
            f"{r['kappa2']:>7.4f} {r['cv_r']:>7.3f} {r['mean_cos']:>7.3f}  {r['verdict']}"
        )
    print("=" * 90)
