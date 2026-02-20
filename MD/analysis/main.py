import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from numpy.linalg import eigvalsh


def analyze(coordinates_path, res_name, plot_output_path):
    u = mda.Universe(coordinates_path)
    u.atoms.masses = 36

    beads = u.select_atoms(f"resname {res_name}")
    # need to extract the with name T* or S* to extract the bead types for the atom (if mentioned)
    # accordingly assign masses conditionally
    # verify the the rg_threshold is optimal for our case

    chain_rgs = []
    chain_cogs = []
    chain_coms = []
    chain_end_to_end_distances = []
    exploded_count = 0
    INTEGRITY_THRESHOLD_RG = 300.0

    chain_length = 300
    for i in range(0, len(beads), chain_length):
        chain = beads[i : i + chain_length]
        chain_rg = chain.radius_of_gyration()
        # print("rg >>", chain_rg)
        chain_cog = chain.center_of_geometry()
        # print("cog >>", chain_cog)

        chain_com = chain.center_of_mass()
        # print("com >>", chain_com)

        # calculate end to end distance
        chain_end_to_end_distance = np.linalg.norm(
            chain[-1].position - chain[0].position
        )
        # print("end to end distance >>", chain_end_to_end_distance)

        if chain_rg > INTEGRITY_THRESHOLD_RG:
            exploded_count += 1

        chain_rgs.append(chain_rg)
        chain_cogs.append(chain_cog)
        chain_coms.append(chain_com)
        chain_end_to_end_distances.append(chain_end_to_end_distance)

    # print("[+] Number of chains:", len(chain_rgs))
    # print("[+] Number of exploded chains:", exploded_count)
    # print("[+] Radius of gyration of each chain:", chain_rgs)
    # print("[+] Center of mass of each chain:", chain_cogs)

    # mean
    mean_rg = np.mean(chain_rgs)
    mean_end_to_end_distance = np.mean(chain_end_to_end_distances)

    # std
    std_rg = np.std(chain_rgs)
    std_end_to_end_distance = np.std(chain_end_to_end_distances)

    print("[+] Mean radius of gyration:", mean_rg)
    print("[+] Mean end to end distance:", mean_end_to_end_distance)
    print("[+] Standard deviation of radius of gyration:", std_rg)
    print("[+] Standard deviation of end to end distance:", std_end_to_end_distance)

    # Morpholigical analysis
    chain_coms = np.array(chain_coms)
    system_com = np.mean(chain_coms, axis=0)

    centered_coords = chain_coms - system_com

    gyration_tensor = np.dot(centered_coords.T, centered_coords) / len(chain_coms)

    eigvals = np.sort(eigvalsh(gyration_tensor))
    L1, L2, L3 = eigvals
    print(f"Eigenvalues: λ1={L1:.1f}, λ2={L2:.1f}, λ3={L3:.1f}")

    # 5. Relative Shape Anisotropy (Kappa Squared)
    # 0 = Sphere, 1 = Line
    numerator = L1**2 + L2**2 + L3**2
    denominator = (L1 + L2 + L3) ** 2
    kappa2 = 1.5 * (numerator / denominator) - 0.5

    print(f"Relative Shape Anisotropy (κ²): {kappa2:.4f}")

    if kappa2 < 0.05:
        print("CONCLUSION: The chains are arranged in a SPHERICAL pattern.")
    elif kappa2 < 0.25:
        print("CONCLUSION: The arrangement is GLOBULAR but slightly distorted.")
    else:
        print("CONCLUSION: The arrangement is NON-SPHERICAL (Rod or Disk-like).")

    # save graphs on distribution of rg, cog, com, end-to-end distances
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(chain_rgs, bins=50, color="blue", alpha=0.7)
    plt.title("Distribution of Radius of Gyration (rg)")
    plt.xlabel("rg")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 2)
    # Flattening 3D coordinates to avoid matplotlib treating them as multiple datasets
    plt.hist(np.array(chain_cogs).flatten(), bins=50, color="green", alpha=0.7)
    plt.title("Distribution of Center of Geometry (cog)")
    plt.xlabel("cog")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 3)
    # chain_coms is already a numpy array from morphological analysis
    plt.hist(chain_coms.flatten(), bins=50, color="red", alpha=0.7)
    plt.title("Distribution of Center of Mass (com)")
    plt.xlabel("com")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    plt.hist(chain_end_to_end_distances, bins=50, color="purple", alpha=0.7)
    plt.title("Distribution of End-to-End Distances")
    plt.xlabel("End-to-End Distance")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{plot_output_path}/distribution_plots.png")
    # plt.show()


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
    output_path = "./output/system_mol_100"
    for i, j in zip(coordinate_paths, output_paths):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("Analysing for file >>", i)
        # RES_NAME = "PEG40PLA60"
        RES_NAME = "PEG2 PLA2"
        analyze(i, RES_NAME, output_path)
