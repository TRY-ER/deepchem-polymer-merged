import os
import shutil
import subprocess

from defaults import md_prod_default_params
from utils import GromacsTopUpdater, MDPWritter


class Status:
    def __init__(self, state: str, message: str):
        self.state = state
        self.message = message


class ExecutorStatus:
    SUCCESS = Status(state="SUCCESS", message="__unspecified__")
    FAILURE = Status(state="FAILURE", message="__unspecified__")
    SKIPPED = Status(state="SKIPPED", message="__unspecified__")


def terminal_exec(command: str, logger) -> ExecutorStatus:
    try:
        import subprocess

        subprocess.run(command, shell=True, check=True)
        logger.info(f"Command executed successfully: {command}")
        return ExecutorStatus.SUCCESS
    except Exception as e:
        logger.alert("Ensure environment activated before execution.")
        logger.error(f"Error occurred while executing command: {command}")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE


def polyply_setup(global_args, params, logger):
    output_dir = global_args.get("output_dir", "output")
    src_martini_base = params.get("src_martini_base", "./src/martini_base.itp")
    src_martini_solvent = params.get("src_martini_solvent", "./src/martini_solvent.itp")
    check_cmd = params.get("check_cmd", "import polyply")
    if not os.path.exists(src_martini_base):
        logger.error(f"Martini base file not found: {src_martini_base}")
        return ExecutorStatus.FAILURE
    if not os.path.exists(src_martini_solvent):
        logger.error(f"Martini solvent file not found: {src_martini_solvent}")
        return ExecutorStatus.FAILURE
    try:
        # copy the src_martini_base file to the output directory
        shutil.copy(src_martini_base, output_dir)
        # copy the src_martini_solvent file to the output directory
        shutil.copy(src_martini_solvent, output_dir)
        # execute the check command to ensure polyply is installed
        exec(check_cmd)
        logger.info(f"Polyply setup successful")
        return ExecutorStatus.SUCCESS
    except Exception as e:
        logger.error(f"Error occurred while setting up Polyply")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE


def poly_gen_params(global_args, params, logger):
    output_dir = global_args.get("output_dir", "output")
    mol_name = global_args.get("mol_name", "TEST_CHAIN_MOL")
    seq = params.get("seq", {})
    output_path = params.get("output_path", f"{mol_name}.itp")
    seq_str = ""
    for key, value in seq.items():
        seq_str += f"{key}:{value} "
    cmd = f"polyply gen_params -lib martini3 -o {output_dir}/{output_path} -name {mol_name} -seq {seq_str}"
    try:
        logger.info(f"[+] Running Command >> {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Polymer parameters generated successfully")
        return ExecutorStatus.SUCCESS
    except Exception as e:
        logger.error(f"Error occurred while generating polymer parameters")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE


def poly_gen_coords(global_args, params, logger):
    output_dir = global_args.get("output_dir", "output")
    mol_name = global_args.get("mol_name", "molecule")
    name_outro = params.get("name_outro", "SINGLE")
    itp_path = params.get("itp_path", "test_mol.itp")
    top_file_path = params.get("top_file_path", f"{mol_name}_single.top")
    solvent_top_file_path = params.get(
        "solvent_top_file_path", f"{mol_name}_solvent.top"
    )
    logger.info("Generating single molecule topology file")
    single_mol_content = f"""
    #include "martini_v3.0.0.itp
    #include "{itp_path}

    [ system ]
    {mol_name.upper()}_{name_outro.upper()}

    [ molecules ]
    {mol_name} 1
    """

    with open(f"{output_dir}/{top_file_path}", "w") as top_file:
        top_file.write(single_mol_content)

    logger.info("Single molecule topology file generated successfully")

    logger.info("Generating single water topology file")
    single_water_content = """
    #include "martini_v3.0.0_solvents_v1.itp"
    #include "martini_v3.0.0.itp"

    [ system ]
    WATER_SINGLE

    [ molecules ]
    W 1

    """

    with open(f"{output_dir}/{solvent_top_file_path}", "w") as top_file:
        top_file.write(single_water_content)

    logger.info("Single water topology file generated successfully")
    mol_output_path = params.get("mol_output_path", f"{name_outro}_sol.gro")
    sol_output_path = params.get("sol_output_path", f"{name_outro}_mol.gro")
    mol_cmd = f"polyply gen_coords -p {output_dir}/{top_file_path} -o {output_dir}/{mol_output_path} -name {mol_name}_{name_outro} -dens 1000"
    sol_cmd = f"polyply gen_coords -p {output_dir}/{solvent_top_file_path} -o {output_dir}/{sol_output_path} -name WATER_{name_outro} -dens 1000"
    try:
        logger.info(f"[+] Running Command >> {mol_cmd}")
        subprocess.run(mol_cmd, shell=True, check=True)
        logger.info("Polymer coordinates generated successfully")
        subprocess.run(sol_cmd, shell=True, check=True)
        logger.info(f"[+] Running Command >> {sol_cmd}")
        logger.info("Water coordinates generated successfully")
        return ExecutorStatus.SUCCESS
    except Exception as e:
        logger.error("Error occurred while generating polymer coordinates")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE


def gmx_solvate_box(global_args, params, logger):
    output_dir = global_args.get("output_dir", "output")
    mol_name = global_args.get("mol_name", "molecule")
    solvent_gro_file_path = params.get("solvent_gro_file_path", f"{mol_name}.gro")
    gro_output_path = params.get("gro_output_path", f"{mol_name}_solvated.gro")
    solvated_topology_path = params.get(
        "solvated_topology_path", f"{mol_name}_solvated.top"
    )
    box_size = params.get("box_size", [20, 20, 20])
    command = f"gmx solvate -cs {output_dir}/{solvent_gro_file_path} -o {output_dir}/{gro_output_path} -p {output_dir}/{solvated_topology_path} -box {box_size[0]} {box_size[1]} {box_size[2]}"
    try:
        logger.info(f"[+] Running Command >> {command}")
        subprocess.run(command, shell=True, check=True)
        logger.info("Box solvated successfully")
        return ExecutorStatus.SUCCESS
    except Exception as e:
        logger.error("Error occurred while solvating box")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE


def gmx_insert_mols(global_args, params, logger):
    output_dir = global_args.get("output_dir", "output")
    mol_name = global_args.get("mol_name", "molecule")
    box_input_path = params.get("box_input_path", "solvated_box.gro")
    mol_input_path = params.get("mol_input_path", "test_mol_single.gro")
    solvated_topology_path = params.get("solvated_topology_path", "test_mol.top")
    system_output_path = params.get("system_output_path", "test_system.gro")
    num_mols = params.get("num_mols", 10)
    command = f"gmx insert-molecules -f {output_dir}/{box_input_path} -ci {output_dir}/{mol_input_path} -nmol {num_mols} -o {output_dir}/{system_output_path} -replace W"
    try:
        logger.info(f"[+] Running Command >> {command}")
        subprocess.run(command, shell=True, check=True)
        logger.info("Molecules inserted successfully")
    except Exception as e:
        logger.error("Error occurred while inserting molecules")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE

    try:
        logger.info("[+] Updating topology")
        updater = GromacsTopUpdater(f"{output_dir}/{solvated_topology_path}")
        updater.update_molecules(
            f"{output_dir}/{system_output_path}", mol_name, num_mols, "W"
        )
        updater.write(f"{output_dir}/{solvated_topology_path}")
        return ExecutorStatus.SUCCESS
    except Exception as e:
        logger.error("Error occurred while updating topology")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE


def modify_write_mdp_params(output_dir, source_mdp, params, logger):
    if not os.path.exists(source_mdp) or params["do_override"]:
        logger.info(f"[+] Creating mdp file at >> {source_mdp}")
        mdp_writer = MDPWritter(
            write_type="em_min", output_file=f"{output_dir}/{source_mdp}"
        )
        if "override_params" in params:
            override = params["override_params"]
            mdp_writer.write(override)
        else:
            mdp_writer.write()
    else:
        logger.info(f"[+] Using existing mdp file at >> {params['source_mdp']}")


def gmx_em_min(global_args, params, logger):
    output_dir = global_args.get("output_dir", "output")
    if "source_mdp" in params:
        source_mdp = params["source_mdp"]
        modify_write_mdp_params(output_dir, source_mdp, params, logger)
        return ExecutorStatus.SUCCESS
    else:
        logger.error("Source mdp file not provided")
        return ExecutorStatus.FAILURE


executor_map = {
    "polyply:setup": polyply_setup,
    "polyply:gen-params": poly_gen_params,
    "polyply:gen-coords": poly_gen_coords,
    "gmx:solvate-box": gmx_solvate_box,
    "gmx:insert-mols": gmx_insert_mols,
    "gmx:em-min": gmx_em_min,
}
