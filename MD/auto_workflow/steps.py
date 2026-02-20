import os
import shutil
import subprocess


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
    output_path = params.get("output_path", f"{name_outro}.gro")
    mol_cmd = f"polyply gen_coords -p {output_dir}/{top_file_path} -o {output_dir}/{output_path} -name {mol_name}_{name_outro} -dens 1000"
    sol_cmd = f"polyply gen_coords -p {output_dir}/{solvent_top_file_path} -o {output_dir}/{output_path} -name WATER_{name_outro} -dens 1000"
    try:
        subprocess.run(mol_cmd, shell=True, check=True)
        logger.info("Polymer coordinates generated successfully")
        subprocess.run(sol_cmd, shell=True, check=True)
        logger.info("Water coordinates generated successfully")
        return ExecutorStatus.SUCCESS
    except Exception as e:
        logger.error("Error occurred while generating polymer coordinates")
        logger.error(f"Error details: {e}")
        return ExecutorStatus.FAILURE


executor_map = {
    "polyply:setup": polyply_setup,
    "polyply:gen-params": poly_gen_params,
    "polyply:gen-coords": poly_gen_coords,
}
