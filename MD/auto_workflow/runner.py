import json
import logging
import os

from steps import ExecutorStatus, executor_map


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def map_step_id_params(config):
    global_params = {k: v for k, v in config.items() if k != "steps"}
    for step in config["steps"]:
        step_id = step["id"]
        step_params = step.get("params", {})
        if step_id not in executor_map:
            raise ValueError(f"Unknown step id: {step_id}")
        yield executor_map[step_id](global_params, step_params, logger)


def runner(config, logger):
    exp_name = config["exp_name"] if "exp_name" in config else None
    assert exp_name is not None, "exp_name is not specified"
    for status in map_step_id_params(config):
        try:
            if status.state == ExecutorStatus.SUCCESS.state:
                logger.info(f"Step {status.state} succeeded")
            elif status.state == ExecutorStatus.FAILURE.state:
                logger.error(f"Step {status.state} failed")
            elif status.state == ExecutorStatus.SKIPPED.state:
                logger.warning(f"Step {status.state} skipped")
            else:
                logger.error(f"Unknown status: {status.state}")
        except Exception as e:
            logger.error(f"Error occurred while running step: {e}")
            raise


if __name__ == "__main__":
    config = load_config("./configs/run_4.json")
    output_dir = config.get("output_dir", "output")
    exp_name = config.get("exp_name", "default")
    os.makedirs(output_dir, exist_ok=True)
    # setup a verbose logger with logging to a .log file as well as show in terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/{exp_name}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(exp_name)
    logger.info(f"Output directory: {output_dir}")

    runner(config, logger)
