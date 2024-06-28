import tempfile
from py_experimenter.experimenter import PyExperimenter, ResultProcessor
from omegaconf import OmegaConf
from typing import Dict
import os


def create_pyexperimenter(config: OmegaConf, use_ssh_tunnel: bool = True) -> PyExperimenter:
    if "nhwpfehl" in os.getcwd():
        credentials_file_path = "/bigwork/nhwpfehl/architectures-in-rl/config/database_credentials.yml"
    elif "mnt" in os.getcwd():
        credentials_file_path = "/mnt/home/lfehring/MasterThesis/architectures-in-rl/config/database_credentials.yml"
    else:
        credentials_file_path = "/home/lukas/Desktop/architectures-in-rl/config/database_credentials.yml"
        use_ssh_tunnel = False

    py_experimenter_config = OmegaConf.create({"PY_EXPERIMENTER": config["PY_EXPERIMENTER"]})
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as tmpfile:
        OmegaConf.save(py_experimenter_config, tmpfile)
        tmpfile_path = tmpfile.name
        experimenter = PyExperimenter(
            experiment_configuration_file_path=tmpfile_path,
            database_credential_file_path=credentials_file_path,
            use_ssh_tunnel=use_ssh_tunnel,
            use_codecarbon=False,
        )
        experimenter.create_table()
    return experimenter


def log_results(result_processor: ResultProcessor, logs: Dict, max_counter: int = 10, counter=0):
    # I need to wrap the funciton, becaue the executed jobs may be distributed on various nodes
    # If so there needs to be a seperate ssh passthrough on every node, however they might be
    # closed in one thread finishes before the other, meaning I need to reclose them
    try:
        result_processor.process_logs(logs)

    except Exception as e:
        if counter > max_counter:
            raise e
        log_results(result_processor, logs, max_counter, counter + 1)
