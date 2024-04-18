import tempfile
from py_experimenter.experimenter import PyExperimenter
from omegaconf import OmegaConf


def create_pyexperimenter(
    config: OmegaConf, credentials_file_path="/bigwork/nhwpfehl/architectures-in-rl/config/database_credentials.yml", use_ssh_tunnel: bool = True
) -> PyExperimenter:
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
