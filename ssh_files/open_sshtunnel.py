import os

if __name__ == "__main__":
    import time
    from py_experimenter.experimenter import PyExperimenter

    if "bigwork" in os.getcwd():
        experimenter = PyExperimenter("/bigwork/nhwpfehl/architectures-in-rl/baselines/blackbox_joined_hpo_nn/config/blackbox_joined_hpo_nn.yaml", use_ssh_tunnel=True, use_codecarbon=False)
    else:
        experimenter = PyExperimenter("/mnt/home/lfehring/MasterThesis/architectures-in-rl/baselines/blackbox_only_hpo_1_layer/config/blackbox_only_hpo.yaml", use_ssh_tunnel=True, use_codecarbon=False)
    
    while True:
        time.sleep(3600)
