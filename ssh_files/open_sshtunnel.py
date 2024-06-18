if __name__ == "__main__":
    import time
    from py_experimenter.experimenter import PyExperimenter

    experimenter = PyExperimenter("/bigwork/nhwpfehl/architectures-in-rl/baselines/blackbox_joined_hpo_nn/config/blackbox_joined_hpo_nn.yaml", use_ssh_tunnel=True, use_codecarbon=False)
    while True:
        time.sleep(3600)
