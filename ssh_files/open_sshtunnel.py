if __name__ == "__main__":
    import time
    from py_experimenter.experimenter import PyExperimenter

    experimenter = PyExperimenter("baselines/config/blackbox_ppo.yaml", use_ssh_tunnel=True, use_codecarbon=False)
    while True:
        time.sleep(3600)
