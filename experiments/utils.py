from experiments.classification.gpr.EfficientNet_b0.run_base import RunBase


def setup_run_instance(experiment: str, run: str, phase: str,) -> RunBase:
    """
    Setup run instance

    :param experiment: name of the experiment (e.g., "EfficientNet_b0")
    :param run: name of the run (e.g., "run_16")
    :param phase: name of the phase (e.g., "phase_1")
    :return: run instance
    """
    # Convert experiment, run, and phase to module paths
    run_module = __import__(f'experiments.{experiment}.{run}', fromlist=[phase])
    phase_module = getattr(run_module, phase)
    run_instance = phase_module.Phase()

    return run_instance
