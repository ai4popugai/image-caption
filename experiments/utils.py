import os
from abc import ABCMeta

import torch

from experiments.EfficientNet_b0.run_base import RunBase


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
    run_instance = phase_module.Phase1()

    return run_instance


def setup_pretrained_model(run_instance: RunBase, snapshot_name: str) -> torch.nn.Module:
    """
    Script set up model by experiment, run, phase and snapshot.

    :param run_instance: RunBase instance class
    :param snapshot_name: name of the snapshot from which we take the model
    :return: model with loaded weights.
    """

    model = run_instance.setup_model()

    # load snapshot
    snapshot_path = os.path.join(run_instance.snapshot_dir, snapshot_name)
    checkpoint = torch.load(snapshot_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    return model
