import os

import hydra
import torch

import run as run
import utils as lib


@hydra.main(config_path='config', config_name='default')
def single_experiment_runner(cfg):
    """
    Parses hydra config, check for potential resuming of training
    and launches training
    """

    try:
        try:
            import ray
            ray.cluster_resources()
            lib.LOGGER.info("Experiment running with ray : deactivating TQDM")
            os.environ['TQDM_DISABLE'] = "1"
        except Exception:
            pass
    except ray.exceptions.RaySystemError:
        pass

    cfg.experience.log_dir = lib.expand_path(cfg.experience.log_dir)

    if cfg.experience.resume is not None:
        if os.path.isfile(lib.expand_path(cfg.experience.resume)):
            resume = lib.expand_path(cfg.experience.resume)
        else:
            resume = os.path.join(cfg.experience.log_dir, cfg.experience.experiment_name, 'weights', cfg.experience.resume)

        if not os.path.isfile(resume):
            lib.LOGGER.warning("Checkpoint does not exists")
            return

        at_epoch = torch.load(resume, map_location='cpu')["epoch"]
        if at_epoch >= cfg.experience.max_iter:
            lib.LOGGER.warning(f"Exiting trial, experiment {cfg.experience.experiment_name} already finished")
            return

    elif cfg.experience.maybe_resume:
        state_path = os.path.join(cfg.experience.log_dir, cfg.experience.experiment_name, 'weights', 'rolling.ckpt')
        if os.path.isfile(state_path):
            resume = state_path
            lib.LOGGER.warning(f"Resuming experience because weights were found @ {resume}")
            at_epoch = torch.load(resume, map_location='cpu')["epoch"]
            if at_epoch >= cfg.experience.max_iter:
                lib.LOGGER.warning(f"Exiting trial, experiment {cfg.experience.experiment_name} already finished")
                return
        else:
            resume = None

    else:
        resume = None
        if os.path.isdir(os.path.join(cfg.experience.log_dir, cfg.experience.experiment_name, 'weights')):
            lib.LOGGER.warning(f"Exiting trial, experiment {cfg.experience.experiment_name} already exists")
            return

    metrics = run.run(
        config=cfg,
        base_config=None,
        checkpoint_dir=resume,
    )

    if metrics is not None:
        return metrics[cfg.experience.eval_split][cfg.experience.principal_metric]


if __name__ == '__main__':
    single_experiment_runner()
