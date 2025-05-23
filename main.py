import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import wandb

from utils import set_seed, NullLogger, WandbLogger


@hydra.main(config_path="configs", config_name="main.yaml", version_base="1.3")
def main(cfg):
    set_seed(cfg.seed)

    energy = instantiate(cfg.energy)

    OmegaConf.set_struct(cfg, False)

    output_dir = HydraConfig.get().runtime.output_dir

    if cfg.wandb.use:
        logger = WandbLogger(cfg, output_dir)
    else:
        logger = NullLogger(cfg, output_dir)

    try:
        call(cfg.train_fn, cfg, energy, logger)
    except Exception as e:
        raise e
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
