import hydra
from omegaconf import DictConfig, OmegaConf

from meta_learning import MetaLearning

#run example:
#python main.py --config-name debug.yaml hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled

@hydra.main(config_path="configs")
def main(cfg: DictConfig) -> None:
    print(cfg)

    meta_learning_framework = MetaLearning(cfg)

    meta_learning_framework.train()

if __name__ == "__main__":
    main()
