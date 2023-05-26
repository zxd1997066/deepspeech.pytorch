import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.testing import evaluate

import torch

cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)


@hydra.main(config_path='.', config_name="config")
def hydra_main(cfg: EvalConfig):
    if cfg.device != 'cuda':
        cfg.model.cuda = False
    print(cfg)
    if cfg.precision == "bfloat16":
        print('---- Enable AMP bfloat16')
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            evaluate(cfg=cfg)
    elif cfg.precision == "float16":
        print('---- Enable AMP float16')
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
            evaluate(cfg=cfg)
    else:
        evaluate(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
