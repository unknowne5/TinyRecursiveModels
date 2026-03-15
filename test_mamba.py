import hydra
from omegaconf import DictConfig
from pretrain import PretrainConfig, create_model

class DummyDatasetMetadata:
    vocab_size = 10
    seq_len = 20
    num_puzzle_identifiers = 5

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    config = PretrainConfig(**hydra_config)
    print(f"config.use_mamba = {config.use_mamba}")
    # Don't try to create the full model if dataset or mamba is missing
    extra = config.arch.__pydantic_extra__ or {}
    model_cfg = dict(
        **extra,  
        batch_size=2,
        vocab_size=10,
        seq_len=20,
        num_puzzle_identifiers=5,
        use_mamba=getattr(config, 'use_mamba', False) or extra.get('use_mamba', False),
        causal=False  
    )
    print(f"model_cfg['use_mamba'] = {model_cfg.get('use_mamba')}")

if __name__ == "__main__":
    launch()
