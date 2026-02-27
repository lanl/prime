#!/usr/bin/env python
"""
PyTorch Lightning ESMC model utilities.
"""
import os
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from esm.models.esmc import ESMC
from esm.pretrained import register_local_model, load_local_model, LOCAL_MODEL_REGISTRY
from esm.tokenization import get_esmc_model_tokenizers

def local_ESMC_300M(local_ckpt_path, device: torch.device | str = "cpu", use_flash_attn=True):
    """ ESMC 300M parameter model loader. """
    def builder(dev=device):
        with torch.device(dev):
            model = ESMC(
                d_model=960,
                n_heads=15,
                n_layers=30,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=use_flash_attn,
            ).eval()
        state_dict = torch.load(local_ckpt_path, map_location=dev)
        model.load_state_dict(state_dict)
        return model
    return builder

def local_ESMC_600M(local_ckpt_path, device: torch.device | str = "cpu", use_flash_attn=True):
    """ ESMC 600M parameter model loader. """
    def builder(dev=device):
        with torch.device(dev):
            model = ESMC(
                d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=use_flash_attn,
            ).eval()
        state_dict = torch.load(local_ckpt_path, map_location=dev)
        model.load_state_dict(state_dict)
        return model
    return builder

def load_from_cache(model_name, cache_dir: str = HF_HUB_CACHE):
    """
    Download and register a local instance of model weights. ESMC does not support 
    loading from a set .cache (only default in /home), so this was written. 
    
    Note that the LOCAL_MODEL_REGISTRY resets
    for each fresh Python process. So in a persistent Jupyter notebook, the model will stay
    in the registry after it is initially registered; however, a restart of the notebook
    will cause it to need to be registered again.

    Params:
        model_name: Model name you want to reference when loading the model. Several names
            are already present in the local registry (like 'esmc_300m'), so this prevents
            using that exact string as the model name if you want to load from cache. So
            just include it in the name (ex. 'esmc_300m_local'), and this should pick up
            on the right model.
            
        cache_dir: Path to your cache directory where you want your models downloaded to.
            Defaults to ~/.cache/huggingface/hub.
    """
    REPO_MAP = {
        "esmc_300m": {
            "repo_id": "EvolutionaryScale/esmc-300m-2024-12",
            "ckpt_path": os.path.join("data", "weights", "esmc_300m_2024_12_v0.pth"),
            "builder": local_ESMC_300M,
        },
        "esmc_600m": {
            "repo_id": "EvolutionaryScale/esmc-600m-2024-12",
            "ckpt_path": os.path.join("data", "weights", "esmc_600m_2024_12_v0.pth"),
            "builder": local_ESMC_600M,
        },
    }

    if model_name in LOCAL_MODEL_REGISTRY:
        print(f"{model_name} registered. Loading model...")
    else:
        print(f"{model_name} not registered, downloading and registering now…")

        # Match key based on model_name substring
        matched_key = next((key for key in REPO_MAP if key in model_name), None)
        if matched_key is None:
            raise ValueError(f"Unknown model name '{model_name}'. Valid keys are: {list(REPO_MAP.keys())}")

        repo_info = REPO_MAP[matched_key]

        # Download (returns path to snapshot dir)
        snapshot_path = snapshot_download(repo_id=repo_info["repo_id"], cache_dir=cache_dir)

        # Build checkpoint path relative to snapshot
        ckpt = os.path.join(snapshot_path, repo_info["ckpt_path"])
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt}")

        register_local_model(model_name, repo_info["builder"](ckpt))
        print(f"{model_name} has been successfully downloaded to {cache_dir} (if needed) and registered!")

    # Load via esm’s API
    client = load_local_model(model_name)
    print(f"Loaded ESMC ({model_name}) from {cache_dir} successfully!")
    return client