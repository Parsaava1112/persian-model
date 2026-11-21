import os, random
from pathlib import Path
import numpy as np
import torch

def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}

def save_checkpoint(path, model, optimizer, step, config, best_val, tokenizer_hash):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save({"format_version":3,"model":model.state_dict(),
      "optimizer":optimizer.state_dict() if optimizer else None,"step":int(step),
      "config":dict(config),"best_val":float(best_val),"tokenizer_hash":tokenizer_hash,
      "rng_state":rng_state()},tmp)
    os.replace(tmp,path)

def read_checkpoint(path):
    return torch.load(path,map_location="cpu",weights_only=False)

def load_checkpoint(path,model,optimizer=None,map_location="cpu"):
    ckpt=torch.load(path,map_location=map_location,weights_only=False)
    if ckpt.get("format_version") != 3:
        raise ValueError("Checkpoint format/version mismatch. v7 checkpoints only; old checkpoints are deliberately unsupported.")
    model.load_state_dict(ckpt["model"],strict=True)
    if optimizer is not None and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

def restore_rng(state):
    if not state:return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
