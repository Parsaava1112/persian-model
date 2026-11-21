import json, sys, torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from danory.model import DanoryGPT
cfg=json.loads((root/"configs/model.json").read_text(encoding="utf-8"))
model=DanoryGPT(cfg)
x=torch.randint(0,cfg["vocab_size"],(1,16))
logits,loss=model(x,x)
assert logits.shape==(1,16,cfg["vocab_size"])
assert loss is not None and torch.isfinite(loss)
loss.backward()
print("Model forward/backward test passed.")
