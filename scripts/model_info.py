import json,argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from danory.model import DanoryGPT
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/model.json"); a=p.parse_args()
cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
m=DanoryGPT(cfg)
n=sum(x.numel() for x in m.parameters() if x.requires_grad)
print(f"Parameters: {n:,} ({n/1e6:.3f} M)")
print("Tied embeddings:",m.token_emb.weight.data_ptr()==m.lm_head.weight.data_ptr())
print("Architecture:",cfg)
print(f"FP32 weights only: {n*4/1024**2:.1f} MiB")
