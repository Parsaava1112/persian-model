import json
from pathlib import Path
from tokenizers import Tokenizer
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/"configs/train.json").read_text(encoding="utf-8"))
mc=json.loads((root/cfg["model_config"]).read_text(encoding="utf-8"))
tp=root/cfg["tokenizer_dir"]/"tokenizer.json"
if not tp.exists(): raise SystemExit(f"Missing tokenizer: {tp}")
tok=Tokenizer.from_file(str(tp))
assert tok.get_vocab_size()==mc["vocab_size"],(tok.get_vocab_size(),mc["vocab_size"])
for t in ("<|unk|>","<|bos|>","<|eos|>","<|pad|>"):
    assert tok.token_to_id(t) is not None, f"Missing special token {t}"
for p in ("train_data","val_data"):
    path=root/cfg[p]
    if not path.exists(): raise SystemExit(f"Missing {p}: {path}")
print("Preflight OK: tokenizer, vocabulary, special tokens and data paths.")
