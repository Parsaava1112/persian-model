import json,torch,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from tokenizers import Tokenizer
from danory.model import DanoryGPT
mc=json.loads((root/"configs/model.json").read_text())
tok=Tokenizer.from_file(str(root/"data/tokenizer/tokenizer.json"))
model=DanoryGPT(mc)
ids=tok.encode("سلام، این یک آزمایش است.").ids
x=torch.tensor([ids[:min(len(ids),mc["block_size"])]],dtype=torch.long)
logits,loss=model(x,x)
assert logits.shape[-1]==mc["vocab_size"]
assert loss is not None and torch.isfinite(loss)
print("Smoke test passed. Logits:",tuple(logits.shape),"loss:",float(loss))
