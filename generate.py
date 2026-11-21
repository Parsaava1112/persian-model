import argparse,json,hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
from danory.model import DanoryGPT
from danory.checkpoint import load_checkpoint
from danory.trainer import resolve

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/pretrain_last.pt")
    p.add_argument("--config",default="configs/train.json")
    p.add_argument("--prompt",required=True)
    p.add_argument("--tokens",type=int,default=160)
    p.add_argument("--temperature",type=float,default=.8)
    p.add_argument("--top-k",type=int,default=50)
    p.add_argument("--top-p",type=float,default=.92)
    p.add_argument("--greedy",action="store_true")
    a=p.parse_args()
    cfg=json.loads(resolve(a.config).read_text(encoding="utf-8"))
    mc=json.loads(resolve(cfg["model_config"]).read_text(encoding="utf-8"))
    tp=resolve(cfg["tokenizer_dir"])/"tokenizer.json"
    tok=Tokenizer.from_file(str(tp))
    if tok.get_vocab_size()!=mc["vocab_size"]: raise ValueError("Tokenizer/model vocab mismatch")
    torch.set_num_threads(max(1,int(cfg.get("cpu_threads",4))))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DanoryGPT(mc).to(device)
    ck=load_checkpoint(resolve(a.checkpoint),model,map_location=device)
    digest=hashlib.sha256(tp.read_bytes()).hexdigest()
    if ck.get("tokenizer_hash")!=digest: raise ValueError("Tokenizer fingerprint mismatch")
    ids=tok.encode(a.prompt).ids
    if not ids: raise ValueError("Prompt is empty after tokenization")
    bos=tok.token_to_id("<|bos|>"); eos=tok.token_to_id("<|eos|>")
    x=torch.tensor(([bos] if bos is not None else [])+ids,dtype=torch.long,device=device)[None,:]
    forbidden=[tok.token_to_id(t) for t in ("<|unk|>","<|pad|>","<|bos|>")]
    with torch.inference_mode():
        out=model.generate(x,a.tokens,a.temperature,a.top_k,a.top_p,1.12,eos,forbidden,16,not a.greedy,128,3)
    continuation=out[0,x.size(1):].tolist()
    print("Danory GPT v7 | step:",ck["step"])
    print("\n"+tok.decode(continuation,skip_special_tokens=True))
if __name__=="__main__":main()
