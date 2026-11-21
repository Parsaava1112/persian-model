import csv, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import DanoryGPT
from .data import TokenBlockDataset, collate
from .checkpoint import save_checkpoint, load_checkpoint, read_checkpoint, restore_rng

ROOT=Path(__file__).resolve().parents[1]
def resolve(p):
    p=Path(p); return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def evaluate(model, loader, device, limit):
    old=model.training; model.eval(); total=0.; count=0
    with torch.inference_mode():
        for i,(x,y,m) in enumerate(loader):
            x,y,m=x.to(device),y.to(device),m.to(device)
            _,loss=model(x,y,m)
            n=int((m*y.ne(-100)).sum().item())
            if n and loss is not None and torch.isfinite(loss):
                total += float(loss.item())*n; count += n
            if limit>0 and i+1>=limit: break
    model.train(old)
    return total/count if count else float("nan")
def lr_at(step,cfg):
    warm=int(cfg["warmup_steps"]); total=int(cfg["max_steps"])
    peak=float(cfg["learning_rate"]); floor=float(cfg["min_lr"])
    if step<warm: return peak*(step+1)/max(1,warm)
    progress=min(1.,max(0.,(step-warm)/max(1,total-warm)))
    return floor+0.5*(1+math.cos(math.pi*progress))*(peak-floor)
def run(config_path,stage="pretrain",resume=None,max_steps=None,steps_this_run=None):
    cfg=json.loads(resolve(config_path).read_text(encoding="utf-8"))
    if max_steps is not None: cfg["max_steps"]=int(max_steps)
    if stage not in ("pretrain","sft"): raise ValueError("stage must be pretrain or sft")
    mc=json.loads(resolve(cfg["model_config"]).read_text(encoding="utf-8"))
    torch.set_num_threads(max(1,int(cfg.get("cpu_threads",4))))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    dev=cfg.get("device","auto")
    if dev=="auto": dev="cuda" if torch.cuda.is_available() else "cpu"
    if dev=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(dev)
    seed=int(cfg.get("seed",1337)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from tokenizers import Tokenizer
    tokpath=resolve(cfg["tokenizer_dir"])/"tokenizer.json"
    if not tokpath.is_file(): raise FileNotFoundError(f"Tokenizer missing: {tokpath}")
    tok=Tokenizer.from_file(str(tokpath))
    if tok.get_vocab_size()!=int(mc["vocab_size"]): raise ValueError("Tokenizer vocab size and model vocab_size mismatch")
    data_key="sft_data" if stage=="sft" else "train_data"
    data_path=resolve(cfg[data_key])
    if not data_path.is_file(): raise FileNotFoundError(f"Training data missing: {data_path}. Run scripts/prepare_public_data.py")
    ds=TokenBlockDataset(data_path,tok,mc["block_size"],sft=(stage=="sft"),shuffle_buffer=cfg.get("shuffle_buffer",64))
    loader=DataLoader(ds,batch_size=int(cfg.get("batch_size",1)),num_workers=0,collate_fn=collate)
    val_loader=None
    if stage=="pretrain" and cfg.get("val_data"):
        vp=resolve(cfg["val_data"])
        if vp.is_file():
            vd=TokenBlockDataset(vp,tok,mc["block_size"],shuffle_buffer=0)
            val_loader=DataLoader(vd,batch_size=1,num_workers=0,collate_fn=collate)
    model=DanoryGPT(mc).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),betas=(0.9,0.95),weight_decay=float(cfg.get("weight_decay",0.1)))
    start=0; best=float("inf"); t_hash=sha256(tokpath)
    if stage=="sft" and not resume:
        # SFT starts from this v7 run only; no earlier-version weights are loaded.
        pretrain_path=resolve("checkpoints/pretrain_best.pt")
        if not pretrain_path.is_file():
            pretrain_path=resolve("checkpoints/pretrain_last.pt")
        if not pretrain_path.is_file():
            raise FileNotFoundError("SFT requires this v7 project's pretrain_best.pt or pretrain_last.pt.")
        pre=read_checkpoint(pretrain_path)
        if pre.get("tokenizer_hash")!=t_hash:
            raise ValueError("Pretraining checkpoint tokenizer differs from current SFT tokenizer.")
        model.load_state_dict(pre["model"],strict=True)
        print(f"SFT initialized from this v7 pretraining checkpoint: {pretrain_path}")
    if resume:
        rp=resolve(resume); meta=read_checkpoint(rp)
        if meta.get("tokenizer_hash")!=t_hash: raise ValueError("Tokenizer fingerprint differs from checkpoint.")
        if meta.get("config",{}).get("model_config") and meta["config"].get("model_config")!=cfg.get("model_config"):
            raise ValueError("Model config differs from saved checkpoint.")
        if int(meta.get("config",{}).get("max_steps",cfg["max_steps"])) != int(cfg["max_steps"]):
            raise ValueError("Total max_steps changed. Keep the original LR schedule length when resuming.")
        ck=load_checkpoint(rp,model,opt,device); start=int(ck["step"]); best=float(ck.get("best_val",float("inf")))
        restore_rng(ck.get("rng_state")); print(f"Resuming v7 from step {start}")
    out=resolve(cfg["out_dir"]); logs=resolve(cfg["log_dir"]); out.mkdir(parents=True,exist_ok=True); logs.mkdir(parents=True,exist_ok=True)
    last=out/f"{stage}_last.pt"; bestp=out/f"{stage}_best.pt"; csvp=logs/f"{stage}.csv"
    if not resume and csvp.exists(): csvp.unlink()
    if not csvp.exists():
        with open(csvp,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(["step","loss","lr","grad_norm","val_loss","elapsed_sec"])
    iterator=iter(loader); model.train(); started=time.time()
    accum=int(cfg.get("grad_accum",4)); maxsteps=int(cfg["max_steps"])
    if accum<1: raise ValueError("grad_accum must be >= 1")
    if steps_this_run is not None and int(steps_this_run)<1: raise ValueError("steps_this_run must be positive")
    end_step=min(maxsteps,start+int(steps_this_run)) if steps_this_run is not None else maxsteps
    opt.zero_grad(set_to_none=True)
    for step in range(start,end_step):
        loss_sum=0.
        for micro in range(accum):
            try: batch=next(iterator)
            except StopIteration:
                iterator=iter(loader)
                try: batch=next(iterator)
                except StopIteration: raise RuntimeError("Training data produced zero token blocks.")
            x,y,mask=(z.to(device) for z in batch)
            _,loss=model(x,y,mask)
            if loss is None or not torch.isfinite(loss): raise RuntimeError("Non-finite training loss.")
            (loss/accum).backward(); loss_sum+=float(loss.detach())
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get("grad_clip",1.0)))
        lr=lr_at(step,cfg)
        for group in opt.param_groups: group["lr"]=lr
        opt.step(); opt.zero_grad(set_to_none=True)
        done=step+1; val=float("nan")
        interval=max(1,int(cfg.get("eval_interval",100)))
        if val_loader is not None and (done%interval==0 or done==maxsteps or done==end_step):
            val=evaluate(model,val_loader,device,int(cfg.get("eval_batches",30)))
            if math.isfinite(val) and val<best:
                best=val; save_checkpoint(bestp,model,opt,done,cfg,best,t_hash)
                print(f"New best checkpoint: {bestp} val_loss={best:.5f}")
        save_every=max(1,int(cfg.get("save_interval",100)))
        if done%save_every==0 or done==maxsteps or done==end_step:
            save_checkpoint(last,model,opt,done,cfg,best,t_hash)
        elapsed=time.time()-started
        if done%max(1,int(cfg.get("log_interval",10)))==0 or done==maxsteps or done==end_step:
            msg=f"step={done}/{maxsteps} stage={stage} loss={loss_sum/accum:.4f} lr={lr:.2e} grad_norm={float(grad_norm):.3f} elapsed={elapsed/60:.1f}min"
            if math.isfinite(val): msg+=f" val_loss={val:.5f}"
            print(msg,flush=True)
            with open(csvp,"a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([done,loss_sum/accum,lr,float(grad_norm),val,elapsed])
    if val_loader is not None and not bestp.exists():
        val=evaluate(model,val_loader,device,int(cfg.get("eval_batches",30)))
        if math.isfinite(val):
            best=val; save_checkpoint(bestp,model,opt,maxsteps,cfg,best,t_hash)
    print(f"Training segment finished at step {end_step}/{maxsteps}. Last checkpoint: {last}")
    if bestp.exists(): print(f"Best checkpoint: {bestp} (val_loss={best:.5f})")
