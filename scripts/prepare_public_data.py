"""Stream selected public Persian corpora into bounded JSONL files.
Private/local user data is never read or uploaded by this script.
"""
import argparse, hashlib, json, re, shutil, unicodedata
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset

def clean(s):
    s=unicodedata.normalize("NFC",str(s or ""))
    s=s.replace("ي","ی").replace("ك","ک").replace("ى","ی").replace("ـ","").replace("\r"," ")
    s=re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s

def export_split(ds, out, limit, seen, min_chars=40, max_chars=20000, append=False):
    n=0; chars=0
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("a" if append else "w",encoding="utf-8") as f:
        for row in ds:
            text=clean(row.get("text","") if isinstance(row,dict) else "")
            if len(text)<min_chars: continue
            if len(text)>max_chars: text=text[:max_chars]
            # Basic Persian-script ratio check; allows digits and punctuation.
            persian=sum(1 for c in text if "\u0600"<=c<="\u06ff")
            letters=sum(1 for c in text if c.isalpha())
            if letters and persian/letters < 0.45: continue
            key=hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in seen: continue
            seen.add(key)
            f.write(json.dumps({"text":text},ensure_ascii=False)+"\n")
            n+=1; chars+=len(text)
            if n%10000==0: print(f"{out.name}: {n:,} rows, {chars:,} chars",flush=True)
            if limit and n>=limit: break
    print(f"Finished {out}: {n:,} rows, {chars:,} chars")
    return n

def sft_text(row):
    if not isinstance(row,dict): return None
    # Common instruction/response schemas.
    instruction=row.get("instruction") or row.get("prompt") or row.get("question") or ""
    inp=row.get("input") or row.get("context") or ""
    output=row.get("output") or row.get("response") or row.get("answer") or row.get("completion") or ""
    if instruction and output:
        return {"prompt":clean(str(instruction)+"\n"+str(inp)), "output":clean(output)}
    msgs=row.get("messages") or row.get("conversations")
    if isinstance(msgs,list):
        user=[]; assistant=[]
        for m in msgs:
            if not isinstance(m,dict): continue
            role=str(m.get("role",m.get("from",""))).lower()
            content=m.get("content",m.get("value",""))
            if role in ("user","human"): user.append(str(content))
            elif role in ("assistant","gpt"): assistant.append(str(content))
        if user and assistant: return {"prompt":clean(user[-1]),"output":clean(assistant[-1])}
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--naab-train",type=int,default=300000)
    ap.add_argument("--naab-test",type=int,default=20000)
    ap.add_argument("--merged-rows",type=int,default=50000,help="Supplementary PersianCorpus_merged examples; set 0 to disable")
    ap.add_argument("--train-shards",default="0",help="Comma-separated NAAM train shard indexes, e.g. 0,1")
    ap.add_argument("--test-shards",default="2",help="Comma-separated NAAM test shard indexes; shard 2 is the smaller final shard")
    ap.add_argument("--sft-limit",type=int,default=50000)
    ap.add_argument("--include-sft",action="store_true")
    ap.add_argument("--out",default="data/public")
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    # IMPORTANT: the legacy NAAM dataset builder downloads every shard even
    # when streaming=True. Download only explicitly selected shard files.
    cache_dir=out/"_naab_shards"
    cache_dir.mkdir(parents=True,exist_ok=True)
    train_indexes=[int(x.strip()) for x in a.train_shards.split(",") if x.strip()]
    test_indexes=[int(x.strip()) for x in a.test_shards.split(",") if x.strip()]
    if not train_indexes or any(i<0 or i>125 for i in train_indexes):
        raise ValueError("NAAM train shard indexes must be between 0 and 125")
    if len(train_indexes)>4:
        raise ValueError("Select at most 4 NAAM train shards per run to stay within runner disk capacity")
    if not test_indexes or any(i<0 or i>2 for i in test_indexes):
        raise ValueError("NAAM test shard indexes must be between 0 and 2")
    if len(test_indexes)>1:
        raise ValueError("Select one NAAM test shard per run")
    if not 1<=a.naab_train<=1000000 or not 1<=a.naab_test<=100000:
        raise ValueError("Set train rows <= 1,000,000 and test rows <= 100,000")
    if not 0<=a.merged_rows<=200000 or not 1<=a.sft_limit<=100000:
        raise ValueError("Set merged rows <= 200,000 and SFT limit <= 100,000")
    def get_lines(split,indexes,total):
        paths=[]
        for i in indexes:
            filename=(f"data/train-{i:05d}-of-00126.txt" if split=="train"
                      else f"data/test-{i:05d}-of-00003.txt")
            print(f"Downloading selected NAAM shard: {filename}",flush=True)
            paths.append(Path(hf_hub_download(repo_id="SLPL/naab",filename=filename,
                repo_type="dataset",local_dir=str(cache_dir))))
        def rows():
            for path in paths:
                with path.open(encoding="utf-8",errors="replace") as f:
                    for line in f:
                        if line.strip(): yield {"text":line}
        return rows()
    try:
        seen=set()
        export_split(get_lines("train",train_indexes,a.naab_train),out/"train.jsonl",a.naab_train,seen)
        val_seen=set()
        export_split(get_lines("test",test_indexes,a.naab_test),out/"validation.jsonl",a.naab_test,val_seen,min_chars=40)
        if a.merged_rows>0:
            print("Streaming supplementary PersianCorpus_merged rows...")
            merged=load_dataset("mshojaei77/PersianCorpus_merged",split="train",streaming=True)
            # Exclude exact matches from both the selected NAAM train and validation samples.
            merged_seen=seen.union(val_seen)
            export_split(merged,out/"train.jsonl",a.merged_rows,merged_seen,append=True)
    finally:
        shutil.rmtree(cache_dir,ignore_errors=True)
    if a.include_sft:
        print("Loading Persian SFT dataset (fa config) in streaming mode...")
        try:
            sft=load_dataset("amiralifirouzi/Persian-SFT-102K","fa",split="train",streaming=True)
        except Exception as e:
            print(f"Could not load SFT dataset: {e}. Pretraining data is ready; SFT skipped.")
            return
        n=0; seen_s=set()
        with (out/"sft.jsonl").open("w",encoding="utf-8") as f:
            for row in sft:
                item=sft_text(row)
                if not item or len(item["output"])<8: continue
                key=hashlib.sha256((item["prompt"]+"\n"+item["output"]).encode()).hexdigest()
                if key in seen_s: continue
                seen_s.add(key)
                f.write(json.dumps(item,ensure_ascii=False)+"\n"); n+=1
                if n>=a.sft_limit: break
        print(f"SFT rows: {n:,}")
    else:
        print("SFT download skipped by default; review its dataset card/license before opting in.")
    print("Public data preparation completed.")
if __name__=="__main__": main()
