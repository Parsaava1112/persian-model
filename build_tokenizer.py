import argparse,json
from pathlib import Path
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,normalizers,decoders

def stream(paths,cap):
    used=0
    for p in paths:
        with p.open(encoding="utf-8",errors="replace") as f:
            for line in f:
                try: row=json.loads(line)
                except Exception: continue
                text=row.get("text","") if isinstance(row,dict) else ""
                if not isinstance(text,str) or not text.strip(): continue
                remain=cap-used
                if remain<=0:return
                text=text[:remain]; used+=len(text); yield text
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--input",default="data/public/train.jsonl")
    a.add_argument("--out",default="data/tokenizer")
    a.add_argument("--vocab-size",type=int,default=16000)
    a.add_argument("--max-chars",type=int,default=200000000)
    x=a.parse_args()
    paths=[Path(p) for p in x.input.split(",")]
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    tok=Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.normalizer=normalizers.Sequence([normalizers.NFC(),normalizers.Replace("ي","ی"),normalizers.Replace("ك","ک"),normalizers.Replace("ى","ی"),normalizers.Replace("ـ","")])
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    special=["<|unk|>","<|bos|>","<|eos|>","<|pad|>"]
    tr=trainers.BpeTrainer(vocab_size=x.vocab_size,min_frequency=2,special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=True)
    tok.train_from_iterator(stream(paths,x.max_chars),trainer=tr)
    dest=Path(x.out); dest.mkdir(parents=True,exist_ok=True); tok.save(str(dest/"tokenizer.json"))
    report={"requested_vocab_size":x.vocab_size,"actual_vocab_size":tok.get_vocab_size(),
      "max_chars":x.max_chars,"special_tokens":{s:tok.token_to_id(s) for s in special}}
    (dest/"tokenizer_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if tok.get_vocab_size()!=x.vocab_size: raise SystemExit("Tokenizer did not reach requested vocab size; adjust model.json to actual_vocab_size.")
if __name__=="__main__":main()
