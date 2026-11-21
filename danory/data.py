import json, random
from pathlib import Path
import torch
from torch.utils.data import IterableDataset

class TokenBlockDataset(IterableDataset):
    def __init__(self, path, tokenizer, block_size, sft=False, shuffle_buffer=64):
        super().__init__()
        self.path, self.tokenizer = str(path), tokenizer
        self.block_size, self.sft = int(block_size), bool(sft)
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.bos = tokenizer.token_to_id("<|bos|>")
        self.eos = tokenizer.token_to_id("<|eos|>")
        if not Path(self.path).is_file():
            raise FileNotFoundError(self.path)

    def _records(self):
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try: row = json.loads(line)
                except Exception: continue
                if not isinstance(row, dict): continue
                if self.sft:
                    prompt = str(row.get("prompt", row.get("instruction", "")) or "")
                    extra = str(row.get("input", "") or "")
                    answer = str(row.get("output", row.get("response", "")) or "")
                    if not answer.strip(): continue
                    if extra: prompt = prompt + "\n" + extra
                    p = self.tokenizer.encode(prompt).ids
                    a = self.tokenizer.encode(answer).ids
                    cap = self.block_size
                    overhead = int(self.bos is not None) + int(self.eos is not None)
                    room = cap + 1 - overhead
                    if room < 2: continue
                    a = a[:room]
                    p = p[-max(0, room-len(a)):] if len(a) < room else []
                    ids = ([self.bos] if self.bos is not None else []) + p + a + ([self.eos] if self.eos is not None else [])
                    yield ids, 1 + len(p)
                else:
                    text = row.get("text", "")
                    if not isinstance(text, str) or not text.strip(): continue
                    ids = self.tokenizer.encode(text).ids
                    if self.bos is not None: ids.insert(0, self.bos)
                    if self.eos is not None: ids.append(self.eos)
                    if len(ids) >= 2: yield ids, 0

    def _shuffled(self):
        if self.shuffle_buffer <= 1:
            yield from self._records(); return
        buf=[]
        for rec in self._records():
            buf.append(rec)
            if len(buf)>=self.shuffle_buffer:
                i=random.randrange(len(buf)); yield buf.pop(i)
        random.shuffle(buf); yield from buf

    def _item(self, ids, prompt_len=0):
        x=torch.zeros(self.block_size,dtype=torch.long)
        y=torch.full((self.block_size,),-100,dtype=torch.long)
        mask=torch.zeros(self.block_size,dtype=torch.float32)
        n=min(self.block_size+1,len(ids))
        if n<2: return x,y,mask
        seq=torch.tensor(ids[:n],dtype=torch.long); length=n-1
        x[:length]=seq[:-1]; y[:length]=seq[1:]
        if self.sft:
            mask[:length]=(torch.arange(length)+1 >= prompt_len).float()
        else: mask[:length]=1.0
        return x,y,mask

    def __iter__(self):
        if self.sft:
            for ids,prompt_len in self._shuffled():
                yield self._item(ids,prompt_len)
            return
        buf=[]; width=self.block_size+1
        for ids,_ in self._shuffled():
            buf.extend(ids)
            while len(buf)>=width:
                yield self._item(buf[:width])
                del buf[:self.block_size]
        if len(buf)>=2: yield self._item(buf)

def collate(batch):
    return tuple(torch.stack(items) for items in zip(*batch))
