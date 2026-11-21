import argparse
from danory.trainer import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Danory GPT v7 — fresh pretraining")
    p.add_argument("--config", default="configs/train.json")
    p.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")
    p.add_argument("--resume", default=None, help="Resume a Danory v7 checkpoint only")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override total schedule length in optimizer steps")
    p.add_argument("--steps-this-run", type=int, default=None,
                   help="Stop after this many additional steps while preserving the configured total schedule")
    a = p.parse_args()
    run(a.config, a.stage, a.resume, a.max_steps, a.steps_this_run)
