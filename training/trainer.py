import os, math, torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

class CodeMindTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.max_steps = config.get("max_steps", 100000)
        self.grad_accum = config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.save_steps = config.get("save_steps", 1000)
        self.eval_steps = config.get("eval_steps", 500)
        self.log_steps = config.get("logging_steps", 50)
        self.output_dir = config.get("output_dir", "./checkpoints")
        self.fp16 = config.get("fp16", True) and device.type == "cuda"
        self.scaler = GradScaler() if self.fp16 else None
        self.global_step = 0
        self.best_val_loss = float("inf")
        os.makedirs(self.output_dir, exist_ok=True)
        self.use_wandb = HAS_WANDB and os.environ.get("WANDB_API_KEY")
        if self.use_wandb:
            wandb.init(project="CodeMind-LLM", config=config)

    def _loss(self, batch):
        ids = batch["input_ids"].to(self.device)
        lbl = batch["labels"].to(self.device)
        msk = batch["attention_mask"].to(self.device)
        with autocast(enabled=self.fp16):
            return self.model(input_ids=ids, attention_mask=msk, labels=lbl)["loss"]

    def _step(self, batch, idx):
        self.model.train()
        loss = self._loss(batch) / self.grad_accum
        if self.fp16: self.scaler.scale(loss).backward()
        else: loss.backward()
        if (idx + 1) % self.grad_accum == 0:
            if self.fp16: self.scaler.unscale_(self.optimizer)
            gn = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.fp16: self.scaler.step(self.optimizer); self.scaler.update()
            else: self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            return loss.item() * self.grad_accum, gn.item()
        return loss.item() * self.grad_accum, 0.0

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total, n = 0.0, 0
        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            total += self._loss(batch).item()
            n += 1
            if n >= 50: break
        avg = total / max(n, 1)
        print(f"  Val Loss: {avg:.4f} | PPL: {math.exp(min(avg,20)):.2f}")
        return avg

    def _save(self, tag):
        path = os.path.join(self.output_dir, f"checkpoint-{tag}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        torch.save({"step": self.global_step, "opt": self.optimizer.state_dict(), "sched": self.scheduler.state_dict()}, os.path.join(path,"state.pt"))
        print(f"Saved: {path}")

    def train(self):
        print(f"\n{'='*50}\nCodeMind Training | Device:{self.device} | FP16:{self.fp16}\n{'='*50}")
        self.optimizer.zero_grad()
        rl, lc = 0.0, 0
        it = iter(self.train_loader)
        idx = 0
        while self.global_step < self.max_steps:
            try: batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
            loss, gn = self._step(batch, idx)
            idx += 1; rl += loss; lc += 1
            if self.global_step > 0 and self.global_step % self.log_steps == 0:
                avg = rl / lc
                lr = self.scheduler.get_last_lr()[0]
                print(f"Step {self.global_step:6d} | Loss:{avg:.4f} | PPL:{math.exp(min(avg,20)):.2f} | LR:{lr:.2e}")
                if self.use_wandb: wandb.log({"loss":avg,"lr":lr,"step":self.global_step})
                rl, lc = 0.0, 0
            if self.global_step > 0 and self.global_step % self.eval_steps == 0:
                vl = self._validate()
                if vl < self.best_val_loss:
                    self.best_val_loss = vl
                    self._save("best")
            if self.global_step > 0 and self.global_step % self.save_steps == 0:
                self._save(str(self.global_step))
        self._save("final")
        if self.use_wandb: wandb.finish()
        print(f"\nDone! Best val loss: {self.best_val_loss:.4f}")
