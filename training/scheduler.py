import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)) + min_lr_ratio
    return LambdaLR(optimizer, lr_lambda)
