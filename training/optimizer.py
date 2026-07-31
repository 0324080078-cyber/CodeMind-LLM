from torch.optim import AdamW

def get_optimizer(model, learning_rate=3e-4, weight_decay=0.1, beta1=0.9, beta2=0.95):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith(".bias") or "layer_norm" in name.lower() or "ln" in name or "embed" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return AdamW(groups, lr=learning_rate, betas=(beta1, beta2), eps=1e-8)
