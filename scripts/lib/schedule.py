"""Learning-rate schedules and step planning for pretraining."""
import math


def official_lr(step, total, initial):
    """Cosine schedule from the validated P1.5 pretraining run."""
    if not 0 <= step <= total or total < 1 or initial <= 0:
        raise ValueError("invalid schedule")
    return initial * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))


def plan(rows, micro, accum, epochs):
    if min(rows, micro, accum, epochs) < 1:
        raise ValueError("invalid training dimensions")
    microbatches = (rows + micro - 1) // micro
    updates = (microbatches + accum - 1) // accum
    return {
        "microbatches_per_epoch": microbatches,
        "updates_per_epoch": updates,
        "total_updates": epochs * updates,
        "total_microbatches": epochs * microbatches,
        "effective_batch": micro * accum,
    }


def finish_epoch(progress, rows, plan_info, targets_per_epoch):
    """Validate conservation before advancing the resumable cursor."""
    epoch = progress["epoch"] + 1
    if (
        progress["next_sample"] != rows
        or progress["step"] != epoch * plan_info["updates_per_epoch"]
        or progress["micro"] != epoch * plan_info["microbatches_per_epoch"]
        or progress["supervised_tokens"] != epoch * targets_per_epoch
    ):
        raise ValueError("epoch sample, step or token conservation failed")
    progress["epoch"] = epoch
    progress["next_sample"] = 0


class CosineSchedule:
    """Stateful optimizer LR scheduler with exact resume support."""

    def __init__(self, optimizer, lr, total_micro):
        self.optimizer = optimizer
        self.lr = lr
        self.total_micro = total_micro
        self.micro = 0

    def advance(self, micro):
        self.micro = micro
        rate = official_lr(micro, self.total_micro, self.lr)
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def state_dict(self):
        return {"micro": self.micro, "lr": self.lr, "total_micro": self.total_micro}

    def load_state_dict(self, state):
        if state["lr"] != self.lr or state["total_micro"] != self.total_micro:
            raise ValueError("scheduler contract mismatch")
        self.advance(state["micro"])
