

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class History:
    """Per-epoch metrics, kept for plotting."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome of one training run."""

    test_acc: float
    val_acc: float
    train_acc: float
    best_epoch: int
    epochs_run: int
    n_params: int
    history: History

    @property
    def generalisation_gap(self) -> float:
        
        return self.train_acc - self.test_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        total_loss += criterion(logits, batch_y).item() * batch_y.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    patience: int = 5,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    label_smoothing: float = 0.0,
    verbose: bool = True,
) -> RunResult:
    
    device = device or pick_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = History()
    best_val_acc, best_epoch, best_state = -1.0, 0, None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimiser.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimiser.step()

            running_loss += loss.item() * batch_y.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss, train_acc = running_loss / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        if verbose:
            print(
                f"epoch {epoch:>3}/{epochs}  "
                f"train loss {train_loss:.4f} acc {train_acc:.4f}  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}; best was {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # The test set is read exactly once, after model selection is complete.
    _, test_acc = evaluate(model, test_loader, criterion, device)
    _, final_train_acc = evaluate(model, train_loader, criterion, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return RunResult(
        test_acc=test_acc,
        val_acc=best_val_acc,
        train_acc=final_train_acc,
        best_epoch=best_epoch,
        epochs_run=len(history.train_loss),
        n_params=n_params,
        history=history,
    )
