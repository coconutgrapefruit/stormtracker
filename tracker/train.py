from typing import Tuple
import torch
import tqdm


def train_epoch(model, optimizer, loader, loss_func, device, epoch):
    model.train()
    pbar = tqdm.notebook.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    for x1, x2, y in pbar:
        optimizer.zero_grad()
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        y_hat = model(x1, x2)
        loss = loss_func(y_hat, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def eval_model(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    obs = []
    preds = []
    with torch.inference_mode():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_hat = model(x1, x2)
            obs.append(y)
            preds.append(y_hat)

    return torch.cat(obs), torch.cat(preds)


def calc_loss(x, y, device, area=(70, 120, 0, 220)):
    lat_scale = area[0] - area[2]
    lon_scale = area[3] - area[1]
    scale_mat = torch.tensor([lat_scale ** 2, lon_scale ** 2], dtype=torch.float32, device=device)

    return torch.mean(((x - y)**2) @ scale_mat)

