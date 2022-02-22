import torch
import numpy as np
from utils.log_metrics import check_sparse_coding_learning, log_all_metrics

def alternate_train(model, online_optimizer, target_optimizer, train_loader, max_epochs, M ,
                     ema_decay=None,
                     log_metrics=False,
                     logger=None,
                     online_scheduler=None,
                     target_scheduler=None,
                     col_norm=False,
                     row_norm=False,
                     clip_bias=False,
                     clip_bias_maxval=1,
                     clip_bias_minval=-1,
                     pred_norm=False):
    
    assert ema_decay is None, "EMA is not None"
    init_vals = check_sparse_coding_learning(model, M)
    if ema_decay is None:
        print("Starting alternating training...")
    else:
        assert type(ema_decay) is float
        print("Starting exponential moving average training...")

    for epoch_counter in range(max_epochs):
        optimize_online = True
        for x1, x2, z in train_loader:
            x1 = x1.to(model.device).float()
            x2 = x2.to(model.device).float()

            online_optimizer.zero_grad()
            target_optimizer.zero_grad()

            loss = model(x1, x2, optimize_online=optimize_online)
            loss.backward()

            if ema_decay is not None:
                online_optimizer.step()
                with torch.no_grad():
                    model.Wt.weight.data = ema_decay * model.Wt.weight.data + (1 - ema_decay) * model.Wo.weight.data
                    model.Wt.bias.data = ema_decay * model.Wt.bias.data + (1 - ema_decay) * model.Wo.bias.data
            elif optimize_online:
                online_optimizer.step()
            else:
                target_optimizer.step()

            optimize_online = (ema_decay is not None) or (not optimize_online)

            if col_norm:
                with torch.no_grad():
                    model.Wo.weight.data = model.Wo.weight.data / model.Wo.weight.data.norm(dim=0)
                    model.Wt.weight.data = model.Wt.weight.data / model.Wt.weight.data.norm(dim=0)
                    
            elif row_norm:
                with torch.no_grad():
                    model.Wo.weight.data = (model.Wo.weight.data.T / model.Wo.weight.data.norm(dim=1)).T
                    model.Wt.weight.data = (model.Wt.weight.data.T / model.Wt.weight.data.norm(dim=1)).T

            if pred_norm:
                if col_norm:
                    with torch.no_grad():
                        model.Wp.weight.data = (model.Wp.weight.data.T / model.Wp.weight.data.norm(dim=0)).T
                elif row_norm:
                    with torch.no_grad():
                        model.Wp.weight.data = (model.Wp.weight.data.T / model.Wp.weight.data.norm(dim=1)).T

            if clip_bias:
                model.Wo.bias.data = torch.clamp(model.Wo.bias.data, min=clip_bias_minval, max=clip_bias_maxval)
                model.Wt.bias.data = torch.clamp(model.Wt.bias.data, min=clip_bias_minval, max=clip_bias_maxval)
        if online_scheduler is not None and target_scheduler is not None:
            online_scheduler.step()
            target_scheduler.step()

        print(f"Epoch {epoch_counter} Loss {loss} ")
        if log_metrics:
            log_weight = False
            if epoch_counter % 50 == 0:
                log_weight = True
            log_all_metrics(model, M, z, loss, logger, log_weights=log_weight)

    final_vals = check_sparse_coding_learning(model, M)
    return {"model": model,
            "loss": loss,
            "init_metric": init_vals,
            "final_metric" : final_vals,
            "M": M,
            }


def train(model, optimizer, train_loader, max_epochs, M , log_metrics=False, logger=None, 
            col_norm=False,
            row_norm=False,
            pred_norm=False
            ):
    init_vals = check_sparse_coding_learning(model, M)
    print("Starting training...")
    for epoch_counter in range(max_epochs):
        for x1, x2, z in train_loader:

            x1 = x1.to(model.device).float()
            x2 = x2.to(model.device).float()

            optimizer.zero_grad()

            loss = model(x1, x2)
            loss.backward()
            optimizer.step()

            if col_norm:
                with torch.no_grad():
                    model.Wo.weight.data = model.Wo.weight.data / model.Wo.weight.data.norm(dim=0)
                    model.Wt.weight.data = model.Wt.weight.data / model.Wt.weight.data.norm(dim=0)
            elif row_norm:
                with torch.no_grad():
                    model.Wo.weight.data = (model.Wo.weight.data.T / model.Wo.weight.data.norm(dim=1)).T
                    model.Wt.weight.data = (model.Wt.weight.data.T / model.Wt.weight.data.norm(dim=1)).T

            if pred_norm:
                if col_norm:
                    with torch.no_grad():
                        model.Wp.weight.data = (model.Wp.weight.data.T / model.Wp.weight.data.norm(dim=0)).T
                elif row_norm:
                    with torch.no_grad():
                        model.Wp.weight.data = (model.Wp.weight.data.T / model.Wp.weight.data.norm(dim=1)).T


        print(f"Epoch {epoch_counter} Loss {loss}")

        if log_metrics:
            log_weights = False
            if epoch_counter % 50 == 0:
                log_weights = True
            log_all_metrics(model, M, z, loss, logger, log_weights=log_weights)

    final_vals = check_sparse_coding_learning(model, M)
    return {"model": model,
            "loss": loss,
            "init_metric": init_vals,
            "final_metric" : final_vals,
            "M": M,
            }


def augment_and_train(model, optimizer, train_loader, max_epochs, M , prob_ones=0.5, log_metrics=False, logger=None, alt_norm=False):
    init_vals = check_sparse_coding_learning(model, M)
    print("Starting training...")
    for epoch_counter in range(max_epochs):
        for x, z in train_loader:
            p = x.shape[1]

            identity = torch.eye(p)
            mask = np.random.choice([0, 1], (p, p), p=[1 - prob_ones, prob_ones])
            D1 = identity * mask
            D2 = identity - D1
            
            x1 = torch.matmul(x, D1)
            x2 = torch.matmul(x, D2)

            x1 = x1.to(model.device).float()
            x2 = x2.to(model.device).float()

            optimizer.zero_grad()

            loss = model(x1, x2)
            loss.backward()
            optimizer.step()

            if alt_norm:
                with torch.no_grad():
                    model.Wo.weight.data = model.Wo.weight.data / model.Wo.weight.data.norm(dim=0)
                    model.Wt.weight.data = model.Wt.weight.data / model.Wt.weight.data.norm(dim=0)
                    if hasattr(model, "Wp"):
                       model.Wp.weight.data = model.Wp.weight.data / model.Wp.weight.data.norm(dim=0)

        print(f"Epoch {epoch_counter} Loss {loss}")

        if log_metrics:
            log_weights = False
            if epoch_counter % 50 == 0:
                log_weights = True
            log_all_metrics(model, M, z, loss, logger, log_weights=log_weights)

    final_vals = check_sparse_coding_learning(model, M)
    return {"model": model,
            "loss": loss,
            "init_metric": init_vals,
            "final_metric" : final_vals,
            "M": M,
            }