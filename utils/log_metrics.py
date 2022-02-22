import torch
import numpy as np
import wandb 
import matplotlib.pyplot as plt
import seaborn as sns 

def check_sparse_coding_learning(model, M):
    with torch.no_grad():
        Wo = model.Wo.weight.cpu().detach().numpy()
    d = model.d
    Wo_dot_M =  Wo @ M
    col_normsW = np.linalg.norm(Wo.T, axis=0,keepdims=True)
    col_normsM = np.linalg.norm(M, axis=0, keepdims=True)
    Wo_dot_M = Wo_dot_M / (col_normsW.T @ col_normsM)
    max_mean_diff = []
    for i in range(d):
        row_abs = abs(Wo_dot_M[:, i])
        max_mean_diff.append(max(row_abs))

    max_val, med_val, min_val = np.max(max_mean_diff), np.median(max_mean_diff), np.min(max_mean_diff)
    return  max_val, med_val, min_val, Wo_dot_M

def log_all_metrics(model, M, z, loss, logger, vals_to_log={}, log_weights=False):
    with torch.no_grad():
        max_val, med_val, min_val, Wo_dot_M = check_sparse_coding_learning(model, M)
        support, falsesupport, sparse = check_support(model, M, z)

        val_dict = {    "Max Wo_dot_M": max_val,
                        "Med Wo_dot_M": med_val,
                        "Min Wo_dot_M": min_val,
                        "loss": loss,
                        "support": support,
                        "false_support": falsesupport,
                        "sparse": sparse,
                        "Wo_norm": torch.norm(model.Wo.weight),
                        "bo norm": torch.norm(model.Wo.bias) if model.Wo.bias is not None else torch.norm(model.bo),
                    }
        sigma_o = np.linalg.svd(model.Wo.weight.cpu().numpy(), compute_uv=False)
        val_dict["Wo_largest_singular_val"] = np.max(sigma_o)
        val_dict["Wo_smallest_singular_val"] = np.min(sigma_o)
        val_dict["Wo_condition number"] = val_dict["Wo_largest_singular_val"] / val_dict["Wo_smallest_singular_val"]

        if 'simsiam' in model.name:
            sigma_p = np.linalg.svd(model.Wp.weight.cpu().numpy(), compute_uv=False)
            val_dict["Wp_largest_singular_val"] = np.max(sigma_p)
            val_dict["Wp_smallest_singular_val"] = np.min(sigma_p)
            val_dict["Wp_condition number"] = val_dict["Wp_largest_singular_val"] / val_dict["Wp_smallest_singular_val"]

            if model.name == 'simsiam':
                sigma_proj = np.linalg.svd(model.Wproj.weight.cpu().numpy(), compute_uv=False)
                val_dict["Wproj_largest_singular_val"] = np.max(sigma_proj)
                val_dict["Wproj_smallest_singular_val"] = np.min(sigma_proj)
                val_dict["Wproj_condition number"] = val_dict["Wproj_largest_singular_val"] / val_dict["Wproj_smallest_singular_val"]

            if log_weights:
                plt.clf()
                ax = sns.heatmap(model.Wo.weight.detach().cpu())
                val_dict["w_e"] = wandb.Image(ax)

                if model.name == 'simsiam':
                    plt.clf()
                    ax = sns.heatmap(model.Wproj.weight.detach().cpu())
                    val_dict["w_proj"] = wandb.Image(ax)
                
                if hasattr(model.Wp, 'weight'):
                    plt.clf()
                    ax = sns.heatmap(model.Wp.weight.detach().cpu())
                    val_dict["w_pred"] = wandb.Image(ax)

        if "simplified" in model.name or model.name == "simplest":
            val_dict["Wt_norm"] =  torch.norm(model.Wt.weight)
            val_dict["weight diff"] = torch.norm(model.Wo.weight - model.Wt.weight)
            val_dict["bt norm"] = torch.norm(model.Wt.bias) if model.Wt.bias is not None else torch.norm(model.bt)
            
            sigma_o = np.linalg.svd(model.Wo.weight.cpu().numpy(), compute_uv=False)
            val_dict["Wo_largest_singular_val"] = np.max(sigma_o)
            val_dict["Wo_smallest_singular_val"] = np.min(sigma_o)
            val_dict["Wo_condition number"] = val_dict["Wo_largest_singular_val"] / val_dict["Wo_smallest_singular_val"]
            
            sigma_t = np.linalg.svd(model.Wo.weight.cpu().numpy(), compute_uv=False)
            val_dict["Wt_largest_singular_val"] = np.max(sigma_t)
            val_dict["Wt_smallest_singular_val"] = np.min(sigma_t)
            val_dict["Wt_condition number"] = val_dict["Wt_largest_singular_val"] / val_dict["Wt_smallest_singular_val"]

            if "pred" in model.name:
                sigma_p = np.linalg.svd(model.Wp.weight.cpu().numpy(), compute_uv=False)
                val_dict["Wp_largest_singular_val"] = np.max(sigma_p)
                val_dict["Wp_smallest_singular_val"] = np.max(sigma_p)
                val_dict["Wp_condition number"] = val_dict["Wp_largest_singular_val"] / val_dict["Wp_smallest_singular_val"]

            if log_weights:
                plt.clf()
                ax = sns.heatmap(model.Wo.weight.detach().cpu())
                val_dict["Wo"] = wandb.Image(ax)
                
                plt.clf()
                ax = sns.heatmap(model.Wt.weight.detach().cpu())
                val_dict["Wt"] = wandb.Image(ax)

                if "pred" in model.name:
                    plt.clf()
                    ax = sns.heatmap(model.Wp.weight.detach().cpu())
                    val_dict["Wpred"] = wandb.Image(ax)
                
    for k, v in vals_to_log.items():
        val_dict[k] = v
    logger(val_dict)

def check_support(model, M, z):
    with torch.no_grad():
        Wo = model.Wo.weight.cpu().detach().numpy()
        pred_rep = model.predicted_rep.cpu().detach().numpy().T
        z = z.cpu().numpy().T
    m = model.m
    d = model.d
    assert z.shape[0] == d, "Z has incorrect shape"
    assert pred_rep.shape[0] == m, "predicted rep being passed in check support has incorrect shape"
    Wo_dot_M = Wo @ M # m * d
    col_normsW = np.linalg.norm(Wo.T, axis=0, keepdims=True)
    col_normsM = np.linalg.norm(M, axis=0, keepdims=True)
    Wo_dot_M = Wo_dot_M / (col_normsW.T @ col_normsM)
    latentsord = []
    neuronord = []
    # sort the neuron-latents pair by their cosine value (decreasing ordering)
    sorted_ind = np.array(np.unravel_index(np.argsort(-Wo_dot_M, axis=None), Wo_dot_M.shape))
    #print(sorted_ind)
    j = 0
    while j < d: 
        neuronnow = sorted_ind[0,0]
        latentnow = sorted_ind[1,0]
        latentsord.append(latentnow) 
        neuronord.append(neuronnow)
        sorted_ind = sorted_ind[:,sorted_ind[0,:]!=neuronnow]
        sorted_ind = sorted_ind[:,sorted_ind[1,:]!=latentnow]
        j+=1
    z_est = np.sign(pred_rep) # (m, batch_size) m>=d, z:(d,batch_size)
    match = []
    mismatch = []
    sparse = []
    for i in range(z_est.shape[1]):
        if np.sum(abs(z[:,i]))>0:
            match.append(np.sum(abs(z[latentsord,i]*z_est[neuronord,i]))/np.sum(abs(z[:,i]))) # get the match between m and d
            mismatch.append(np.sum((z[latentsord,i]==0)*abs(z_est[neuronord,i]))/max(np.sum(abs(z_est[:,i])),1))
            sparse.append(np.mean(abs(z_est[:,i]))-np.mean(abs(z[:,i]))) 
        else:
            match.append(0)
            mismatch.append(0)
            sparse.append(0)
                
    return np.mean(match), np.mean(mismatch), np.mean(sparse)