

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from sklearn.decomposition import IncrementalPCA
from utils.cli_utils import *
import time


class SaTeen(nn.Module):

    def __init__(self, model, args, optimizer, steps=1, episodic=False, sateen_margin=0.2 * math.log(1000)):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.args = args

        self.steps = steps
        self.episodic = episodic
        args.counts = [1e-6, 1e-6, 1e-6, 1e-6]
        args.correct_counts = [0, 0, 0, 0]

        self.sateen_margin = sateen_margin
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.k = args.k

    def forward(self, x, iter_, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()

        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_sateen(x, iter_,
                                                                                   self.model,
                                                                                   self.args,
                                                                                   self.optimizer,
                                                                                   self.sateen_margin,
                                                                                   self.lambda_1,
                                                                                   self.lambda_2,
                                                                                   self.k,
                                                                                   targets, flag,
                                                                                   group,
                                                                                   )
                else:
                    outputs = forward_and_adapt_sateen(x, iter_,
                                                       self.model,
                                                       self.args,
                                                       self.optimizer,
                                                       self.sateen_margin,
                                                       self.lambda_1,
                                                       self.lambda_2,
                                                       self.k,
                                                       targets, flag,
                                                       group,
                                                       )
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_sateen(x, iter_,
                                                                                                     self.model,
                                                                                                     self.args,
                                                                                                     self.optimizer,
                                                                                                     self.sateen_margin,
                                                                                                     self.lambda_1,
                                                                                                     self.lambda_2,
                                                                                                     self.k,
                                                                                                     targets, flag,
                                                                                                     group,
                                                                                                     )
                else:
                    outputs = forward_and_adapt_sateen(x, iter_,
                                                         self.model,
                                                         self.args,
                                                         self.optimizer,
                                                         self.sateen_margin,
                                                         self.lambda_1,
                                                         self.lambda_2,
                                                         self.k,
                                                         targets, flag,
                                                         group,
                                                         )
        if targets is None:
            if flag:
                return outputs, backward, final_backward
            else:
                return outputs
        else:
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None



def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # temprature = 1.1 #0.9 #1.2
    # x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def CrossEntropy(logits_p, logits_q, dim=1, eps=1e-8):
    p = F.softmax(logits_p, dim=dim)
    q = F.softmax(logits_q, dim=dim)

    #
    p = p + eps
    q = q + eps

    #
    p = p / p.sum(dim=dim, keepdim=True)
    q = q / q.sum(dim=dim, keepdim=True)

    kl_qp = (q * (- p.log())).sum(dim=dim)

    return (kl_qp)


#
def get_random_index_with_no_self_matching(size):

    torch.manual_seed(int(time.time()))
    random_index = torch.randint(0, size, (size,))

    #
    # while torch.any(random_index == torch.arange(size)):
    # random_index = torch.randint(0, size, (size,))

    return random_index


# @torch.no_grad()
@torch.enable_grad()
def forward_and_adapt_sateen(x, iter_, model, args, optimizer, sateen_margin, _LAMBDA_1 , _LAMBDA_2 , _K_PCS, targets=None, flag=True, group=None ,_IPCA_WARM_STEPS = 100,  _IPCA_BATCH =32):

    if not hasattr(forward_and_adapt_sateen, 'st'):
        forward_and_adapt_sateen.st = {
            "ipca": IncrementalPCA(n_components=_K_PCS, batch_size=_IPCA_BATCH),
            "running_mean": None,
            "buffer": [],
            "steps": 0,
            "u_cache": None,
            "count": 0,
        }
    st = forward_and_adapt_sateen.st
    outputs = model(x)
    if not flag:
        return outputs

    optimizer.zero_grad()

    x_prime = x.detach()

    if args.aug_type == 'occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start + args.occlusion_size,
        args.column_start:args.column_start + args.occlusion_size] = occlusion_window
    elif args.aug_type == 'patch':
        resize_t = torchvision.transforms.Resize(
            ((x.shape[-1] // args.patch_len) * args.patch_len, (x.shape[-1] // args.patch_len) * args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1], x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type == 'pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    elif args.aug_type == 'random':
        random_index = get_random_index_with_no_self_matching(x.size(0))
        x_prime = x_prime[random_index]
    elif args.aug_type == 'random+patch':
        random_index = get_random_index_with_no_self_matching(x.size(0))
        x_prime = x_prime[random_index]
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start + args.occlusion_size,
        args.column_start:args.column_start + args.occlusion_size] = occlusion_window

    with torch.no_grad():
        outputs_prime = model(x_prime)

    entropys_full = softmax_entropy(outputs)
    L_null = CrossEntropy(outputs, outputs_prime).clamp_max(math.log(1000))
    l1 = entropys_full - _LAMBDA_1 * L_null
    if args.filter_ent:
        filter_ids_1 = (l1 < sateen_margin)
    else:
        filter_ids_1 = (l1 > - math.log(1000))

    entropys = entropys_full[filter_ids_1]
    backward = len(entropys)

    if backward == 0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0
    final_backward = len(entropys)

    if targets is not None:
        corr_pl_1 = 0

    if final_backward == 0:
        del x_prime

        if targets is not None:
            return outputs, backward, 0, corr_pl_1, 0
        return outputs, backward, 0

    if targets is not None:
        corr_pl_2 = 0

    if args.reweight_ent or args.reweight_plpd:
        coeff = 1 / (torch.exp(l1.detach() - sateen_margin))
        l1 = l1.mul(coeff)

    loss = l1[filter_ids_1].mean(0)

    # ===================== IPCA =====================
    z_all = outputs.detach()
    st["steps"] += 1
    if filter_ids_1.any():
        z_rel = z_all[filter_ids_1].cpu().numpy()  # (Br, d)
        st["buffer"].append(z_rel)
    if st["buffer"] and (sum(len(b) for b in st["buffer"]) >= _IPCA_BATCH):
    #if st["buffer"]:
        Xc = np.concatenate(st["buffer"], axis=0)
        try:
            st["ipca"].partial_fit(Xc)
        except Exception:
            pass
        st["buffer"].clear()
        st["u_cache"] = None

    do_null = (st["steps"] >= _IPCA_WARM_STEPS) and hasattr(st["ipca"], "components_")
    if do_null and getattr(st["ipca"], "components_", None) is not None:
        if st["u_cache"] is None:
            C = torch.from_numpy(st["ipca"].components_.copy()).to(z_all.device, z_all.dtype)  # (k, d)
            U = C.T.contiguous()
            U, _ = torch.linalg.qr(U, mode="reduced")
            st["u_cache"] = U.detach()
        U = st["u_cache"]
        I = torch.eye(U.shape[0], device=z_all.device, dtype=z_all.dtype)
        P_perp = (I - U @ U.T) # (d, d)
        z_a = outputs
        z_a_perp = (P_perp @ z_a.T).T
        L_null = torch.norm(z_a_perp, p=2, dim=1)
        loss = loss + _LAMBDA_2 * L_null.mul(coeff).mean()

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime

    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


