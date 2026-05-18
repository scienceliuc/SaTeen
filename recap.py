"""
Copyright to ReCAP Authors, ICML 2025 Poster.
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from einops import rearrange

class ReCAP(nn.Module):

    def __init__(self, model, optimizer, sigmas, batch_size, steps=1, episodic=False, margin = 0.8 * math.log(1000), \
        reset_constant_em=0.2, margin_L0 = 0.7 * math.log(1000), weight_reg = 0.5, reweight_threshold = 2.0, weight_tau = 1.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "ReCAP requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.batch_size = batch_size

        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme, follow SAR
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.margin = margin  # margin \tau_RE in Eqn. (9)
        self.margin_L0 = margin_L0 # L_0 in Eqn. (9)
        self.weight_reg = weight_reg
        self.reweight_threshold = reweight_threshold
        self.weight_tau = weight_tau
        self.sigma_t = sigmas

        try:
            self.W = model.fc.weight # ResNet
        except:
            self.W = model.head.weight # ViT

        self.W_cpu = self.W.cpu()
        self._refresh_prob_aug()

    def _refresh_prob_aug(self, scale = 0.1):
        with torch.no_grad():
            sigma_t = self.sigma_t.view(1, 1, -1)
            region = sigma_t * self.weight_tau / scale
            sqrt_region = torch.sqrt(region).cpu()
            diff = (self.W_cpu.unsqueeze(0) - self.W_cpu.unsqueeze(1)) * sqrt_region
            self.prob_aug = torch.exp(0.5 * torch.einsum('ijb,ijb->ij', diff, diff))
            self.prob_aug = self.prob_aug.cuda()
            self.normW = 0.1 / 2 * (scale ** 2)  * (torch.norm(self.W, dim=1) ** 2)

    @torch.jit.script
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def L_RE(self, x: torch.Tensor) -> torch.Tensor:
        prob_anchor = x.softmax(1)
        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        prob = (x + self.normW).softmax(1)
        return (-prob * torch.log(prob_anchor) + prob * torch.log(prob_aug)).sum(1)

    def L_RI(self, x: torch.Tensor) -> torch.Tensor:
        prob_anchor = x.softmax(1)
        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        return (prob_anchor * torch.log(prob_aug)).sum(1)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_recap(self, x, ema):
        
        self.optimizer.zero_grad()
        
        outputs = self.model(x)
        
        L_RE = self.L_RE(outputs)
        L_RI = self.L_RI(outputs)


        loss = (L_RE + self.weight_reg * L_RI).mean(0) # ReCAP replacement for entropy loss

        if not np.isnan(loss.item()):
            ema = update_ema(ema, loss.item() / 2) # record moving average loss values for model recovery
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                print("ema < 0.2, now reset the model")
                reset_flag = True

        return outputs, ema, reset_flag


    def forward(self, x, no_adapt = False):
        if self.episodic:
            self.reset()

        if no_adapt == True:
            return self.model(x)

        for _ in range(self.steps):
            outputs, ema, reset_flag = self.forward_and_adapt_recap(x, self.ema)

            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss
 
        return outputs

    def get_features(self, x):
        x = self.model.forward_features(x)
        try:
            x = self.model.global_pool(x) # ResNet
        except:
            x = x[:, 0, :]  # ViT
        return x
        
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None