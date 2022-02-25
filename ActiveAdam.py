import math
import torch
from torch.optim.optimizer import Optimizer, required

class ActiveAdam(Optimizer):


    def __init__(self, params, stepSize, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, lrHigh=2., lrLow=.5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        lrHigh=lrHigh, lrLow=lrLow, stepSize=stepSize)
        super(ActiveAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ActiveAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue


                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['gai'] = torch.ones_like(p, memory_format=torch.preserve_format)
                    state['cumm'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Accumulate gradients for the epoch
                state['cumm']+=(p.grad)

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * state['gai'] * group['weight_decay'])


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']


                state['exp_avg'] = beta1 * (exp_avg) + (1-beta1)*(grad)

                state['exp_avg_sq'] = beta2 * exp_avg_sq + (1-beta2)*grad.pow(2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Correction
                exp_avgCorr = state['exp_avg']/(1-beta1**state['step'])

                exp_avg_sqCorr = state['exp_avg_sq']/(1-beta2**state['step'])

                step_size = group['lr']

                p -= step_size*state['gai']*(exp_avgCorr/(exp_avg_sqCorr.sqrt()+group['eps']))

                # SetLR if i>0
                if state['step']/group['stepSize'] > 1 and state['step']%group['stepSize']==0:
                    tmp2 = state['gradOld'].clone()
                    tmp3 = state['cumm'].clone()
                    tmp5 = state['gai'].clone()

                    state['gai'] = torch.where(tmp2*tmp3<=0, tmp5.mul(group['lrLow']), tmp5.add(group['lrHigh']))


                # Resetting the accumulated gradients after each epoch
                if state['step']%group['stepSize']==0:
                    cumm = state['cumm']
                    state['gradOld'] = cumm.clone()
                    state['cumm'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        return loss
