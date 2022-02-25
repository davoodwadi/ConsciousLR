import math
import torch
from torch.optim.optimizer import Optimizer, required

class ActiveRAdam(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, stepSize, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, lrHigh=.05, lrLow=.95):
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
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdamConsciousLR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ActiveRAdam, self).__setstate__(state)
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
#                 print('step', state['step'], 'cumm', state['cumm'], 'grad', p.grad.item())




                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 print('exp_avg',state['exp_avg'])
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
#                 exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['exp_avg'] = beta1 * (exp_avg) + (1-beta1)*(grad)
#                 print('exp_avg',state['exp_avg'])
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['exp_avg_sq'] = beta2 * exp_avg_sq + (1-beta2)*grad.pow(2)

                # RAdam Start1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                # RAdam End1

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * state['gai'] * group['weight_decay'])

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
#                     p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p -= step_size*state['gai']*(exp_avg/denom)
                else:
#                     p_data_fp32.add_(-step_size, exp_avg)
                    p -= step_size*state['gai']*exp_avg


                # SetLR if i>0
                if state['step']/group['stepSize'] > 1 and state['step']%group['stepSize']==0:
                    tmp2 = state['gradOld'].clone().cpu()##could be eliminated
                    tmp3 = state['cumm'].clone().cpu()##could be eliminated
                    tmp5 = state['gai'].clone().cpu()##may be the one that needs cloning

                    state['gai'] = torch.as_tensor(np.where(tmp2*tmp3<=0, tmp5.mul(group['lrLow']), tmp5.add(group['lrHigh'])),dtype=p.dtype , device=p.device)

                # Resetting the accumulated gradients after each epoch
                if state['step']%group['stepSize']==0:
                    cumm = state['cumm']
                    state['gradOld'] = cumm.clone()
                    state['cumm'] = torch.zeros_like(p, memory_format=torch.preserve_format)



        return loss
