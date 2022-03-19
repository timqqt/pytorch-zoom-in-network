from functools import partial

import torch

def mean_forward(accum, sub):
    return accum + sub

def max_forward(accum, sub):
    return torch.where(sub > accum, sub, accum)

def logsumexp_forward(accum, sub):
    return accum + torch.exp(sub)

def torch_exclusive_cumsum_in_dim_1_shape2(x):
    x = torch.cumsum(x, dim=1)
    x = torch.cat((torch.zeros((x.shape[0], 1)), x[:, :-1]), dim=1)
    return x


def torch_exclusive_cumsum_in_dim_1_shape3(x):
    x = torch.cumsum(x, dim=1)
    x = torch.cat((torch.zeros((x.shape[0], 1, x.shape[2])), x[:, :-1, :]), dim=1)
    return x

def to_tensor(x, dtype=torch.int32, device=None):
    """If x is a Tensor return it as is otherwise return a constant tensor of
    type dtype."""
    device = torch.device('cpu') if device is None else device
    if torch.is_tensor(x):
        return x.to(device)

    return torch.tensor(x, dtype=dtype, device=device)


def to_dtype(x, dtype):
    """Cast Tensor x to the dtype """
    return x.type(dtype)


to_float16 = partial(to_dtype, dtype=torch.float16)
to_float32 = partial(to_dtype, dtype=torch.float32)
to_float64 = partial(to_dtype, dtype=torch.float64)
to_double = to_float64
to_int8 = partial(to_dtype, dtype=torch.int8)
to_int16 = partial(to_dtype, dtype=torch.int16)
to_int32 = partial(to_dtype, dtype=torch.int32)
to_int64 = partial(to_dtype, dtype=torch.int64)


def expand_many(x, axes):
    """Call expand_dims many times on x once for each item in axes."""
    for ax in axes:
        x = torch.unsqueeze(x, ax)
    return x

class ExpectationWithoutReplacement(torch.autograd.Function):
    """ Custom pytorch layer for calculating the expectation of the sampled patches
        without replacement.
    """

    @staticmethod
    def forward(ctx, weights, attention, features):
        # Reshape the passed weights and attention in feature compatible shapes
        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)
        af = expand_many(attention, axes)

        # Compute how much of the probablity mass was available for each sample
        pm = 1 - torch.cumsum(attention, axis=1)
        pmf = expand_many(pm, axes)

        # Compute the features
        Fa = af * features
        Fpm = pmf * features
        Fa_cumsum = torch.cumsum(Fa, axis=1)
        F_estimator = Fa_cumsum + Fpm

        F = torch.sum(wf * F_estimator, axis=1)

        ctx.save_for_backward(weights, attention, features, pm, pmf, Fa, Fpm, Fa_cumsum, F_estimator)

        return F

    @staticmethod
    def backward(ctx, grad_output):
        weights, attention, features, pm, pmf, Fa, Fpm, Fa_cumsum, F_estimator = ctx.saved_tensors
        device = weights.device

        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)
        af = expand_many(attention, axes)

        N = attention.shape[1]
        probs = attention / pm
        probsf = expand_many(probs, axes)
        grad = torch.unsqueeze(grad_output, 1)

        # Gradient wrt to the attention
        ga1 = F_estimator / probsf
        ga2 = (
                torch.cumsum(features, axis=1) -
                expand_many(to_float32(torch.arange(0, N, device=device)), [0] + axes) * features
        )
        ga = grad * (ga1 + ga2)
        ga = torch.sum(ga, axis=list(range(2, len(ga.shape))))
        ga = ga * weights

        # Gradient wrt to the features
        gf = expand_many(to_float32(torch.arange(N-1, -1, -1, device=device)), [0] + axes)
        gf = pmf + gf * af
        gf = wf * gf
        gf = gf * grad

        return None, ga, gf
    
    
class ExpectationWithReplacement(torch.autograd.Function):
    """ Custom pytorch layer for calculating the expectation of the sampled patches
        with replacement.
    """
    @staticmethod
    def forward(ctx, weights, attention, features):

        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)

        F = torch.sum(wf * features, dim=1)

        ctx.save_for_backward(weights, attention, features, F)
        return F

    @staticmethod
    def backward(ctx, grad_output):
        weights, attention, features, F = ctx.saved_tensors
        axes = [-1] * (len(features.shape) - 2)
        wf = expand_many(weights, axes)

        grad = torch.unsqueeze(grad_output, 1)

        # Gradient wrt to the attention
        ga = grad * features
        ga = torch.sum(ga, axis=list(range(2, len(ga.shape))))
        ga = ga * weights / attention

        # Gradient wrt to the features
        gf = wf * grad

        return None, ga, gf