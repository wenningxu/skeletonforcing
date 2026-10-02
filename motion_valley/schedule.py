"""Noise level sigma: 1=noise, 0=clean. Anchors are always clean."""
import torch

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def graph_distances(device=None):
    d = torch.full((22, 22), 100., device=device)
    d.fill_diagonal_(0)
    for j, p in enumerate(PARENTS):
        if p >= 0:
            d[j, p] = d[p, j] = 1
    for k in range(22):
        d = torch.minimum(d, d[:, k:k+1] + d[k:k+1, :])
    return d


def distance_field(mask, temporal_scale=20., joint_scale=2.):
    """Nearest anchor distance on time x skeleton graph; B,T,J boolean mask."""
    if mask.ndim != 3 or mask.shape[-1] != 22 or mask.dtype != torch.bool:
        raise ValueError('mask must be boolean B,T,22')
    if temporal_scale <= 0 or joint_scale <= 0:
        raise ValueError('distance scales must be positive')
    b, t, j = mask.shape
    graph = graph_distances(mask.device) / joint_scale
    frames = torch.arange(t, device=mask.device)
    field = torch.empty((b, t, j), device=mask.device)
    for i in range(b):
        anchors = mask[i].nonzero()
        if len(anchors) == 0:
            field[i] = 1.
            continue
        dt = (frames[:, None] - anchors[:, 0]).abs() / temporal_scale
        distance = dt[:, None, :] + graph[:, anchors[:, 1]][None]
        field[i] = distance.amin(-1)
    return field


def noise_levels(progress, mask, distance, mode='valley', spread=.75):
    """Monotone piecewise-linear front; every unknown starts at sigma=1.

    delay = spread*d/(1+d). Near anchors start denoising first. At p=1 all
    unknowns are clean. This avoids the invalid partially-clean Gaussian init.
    """
    if not 0 <= spread < 1:
        raise ValueError('spread must be in [0,1)')
    p = torch.as_tensor(progress, device=mask.device, dtype=torch.float32)
    while p.ndim < 3:
        p = p.unsqueeze(-1)
    if mode == 'valley':
        delay = spread * distance / (1 + distance)
    elif mode == 'uniform':
        delay = torch.zeros_like(distance)
    elif mode == 'temporal':
        delay = spread * torch.linspace(0, 1, mask.shape[1], device=mask.device)[None, :, None]
    else:
        raise ValueError(mode)
    sigma = 1 - ((p-delay)/(1-delay)).clamp(0, 1)
    return sigma.expand_as(distance).masked_fill(mask, 0.)


def corrupt(clean, noise, sigma, mask):
    mixed = (1-sigma[..., None])*clean + sigma[..., None]*noise
    return torch.where(mask[..., None], clean, mixed)


def trajectory_step(index, steps, mask, distance, mode='valley', spread=.75):
    """Single source of truth for the training AND inference time fields.

    Training may sample a global step index, never independent per-token times.
    A batch of indices selects complete fields from the actual sampler lattice.
    """
    if steps<1: raise ValueError('steps must be positive')
    k=torch.as_tensor(index,device=mask.device)
    if (k<0).any() or (k>=steps).any(): raise ValueError('step index outside trajectory')
    current=noise_levels(k/steps,mask,distance,mode,spread)
    following=noise_levels((k+1)/steps,mask,distance,mode,spread)
    return current,following


@torch.no_grad()
def sample(model, anchors, mask, text, valid, steps=32, mode='valley',
           spread=.75, generator=None, initial_noise=None, callback=None):
    if steps < 1:
        raise ValueError('steps must be positive')
    distance = distance_field(mask)
    x = (torch.randn(anchors.shape, device=anchors.device, generator=generator)
         if initial_noise is None else initial_noise.clone())
    x = torch.where(mask[..., None], anchors, x)
    if callback:
        callback(0, x)
    for k in range(steps):
        sigma,nxt = trajectory_step(k,steps,mask,distance,mode,spread)
        # Model predicts dx/dsigma = epsilon - x_clean, not dx/dglobal_progress.
        velocity = model(x, sigma, mask, text, valid)
        x = x + (nxt-sigma)[..., None] * velocity.float()
        x = torch.where(mask[..., None], anchors, x)
        sigma = nxt
        if callback:
            callback(k+1, x)
    return x
