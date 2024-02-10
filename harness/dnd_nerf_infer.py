
import torch
import dnd
from mlzoo.nerf import model

net = model.Nerf(model.Nerf.default_nerf_config) \
    .to(dnd.env.dtype).to(dnd.env.dev)

rays_o = torch.randn(
    (dnd.env.bs, 3),
    dtype=dnd.env.dtype,
    device=dnd.env.dev)

rays_d = torch.randn(
    (dnd.env.bs, 3),
    dtype=dnd.env.dtype,
    device=dnd.env.dev)

def roi(rays_o, rays_d):
    with dnd.trace_region('forward'):
        net.render_rays(rays_o, rays_d, None, 8, near=2, far=6)

dnd.profile(roi, rays_o, rays_d)
