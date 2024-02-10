
import torch
import dnd
from mlzoo.pinn import navierstokes as NS

lb = torch.tensor([-1.0, -1.0, -1.0])
ub = torch.tensor([1.0, 1.0, 1.0])

net = NS.PinnNs(NS.PinnNs.Config(lb=lb, ub=ub, layers=[20] * 8)) \
    .to(dnd.env.dtype).to(dnd.env.dev).train()

x = torch.randn((dnd.env.bs, 1), dtype=dnd.env.dtype, device=dnd.env.dev, requires_grad=True)
y = torch.randn((dnd.env.bs, 1), dtype=dnd.env.dtype, device=dnd.env.dev, requires_grad=True)
t = torch.randn((dnd.env.bs, 1), dtype=dnd.env.dtype, device=dnd.env.dev, requires_grad=True)
u = torch.randn((dnd.env.bs, 1), dtype=dnd.env.dtype, device=dnd.env.dev, requires_grad=True)
v = torch.randn((dnd.env.bs, 1), dtype=dnd.env.dtype, device=dnd.env.dev, requires_grad=True)

# N.B. Torch Compile currently doesn't support double backward for higher order
# derivatives. We use the `instrument` function which allows us to use torch's
# fx.symbolic_trace which can capture the higher order derivatives and translate
# them to matmuls.

instrumented = dnd.instrument(net, use_fx=True)

def roi(x, y, t, u, v):
    with dnd.trace_region('forward'):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = instrumented(x, y, t)

    with dnd.trace_region('loss'):
        loss = torch.sum(torch.square(u - u_pred)) + \
            torch.sum(torch.square(v - v_pred)) + \
            torch.sum(torch.square(f_u_pred)) + \
            torch.sum(torch.square(f_v_pred))

    with dnd.trace_region('backward'):
        loss.backward()

dnd.profile(roi, x, y, t, u, v, no_compile=True)
