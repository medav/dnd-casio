
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

# N.B. Torch Compile currently doesn't support double backward for higher order
# derivatives. We use the `instrument` function which allows us to use torch's
# fx.symbolic_trace which can capture the higher order derivatives and translate
# them to matmuls.

instrumented = dnd.instrument(net, use_fx=True)

def roi(x, y, t):
    with dnd.trace_region('forward'):
        instrumented(x, y, t)


dnd.profile(roi, x, y, t, no_compile=True)
