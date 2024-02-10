
import torch
import dnd
from mlzoo.resnet50 import model

net = model.ResNet(model.ResNet.config_resnet50_mlperf) \
    .to(dnd.env.dtype).to(dnd.env.dev)

x = torch.randn(
    (dnd.env.bs, 3, 224, 224),
    dtype=dnd.env.dtype,
    device=dnd.env.dev)

def roi(x):
    with dnd.trace_region('forward'):
        y = net(x)

    with dnd.trace_region('loss'):
        loss = y.sum()

    with dnd.trace_region('backward'):
        loss.backward()

dnd.profile(roi, x)
