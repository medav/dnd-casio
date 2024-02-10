
import torch
import dnd
from mlzoo.mgn import graphnet as GNN
from mlzoo.mgn import hyperel

ds = hyperel.HyperElasticitySyntheticData(10, 10, 10, 10, 10, 10, dnd.env.bs)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=dnd.env.bs,
    shuffle=False,
    collate_fn=lambda b: \
        GNN.collate_common(b, hyperel.HyperElasticitySampleBatch)
)

batch : hyperel.HyperElasticitySampleBatch
batch = next(iter(dl)).asdtype(dnd.env.dtype).todev(dnd.env.dev)

net = hyperel.HyperElasticityModel(
    hyperel.HyperElasticityModel.default_config_3d) \
        .to(dnd.env.dtype).to(dnd.env.dev)


def roi(batch):
    with dnd.trace_region('forward'):
        y = net.forward(batch)

    with dnd.trace_region('loss'):
        loss = y.sum()

    with dnd.trace_region('backward'):
        loss.backward()

dnd.profile(roi, batch)
