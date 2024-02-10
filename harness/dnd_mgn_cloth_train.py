
import torch
import dnd
from mlzoo.mgn import graphnet as GNN
from mlzoo.mgn import cloth

ds = cloth.ClothSyntheticData(10, 10, dnd.env.bs)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=dnd.env.bs,
    shuffle=False,
    collate_fn=lambda b: \
        GNN.collate_common(b, cloth.ClothSampleBatch)
)

batch : cloth.ClothSampleBatch
batch = next(iter(dl)).asdtype(dnd.env.dtype).todev(dnd.env.dev)

net = cloth.ClothModel(cloth.ClothModel.default_config) \
    .to(dnd.env.dtype).to(dnd.env.dev)


def roi(batch):
    with dnd.trace_region('forward'):
        net.forward(batch)

dnd.profile(roi, batch)
