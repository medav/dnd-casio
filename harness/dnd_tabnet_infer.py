
import torch
import dnd
from mlzoo.tabnet import model
from mlzoo.tabnet import dataset

def convert_float_tensor(t : torch.Tensor, dtype):
    if torch.is_floating_point(t):
        return t.to(dtype)
    return t

net = model.TabNet(model.TabNet.covertype_config) \
    .to(dnd.env.dtype).to(dnd.env.dev).eval()

ds = dataset.CovertypeSyntheticDataset(dnd.env.bs)
dl = torch.utils.data.DataLoader(ds, batch_size=dnd.env.bs, shuffle=False)

cols, labels = next(iter(dl))

cols = [
    convert_float_tensor(c, dtype=dnd.env.dtype).to(dnd.env.dev)
    for c in cols
]

labels = labels.to(dnd.env.dev)

def roi(cols):
    with dnd.trace_region('forward'):
        net(cols)

dnd.profile(roi, cols)
