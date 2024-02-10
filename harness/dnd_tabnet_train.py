
import torch
import dnd
from mlzoo.tabnet import model
from mlzoo.tabnet import dataset

def convert_float_tensor(t : torch.Tensor, dtype):
    if torch.is_floating_point(t):
        return t.to(dtype)
    return t

net = model.TabNet(model.TabNet.covertype_config) \
    .to(dnd.env.dtype).to(dnd.env.dev)

ds = dataset.CovertypeSyntheticDataset(dnd.env.bs)
dl = torch.utils.data.DataLoader(ds, batch_size=dnd.env.bs, shuffle=False)

cols, labels = next(iter(dl))

# N.B. The synthetic generator for the labels doesn't understand the range it
# should be generating in, so we correct it here. It's a bit hacky, but it
# doesn't matter too much.
labels = (labels % model.TabNet.covertype_config.num_classes) + 1

cols = [
    convert_float_tensor(c, dtype=dnd.env.dtype).to(dnd.env.dev)
    for c in cols
]

labels = labels.to(dnd.env.dev)

def roi(cols, labels):
    with dnd.trace_region('forward'):
        loss = net.loss(cols, labels)

    with dnd.trace_region('backward'):
        loss.backward()

dnd.profile(roi, cols, labels)
