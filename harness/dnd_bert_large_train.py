
import torch
import dnd
from mlzoo.bert import model

cfg = model.bert_large_conf(512)
net = model.Bert(cfg).to(dnd.env.dtype).to(dnd.env.dev)

input_ids = torch.randint(
    0,
    cfg.vocab_size,
    (dnd.env.bs, 512),
    device=dnd.env.dev)

token_type_ids = torch.randint(
    0,
    1,
    (dnd.env.bs, 512),
    device=dnd.env.dev)

def roi(input_ids, token_type_ids):
    with dnd.trace_region('forward'):
        y = net(input_ids, token_type_ids)

    with dnd.trace_region('loss'):
        loss = y.sum()

    with dnd.trace_region('backward'):
        loss.backward()

dnd.profile(roi, input_ids, token_type_ids)
