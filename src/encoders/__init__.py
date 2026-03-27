from .compgcn import CompGCNEncoder
from .rgcn import RGCNEncoder
from .rgat import RGATEncoder
from .kge import DistMultKGEEncoder, TransEEncoder, RotatEEncoder, Node2VecEncoder

ENCODERS = {
    # GNN encoders
    'compgcn': CompGCNEncoder,
    'rgcn': RGCNEncoder,
    'rgat': RGATEncoder,
    # KGE encoders (no message passing)
    'distmult_kge': DistMultKGEEncoder,
    'transe': TransEEncoder,
    'rotate': RotatEEncoder,
    'node2vec': Node2VecEncoder,
}

def get_encoder(name):
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODERS.keys())}")
    return ENCODERS[name]
