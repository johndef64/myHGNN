from .distmult import DistMultDecoder
from .transe import TransEDecoder
from .rotate import RotatEDecoder
from .node_classifier import NodeClassifier
from .graph_classifier import GraphClassifier

DECODERS = {
    'distmult': DistMultDecoder,
    'transe': TransEDecoder,
    'rotate': RotatEDecoder,
    'node_classifier': NodeClassifier,
    'graph_classifier': GraphClassifier,
}

def get_decoder(name):
    if name not in DECODERS:
        raise ValueError(f"Unknown decoder '{name}'. Available: {list(DECODERS.keys())}")
    return DECODERS[name]
