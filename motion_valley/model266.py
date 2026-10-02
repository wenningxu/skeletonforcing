from .model263 import Motion263Flow
from .representation266 import Layout266


class Motion266Flow(Motion263Flow):
    """Same joint-time backbone; lossless 266D motion packing and 4096D UMT5 tokens."""
    def __init__(self,width=256,depth=6,heads=8,text_dim=4096):
        super().__init__(width,depth,heads,text_dim)
        self.layout=Layout266()
