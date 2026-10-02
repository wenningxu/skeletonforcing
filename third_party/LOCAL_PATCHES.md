# Local compatibility changes

`FloodDiffusion/models/tools/t5.py`: T5EncoderModel's default device changed
from the import-time call `torch.cuda.current_device()` to the string `cpu`.
Every production cache invocation passes a device explicitly. Tokenization,
architecture, attention, checkpoint loading and forward math are unchanged.

This permits importing the upstream encoder on a machine with CPU-only torch.
All existing attribution/license notices remain intact.
