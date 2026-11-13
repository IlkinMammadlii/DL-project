"""Restricted loading of the mirror's NumPy-containing PyTorch archives."""
import io,numpy as np,torch
def safe_load(f):
    allowed=[(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),np._core.multiarray._reconstruct,np.ndarray,np.dtype]+[type(np.dtype(t)) for t in ['float32','int64','int32','uint8','uint32','float64','bool','int16','uint16']]
    with torch.serialization.safe_globals(allowed): return torch.load(io.BytesIO(f.read()),map_location='cpu',weights_only=True)
