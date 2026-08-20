from fastai.vision.all import *
from torch import linspace

def get_slice(vol, plane, i):
    if plane in (0, 'a', 'axial'):    return vol[i]
    if plane in (1, 'c', 'coronal'):  return vol[:,i]
    if plane in (2, 's', 'sagittal'): return vol[:,:,i]

class TensorMri(TensorImage): pass

def show_mri(t, plane, slice, ctx=None, **kwargs):
    t = TensorImage(get_slice(t, plane, slice))
    return show_image(t, ctx=ctx, **{**t._show_args, **kwargs})

@patch
def show(self:TensorMri, plane='a', slice=None, ctx=None, **kw):
    if slice is None: slice = self.shape[2]//2
    show_mri(self, plane, slice, ctx=ctx, **kw)

def bin_mids(start, end, n):
    o = linspace(start, end, n+1)
    return ((o[:-1]+o[1:])/2).int()

@patch
def show_all(self:TensorMri, n=5, **kw):
    _,axs = subplots(3,n)
    for i,ax in enumerate(axs):
        slices = bin_mids(0,self.shape[i]-1,5)
        for s,a in zip(slices,ax): self.show(i,s,ctx=a)

__all__ = 'get_slice TensorMri'
