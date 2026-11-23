"""Direct packed-weight restore, ported from the frozen pi05 precision path."""
from pathlib import Path
import hashlib,json,math
import torch
from torch import nn
from safetensors.torch import save_file,load_file
from vvla_runtime.precision import QuantizedLinear

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def from_packed(original,precision,tensors):
 k,n=original.in_features,original.out_features;alignment=32 if precision=='nvfp4' else 16
 if not isinstance(original,nn.Linear) or original.weight.dtype!=torch.bfloat16 or k%alignment or n%alignment:raise ValueError('Invalid packed BF16 source geometry')
 spec={'packed_weight':((n,k),torch.float8_e4m3fn),'weight_scale':((n,1),torch.float32)}
 if precision=='nvfp4':
  if torch.cuda.get_device_capability()!=(12,0):raise ValueError('NVFP4 requires SM120')
  spec={'packed_weight':((n,k//2),torch.uint8),'weight_scale':((((n+127)//128)*128,((k+63)//64)*4),torch.float8_e4m3fn),'input_global_scale':((1,),torch.float32),'alpha':((1,),torch.float32)}
 if set(tensors)!=set(spec):raise ValueError('Wrong packed keys')
 for name,(shape,dtype) in spec.items():
  t=tensors[name]
  if tuple(t.shape)!=shape or t.dtype!=dtype or not t.is_contiguous():raise ValueError('Invalid packed tensor '+name)
  if dtype==torch.float32 and (not torch.isfinite(t).all() or not (t>0).all()):raise ValueError('Invalid scale')
 self=QuantizedLinear.__new__(QuantizedLinear);nn.Module.__init__(self)
 self.weight,self.bias=original.weight,original.bias;self.in_features,self.out_features=k,n;self.precision=precision
 self.padded_in_features=k;self.padded_out_features=n;self._geometry=(k,n,k,n)
 for name in ['packed_weight','weight_scale','input_global_scale','alpha']:
  t=tensors.get(name);self.register_buffer(name,None if t is None else t.to(original.weight.device).clone())
 self._source=self._source_signature();self._packed=self._packed_signature();self.train(original.training)
 return self

class PackedStore:
 def __init__(self,path,mode,calibration):
  self.path=Path(path);self.mode=mode;self.meta_path=self.path.with_suffix('.json')
  self.cal_sha=hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest();self.data=None;self.sources={}
  if self.path.exists():
   meta=json.loads(self.meta_path.read_text())
   if meta['mode']!=mode or meta['calibration_sha256']!=self.cal_sha:raise ValueError('Packed recipe mismatch')
   if hashlib.file_digest(self.path.open('rb'),'sha256').hexdigest()!=meta['sha256']:raise ValueError('Packed file SHA mismatch')
   self.sources=meta['source_weight_sha256'];self.data=load_file(str(self.path))
 def restore(self,key,original,precision):
  if self.data is None:return None
  if tensor_sha(original.weight)!=self.sources[key]:raise ValueError('Packed source weight SHA mismatch: '+key)
  prefix=key+'.';ts={k[len(prefix):]:v for k,v in self.data.items() if k.startswith(prefix)}
  return from_packed(original,precision,ts)
 def save(self,layers):
  if self.data is not None:return
  ts={};sources={}
  for i,l in enumerate(layers):
   for n in ['gate_proj','up_proj','down_proj']:
    key=f'vlm.layers.{i}.mlp.{n}';m=getattr(l.mlp,n);sources[key]=tensor_sha(m.weight)
    for name in ['packed_weight','weight_scale','input_global_scale','alpha']:
     t=getattr(m,name)
     if t is not None:ts[key+'.'+name]=t.detach().cpu().contiguous()
  self.path.parent.mkdir(parents=True,exist_ok=True);tmp=self.path.with_suffix('.partial');save_file(ts,str(tmp));tmp.replace(self.path)
  self.meta_path.write_text(json.dumps({'mode':self.mode,'calibration_sha256':self.cal_sha,'sha256':hashlib.file_digest(self.path.open('rb'),'sha256').hexdigest(),'source_weight_sha256':sources,'modules':108},indent=2))
