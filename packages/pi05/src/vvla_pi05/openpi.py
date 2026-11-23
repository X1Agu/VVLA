"""Experimental native OpenPI LIBERO Pi0.5 preprocessing contract.

Keep this separate from the historical LeRobot processor: PIL uint8 resize,
SentencePiece BOS/separate newline, quantile epsilon and action semantics differ.
Sources and qualification boundary are recorded in docs/OPENPI_CONTRACT.md.
"""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch


CAMERAS=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')


class OpenPITokenizer:
    def __init__(self,path):
        import sentencepiece
        self.path=Path(path).resolve(strict=True)
        self.model=sentencepiece.SentencePieceProcessor(model_file=str(self.path))
    def tokens(self,prompt,state,limit):
        cleaned=prompt.strip().replace('_',' ').replace('\n',' ')
        if state is None:
            ids=self.model.encode(cleaned,add_bos=True)+self.model.encode('\n')
        else:
            bins=np.digitize(state,np.linspace(-1,1,257)[:-1])-1
            text=f"Task: {cleaned}, State: {' '.join(map(str,bins))};\nAction: "
            ids=self.model.encode(text,add_bos=True)
        length=min(len(ids),limit)
        return ids[:limit]+[0]*(limit-length),[True]*length+[False]*(limit-length)
    def get_vocab(self):
        # Used only by cold provenance construction.
        return {self.model.id_to_piece(i):i for i in range(self.model.vocab_size())}


def resized_rgb_uint8(value,height,width):
    array=np.asarray(value)
    if array.ndim!=3:raise ValueError('OpenPI RGB image must have rank 3')
    if array.shape[0]==3 and array.shape[-1]!=3:array=np.moveaxis(array,0,-1)
    if array.shape[-1]!=3:raise ValueError('OpenPI image must have three channels')
    if array.dtype.kind=='f':
        if not np.isfinite(array).all() or array.min()<0 or array.max()>1:raise ValueError('Floating image must be finite in [0,1]')
        array=(array*255).astype(np.uint8)
    if array.dtype!=np.uint8:raise ValueError('OpenPI image must be uint8 or floating [0,1]')
    if array.shape[:2]==(height,width):return np.array(array,copy=True)
    h,w=array.shape[:2];ratio=max(h/height,w/width)
    nh,nw=int(h/ratio),int(w/ratio)
    if min(nh,nw)<1:raise ValueError('Image aspect ratio produces an empty resize')
    resized=np.asarray(Image.fromarray(array).resize((nw,nh),Image.Resampling.BILINEAR))
    output=np.zeros((height,width,3),np.uint8)
    top,left=(height-nh)//2,(width-nw)//2
    output[top:top+nh,left:left+nw]=resized
    return output


class OpenPiLiberoProcessor:
    image_preprocess_version='openpi-pil-uint8-resize-then-siglip-v1'
    image_prepare_hook_version='unsupported-openpi'
    metadata_prepare_hook_version='unsupported-openpi'
    def __init__(self,config,tokenizer,device):
        if (config.chunk_size!=10 or config.discrete_state_input or config.use_relative_actions
                or config.state_dim!=8 or config.action_dim!=7 or config.max_action_dim!=32
                or tuple(config.image_feature_keys)!=CAMERAS or config.max_cameras!=3):
            raise ValueError('Native pi05_libero requires H10/no state tokens/no extra delta transform/state8/action7/camera order')
        self.config,self.tokenizer,self.device=config,tokenizer,device
        self._stats={}
        for key,width in (('state',8),('action',7)):
            record=config.norm_stats[key]
            lo,hi=(np.asarray(record[k],dtype=np.float32) for k in ('q01','q99'))
            if lo.shape!=(width,) or hi.shape!=(width,) or not np.isfinite([lo,hi]).all() or (hi<lo).any():
                raise ValueError('OpenPI requires finite ordered per-dimension quantile stats')
            self._stats[key]=(lo.copy(),hi.copy())
        self._state_norm=None  # State is validated/normalized independently; not serialized into text.

    def normalized_state(self,state):
        values=np.asarray(state)
        if values.shape!=(8,) or values.dtype.kind not in 'fi' or not np.isfinite(values).all():
            raise ValueError('OpenPI LIBERO state must be a finite vector of length eight')
        lo,hi=self._stats['state']
        normalized=(values-lo)/(hi-lo+1e-6)*2.0-1.0
        return np.pad(normalized,(0,24))

    def build_model_inputs(self,observation,*,return_host_layout=False):
        self.normalized_state(observation['state'])
        current=observation['images']
        if any(current.get(k) is None for k in CAMERAS[:2]):raise ValueError('OpenPI LIBERO requires base and left wrist images')
        if current.get(CAMERAS[2]) is not None:raise ValueError('pi05_libero right wrist is a masked padding camera')
        ids,valid=self.tokenizer.tokens(observation['prompt'],None,self.config.tokenizer_max_length)
        h,w=self.config.image_resolution
        images=[]
        for key in CAMERAS:
            array=np.zeros((h,w,3),np.uint8) if key==CAMERAS[2] else resized_rgb_uint8(current[key],h,w)
            # Match official Torch Observation.from_dict: uint8 H2D, then FP32
            # normalization on the model device (not a NumPy approximation).
            tensor=torch.from_numpy(array).permute(2,0,1)[None].to(self.device,dtype=torch.float32)
            images.append(tensor/255.0*2.0-1.0)
        masks=[torch.tensor([v],device=self.device,dtype=torch.bool) for v in (True,True,False)]
        tokens=torch.tensor([ids],device=self.device,dtype=torch.long)
        token_mask=torch.tensor([valid],device=self.device,dtype=torch.bool)
        if return_host_layout:
            from .lite.layout import HostMaskLayout
            layout=HostMaskLayout.from_processor((True,True,False),valid,images,masks,token_mask)
            return images,masks,tokens,token_mask,layout
        return images,masks,tokens,token_mask

    def build_model_outputs(self,actions,observation):
        array=actions.detach().float().cpu().numpy() if isinstance(actions,torch.Tensor) else np.asarray(actions)
        lo,hi=self._stats['action']
        result=(array[...,:7]+1.0)/2.0*(hi-lo+1e-6)+lo
        if result.shape[0]==1:result=result[0]
        return np.array(result,dtype=np.float32,order='C',copy=True)

