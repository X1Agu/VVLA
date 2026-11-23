"""Fresh coalesced inputs and layout routing for true B1/2/4/8/16/32.

Source497's arithmetic/helpers and current typed metadata are retained. This is
a separate, unqualified geometry adaptation; neither B32 evidence nor CPU shape
tests imply same-card CUDA parity, model quality, or service performance.
"""
from contextlib import ExitStack
import hashlib
from pathlib import Path

import torch

from .batch_coalesced_prepare import _source_rows,_images,_metadata
from .batch_layout import BatchHostMaskLayout,BatchHostLayoutCache,validate_fixed_batch_size
from .batch_processor import Pi05BatchProcessor,validate_observations
from .processor import Pi05LiteProcessor
from vvla_repro_pi05.models.pi05.processor_pi05 import apply_norm,as_state_vector,build_pi05_prompt,tokenize_prompt


def build_dynamic_inputs(processor,observations,*,batch_size,return_host_layout=False):
    batch=validate_fixed_batch_size(batch_size)
    validate_observations(observations,batch)
    if (type(processor) is not Pi05LiteProcessor or
            getattr(processor.build_model_inputs,'__func__',None) is not Pi05LiteProcessor.build_model_inputs):
        raise ValueError('Dynamic coalesced preparation requires its private original processor')
    if torch.is_autocast_enabled(torch.device(processor.device).type):
        raise ValueError('Dynamic preparation requires explicit selected dtypes')
    source=_source_rows(processor,observations)
    if source is None:
        return Pi05BatchProcessor(processor,batch).build_model_inputs(observations,return_host_layout=return_host_layout)
    from .gpu_image_prepare import GPUUint8Preparation
    from .packed_input_metadata import PackedInputMetadata
    hooks=(('_lite_image_prepare','_lite_gpu_uint8_prepare_installation',GPUUint8Preparation),
           ('_lite_metadata_prepare','_lite_packed_input_metadata_installation',PackedInputMetadata))
    gpu_uint8=False
    with ExitStack() as locks:
        for hook,installation,expected in hooks:
            owner=getattr(processor,hook,None);installed=getattr(processor,installation,None)
            if owner is None and installed is None:continue
            if type(owner) is not expected or installed is not owner:
                raise ValueError('Dynamic preparation cannot bypass an unknown hook')
            if not owner._lock.acquire(blocking=False):raise RuntimeError('Preparation owner is busy')
            locks.callback(owner._lock.release)
            if expected is GPUUint8Preparation:owner._assert_installed();gpu_uint8=True
            else:owner._active_processor()
        rows,shape=source;config=processor.config
        cameras=[[image is not None for image in row] for row in rows]
        ids=[];masks=[]
        for observation in observations:
            state=as_state_vector(observation.get('state'),config.state_dim)
            normalized=apply_norm(torch.from_numpy(state),processor._state_norm).numpy()
            prompt=build_pi05_prompt(task=observation.get('prompt','') or '',
                normalized_state=normalized,state_num_bins=config.state_num_bins)
            row_ids,row_mask=tokenize_prompt(processor.tokenizer,prompt,config.tokenizer_max_length)
            if len(row_ids)!=config.tokenizer_max_length or len(row_mask)!=len(row_ids):
                raise ValueError('Actual tokenizer geometry changed')
            ids.append(row_ids);masks.append([bool(x) for x in row_mask])
        height,width=config.image_resolution
        images=_images(rows,shape,height,width,processor.device,gpu_uint8)
        image_masks,tokens,language_masks=_metadata(cameras,ids,masks,processor.device)
        if return_host_layout:
            from vvla_repro_pi05.models.pi05.inference_cache_pi05 import tensor_identity
            host=BatchHostMaskLayout(tuple(tuple(row) for row in cameras),tuple(tuple(row) for row in masks),
                tuple(tensor_identity(x) for x in (*images,*image_masks,language_masks)),batch_size=batch)
            host.validate(images,image_masks,language_masks)
            return images,image_masks,tokens,language_masks,host
        return images,image_masks,tokens,language_masks


class DynamicCoalescedProcessor(Pi05BatchProcessor):
    def __init__(self,processor,batch_size,*,output_processor=None):
        validate_fixed_batch_size(batch_size)
        if type(processor) is not Pi05LiteProcessor:raise ValueError('Private original processor required')
        super().__init__(processor,batch_size,output_processor=output_processor)

    def build_model_inputs(self,observations,*,return_host_layout=False):
        return build_dynamic_inputs(self._single,observations,batch_size=self.batch_size,return_host_layout=return_host_layout)


class DynamicHostLayoutCache:
    def __init__(self,capacity=64):
        self.caches={batch:BatchHostLayoutCache(capacity=capacity,batch_size=batch) for batch in (1,2,4,8,16,32)}
    def clear(self):
        for cache in self.caches.values():cache.clear()
    def get(self,model,host,images,image_masks,lang_masks):
        if type(host) is not BatchHostMaskLayout:raise ValueError('Dynamic layout requires actual processor-owned masks')
        batch=validate_fixed_batch_size(host.batch_size)
        return self.caches[batch].get(model,host,images,image_masks,lang_masks)


def implementation_manifest():
    here=Path(__file__).resolve().parent
    return dict(implementation='source497-fresh-dynamic-coalesced-inputs-v1',batch_sizes=[1,2,4,8,16,32],
        source_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest()
            for name in ('dynamic_batch_prepare.py','batch_coalesced_prepare.py','batch_layout.py','batch_processor.py')},
        shared_arithmetic='original normalization, prompt/tokenizer, image upload/transform and output inverse transform',
        current_inputs_only=True,dummy_lanes=False,inherited_gpu_qualification=False)
