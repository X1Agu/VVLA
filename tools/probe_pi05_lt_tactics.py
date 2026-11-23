"""Qualify all six actual Lt batch geometries with real weights, synthetic current inputs.

This is an isolated primitive diagnostic, not a policy or accuracy result. Ported from the retained p0-20261111 primitive harness; uses installed VVLA packages.
This creates NEW target-GPU selections, never restores the missing historical table.
Every algorithm is resolved from the actual top16, screened with a full FP32 oracle,
then the selected callable is compiled/captured and timed against F.linear in
ABBA order. No B1 algorithm is treated as true-B evidence.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

_UNRETIRED=[]


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def tensor_sha(torch,value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def checkpoint_weights(torch,path,label):
    from safetensors import safe_open
    with safe_open(path,framework='pt',device='cpu')as source:
        names=tuple(source.keys());records=[]
        def get(suffix):
            keys=[name for name in names if name.endswith(suffix)]
            if len(keys)!=1:raise ValueError('Expected exactly one checkpoint weight '+suffix)
            value=source.get_tensor(keys[0])
            if value.dtype!=torch.bfloat16 or not bool(torch.isfinite(value).all()):raise ValueError('Non-original finite BF16 weight')
            value=value.contiguous().to('cuda');records.append(dict(key=keys[0],shape=list(value.shape),sha256=tensor_sha(torch,value)))
            return value
        if label=='vision_fc2':
            prefix='paligemma.model.vision_tower.vision_model.encoder.layers.0.mlp.fc2.'
            # Checkpoint releases omit the internal vision_model wrapper.
            if not any(name.endswith(prefix+'weight')for name in names):
                prefix='paligemma.model.vision_tower.encoder.layers.0.mlp.fc2.'
            weight,bias=get(prefix+'weight'),get(prefix+'bias')
        else:
            prefix='gemma_expert.model.layers.0.'
            if label=='qkv':weight=torch.cat([get(prefix+'self_attn.'+name+'.weight')for name in ('q_proj','k_proj','v_proj')],dim=0)
            elif label=='gate_up_packed':weight=torch.cat([get(prefix+'mlp.'+name+'.weight')for name in ('gate_proj','up_proj')],dim=0)
            else:weight=get(prefix+'self_attn.o_proj.weight')
            bias=None
        return weight,bias,records


def timing(torch,function,count=40):
    values=[]
    for _ in range(5):function()
    for _ in range(count):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();result=function();end.record();end.synchronize();values.append(float(begin.elapsed_time(end)));del result
    return values


def numerical(torch,actual,reference):
    difference=actual.float()-reference
    violations=difference.abs()>(.03125+.015625*reference.abs())
    return dict(passed=not bool(violations.any()),max_abs=float(difference.abs().max()),
                rmse=float(difference.square().mean().sqrt()),violations=int(violations.sum()),atol=.03125,rtol=.015625)


def exact(torch,left,right):
    if left.shape!=right.shape or left.dtype!=right.dtype or not torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8)):
        raise RuntimeError('Compiled/captured true-B Lt output bytes changed')


@contextmanager
def capture(torch,function,args):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph();compiled=output=None;retired=False
    try:
        compiled=torch.compile(function,fullgraph=True,dynamic=False,
            options={'triton.cudagraphs':False,'emulate_precision_casts':True,'use_fast_math':False})
        with torch.cuda.stream(stream):
            first=compiled(*args);second=compiled(*args);exact(torch,first,second)
            with torch.cuda.graph(graph,stream=stream):output=compiled(*args)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();exact(torch,output,first)
        yield graph,stream,output
    finally:
        try:torch.cuda.synchronize();graph.reset();retired=True
        finally:
            if not retired:_UNRETIRED.append((graph,stream,compiled,function,args,output))


class _Owner:
    def assert_execution_owner(self):pass # Diagnostic plan, no installed model.


def case(torch,module,ffi,args,label,output,api,batch):
    import numpy as np
    spec_shape=module.geometries(batch)[label];m,k,n=(spec_shape[key]for key in ('m','k','n'))
    weight,bias,records=checkpoint_weights(torch,args.checkpoint,label)
    if weight.shape!=(n,k)or(bias is not None and bias.shape!=(n,)):raise ValueError('Checkpoint geometry changed '+label)
    generator=torch.Generator(device='cuda').manual_seed(7123+m+k+n)
    x=torch.randn((2*batch,256,k)if bias is not None else(batch,50,k),device='cuda',dtype=torch.bfloat16,generator=generator)*.25
    raw_x=x.reshape(m,k);packed=weight.T.contiguous()if spec_shape['layout']=='contiguous_kn'else weight
    workspace=torch.empty(ffi.WORKSPACE_BYTES,dtype=torch.uint8,device='cuda');y=torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
    # Float32 oracle is calculated before BF16 rounding and before bias output.
    oracle=raw_x.float().mm(weight.float().T)
    if bias is not None:oracle=oracle+bias.float()
    native=lambda:torch.nn.functional.linear(x,weight,bias)
    native_y=native();native_numeric=numerical(torch,native_y.reshape(m,n),oracle)
    plan=module.DynamicBiasedPlan(api,bias,batch)if bias is not None else ffi.Plan(api,m,k,n,spec_shape['layout'],'library-default')
    choices=[]
    try:
        for rank,result in enumerate(plan.heuristics()):
            info=api.algorithm_info(result.algo)
            reason=ffi.candidate_gate(result,info,'library-default')
            if info['numerical_impl_flags']&0xff0000!=0x020000:reason='not_exclusively_BF16_input'
            row=dict(rank=rank,algorithm={key:value for key,value in info.items()if key!='opaque_algo_hex'},workspace_bytes=int(result.workspaceSize))
            choices.append(row)
            if reason:row['rejected']=reason;continue
            try:
                def launch():plan.launch(raw_x,packed,y,result.algo,workspace,torch.cuda.current_stream());return y
                candidate=launch();torch.cuda.synchronize();row['oracle']=numerical(torch,candidate,oracle)
                if not row['oracle']['passed']:row['rejected']='full_FP32_oracle';continue
                row['raw_ms']=timing(torch,launch);row['p50_ms']=statistics.median(row['raw_ms'])
            except ffi.LtError as error:
                row['rejected']=str(error)
        eligible=[row for row in choices if 'p50_ms'in row]
        if not eligible:raise RuntimeError('No valid new true-B Lt algorithm for '+label)
        selected=min(eligible,key=lambda row:row['p50_ms'])
        specification=dict(**spec_shape,algorithm=selected['algorithm'],workspace_bytes=selected['workspace_bytes'])
    finally:
        torch.cuda.synchronize();plan.close()
    del workspace,y;gc.collect()
    owner=_Owner()
    bound=module.DynamicBiasedBoundPlan(api,weight,bias,specification,batch)if bias is not None else ffi.BoundPlan(api,specification,'cuda:0')
    bound.bind_owner(owner)
    try:
        if bias is not None:
            def selected_call(current,w,b):return module.biased.vision_fc2_cublaslt(current,w,b,bound.key)
            call_args=(x,weight,bias)
        else:
            def selected_call(current,w):return ffi.expert_cublaslt(current,w,bound.key,n)
            call_args=(x,packed)
        eager=selected_call(*call_args);numeric=numerical(torch,eager.reshape(m,n),oracle)
        if not numeric['passed']:raise RuntimeError('Actual true-B opaque Lt operator failed full oracle')
        first=eager.clone();original=x.clone();trials={}
        def native_call(current,w,b):return torch.nn.functional.linear(current,w,b)
        with capture(torch,native_call,(x,weight,bias))as(ga,sa,oa),capture(torch,selected_call,call_args)as(gb,sb,ob):
            exact(torch,ob,eager);exact(torch,oa,native_y)
            x.mul_(.875);expected=selected_call(*call_args);torch.cuda.synchronize();gb.replay();torch.cuda.synchronize();exact(torch,ob,expected)
            if torch.equal(ob.view(torch.uint8),first.view(torch.uint8)):raise RuntimeError('Current input update did not affect true-B output')
            x.copy_(original);torch.cuda.synchronize();ga.replay();gb.replay();torch.cuda.synchronize();exact(torch,ob,eager);exact(torch,oa,native_y)
            for name,graph,stream in [('A1_native',ga,sa),('B1_Lt',gb,sb),('B2_Lt',gb,sb),('A2_native',ga,sa)]:
                with torch.cuda.stream(stream):values=timing(torch,graph.replay)
                trials[name]=dict(raw_ms=values,p50_ms=statistics.median(values))
            exact(torch,eager,first)
        # Preserve true source inputs for review; output/oracle are clearly
        # sampled rows/columns while full-shape numerical checks remain above.
        rows=torch.arange(0,m,max(1,m//32),device='cuda',dtype=torch.int64)
        columns=torch.arange(0,n,max(1,n//64),device='cuda',dtype=torch.int64)
        array=lambda value:value.contiguous().cpu().numpy().copy()
        arrays=dict(rows=array(rows),columns=array(columns),input_bf16_bits=array(raw_x[rows].view(torch.uint16)),
            weight_bf16_bits=array(weight[columns].view(torch.uint16)),output_bf16_bits=array(eager.reshape(m,n)[rows][:,columns].contiguous().view(torch.uint16)),
            oracle_fp32=array(oracle[rows][:,columns]))
        if bias is not None:arrays['bias_bf16_bits']=array(bias[columns].view(torch.uint16))
        raw=output/(label+'-RAW.npz')
        with raw.open('xb')as stream:np.savez_compressed(stream,**arrays)
        faster_a=min(trials[name]['p50_ms']for name in ('A1_native','A2_native'))
        slower_b=max(trials[name]['p50_ms']for name in ('B1_Lt','B2_Lt'))
        return dict(shape=spec_shape,checkpoint_weights=records,synthetic_activations=True,selected_spec=specification,
            support_passed=True,fullgraph_passed=True,cuda_graph_passed=True,oracle_passed=True,timing_samples=40,
            native_oracle=native_numeric,selected_oracle=numeric,heuristics=choices,trials=trials,
            conservative_primitive_ratio=slower_b/faster_a,raw_path=raw.name,raw_sha256=sha(raw),
            raw_scope='32 distributed rows/64 output columns; full K; full-shape oracle and output hashes calculated before sampling',
            input_sha256=tensor_sha(torch,x),weight_sha256=tensor_sha(torch,weight),output_sha256=tensor_sha(torch,eager),
            full_oracle_sha256=tensor_sha(torch,oracle),policy_quality_passed=False)
    finally:
        if not _UNRETIRED:torch.cuda.synchronize();bound.close()
        else:_UNRETIRED.append((bound,owner,x,packed,weight,bias))


def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.checkpoint=args.checkpoint.resolve(strict=True)
    checkpoint_sha256=sha(args.checkpoint)
    with open('/tmp/vvla-gpu-'+args.gpu_uuid+'.lock','a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=10)
        if args.gpu_uuid in active:raise RuntimeError('Another GPU owner remains active')
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID')
        import torch
        from vvla_pi05.lite import dynamic_cublaslt as module
        from vvla_pi05.lite import cublaslt_runtime as ffi
        original=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        cases={};api=None
        try:
            torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
            api,identity=module.load_api('cuda:0')
            actual_uuid=str(torch.cuda.get_device_properties(0).uuid)
            if actual_uuid.removeprefix('GPU-')!=args.gpu_uuid.removeprefix('GPU-'):raise RuntimeError('Wrong actual physical GPU')
            identity.update(gpu_uuid=args.gpu_uuid,device_name=torch.cuda.get_device_name(0),multiprocessor_count=torch.cuda.get_device_properties(0).multi_processor_count)
            with torch.inference_mode():
                for batch in module.BATCH_SIZES:
                    cases[str(batch)]={};directory=args.output/('batch'+str(batch));directory.mkdir()
                    for label in module.BASE_SHAPES:
                        row=case(torch,module,ffi,args,label,directory,api,batch)
                        row['raw_path']=str(Path(directory.name)/row['raw_path'])
                        cases[str(batch)][label]=row
                        save(directory/(label+'.json'),row);gc.collect();torch.cuda.empty_cache()
                        print(json.dumps(dict(batch=batch,case=label,status='passed',ratio=row['conservative_primitive_ratio'])),flush=True)
                        # Each primitive's two captured graphs are retired by
                        # case(). Clear only this diagnostic's specialization
                        # cache before moving to the next independent geometry.
                        torch.compiler.reset()
            report=dict(status='complete',implementation=module.VERSION,batch_sizes=list(module.BATCH_SIZES),gpu_uuid=args.gpu_uuid,
                library_sha256=identity['library_sha256'],runtime=identity,checkpoint_sha256=checkpoint_sha256,selection_origin='fresh_target_gpu_probe',
                source_sha256=module.sources(),probe_sha256=sha(Path(__file__)),batches=cases,
                policy_loaded=False,policy_quality_passed=False,diagnostic_only=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        finally:
            torch.set_float32_matmul_precision(original[0]);torch.backends.cuda.matmul.allow_tf32=original[1]
            torch.backends.cudnn.allow_tf32=original[2];torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original[3]
            restored=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                      torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)==original
            save(args.output/'CLEANUP.json',dict(graphs_retired=not _UNRETIRED,settings_restored=restored))
        if not restored or _UNRETIRED:raise RuntimeError('true-B Lt cleanup failed')
        save(args.output/'BACKEND.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--describe',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--gpu-uuid')
    args=parser.parse_args()
    if args.describe:
        print(json.dumps(dict(batch_sizes=[1,2,4,8,16,32],geometries=dict(expert_m='50B',vision_m='512B'),policy_loaded=False,
            actual_checkpoint_weights=True,synthetic_activations=True,quality_claim=False)));return
    if args.output is None or args.checkpoint is None or args.gpu_uuid is None:parser.error('output/checkpoint/gpu-uuid required')
    try:run(args)
    except BaseException:
        if args.output.exists():save(args.output/'FAILED.json',dict(status='failed',error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
