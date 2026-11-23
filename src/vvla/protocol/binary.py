"""Length-prefixed JSON metadata plus raw array bytes; never pickle.

Each decoded array owns its storage. Metadata and total payload are bounded
before allocation, and offsets must cover the payload exactly once in order.
"""
import io
import json
import math
import struct

import numpy as np

from .wire import MAX_ARRAY_BYTES, MAX_MESSAGE_BYTES

MAGIC=b'VVLA\x02'
PREFIX=struct.Struct('!5sIQ')
MAX_HEADER_BYTES=4*1024*1024
MAX_DEPTH=64
END_OF_STREAM=object()


def encode_parts(value):
    """Snapshot mutable arrays once; keep immutable segments for pipe writes."""
    chunks=[]
    size=0
    def visit(item,depth=0):
        nonlocal size
        if depth>MAX_DEPTH:raise ValueError('Wire nesting exceeds limit')
        if isinstance(item,np.ndarray):
            if item.dtype.kind not in 'buif' or item.ndim>8 or item.nbytes>MAX_ARRAY_BYTES:
                raise ValueError('Invalid array dtype, rank or byte size')
            raw=item.tobytes(order='C')
            record={'$array':True,'dtype':item.dtype.str,'shape':list(item.shape),'offset':size,'bytes':len(raw)}
            chunks.append(raw);size+=len(raw)
            if size>MAX_MESSAGE_BYTES:raise ValueError('Payload exceeds transport limit')
            return record
        if isinstance(item,np.generic):return visit(item.item(),depth)
        if type(item) is dict:
            if any(type(k) is not str for k in item) or '$array' in item:
                raise ValueError('String keys required; $array is reserved')
            return {key:visit(value,depth+1) for key,value in item.items()}
        if type(item) in (tuple,list):return [visit(x,depth+1) for x in item]
        # Version metadata may be a str subclass (e.g. torch.torch_version.TorchVersion).
        # Normalize only actual strings; keep rejecting arbitrary objects.
        if isinstance(item,str):return str(item)
        if item is None or type(item) in (str,int,bool):return item
        if type(item) is float and math.isfinite(item):return item
        raise ValueError('Unsupported binary wire value: '+type(item).__name__)
    header=json.dumps(visit(value),allow_nan=False,separators=(',',':')).encode()
    if len(header)>MAX_HEADER_BYTES or len(header)+size>MAX_MESSAGE_BYTES:
        raise ValueError('Message exceeds transport limit')
    return (PREFIX.pack(MAGIC,len(header),size),header,*chunks)


def encode_frame(value):
    """Contiguous form for stored frames and callers that explicitly need bytes."""
    return b''.join(encode_parts(value))


def write_parts(stream,parts):
    for part in parts:
        view=memoryview(part)
        while view:
            count=stream.write(view)
            if not count:raise EOFError('Worker output pipe closed')
            if count<0 or count>len(view):raise OSError('Invalid pipe write count')
            view=view[count:]
    stream.flush()


def _exact(stream,size):
    data=bytearray(size)
    view=memoryview(data)
    position=0
    while position<size:
        count=stream.readinto(view[position:])
        if not count:raise EOFError('Truncated worker frame')
        position+=count
    return data


def read_frame(stream):
    initial=stream.read(1)
    if not initial:return END_OF_STREAM
    magic,header_size,payload_size=PREFIX.unpack(initial+_exact(stream,PREFIX.size-1))
    if (magic!=MAGIC or header_size>MAX_HEADER_BYTES
            or header_size+payload_size>MAX_MESSAGE_BYTES):
        raise ValueError('Invalid binary frame version or length')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate wire key')
            result[key]=value
        return result
    header=json.loads(_exact(stream,header_size),object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
    payload=_exact(stream,payload_size)
    cursor=0
    def visit(item,depth=0):
        nonlocal cursor
        if depth>MAX_DEPTH:raise ValueError('Wire nesting exceeds limit')
        if type(item) is dict:
            if '$array' in item:
                if set(item)!={'$array','dtype','shape','offset','bytes'} or item['$array'] is not True:
                    raise ValueError('Invalid binary array envelope')
                dtype,shape=np.dtype(item['dtype']),item['shape']
                if (dtype.kind not in 'buif' or type(shape) is not list or len(shape)>8
                        or any(type(x) is not int or not 0<=x<=MAX_ARRAY_BYTES for x in shape)):
                    raise ValueError('Invalid binary array dtype or shape')
                size=math.prod(shape)*dtype.itemsize
                if (size>MAX_ARRAY_BYTES or type(item['offset']) is not int or item['offset']!=cursor
                        or type(item['bytes']) is not int or item['bytes']!=size or cursor+size>len(payload)):
                    raise ValueError('Invalid array payload span')
                result=np.frombuffer(payload,dtype=dtype,count=math.prod(shape),offset=cursor).reshape(shape).copy()
                cursor+=size
                return result
            return {k:visit(v,depth+1) for k,v in item.items()}
        if type(item) is list:return [visit(v,depth+1) for v in item]
        return item
    result=visit(header)
    if cursor!=len(payload):raise ValueError('Unclaimed binary payload')
    return result


def decode_frame(data):
    if len(data) < PREFIX.size:
        raise ValueError('Truncated VVLA frame')
    magic, header_size, payload_size = PREFIX.unpack_from(data)
    if magic != MAGIC or PREFIX.size + header_size + payload_size != len(data):
        raise ValueError('Invalid VVLA frame length or version')
    stream=io.BytesIO(data)
    result=read_frame(stream)
    if result is END_OF_STREAM or stream.read(1):raise ValueError('Expected exactly one binary frame')
    return result
