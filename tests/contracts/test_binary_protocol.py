import io
import json
import numpy as np
import pytest

from vvla.protocol.binary import encode_frame,encode_parts,write_parts,decode_frame,read_frame,PREFIX,MAGIC,END_OF_STREAM


def test_nested_torch_version_metadata_roundtrips_as_plain_string():
    from torch.torch_version import TorchVersion
    value={'metadata':{'native':{'torch_version':TorchVersion('2.13.0+cu130')}}}
    result=decode_frame(encode_frame(value))
    assert result==value
    assert type(result['metadata']['native']['torch_version']) is str
    with pytest.raises(ValueError,match='Unsupported binary wire value'):
        encode_frame({'metadata':object()})


def test_binary_payload_is_raw_and_outputs_are_independent():
    source=np.arange(120,dtype=np.float32).reshape(10,12)[:,::2]
    frame=encode_frame({'state':source,'prompt':'取杯子','empty':np.empty((0,7),np.float32)})
    assert source.copy().tobytes() in frame
    out=decode_frame(frame)
    np.testing.assert_array_equal(out['state'],source)
    assert out['state'].flags.owndata and out['empty'].shape==(0,7)
    source[:]=3
    assert out['state'][0,1]==2 and out['prompt']=='取杯子'
    assert decode_frame(encode_frame(None)) is None


def test_short_reads_and_multiple_frames_do_not_mix_requests():
    class ShortReads(io.BytesIO):
        def readinto(self,buffer):return super().readinto(buffer[:3])
    stream=ShortReads(encode_frame({'id':1,'x':np.ones(4,np.uint8)})+encode_frame({'id':2}))
    assert read_frame(stream)['id']==1
    assert read_frame(stream)=={'id':2}
    assert read_frame(stream) is END_OF_STREAM


def test_segmented_write_handles_partial_io_and_snapshots_caller_arrays():
    source=np.arange(60,dtype=np.float32).reshape(10,6)[:,::2]
    expected=source.copy()
    parts=encode_parts({'x':source,'empty':np.empty((0,7),np.float32)})
    assert len(parts)==4 and all(type(p) is bytes for p in parts)
    source[:]=-1
    class ShortWrites(io.BytesIO):
        def write(self,data):return super().write(data[:3])
    stream=ShortWrites()
    write_parts(stream,parts)
    write_parts(stream,encode_parts({'second':True}))
    stream.seek(0)
    out=read_frame(stream)
    np.testing.assert_array_equal(out['x'],expected)
    assert out['x'].flags.owndata and out['empty'].flags.owndata
    assert read_frame(stream)=={'second':True}
    assert read_frame(stream) is END_OF_STREAM


def test_segmented_write_detects_closed_pipe():
    class ClosedPipe:
        def write(self,value):return 0
    with pytest.raises(EOFError):write_parts(ClosedPipe(),encode_parts({'x':np.ones(4)}))


@pytest.mark.parametrize('kind',['truncated','oversize','overlap','object','unclaimed','duplicate'])
def test_binary_rejects_invalid_frames_before_array_use(kind):
    header={'$array':True,'dtype':'<f4','shape':[2],'offset':0,'bytes':8}
    payload=b'\0'*8
    if kind=='overlap':header['offset']=1
    if kind=='object':header['dtype']='O'
    if kind=='unclaimed':payload+=b'\0'
    raw=json.dumps(header).encode() if kind!='duplicate' else b'{"id":1,"id":2}'
    frame=PREFIX.pack(MAGIC,len(raw),len(payload))+raw+payload
    if kind=='truncated':frame=frame[:-1]
    if kind=='oversize':frame=PREFIX.pack(MAGIC,5*1024**2,0)
    with pytest.raises((ValueError,EOFError)):decode_frame(frame)
