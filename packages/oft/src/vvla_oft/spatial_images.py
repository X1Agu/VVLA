"""Apply the pinned OFT Lanczos and crop operators on device using TensorFlow's coefficients."""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .paired_images import PairedCameraProcessor, _normalize_planes


@triton.jit
def _lanczos(X, START, WEIGHT, OUT, IN_H:tl.constexpr, IN_W:tl.constexpr,
             OUT_H:tl.constexpr, OUT_W:tl.constexpr, VERTICAL:tl.constexpr,
             ROUND:tl.constexpr, TAPS:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < 2 * OUT_H * OUT_W * 3
    ch = i % 3
    x = (i // 3) % OUT_W
    y = (i // (3 * OUT_W)) % OUT_H
    cam = i // (3 * OUT_W * OUT_H)
    axis = y if VERTICAL else x
    start = tl.load(START + axis)
    value = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(TAPS):
        weight = tl.load(WEIGHT + axis * TAPS + j)
        sy = tl.minimum(start + j, IN_H-1) if VERTICAL else y
        sx = x if VERTICAL else tl.minimum(start + j, IN_W-1)
        sample = tl.load(X + ((cam*IN_H+sy)*IN_W+sx)*3+ch, valid, 0).to(tl.float32)
        value = value + weight * sample
    if ROUND:
        value = tl.minimum(255., tl.maximum(0., libdevice.rint(value)))
    tl.store(OUT + i, value, valid)


@triton.jit
def _crop(X, COORD, OUT, WIDTH:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < 2 * WIDTH * WIDTH * 3
    x = (i//3) % WIDTH
    y = (i//(3*WIDTH)) % WIDTH
    cam = i//(3*WIDTH*WIDTH)
    ch = i % 3
    fx = tl.load(COORD+x)
    fy = tl.load(COORD+y)
    x0 = fx.to(tl.int32)
    y0 = fy.to(tl.int32)
    dx = fx-x0.to(tl.float32)
    dy = fy-y0.to(tl.float32)
    base = ((cam*WIDTH+y0)*WIDTH+x0)*3+ch
    a = tl.load(X+base, valid, 0).to(tl.float32)*(1./255.)
    b = tl.load(X+base+3, valid, 0).to(tl.float32)*(1./255.)
    c = tl.load(X+base+3*WIDTH, valid, 0).to(tl.float32)*(1./255.)
    d = tl.load(X+base+3*WIDTH+3, valid, 0).to(tl.float32)*(1./255.)
    upper = a + (b-a)*dx
    lower = c + (d-c)*dx
    value = upper + (lower-upper)*dy
    tl.store(OUT+i, tl.minimum(255., tl.maximum(0., value*255.5)), valid)


class SpatialCameraProcessor(PairedCameraProcessor):
    def __init__(self, processor, utils, device, *, native_jpeg=False):
        super().__init__(processor, utils, device)
        tf = self.tf
        # Impulses recover all taps from the frozen TensorFlow sampling geometry.
        # Only shape-dependent interpolation coefficients are stored.
        basis = tf.eye(256, dtype=tf.float32)[None,:,None,:]
        weights = tf.image.resize(basis, (224,1), method="lanczos3", antialias=True).numpy()[0,:,0]
        starts = np.array([np.flatnonzero(w)[0] for w in weights], dtype=np.int32)
        taps = max(np.flatnonzero(w)[-1]-s+1 for w,s in zip(weights,starts))
        self.taps = int(taps)
        compact = np.zeros((224,self.taps),dtype=np.float32)
        for i,start in enumerate(starts):
            tail=weights[i,start:start+self.taps]
            compact[i,:len(tail)]=tail
        self.starts = torch.from_numpy(starts).to(device)
        self.weights = torch.from_numpy(compact).to(device)
        ramp = tf.broadcast_to(tf.range(224,dtype=tf.float32)[None,:,None],(224,224,1))
        self.coords = torch.from_numpy(utils.crop_and_resize(ramp,0.9,1).numpy()[0,:,0].copy()).to(device)
        spec = tf.TensorSpec([256,256,3], tf.uint8)
        @tf.function(input_signature=[spec,spec], autograph=False)
        def decode_pair(full, wrist):
            def one(img):
                return tf.io.decode_image(tf.io.encode_jpeg(img),channels=3,expand_animations=False)
            return tf.stack([one(full),one(wrist)])
        self.decode_pair = decode_pair
        self.jpeg_kernel = self.jpeg_manifest = None
        if native_jpeg:
            from vvla_runtime.execution.paired_jpeg import load_paired_jpeg
            self.jpeg_kernel, self.jpeg_manifest = load_paired_jpeg()

    def decode_array(self, full, wrist):
        if self.jpeg_kernel is None:
            return self.decode_pair(full,wrist).numpy()
        return self.jpeg_kernel.roundtrip(np.ascontiguousarray(full), np.ascontiguousarray(wrist))

    def resized_pixels(self, raw, *, vertical_first=True, fusion=False):
        temp_shape = (2,224,256,3) if vertical_first else (2,256,224,3)
        temp = torch.empty(temp_shape, device=self.device, dtype=torch.float32)
        out = torch.empty((2,224,224,3), device=self.device, dtype=torch.uint8)
        for src,dst,vertical,finish in ((raw,temp,vertical_first,False),(temp,out,not vertical_first,True)):
            _lanczos[(triton.cdiv(dst.numel(),256),)](src,self.starts,self.weights,dst,
                *src.shape[1:3],*dst.shape[1:3],vertical,finish,self.taps,256,enable_fp_fusion=fusion)
        return out

    def crop_pixels(self, resized, *, fusion=False):
        out = torch.empty_like(resized)
        _crop[(triton.cdiv(out.numel(),256),)](resized,self.coords,out,224,256,enable_fp_fusion=fusion)
        return out

    def cropped_pixels(self, full, wrist):
        if full.shape == wrist.shape == (256,256,3):
            decoded = torch.from_numpy(self.decode_array(full,wrist)).to(self.device)
            return self.crop_pixels(self.resized_pixels(decoded))
        if full.shape == wrist.shape == (224,224,3):
            return self.crop_pixels(torch.from_numpy(np.stack([full,wrist])).to(self.device))
        return torch.from_numpy(self._transform(full,wrist).numpy()).to(self.device)

    def __call__(self, full, wrist):
        images = self.cropped_pixels(full,wrist)
        result = torch.empty((12,224,224),device=self.device,dtype=torch.bfloat16)
        count = 224*224
        _normalize_planes[(triton.cdiv(count,256),12)](images,self.table,result,count,256)
        return result
