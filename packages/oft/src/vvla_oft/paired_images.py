"""Paired preprocessing for the pinned 224-pixel dual-backbone OFT processor."""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _normalize_planes(images, table, output, pixels: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    plane = tl.program_id(1)
    camera = plane // 6
    channel = plane % 3
    value = tl.load(images + (camera * pixels + index) * 3 + channel, mask=index < pixels, other=0)
    normalized = tl.load(table + (plane % 6) * 256 + value.to(tl.int32))
    tl.store(output + plane * pixels + index, normalized, mask=index < pixels)


class PairedCameraProcessor:
    def __init__(self, processor, utils, device):
        import tensorflow as tf
        from PIL import Image
        cls = type(processor)
        image = processor.image_processor
        if (cls.__module__, cls.__name__) != ("prismatic.extern.hf.processing_prismatic", "PrismaticProcessor"):
            raise ValueError("Paired camera preprocessing requires the OFT PrismaticProcessor")
        if type(image).__module__ != cls.__module__ or type(image).__name__ != "PrismaticImageProcessor":
            raise ValueError("Paired camera preprocessing requires the pinned PrismaticImageProcessor")
        if image.tvf_do_letterbox or len(image.input_sizes) != 2 or any(tuple(size) != (3,224,224) for size in image.input_sizes):
            raise ValueError("Paired camera preprocessing requires two 224x224 RGB transforms")
        if (any(tuple(p["size"]) != (224,224) for p in image.tvf_resize_params)
                or any(tuple(p["output_size"]) != (224,224) for p in image.tvf_crop_params)):
            raise ValueError("Paired image normalization requires identity final resize and crop")
        self.device = device
        self.tf = tf
        # On 224x224 RGB inputs the remaining two transforms are pointwise.
        # Obtain all 256 normalized values for each channel from the real processor.
        ramp = np.arange(224*224,dtype=np.int32).astype(np.uint8).reshape(224,224)
        rgb = np.repeat(ramp[:,:,None], 3, axis=2)
        sample = image.apply_transform(Image.fromarray(rgb)).reshape(6,-1)
        self.table = sample[:,:256].contiguous().to(device=device,dtype=torch.bfloat16)

        def one(raw):
            def resize():
                encoded = tf.io.encode_jpeg(raw)
                decoded = tf.io.decode_image(encoded, channels=3, expand_animations=False)
                decoded.set_shape([None,None,3])
                resized = tf.image.resize(decoded, (224,224), method="lanczos3", antialias=True)
                return tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8)
            image224 = tf.cond(tf.reduce_all(tf.equal(tf.shape(raw)[:2], [224,224])), lambda:raw, resize)
            image224.set_shape([224,224,3])
            scaled = tf.image.convert_image_dtype(image224, tf.float32)
            cropped = utils.crop_and_resize(scaled, 0.9, 1)
            return tf.image.convert_image_dtype(tf.clip_by_value(cropped,0,1),tf.uint8,saturate=True)

        spec = tf.TensorSpec([None,None,3], tf.uint8)
        @tf.function(input_signature=[spec,spec], autograph=False)
        def pair(full, wrist):
            return tf.stack([one(full), one(wrist)])
        self._transform = pair

    def __call__(self, full, wrist):
        images = self._transform(self.tf.convert_to_tensor(full),self.tf.convert_to_tensor(wrist)).numpy()
        raw = torch.from_numpy(images).to(self.device)
        result = torch.empty((12,224,224), device=self.device, dtype=torch.bfloat16)
        count = 224*224
        _normalize_planes[(triton.cdiv(count,256),12)](raw,self.table,result,pixels=count,block=256)
        return result
