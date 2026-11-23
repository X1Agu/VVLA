"""Load the optional CPU libjpeg operator before capturing the OFT model."""
import hashlib
from pathlib import Path

def load_paired_jpeg():
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name("csrc")/"paired_jpeg.cpp"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    extension = load(name="vvla_oft_paired_jpeg_"+digest[:16],sources=[str(source)],
                     extra_cflags=["-O3","-fopenmp"],extra_ldflags=["-ljpeg","-fopenmp","-ldl"],with_cuda=False)
    library = Path(extension.library_path()).resolve(strict=True)
    receipt = {"source_sha256":digest, "binary_sha256":hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
               "jpeg_library":str(library), "jpeg_library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),
               "quality":95,"jpeg_chroma_subsampling":"4:2:0","idct":"IFAST","parallel_cameras":2}
    return extension, receipt
