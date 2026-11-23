// SPDX-License-Identifier: Apache-2.0
// JPEG round-trip for two independent uint8 OFT camera images.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <jpeglib.h>
#include <dlfcn.h>

namespace py = pybind11;
namespace {
struct Errors { jpeg_error_mgr base; jmp_buf jump; char* message; };
struct Buffer { unsigned char* data = nullptr; unsigned long length = 0; };

void jpeg_error(j_common_ptr c) {
  auto* e = reinterpret_cast<Errors*>(c->err);
  c->err->format_message(c, e->message);
  longjmp(e->jump, 1);
}

bool encode(const unsigned char* input, Buffer* buffer, char* message) {
  jpeg_compress_struct c{};
  Errors e{};
  c.err = jpeg_std_error(&e.base);
  e.base.error_exit = jpeg_error;
  e.message = message;
  if (setjmp(e.jump)) { jpeg_destroy_compress(&c); return false; }
  jpeg_create_compress(&c);
  jpeg_mem_dest(&c, &buffer->data, &buffer->length);
  c.image_width = c.image_height = 256;
  c.input_components = 3;
  c.in_color_space = JCS_RGB;
  jpeg_set_defaults(&c);
  jpeg_set_quality(&c, 95, TRUE);
  jpeg_start_compress(&c, TRUE);
  while (c.next_scanline < c.image_height) {
    JSAMPROW row = const_cast<JSAMPROW>(input + c.next_scanline * 256 * 3);
    jpeg_write_scanlines(&c, &row, 1);
  }
  jpeg_finish_compress(&c);
  jpeg_destroy_compress(&c);
  return true;
}

bool decode(const Buffer& buffer, unsigned char* output, bool fast_dct, char* message) {
  jpeg_decompress_struct c{};
  Errors e{};
  c.err = jpeg_std_error(&e.base);
  e.base.error_exit = jpeg_error;
  e.message = message;
  if (setjmp(e.jump)) { jpeg_destroy_decompress(&c); return false; }
  jpeg_create_decompress(&c);
  jpeg_mem_src(&c,buffer.data,buffer.length);
  jpeg_read_header(&c,TRUE);
  c.out_color_space = JCS_RGB;
  c.dct_method = fast_dct ? JDCT_IFAST : JDCT_ISLOW;
  c.do_fancy_upsampling = TRUE;
  jpeg_start_decompress(&c);
  while (c.output_scanline < c.output_height) {
    JSAMPROW row = output + c.output_scanline * 256 * 3;
    jpeg_read_scanlines(&c,&row,1);
  }
  jpeg_finish_decompress(&c);
  jpeg_destroy_decompress(&c);
  return true;
}

using RGB = py::array_t<uint8_t,py::array::c_style>;
py::array_t<uint8_t> roundtrip(const RGB& full, const RGB& wrist, bool fast_dct, bool parallel) {
  for (const auto* image : {&full,&wrist}) {
    if (image->ndim()!=3 || image->shape(0)!=256 || image->shape(1)!=256 || image->shape(2)!=3)
      throw py::value_error("Native JPEG requires two contiguous 256x256 RGB arrays");
  }
  const uint8_t* inputs[2] = {full.data(),wrist.data()};
  py::array_t<uint8_t> result({2,256,256,3});
  uint8_t* destination = result.mutable_data();
  char messages[2][JMSG_LENGTH_MAX] = {};
  {
    py::gil_scoped_release release;
    #pragma omp parallel for num_threads(2) if(parallel)
    for (int view=0;view<2;++view) {
      Buffer bytes;
      if (encode(inputs[view],&bytes,messages[view]))
        decode(bytes,destination + view*256*256*3,fast_dct,messages[view]);
      std::free(bytes.data);
    }
  }
  for (const auto& message : messages) if (message[0]) throw std::runtime_error(message);
  return result;
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roundtrip",&roundtrip,py::arg("full").noconvert(),py::arg("wrist").noconvert(),
        py::arg("fast_dct")=true,py::arg("parallel")=true);
  m.def("library_path",[]() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&jpeg_std_error),&info)) throw std::runtime_error("Cannot resolve libjpeg");
    return std::string(info.dli_fname);
  });
}
