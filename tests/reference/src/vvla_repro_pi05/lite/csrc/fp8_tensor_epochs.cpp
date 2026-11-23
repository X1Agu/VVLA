// Host-only stateless validation. All current fields are read on every call.
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <c10/util/SmallVector.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
struct State {
  PyObject* object;
  void* data;
  bool version_enabled;
  uint64_t version;
  PyObject* dtype;
  c10::Device device;
  c10::SmallVector<int64_t, 4> shape;
};

bool tuple_size(PyObject* value, size_t size) {
  return PyTuple_CheckExact(value) && static_cast<size_t>(PyTuple_GET_SIZE(value)) == size;
}

uint64_t integer(PyObject* value) {
  if (!PyLong_CheckExact(value)) throw py::type_error("Epoch fields require exact nonnegative integers");
  const auto result = PyLong_AsUnsignedLongLong(value);
  if (PyErr_Occurred()) throw py::error_already_set();
  return result;
}

State read_tensor(PyObject* object, const std::array<py::object, 5>& keys) {
  if (!THPVariable_CheckExact(object))
    throw py::type_error("Native FP8 epochs support exact Tensor/Parameter objects only");
  PyObject** dictionary = _PyObject_GetDictPtr(object);
  if (dictionary && *dictionary) {
    if (!PyDict_CheckExact(*dictionary)) throw py::type_error("Tensor state must be an ordinary dict");
    for (const auto& key : keys) {
      if (PyDict_GetItemWithError(*dictionary, key.ptr()))
        throw py::type_error("Native FP8 epochs reject shadowed tensor identity fields");
      if (PyErr_Occurred()) throw py::error_already_set();
    }
  }
  // Never retain this Tensor/TensorImpl reference between calls.
  const at::Tensor& tensor = THPVariable_Unpack(object);
  const auto& counter = tensor.unsafeGetTensorImpl()->version_counter();
  const bool enabled = counter.enabled();
  const uint64_t version = enabled ? counter.current_version() : 0;
  const auto sizes = tensor.sizes();
  c10::SmallVector<int64_t, 4> shape(sizes.begin(), sizes.end());
  void* data = tensor.data_ptr();  // Includes current storage_offset.
  PyObject* dtype = reinterpret_cast<PyObject*>(torch::getTHPDtype(tensor.scalar_type()));
  const auto device = tensor.device();
  return {object, data, enabled, version, dtype, device, std::move(shape)};
}

void validate(py::object tensors, py::object expected) {
  if (!PyGILState_Check()) throw py::value_error("Native FP8 epochs require the GIL");
  if (!PyTuple_CheckExact(tensors.ptr()) || !PyTuple_CheckExact(expected.ptr()) ||
      PyTuple_GET_SIZE(tensors.ptr()) != PyTuple_GET_SIZE(expected.ptr()) ||
      PyTuple_GET_SIZE(tensors.ptr()) == 0)
    throw py::value_error("Native FP8 epochs require equal nonempty exact tuples");
  const std::array<py::object, 5> keys{py::str("data_ptr"), py::str("_version"),
      py::str("dtype"), py::str("device"), py::str("shape")};
  std::vector<State> current;
  current.reserve(PyTuple_GET_SIZE(tensors.ptr()));
  // Read all six fields of every current tensor before inspecting any expected
  // record. No early mismatch can hide a later live read/error.
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(tensors.ptr()); ++i)
    current.push_back(read_tensor(PyTuple_GET_ITEM(tensors.ptr(), i), keys));
  bool equal = true;
  for (size_t i = 0; i < current.size(); ++i) {
    PyObject* old = PyTuple_GET_ITEM(expected.ptr(), i);
    if (!tuple_size(old, 6)) throw py::value_error("Expected tensor epoch must have exactly six fields");
    const auto& now = current[i];
    const auto object = integer(PyTuple_GET_ITEM(old, 0));
    const auto data = integer(PyTuple_GET_ITEM(old, 1));
    PyObject* old_version = PyTuple_GET_ITEM(old, 2);
    const bool version_equal = old_version == Py_None ? !now.version_enabled :
        (integer(old_version) == now.version && now.version_enabled);
    PyObject* device = PyTuple_GET_ITEM(old, 4);
    if (Py_TYPE(device) != &THPDeviceType) throw py::type_error("Expected epoch device must be an exact torch.device");
    PyObject* shape = PyTuple_GET_ITEM(old, 5);
    if (!PyTuple_CheckExact(shape)) throw py::type_error("Expected epoch shape must be an exact tuple");
    bool shape_equal = static_cast<size_t>(PyTuple_GET_SIZE(shape)) == now.shape.size();
    for (Py_ssize_t d = 0; d < PyTuple_GET_SIZE(shape); ++d) {
      const auto dimension = integer(PyTuple_GET_ITEM(shape, d));
      if (static_cast<size_t>(d) >= now.shape.size() || dimension != static_cast<uint64_t>(now.shape[d]))
        shape_equal = false;
    }
    equal = equal && object == reinterpret_cast<uintptr_t>(now.object) &&
        data == reinterpret_cast<uintptr_t>(now.data) && version_equal &&
        PyTuple_GET_ITEM(old, 3) == now.dtype &&
        reinterpret_cast<THPDevice*>(device)->device == now.device && shape_equal;
  }
  if (!equal) throw std::runtime_error("FP8 immutable packing epoch changed; retire and reinstall");
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("validate_tensor_epochs", &validate);
  module.def("compiler", []() { return std::string(__VERSION__); });
}
