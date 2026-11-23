// SPDX-License-Identifier: Apache-2.0
// Explicit v3: read a complete fresh native snapshot, then optionally reuse
// the caller's immutable lifecycle triple. No inventory/result is retained.
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <c10/util/SmallVector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <unordered_set>
#include <vector>

namespace {
using Objects = std::vector<py::object>;

py::object owned(PyObject* object) {
  if (!object) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(object);
}

py::object object_id(PyObject* object) { return owned(PyLong_FromVoidPtr(object)); }

struct Keys {
  py::object state = owned(PyUnicode_InternFromString("__dict__"));
  py::object parameters = owned(PyUnicode_InternFromString("_parameters"));
  py::object buffers = owned(PyUnicode_InternFromString("_buffers"));
  py::object modules = owned(PyUnicode_InternFromString("_modules"));
  std::array<py::object, 5> identity_fields{
      owned(PyUnicode_InternFromString("data_ptr")), owned(PyUnicode_InternFromString("_version")),
      owned(PyUnicode_InternFromString("dtype")), owned(PyUnicode_InternFromString("device")),
      owned(PyUnicode_InternFromString("shape"))};
};

PyObject* registry(PyObject* state, const py::object& key, const char* name) {
  PyObject* value = PyDict_GetItemWithError(state, key.ptr());
  if (!value) {
    if (PyErr_Occurred()) throw py::error_already_set();
    throw py::key_error(name);
  }
  if (!PyDict_CheckExact(value))
    throw py::type_error("Native lifecycle requires ordinary dict registries");
  return value;
}

void append_registry(PyObject* dictionary, Objects& names, Objects& values) {
  Py_ssize_t offset = 0;
  PyObject *name, *value;
  while (PyDict_Next(dictionary, &offset, &name, &value)) {
    names.push_back(py::reinterpret_borrow<py::object>(name));
    values.push_back(py::reinterpret_borrow<py::object>(value));
  }
}

struct Node {
  PyObject* object;  // Retained by model or the temporary children vector.
  py::object type;  // Capture the type now, before subsequent __dict__ callbacks.
  Py_ssize_t parameters, buffers, children;
};

struct TensorState {
  PyObject* object;  // Retained by the temporary registration value vectors.
  void* data;
  bool version_enabled;
  uint64_t version;
  py::object dtype;
  c10::Device device;
  c10::SmallVector<int64_t, 4> shape;
};

TensorState read_tensor(PyObject* object, const Keys& keys) {
  if (!THPVariable_CheckExact(object))
    throw py::type_error("Native lifecycle supports exact Tensor/Parameter objects only");
  PyObject** dictionary = _PyObject_GetDictPtr(object);
  if (dictionary && *dictionary) {
    for (const auto& key : keys.identity_fields) {
      if (PyDict_GetItemWithError(*dictionary, key.ptr()))
        throw py::type_error("Native lifecycle rejects shadowed tensor identity fields");
      if (PyErr_Occurred()) throw py::error_already_set();
    }
  }
  const at::Tensor& tensor = THPVariable_Unpack(object);
  const auto& counter = tensor.unsafeGetTensorImpl()->version_counter();
  const bool enabled = counter.enabled();
  const uint64_t version = enabled ? counter.current_version() : 0;
  const auto sizes = tensor.sizes();
  c10::SmallVector<int64_t, 4> shape(sizes.begin(), sizes.end());
  // All six fields are read unconditionally, before any comparison with old
  // evidence. In particular a changed id/topology cannot hide a later error.
  void* data = tensor.data_ptr();
  py::object dtype = py::reinterpret_borrow<py::object>(
      reinterpret_cast<PyObject*>(torch::getTHPDtype(tensor.scalar_type())));
  const auto device = tensor.device();
  return {object, data, enabled, version, std::move(dtype), device, std::move(shape)};
}

std::vector<TensorState> read_unique(const Objects& values, const Keys& keys,
                                    std::pmr::memory_resource& arena) {
  std::pmr::unordered_set<PyObject*> seen{&arena};
  seen.reserve(values.size());
  std::vector<TensorState> result;
  result.reserve(values.size());
  for (const auto& value : values)
    if (!value.is_none() && seen.insert(value.ptr()).second)
      result.push_back(read_tensor(value.ptr(), keys));
  return result;
}

struct Snapshot {
  std::vector<Node> nodes;
  Objects parameter_names, parameters, buffer_names, buffers, child_names, children;
  std::vector<TensorState> parameter_state, buffer_state;
};

Snapshot read_snapshot(PyObject* model, const Keys& keys, std::pmr::memory_resource& arena) {
  Snapshot result;
  std::pmr::unordered_set<PyObject*> visited{&arena};
  std::vector<PyObject*> stack{model};
  while (!stack.empty()) {
    PyObject* module = stack.back();
    stack.pop_back();
    if (module == Py_None || !visited.insert(module).second) continue;
    py::object state = owned(PyObject_GetAttr(module, keys.state.ptr()));
    if (!PyDict_CheckExact(state.ptr()))
      throw py::type_error("Native lifecycle requires an ordinary module __dict__");
    PyObject* ps = registry(state.ptr(), keys.parameters, "_parameters");
    PyObject* bs = registry(state.ptr(), keys.buffers, "_buffers");
    PyObject* cs = registry(state.ptr(), keys.modules, "_modules");
    result.nodes.push_back({module,
        py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(Py_TYPE(module))),
        PyDict_Size(ps), PyDict_Size(bs), PyDict_Size(cs)});
    append_registry(ps, result.parameter_names, result.parameters);
    append_registry(bs, result.buffer_names, result.buffers);
    const size_t begin = result.children.size();
    append_registry(cs, result.child_names, result.children);
    for (size_t i = result.children.size(); i > begin; --i)
      stack.push_back(result.children[i - 1].ptr());
  }
  // Preserve v2's phase order and separate alias deduplication. Registry values
  // remain strongly referenced through both passes and materialization/reuse.
  result.parameter_state = read_unique(result.parameters, keys, arena);
  result.buffer_state = read_unique(result.buffers, keys, arena);
  return result;
}

bool tuple_size(PyObject* object, size_t size) {
  return PyTuple_CheckExact(object) && static_cast<size_t>(PyTuple_GET_SIZE(object)) == size;
}

bool integer_equals(PyObject* object, uint64_t value) {
  if (!PyLong_CheckExact(object)) return false;  // Never invoke __index__/__eq__.
  const auto number = PyLong_AsUnsignedLongLong(object);
  if (PyErr_Occurred()) {
    // Negative/oversized exact integers are malformed old evidence, not an
    // error in the live snapshot. Materialize and leave refresh semantics alone.
    if (!PyErr_ExceptionMatches(PyExc_OverflowError)) throw py::error_already_set();
    PyErr_Clear();
    return false;
  }
  return number == value;
}

bool pointer_equals(PyObject* object, const void* value) {
  static_assert(sizeof(uintptr_t) <= sizeof(uint64_t));
  return integer_equals(object, reinterpret_cast<uintptr_t>(value));
}

bool names_equal(PyObject* previous, const Objects& names) {
  if (!tuple_size(previous, names.size())) return false;
  for (size_t i = 0; i < names.size(); ++i) {
    PyObject* old = PyTuple_GET_ITEM(previous, i);
    PyObject* now = names[i].ptr();
    if (!PyUnicode_CheckExact(old) || !PyUnicode_CheckExact(now)) return false;
    const int compared = PyUnicode_Compare(old, now);
    if (compared == -1 && PyErr_Occurred()) throw py::error_already_set();
    if (compared != 0) return false;
  }
  return true;
}

bool ids_equal(PyObject* previous, const Objects& values) {
  if (!tuple_size(previous, values.size())) return false;
  for (size_t i = 0; i < values.size(); ++i)
    if (!pointer_equals(PyTuple_GET_ITEM(previous, i), values[i].ptr())) return false;
  return true;
}

bool tensor_states_equal(PyObject* previous, const std::vector<TensorState>& values) {
  if (!tuple_size(previous, values.size())) return false;
  for (size_t i = 0; i < values.size(); ++i) {
    PyObject* old = PyTuple_GET_ITEM(previous, i);
    if (!tuple_size(old, 6)) return false;
    const auto& now = values[i];
    if (!pointer_equals(PyTuple_GET_ITEM(old, 0), now.object) ||
        !pointer_equals(PyTuple_GET_ITEM(old, 1), now.data)) return false;
    PyObject* version = PyTuple_GET_ITEM(old, 2);
    if (now.version_enabled ? !integer_equals(version, now.version) : version != Py_None) return false;
    if (PyTuple_GET_ITEM(old, 3) != now.dtype.ptr()) return false;
    PyObject* device = PyTuple_GET_ITEM(old, 4);
    if (Py_TYPE(device) != &THPDeviceType || reinterpret_cast<THPDevice*>(device)->device != now.device)
      return false;
    PyObject* shape = PyTuple_GET_ITEM(old, 5);
    if (!tuple_size(shape, now.shape.size())) return false;
    for (size_t d = 0; d < now.shape.size(); ++d)
      if (!integer_equals(PyTuple_GET_ITEM(shape, d), static_cast<uint64_t>(now.shape[d]))) return false;
  }
  return true;
}

bool topology_equal(PyObject* previous, const Snapshot& snapshot) {
  if (!tuple_size(previous, 7)) return false;
  PyObject* nodes = PyTuple_GET_ITEM(previous, 0);
  if (!tuple_size(nodes, snapshot.nodes.size())) return false;
  for (size_t i = 0; i < snapshot.nodes.size(); ++i) {
    PyObject* old = PyTuple_GET_ITEM(nodes, i);
    if (!tuple_size(old, 5)) return false;
    const auto& now = snapshot.nodes[i];
    PyObject* old_type = PyTuple_GET_ITEM(old, 1);
    // Metaclass equality may execute Python. Do not call it earlier than the
    // existing cache.refresh() would; materialize instead, even for same type.
    if (!PyType_Check(old_type) || Py_TYPE(old_type) != &PyType_Type ||
        Py_TYPE(now.type.ptr()) != &PyType_Type || old_type != now.type.ptr()) return false;
    if (!pointer_equals(PyTuple_GET_ITEM(old, 0), now.object) ||
        !integer_equals(PyTuple_GET_ITEM(old, 2), now.parameters) ||
        !integer_equals(PyTuple_GET_ITEM(old, 3), now.buffers) ||
        !integer_equals(PyTuple_GET_ITEM(old, 4), now.children)) return false;
  }
  return names_equal(PyTuple_GET_ITEM(previous, 1), snapshot.parameter_names) &&
      ids_equal(PyTuple_GET_ITEM(previous, 2), snapshot.parameters) &&
      names_equal(PyTuple_GET_ITEM(previous, 3), snapshot.buffer_names) &&
      ids_equal(PyTuple_GET_ITEM(previous, 4), snapshot.buffers) &&
      names_equal(PyTuple_GET_ITEM(previous, 5), snapshot.child_names) &&
      ids_equal(PyTuple_GET_ITEM(previous, 6), snapshot.children);
}

bool reusable(PyObject* previous, const Snapshot& snapshot) {
  // Called only after every live field has been read; early mismatch here is
  // safe. Success requires exact equality of every field with no Python eq.
  return tuple_size(previous, 3) &&
      tensor_states_equal(PyTuple_GET_ITEM(previous, 0), snapshot.parameter_state) &&
      tensor_states_equal(PyTuple_GET_ITEM(previous, 1), snapshot.buffer_state) &&
      topology_equal(PyTuple_GET_ITEM(previous, 2), snapshot);
}

py::tuple objects_tuple(Objects& values) {
  py::tuple result(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    PyTuple_SET_ITEM(result.ptr(), i, values[i].release().ptr());
  return result;
}

py::tuple ids(const Objects& values) {
  py::tuple result(values.size());
  for (size_t i = 0; i < values.size(); ++i) result[i] = object_id(values[i].ptr());
  return result;
}

py::tuple materialize_tensors(const std::vector<TensorState>& values) {
  py::tuple result(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const auto& value = values[i];
    py::tuple shape(value.shape.size());
    for (size_t j = 0; j < value.shape.size(); ++j) shape[j] = py::int_(value.shape[j]);
    py::object version = value.version_enabled ? py::cast(value.version) : py::none();
    result[i] = py::make_tuple(object_id(value.object), owned(PyLong_FromVoidPtr(value.data)), version,
                             value.dtype, owned(THPDevice_New(value.device)), shape);
  }
  return result;
}

py::tuple materialize(Snapshot& snapshot) {
  py::tuple parameter_ids = ids(snapshot.parameters), buffer_ids = ids(snapshot.buffers), child_ids = ids(snapshot.children);
  py::tuple parameters = materialize_tensors(snapshot.parameter_state);
  py::tuple buffers = materialize_tensors(snapshot.buffer_state);
  py::tuple nodes(snapshot.nodes.size());
  for (size_t i = 0; i < snapshot.nodes.size(); ++i) {
    const auto& node = snapshot.nodes[i];
    nodes[i] = py::make_tuple(object_id(node.object), node.type, node.parameters, node.buffers, node.children);
  }
  return py::make_tuple(parameters, buffers,
      py::make_tuple(nodes, objects_tuple(snapshot.parameter_names), parameter_ids,
                     objects_tuple(snapshot.buffer_names), buffer_ids, objects_tuple(snapshot.child_names), child_ids));
}

py::tuple collect(py::object model, py::object previous, const Keys& keys) {
  // GIL and request ownership stay held. No state, inventory, or result survives
  // this call beyond immutable tuples already owned by the caller's cache.
  std::byte arena_buffer[64 * 1024];
  std::pmr::monotonic_buffer_resource arena(arena_buffer, sizeof(arena_buffer));
  auto snapshot = read_snapshot(model.ptr(), keys, arena);
  if (reusable(previous.ptr(), snapshot)) return py::reinterpret_borrow<py::tuple>(previous);
  // A mismatch/unsupported previous object never starts a second model walk.
  return materialize(snapshot);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  auto keys = std::make_shared<Keys>();
  module.def("collect", [keys](py::object model, py::object previous) {
    return collect(std::move(model), std::move(previous), *keys);
  }, py::arg("model"), py::arg("previous") = py::none());
  module.def("compiler", []() { return __VERSION__; });
}
