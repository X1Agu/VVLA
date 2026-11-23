// SPDX-License-Identifier: Apache-2.0
// Allocation variant of native_lifecycle.cpp; identical live signature fields.
// Only fixed API-name strings live across calls. Each invocation rebuilds all
// registries/sets/identities in fresh storage. No tensor work or CUDA APIs.
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>

#include <unordered_set>
#include <vector>
#include <array>
#include <cstddef>
#include <memory>
#include <memory_resource>

namespace {
using Objects = std::vector<py::object>;

py::object owned(PyObject* object) {
  if (object == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(object);
}

py::object object_id(PyObject* object) {
  return owned(PyLong_FromVoidPtr(object));
}

// Immutable API names only. Owned by the Python function's closure so teardown
// runs under the GIL; no process-static py::object destruction after shutdown.
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

py::tuple object_tuple(Objects& objects) {
  py::tuple result(objects.size());
  // Fresh tuple has null slots; steal our already-owned references directly.
  // This helper consumes its vector only after all dependent reads finish.
  for (size_t i = 0; i < objects.size(); ++i)
    PyTuple_SET_ITEM(result.ptr(), i, objects[i].release().ptr());
  return result;
}

py::tuple ids(const Objects& objects) {
  py::tuple result(objects.size());
  for (size_t i = 0; i < objects.size(); ++i) result[i] = object_id(objects[i].ptr());
  return result;
}

PyObject* registry(PyObject* state, const py::object& key, const char* name) {
  PyObject* value = PyDict_GetItemWithError(state, key.ptr());  // borrowed, under GIL
  if (!value) {
    if (PyErr_Occurred()) throw py::error_already_set();
    throw py::key_error(name);
  }
  // A subclass could override keys/values/len and change the Python reference.
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

py::tuple tensor_identity(PyObject* object, const Keys& keys) {
  // Python tensor subclasses can override any of the reference fields.
  // Reject them instead of silently reading a different ATen identity.
  if (!THPVariable_CheckExact(object))
    throw py::type_error("Native lifecycle supports exact Tensor/Parameter objects only");
  // Even exact Tensor instances may shadow data_ptr in their __dict__.
  // CPython exposes the dictionary without creating one for each tensor.
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
  py::object version = counter.enabled() ? py::cast(counter.current_version()) : py::none();
  py::tuple shape(tensor.dim());
  const auto sizes = tensor.sizes();
  for (size_t i = 0; i < sizes.size(); ++i) shape[i] = py::int_(sizes[i]);
  return py::make_tuple(
      object_id(object), owned(PyLong_FromVoidPtr(tensor.data_ptr())), version,
      py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(torch::getTHPDtype(tensor.scalar_type()))),
      owned(THPDevice_New(tensor.device())), shape);
}

py::tuple unique_identities(const Objects& values, const Keys& keys, std::pmr::memory_resource& arena) {
  std::pmr::unordered_set<PyObject*> seen{&arena};
  seen.reserve(values.size());
  Objects result;
  result.reserve(values.size());
  for (const auto& value : values) {
    if (!value.is_none() && seen.insert(value.ptr()).second)
      result.push_back(tensor_identity(value.ptr(), keys));
  }
  return object_tuple(result);
}

py::tuple collect(py::object model, const Keys& keys) {
  // Never release the GIL: registration dictionaries must not mutate while
  // iterated, and identity/version reads form the same host check as Python.
  // Memory is owned only by this invocation. Large/dynamically expanded models
  // transparently use upstream allocations, also freed before return.
  std::byte arena_buffer[64 * 1024];
  std::pmr::monotonic_buffer_resource arena(arena_buffer, sizeof(arena_buffer));
  std::pmr::unordered_set<PyObject*> visited{&arena};
  // Root is retained by the argument and every pushed child by `children`.
  // Borrowed stack pointers avoid duplicate ownership but remain live even
  // when a custom module __dict__ access rewires an earlier registration.
  std::vector<PyObject*> stack{model.ptr()};
  Objects nodes, parameter_names, parameters, buffer_names, buffers,
      child_names, children;
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
    nodes.push_back(py::make_tuple(
        object_id(module),
        py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(Py_TYPE(module))),
        PyDict_Size(ps), PyDict_Size(bs), PyDict_Size(cs)));
    append_registry(ps, parameter_names, parameters);
    append_registry(bs, buffer_names, buffers);
    const size_t child_begin = children.size();
    append_registry(cs, child_names, children);
    for (size_t i = children.size(); i > child_begin; --i) stack.push_back(children[i - 1].ptr());
  }
  // Strong references above retain all registration values until both tensor
  // passes finish, matching the Python collector's lists (including aliases).
  py::tuple parameter_ids = ids(parameters), buffer_ids = ids(buffers), child_ids = ids(children);
  py::tuple parameter_signature = unique_identities(parameters, keys, arena);
  py::tuple buffer_signature = unique_identities(buffers, keys, arena);
  return py::make_tuple(parameter_signature, buffer_signature,
      py::make_tuple(object_tuple(nodes), object_tuple(parameter_names), parameter_ids,
                     object_tuple(buffer_names), buffer_ids, object_tuple(child_names), child_ids));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  auto keys = std::make_shared<Keys>();
  module.def("collect", [keys](py::object model) { return collect(std::move(model), *keys); });
  module.def("compiler", []() { return __VERSION__; });
}
