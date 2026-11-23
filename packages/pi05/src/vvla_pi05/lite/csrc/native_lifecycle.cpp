// SPDX-License-Identifier: Apache-2.0
// Live host metadata only. No tensor operators, content reads, CUDA APIs, or
// persistent inventory. The caller holds the same request lock as Python.
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>

#include <unordered_set>
#include <vector>

namespace {
using Objects = std::vector<py::object>;

py::object owned(PyObject* object) {
  if (object == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(object);
}

py::object object_id(PyObject* object) {
  return owned(PyLong_FromVoidPtr(object));
}

py::tuple object_tuple(const Objects& objects) {
  py::tuple result(objects.size());
  for (size_t i = 0; i < objects.size(); ++i) result[i] = objects[i];
  return result;
}

py::tuple ids(const Objects& objects) {
  py::tuple result(objects.size());
  for (size_t i = 0; i < objects.size(); ++i) result[i] = object_id(objects[i].ptr());
  return result;
}

PyObject* registry(PyObject* state, const char* name) {
  PyObject* value = PyDict_GetItemString(state, name);  // borrowed, under GIL
  if (!value) throw py::key_error(name);
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

py::tuple tensor_identity(PyObject* object) {
  // Python tensor subclasses can override any of the reference fields.
  // Reject them instead of silently reading a different ATen identity.
  if (!THPVariable_CheckExact(object))
    throw py::type_error("Native lifecycle supports exact Tensor/Parameter objects only");
  // Even exact Tensor instances may shadow data_ptr in their __dict__.
  // CPython exposes the dictionary without creating one for each tensor.
  PyObject** dictionary = _PyObject_GetDictPtr(object);
  if (dictionary && *dictionary) {
    for (const char* name : {"data_ptr", "_version", "dtype", "device", "shape"}) {
      if (PyDict_GetItemString(*dictionary, name))
        throw py::type_error("Native lifecycle rejects shadowed tensor identity fields");
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

py::tuple unique_identities(const Objects& values) {
  std::unordered_set<PyObject*> seen;
  seen.reserve(values.size());
  Objects result;
  result.reserve(values.size());
  for (const auto& value : values) {
    if (!value.is_none() && seen.insert(value.ptr()).second)
      result.push_back(tensor_identity(value.ptr()));
  }
  return object_tuple(result);
}

py::tuple collect(py::object model) {
  // Never release the GIL: registration dictionaries must not mutate while
  // iterated, and identity/version reads form the same host check as Python.
  std::unordered_set<PyObject*> visited;
  Objects stack{model}, nodes, parameter_names, parameters, buffer_names, buffers,
      child_names, children;
  while (!stack.empty()) {
    py::object module = std::move(stack.back());
    stack.pop_back();
    if (module.is_none() || !visited.insert(module.ptr()).second) continue;
    py::object state = owned(PyObject_GetAttrString(module.ptr(), "__dict__"));
    if (!PyDict_CheckExact(state.ptr()))
      throw py::type_error("Native lifecycle requires an ordinary module __dict__");
    PyObject* ps = registry(state.ptr(), "_parameters");
    PyObject* bs = registry(state.ptr(), "_buffers");
    PyObject* cs = registry(state.ptr(), "_modules");
    nodes.push_back(py::make_tuple(
        object_id(module.ptr()),
        py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(Py_TYPE(module.ptr()))),
        PyDict_Size(ps), PyDict_Size(bs), PyDict_Size(cs)));
    append_registry(ps, parameter_names, parameters);
    append_registry(bs, buffer_names, buffers);
    const size_t child_begin = children.size();
    append_registry(cs, child_names, children);
    for (size_t i = children.size(); i > child_begin; --i) stack.push_back(children[i - 1]);
  }
  // Strong references above retain all registration values until both tensor
  // passes finish, matching the Python collector's lists (including aliases).
  py::tuple parameter_ids = ids(parameters), buffer_ids = ids(buffers), child_ids = ids(children);
  py::tuple parameter_signature = unique_identities(parameters);
  py::tuple buffer_signature = unique_identities(buffers);
  return py::make_tuple(parameter_signature, buffer_signature,
      py::make_tuple(object_tuple(nodes), object_tuple(parameter_names), parameter_ids,
                     object_tuple(buffer_names), buffer_ids, object_tuple(child_names), child_ids));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("collect", &collect);
  module.def("compiler", []() { return __VERSION__; });
}
