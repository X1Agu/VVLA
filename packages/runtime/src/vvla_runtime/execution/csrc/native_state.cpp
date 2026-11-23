// SPDX-License-Identifier: Apache-2.0
// OFT cached-registration comparator. The caller owns every model reference.
// Keep the GIL and original check order. No tensor ops, GPU reads, or state cache.
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <ATen/PythonTorchFunctionTLS.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_set>

namespace {
py::object owned(PyObject* value) {
  if (!value) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(value);
}
bool tuple_size(PyObject* value, Py_ssize_t n) {
  return PyTuple_CheckExact(value) && PyTuple_GET_SIZE(value) == n;
}
PyObject* item(PyObject* value, Py_ssize_t i) { return PyTuple_GET_ITEM(value, i); }
bool differs(PyObject* a, PyObject* b) {
  const int result = PyObject_RichCompareBool(a, b, Py_NE);
  if (result < 0) throw py::error_already_set();
  return result != 0;
}
bool truth(PyObject* value) {
  const int result = PyObject_IsTrue(value);
  if (result < 0) throw py::error_already_set();
  return result != 0;
}
bool integer_equals(PyObject* value, uint64_t expected) {
  if (!PyLong_CheckExact(value)) return false;
  const auto result = PyLong_AsUnsignedLongLong(value);
  if (PyErr_Occurred()) throw py::error_already_set();
  return result == expected;
}
bool dimensions_equal(PyObject* previous, at::IntArrayRef current) {
  if (!tuple_size(previous, static_cast<Py_ssize_t>(current.size()))) return false;
  for (size_t i = 0; i < current.size(); ++i) {
    if (!PyLong_CheckExact(item(previous, i))) return false;
    const auto old = PyLong_AsLongLong(item(previous, i));
    if (PyErr_Occurred()) throw py::error_already_set();
    if (old != current[i]) return false;
  }
  return true;
}

struct Keys {
  std::array<py::object, 3> enumerators{
      owned(PyUnicode_InternFromString("named_modules")),
      owned(PyUnicode_InternFromString("named_parameters")),
      owned(PyUnicode_InternFromString("named_buffers"))};
};

// Raw type-dict lookup does not invoke descriptors, __getattribute__, __eq__, or
// custom metaclass behavior. Such models keep the original Python comparator.
PyObject* raw_type_attribute(PyTypeObject* type, const char* name) {
  PyObject* mro = type->tp_mro;
  if (!mro || !PyTuple_CheckExact(mro)) return nullptr;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(mro); ++i) {
    auto* base = reinterpret_cast<PyTypeObject*>(item(mro, i));
#if PY_VERSION_HEX >= 0x030C0000
    // CPython 3.12 keeps static builtin dictionaries (including object) in
    // interpreter state; base->tp_dict may be null. The public raw-dict API
    // returns a new reference and never invokes a Python attribute hook.
    auto dictionary = py::reinterpret_steal<py::object>(PyType_GetDict(base));
#else
    auto dictionary = py::reinterpret_borrow<py::object>(base->tp_dict);
#endif
    if (!dictionary) {
      if (PyErr_Occurred()) throw py::error_already_set();
      return nullptr;
    }
    if (!PyDict_CheckExact(dictionary.ptr())) return nullptr;
    PyObject* value = PyDict_GetItemString(dictionary.ptr(), name);
    // The live type/interpreter still owns its dictionary after the temporary
    // API reference is released. Keep the GIL and return its borrowed value.
    if (value) return value;
  }
  return nullptr;
}

bool supported_modules(PyObject* modules, PyObject* enumeration_types) {
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(enumeration_types); ++i) {
    PyObject* row = item(enumeration_types, i);
    if (!tuple_size(row, 2) || !PyType_Check(item(row, 0)) || !PyTuple_CheckExact(item(row, 1))) return false;
    auto* type = reinterpret_cast<PyTypeObject*>(item(row, 0));
    if (Py_TYPE(type) != &PyType_Type) return false;
    PyObject* type_getattribute = raw_type_attribute(type, "__getattribute__");
    PyObject* object_getattribute = raw_type_attribute(&PyBaseObject_Type, "__getattribute__");
    if (!object_getattribute || type_getattribute != object_getattribute) return false;
    PyObject* dictionary = raw_type_attribute(type, "__dict__");
    if (!dictionary || Py_TYPE(dictionary) != &PyGetSetDescr_Type) return false;
    for (const char* name : {"training", "_modules", "_parameters", "_buffers"})
      if (raw_type_attribute(type, name)) return false;
    for (const char* name : {"named_modules", "named_parameters", "named_buffers"}) {
      PyObject* method = raw_type_attribute(type, name);
      if (!method || !PyFunction_Check(method)) return false;
    }
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(modules); ++i) {
    PyObject* row = item(modules, i);
    if (!tuple_size(row, 5) || !PyType_Check(item(row, 1)) ||
        !PyUnicode_CheckExact(item(row, 2)) || !PyUnicode_CheckExact(item(row, 3)) ||
        !PyTuple_CheckExact(item(row, 4))) return false;
    // A type replacement is detected by the main comparator before attributes.
    if (reinterpret_cast<PyObject*>(Py_TYPE(item(row, 0))) != item(row, 1)) continue;
    PyObject** state = _PyObject_GetDictPtr(item(row, 0));
    if (!state || !*state || !PyDict_CheckExact(*state)) return false;
    for (const char* name : {"training", "_modules", "_parameters", "_buffers"})
      if (!PyDict_GetItemString(*state, name)) return false;
    if (!PyBool_Check(PyDict_GetItemString(*state, "training"))) return false;
    for (auto raw_registry : py::reinterpret_borrow<py::tuple>(item(row, 4))) {
      PyObject* registry = raw_registry.ptr();
      if (!tuple_size(registry, 3) || !PyUnicode_CheckExact(item(registry, 0)) ||
          !PyDict_CheckExact(item(registry, 1)) || !PyTuple_CheckExact(item(registry, 2))) return false;
      Py_ssize_t offset = 0;
      PyObject *key, *value;
      while (PyDict_Next(item(registry, 1), &offset, &key, &value))
        if (!PyUnicode_CheckExact(key)) return false;
      for (auto raw_slot : py::reinterpret_borrow<py::tuple>(item(registry, 2)))
        if (!tuple_size(raw_slot.ptr(), 2) || !PyUnicode_CheckExact(item(raw_slot.ptr(), 0))) return false;
    }
  }
  return true;
}

// Check unsupported tensor APIs before invoking any module getters. In these
// cases None asks the caller to execute its original Python comparator once.
bool supported_tensors(PyObject* tensors, PyObject* api) {
  if (!PyTuple_CheckExact(tensors) || !PyTuple_CheckExact(api)) return false;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(api); ++i) {
    PyObject* guard = item(api, i);
    if (!tuple_size(guard, 2) || !PyType_Check(item(guard, 0)) || !PyTuple_CheckExact(item(guard, 1))) return false;
    PyObject* fields = item(guard, 1);
    for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(fields); ++j) {
      PyObject* field = item(fields, j);
      if (!tuple_size(field, 2) || !PyUnicode_CheckExact(item(field, 0))) return false;
      const char* name = PyUnicode_AsUTF8(item(field, 0));
      if (!name) throw py::error_already_set();
      // Do not invoke an unknown descriptor merely to decide eligibility.
      // Python fallback must be its first instance-access observation.
      PyObject* current = raw_type_attribute(reinterpret_cast<PyTypeObject*>(item(guard, 0)), name);
      if (current != item(field, 1)) return false;
    }
  }
  constexpr std::array<const char*, 6> names = {"data_ptr", "_version", "shape", "stride", "dtype", "device"};
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(tensors); ++i) {
    PyObject* row = item(tensors, i);
    if (!tuple_size(row, 2) || !tuple_size(item(row, 1), 7)) return false;
    PyObject* object = item(row, 0);
    if (!THPVariable_CheckExact(object)) return false;
    PyObject** state = _PyObject_GetDictPtr(object);
    if (state && *state) {
      if (!PyDict_CheckExact(*state)) return false;
      for (const char* name : names)
        if (PyDict_GetItemString(*state, name)) return false;
    }
    // Preserve Python behavior for unusual tensor layouts rather than changing
    // the order/errors of their data_ptr/stride accessors.
    if (THPVariable_Unpack(object).layout() != c10::kStrided) return false;
    PyObject* signature = item(row, 1);
    if (!PyLong_CheckExact(item(signature, 0)) || !PyLong_CheckExact(item(signature, 1)) ||
        !PyLong_CheckExact(item(signature, 2)) || !PyTuple_CheckExact(item(signature, 3)) ||
        !PyTuple_CheckExact(item(signature, 4)) || Py_TYPE(item(signature, 6)) != &THPDeviceType)
      return false;
    for (int index : {3, 4})
      for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(item(signature, index)); ++j)
        if (!PyLong_CheckExact(item(item(signature, index), j))) return false;
  }
  return true;
}

py::object compare_tensor_fields(py::tuple tensors) {
  for (auto raw : tensors) {
    PyObject* row = raw.ptr();
    const auto& tensor = THPVariable_Unpack(item(row, 0));
    if (tensor.layout() != c10::kStrided) return py::none();
    PyObject* signature = item(row, 1);
    const auto& version = tensor.unsafeGetTensorImpl()->version_counter();
    if (!version.enabled())
      throw py::value_error("Persistent model tensors need version counters; construct/install outside inference_mode");
    if (!integer_equals(item(signature, 2), version.current_version())) return py::bool_(false);
    if (!integer_equals(item(signature, 1), reinterpret_cast<uintptr_t>(tensor.data_ptr()))) return py::bool_(false);
    if (item(signature, 5) != reinterpret_cast<PyObject*>(torch::getTHPDtype(tensor.scalar_type()))) return py::bool_(false);
    if (reinterpret_cast<THPDevice*>(item(signature, 6))->device != tensor.device()) return py::bool_(false);
    if (!dimensions_equal(item(signature, 3), tensor.sizes()) ||
        !dimensions_equal(item(signature, 4), tensor.strides())) return py::bool_(false);
  }
  return py::bool_(true);
}

py::object compare(py::object model, py::tuple modules, py::tuple tensors,
                   py::tuple enumeration_types, py::tuple tensor_api, const Keys& keys) {
  if (modules.size() == 0) return py::bool_(false);
  if (!tuple_size(item(modules.ptr(), 0), 5)) return py::none();
  if (item(item(modules.ptr(), 0), 0) != model.ptr()) return py::bool_(false);
  // Metadata access can participate in a TorchFunctionMode. Do not bypass a
  // caller's mode even if it currently disables subclass dispatch internally.
  if (at::impl::PythonTorchFunctionTLS::stack_len() != 0) return py::none();
  if (!supported_modules(modules.ptr(), enumeration_types.ptr()) ||
      !supported_tensors(tensors.ptr(), tensor_api.ptr())) return py::none();
  for (auto raw : enumeration_types) {
    PyObject* row = raw.ptr();
    if (!tuple_size(row, 2) || !PyTuple_CheckExact(item(row, 1))) return py::none();
    for (auto raw_method : py::reinterpret_borrow<py::tuple>(item(row, 1))) {
      PyObject* method = raw_method.ptr();
      if (!tuple_size(method, 2)) throw py::type_error("Malformed OFT enumerator snapshot");
      auto current = owned(PyObject_GetAttr(item(row, 0), item(method, 0)));
      if (current.ptr() != item(method, 1)) return py::bool_(false);
    }
  }
  for (auto raw : modules) {
    PyObject* row = raw.ptr();
    PyObject* module = item(row, 0);
    PyObject* module_type = item(row, 1);
    if (reinterpret_cast<PyObject*>(Py_TYPE(module)) != module_type) return py::bool_(false);
    auto type_module = owned(PyObject_GetAttrString(module_type, "__module__"));
    if (!PyUnicode_CheckExact(type_module.ptr())) return py::none();
    if (differs(type_module.ptr(), item(row, 2))) return py::bool_(false);
    auto type_name = owned(PyObject_GetAttrString(module_type, "__qualname__"));
    if (!PyUnicode_CheckExact(type_name.ptr())) return py::none();
    if (differs(type_name.ptr(), item(row, 3))) return py::bool_(false);
    // Match the original ordinary-module access order. Custom accessors were
    // rejected without invoking them and will run once via Python fallback.
    for (const auto& key : keys.enumerators) {
      auto state = owned(PyObject_GetAttrString(module, "__dict__"));
      const int contains = PySequence_Contains(state.ptr(), key.ptr());
      if (contains < 0) throw py::error_already_set();
      if (contains) return py::bool_(false);
    }
    auto training = owned(PyObject_GetAttrString(module, "training"));
    if (truth(training.ptr())) throw py::value_error("All model submodules must remain in eval mode");
    for (auto raw_registry : py::reinterpret_borrow<py::tuple>(item(row, 4))) {
      PyObject* registry = raw_registry.ptr();
      auto current = owned(PyObject_GetAttr(module, item(registry, 0)));
      PyObject* slots = item(registry, 2);
      if (current.ptr() != item(registry, 1)) return py::bool_(false);
      const auto length = PyObject_Length(current.ptr());
      if (length < 0) throw py::error_already_set();
      if (length != PyTuple_GET_SIZE(slots)) return py::bool_(false);
      auto iterator = owned(PyObject_GetIter(current.ptr()));
      for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(slots); ++j) {
        PyObject* slot = item(slots, j);
        if (!tuple_size(slot, 2)) throw py::type_error("Malformed OFT registration snapshot");
        PyObject* next = PyIter_Next(iterator.ptr());
        if (!next) {
          if (PyErr_Occurred()) throw py::error_already_set();
          return py::bool_(false);
        }
        auto key = owned(next);
        if (differs(key.ptr(), item(slot, 0))) return py::bool_(false);
        auto value = owned(PyObject_GetItem(current.ptr(), item(slot, 0)));
        if (value.ptr() != item(slot, 1)) return py::bool_(false);
      }
    }
  }
  return compare_tensor_fields(tensors);
}
}  // namespace

#if PY_VERSION_HEX >= 0x030C0000 && PY_VERSION_HEX < 0x030D0000
// CPython 3.12 gives every dictionary write a new version. A version hit only
// shortens the registration traversal. TensorImpl geometry and versions stay live.
uint64_t dict_version(PyObject* dict) {
  return reinterpret_cast<PyDictObject*>(dict)->ma_version_tag;
}

struct WatchedInstance {
  PyObject* object;
  PyTypeObject* type;
  PyObject* dictionary = nullptr;
  uint64_t version;

  explicit WatchedInstance(PyObject* value) : object(value), type(Py_TYPE(value)), version(0) {
    PyObject** slot = _PyObject_GetDictPtr(value);
    if (slot && *slot) {
      dictionary = *slot;
      version = dict_version(*slot);
    }
  }

  bool unchanged() const {
    if (Py_TYPE(object) != type) return false;
    PyObject** slot = _PyObject_GetDictPtr(object);
    PyObject* now = slot ? *slot : nullptr;
    return now == dictionary && (!now || dict_version(now) == version);
  }
};

struct WatchedType {
  PyTypeObject* type;
  py::object mro;
  PyObject* dictionary;
  uint64_t version;
  explicit WatchedType(PyTypeObject* t) : type(t), mro(py::reinterpret_borrow<py::object>(t->tp_mro)) {
    auto now = owned(PyType_GetDict(t));
    dictionary = now.ptr();
    version = dict_version(now.ptr());
  }
  bool unchanged() const {
    auto now = owned(PyType_GetDict(type));
    return mro.ptr() == type->tp_mro && dictionary == now.ptr() && version == dict_version(now.ptr());
  }
};

class VersionedComparator {
 public:
  py::object check(py::object model, py::tuple modules, py::tuple tensors, py::tuple types, py::tuple api) {
    if (at::impl::PythonTorchFunctionTLS::stack_len() != 0) return py::none();
    if (model.ptr() == model_.ptr() && modules.ptr() == modules_.ptr() && tensors.ptr() == tensors_.ptr()
        && types.ptr() == types_.ptr() && api.ptr() == api_.ptr() && structure_unchanged()) {
      return compare_tensor_fields(tensors);
    }
    auto result = compare(model, modules, tensors, types, api, keys_);
    if (result.ptr() == Py_True) {
      snapshot(model, modules, tensors, types, api);
    } else {
      clear();
    }
    return result;
  }

 private:
  Keys keys_;
  py::object model_, modules_, tensors_, types_, api_;
  std::vector<WatchedInstance> instances_;
  std::vector<std::pair<PyObject*,uint64_t>> registries_;
  std::vector<WatchedType> type_guards_;

  void clear() {
    instances_.clear(); registries_.clear(); type_guards_.clear(); labels_.clear();
    model_ = modules_ = tensors_ = types_ = api_ = py::object();
  }
  void snapshot(py::object model, py::tuple modules, py::tuple tensors, py::tuple types, py::tuple api) {
    clear();
    model_ = model; modules_ = modules; tensors_ = tensors; types_ = types; api_ = api;
    std::unordered_set<PyTypeObject*> seen;
    auto include_type = [&](PyObject* object) {
      auto* type = reinterpret_cast<PyTypeObject*>(object);
      for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(type->tp_mro); ++i) {
        auto* base = reinterpret_cast<PyTypeObject*>(item(type->tp_mro, i));
        if (seen.insert(base).second) type_guards_.emplace_back(base);
      }
    };
    for (auto row : types) {
      auto* t = item(row.ptr(),0);
      include_type(t);
      labels_.push_back({t, owned(PyObject_GetAttrString(t, "__module__")), owned(PyObject_GetAttrString(t, "__qualname__"))});
    }
    for (auto row : api) include_type(item(row.ptr(),0));
    for (auto row : modules) {
      instances_.emplace_back(item(row.ptr(),0));
      for (auto registry : py::reinterpret_borrow<py::tuple>(item(row.ptr(),4))) {
        PyObject* dict = item(registry.ptr(),1);
        registries_.emplace_back(dict,dict_version(dict));
      }
    }
    for (auto row : tensors) instances_.emplace_back(item(row.ptr(),0));
  }
  bool structure_unchanged() const {
    for (const auto& type : type_guards_) if (!type.unchanged()) return false;
    // Heap type names can change independently of the namespace dictionary.
    for (const auto& instance : instances_) if (!instance.unchanged()) return false;
    for (const auto& entry : registries_) if (dict_version(entry.first) != entry.second) return false;
    for (const auto& label : labels_) {
      auto module = owned(PyObject_GetAttrString(label.type, "__module__"));
      auto name = owned(PyObject_GetAttrString(label.type, "__qualname__"));
      if (module.ptr() != label.module.ptr() || name.ptr() != label.name.ptr()) return false;
    }
    return true;
  }
  struct Label { PyObject* type; py::object module; py::object name; };
  std::vector<Label> labels_;
};
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
#if PY_VERSION_HEX >= 0x030C0000 && PY_VERSION_HEX < 0x030D0000
  py::class_<VersionedComparator>(module, "VersionedComparator").def(py::init<>()).def("compare", &VersionedComparator::check);
#endif
  auto keys = std::make_shared<Keys>();
  module.def("compare", [keys](py::object model, py::tuple modules, py::tuple tensors,
                              py::tuple enumeration_types, py::tuple tensor_api) {
    return compare(std::move(model), std::move(modules), std::move(tensors),
                   std::move(enumeration_types), std::move(tensor_api), *keys);
  });
  module.def("compiler", [] { return __VERSION__; });
}
