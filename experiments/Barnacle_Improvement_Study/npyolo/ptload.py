"""Read an Ultralytics/PyTorch checkpoint WITHOUT PyTorch.

The pickle is treated as untrusted data: find_class never imports real code.
torch/ultralytics classes become inert stub records (class name + state dict),
tensors become numpy arrays read from the zip's raw storage files.
"""
import pickle
import zipfile
import collections
import numpy as np

_DTYPES = {
    'FloatStorage': np.float32, 'HalfStorage': np.float16, 'DoubleStorage': np.float64,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
    'BFloat16Storage': None,
}


class Stub:
    """Inert stand-in for any torch/ultralytics object."""
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = {**(state[0] or {}), **state[1]}
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__['_state'] = state

    def __repr__(self):
        return f'<Stub {getattr(self, "_cls", "?")}>'


def _stub_class(module, name):
    return type(name, (Stub,), {'_cls': f'{module}.{name}'})


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
    arr, = storage
    n = int(np.prod(size)) if len(size) else 1
    if not len(size):
        return arr[offset].copy()
    itemsize = arr.itemsize
    view = np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s*itemsize for s in stride))
    return np.array(view, copy=True)


def _rebuild_parameter(data, *rest):
    return data


def _rebuild_from_type_v2(func, new_type, args, state):
    return func(*args)


_SAFE = {
    ('collections', 'OrderedDict'): collections.OrderedDict,
    ('torch._utils', '_rebuild_tensor_v2'): _rebuild_tensor_v2,
    ('torch._utils', '_rebuild_parameter'): _rebuild_parameter,
    ('torch._tensor', '_rebuild_from_type_v2'): _rebuild_from_type_v2,
    ('builtins', 'set'): set,
    ('__builtin__', 'set'): set,
    ('builtins', 'dict'): dict,
    ('builtins', 'list'): list,
    ('builtins', 'tuple'): tuple,
    ('builtins', 'frozenset'): frozenset,
    ('builtins', 'slice'): slice,
    ('builtins', 'getattr'): None,  # refused below
}


class SafeUnpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf = zf
        self.prefix = prefix
        self.cache = {}

    def find_class(self, module, name):
        if (module, name) in _SAFE and _SAFE[(module, name)] is not None:
            return _SAFE[(module, name)]
        if module == 'torch' and name.endswith('Storage'):
            return _StorageType(name)
        if module.startswith(('torch', 'ultralytics', 'numpy', '__main__', 'pathlib',
                              'types', 'argparse', 'datetime', 'copy', 'functools')):
            return _stub_class(module, name)
        raise pickle.UnpicklingError(f'Refusing to load {module}.{name}')

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        typ, stype, key, location, numel = pid
        if key not in self.cache:
            name = stype.name if isinstance(stype, _StorageType) else str(stype)
            dtype = _DTYPES.get(name, np.float32)
            raw = self.zf.read(f'{self.prefix}/data/{key}')
            self.cache[key] = np.frombuffer(raw, dtype=dtype)
        return (self.cache[key],)


def load(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith('data.pkl')][0]
    prefix = pkl[:-len('/data.pkl')]
    with zf.open(pkl) as f:
        return SafeUnpickler(f, zf, prefix).load()


def modules(stub):
    return stub.__dict__.get('_modules', {})


def params(stub):
    out = dict(stub.__dict__.get('_parameters', {}) or {})
    out.update(stub.__dict__.get('_buffers', {}) or {})
    return {k: v for k, v in out.items() if v is not None}


def cls(stub):
    return getattr(stub, '_cls', type(stub).__name__).split('.')[-1]
