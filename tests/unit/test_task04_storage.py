"""The small storage interface (local disk by default, S3 swappable later).

The task text does not name it, so these tests discover a module under backend/
holding a storage abstraction and a local-disk implementation, then exercise the
put/open/delete-style methods it exposes.
"""
import hashlib
import importlib
import inspect
import io
import pkgutil
import re

import pytest

from rs4_helpers import REPO, STORAGE_ENV_VARS, call, read_all, sha

SAVE_NAMES = ("put", "save", "store", "write", "upload", "put_stream", "save_stream",
              "put_file", "save_file", "put_object")
OPEN_NAMES = ("open", "get", "read", "open_read", "get_stream", "download", "stream",
              "get_file", "get_object", "read_bytes")
DELETE_NAMES = ("delete", "remove", "unlink", "rm", "delete_object")
EXISTS_NAMES = ("exists", "has", "contains", "is_file", "isfile")
FACTORY_NAMES = ("get_storage", "default_storage", "get_default_storage", "storage_from_env",
                 "make_storage", "create_storage", "build_storage", "get_file_storage")
STREAM_PARAM = re.compile(r"stream|file|fileobj|fp|data|content|src|source|body|reader", re.I)


def _storage_modules():
    import backend

    mods = []
    for info in pkgutil.walk_packages(backend.__path__, "backend."):
        if re.search(r"stor|file|blob|media", info.name.rsplit(".", 1)[-1], re.I):
            try:
                mods.append(importlib.import_module(info.name))
            except Exception:  # noqa: BLE001
                continue
    return mods


def _classes(mod):
    return [c for _, c in inspect.getmembers(mod, inspect.isclass) if c.__module__.startswith("backend")]


@pytest.fixture(scope="module")
def storage_mods():
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    mods = _storage_modules()
    assert mods, "no module under backend/ whose name mentions storage/files/blob/media"
    return mods


@pytest.fixture(scope="module")
def local_cls(storage_mods):
    for mod in storage_mods:
        for c in _classes(mod):
            if re.search(r"local|disk|filesystem|fs", c.__name__, re.I) and re.search(
                    r"stor|backend|store", c.__name__, re.I):
                return c
    pytest.fail("no local-disk storage class (name like LocalStorage / LocalDiskStorage) "
                f"in {[m.__name__ for m in storage_mods]}")


@pytest.fixture()
def storage(local_cls, storage_mods, tmp_path, monkeypatch):
    root = tmp_path / "outer" / "root"
    for var in STORAGE_ENV_VARS:
        monkeypatch.setenv(var, str(root))
    inst = None
    errors = []
    for kwargs in ({}, {"root": root}, {"base_dir": root}, {"base_path": root}, {"directory": root},
                   {"path": root}, {"root_dir": root}, {"storage_dir": root}, {"base": root}):
        try:
            inst = local_cls(**kwargs) if kwargs else local_cls(root)
            break
        except TypeError as exc:
            errors.append(repr(exc))
    if inst is None:
        try:
            inst = local_cls()
        except Exception as exc:  # noqa: BLE001
            for mod in storage_mods:
                for name in FACTORY_NAMES:
                    fn = getattr(mod, name, None)
                    if callable(fn):
                        inst = fn()
                        break
                if inst is not None:
                    break
            if inst is None:
                pytest.fail(f"cannot construct {local_cls.__name__}: {exc!r}; {errors}")
    inst._rs4_root = root
    return inst


def _method(obj, names):
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return n, fn
    return None, None


def _save(storage, key, stream):
    """Save `stream` under `key`. Supports save(key, stream) and the coder's
    save(drive_id, name, source), which returns the storage key."""
    name, fn = _method(storage, SAVE_NAMES)
    assert fn is not None, f"no save-like method among {SAVE_NAMES} on {type(storage).__name__}"
    params = [p for p in inspect.signature(fn).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in params if p.default is p.empty]
    if len(required) == 3:
        # (drive_id, name, source): split a nested key into drive id + file name
        head, _, tail = key.partition("/")
        if not tail:
            head, tail = "d", key
        saved = call(fn, head, tail, stream)
        return saved if isinstance(saved, str) else f"{head}/{tail}"
    assert len(required) <= 2, f"{name} needs more than (key, stream): {[p.name for p in required]}"
    first_is_stream = bool(params) and STREAM_PARAM.search(params[0].name) and not re.search(
        r"key|path|name", params[0].name, re.I)
    args = (stream, key) if first_is_stream else (key, stream)
    call(fn, *args)
    return key


def _open(storage, key):
    name, fn = _method(storage, OPEN_NAMES)
    assert fn is not None, f"no open/get-like method among {OPEN_NAMES} on {type(storage).__name__}"
    return read_all(call(fn, key))


def _delete(storage, key):
    name, fn = _method(storage, DELETE_NAMES)
    assert fn is not None, f"no delete-like method among {DELETE_NAMES} on {type(storage).__name__}"
    return call(fn, key)


def _files(root):
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


class LazyStream(io.RawIOBase):
    """A non-materialised file-like of `total` bytes that records how it is read."""

    def __init__(self, total: int):
        self.total = total
        self.pos = 0
        self.max_request = 0
        self.unbounded_reads = 0
        self.hasher = hashlib.sha256()
        self._chunk = bytes(range(251)) * 1000

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0 and offset == 0:
            self.pos = 0
            self.hasher = hashlib.sha256()
            return 0
        if whence == 2 and offset == 0:
            return self.total
        raise io.UnsupportedOperation("only rewind supported")

    def read(self, size=-1):
        if size is None or size < 0:
            self.unbounded_reads += 1
            size = self.total
        self.max_request = max(self.max_request, size)
        n = min(size, self.total - self.pos, 1 << 20)
        if n <= 0:
            return b""
        out = (self._chunk * (n // len(self._chunk) + 1))[:n]
        self.pos += n
        self.hasher.update(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


# ----------------------------------------------------------------- interface


def test_storage_module_has_an_abstraction_and_a_local_implementation(storage_mods, local_cls):
    classes = [c for m in storage_mods for c in _classes(m)]
    others = [c for c in classes if c is not local_cls and re.search(r"stor|backend", c.__name__, re.I)]
    # The abstraction is either a base class / Protocol of the local one, or at least a
    # separate named type that shares the method names.
    bases = [b for b in inspect.getmro(local_cls)[1:] if b is not object]
    assert bases or others, "no storage interface (ABC/Protocol/base class) next to the local class"


def test_local_storage_exposes_save_open_delete(storage):
    assert _method(storage, SAVE_NAMES)[1], f"no save-like method {SAVE_NAMES}"
    assert _method(storage, OPEN_NAMES)[1], f"no open-like method {OPEN_NAMES}"
    assert _method(storage, DELETE_NAMES)[1], f"no delete-like method {DELETE_NAMES}"


def test_put_then_open_round_trips_bytes(storage):
    payload = b"RoadScout storage round trip \x00\x01\x02" * 100
    _save(storage, "drive-1/video.mp4", io.BytesIO(payload))
    assert _open(storage, "drive-1/video.mp4") == payload


def test_put_writes_to_the_local_directory(storage):
    payload = b"on disk " * 1000
    _save(storage, "d1/clip.bin", io.BytesIO(payload))
    root = storage._rs4_root
    assert any(sha(p.read_bytes()) == sha(payload) for p in _files(root)), (
        f"payload not found under {root}")


def test_put_creates_missing_directories(storage):
    _save(storage, "newdrive/deep.bin", io.BytesIO(b"deep"))
    assert _open(storage, "newdrive/deep.bin") == b"deep"


def test_put_is_isolated_per_key(storage):
    _save(storage, "k1/f.bin", io.BytesIO(b"one"))
    _save(storage, "k2/f.bin", io.BytesIO(b"two"))
    assert _open(storage, "k1/f.bin") == b"one"
    assert _open(storage, "k2/f.bin") == b"two"


def test_put_handles_empty_stream(storage):
    _save(storage, "e/empty.bin", io.BytesIO(b""))
    assert _open(storage, "e/empty.bin") == b""


def test_exists_if_provided(storage):
    _, fn = _method(storage, EXISTS_NAMES)
    if fn is None:
        pytest.skip("storage exposes no exists()")
    assert not call(fn, "nope/missing.bin")
    _save(storage, "yes/there.bin", io.BytesIO(b"x"))
    assert call(fn, "yes/there.bin")


def test_delete_removes_the_object(storage):
    _save(storage, "gone/x.bin", io.BytesIO(b"bye"))
    _delete(storage, "gone/x.bin")
    with pytest.raises(Exception):  # noqa: B017  (FileNotFoundError / KeyError / custom)
        _open(storage, "gone/x.bin")
    assert not any(p.name == "x.bin" for p in _files(storage._rs4_root))


def test_open_missing_key_raises(storage):
    with pytest.raises(Exception):  # noqa: B017
        _open(storage, "never/existed.bin")


def test_path_traversal_keys_cannot_escape_the_root(storage):
    root = storage._rs4_root
    outside = root.parent / "escape.bin"
    try:
        _save(storage, "../escape.bin", io.BytesIO(b"escaped"))
    except Exception:  # noqa: BLE001  (rejecting the key is fine)
        pass
    assert not outside.exists(), "a '../' key wrote outside the storage root"
    try:
        _save(storage, "../../escape2.bin", io.BytesIO(b"escaped"))
    except Exception:  # noqa: BLE001
        pass
    assert not (root.parent.parent / "escape2.bin").exists()


def test_put_streams_in_chunks_instead_of_loading_everything(storage):
    """A 48 MiB non-materialised stream: no unbounded read(), requests stay chunk-sized."""
    total = 48 * 1024 * 1024
    stream = LazyStream(total)
    _save(storage, "big/stream.bin", stream)
    assert stream.unbounded_reads == 0, "storage called read() without a size (loads all in memory)"
    assert stream.max_request <= 16 * 1024 * 1024, (
        f"storage asked for {stream.max_request} bytes in one read")
    assert stream.pos == total
    stored = [p for p in _files(storage._rs4_root) if p.name == "stream.bin"]
    assert stored and stored[0].stat().st_size == total
    h = hashlib.sha256()
    with open(stored[0], "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    assert h.hexdigest() == stream.hasher.hexdigest()


def test_open_returns_something_streamable_or_bytes(storage):
    """open() hands back a file-like/iterator/bytes; reading it back works."""
    _save(storage, "s/open.bin", io.BytesIO(b"abc" * 1000))
    _, fn = _method(storage, OPEN_NAMES)
    obj = call(fn, "s/open.bin")
    assert isinstance(obj, (bytes, bytearray)) or hasattr(obj, "read") or hasattr(obj, "__iter__") \
        or hasattr(obj, "__enter__") or hasattr(obj, "__fspath__")
    assert read_all(obj) == b"abc" * 1000


def test_local_storage_root_comes_from_environment_when_no_argument(local_cls, tmp_path, monkeypatch):
    """The default (no-argument) construction must honour ROADSCOUT_STORAGE_DIR."""
    root = tmp_path / "envroot"
    monkeypatch.setenv("ROADSCOUT_STORAGE_DIR", str(root))
    try:
        inst = local_cls()
    except TypeError:
        pytest.skip("local storage requires an explicit root argument")
    _save(inst, "env/k.bin", io.BytesIO(b"env"))
    assert any(p.read_bytes() == b"env" for p in _files(root)), (
        "no-argument storage ignored ROADSCOUT_STORAGE_DIR")
