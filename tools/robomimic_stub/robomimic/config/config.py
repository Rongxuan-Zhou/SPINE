"""Minimal Config implementation for MimicGen compatibility."""

from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Any, Dict, Iterable, Iterator, Tuple


class Config:
    """
    Lightweight nested config with attribute access and optional key locking.

    This is intentionally minimal to satisfy MimicGen's config usage.
    """

    def __init__(self, dict_to_load: Dict[str, Any] | None = None) -> None:
        super().__setattr__("_data", {})
        super().__setattr__("_locked", False)
        super().__setattr__("_do_not_lock", False)
        if dict_to_load is not None:
            self._load_from_dict(dict_to_load)

    def _load_from_dict(self, dic: Dict[str, Any]) -> None:
        for key, value in dic.items():
            if isinstance(value, dict):
                child = Config(value)
                self._data[key] = child
            else:
                self._data[key] = value

    def _assert_mutable(self, key: str) -> None:
        if self._locked and not self._do_not_lock and key not in self._data:
            raise KeyError(f"Config is locked. Key '{key}' does not exist.")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name not in self._data:
            if self._locked and not self._do_not_lock:
                raise AttributeError(f"Config has no attribute '{name}' (locked).")
            child = Config()
            self._data[name] = child
            return child
        return self._data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        self._assert_mutable(name)
        if isinstance(value, dict):
            value = Config(value)
        self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._assert_mutable(key)
        if isinstance(value, dict):
            value = Config(value)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        if self._locked and not self._do_not_lock:
            raise KeyError(f"Config is locked. Key '{key}' cannot be deleted.")
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        return self._data.items()

    def values(self) -> Iterable[Any]:
        return self._data.values()

    def to_dict(self) -> Dict[str, Any]:
        def convert(val: Any) -> Any:
            if isinstance(val, Config):
                return {k: convert(v) for k, v in val.items()}
            return val
        return {k: convert(v) for k, v in self._data.items()}

    def dump(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=False, indent=4)

    def update(self, other: Dict[str, Any] | "Config") -> None:
        if isinstance(other, Config):
            other = other.to_dict()
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self._data or not isinstance(self._data[key], Config):
                    self._assert_mutable(key)
                    self._data[key] = Config()
                self._data[key].update(value)
            else:
                self._assert_mutable(key)
                self._data[key] = value

    def do_not_lock_keys(self) -> None:
        self._do_not_lock = True
        self._locked = False

    def lock_keys(self) -> None:
        if self._do_not_lock:
            return
        self._locked = True
        for value in self._data.values():
            if isinstance(value, Config):
                value.lock_keys()

    @contextmanager
    def values_unlocked(self) -> Iterator["Config"]:
        previous = []

        def _collect(node: Config) -> None:
            previous.append((node, node._locked))
            node._locked = False
            for child in node._data.values():
                if isinstance(child, Config):
                    _collect(child)

        _collect(self)
        try:
            yield self
        finally:
            for node, locked in previous:
                node._locked = locked
