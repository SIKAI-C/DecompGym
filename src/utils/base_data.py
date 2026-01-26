"""Shared helpers for serializing dataclass-style problem inputs to JSON and
providing a base interface for synthetic instance generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any, Generic, TypeVar, get_args, get_origin

import numpy as np


TInst = TypeVar("TInst")


class JsonDataMixin:
    """Mixin that provides JSON round-tripping for dataclass containers."""

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} must be a dataclass to serialize")

        return {f.name: self._serialize_value(getattr(self, f.name)) for f in fields(self)}

    def to_json_file(self, path: str | Path, *, indent: int = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=indent)
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to deserialize")

        kwargs = {}
        for f in fields(cls):
            if f.name not in payload:
                raise KeyError(f"Missing field '{f.name}' in payload for {cls.__name__}")
            kwargs[f.name] = cls._deserialize_value(payload[f.name], f.type)
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> Any:
        with Path(path).open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls.from_dict(payload)

    @classmethod
    def _serialize_value(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, JsonDataMixin):
            return value.to_dict()
        if isinstance(value, list):
            return [cls._serialize_value(v) for v in value]
        if isinstance(value, tuple):
            return [cls._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._serialize_value(v) for k, v in value.items()}
        return value

    @classmethod
    def _deserialize_value(cls, value: Any, field_type: Any) -> Any:
        origin = get_origin(field_type)
        args = get_args(field_type)

        if value is None:
            return None

        if origin is list:
            inner_type = args[0] if args else Any
            return [cls._deserialize_value(v, inner_type) for v in value]

        if origin is tuple:
            inner_type = args[0] if args else Any
            return tuple(cls._deserialize_value(v, inner_type) for v in value)

        if origin is dict:
            key_type = args[0] if len(args) > 0 else Any
            val_type = args[1] if len(args) > 1 else Any
            return {cls._deserialize_value(k, key_type): cls._deserialize_value(v, val_type) for k, v in value.items()}

        if origin is not None:
            # Optional[T] shows up as Union[T, NoneType]
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return cls._deserialize_value(value, non_none[0])

        if isinstance(field_type, type) and issubclass(field_type, JsonDataMixin):
            return field_type.from_dict(value)

        if field_type is np.ndarray or field_type == np.ndarray:
            return np.asarray(value)

        return value


class BaseInstanceGenerator(ABC, Generic[TInst]):
    """Interface for generating, saving, and visualizing benchmark instances."""

    @abstractmethod
    def generate_instance(self, *, seed: int | None = None) -> TInst:
        """Create and return a single random/parametric instance."""

    def save_instance(self, inst: TInst, path: str | Path, *, indent: int = 2) -> Path:
        """Persist an instance to JSON using the JsonDataMixin contract."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(inst, JsonDataMixin):
            return inst.to_json_file(path, indent=indent)

        if is_dataclass(inst):
            payload = {f.name: JsonDataMixin._serialize_value(getattr(inst, f.name)) for f in fields(inst)}
            with path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=indent)
            return path

        raise TypeError("Instances must be dataclasses or JsonDataMixin subclasses to be saved as JSON.")

    def plot_instance(self, inst: TInst, output_path: str | Path | None = None) -> None:
        """Optional visualization of a generated instance; save to output_path if provided."""
        raise NotImplementedError("plot_instance should be implemented in concrete generators.")

    def plot_solution(self, inst: TInst, result: Any, output_path: str | Path | None = None) -> None:
        """Optional visualization of a solution on top of the instance; save to output_path if provided.

        The `result` is typically the finalized result dict returned by a baseline
        or Benders run; concrete generators should parse it and extract plotting
        data internally.
        """
        raise NotImplementedError("plot_solution should be implemented in concrete generators.")
