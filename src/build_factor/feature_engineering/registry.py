"""
feature_engineering/registry.py
---------------------------------
Global stage registry. Use @register to make a FeatureStage discoverable
by name, so run_experiment.py can build pipelines from string configs.

Usage
-----
    from feature_engineering.registry import register, get_stage, list_stages

    @register
    class MyStage(FeatureStage):
        name = "my_stage"
        ...

    stage = get_stage("my_stage")()   # instantiate
"""

from __future__ import annotations

from typing import Type
from .base import FeatureStage

_REGISTRY: dict[str, Type[FeatureStage]] = {}


def register(cls: Type[FeatureStage]) -> Type[FeatureStage]:
    """Class decorator — add cls to the global stage registry."""
    _REGISTRY[cls.name.fget(cls)] = cls  # type: ignore[attr-defined]
    return cls


def get_stage(name: str) -> Type[FeatureStage]:
    if name not in _REGISTRY:
        raise KeyError(f"Stage '{name}' not found. Available: {list_stages()}")
    return _REGISTRY[name]


def list_stages() -> list[str]:
    return sorted(_REGISTRY)


def build_pipeline_from_names(
    names: list[str],
    **pipeline_kwargs,
):
    """Convenience: build a FeaturePipeline from a list of stage names."""
    from .base import FeaturePipeline
    stages = [get_stage(n)() for n in names]
    return FeaturePipeline(stages, **pipeline_kwargs)
