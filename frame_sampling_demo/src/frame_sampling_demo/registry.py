from __future__ import annotations

from typing import Any

SAMPLERS: dict[str, type] = {}


def register_sampler(name: str):
    def decorator(cls):
        if name in SAMPLERS:
            raise ValueError(f"Sampler already registered: {name}")
        SAMPLERS[name] = cls
        cls.name = name
        return cls

    return decorator


def get_sampler(name: str):
    try:
        return SAMPLERS[name]
    except KeyError as error:
        available = ", ".join(sorted(SAMPLERS))
        raise ValueError(f"Unknown sampler '{name}'. Available: {available}") from error


def available_samplers() -> list[str]:
    return sorted(SAMPLERS)
