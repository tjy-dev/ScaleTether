from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


EVENT_KINDS = {"compute", "collective", "memory", "synchronization"}
COLLECTIVES = {
    "unknown",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_to_all",
    "reduce",
    "gather",
    "scatter",
    "barrier",
    "send",
    "recv",
}


@dataclass(frozen=True)
class TraceEvent:
    id: str
    name: str
    kind: str
    duration_us: float
    stream: str = "0"
    rank: int = 0
    device: int = 0
    dependencies: tuple[str, ...] = ()
    observed_start_us: float | None = None
    collective: str | None = None
    message_bytes: int | None = None
    group_role: str | None = None
    group_size: int | None = None
    sm_fraction: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        raw_message_bytes = data.get("message_bytes")
        metadata = dict(data.get("metadata", {}))
        message_bytes = None if raw_message_bytes is None else int(raw_message_bytes)
        if message_bytes == 0 and data.get("collective") == "unknown":
            # Recover traces emitted by the early v0.1 profiler adapter, which
            # represented NCCL bookkeeping/zero-element records as a literal
            # zero-byte unknown collective. Keep an explicit audit marker;
            # known collectives and negative sizes remain validation errors.
            metadata["message_bytes_recovery"] = {
                "status": "normalized-zero-to-unknown",
                "observed_message_bytes": 0,
            }
            message_bytes = None
        event = cls(
            id=str(data["id"]),
            name=str(data["name"]),
            kind=str(data["kind"]),
            duration_us=float(data["duration_us"]),
            stream=str(data.get("stream", "0")),
            rank=int(data.get("rank", 0)),
            device=int(data.get("device", 0)),
            dependencies=tuple(str(value) for value in data.get("dependencies", ())),
            observed_start_us=(
                None
                if data.get("observed_start_us") is None
                else float(data["observed_start_us"])
            ),
            collective=data.get("collective"),
            message_bytes=message_bytes,
            group_role=data.get("group_role"),
            group_size=None
            if data.get("group_size") is None
            else int(data["group_size"]),
            sm_fraction=(
                None if data.get("sm_fraction") is None else float(data["sm_fraction"])
            ),
            metadata=metadata,
        )
        event.validate()
        return event

    def validate(self) -> None:
        if not self.id or not self.name:
            raise ValueError("trace events require non-empty id and name")
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unsupported event kind {self.kind!r}")
        if self.duration_us < 0:
            raise ValueError(f"event {self.id!r} has negative duration")
        if self.kind == "collective" and self.collective not in COLLECTIVES:
            raise ValueError(
                f"collective event {self.id!r} requires one of {sorted(COLLECTIVES)}"
            )
        if self.message_bytes is not None and self.message_bytes <= 0:
            raise ValueError(f"event {self.id!r} has invalid message_bytes")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError(f"event {self.id!r} has invalid group_size")
        if self.sm_fraction is not None and not 0.0 <= self.sm_fraction <= 1.0:
            raise ValueError(f"event {self.id!r} has invalid sm_fraction")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dependencies"] = list(self.dependencies)
        return result


@dataclass(frozen=True)
class WorkloadTrace:
    events: tuple[TraceEvent, ...]
    source: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadTrace":
        if str(data.get("schema_version", "0.1")) != "0.1":
            raise ValueError("unsupported workload schema version")
        trace = cls(
            events=tuple(TraceEvent.from_dict(item) for item in data.get("events", ())),
            source=dict(data.get("source", {})),
            metadata=dict(data.get("metadata", {})),
            schema_version="0.1",
        )
        trace.validate()
        return trace

    @classmethod
    def load(cls, path: Path) -> "WorkloadTrace":
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if not self.events:
            raise ValueError("captured workload contains no events")
        ids = {event.id for event in self.events}
        if len(ids) != len(self.events):
            raise ValueError("trace event ids must be unique")
        for event in self.events:
            unknown = set(event.dependencies) - ids
            if unknown:
                raise ValueError(
                    f"event {event.id!r} has unknown dependencies {sorted(unknown)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "metadata": self.metadata,
            "events": [event.to_dict() for event in self.events],
        }

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
