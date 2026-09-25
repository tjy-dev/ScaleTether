from __future__ import annotations


CTA_POLICY_FLAGS = {
    "default": 0x00,
    "efficiency": 0x01,
    "zero": 0x02,
    "efficiency+zero": 0x03,
}


def parse_cta_policy_mode(value: str | int | None) -> int:
    if value is None:
        return CTA_POLICY_FLAGS["default"]
    if isinstance(value, int):
        if value not in set(CTA_POLICY_FLAGS.values()):
            raise ValueError(
                "NCCL CTA policy must be a supported flag value 0 through 3"
            )
        return value
    normalized = value.strip().lower().replace("|", "+")
    try:
        return CTA_POLICY_FLAGS[normalized]
    except KeyError as error:
        raise ValueError(
            "NCCL CTA policy must be default, efficiency, zero, or efficiency+zero"
        ) from error


def cta_policy_name(value: int) -> str:
    for name, flag in CTA_POLICY_FLAGS.items():
        if flag == value:
            return name
    raise ValueError(f"unsupported NCCL CTA policy flag {value}")


def parse_cga_cluster_size(value: int | None) -> int | None:
    """Validate NCCL's public thread-block-cluster X dimension.

    ``None`` leaves NCCL's automatic selection untouched.  NCCL documents
    explicit values from 0 (default behavior) through 8.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 8:
        raise ValueError("NCCL CGA cluster size must be an integer from 0 through 8")
    return value
