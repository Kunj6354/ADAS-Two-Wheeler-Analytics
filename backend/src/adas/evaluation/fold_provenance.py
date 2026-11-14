"""Independent research-only fold provenance validation."""

from collections.abc import Mapping
import re

PROVENANCE_VERSION = "adas.fold-provenance.v1"
WINDOW_IDENTITY_VERSION = "adas.window-identity.sorted-json-lf.v1"

REQUIRED = frozenset({
    "provenance_schema_version",
    "window_identity_version",
    "window_inventory_sha256",
    "eligible_window_count",
    "runtime",
    "solver_configuration",
})

RUNTIME_FIELDS = frozenset({
    "python", "numpy", "scipy",
})

SOLVER_FIELDS = frozenset({
    "method", "objective", "time_limit_seconds",
})


def _digest(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def validate_fold_provenance(
    provenance,
    *,
    expected_window_sha256,
    expected_window_count,
    expected_runtime,
    expected_solver_configuration,
):
    """Compare provenance with independently supplied expectations.

    Never interprets a validated draft as an approved experiment.
    Performs no filesystem access.
    """
    if not isinstance(provenance, Mapping):
        raise ValueError("Invalid provenance object")

    if set(provenance) != REQUIRED:
        raise ValueError("Missing or unexpected provenance fields")

    if provenance["provenance_schema_version"] != PROVENANCE_VERSION:
        raise ValueError("Unsupported provenance version")

    if provenance["window_identity_version"] != WINDOW_IDENTITY_VERSION:
        raise ValueError("Unsupported window-identity version")

    if not _digest(expected_window_sha256):
        raise ValueError("Invalid trusted window digest")

    if (
        not _digest(provenance["window_inventory_sha256"])
        or provenance["window_inventory_sha256"]
        != expected_window_sha256
    ):
        raise ValueError("Window-inventory fingerprint mismatch")

    if (
        type(expected_window_count) is not int
        or expected_window_count <= 0
        or type(provenance["eligible_window_count"]) is not int
        or provenance["eligible_window_count"] != expected_window_count
    ):
        raise ValueError("Eligible-window count mismatch")

    runtime = provenance["runtime"]

    if (
        not isinstance(runtime, Mapping)
        or not isinstance(expected_runtime, Mapping)
        or set(runtime) != RUNTIME_FIELDS
        or set(expected_runtime) != RUNTIME_FIELDS
    ):
        raise ValueError("Runtime inventory mismatch")

    for field in RUNTIME_FIELDS:
        expected = expected_runtime[field]
        actual = runtime[field]

        if (
            not isinstance(expected, str)
            or not expected.strip()
            or not isinstance(actual, str)
            or actual != expected
        ):
            raise ValueError(f"Runtime mismatch: {field}")

    solver = provenance["solver_configuration"]

    if (
        not isinstance(solver, Mapping)
        or not isinstance(expected_solver_configuration, Mapping)
        or set(solver) != SOLVER_FIELDS
        or set(expected_solver_configuration) != SOLVER_FIELDS
    ):
        raise ValueError("Solver configuration inventory mismatch")

    for field in ("method", "objective"):
        expected = expected_solver_configuration[field]
        if (
            not isinstance(expected, str)
            or not expected.strip()
            or solver[field] != expected
        ):
            raise ValueError(f"Solver configuration mismatch: {field}")

    limit = solver["time_limit_seconds"]
    trusted_limit = expected_solver_configuration["time_limit_seconds"]

    if (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or isinstance(trusted_limit, bool)
        or not isinstance(trusted_limit, (int, float))
        or not (0 < limit == trusted_limit < float("inf"))
    ):
        raise ValueError("Solver time-limit mismatch")

    return {
        "valid": True,
        "approval_status": "NOT_APPROVED",
        "window_count": expected_window_count,
        "window_inventory_sha256": expected_window_sha256,
        "provenance_schema_version": PROVENANCE_VERSION,
    }
