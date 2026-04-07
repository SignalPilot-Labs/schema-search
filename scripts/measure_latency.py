#!/usr/bin/env python3
"""Standalone CLI script to measure TCP socket connection latency to a database host.

Reports min/avg/max/jitter stats — useful for diagnosing slow schema-search
connections to remote databases.

Usage:
    python scripts/measure_latency.py --host localhost --port 5432
    python scripts/measure_latency.py --url postgresql://user:pass@db.host.com:5432/mydb
"""

import argparse
import socket
import statistics
import sys
import time
from urllib.parse import urlparse

DEFAULT_PORTS: dict[str, int] = {
    "postgresql": 5432,
    "postgres": 5432,
    "mysql": 3306,
    "snowflake": 443,
    "bigquery": 443,
    "databricks": 443,
}

FALLBACK_PORT: int = 443


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Measure TCP connection latency to a database host.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        type=str,
        help="SQLAlchemy-style connection URL (e.g. postgresql://user:pass@host:5432/db)",
    )
    group.add_argument(
        "--host",
        type=str,
        help="Raw hostname or IP address to probe.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to probe. Overrides or supplements the port from --url. Default: 443.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of probe attempts. Default: 10.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-probe socket timeout in seconds. Default: 5.0.",
    )
    return parser.parse_args()


def extract_host_port(url: str, port_override: int | None) -> tuple[str, int]:
    """Parse a SQLAlchemy-style connection URL and return the host and port.

    Maps database dialect to a sensible default port when the URL omits one.
    The port_override, if provided, always wins.

    Args:
        url: A SQLAlchemy-style connection URL such as
            ``postgresql://user:pass@db.host.com:5432/mydb`` or
            ``snowflake://account.region/db``.
        port_override: An explicit port supplied via --port, or None.

    Returns:
        A (host, port) tuple.

    Raises:
        ValueError: If the URL cannot be parsed or no host is found.
    """
    # Strip dialect+driver prefix noise so urlparse handles it cleanly.
    # e.g. "snowflake+connector://..." -> "snowflake://..."
    # urlparse needs a proper scheme to work correctly.
    parsed = urlparse(url)
    scheme = parsed.scheme  # may be "postgresql+psycopg2", "snowflake", etc.
    dialect = scheme.split("+")[0].lower() if scheme else ""

    host = parsed.hostname
    if not host:
        raise ValueError(f"Could not extract host from URL: {url!r}")

    # Determine port: URL-explicit > dialect default
    url_port: int | None = parsed.port
    dialect_default: int = DEFAULT_PORTS.get(dialect, FALLBACK_PORT)
    resolved_port: int = url_port if url_port is not None else dialect_default

    # port_override always takes precedence
    if port_override is not None:
        resolved_port = port_override

    return host, resolved_port


def probe_once(host: str, port: int, timeout: float) -> tuple[float | None, str]:
    """Attempt a single TCP connection and return the latency in milliseconds.

    Args:
        host: Hostname or IP address to connect to.
        port: TCP port to connect to.
        timeout: Socket timeout in seconds.

    Returns:
        Tuple of (latency_ms, error_detail). On success latency_ms is a float
        and error_detail is empty. On failure latency_ms is None and error_detail
        describes the failure reason.
    """
    start = time.perf_counter()
    try:
        conn = socket.create_connection((host, port), timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        conn.close()
        return elapsed_ms, ""
    except socket.timeout:
        return None, "Timed out"
    except OSError as exc:
        return None, exc.strerror if exc.strerror else str(exc)


def compute_stats(samples: list[float]) -> dict[str, float]:
    """Compute min, avg, max, and jitter from a list of latency samples.

    Args:
        samples: Non-empty list of latency values in milliseconds.

    Returns:
        Dictionary with keys: ``min``, ``avg``, ``max``, ``jitter``.
    """
    minimum = min(samples)
    maximum = max(samples)
    avg = statistics.mean(samples)
    jitter = statistics.stdev(samples) if len(samples) >= 2 else 0.0
    return {"min": minimum, "avg": avg, "max": maximum, "jitter": jitter}


def main() -> None:
    """Orchestrate argument parsing, probing, stats computation, and output."""
    args = parse_args()

    if args.url:
        try:
            host, port = extract_host_port(args.url, args.port)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        host = args.host
        port = args.port if args.port is not None else FALLBACK_PORT

    count: int = args.count
    timeout: float = args.timeout

    print(f"Measuring latency to {host}:{port} ({count} probes, timeout={timeout}s)")
    print()

    samples: list[float] = []
    failures: int = 0
    probe_width = len(str(count))

    for i in range(1, count + 1):
        latency, error_detail = probe_once(host, port, timeout)
        if latency is not None:
            samples.append(latency)
            print(f"Probe {i:{probe_width}d}: {latency:7.2f} ms")
        else:
            failures += 1
            print(f"Probe {i:{probe_width}d}:   FAILED ({error_detail})")

    print()

    successful = len(samples)
    print(f"Results ({successful}/{count} successful):")

    if successful == 0:
        print("  No successful probes. Cannot compute stats.")
        sys.exit(1)

    stats = compute_stats(samples)
    print(f"  Min:    {stats['min']:7.2f} ms")
    print(f"  Avg:    {stats['avg']:7.2f} ms")
    print(f"  Max:    {stats['max']:7.2f} ms")
    print(f"  Jitter: {stats['jitter']:7.2f} ms")


if __name__ == "__main__":
    main()
