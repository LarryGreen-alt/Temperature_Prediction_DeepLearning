from __future__ import annotations

import argparse
import os
import shutil
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.preprocess import preprocess
from src.data.split_dataset import split_dataset
from src.utils.city_coordinates import CITY_COORDINATES, get_coordinates

"""
The script will:

Read every city from CITY_COORDINATES.
Download each city from 2015-01-01 through 2026-07-01.
Validate that each downloaded CSV covers the full requested range.
Preserve the previous city CSV if a replacement download fails.
Run preprocessing once after every city succeeds.
Rebuild train.csv, dev.csv, and test.csv.
Stop before publishing mixed date ranges when any city fails.
"""


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
CACHE_DIR = RAW_DIR / ".historical_download_cache"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2026-07-01"
DEFAULT_CHUNK_DAYS = 365

HOURLY_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "is_day",
]

REQUIRED_COLUMNS = ["time", *HOURLY_COLUMNS]

RETRYABLE_HTTP_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD."
        ) from error


def city_slug(city_name: str) -> str:
    return (
        city_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


def create_directories() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        SPLIT_DIR,
        CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def build_session(
    retries: int,
    backoff_factor: float,
) -> requests.Session:
    """
    Create one HTTP session with connection pooling and automatic retries.

    Retries apply to connection errors, read errors, rate limits, and common
    temporary server errors. Retry-After is honored when Open-Meteo returns it.
    """
    retry_policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=sorted(RETRYABLE_HTTP_CODES),
        backoff_factor=backoff_factor,
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "WeatherTemperatureAI/1.0 "
                "(historical-dataset-updater)"
            )
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def iter_date_chunks(
    start_date: date,
    end_date: date,
    chunk_days: int,
) -> Iterator[tuple[date, date]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1.")

    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(
            chunk_start + timedelta(days=chunk_days - 1),
            end_date,
        )
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


def expected_hour_count(
    start_date: date,
    end_date: date,
) -> int:
    return ((end_date - start_date).days + 1) * 24


def parse_api_error(response: requests.Response) -> str:
    """Extract a useful Open-Meteo error reason from a failed response."""
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        reason = payload.get("reason") or payload.get("message")
        if reason:
            return str(reason)

    body = response.text.strip()
    if body:
        return body[:500]

    return response.reason or "Unknown HTTP error"


def validate_hourly_dataframe(
    dataframe: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    source_name: str,
) -> pd.DataFrame:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{source_name} is missing columns: {missing_columns}"
        )

    dataframe = dataframe[REQUIRED_COLUMNS].copy()

    timestamps = pd.to_datetime(
        dataframe["time"],
        errors="coerce",
        utc=True,
    )
    invalid_time_count = int(timestamps.isna().sum())
    if invalid_time_count:
        raise ValueError(
            f"{source_name} contains {invalid_time_count:,} "
            "invalid timestamps."
        )

    dataframe["_timestamp"] = timestamps
    dataframe = (
        dataframe.sort_values("_timestamp")
        .drop_duplicates(subset=["_timestamp"], keep="last")
        .reset_index(drop=True)
    )

    first_expected = pd.Timestamp(
        start_date,
        tz="UTC",
    )
    last_expected = pd.Timestamp(
        end_date + timedelta(days=1),
        tz="UTC",
    ) - pd.Timedelta(hours=1)

    actual_first = dataframe["_timestamp"].min()
    actual_last = dataframe["_timestamp"].max()

    if actual_first != first_expected or actual_last != last_expected:
        raise ValueError(
            f"{source_name} covers {actual_first} through {actual_last}; "
            f"expected {first_expected} through {last_expected}."
        )

    expected_rows = expected_hour_count(start_date, end_date)
    if len(dataframe) != expected_rows:
        raise ValueError(
            f"{source_name} contains {len(dataframe):,} unique hourly "
            f"rows; expected {expected_rows:,}. This indicates missing "
            "or duplicated hours."
        )

    null_counts = dataframe[HOURLY_COLUMNS].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(
            f"{source_name} contains missing weather values:\n"
            f"{null_counts.to_string()}"
        )

    dataframe = dataframe.drop(columns="_timestamp")
    return dataframe


def load_and_validate_csv(
    path: Path,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return validate_hourly_dataframe(
        dataframe,
        start_date=start_date,
        end_date=end_date,
        source_name=str(path),
    )


def chunk_cache_path(
    city_name: str,
    model_name: str,
    chunk_start: date,
    chunk_end: date,
) -> Path:
    city_cache = CACHE_DIR / city_slug(city_name) / model_name
    city_cache.mkdir(parents=True, exist_ok=True)

    return city_cache / (
        f"{chunk_start.isoformat()}__{chunk_end.isoformat()}.csv"
    )


def download_chunk(
    session: requests.Session,
    *,
    city_name: str,
    latitude: float,
    longitude: float,
    chunk_start: date,
    chunk_end: date,
    model_name: str,
    connect_timeout: float,
    read_timeout: float,
) -> pd.DataFrame:
    params: dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": chunk_start.isoformat(),
        "end_date": chunk_end.isoformat(),
        "hourly": ",".join(HOURLY_COLUMNS),
        "timezone": "GMT",
    }

    # Omitting "models" asks Open-Meteo for its default best-match series.
    if model_name != "best_match":
        params["models"] = model_name

    try:
        response = session.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            timeout=(connect_timeout, read_timeout),
        )
    except requests.Timeout as error:
        raise RuntimeError(
            f"Open-Meteo timed out for {city_name.title()} "
            f"({chunk_start} through {chunk_end})."
        ) from error
    except requests.ConnectionError as error:
        raise RuntimeError(
            f"Could not connect to Open-Meteo for "
            f"{city_name.title()} ({chunk_start} through {chunk_end})."
        ) from error
    except requests.RequestException as error:
        raise RuntimeError(
            f"Open-Meteo request failed for {city_name.title()} "
            f"({chunk_start} through {chunk_end}): {error}"
        ) from error

    if not response.ok:
        reason = parse_api_error(response)
        raise RuntimeError(
            f"Open-Meteo returned HTTP {response.status_code} for "
            f"{city_name.title()} ({chunk_start} through {chunk_end}): "
            f"{reason}"
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Open-Meteo returned invalid JSON for "
            f"{city_name.title()} ({chunk_start} through {chunk_end})."
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Unexpected Open-Meteo response type for "
            f"{city_name.title()}: {type(payload).__name__}"
        )

    if payload.get("error"):
        raise RuntimeError(
            f"Open-Meteo rejected the request for "
            f"{city_name.title()}: {payload.get('reason', payload)}"
        )

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError(
            f"Open-Meteo response for {city_name.title()} does not "
            "contain an hourly data object."
        )

    dataframe = pd.DataFrame(hourly)
    return validate_hourly_dataframe(
        dataframe,
        start_date=chunk_start,
        end_date=chunk_end,
        source_name=(
            f"Open-Meteo response for {city_name.title()} "
            f"{chunk_start} through {chunk_end}"
        ),
    )


def get_or_download_chunk(
    session: requests.Session,
    *,
    city_name: str,
    latitude: float,
    longitude: float,
    chunk_start: date,
    chunk_end: date,
    model_name: str,
    connect_timeout: float,
    read_timeout: float,
    pause_seconds: float,
    force: bool,
) -> pd.DataFrame:
    cache_path = chunk_cache_path(
        city_name,
        model_name,
        chunk_start,
        chunk_end,
    )

    if cache_path.exists() and not force:
        try:
            dataframe = load_and_validate_csv(
                cache_path,
                start_date=chunk_start,
                end_date=chunk_end,
            )
            print(
                f"  Reused cached chunk: {chunk_start} through "
                f"{chunk_end}"
            )
            return dataframe
        except Exception as error:
            print(
                f"  Cached chunk is invalid and will be replaced: "
                f"{error}"
            )
            cache_path.unlink(missing_ok=True)

    print(
        f"  Downloading chunk: {chunk_start} through {chunk_end}"
    )
    dataframe = download_chunk(
        session,
        city_name=city_name,
        latitude=latitude,
        longitude=longitude,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        model_name=model_name,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )

    temporary_path = cache_path.with_suffix(
        cache_path.suffix + ".part"
    )
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, cache_path)

    if pause_seconds > 0:
        time.sleep(pause_seconds)

    return dataframe


def load_existing_city_source(
    city_path: Path,
) -> pd.DataFrame | None:
    """
    Load an existing raw city file once so its valid hours can be reused.

    This performs structural validation but does not require the file to have
    the exact requested first and last dates. Exact range validation is done
    when a requested period or chunk is extracted.
    """
    if not city_path.exists():
        return None

    try:
        columns = pd.read_csv(city_path, nrows=0).columns.tolist()
    except Exception as error:
        print(
            f"[Existing file] Could not read {city_path}: {error}"
        )
        return None

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in columns
    ]
    if missing_columns:
        print(
            f"[Existing file] {city_path} cannot be reused because it "
            f"is missing columns: {missing_columns}"
        )
        return None

    try:
        dataframe = pd.read_csv(
            city_path,
            usecols=REQUIRED_COLUMNS,
        )
    except Exception as error:
        print(
            f"[Existing file] Could not load {city_path}: {error}"
        )
        return None

    timestamps = pd.to_datetime(
        dataframe["time"],
        errors="coerce",
        utc=True,
    )
    invalid_count = int(timestamps.isna().sum())
    if invalid_count:
        print(
            f"[Existing file] {city_path} contains "
            f"{invalid_count:,} invalid timestamps and cannot be reused."
        )
        return None

    dataframe["_timestamp"] = timestamps
    dataframe = (
        dataframe.sort_values("_timestamp")
        .drop_duplicates(subset=["_timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return dataframe


def extract_valid_existing_range(
    existing: pd.DataFrame | None,
    *,
    start_date: date,
    end_date: date,
    source_name: str,
) -> pd.DataFrame | None:
    """
    Extract and fully validate one requested date range from existing data.

    A range is reusable only when every expected hour is present and all
    required weather values are non-null.
    """
    if existing is None or existing.empty:
        return None

    first_timestamp = pd.Timestamp(start_date, tz="UTC")
    end_exclusive = pd.Timestamp(
        end_date + timedelta(days=1),
        tz="UTC",
    )

    selected = existing.loc[
        (existing["_timestamp"] >= first_timestamp)
        & (existing["_timestamp"] < end_exclusive),
        REQUIRED_COLUMNS,
    ].copy()

    if selected.empty:
        return None

    try:
        return validate_hourly_dataframe(
            selected,
            start_date=start_date,
            end_date=end_date,
            source_name=source_name,
        )
    except Exception:
        return None


def verify_existing_city_file(
    city_path: Path,
    *,
    start_date: date,
    end_date: date,
) -> tuple[bool, str, pd.DataFrame | None]:
    """
    Verify whether an existing city CSV completely covers the requested range.

    Returns:
        (is_complete, status_message, reusable_source_dataframe)
    """
    if not city_path.exists():
        return False, "file is missing", None

    existing = load_existing_city_source(city_path)
    if existing is None:
        return False, "file could not be validated", None

    reusable = extract_valid_existing_range(
        existing,
        start_date=start_date,
        end_date=end_date,
        source_name=str(city_path),
    )

    actual_start = existing["_timestamp"].min()
    actual_end = existing["_timestamp"].max()

    if reusable is not None:
        return (
            True,
            (
                f"complete: {actual_start.date()} through "
                f"{actual_end.date()}, with all requested hours present"
            ),
            existing,
        )

    expected_start = pd.Timestamp(start_date, tz="UTC")
    expected_end = pd.Timestamp(
        end_date + timedelta(days=1),
        tz="UTC",
    ) - pd.Timedelta(hours=1)

    missing_before = actual_start > expected_start
    missing_after = actual_end < expected_end

    if missing_before and missing_after:
        detail = (
            f"partial: available {actual_start.date()} through "
            f"{actual_end.date()}, missing both ends of the requested range"
        )
    elif missing_before:
        detail = (
            f"partial: begins {actual_start.date()}, but requested start is "
            f"{start_date}"
        )
    elif missing_after:
        detail = (
            f"partial: ends {actual_end.date()}, but requested end is "
            f"{end_date}"
        )
    else:
        detail = (
            "range endpoints are present, but one or more hourly records or "
            "required values are missing"
        )

    return False, detail, existing


def write_dataframe_atomically(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    temporary_destination = destination.with_suffix(
        destination.suffix + ".part"
    )
    dataframe.to_csv(temporary_destination, index=False)
    os.replace(temporary_destination, destination)


def update_city(
    session: requests.Session,
    *,
    city_name: str,
    start_date: date,
    end_date: date,
    chunk_days: int,
    model_name: str,
    connect_timeout: float,
    read_timeout: float,
    pause_seconds: float,
    force: bool,
) -> tuple[Path, bool]:
    """
    Update one city while reusing all verified existing hours.

    Returns:
        (destination_path, raw_file_changed)
    """
    latitude, longitude = get_coordinates(city_name)
    destination = RAW_DIR / f"{city_name}.csv"

    existing_source: pd.DataFrame | None = None

    if not force:
        is_complete, verification_message, existing_source = (
            verify_existing_city_file(
                destination,
                start_date=start_date,
                end_date=end_date,
            )
        )

        print(
            f"\n[{city_name.title()}] Existing-data verification: "
            f"{verification_message}"
        )

        if is_complete:
            requested_data = extract_valid_existing_range(
                existing_source,
                start_date=start_date,
                end_date=end_date,
                source_name=str(destination),
            )
            assert requested_data is not None

            # If the file contains dates outside the requested interval, trim it
            # once so every raw city file has the same exact date boundaries.
            if existing_source is not None and (
                len(existing_source) != len(requested_data)
            ):
                write_dataframe_atomically(
                    requested_data,
                    destination,
                )
                print(
                    f"[{city_name.title()}] Existing file was valid and "
                    "was trimmed to the exact requested range."
                )
                return destination, True

            print(
                f"[{city_name.title()}] All requested hours already exist. "
                "No download is needed."
            )
            return destination, False

    print("\n" + "=" * 68)
    print(f"Updating {city_name.title()}")
    print("=" * 68)
    print(f"Coordinates : {latitude}, {longitude}")
    print(f"Date range  : {start_date} through {end_date}")
    print(f"Model       : {model_name}")
    print(f"Chunk size  : {chunk_days} days")

    frames: list[pd.DataFrame] = []
    reused_existing_chunks = 0
    downloaded_or_cached_chunks = 0

    for chunk_start, chunk_end in iter_date_chunks(
        start_date,
        end_date,
        chunk_days,
    ):
        existing_chunk = None

        if not force:
            existing_chunk = extract_valid_existing_range(
                existing_source,
                start_date=chunk_start,
                end_date=chunk_end,
                source_name=(
                    f"existing {city_name.title()} data "
                    f"{chunk_start} through {chunk_end}"
                ),
            )

        if existing_chunk is not None:
            print(
                f"  Reused existing raw data: {chunk_start} through "
                f"{chunk_end}"
            )
            frame = existing_chunk
            reused_existing_chunks += 1
        else:
            frame = get_or_download_chunk(
                session,
                city_name=city_name,
                latitude=latitude,
                longitude=longitude,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                model_name=model_name,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                pause_seconds=pause_seconds,
                force=force,
            )
            downloaded_or_cached_chunks += 1

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = validate_hourly_dataframe(
        combined,
        start_date=start_date,
        end_date=end_date,
        source_name=f"combined {city_name.title()} dataset",
    )

    write_dataframe_atomically(
        combined,
        destination,
    )

    print(
        f"[{city_name.title()}] Saved {len(combined):,} verified "
        f"hourly rows to {destination}"
    )
    print(
        f"[{city_name.title()}] Chunks reused from existing file: "
        f"{reused_existing_chunks}; chunks loaded from cache/API: "
        f"{downloaded_or_cached_chunks}"
    )
    return destination, True


def clear_generated_outputs() -> None:
    """
    Clear processed and split outputs while preserving raw files and cache.
    """
    for directory in (PROCESSED_DIR, SPLIT_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    print("\nCleared previous processed and split outputs.")


def validate_split_outputs() -> None:
    expected_paths = [
        SPLIT_DIR / "train.csv",
        SPLIT_DIR / "dev.csv",
        SPLIT_DIR / "test.csv",
    ]
    missing = [
        str(path)
        for path in expected_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Preprocessing/splitting completed without creating: "
            f"{missing}"
        )

    print("\nGenerated split files:")
    for path in expected_paths:
        dataframe = pd.read_csv(path)
        print(f" - {path.name}: {len(dataframe):,} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reliably download, validate, preprocess, and split historical "
            "weather data for every city in CITY_COORDINATES."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        default=date.fromisoformat(DEFAULT_START_DATE),
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        default=date.fromisoformat(DEFAULT_END_DATE),
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
        help=(
            "Maximum number of days requested per API call. "
            f"Default: {DEFAULT_CHUNK_DAYS}"
        ),
    )
    parser.add_argument(
        "--model",
        choices=("era5", "era5_land", "best_match"),
        default="era5",
        help=(
            "Historical source. ERA5 is recommended for consistent "
            "2015-2026 model training. Default: era5"
        ),
    )
    parser.add_argument(
        "--http-retries",
        type=int,
        default=6,
        help="Automatic HTTP retries per chunk. Default: 6",
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=2.0,
        help="Exponential retry backoff factor. Default: 2.0",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=20.0,
        help="Connection timeout in seconds. Default: 20",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=180.0,
        help="Response read timeout in seconds. Default: 180",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help=(
            "Seconds to pause after each newly downloaded chunk. "
            "Default: 1"
        ),
    )
    # Kept for compatibility with commands written for the prior version.
    # Complete-file verification is now automatic, so this option is harmless.
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Check every existing raw city file against the requested date "
            "range, print the results, and exit without downloading, "
            "preprocessing, or splitting."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore valid cached chunks and download them again.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Update raw city files without preprocessing or splitting.",
    )
    parser.add_argument(
        "--clean-generated",
        action="store_true",
        help=(
            "Clear data/processed and data/splits before rebuilding. "
            "Raw files and cached chunks are preserved."
        ),
    )

    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date.")
    if args.chunk_days < 1:
        parser.error("--chunk-days must be at least 1.")
    if args.http_retries < 0:
        parser.error("--http-retries cannot be negative.")
    if args.backoff_factor < 0:
        parser.error("--backoff-factor cannot be negative.")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be positive.")
    if args.read_timeout <= 0:
        parser.error("--read-timeout must be positive.")
    if args.pause < 0:
        parser.error("--pause cannot be negative.")

    return args


def main() -> None:
    os.chdir(PROJECT_ROOT)
    create_directories()
    args = parse_args()

    city_names = list(CITY_COORDINATES)
    if not city_names:
        raise ValueError("CITY_COORDINATES is empty.")

    print("\n" + "=" * 68)
    print("Resilient Historical Weather Dataset Update")
    print("=" * 68)
    print(f"Cities     : {len(city_names)}")
    print(f"Date range : {args.start_date} through {args.end_date}")
    print(f"Model      : {args.model}")
    print(f"Chunk days : {args.chunk_days}")
    print("Existing-file verification and reuse: automatic")
    print("=" * 68)

    if args.verify_only:
        complete_count = 0
        incomplete_count = 0

        print("\nExisting Raw Dataset Verification")
        print("=" * 68)

        for city_name in city_names:
            city_path = RAW_DIR / f"{city_name}.csv"
            is_complete, message, _ = verify_existing_city_file(
                city_path,
                start_date=args.start_date,
                end_date=args.end_date,
            )

            status = "COMPLETE" if is_complete else "INCOMPLETE"
            print(
                f"{city_name.title():24} {status:10} {message}"
            )

            if is_complete:
                complete_count += 1
            else:
                incomplete_count += 1

        print("=" * 68)
        print(f"Complete   : {complete_count}")
        print(f"Incomplete : {incomplete_count}")
        return

    session = build_session(
        retries=args.http_retries,
        backoff_factor=args.backoff_factor,
    )

    successes: list[str] = []
    failures: dict[str, str] = {}

    try:
        for city_name in city_names:
            try:
                _, raw_file_changed = update_city(
                    session,
                    city_name=city_name,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    chunk_days=args.chunk_days,
                    model_name=args.model,
                    connect_timeout=args.connect_timeout,
                    read_timeout=args.read_timeout,
                    pause_seconds=args.pause,
                    force=args.force,
                )
                successes.append(city_name)
            except Exception as error:
                failures[city_name] = (
                    f"{type(error).__name__}: {error}"
                )
                print(
                    f"\n[{city_name.title()}] FAILED\n"
                    f"{type(error).__name__}: {error}"
                )
    finally:
        session.close()

    print("\n" + "=" * 68)
    print("Download Summary")
    print("=" * 68)
    print(f"Successful : {len(successes)}")
    print(f"Failed     : {len(failures)}")

    if failures:
        for city_name, error_message in failures.items():
            print(f" - {city_name.title()}: {error_message}")

        raise RuntimeError(
            "At least one city failed. Successful chunks are cached. "
            "Run the same command again to resume only the missing or "
            "invalid chunks. Preprocessing was not started."
        )

    if args.download_only:
        print("\nDownload-only update completed successfully.")
        return

    if args.clean_generated:
        clear_generated_outputs()

    print("\nPreprocessing all verified city datasets...\n")
    preprocess()

    print("\nChronologically splitting the processed data...\n")
    split_dataset()

    validate_split_outputs()

    print("\n" + "=" * 68)
    print("All Datasets Updated Successfully")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDataset update cancelled by user.")
        raise SystemExit(130)
    except Exception as error:
        print(f"\nDataset update failed: {error}")
        raise