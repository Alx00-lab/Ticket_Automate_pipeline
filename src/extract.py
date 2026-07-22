from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Final, FrozenSet, Tuple
import pandas as pd


# Looging Setup 
logger = logging.getLogger(__name__)

#  --- Pipeline Cotract & Constants ----

# 1. Structural Schema Specification (20 Columns, Strict Order)
EXPECTED_COLUMNS: Final[Tuple[str, ...]] = (
    "ticket_id",
    "created_at",
    "customer_id",
    "customer_segment",
    "channel",
    "product_area",
    "issue_type",
    "priority",
    "status",
    "sla_plan",
    "initial_message",
    "agent_first_reply",
    "resolution_summary",
    "resolution_time_hours",
    "reopened",
    "customer_sentiment",
    "csat_score",
    "has_attachment",
    "platform",
    "region",
)

# 2. Domain Categorical Expectations
VALID_STATUSES: Final[FrozenSet[str]] = frozenset(
    {"OPEN", "IN_PROGRESS", "ON_HOLD", "RESOLVED", "CLOSED_NO_ACTION"}
)

VALID_PRIORITIES: Final[FrozenSet[str]] = frozenset(
    {"LOW", "MEDIUM", "HIGH", "URGENT"}
)

# 3. Volume Lower Bound (Safety margin against truncated upstream exports)
MIN_EXPECTED_ROWS: Final[int] = 95_000

# 4. Temporal Bounds (Sanity check range for creation dates)
MIN_EXPECTED_DATE: Final[datetime] = datetime(2022, 1, 1, tzinfo=timezone.utc)
MAX_EXPECTED_DATE: Final[datetime] = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


# ---- CUSTOM EXCEPTIONS  ------
class ExtractionError(Exception):
    """Base exception for extraction phase failures."""
    pass

class SchemaMismatchError(ExtractionError):
    """Raised when columns do not strictly match expected contract."""
    pass

class TruncateDataError(ExtractionError):
    """Raised when row count falls below minimum expected safety threshold."""
    pass


# ----- Main extraction Funct. -----
def extract_raw_tickets(file_path: str | Path) -> pd.DataFrame:
    target_path = Path(file_path)
    if not target_path.exists():
        logger.critical(f"Extraction failed: File not found at {target_path}")
        raise FileNotFoundError(f"source file missing: {target_path}")
    
    logger.info(f"Starting extraction from: {target_path}")

    # ----------------------
    #    READ CSV DOC...
    # ----------------------
    df_raw = pd.read_csv(target_path, dtype=str)
    total_rows = len(df_raw)

    # ---------------------------------------
    # Validate Column Structure (Exact Match)
    # --------------------------------------- 
    actual_columns = tuple(df_raw.columns)
    if actual_columns != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(actual_columns)
        unexpected = set(actual_columns) - set(EXPECTED_COLUMNS)

        error_msg = (
            f"Schema validation failed!\n"
            f"Expected {len(EXPECTED_COLUMNS)} columns, got {len(actual_columns)}.\n"
            f"Missing columns: {missing}\n"
            f"Unexpected columns: {unexpected}\n"
            f"Exact order match: {actual_columns == EXPECTED_COLUMNS}"
        )
        logger.error(error_msg)
        raise SchemaMismatchError(error_msg)
    
    # ----------------------------------
    # 3. Validate Row count Safety Margin
    # ----------------------------------

    if total_rows < MIN_EXPECTED_ROWS:
        error_msg = (
            f"Data volume check failed! Found {total_rows:,} rows, "
            f"expected at least {MIN_EXPECTED_ROWS:,}. File may be truncated."
        )
        logger.error(error_msg)
        raise TruncateDataError(error_msg)
    
   
    # -----------------------------------------------------
    # 4. Parse Dates (Validation only, don't mutate raw DF)
    # ------------------------------------------------------
    parsed_dates = pd.to_datetime(df_raw["created_at"], errors="coerce", utc=True)
    unparseable_count = parsed_dates.isna().sum()

    if unparseable_count > 0:
        logger.warning(
            f"Date Quality Issue: Found {unparseable_count:,} rows with unparseable "
            f"'created_at' values out of {total_rows:,} total rows."
        )

    # --------------------------
    # 5. Confirm Temporal Bounds
    # --------------------------
    valid_dates = parsed_dates.dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min()
        max_date = valid_dates.max()

        if min_date < MIN_EXPECTED_DATE or max_date > MAX_EXPECTED_DATE:
            logger.warning(
                f"Date Bound Anomaly Detected!\n"
                f"Observed Range: [{min_date}] to [{max_date}]\n"
                f"Expected Range: [{MIN_EXPECTED_DATE}] to [{MAX_EXPECTED_DATE}]"
            )
        else:
            min_date, max_date = None, None
            logger.warning("No valid dates found in 'created_at' column to inspect bounds.")

    
    # -------------
    # 6. Categorical Anomaly Checking (Informational, non-Blocking)
    # -------------
    # Status Check
    actual_statuses = set(df_raw["status"].dropna().unique())
    unexpected_statuses = actual_statuses - VALID_STATUSES
    if unexpected_statuses:
        logger.warning(
            f"New/Unexpected Status values observed in data: {unexpected_statuses}. "
            f"Allowed known statuses: {set(VALID_STATUSES)}"
        )
    
    # Priority Check
    actual_priorities = set(df_raw["prority"].dropna().unique())
    unexpected_priorities = actual_priorities - VALID_PRIORITIES
    if unexpected_priorities:
        logger.warning(
            f"New/Unexpected Priority values observed in data: {unexpected_priorities}. "
            f"Allowed known priorities: {set(VALID_PRIORITIES)}"
        )

    # ------------------
    # 7. Summary Logging
    # ------------------
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY & SANITY CHECK COMPLETE")
    logger.info(f"Total Rows Extracted : {total_rows:,}")
    logger.info(f"Columns Verified     : {len(actual_columns)} (Matches Contract)")
    logger.info(f"Date Min Observed    : {min_date}")
    logger.info(f"Date Max Observed    : {max_date}")
    logger.info(f"Unparseable Dates    : {unparseable_count:,}")
    logger.info(f"Status Anomalies     : {unexpected_statuses if unexpected_statuses else 'None'}")
    logger.info(f"Priority Anomalies   : {unexpected_priorities if unexpected_priorities else 'None'}")
    logger.info("=" * 60)

    # Last: Return Pure Raw Data
    return df_raw
