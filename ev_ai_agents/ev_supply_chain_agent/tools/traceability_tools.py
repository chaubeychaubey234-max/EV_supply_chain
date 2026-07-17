import os
import pandas as pd
from langchain.tools import tool
from . import ToolError


# ---------------------------------------------------------------------------
# Load actual datasets
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASETS_DIR = os.path.join(_BASE_DIR, "..", "..", "..", "datasets")

_supply_chain_df = pd.read_csv(os.path.join(_DATASETS_DIR, "ev_supply_chain.csv"))


def _build_batch_db() -> dict:
    """Build batch traceability data from ev_supply_chain.csv."""
    db = {}
    for batch_id, group in _supply_chain_df.groupby("batch_id"):
        row = group.iloc[0]
        db[batch_id] = {
            "material": row["material"],
            "supplier_id": row["supplier_id"],
            "supplier_name": row["supplier_name"],
            "origin_country": row["country"],
            "battery_type": row["battery_type"],
            "cell_ids": group["cell_id"].unique().tolist(),
        }
    return db


def _build_cell_to_pack() -> dict:
    """Build cell-to-pack mapping from ev_supply_chain.csv."""
    return dict(zip(_supply_chain_df["cell_id"], _supply_chain_df["pack_id"]))


def _build_pack_to_vehicle() -> dict:
    """Build pack-to-vehicle mapping from ev_supply_chain.csv."""
    db = {}
    for pack_id, group in _supply_chain_df.groupby("pack_id"):
        row = group.iloc[0]
        db[pack_id] = {
            "vehicle_id": row["vehicle_id"],
        }
    return db


_BATCH_DB = _build_batch_db()
_CELL_TO_PACK = _build_cell_to_pack()
_PACK_TO_VEHICLE = _build_pack_to_vehicle()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def trace_material_batch(batch_id: str) -> dict:
    """Trace a raw material batch through the full EV supply chain.

    Given a batch ID (e.g. 'BAT-2024-001'), returns the complete trace
    path from raw material extraction through cell production, pack
    assembly, and vehicle integration.

    The trace path follows: Supplier → Cell → Pack → Vehicle.

    Use this to audit material provenance, verify conflict-free sourcing,
    or investigate quality issues back to their origin.
    """
    try:
        if not batch_id or not isinstance(batch_id, str):
            raise ToolError("batch_id must be a non-empty string")

        batch = _BATCH_DB.get(batch_id.upper())
        if batch is None:
            raise ToolError(
                f"Batch '{batch_id}' not found. "
                f"Valid IDs: {', '.join(_BATCH_DB.keys())}"
            )

        # Build the full trace path for each cell in the batch
        trace_paths = []
        for cell_id in batch["cell_ids"]:
            pack_id = _CELL_TO_PACK.get(cell_id)
            vehicle_info = _PACK_TO_VEHICLE.get(pack_id, {}) if pack_id else {}
            trace_paths.append({
                "cell_id": cell_id,
                "pack_id": pack_id or "UNASSIGNED",
                "vehicle_id": vehicle_info.get("vehicle_id", "UNASSIGNED"),
            })

        return {
            "batch_id": batch_id.upper(),
            "material": batch["material"],
            "supplier_id": batch["supplier_id"],
            "supplier_name": batch["supplier_name"],
            "origin_country": batch["origin_country"],
            "battery_type": batch["battery_type"],
            "trace_paths": trace_paths,
            "traceability_status": "Complete",
            "total_cells": len(batch["cell_ids"]),
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Batch tracing failed: {exc}") from exc


@tool
def map_cell_to_pack(cell_id: str) -> dict:
    """Map a battery cell to its parent battery pack.

    Given a cell ID (e.g. 'CELL-A100'), returns the pack it was assembled
    into along with other cells in the same pack and the destination vehicle.

    Use this when investigating a specific cell's location in the pack
    hierarchy or tracing a defective cell to its pack and vehicle.
    """
    try:
        if not cell_id or not isinstance(cell_id, str):
            raise ToolError("cell_id must be a non-empty string")

        pack_id = _CELL_TO_PACK.get(cell_id.upper())
        if pack_id is None:
            raise ToolError(
                f"Cell '{cell_id}' not found. "
                f"Valid IDs: {', '.join(_CELL_TO_PACK.keys())}"
            )

        # Find sibling cells in the same pack
        sibling_cells = [
            cid for cid, pid in _CELL_TO_PACK.items() if pid == pack_id
        ]

        # Find the source batch for this cell
        source_batch = None
        for bid, bdata in _BATCH_DB.items():
            if cell_id.upper() in bdata["cell_ids"]:
                source_batch = bid
                break

        vehicle_info = _PACK_TO_VEHICLE.get(pack_id, {})

        return {
            "cell_id": cell_id.upper(),
            "pack_id": pack_id,
            "source_batch": source_batch or "UNKNOWN",
            "sibling_cells": sibling_cells,
            "cells_in_pack": len(sibling_cells),
            "vehicle_id": vehicle_info.get("vehicle_id", "UNASSIGNED"),
            "mapping_status": "Verified",
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Cell-to-pack mapping failed: {exc}") from exc


@tool
def map_pack_to_vehicle(pack_id: str) -> dict:
    """Map a battery pack to the vehicle it was installed in.

    Given a pack ID (e.g. 'PACK-001'), returns the vehicle details
    including VIN. Also returns all cells contained in the pack for
    downward traceability.

    Use this when tracing a pack-level issue to the affected vehicle,
    or for recall scope analysis.
    """
    try:
        if not pack_id or not isinstance(pack_id, str):
            raise ToolError("pack_id must be a non-empty string")

        vehicle_info = _PACK_TO_VEHICLE.get(pack_id.upper())
        if vehicle_info is None:
            raise ToolError(
                f"Pack '{pack_id}' not found. "
                f"Valid IDs: {', '.join(_PACK_TO_VEHICLE.keys())}"
            )

        # Find all cells in this pack
        cells_in_pack = [
            cid for cid, pid in _CELL_TO_PACK.items() if pid == pack_id.upper()
        ]

        # Trace cells back to batches
        batch_sources = set()
        for cell_id in cells_in_pack:
            for bid, bdata in _BATCH_DB.items():
                if cell_id in bdata["cell_ids"]:
                    batch_sources.add(bid)

        return {
            "pack_id": pack_id.upper(),
            "vehicle_id": vehicle_info["vehicle_id"],
            "cells_in_pack": cells_in_pack,
            "total_cells": len(cells_in_pack),
            "source_batches": list(batch_sources),
            "mapping_status": "Verified",
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Pack-to-vehicle mapping failed: {exc}") from exc


@tool
def verify_traceability_completeness(batch_data: dict) -> dict:
    """Verify end-to-end traceability completeness for a material batch.

    Expects a dict with keys:
      - batch_id (str): The batch to verify

    Checks that every link in the chain (Supplier → Cell → Pack → Vehicle)
    is present and accounted for. Identifies missing links that break the
    traceability chain.

    Use this for regulatory compliance verification (EU Battery Passport),
    audit preparation, or traceability gap analysis.
    """
    try:
        if not isinstance(batch_data, dict):
            raise ToolError("batch_data must be a dictionary")

        batch_id = batch_data.get("batch_id")
        if not batch_id:
            raise ToolError("batch_data must contain a 'batch_id' key")

        batch = _BATCH_DB.get(batch_id.upper())
        if batch is None:
            raise ToolError(f"Batch '{batch_id}' not found in traceability system")

        missing_links = []
        verified_links = []
        total_checks = 0

        # Check each cell in the batch
        for cell_id in batch["cell_ids"]:
            total_checks += 1

            # Cell → Pack link
            pack_id = _CELL_TO_PACK.get(cell_id)
            if pack_id:
                verified_links.append(f"{cell_id} → {pack_id}")
            else:
                missing_links.append({
                    "link_type": "cell_to_pack",
                    "cell_id": cell_id,
                    "issue": "Cell not assigned to any pack",
                })
                continue

            # Pack → Vehicle link
            vehicle_info = _PACK_TO_VEHICLE.get(pack_id)
            if vehicle_info:
                verified_links.append(
                    f"{pack_id} → {vehicle_info['vehicle_id']}"
                )
            else:
                missing_links.append({
                    "link_type": "pack_to_vehicle",
                    "pack_id": pack_id,
                    "issue": "Pack not assigned to any vehicle",
                })

        completeness_pct = round(
            (len(verified_links) / max(total_checks * 2, 1)) * 100, 1
        )

        if completeness_pct == 100:
            status = "Complete"
        elif completeness_pct >= 75:
            status = "Partial – Minor Gaps"
        elif completeness_pct >= 50:
            status = "Partial – Significant Gaps"
        else:
            status = "Incomplete – Critical Gaps"

        return {
            "batch_id": batch_id.upper(),
            "material": batch["material"],
            "supplier_id": batch["supplier_id"],
            "trace_path": (
                f"{batch['supplier_name']} ({batch['origin_country']}) → "
                f"Cells ({len(batch['cell_ids'])}) → "
                f"Packs → Vehicles"
            ),
            "traceability_status": status,
            "completeness_percent": completeness_pct,
            "verified_links": verified_links,
            "missing_links": missing_links,
            "total_checks": total_checks * 2,
            "eu_battery_passport_ready": completeness_pct == 100,
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            f"Traceability completeness verification failed: {exc}"
        ) from exc
