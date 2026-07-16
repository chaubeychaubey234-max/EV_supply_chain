import random
from langchain.tools import tool
from . import ToolError


# ---------------------------------------------------------------------------
# Mock traceability data – Cell → Pack → Vehicle lineage
# ---------------------------------------------------------------------------
_BATCH_DB = {
    "BAT-2024-001": {
        "material": "lithium",
        "supplier_id": "SUP-001",
        "supplier_name": "Ganfeng Lithium Co.",
        "origin_country": "China",
        "extraction_date": "2024-01-15",
        "purity_grade": "99.5%",
        "weight_kg": 500,
        "cell_ids": ["CELL-A100", "CELL-A101", "CELL-A102"],
    },
    "BAT-2024-002": {
        "material": "cobalt",
        "supplier_id": "SUP-003",
        "supplier_name": "Glencore Congo SARL",
        "origin_country": "Congo",
        "extraction_date": "2024-02-20",
        "purity_grade": "99.2%",
        "weight_kg": 120,
        "cell_ids": ["CELL-B200", "CELL-B201"],
    },
    "BAT-2024-003": {
        "material": "nickel",
        "supplier_id": "SUP-006",
        "supplier_name": "Norilsk Nickel",
        "origin_country": "Russia",
        "extraction_date": "2024-03-10",
        "purity_grade": "99.8%",
        "weight_kg": 350,
        "cell_ids": ["CELL-C300", "CELL-C301", "CELL-C302", "CELL-C303"],
    },
    "BAT-2024-004": {
        "material": "LFP",
        "supplier_id": "SUP-005",
        "supplier_name": "CATL Battery Co.",
        "origin_country": "China",
        "extraction_date": "2024-04-05",
        "purity_grade": "99.6%",
        "weight_kg": 800,
        "cell_ids": ["CELL-D400", "CELL-D401", "CELL-D402"],
    },
}

_CELL_TO_PACK = {
    "CELL-A100": "PACK-001",
    "CELL-A101": "PACK-001",
    "CELL-A102": "PACK-002",
    "CELL-B200": "PACK-002",
    "CELL-B201": "PACK-003",
    "CELL-C300": "PACK-003",
    "CELL-C301": "PACK-004",
    "CELL-C302": "PACK-004",
    "CELL-C303": "PACK-005",
    "CELL-D400": "PACK-005",
    "CELL-D401": "PACK-006",
    "CELL-D402": "PACK-006",
}

_PACK_TO_VEHICLE = {
    "PACK-001": {
        "vehicle_id": "VIN-EV-20240001",
        "model": "Model E Pro",
        "manufacturer": "NovaDrive Motors",
        "assembly_plant": "Shanghai, China",
        "assembly_date": "2024-06-15",
    },
    "PACK-002": {
        "vehicle_id": "VIN-EV-20240002",
        "model": "Model E Pro",
        "manufacturer": "NovaDrive Motors",
        "assembly_plant": "Shanghai, China",
        "assembly_date": "2024-06-18",
    },
    "PACK-003": {
        "vehicle_id": "VIN-EV-20240003",
        "model": "Volt X",
        "manufacturer": "ElectraAuto",
        "assembly_plant": "Munich, Germany",
        "assembly_date": "2024-07-01",
    },
    "PACK-004": {
        "vehicle_id": "VIN-EV-20240004",
        "model": "Volt X",
        "manufacturer": "ElectraAuto",
        "assembly_plant": "Munich, Germany",
        "assembly_date": "2024-07-05",
    },
    "PACK-005": {
        "vehicle_id": "VIN-EV-20240005",
        "model": "ZeroKm S",
        "manufacturer": "ZeroKm Mobility",
        "assembly_plant": "Fremont, USA",
        "assembly_date": "2024-07-20",
    },
    "PACK-006": {
        "vehicle_id": "VIN-EV-20240006",
        "model": "ZeroKm S",
        "manufacturer": "ZeroKm Mobility",
        "assembly_plant": "Fremont, USA",
        "assembly_date": "2024-07-22",
    },
}


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
                "vehicle_model": vehicle_info.get("model", "N/A"),
            })

        return {
            "batch_id": batch_id.upper(),
            "material": batch["material"],
            "supplier_id": batch["supplier_id"],
            "supplier_name": batch["supplier_name"],
            "origin_country": batch["origin_country"],
            "extraction_date": batch["extraction_date"],
            "purity_grade": batch["purity_grade"],
            "weight_kg": batch["weight_kg"],
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
            "vehicle_model": vehicle_info.get("model", "N/A"),
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
    including VIN, model, manufacturer, assembly plant, and assembly date.
    Also returns all cells contained in the pack for downward traceability.

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
            "vehicle_model": vehicle_info["model"],
            "manufacturer": vehicle_info["manufacturer"],
            "assembly_plant": vehicle_info["assembly_plant"],
            "assembly_date": vehicle_info["assembly_date"],
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
      - or any output from trace_material_batch

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

        # If we only have the batch_id, look up the full trace
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
