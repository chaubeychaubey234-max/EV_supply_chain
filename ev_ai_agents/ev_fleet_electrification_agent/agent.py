"""
agent.py — Orchestration layer for the Fleet Electrification Readiness Agent.

Responsibilities
----------------
1. Accept a natural-language user query.
2. Route intent to the correct subset of existing tools.
3. Execute each selected tool and collect outputs.
4. Pass collected outputs to ``generate_llm_response()`` (Groq stub).
5. Return a single structured response dictionary.

LLM integration
---------------
All future LLM interaction is isolated to ``generate_llm_response()``.
When Groq is ready, **only that one function** needs to be updated.
No other code in this file needs to change.

Tools used
----------
The agent imports and calls the six existing tools directly from:
    features/fleet_electrification_readiness/tools/

No new tools are created. No business logic lives here.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Any

# ─── Path bootstrap ───────────────────────────────────────────────────────────
# Resolve project root so imports work regardless of the CWD used to run the agent.
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))                   # …/ev_fleet_electrification_agent/
_FEAT_DIR  = os.path.dirname(_AGENT_DIR)                                   # …/fleet_electrification_readiness/
_FEATS_DIR = os.path.dirname(_FEAT_DIR)                                    # …/features/
_ROOT_DIR  = os.path.dirname(_FEATS_DIR)                                   # project root
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# ─── Existing tool imports (no new tools created) ─────────────────────────────
from ev_ai_agents.ev_fleet_electrification_agent.tools.fleet_data_tool      import fetch_vehicle_data
from ev_ai_agents.ev_fleet_electrification_agent.tools.readiness_score_tool  import calculate_readiness_score
from ev_ai_agents.ev_fleet_electrification_agent.tools.ev_matching_tool      import recommend_ev_replacement
from ev_ai_agents.ev_fleet_electrification_agent.tools.roi_tool              import calculate_roi
from ev_ai_agents.ev_fleet_electrification_agent.tools.route_analysis_tool   import analyze_vehicle_route
from ev_ai_agents.ev_fleet_electrification_agent.tools.procurement_tool      import recommend_procurement

from .state import AgentState

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# Maps a stable string key → the LangChain @tool callable.
# Agent orchestration references these keys only — never the callables directly.
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Any] = {
    "fleet_data_tool":       fetch_vehicle_data,
    "readiness_score_tool":  calculate_readiness_score,
    "ev_matching_tool":      recommend_ev_replacement,
    "roi_tool":              calculate_roi,
    "route_analysis_tool":   analyze_vehicle_route,
    "procurement_tool":      recommend_procurement,
}

# ─────────────────────────────────────────────────────────────────────────────
# Intent → tool mapping
# Each intent group defines which tools should run for matching queries.
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_MAP: list[dict[str, Any]] = [
    # Full electrification / transition evaluation
    {
        "keywords": {
            "transition", "electrif", "ev adoption", "convert", "replace",
            "procurement", "fleet ev", "go electric", "switch to ev",
        },
        "tools": [
            "fleet_data_tool",
            "readiness_score_tool",
            "ev_matching_tool",
            "roi_tool",
            "procurement_tool",
        ],
    },
    # Charging infrastructure / route-based queries
    {
        "keywords": {
            "charg", "infrastructure", "station", "charger", "charging point",
            "route", "range", "overnight", "window",
        },
        "tools": [
            "fleet_data_tool",
            "route_analysis_tool",
        ],
    },
    # Financial / ROI / savings
    {
        "keywords": {
            "roi", "return on investment", "saving", "cost", "payback",
            "financial", "annual saving", "profit", "break even",
        },
        "tools": [
            "roi_tool",
        ],
    },
    # Carbon / emissions
    {
        "keywords": {
            "carbon", "emission", "co2", "co₂", "green", "environment",
            "sustainability", "footprint", "ghg",
        },
        "tools": [
            "procurement_tool",
        ],
    },
    # EV model / recommendation
    {
        "keywords": {
            "recommend", "suitable ev", "which ev", "best ev", "ev model",
            "replacement", "match", "spec",
        },
        "tools": [
            "ev_matching_tool",
        ],
    },
    # Readiness scoring
    {
        "keywords": {
            "readiness", "ready", "score", "assess", "feasib", "viable",
        },
        "tools": [
            "readiness_score_tool",
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM placeholder
# ─────────────────────────────────────────────────────────────────────────────

def generate_llm_response(
    user_query: str,
    tool_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Generate a natural-language summary from tool outputs via an LLM.

    **Current status:** Groq integration is pending.
    This function returns a structured placeholder response that includes
    all raw tool outputs so the rest of the pipeline remains fully functional.

    When Groq (or any other LLM) is integrated, replace **only** the body
    of this function. No other code in agent.py needs to change.

    Args:
        user_query:   The original user question.
        tool_outputs: Collected outputs from all executed tools.

    Returns:
        A structured response dict containing at minimum:
        ``status``, ``summary``, ``tool_outputs``, ``recommendations``, ``next_steps``.
    """
    # ── Groq integration point ──────────────────────────────────────────────
    # TODO: Replace this block with a Groq API call, e.g.:
    #
    #   from groq import Groq
    #   client = Groq(api_key=os.environ["GROQ_API_KEY"])
    #   completion = client.chat.completions.create(
    #       model="llama-3.3-70b-versatile",
    #       messages=[
    #           {"role": "system", "content": SYSTEM_PROMPT},
    #           {"role": "user",   "content": _build_prompt(user_query, tool_outputs)},
    #       ],
    #   )
    #   return _parse_groq_response(completion)
    # ───────────────────────────────────────────────────────────────────────

    log.info("generate_llm_response: LLM not yet integrated — returning placeholder.")

    # Build lightweight placeholder recommendations from raw tool data
    recommendations: list[str] = []
    next_steps: list[str] = []

    if "readiness_score_tool" in tool_outputs:
        rs = tool_outputs["readiness_score_tool"]
        score = rs.get("readiness_score", "N/A")
        cls   = rs.get("classification", "")
        recommendations.append(
            f"Readiness score is {score}/100 ({cls}). "
            "Review the classification rationale before committing to procurement."
        )
        next_steps.append("Commission a formal route-optimisation study.")

    if "roi_tool" in tool_outputs:
        roi = tool_outputs["roi_tool"]
        savings   = roi.get("total_annual_savings_usd", 0)
        payback   = roi.get("estimated_payback_years", "N/A")
        roi_pct   = roi.get("roi_percent_over_10_years", "N/A")
        recommendations.append(
            f"Estimated annual savings: USD {savings:,.0f}. "
            f"Payback period: {payback} years. "
            f"10-year ROI: {roi_pct}%."
        )
        next_steps.append("Engage fleet finance team to validate ROI assumptions.")

    if "ev_matching_tool" in tool_outputs:
        ev = tool_outputs["ev_matching_tool"]
        recommendations.append(
            f"Recommended EV: {ev.get('recommended_ev', 'N/A')} "
            f"(compatibility score: {ev.get('compatibility_score', 'N/A')})."
        )
        next_steps.append("Request EV supplier demo and total-cost-of-ownership comparison.")

    if "procurement_tool" in tool_outputs:
        proc = tool_outputs["procurement_tool"]
        recommendations.append(
            f"Procurement recommendation: {proc.get('recommendation', 'N/A')} "
            f"(window: {proc.get('recommended_purchase_window', 'N/A')})."
        )
        next_steps.append("Initiate procurement process per recommended purchase window.")

    if "route_analysis_tool" in tool_outputs:
        route = tool_outputs["route_analysis_tool"]
        window = route.get("available_charging_window_hours", "N/A")
        recommendations.append(
            f"Charging window available: {window} hours. "
            "Verify depot charging infrastructure capacity."
        )
        next_steps.append("Survey depot for Level 2 / DC fast charger installation feasibility.")

    if not recommendations:
        recommendations = ["Review full tool outputs for detailed insights."]
    if not next_steps:
        next_steps = ["Consult fleet operations team for next steps."]

    return {
        "status":          "LLM integration pending",
        "summary":         (
            f"Query '{user_query}' processed. "
            f"{len(tool_outputs)} tool(s) executed successfully. "
            "Groq LLM narrative will replace this placeholder on integration."
        ),
        "tool_outputs":    tool_outputs,
        "recommendations": recommendations,
        "next_steps":      next_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Intent routing
# ─────────────────────────────────────────────────────────────────────────────

def _select_tools(query: str) -> list[str]:
    """Map a user query to the required tool keys via keyword matching.

    Supports multi-intent queries (e.g. "electrification and savings" triggers
    both the transition group and the ROI group).

    Args:
        query: Raw user query string.

    Returns:
        Ordered, deduplicated list of tool-registry keys to execute.
        Falls back to ``["fleet_data_tool", "readiness_score_tool"]`` when
        no keywords match.
    """
    query_lower = query.lower()
    matched: list[str] = []

    for intent in _INTENT_MAP:
        if any(kw in query_lower for kw in intent["keywords"]):
            for tool_key in intent["tools"]:
                if tool_key not in matched:
                    matched.append(tool_key)

    if not matched:
        log.warning("_select_tools: no intent matched — falling back to readiness check.")
        matched = ["fleet_data_tool", "readiness_score_tool"]

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────────────────────────

def _execute_tool(tool_key: str, vehicle_id: str) -> tuple[bool, Any]:
    """Invoke a single tool from the registry and return its output.

    Args:
        tool_key:   Key in ``_TOOL_REGISTRY``.
        vehicle_id: Vehicle identifier forwarded to the tool.

    Returns:
        ``(success, output)`` — output is the tool result dict on success,
        or an error dict on failure.
    """
    callable_tool = _TOOL_REGISTRY.get(tool_key)
    if callable_tool is None:
        err = f"Tool '{tool_key}' is not registered in _TOOL_REGISTRY."
        log.error(err)
        return False, {"error": err}

    try:
        log.info("  Executing tool: %s (vehicle_id=%s)", tool_key, vehicle_id)
        result = callable_tool.invoke({"vehicle_id": vehicle_id})
        return True, result
    except Exception as exc:
        log.error("  Tool '%s' failed: %s", tool_key, exc)
        return False, {"error": str(exc), "tool": tool_key}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(user_query: str, vehicle_id: str = "VEH-002") -> dict[str, Any]:
    """Execute the Fleet Electrification Readiness Agent.

    Orchestration flow
    ------------------
    1. Validate inputs.
    2. Select tools via intent routing.
    3. Execute each selected tool; collect outputs.
    4. Pass outputs to ``generate_llm_response()``.
    5. Populate :class:`AgentState` and return the final response dict.

    Args:
        user_query: Natural-language question from the user.
        vehicle_id: Fleet vehicle identifier to evaluate (default ``"VEH-002"``).

    Returns:
        Structured response dictionary::

            {
                "status":          "success" | "partial" | "error",
                "selected_tools":  [...],
                "tool_outputs":    {...},
                "summary":         "...",
                "recommendations": [...],
                "next_steps":      [...],
            }
    """
    log.info("=" * 60)
    log.info("Fleet Electrification Agent — START")
    log.info("  Query     : %s", user_query)
    log.info("  Vehicle   : %s", vehicle_id)
    log.info("=" * 60)

    # ── 1. Input validation ──────────────────────────────────────────────────
    state = AgentState()

    if not isinstance(user_query, str) or not user_query.strip():
        log.error("Invalid user_query: must be a non-empty string.")
        state.mark_error()
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         "Invalid query. Please provide a non-empty question.",
            "recommendations": [],
            "next_steps":      [],
        }

    if not isinstance(vehicle_id, str) or not vehicle_id.strip():
        log.error("Invalid vehicle_id: must be a non-empty string.")
        state.mark_error()
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         "Invalid vehicle_id. Please provide a valid fleet identifier.",
            "recommendations": [],
            "next_steps":      [],
        }

    state.user_query = user_query.strip()
    state.add_history("user", state.user_query)

    # ── 2. Intent routing ─────────────────────────────────────────────────────
    selected = _select_tools(state.user_query)
    state.selected_tools = selected
    log.info("Selected tools: %s", selected)

    # ── 3. Tool execution ─────────────────────────────────────────────────────
    failed_tools: list[str] = []

    for tool_key in selected:
        ok, output = _execute_tool(tool_key, vehicle_id.strip())
        state.add_tool_output(tool_key, output)
        if not ok:
            failed_tools.append(tool_key)

    # ── 4. Determine execution status ─────────────────────────────────────────
    if not failed_tools:
        state.mark_success()
    elif len(failed_tools) < len(selected):
        state.mark_partial()
        log.warning("Partial execution: failed tools — %s", failed_tools)
    else:
        state.mark_error()
        log.error("All tools failed: %s", failed_tools)

    # ── 5. Generate response (LLM stub) ───────────────────────────────────────
    llm_output = generate_llm_response(state.user_query, state.tool_outputs)

    # ── 6. Assemble final response ────────────────────────────────────────────
    state.final_response = {
        "status":          state.execution_status,
        "selected_tools":  state.selected_tools,
        "tool_outputs":    state.tool_outputs,
        "summary":         llm_output.get("summary", ""),
        "recommendations": llm_output.get("recommendations", []),
        "next_steps":      llm_output.get("next_steps", []),
    }

    state.add_history("assistant", state.final_response["summary"])

    log.info("Agent completed with status: %s", state.execution_status)
    return state.final_response


# ─────────────────────────────────────────────────────────────────────────────
# Local test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    TEST_QUERY      = "Evaluate my delivery fleet for electrification and estimate annual savings."
    TEST_VEHICLE_ID = "VEH-002"

    print("=" * 60)
    print("Fleet Electrification Agent — Test Run")
    print("=" * 60)
    print(f"Query     : {TEST_QUERY}")
    print(f"Vehicle ID: {TEST_VEHICLE_ID}")
    print()

    response = run_agent(TEST_QUERY, vehicle_id=TEST_VEHICLE_ID)

    print()
    print("=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)
    print(f"Status         : {response['status']}")
    print(f"Selected Tools : {response['selected_tools']}")
    print()
    print(f"Summary:")
    print(f"  {response['summary']}")
    print()

    print("Recommendations:")
    for i, rec in enumerate(response["recommendations"], 1):
        print(f"  {i}. {rec}")
    print()

    print("Next Steps:")
    for i, ns in enumerate(response["next_steps"], 1):
        print(f"  {i}. {ns}")
    print()

    print("Tool Outputs (JSON):")
    print(json.dumps(response["tool_outputs"], indent=2, default=str))
