from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agents.rl1_load_referral_context import load_referral_context
from agents.rl2_classify_urgency import classify_urgency
from agents.rl3_extract_clinical_history import extract_clinical_history
from agents.rl4_generate_referral_letter import generate_referral_letter
from agents.rl5_assemble_document import assemble_document
from agents.rl6_route_and_notify import route_and_notify
from shared.models import ReferralContext, UrgencyClassification, UrgencyLevel


class ReferralState(TypedDict):
    """
    Typed state schema for the referral letter pipeline.
    Each field is populated by the corresponding RL step and passed forward.
    """
    service_request_id: str
    context: Optional[ReferralContext]
    urgency: Optional[UrgencyClassification]
    clinical_history: Optional[str]
    letter_text: Optional[str]
    document: Optional[dict]
    document_reference_id: Optional[str]
    error: Optional[str]


def node_rl1(state: ReferralState) -> ReferralState:
    """RL-1: Load referral context from FHIR."""
    try:
        context = load_referral_context(state["service_request_id"])
        return {**state, "context": context}
    except Exception as e:
        return {**state, "error": f"RL-1 failed: {str(e)}"}


def node_rl2(state: ReferralState) -> ReferralState:
    """RL-2: Classify urgency -- output drives conditional edge."""
    if state.get("error"):
        return state
    try:
        urgency = classify_urgency(state["context"])
        return {**state, "urgency": urgency}
    except Exception as e:
        return {**state, "error": f"RL-2 failed: {str(e)}"}


def node_rl3(state: ReferralState) -> ReferralState:
    """RL-3: Extract clinical history scoped to referral reason."""
    if state.get("error"):
        return state
    try:
        history = extract_clinical_history(state["context"])
        return {**state, "clinical_history": history}
    except Exception as e:
        return {**state, "error": f"RL-3 failed: {str(e)}"}


def node_rl4(state: ReferralState) -> ReferralState:
    """RL-4: Generate LOINC 57133-1 referral letter."""
    if state.get("error"):
        return state
    try:
        letter_text = generate_referral_letter(
            state["context"],
            state["urgency"],
            state["clinical_history"],
        )
        return {**state, "letter_text": letter_text}
    except Exception as e:
        return {**state, "error": f"RL-4 failed: {str(e)}"}


def node_rl5(state: ReferralState) -> ReferralState:
    """RL-5: Assemble FHIR DocumentReference."""
    if state.get("error"):
        return state
    try:
        document = assemble_document(
            state["context"],
            state["urgency"],
            state["letter_text"],
        )
        return {**state, "document": document}
    except Exception as e:
        return {**state, "error": f"RL-5 failed: {str(e)}"}


def node_rl6(state: ReferralState) -> ReferralState:
    """RL-6: DLP inspect, FHIR write, Pub/Sub route, Firestore audit."""
    if state.get("error"):
        return state
    try:
        document_id = route_and_notify(
            state["document"],
            state["urgency"],
            state["context"],
        )
        return {**state, "document_reference_id": document_id}
    except Exception as e:
        return {**state, "error": f"RL-6 failed: {str(e)}"}


def route_by_urgency(state: ReferralState) -> str:
    """
    Conditional edge function at RL-2.
    Returns the next node name based on urgency classification.
    This is the key LangGraph vs ADK comparison point:
    routing is explicit and deterministic here, not LLM-driven at runtime.
    """
    if state.get("error"):
        return "end"
    urgency = state.get("urgency")
    if urgency and urgency.urgency == UrgencyLevel.EMERGENT:
        return "rl3_emergent"
    return "rl3"


def node_rl3_emergent(state: ReferralState) -> ReferralState:
    """
    RL-3 variant for EMERGENT referrals.
    Same extraction logic but logs the EMERGENT fast-track path.
    In a production system this node could trigger immediate alerts
    before letter generation completes.
    """
    if state.get("error"):
        return state
    try:
        history = extract_clinical_history(state["context"])
        return {**state, "clinical_history": history}
    except Exception as e:
        return {**state, "error": f"RL-3 (EMERGENT) failed: {str(e)}"}


def build_graph() -> StateGraph:
    """
    Build and compile the referral letter StateGraph.

    Graph structure:
        rl1 -> rl2 -> [conditional edge] -> rl3 or rl3_emergent
        rl3 -> rl4 -> rl5 -> rl6 -> END
        rl3_emergent -> rl4 -> rl5 -> rl6 -> END
    """
    graph = StateGraph(ReferralState)

    graph.add_node("rl1", node_rl1)
    graph.add_node("rl2", node_rl2)
    graph.add_node("rl3", node_rl3)
    graph.add_node("rl3_emergent", node_rl3_emergent)
    graph.add_node("rl4", node_rl4)
    graph.add_node("rl5", node_rl5)
    graph.add_node("rl6", node_rl6)

    graph.set_entry_point("rl1")
    graph.add_edge("rl1", "rl2")

    graph.add_conditional_edges(
        "rl2",
        route_by_urgency,
        {
            "rl3": "rl3",
            "rl3_emergent": "rl3_emergent",
            "end": END,
        },
    )

    graph.add_edge("rl3", "rl4")
    graph.add_edge("rl3_emergent", "rl4")
    graph.add_edge("rl4", "rl5")
    graph.add_edge("rl5", "rl6")
    graph.add_edge("rl6", END)

    return graph.compile()


referral_graph = build_graph()
