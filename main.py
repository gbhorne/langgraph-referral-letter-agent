import json
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from agents.graph import referral_graph


def run(service_request_id: str) -> dict:
    """
    Run the referral letter pipeline for a given ServiceRequest ID.
    LangSmith tracing is enabled automatically via LANGCHAIN_API_KEY
    and LANGSMITH_TRACING environment variables.
    """
    initial_state = {
        "service_request_id": service_request_id,
        "context": None,
        "urgency": None,
        "clinical_history": None,
        "letter_text": None,
        "document": None,
        "document_reference_id": None,
        "error": None,
    }

    print(f"\nRunning referral letter pipeline for ServiceRequest: {service_request_id}")
    print("-" * 60)

    final_state = referral_graph.invoke(initial_state)

    if final_state.get("error"):
        print(f"\nPipeline error: {final_state['error']}")
        return final_state

    result = {
        "status": "success",
        "document_reference_id": final_state.get("document_reference_id"),
        "service_request_id": service_request_id,
        "urgency": final_state["urgency"].urgency.value if final_state.get("urgency") else None,
        "urgency_rationale": final_state["urgency"].clinical_rationale if final_state.get("urgency") else None,
        "confidence": final_state["urgency"].confidence if final_state.get("urgency") else None,
        "specialty": final_state["context"].specialty if final_state.get("context") else None,
        "referral_reason": final_state["context"].referral_reason if final_state.get("context") else None,
    }

    print("\nPipeline completed successfully.")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <service_request_id>")
        print("Example: python main.py 2c3acb29-c1d2-4d43-af58-e62d2a60896d")
        sys.exit(1)

    run(sys.argv[1])
