from src.state import WorkflowState
from src.data_loader import get_patient_rows

class HealthcareMultiAgentWorkflow:
    """Supervisor + specialized agents.

    Uses LangGraph when installed. Falls back to the same deterministic sequence
    if LangGraph is unavailable, which makes the repository easy to inspect/run.
    """

    def __init__(
        self,
        data,
        patient_agent,
        clinical_agent,
        claims_agent,
        risk_agent,
        evidence_agent,
        synthesis_agent,
        safety_agent,
    ):
        self.data = data
        self.patient_agent = patient_agent
        self.clinical_agent = clinical_agent
        self.claims_agent = claims_agent
        self.risk_agent = risk_agent
        self.evidence_agent = evidence_agent
        self.synthesis_agent = synthesis_agent
        self.safety_agent = safety_agent
        self.graph = self._build_graph()

    def _supervisor(self, state: WorkflowState):
        rows = get_patient_rows(self.data, state["patient_id"])
        if not rows["patient"]:
            raise ValueError(f"Unknown patient_id: {state['patient_id']}")
        return {
            "patient_rows": rows,
            "plan": [
                "patient_data_agent",
                "clinical_nlp_agent",
                "claims_agent",
                "risk_agent",
                "evidence_agent",
                "synthesis_agent",
                "safety_agent",
            ],
        }

    def _patient_node(self, state):
        return {"patient_summary": self.patient_agent.run(state["patient_rows"])}

    def _clinical_node(self, state):
        return {"clinical_summary": self.clinical_agent.run(state["patient_rows"])}

    def _claims_node(self, state):
        return {"claims_summary": self.claims_agent.run(state["patient_rows"])}

    def _risk_node(self, state):
        return {"risk_summary": self.risk_agent.run(state["patient_rows"])}

    def _evidence_node(self, state):
        return {
            "evidence_summary": self.evidence_agent.run(
                state["query"],
                state["patient_summary"],
                state["clinical_summary"],
                state["risk_summary"],
            )
        }

    def _synthesis_node(self, state):
        return {"report_data": self.synthesis_agent.run(state)}

    def _safety_node(self, state):
        return {
            "safety": self.safety_agent.run(
                state["report_data"]["report"],
                state["evidence_summary"],
            )
        }

    def _repair_node(self, state):
        repaired = self.safety_agent.repair(
            state["report_data"]["report"],
            state["evidence_summary"],
        )
        return {"final_report": repaired}

    def _finalize_node(self, state):
        return {"final_report": state["report_data"]["report"]}

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, START, END
        except ImportError:
            return None

        builder = StateGraph(WorkflowState)
        builder.add_node("supervisor", self._supervisor)
        builder.add_node("patient_agent", self._patient_node)
        builder.add_node("clinical_nlp_agent", self._clinical_node)
        builder.add_node("claims_agent", self._claims_node)
        builder.add_node("risk_agent", self._risk_node)
        builder.add_node("evidence_agent", self._evidence_node)
        builder.add_node("synthesis_agent", self._synthesis_node)
        builder.add_node("safety_agent", self._safety_node)
        builder.add_node("repair", self._repair_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "supervisor")

        # These three specialist agents run in parallel.
        builder.add_edge("supervisor", "patient_agent")
        builder.add_edge("supervisor", "clinical_nlp_agent")
        builder.add_edge("supervisor", "claims_agent")

        # Wait for all three branches before risk/evidence synthesis.
        builder.add_edge(
            ["patient_agent", "clinical_nlp_agent", "claims_agent"],
            "risk_agent"
        )
        builder.add_edge("risk_agent", "evidence_agent")
        builder.add_edge("evidence_agent", "synthesis_agent")
        builder.add_edge("synthesis_agent", "safety_agent")

        def route_after_safety(state):
            return "finalize" if state["safety"]["passed"] else "repair"

        builder.add_conditional_edges(
            "safety_agent",
            route_after_safety,
            {"finalize": "finalize", "repair": "repair"},
        )
        builder.add_edge("finalize", END)
        builder.add_edge("repair", END)
        return builder.compile()

    def invoke(self, patient_id: str, query: str):
        initial = {"patient_id": patient_id, "query": query}
        if self.graph is not None:
            return self.graph.invoke(initial)

        # Dependency-light fallback for code review or environments without LangGraph.
        state = dict(initial)
        state.update(self._supervisor(state))
        state.update(self._patient_node(state))
        state.update(self._clinical_node(state))
        state.update(self._claims_node(state))
        state.update(self._risk_node(state))
        state.update(self._evidence_node(state))
        state.update(self._synthesis_node(state))
        state.update(self._safety_node(state))
        if state["safety"]["passed"]:
            state.update(self._finalize_node(state))
        else:
            state.update(self._repair_node(state))
        return state
