from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from .tools.pubmed_tool import search_pubmed
import os

@CrewBase
class BloodCancerCrew:
    """Sequential, bounded hematology research workflow."""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _agent(self, name, tools=None, iterations=5):
        return Agent(
            config=self.agents_config[name],
            tools=tools or [],
            max_iter=iterations,
            max_rpm=8,
            max_execution_time=180,
            allow_delegation=False,
            verbose=True,
        )

    def _web_tools(self):
        return [SerperDevTool()] if os.getenv("SERPER_API_KEY") else []

    @agent
    def publication_researcher(self):
        return self._agent("publication_researcher", [search_pubmed], 5)

    @agent
    def clinical_guideline_researcher(self):
        return self._agent("clinical_guideline_researcher", self._web_tools(), 5)

    @agent
    def symptom_analyst(self):
        return self._agent("symptom_analyst", [], 3)

    @agent
    def diagnostic_analyst(self):
        return self._agent("diagnostic_analyst", [], 4)

    @agent
    def laboratory_analyst(self):
        return self._agent("laboratory_analyst", [], 4)

    @agent
    def treatment_researcher(self):
        return self._agent("treatment_researcher", [search_pubmed] + self._web_tools(), 5)

    @agent
    def evidence_auditor(self):
        return self._agent("evidence_auditor", [], 4)

    @agent
    def report_manager(self):
        return self._agent("report_manager", [], 4)

    @task
    def collect_publications(self):
        return Task(config=self.tasks_config["collect_publications"])

    @task
    def collect_guidelines(self):
        return Task(config=self.tasks_config["collect_guidelines"])

    @task
    def analyze_symptoms(self):
        return Task(config=self.tasks_config["analyze_symptoms"])

    @task
    def review_diagnostics(self):
        return Task(config=self.tasks_config["review_diagnostics"])

    @task
    def review_laboratory_tests(self):
        return Task(config=self.tasks_config["review_laboratory_tests"])

    @task
    def review_treatments(self):
        return Task(config=self.tasks_config["review_treatments"])

    @task
    def audit_evidence(self):
        return Task(config=self.tasks_config["audit_evidence"])

    @task
    def synthesize_report(self):
        return Task(config=self.tasks_config["synthesize_report"])

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            tracing=False,
            max_rpm=8,
        )
