from mcp.server.fastmcp import FastMCP
from datetime import datetime
import json

mcp = FastMCP("advanced_allergy_clinic_server")

# ---------------------------------------------------------------------
# Fictional demo data only. This server is for MCP learning, not care.
# ---------------------------------------------------------------------

PATIENTS = {
    "emma": {
        "age": 42,
        "known_allergens": ["grass pollen", "penicillin"],
        "medications": ["cetirizine"],
        "recent_vitals": {
            "blood_pressure": "118/76 mmHg",
            "heart_rate": 72,
            "temperature_f": 98.5,
            "oxygen_saturation": 98
        }
    },
    "john": {
        "age": 67,
        "known_allergens": ["dust mites"],
        "medications": ["lisinopril"],
        "recent_vitals": {
            "blood_pressure": "132/82 mmHg",
            "heart_rate": 78,
            "temperature_f": 98.7,
            "oxygen_saturation": 97
        }
    }
}


EPISODES = {
    "emma": {
        "episode-001": {
            "trigger": "grass pollen exposure during a park walk",
            "exposure_time": "2026-08-26 14:00",
            "symptoms": [
                {
                    "symptom": "sneezing",
                    "body_system": "respiratory",
                    "severity": 3,
                    "onset_minutes": 5
                },
                {
                    "symptom": "itchy watery eyes",
                    "body_system": "eyes_nose",
                    "severity": 4,
                    "onset_minutes": 7
                },
                {
                    "symptom": "runny nose",
                    "body_system": "respiratory",
                    "severity": 3,
                    "onset_minutes": 10
                }
            ]
        }
    },
    "john": {}
}


MEDICATION_ALLERGY_MAP = {
    "amoxicillin": "penicillin",
    "ampicillin": "penicillin",
    "augmentin": "penicillin"
}


# These are conservative red-flag phrases for this educational demo.
EMERGENCY_KEYWORDS = {
    "difficulty breathing",
    "trouble breathing",
    "shortness of breath",
    "throat tightness",
    "throat swelling",
    "tongue swelling",
    "wheezing",
    "fainting",
    "passing out",
    "weak pulse"
}


REFERENCE = {
    "early_allergy": {
        "summary": (
            "Early allergic-rhinitis symptoms can include sneezing, runny or "
            "stuffy nose, itchy nose/eyes/throat, and watery red eyes. Symptoms "
            "may begin soon after allergen exposure."
        ),
        "sources": [
            {
                "name": "Mayo Clinic - Hay fever",
                "url": "https://www.mayoclinic.org/diseases-conditions/hay-fever/symptoms-causes/syc-20373039"
            },
            {
                "name": "Mayo Clinic - Allergies",
                "url": "https://www.mayoclinic.org/diseases-conditions/allergies/symptoms-causes/syc-20351497"
            }
        ]
    },
    "allergy_vs_flu": {
        "summary": (
            "Itching, watery eyes, sneezing, and a clear exposure pattern favor "
            "allergy. Flu more often has abrupt systemic symptoms such as fever "
            "or chills, body aches, headache, cough, and fatigue. Symptoms can overlap."
        ),
        "sources": [
            {
                "name": "Mayo Clinic - Hay fever",
                "url": "https://www.mayoclinic.org/diseases-conditions/hay-fever/symptoms-causes/syc-20373039"
            },
            {
                "name": "CDC - Signs and Symptoms of Flu",
                "url": "https://www.cdc.gov/flu/signs-symptoms/index.html"
            }
        ]
    },
    "anaphylaxis": {
        "summary": (
            "Severe allergic-reaction warning signs can include throat or tongue "
            "swelling, difficulty breathing, wheezing, fainting, weak pulse, or "
            "other rapidly progressing symptoms. Emergency symptoms require immediate care."
        ),
        "sources": [
            {
                "name": "AAAAI - Anaphylaxis",
                "url": "https://www.aaaai.org/tools-for-the-public/conditions-library/allergies/anaphylaxis"
            }
        ]
    }
}


def get_patient(name: str) -> dict:
    patient = PATIENTS.get(name.lower())
    if not patient:
        raise ValueError(f"Patient '{name}' was not found.")
    return patient


def get_episode(name: str, episode_id: str) -> dict:
    episodes = EPISODES.get(name.lower(), {})
    episode = episodes.get(episode_id)
    if not episode:
        raise ValueError(
            f"Episode '{episode_id}' was not found for patient '{name}'."
        )
    return episode


def _symptom_texts(episode: dict) -> list[str]:
    return [item["symptom"].lower() for item in episode["symptoms"]]


def _max_severity(episode: dict) -> int:
    return max(
        (item["severity"] for item in episode["symptoms"]),
        default=0
    )


@mcp.tool()
async def get_patient_profile(name: str) -> dict:
    """Return a fictional patient's allergy-focused profile."""
    patient = get_patient(name)
    return {
        "name": name.title(),
        "age": patient["age"],
        "known_allergens": patient["known_allergens"],
        "medications": patient["medications"],
        "recent_vitals": patient["recent_vitals"],
        "episode_ids": list(EPISODES.get(name.lower(), {}).keys())
    }


@mcp.tool()
async def start_allergy_episode(
    name: str,
    trigger: str,
    exposure_time: str
) -> dict:
    """Start a new fictional allergy-symptom episode."""
    get_patient(name)

    episode_id = f"episode-{len(EPISODES.setdefault(name.lower(), {})) + 1:03d}"

    EPISODES[name.lower()][episode_id] = {
        "trigger": trigger,
        "exposure_time": exposure_time,
        "symptoms": []
    }

    return {
        "status": "created",
        "patient": name.title(),
        "episode_id": episode_id,
        "trigger": trigger,
        "exposure_time": exposure_time
    }


@mcp.tool()
async def record_episode_symptom(
    name: str,
    episode_id: str,
    symptom: str,
    body_system: str,
    severity: int,
    onset_minutes: int
) -> dict:
    """Record one symptom in a fictional episode.

    severity: 1-10
    onset_minutes: minutes after the suspected exposure
    """
    if not 1 <= severity <= 10:
        raise ValueError("Severity must be between 1 and 10.")

    if onset_minutes < 0:
        raise ValueError("onset_minutes cannot be negative.")

    episode = get_episode(name, episode_id)

    item = {
        "symptom": symptom.strip().lower(),
        "body_system": body_system.strip().lower(),
        "severity": severity,
        "onset_minutes": onset_minutes
    }

    episode["symptoms"].append(item)

    return {
        "status": "recorded",
        "patient": name.title(),
        "episode_id": episode_id,
        "symptom": item
    }


@mcp.tool()
async def get_episode_timeline(name: str, episode_id: str) -> dict:
    """Return the trigger and symptom timeline for a fictional episode."""
    episode = get_episode(name, episode_id)

    symptoms = sorted(
        episode["symptoms"],
        key=lambda item: item["onset_minutes"]
    )

    return {
        "patient": name.title(),
        "episode_id": episode_id,
        "trigger": episode["trigger"],
        "exposure_time": episode["exposure_time"],
        "symptoms": symptoms
    }


@mcp.tool()
async def get_allergy_reference(topic: str) -> dict:
    """Return concise, source-linked educational guidance.

    Valid topics: early_allergy, allergy_vs_flu, anaphylaxis
    """
    topic_key = topic.strip().lower()

    if topic_key not in REFERENCE:
        raise ValueError(
            "Unknown topic. Use early_allergy, allergy_vs_flu, or anaphylaxis."
        )

    return {
        "topic": topic_key,
        **REFERENCE[topic_key],
        "note": "Educational reference only; not a diagnosis."
    }


@mcp.tool()
async def compare_allergy_vs_flu(name: str, episode_id: str) -> dict:
    """Compare the fictional episode with simple allergy-like and flu-like clues.

    This rule-based comparison is educational and cannot diagnose either condition.
    """
    patient = get_patient(name)
    episode = get_episode(name, episode_id)

    symptom_text = " | ".join(_symptom_texts(episode))
    temperature = patient["recent_vitals"]["temperature_f"]

    allergy_clues = []
    flu_clues = []

    if "itch" in symptom_text:
        allergy_clues.append("itching")
    if "sneez" in symptom_text:
        allergy_clues.append("sneezing")
    if "watery" in symptom_text:
        allergy_clues.append("watery eyes")
    if "runny nose" in symptom_text:
        allergy_clues.append("runny nose")
    if any(item["onset_minutes"] <= 30 for item in episode["symptoms"]):
        allergy_clues.append("symptoms soon after exposure")

    if temperature >= 100.4:
        flu_clues.append("fever")
    if "chills" in symptom_text:
        flu_clues.append("chills")
    if "body ache" in symptom_text or "muscle ache" in symptom_text:
        flu_clues.append("body aches")
    if "headache" in symptom_text:
        flu_clues.append("headache")
    if "fatigue" in symptom_text:
        flu_clues.append("fatigue")

    if len(allergy_clues) >= 3 and len(flu_clues) == 0:
        pattern = "more allergy-like in this demo"
    elif len(flu_clues) >= 2 and len(allergy_clues) < 2:
        pattern = "more flu-like in this demo"
    else:
        pattern = "mixed or uncertain"

    return {
        "patient": name.title(),
        "episode_id": episode_id,
        "allergy_clues": allergy_clues,
        "flu_clues": flu_clues,
        "pattern": pattern,
        "disclaimer": (
            "This is a simple educational comparison. Symptoms overlap, "
            "and this output is not a diagnosis."
        )
    }


@mcp.tool()
async def assess_emergency_warning(name: str, episode_id: str) -> dict:
    """Check the fictional episode for severe allergic-reaction warning signs.

    This does not replace clinical judgment or an emergency action plan.
    """
    episode = get_episode(name, episode_id)

    symptom_names = _symptom_texts(episode)
    matches = sorted({
        keyword
        for keyword in EMERGENCY_KEYWORDS
        if any(keyword in symptom for symptom in symptom_names)
    })

    high_severity = _max_severity(episode) >= 8

    emergency_warning = bool(matches)

    if emergency_warning:
        action = (
            "Severe allergic-reaction warning symptom recorded. "
            "Use the person's prescribed emergency plan, including epinephrine "
            "if prescribed for anaphylaxis, and call emergency services."
        )
    elif high_severity:
        action = (
            "A high-severity symptom is recorded without a listed emergency keyword. "
            "Prompt clinical evaluation is appropriate."
        )
    else:
        action = (
            "No configured emergency warning symptom is recorded in this demo episode. "
            "Continue appropriate clinical follow-up and monitor for worsening symptoms."
        )

    return {
        "patient": name.title(),
        "episode_id": episode_id,
        "emergency_warning": emergency_warning,
        "matched_warning_signs": matches,
        "max_severity": _max_severity(episode),
        "action": action,
        "disclaimer": "Educational demo only; not a medical triage system."
    }


@mcp.tool()
async def check_medication_allergy(name: str, medication: str) -> dict:
    """Check a medication against a tiny fictional allergy mapping."""
    patient = get_patient(name)

    medication_key = medication.strip().lower()
    mapped_allergen = MEDICATION_ALLERGY_MAP.get(medication_key)

    conflict = (
        mapped_allergen in patient["known_allergens"]
        if mapped_allergen
        else False
    )

    return {
        "patient": name.title(),
        "medication": medication,
        "known_allergens": patient["known_allergens"],
        "possible_conflict": conflict,
        "matched_allergen": mapped_allergen if conflict else None,
        "note": (
            "Demo mapping only. Real medication-allergy assessment requires "
            "clinical verification."
        )
    }


@mcp.tool()
async def create_clinician_handoff(name: str, episode_id: str) -> str:
    """Create a structured visit handoff from the fictional episode."""
    patient = get_patient(name)
    episode = get_episode(name, episode_id)

    symptoms = sorted(
        episode["symptoms"],
        key=lambda item: item["onset_minutes"]
    )

    timeline_lines = [
        (
            f"- +{item['onset_minutes']} min: {item['symptom']} "
            f"({item['body_system']}, severity {item['severity']}/10)"
        )
        for item in symptoms
    ]

    return (
        f"ALLERGY EPISODE HANDOFF\n"
        f"Patient: {name.title()}\n"
        f"Episode: {episode_id}\n"
        f"Age: {patient['age']}\n"
        f"Known allergens: {', '.join(patient['known_allergens'])}\n"
        f"Current medications: {', '.join(patient['medications'])}\n"
        f"Suspected trigger: {episode['trigger']}\n"
        f"Exposure time: {episode['exposure_time']}\n"
        f"Symptom timeline:\n" + "\n".join(timeline_lines) + "\n"
        f"Recent vitals: {json.dumps(patient['recent_vitals'])}\n"
        f"Note: Fictional educational record; clinician review required."
    )


@mcp.resource("healthcare://patient/{name}/episode/{episode_id}")
async def allergy_episode_resource(name: str, episode_id: str) -> str:
    """Return a complete fictional allergy episode as JSON."""
    patient = get_patient(name)
    episode = get_episode(name, episode_id)

    return json.dumps(
        {
            "patient": name.title(),
            "profile": patient,
            "episode_id": episode_id,
            "episode": episode
        },
        indent=2
    )


@mcp.resource("healthcare://allergy/reference")
async def allergy_reference_resource() -> str:
    """Return the source-linked educational allergy reference."""
    return json.dumps(REFERENCE, indent=2)


@mcp.prompt()
def early_allergy_review(name: str, episode_id: str) -> str:
    """Reusable prompt for reviewing one fictional early-allergy episode."""
    return (
        f"Review the fictional allergy episode {episode_id} for {name}. "
        "Use the MCP tools to inspect the timeline, compare allergy-like versus "
        "flu-like clues, check emergency warning signs, and create a concise "
        "clinician handoff. Do not diagnose."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
