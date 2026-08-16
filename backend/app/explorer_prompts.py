import json

SUPPORTED_TOPICS = {
    "overview",
    "symptoms",
    "causes",
    "spread",
    "severity",
    "management",
    "prevention",
    "monitoring",
}


def build_explorer_prompt(plant: str, disease: str, topics: list[str], knowledge_context: dict) -> str:
    requested_topics = ", ".join(topics)
    topic_keys = json.dumps(topics)

    # Build a short verified-snippet for each requested topic
    snippets = []
    for t in topics:
        value = knowledge_context.get(t) if isinstance(knowledge_context, dict) else None
        if not value:
            snippet = f"{t}: (no verified information available)"
        elif isinstance(value, list):
            snippet = f"{t}: " + " | ".join(v for v in value)
        else:
            snippet = f"{t}: {value}"
        snippets.append(snippet)
    verified_block = "\n".join(snippets)

    return f"""
You are a careful agricultural advisor. The user wants general information about a crop disease, not a diagnosis from an image.

Plant: {plant}
Disease: {disease}
Requested topics: {requested_topics}

Verified knowledge (only include these items when answering):
{verified_block}

Instructions:
- Answer only the requested topics and use ONLY the verified knowledge above for disease-specific facts.
- Write in plain, farmer-friendly language.
- Do not diagnose from an image or claim certainty beyond the provided crop and disease context.
- Do not invent pesticide dosages, concentrations, schedules, or unsupported chemical recommendations.
- If a topic is not supported by the verified knowledge above, say so briefly and recommend consulting local extension guidance.
- Return valid JSON with exactly these keys in this order: {topic_keys}
- Each value must be a short paragraph or a few concise bullets in plain text.
- Do not add any extra keys or commentary outside the JSON.
""".strip()
