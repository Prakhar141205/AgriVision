import json
import re
from collections import defaultdict

from app.assistant import AssistantService
from app.explorer_prompts import SUPPORTED_TOPICS, build_explorer_prompt
from app.knowledge import get_knowledge_for_disease


def _parse_json_response(raw_response: str) -> dict[str, str]:
    cleaned = raw_response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1)
    elif cleaned.startswith("{") and cleaned.endswith("}"):
        pass
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in response.")
        cleaned = cleaned[start : end + 1]

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("The JSON payload is not an object.")
    return payload


def clean_fragment(value: str) -> str:
    value = value.replace("_", " ").replace("___", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_plant_name(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def normalize_disease_name(value: str) -> str:
    cleaned = value.strip().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def build_plant_disease_catalog(class_names: list[str]) -> dict[str, list[dict[str, str]]]:
    catalog: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()

    for label in class_names:
        if "___" not in label:
            continue
        plant_label, disease_label = label.split("___", maxsplit=1)
        plant_name = normalize_plant_name(plant_label)
        disease_name = normalize_disease_name(disease_label)
        disease_id = slugify(f"{plant_name} {disease_name}")
        key = (plant_name, disease_name)
        if key in seen:
            continue
        seen.add(key)
        catalog[plant_name].append({"id": disease_id, "name": disease_name})

    return {plant: sorted(diseases, key=lambda item: item["name"]) for plant, diseases in sorted(catalog.items())}


class ExplorerService:
    def __init__(self, assistant_service: AssistantService, class_names: list[str]) -> None:
        self.assistant_service = assistant_service
        self.catalog = build_plant_disease_catalog(class_names)

    def list_plants(self) -> list[str]:
        return sorted(self.catalog)

    def list_diseases_for_plant(self, plant: str) -> dict[str, list[dict[str, str]]]:
        normalized = normalize_plant_name(plant)
        if normalized not in self.catalog:
            supported = ", ".join(sorted(self.catalog))
            raise ValueError(f"Unsupported plant '{plant}'. Supported plants: {supported}.")
        return {"plant": normalized, "diseases": self.catalog[normalized]}

    def analyze(self, plant: str, disease: str, topics: list[str]) -> dict[str, object]:
        normalized_plant = normalize_plant_name(plant)
        if normalized_plant not in self.catalog:
            supported = ", ".join(sorted(self.catalog))
            raise ValueError(f"Unsupported plant '{plant}'. Supported plants: {supported}.")

        supported_topics = sorted(SUPPORTED_TOPICS)
        requested_topics = [topic.strip().lower() for topic in topics if isinstance(topic, str) and topic.strip()]
        if not requested_topics:
            raise ValueError("At least one valid topic is required for explorer analysis.")

        invalid = [topic for topic in requested_topics if topic not in SUPPORTED_TOPICS]
        if invalid:
            raise ValueError(
                f"Unsupported topic(s): {', '.join(invalid)}. Supported topics: {', '.join(supported_topics)}."
            )

        disease_value = disease.strip()
        if "___" in disease_value:
            disease_value = disease_value.split("___", maxsplit=1)[1]
        normalized_disease = normalize_disease_name(disease_value)
        allowed_diseases = {normalize_disease_name(item["name"]) for item in self.catalog[normalized_plant]}
        if normalized_disease not in allowed_diseases:
            supported_diseases = ", ".join(sorted(item["name"] for item in self.catalog[normalized_plant]))
            raise ValueError(
                f"Unsupported disease '{disease}' for plant '{normalized_plant}'. Supported diseases: {supported_diseases}."
            )

        knowledge_context = get_knowledge_for_disease(f"{normalized_plant}___{normalized_disease}", normalized_plant)
        prompt = build_explorer_prompt(normalized_plant, normalized_disease, requested_topics, knowledge_context)
        answer = self.assistant_service.generate_prompt_response(prompt, low_confidence=False)

        try:
            payload = _parse_json_response(answer)
        except ValueError as exc:
            raise ValueError("The explorer model did not return valid JSON for the requested topics.") from exc

        if not isinstance(payload, dict):
            raise ValueError("The explorer model returned an unexpected response format.")

        response = {}
        for topic in requested_topics:
            if topic not in payload:
                raise ValueError(f"The explorer response is missing the requested topic '{topic}'.")
            response[topic] = str(payload[topic])

        return {
            "plant": normalized_plant,
            "disease": normalized_disease,
            "information": response,
        }
