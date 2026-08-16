from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

# Structured disease knowledge store.
# Only a small set of well-grounded diseases are populated for the hackathon MVP.
# Fields left empty should be treated as "information unavailable" rather than invented.

@dataclass
class Source:
    title: str
    organization: str | None = None
    url: str | None = None


DISEASE_KB: Dict[str, Dict[str, Any]] = {
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Tomato___Early_blight",
        "overview": "Early blight is a common fungal disease of tomato that typically begins on older leaves as circular lesions with concentric rings.",
        "symptoms": [
            "Circular lesions on older leaves often with concentric rings",
            "Yellowing and browning of leaf tissue, beginning at leaf tips or older foliage",
            "Progressive defoliation under high disease pressure",
        ],
        "causes": ["Fungal pathogens that persist on crop residue and in soil; spores spread by splashing water"],
        "spread": ["Splashing rain or irrigation water; infected plant debris"],
        "favorable_conditions": ["Warm, humid conditions and frequent leaf wetness"],
        "management": [
            "Sanitation: remove infected crop debris and diseased plants",
            "Reduce leaf wetness: avoid late-day irrigation and improve airflow through pruning",
            "Rotate crops where practical to reduce inoculum in the field",
        ],
        "prevention": [
            "Use resistant varieties when available",
            "Maintain good field sanitation and remove volunteer plants",
            "Ensure proper plant spacing for airflow",
        ],
        "monitoring": ["Regularly inspect lower leaves for early symptoms, especially after wet periods"],
        "chemical_control": [],
        "important_notes": [
            "If chemical control is considered, follow locally approved product labels and extension guidance; specific products/dosages are outside this knowledge base."
        ],
        "source": {
            "title": "AgriVision curated summary",
            "organization": "AgriVision",
            "url": None,
        },
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Tomato___Late_blight",
        "overview": "Late blight is a fast-moving disease that can rapidly damage foliage and fruit under cool, wet conditions.",
        "symptoms": [
            "Dark, water-soaked lesions on leaves and stems",
            "Rapid collapse of tissue under persistent moisture",
        ],
        "causes": ["Oomycete pathogens that spread via wind-blown and splash-dispersed spores"],
        "spread": ["High humidity, prolonged leaf wetness, and cool temperatures favor spread"],
        "favorable_conditions": ["Cool, wet weather and dense canopies"],
        "management": [
            "Sanitation and removal of infected material",
            "Improve ventilation and reduce leaf wetness where possible",
        ],
        "prevention": ["Avoid overhead irrigation late in the day; promote airflow through canopy management"],
        "monitoring": ["Monitor fields closely during cool, wet weather; inspect leaves and stems for water-soaked lesions"],
        "chemical_control": [],
        "important_notes": [
            "Late blight can progress quickly; consult local extension services for rapid response options."
        ],
        "source": {"title": "AgriVision curated summary", "organization": "AgriVision", "url": None},
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "disease": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "overview": "Isariopsis leaf spot (commonly reported as a leaf blight in grape) causes small, circular to irregular leaf spots that may coalesce under wet conditions.",
        "symptoms": [
            "Small circular to irregular spots on leaves that may enlarge or coalesce",
            "Defoliation in severe cases leading to reduced canopy vigor",
        ],
        "causes": ["Fungal pathogens that favor warm, wet conditions and spread via splash"],
        "spread": ["Splash dispersal from rain or irrigation and infected plant debris"],
        "favorable_conditions": ["Warm, humid weather and frequent leaf wetting"],
        "management": [
            "Sanitation: remove infected residues and manage canopy to reduce humidity",
            "Avoid prolonged leaf wetness through irrigation management",
        ],
        "prevention": ["Maintain good canopy airflow and remove heavily infected shoots"],
        "monitoring": ["Inspect leaves after wet periods for spot development and monitor disease progression"],
        "chemical_control": [],
        "important_notes": [
            "Follow local extension guidance for any chemical controls; details are intentionally not provided here."
        ],
        "source": {"title": "AgriVision curated summary", "organization": "AgriVision", "url": None},
    },
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple___Apple_scab",
        "overview": "Apple scab is a fungal disease that causes olive-green to black lesions on leaves and fruit, favored by cool, wet weather.",
        "symptoms": [
            "Olive-green to black lesions on leaves and fruit",
            "Premature leaf drop in severe infections",
        ],
        "causes": ["Fungal pathogens whose spores are spread during wet periods"],
        "spread": ["Splash dispersal during rainy periods and overwintering in infected leaves"],
        "favorable_conditions": ["Cool, wet spring weather which favors ascospore release"],
        "management": [
            "Sanitation: remove and destroy fallen infected leaves",
            "Thinning and pruning to improve airflow",
        ],
        "prevention": ["Choose resistant cultivars when available and practice good orchard hygiene"],
        "monitoring": ["Monitor during spring wet periods for ascospore release and early leaf symptoms"],
        "chemical_control": [],
        "important_notes": ["Consult local extension for specific timing of interventions; local labels should guide chemical use."],
        "source": {"title": "AgriVision curated summary", "organization": "AgriVision", "url": None},
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Potato___Late_blight",
        "overview": "Late blight in potato is a fast-spreading disease favored by cool, moist conditions and can affect foliage and tubers.",
        "symptoms": [
            "Dark, water-soaked lesions on foliage",
            "Stem lesions and rapid collapse of plants under favorable conditions",
        ],
        "causes": ["Oomycete pathogens that produce motile spores under wet conditions"],
        "spread": ["Wind and rain-driven spore dispersal and infected seed tubers"],
        "favorable_conditions": ["Cool temperatures with high humidity and frequent rain"],
        "management": [
            "Sanitation and removal of infected material",
            "Avoid moving infected tubers between fields",
        ],
        "prevention": ["Use certified seed and rotate crops where possible"],
        "monitoring": ["Monitor foliage and tubers during humid conditions"],
        "chemical_control": [],
        "important_notes": ["Late blight can cause rapid crop loss; consult local extension for emergency measures."],
        "source": {"title": "AgriVision curated summary", "organization": "AgriVision", "url": None},
    },
}


DEFAULT_CONTEXT = {
    "knowledge_available": False,
    "message": "Use general agricultural guidance: emphasize sanitation, canopy management, irrigation timing, airflow, monitoring, and consult local extension. Disease-specific verified knowledge not available.",
}


def get_disease_knowledge(disease: str, crop: str | None = None) -> dict[str, Any]:
    """Return structured knowledge for a disease key.

    Behavior:
    1. If an exact 'Crop___Disease' key matches, return it.
    2. If a crop is provided, only return a matching disease when the record's crop matches the provided crop.
    3. If only a friendly disease name is provided and it uniquely matches a single record, return that record.
    4. Otherwise return DEFAULT_CONTEXT (knowledge_available=False).

    This avoids silently returning unrelated disease information for a different crop.
    """
    key = disease.strip()

    # Normalize helpers
    def _normalize(s: str) -> str:
        return s.replace("_", " ").strip().lower()

    requested_crop_norm = _normalize(crop) if crop else None

    # Accept labels in the form 'Crop___Disease' (preferred)
    if key in DISEASE_KB:
        # If a crop was provided, ensure it matches the record's crop to avoid returning unrelated info
        record = DISEASE_KB[key]
        record_crop_norm = _normalize(record.get("crop", ""))
        if requested_crop_norm and record_crop_norm != requested_crop_norm:
            result = DEFAULT_CONTEXT.copy()
            result["knowledge_available"] = False
            result["message"] = (
                "Verified agricultural information for this crop+disease combination is not available in the knowledge base."
            )
            result["requested_disease"] = disease
            result["requested_crop"] = crop
            return result
        payload = record.copy()
        payload["knowledge_available"] = True
        return payload

    # Extract friendly suffix (e.g., 'Early_blight' from 'Tomato___Early_blight' or 'Early_blight')
    if "___" in key:
        _, suffix = key.split("___", maxsplit=1)
    else:
        suffix = key

    suffix_norm = _normalize(suffix)

    # If crop provided: search records that match both crop and disease suffix
    if requested_crop_norm:
        for full_key, record in DISEASE_KB.items():
            record_crop = _normalize(record.get("crop", ""))
            record_name = full_key.split("___", maxsplit=1)[1] if "___" in full_key else full_key
            if record_crop == requested_crop_norm and _normalize(record_name) == suffix_norm:
                payload = record.copy()
                payload["knowledge_available"] = True
                return payload
        # No matching record for this crop+disease
        result = DEFAULT_CONTEXT.copy()
        result["knowledge_available"] = False
        result["message"] = (
            "Verified agricultural information for this crop+disease combination is not available in the knowledge base."
        )
        result["requested_disease"] = disease
        result["requested_crop"] = crop
        return result

    # Crop not provided: attempt to find unique suffix match across all records
    matches = []
    for full_key, record in DISEASE_KB.items():
        record_name = full_key.split("___", maxsplit=1)[1] if "___" in full_key else full_key
        if _normalize(record_name) == suffix_norm:
            matches.append(record)

    if len(matches) == 1:
        payload = matches[0].copy()
        payload["knowledge_available"] = True
        return payload

    # Ambiguous or not found
    result = DEFAULT_CONTEXT.copy()
    result["knowledge_available"] = False
    result["message"] = (
        "Verified agricultural information for this disease is not currently available in the knowledge base."
    )
    result["requested_disease"] = disease
    if crop:
        result["requested_crop"] = crop
    return result


# Backwards-compatible alias used by existing code
get_knowledge_for_disease = get_disease_knowledge
