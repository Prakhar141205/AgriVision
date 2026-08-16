from dataclasses import dataclass


SUPPORTED_QUESTION_TYPES = {
    "overview",
    "symptoms",
    "causes",
    "severity",
    "management",
    "prevention",
    "immediate_actions",
}


SAFETY_INSTRUCTIONS = """
You are an agricultural disease assistant. The disease classification was produced by a separate computer vision model and is provided as part of the context.
You are NOT responsible for diagnosing the image. Do not visually diagnose, override, or contradict the CV prediction.
Use the supplied agricultural knowledge as the PRIMARY source for all disease-specific facts; do not invent or add disease facts beyond the supplied knowledge.
Use phrases like "the model predicts" or "the image is consistent with"; do not state that the plant definitively has the disease.
CNN confidence is not disease severity. Never equate confidence with severity.
Avoid unsupported pesticide names, dosages, concentrations, or spray schedules.
When chemical control may be relevant, instruct the user to consult locally approved extension guidance and product labels.
Keep answers practical, farmer-friendly, concise, and explicitly grounded in the supplied knowledge.
""".strip()


LOW_CONFIDENCE_INSTRUCTIONS = """
The CNN confidence is low. Clearly state that the visual diagnosis is uncertain.
Do not present the predicted disease as confirmed.
Recommend a clearer image and/or advice from a local agricultural extension officer or agronomist.
Still answer cautiously using conditional language.
""".strip()


@dataclass(frozen=True)
class PromptContext:
    crop: str
    disease: str
    confidence: float
    top_predictions: str
    # knowledge_context will be a structured dict returned by get_disease_knowledge
    knowledge_context: dict
    low_confidence: bool


def _render_list(items: list | None) -> str:
    if not items:
        return "(no verified information available)"
    return "\n".join(f"- {s}" for s in items)


def _section_text(knowledge: dict, section: str) -> str:
    # Return a short plain-text summary for a given section
    if not knowledge or not knowledge.get("knowledge_available"):
        return "(no verified information available)"
    value = knowledge.get(section)
    if not value:
        return "(no verified information available)"
    if isinstance(value, list):
        return _render_list(value)
    return str(value)


def format_context(context: PromptContext) -> str:
    uncertainty = LOW_CONFIDENCE_INSTRUCTIONS if context.low_confidence else "The CNN confidence is adequate for disease-specific guidance, but still avoid absolute certainty."
    kb = context.knowledge_context or {}
    available = kb.get("knowledge_available", False)
    available_note = "Knowledge available" if available else "Knowledge NOT available"

    return f"""
Crop: {context.crop}
CNN-detected disease: {context.disease}
CNN confidence: {context.confidence:.4f}
Top CNN predictions: {context.top_predictions}
Knowledge status: {available_note}
Confidence handling: {uncertainty}
""".strip()


def build_overview_prompt(context: PromptContext) -> str:
    kb = context.knowledge_context or {}
    overview = _section_text(kb, "overview")
    important = _section_text(kb, "important_notes")
    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Relevant verified knowledge (overview):
{overview}

Important notes:
{important}

Question: What is this disease?
Write a concise farmer-friendly answer in 4 short bullet points.
Cover the disease overview, crop affected, likely disease type, and why it matters.
Use ONLY the verified knowledge above; do not invent details.
Write only the final answer.
""".strip()


def build_symptoms_prompt(context: PromptContext) -> str:
    kb = context.knowledge_context or {}
    symptoms = _section_text(kb, "symptoms")
    monitoring = _section_text(kb, "monitoring")
    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Verified symptoms:
{symptoms}

Monitoring guidance:
{monitoring}

Question: What are the symptoms?
Write a farmer-friendly answer in 4 short bullet points focused on observable signs and what to monitor.
Use ONLY the verified knowledge above; do not invent symptoms.
Write only the final answer.
""".strip()


def build_causes_prompt(context: PromptContext) -> str:
    kb = context.knowledge_context or {}
    causes = _section_text(kb, "causes")
    spread = _section_text(kb, "spread")
    favorable = _section_text(kb, "favorable_conditions")
    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Verified causes:
{causes}

Verified spread routes:
{spread}

Conditions that favor the disease:
{favorable}

Question: What causes it and how does it spread?
Write a farmer-friendly answer in 4 short bullet points, focusing only on verified causes and spread.
Use ONLY the verified knowledge above; do not invent causal factors.
Write only the final answer.
""".strip()


def build_severity_prompt(context: PromptContext) -> str:
    kb = context.knowledge_context or {}
    favorable = _section_text(kb, "favorable_conditions")
    important = _section_text(kb, "important_notes")
    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Known risk conditions:
{favorable}

Important notes:
{important}

Question: How serious is it?
Write a farmer-friendly answer in 4 short bullet points covering potential impact and risk factors.
Important: do not equate CNN confidence with disease severity. Use cautious language.
Write only the final answer.
""".strip()


def build_management_prompt(context: PromptContext) -> str:
    kb = context.knowledge_context or {}
    management = _section_text(kb, "management")
    important = _section_text(kb, "important_notes")
    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Verified management options (from knowledge base):
{management}

Important notes and cautions:
{important}

Question: How can I control or manage it?
Write a farmer-friendly answer in 5 short bullet points drawn ONLY from the verified management options above.
Do NOT add or invent any management actions that are not already present in the verified management options.
If chemical control is relevant but not present in the verified options, respond: "Chemical control: consult local extension and product labels; specific products and dosages are not provided." Do not invent pesticide names, dosages, concentrations, or schedules.
Write only the final answer.
""".strip()


def build_prevention_prompt(context: PromptContext) -> str:
    kb = context.knowledge_context or {}
    prevention = _section_text(kb, "prevention")
    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Verified prevention measures:
{prevention}

Question: How can I prevent it?
Write a farmer-friendly answer in 6 short bullet points drawn ONLY from verified prevention measures above. Do not invent prevention actions not supported by the knowledge base.
Write only the final answer.
""".strip()


def build_immediate_actions_prompt(context: PromptContext) -> str:
    kb = context.knowledge_context or {}
    management = _section_text(kb, "management")
    prevention = _section_text(kb, "prevention")
    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Verified immediate management options:
{management}

Verified prevention measures:
{prevention}

Question: What should I do right now?
Return a concise prioritized action plan using ONLY the verified management and prevention items above.
Structure the response as:

Immediate:
1. ...
2. ...

Next:
1. ...
2. ...
""".strip()


PROMPT_BUILDERS = {
    "overview": build_overview_prompt,
    "symptoms": build_symptoms_prompt,
    "causes": build_causes_prompt,
    "severity": build_severity_prompt,
    "management": build_management_prompt,
    "prevention": build_prevention_prompt,
    "immediate_actions": build_immediate_actions_prompt,
}


def build_custom_prompt(context: PromptContext, question: str) -> str:
    kb = context.knowledge_context or {}
    overview = _section_text(kb, "overview")
    management = _section_text(kb, "management")
    prevention = _section_text(kb, "prevention")
    symptoms = _section_text(kb, "symptoms")

    return f"""
{SAFETY_INSTRUCTIONS}

{format_context(context)}

Verified overview:
{overview}

Verified management:
{management}

Verified prevention:
{prevention}

Verified symptoms:
{symptoms}

User question: {question.strip()}
Primary task: Answer the user's question using ONLY the verified knowledge above. If the question requests information not covered by these verified fields, clearly state that the specific information is unavailable and recommend consulting local extension services or product labels.
Write a concise, farmer-friendly answer. Do not invent details or provide pesticide dosages or schedules.
""".strip()
