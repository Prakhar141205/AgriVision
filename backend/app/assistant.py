import re
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from app.assistant_prompts import PROMPT_BUILDERS, PromptContext, build_custom_prompt
from app.config import Settings
from app.knowledge import get_knowledge_for_disease
from app.model import select_device
from app.schemas import AssistantPredictionItem


@dataclass(frozen=True)
class DiseaseMetadata:
    crop: str
    condition: str
    is_healthy: bool


class DiseaseKnowledgeStore:
    """Small retrieval hook that can later be replaced by a richer knowledge layer."""

    def retrieve(self, disease: str, crop: str | None = None) -> str:
        return get_knowledge_for_disease(disease=disease, crop=crop)


class AssistantService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = select_device(settings.assistant_device)
        self.knowledge_store = DiseaseKnowledgeStore()

        load_kwargs = {}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        self.model_config = AutoConfig.from_pretrained(settings.hf_model_id, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.hf_model_id, **load_kwargs)
        model_class = AutoModelForSeq2SeqLM if self.model_config.is_encoder_decoder else AutoModelForCausalLM
        self.model = model_class.from_pretrained(settings.hf_model_id, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer_predefined(
        self,
        crop: str,
        disease: str,
        confidence: float,
        question_type: str,
        top_predictions: list[AssistantPredictionItem] | None = None,
    ) -> str:
        if question_type not in PROMPT_BUILDERS:
            supported = ", ".join(sorted(PROMPT_BUILDERS))
            raise ValueError(f"Unsupported question_type '{question_type}'. Supported values: {supported}.")

        # Build a context tailored to the requested question_type to avoid sending unnecessary knowledge
        context = self._build_context(crop, disease, confidence, top_predictions, question_type=question_type)
        prompt = PROMPT_BUILDERS[question_type](context)
        return self.generate_prompt_response(prompt, context.low_confidence)


    def answer_custom(
        self,
        crop: str,
        disease: str,
        confidence: float,
        question: str,
        top_predictions: list[AssistantPredictionItem] | None = None,
    ) -> str:
        # For custom questions include the full knowledge context so the user's question can be answered
        context = self._build_context(crop, disease, confidence, top_predictions, question_type=None)
        prompt = build_custom_prompt(context, question)
        return self.generate_prompt_response(prompt, context.low_confidence)


    def generate_prompt_response(self, prompt: str, low_confidence: bool = False) -> str:
        return self._generate(prompt, low_confidence)

    def _build_context(
        self,
        crop: str,
        disease: str,
        confidence: float,
        top_predictions: list[AssistantPredictionItem] | None,
        question_type: str | None = None,
    ) -> PromptContext:
        metadata = parse_disease_metadata(disease)
        resolved_crop = (crop or metadata.crop).strip() or metadata.crop
        resolved_disease = metadata.condition

        # Retrieve the full knowledge record first
        full_knowledge = self.knowledge_store.retrieve(disease, resolved_crop) or {}

        # Map question types to the specific knowledge sections that should be included
        type_to_sections = {
            "overview": ["overview", "important_notes", "source"],
            "symptoms": ["symptoms", "monitoring", "source"],
            "causes": ["causes", "spread", "favorable_conditions", "source"],
            "severity": ["favorable_conditions", "important_notes", "source"],
            "management": ["management", "important_notes", "chemical_control", "source"],
            "prevention": ["prevention", "important_notes", "source"],
            "immediate_actions": ["management", "prevention", "important_notes", "source"],
        }

        if question_type and question_type in type_to_sections:
            filtered = {k: v for k, v in full_knowledge.items() if k in type_to_sections[question_type] or k in ("knowledge_available", "message", "requested_disease", "requested_crop")}
        else:
            # For custom questions or unknown types include the entire knowledge record
            filtered = full_knowledge

        return PromptContext(
            crop=resolved_crop,
            disease=resolved_disease,
            confidence=confidence,
            top_predictions=format_top_predictions(top_predictions),
            knowledge_context=filtered,
            low_confidence=confidence < self.settings.assistant_low_confidence_threshold,
        )

    def _generate(self, prompt: str, low_confidence: bool) -> str:
        messages = [
            {
                "role": "system",
                "content": "You provide cautious, practical agricultural disease guidance based on a CNN prediction.",
            },
            {"role": "user", "content": prompt},
        ]
        if not self.model_config.is_encoder_decoder and hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            model_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            model_input = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nAnswer:"

        encoded = self.tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.tokenizer.model_max_length, 1024),
        ).to(self.device)
        do_sample = self.settings.assistant_temperature > 0

        generation_kwargs = {
            "max_new_tokens": self.settings.assistant_max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.model_config.is_encoder_decoder:
            generation_kwargs["num_beams"] = 4
        if do_sample:
            generation_kwargs["temperature"] = self.settings.assistant_temperature

        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **generation_kwargs)

        if self.model_config.is_encoder_decoder:
            generated_ids = output_ids[0]
        else:
            generated_ids = output_ids[0][encoded["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = clean_answer(answer)
        if low_confidence:
            answer = ensure_low_confidence_notice(answer)
        return answer


def parse_disease_metadata(disease: str) -> DiseaseMetadata:
    if "___" in disease:
        crop, condition = disease.split("___", maxsplit=1)
    else:
        crop, condition = "Unknown crop", disease

    crop = crop.replace("_", " ").strip().title() or "Unknown crop"
    condition = condition.replace("_", " ").strip()
    condition = re.sub(r"\s+", " ", condition)
    is_healthy = condition.lower() == "healthy"
    if is_healthy:
        condition = "healthy plant"

    return DiseaseMetadata(crop=crop, condition=condition, is_healthy=is_healthy)


def format_top_predictions(top_predictions: list[AssistantPredictionItem] | None) -> str:
    if not top_predictions:
        return "Not provided."
    return "; ".join(
        f"{parse_disease_metadata(item.disease).condition} ({item.confidence:.4f})"
        for item in top_predictions[:5]
    )


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^(assistant|answer):\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def ensure_low_confidence_notice(answer: str) -> str:
    notice = (
        "The CNN confidence is low, so this visual diagnosis is uncertain. "
        "Use this as cautious guidance, upload a clearer image if possible, and consult local agricultural support for confirmation."
    )
    if "confidence is low" in answer.lower() or "diagnosis is uncertain" in answer.lower():
        return answer
    return f"{notice}\n\n{answer}".strip()
