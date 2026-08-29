from typing import Any


def normalize_memo_ai_settings(config: Any, fallback_config: Any = None) -> tuple[dict[str, dict[str, str]], str]:
    if not isinstance(config, dict):
        raise ValueError("config must be an object")

    llm = config.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("config missing 'llm' section")
    fallback_llm = fallback_config.get("llm", {}) if isinstance(fallback_config, dict) else {}
    if not isinstance(fallback_llm, dict):
        fallback_llm = {}

    raw_prompts = llm.get("memo_prompts")
    if raw_prompts is None:
        raw_prompts = fallback_llm.get("memo_prompts")
    if not isinstance(raw_prompts, dict) or not raw_prompts:
        raise ValueError("config 'llm.memo_prompts' must be a non-empty object")

    prompts: dict[str, dict[str, str]] = {}
    for raw_prompt_id, raw_meta in raw_prompts.items():
        prompt_id = str(raw_prompt_id).strip()
        if not prompt_id:
            raise ValueError("memo AI prompt ID is required")
        if prompt_id in prompts:
            raise ValueError(f"duplicate memo AI prompt ID: {prompt_id}")
        if not isinstance(raw_meta, dict):
            raise ValueError(f"memo AI prompt '{prompt_id}' must be an object")
        label = raw_meta.get("label")
        template = raw_meta.get("template")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"memo AI prompt '{prompt_id}' label is required")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"memo AI prompt '{prompt_id}' template is required")
        if "{text}" not in template:
            raise ValueError(f"memo AI prompt '{prompt_id}' template must contain {{text}}")
        prompts[prompt_id] = {"label": label.strip(), "template": template.strip()}

    default_prompt_id = llm.get("memo_default_prompt_id")
    if default_prompt_id is None:
        default_prompt_id = fallback_llm.get("memo_default_prompt_id")
    default_prompt_id = str(default_prompt_id or "").strip()
    if not default_prompt_id:
        default_prompt_id = next(iter(prompts))
    if default_prompt_id not in prompts:
        raise ValueError("llm.memo_default_prompt_id must exist in llm.memo_prompts")

    return prompts, default_prompt_id
