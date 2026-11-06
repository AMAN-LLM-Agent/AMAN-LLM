from src.layer3.output_validator import ValidatorConfig

# Disable ML-heavy checks locally — use rule-based fallbacks only
CPU_DEV_CONFIG = ValidatorConfig(
    toxicity_threshold    = 0.7,
    embedding_model       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    toxicity_model        = "unitary/multilingual-toxic-xlm-roberta",
    max_input_length      = 2_000,      # shorter = faster on CPU
    require_tool_evidence = False,      # don't hard-fail when no RAG evidence supplied
    enable_islamic_guard  = True,       # cheap regex — keep enabled even on CPU
)