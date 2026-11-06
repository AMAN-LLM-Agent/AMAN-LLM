# src/pipeline/layer_toggle.py
import yaml, os
from dataclasses import dataclass

@dataclass
class LayerConfig:
    input_sanitizer_enabled: bool = True
    layer1_enabled: bool = True
    layer2_enabled: bool = True
    layer3_enabled: bool = True
    layer4_enabled: bool = True

    @classmethod
    def from_env(cls):
        return cls(
            layer1_enabled=os.getenv("LAYER1_ENABLED", "true").lower() == "true",
            layer2_enabled=os.getenv("LAYER2_ENABLED", "true").lower() == "true",
            layer3_enabled=os.getenv("LAYER3_ENABLED", "true").lower() == "true",
            layer4_enabled=os.getenv("LAYER4_ENABLED", "true").lower() == "true"
        )

    @classmethod
    def from_yaml(cls, path: str = "config/layer_config.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg['layers'])

    def describe(self) -> str:
        status = {
            "Layer 1 (Input Filter)":   "✅ ON" if self.layer1_enabled else "❌ OFF",
            "Layer 2 (Policy Gate)":    "✅ ON" if self.layer2_enabled else "❌ OFF",
            "Layer 3 (Output Monitor)": "✅ ON" if self.layer3_enabled else "❌ OFF",
            "Layer 4 (Self-Critique)":  "✅ ON" if self.layer4_enabled else "❌ OFF",
        }
        return "\n".join(f"  {k}: {v}" for k, v in status.items())