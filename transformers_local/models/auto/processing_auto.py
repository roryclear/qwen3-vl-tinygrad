import importlib
from collections import OrderedDict
from typing import TYPE_CHECKING

from ...image_processing_utils import ImageProcessingMixin
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)

if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    PROCESSOR_MAPPING_NAMES: OrderedDict[str, str | None] = OrderedDict()
else:
    PROCESSOR_MAPPING_NAMES = OrderedDict(
        [
            ("qwen2_5_omni", "Qwen2_5OmniProcessor"),
            ("qwen2_5_vl", "Qwen2_5_VLProcessor"),
            ("qwen2_audio", "Qwen2AudioProcessor"),
            ("qwen2_vl", "Qwen2VLProcessor"),
            ("qwen3_5", "Qwen3VLProcessor"),
            ("qwen3_5_moe", "Qwen3VLProcessor"),
            ("qwen3_omni_moe", "Qwen3OmniMoeProcessor"),
            ("qwen3_vl", "Qwen3VLProcessor"),
            ("qwen3_vl_moe", "Qwen3VLProcessor"),
        ]
    )

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)


def processor_class_from_name(class_name: str):
    for module_name, processors in PROCESSOR_MAPPING_NAMES.items():
        if class_name in processors:
            module_name = model_type_to_module_name(module_name)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for processor in PROCESSOR_MAPPING._extra_content.values():
        if getattr(processor, "__name__", None) == class_name:
            return processor



class AutoProcessor:
    def __init__(self): pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, _ = ImageProcessingMixin.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
        processor_class = config_dict.get("processor_class", None)
        processor_class = processor_class_from_name(processor_class)
        return processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

__all__ = ["PROCESSOR_MAPPING", "AutoProcessor"]
