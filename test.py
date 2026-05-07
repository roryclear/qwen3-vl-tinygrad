from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
from PIL import Image
import requests
from io import BytesIO
import torch
from typing import Callable
import inspect

set_seed(42)

@torch.no_grad()
def generate(
    inputs: torch.Tensor | None = None,
    model=None,
    generation_config=None,
    prefix_allowed_tokens_fn=None,
    synced_gpus: bool | None = None,
    assistant_model= None,
    streamer=None,
    negative_prompt_ids: torch.Tensor | None = None,
    negative_prompt_attention_mask: torch.Tensor | None = None,
    custom_generate= None,
    **kwargs,
):
    generation_mode_kwargs = model._extract_generation_mode_kwargs(
        custom_generate,
        kwargs,
        synced_gpus,
        assistant_model,
        streamer,
    )

    has_default_max_length = (
        kwargs.get("max_length") is None
        and (generation_config is None or generation_config.max_length is None)
        and model.generation_config.max_length is None
    )
    has_default_min_length = (
        kwargs.get("min_length") is None
        and (generation_config is None or generation_config.min_length is None)
        and model.generation_config.min_length is None
    )
    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **kwargs)

    generation_mode = generation_config.get_generation_mode(assistant_model)
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )

    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    model_kwargs["position_ids"] = model._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)


    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # Expand inputs depending on the generation mode
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )


    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[1]
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    model_kwargs["logits_to_keep"] = 1

    max_cache_length = generation_config.max_length - 1
    model._prepare_cache_for_generation(
        generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    )

    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=[],
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=[],
        tokenizer=generation_mode_kwargs.get("tokenizer"),
    )

    model_kwargs["use_cache"] = generation_config.use_cache

    result = model._sample(
        input_ids,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        **generation_mode_kwargs,
        **model_kwargs,
    )

    return result

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

url = "https://img.wort.lu/public/luxemburg/vfka4n-picture-title-binary/alternates/ONE_ONE_256/Picture%20title%20binary"
image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this?<|im_end|>\n<|im_start|>assistant\n"
text_inputs = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)

image_inputs = processor.image_processor(images=image, return_tensors="pt")

merge_size = processor.image_processor.merge_size  # usually 2
image_grid_thw = image_inputs["image_grid_thw"]  # [batch, 3] -> [t, h, w]
num_image_tokens = (image_grid_thw.prod(dim=-1) / (merge_size ** 2)).item()

image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
input_ids = text_inputs["input_ids"][0].tolist()
image_token_positions = [i for i, tid in enumerate(input_ids) if tid == image_token_id]

for pos in reversed(image_token_positions):  # reversed to maintain indices
    input_ids[pos:pos+1] = [image_token_id] * int(num_image_tokens)

mm_token_type_ids = [0] * len(input_ids)
for pos in image_token_positions: mm_token_type_ids[pos:pos + int(num_image_tokens)] = [1] * int(num_image_tokens)

inputs = {
    'input_ids': torch.tensor([input_ids]),
    'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long),
    'mm_token_type_ids': torch.tensor([mm_token_type_ids]),
    'pixel_values': image_inputs['pixel_values'],
    'image_grid_thw': image_inputs['image_grid_thw']
}

print("manual inputs =", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})

outputs = generate(**inputs, model=model, max_new_tokens=128)
#outputs = model.generate(**inputs, max_new_tokens=128)
generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
output = processor.decode(generated_ids, skip_special_tokens=True)
print(output)
assert output == "This is a Ferrari F40, a legendary sports car produced by Ferrari from 1987 to 1992. It is renowned for its sleek design and powerful performance, making it one of the most iconic cars in automotive history."