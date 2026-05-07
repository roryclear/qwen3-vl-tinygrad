from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
from PIL import Image
import requests
from io import BytesIO
import torch
from torch import nn

set_seed(42)

def _update_model_kwargs_for_generation(
    position_ids,
    num_new_tokens=1):
    required_dim = [1] * (position_ids.dim() - 1) + [-1]
    return (
        torch.arange(num_new_tokens, dtype=position_ids.dtype, device=position_ids.device).view(*required_dim)
        + position_ids[..., -1:]
        + 1
    )

def _sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor,
    stopping_criteria,
    generation_config,
    pixel_values,
    **model_kwargs,
):
    pad_token_id = generation_config._pad_token_tensor
    output_scores = generation_config.output_scores
    return_dict_in_generate = generation_config.return_dict_in_generate
    scores = () if (return_dict_in_generate and output_scores) else None
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    prefill_consumed = False
    model_kwargs["pixel_values"] = pixel_values
    outputs = model._prefill(
        input_ids,
        generation_config,
        model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )
    del model_kwargs["pixel_values"]

    while not this_peer_finished:
        if prefill_consumed:
            next_sequence_length = 1 if model_kwargs["use_cache"] else None
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, next_sequence_length=next_sequence_length, **model_kwargs
            )
            with model._optimize_model_for_decode():
                outputs = model(**model_inputs, return_dict=True)
        prefill_consumed = True
        model_kwargs["position_ids"] = _update_model_kwargs_for_generation(
            model_kwargs["position_ids"],
        )
        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        # token selection
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    return input_ids

@torch.no_grad()
def generate(
    inputs: torch.Tensor | None = None,
    model=None,
    generation_config=None,
    prefix_allowed_tokens_fn=None,
    assistant_model= None,
    negative_prompt_ids: torch.Tensor | None = None,
    negative_prompt_attention_mask: torch.Tensor | None = None,
    input_ids=None,
    pixel_values=None,
    image_grid_thw=None,
    max_new_tokens=128
):
    generation_config, _ = model._prepare_generation_config(generation_config, input_ids=input_ids, image_grid_thw=image_grid_thw, max_new_tokens=max_new_tokens)
    
    model_kwargs = {"input_ids": input_ids, "image_grid_thw": image_grid_thw}

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
        has_default_max_length=True,
        has_default_min_length=True,
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
        tokenizer=None,
    )

    model_kwargs["use_cache"] = generation_config.use_cache
    result = _sample(
        model,
        input_ids,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        pixel_values=pixel_values,
        synced_gpus=False,
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


outputs = generate(input_ids=torch.tensor([input_ids]), pixel_values=image_inputs['pixel_values'], image_grid_thw=image_inputs['image_grid_thw'], model=model, max_new_tokens=128)
#outputs = model.generate(**inputs, max_new_tokens=128)
generated_ids = outputs[0][len(input_ids):]
output = processor.decode(generated_ids, skip_special_tokens=True)
print(output)
assert output == "This is a Ferrari F40, a legendary sports car produced by Ferrari from 1987 to 1992. It is renowned for its sleek design and powerful performance, making it one of the most iconic cars in automotive history."