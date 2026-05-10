from transformers import AutoModelForImageTextToText
from transformers.masking_utils import create_causal_mask
from PIL import Image
import requests
from io import BytesIO
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import importlib
import random
import numpy as np
from tinygrad import Tensor
import math


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)



set_seed(42)


class output_class():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def forward1(
    model,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values= None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    **kwargs):
    inputs_embeds = model.language_model.embed_tokens(input_ids)

    pixel_values = pixel_values.type(model.visual.dtype)

    hidden_states = model.visual.patch_embed(pixel_values)

    pos_embeds = model.visual.fast_pos_embed_interpolate(image_grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = model.visual.rot_pos_emb(image_grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(model.visual.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in model.visual.deepstack_visual_indexes:
            deepstack_feature = model.visual.deepstack_merger_list[model.visual.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    merged_hidden_states = model.visual.merger(hidden_states)

    vision_output = [merged_hidden_states, deepstack_feature_lists]

    image_embeds = vision_output[0]
    split_sizes = (image_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    vision_output[0] = image_embeds
    image_outputs = vision_output
    deepstack_image_embeds = image_outputs[1]
    image_embeds = image_embeds[0]
    image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)

    inputs_embeds[image_mask] = image_embeds.view(-1)

    image_mask = image_mask[..., 0]

    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]

    attention_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = model.language_model.rotary_emb(hidden_states, position_ids)
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs
        if deepstack_image_embeds is not None and layer_idx in range(len(deepstack_image_embeds)):
            hidden_states = model.language_model._deepstack_process(
                hidden_states,
                image_mask,
                deepstack_image_embeds[layer_idx],
            )
    hidden_states = model.language_model.norm(hidden_states)
    return hidden_states

def _prefill(
    model,
    input_ids,
    pixel_values,
    past_key_values,
    image_grid_thw):


    position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1)
    hidden_states = forward1(model.model, pixel_values=pixel_values,
                        past_key_values=past_key_values,
                        image_grid_thw=image_grid_thw,
                        position_ids=position_ids,
                        input_ids=input_ids)
    logits = model.lm_head(hidden_states[:, -1:, :])
    return logits

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
    outputs = _prefill(
        model,
        input_ids,
        pixel_values,
        model_kwargs["past_key_values"],
        model_kwargs["image_grid_thw"]
    )

    while not this_peer_finished:
        if prefill_consumed:
            model_inputs = {"input_ids": input_ids[:, -1:], "past_key_values": model_kwargs["past_key_values"], "position_ids": model_kwargs["position_ids"]}
            outputs = model.model(**model_inputs)
            hidden_states = outputs[0]
            outputs = model.lm_head(hidden_states[:, -1:, :])

        prefill_consumed = True
        model_kwargs["position_ids"] = _update_model_kwargs_for_generation(
            model_kwargs["position_ids"],
        )
        
        temp = 0.7
        top_k = 20
        filter_value = -math.inf
        min_tokens_to_keep = 1
        top_p = 0.8

        next_token_logits = outputs[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        scores = next_token_logits / temp


        top_k = min(20, scores.size(-1))  # Safety check
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, filter_value)
        scores = scores_processed


        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, filter_value)
        
        next_token_scores = scores_processed

        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        this_peer_finished = input_ids[0][-1] == 151645 or len(input_ids[0]) == 406
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



    model_kwargs = {"image_grid_thw": image_grid_thw}
    batch_size = input_ids.shape[0]

    device = input_ids.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    model_kwargs["position_ids"] = torch.arange(input_ids.shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1)


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
        model_input_name="input_ids",
        inputs_tensor=input_ids,
        input_ids_length=input_ids_length,
    )


    model_kwargs["logits_to_keep"] = 1

    max_cache_length = generation_config.max_length - 1
    model._prepare_cache_for_generation(
        generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    )

    model_kwargs["use_cache"] = generation_config.use_cache
    result = _sample(
        model,
        input_ids,
        generation_config=generation_config,
        pixel_values=pixel_values,
        **model_kwargs,
    )

    return result

from transformers.models.qwen3_vl import Qwen3VLProcessor
processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# model visual in tiny to start?
#print(model.model)
#print(model.model.visual.patch_embed, type(model.model.visual.patch_embed.proj))

urls = ["https://img.wort.lu/public/luxemburg/vfka4n-picture-title-binary/alternates/ONE_ONE_256/Picture%20title%20binary",
        "https://www.cartell.ie/car_check/wp-content/uploads/2012/03/Nissan-Micra-_4b.jpg"]

expected_outputs = ["This is a Ferrari F40, a legendary sports car produced by Ferrari from 1987 to 1992. It is renowned for its sleek design and powerful performance, making it one of the most iconic cars in automotive history.",
                    "This is a Nissan Micra, a compact car produced by the Japanese automaker Nissan. The Micra is a popular and affordable car, known for its reliability and efficiency.\n\nThe Nissan Micra was first introduced in 1990 as a small, affordable car. It was designed to compete with other small cars in the market, and it quickly gained popularity due to its fuel efficiency and low cost.\n\nThe Micra was produced in several different versions, including the 1.0L and 1.3L engines, which were available in different configurations. The Micra was also available with different body styles, including the standard"]

prompts = ["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this?<|im_end|>\n<|im_start|>assistant\n",
           "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nTell me the history of this car<|im_end|>\n<|im_start|>assistant\n"]

for url, expected_output, prompt in zip(urls, expected_outputs, prompts):
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    text_inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    
    image_inputs = processor.image_processor.preprocess(images=image, return_tensors="pt")

    merge_size = processor.image_processor.merge_size  # usually 2
    image_grid_thw = image_inputs["image_grid_thw"]  # [batch, 3] -> [t, h, w]
    num_image_tokens = (image_grid_thw.prod(dim=-1) / (merge_size ** 2)).item()

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_token_positions = [i for i, tid in enumerate(text_inputs) if tid == image_token_id]

    for pos in reversed(image_token_positions):  # reversed to maintain indices
        text_inputs[pos:pos+1] = [image_token_id] * int(num_image_tokens)

    mm_token_type_ids = [0] * len(text_inputs)
    for pos in image_token_positions: mm_token_type_ids[pos:pos + int(num_image_tokens)] = [1] * int(num_image_tokens)


    outputs = generate(input_ids=torch.tensor([text_inputs]), pixel_values=image_inputs['pixel_values'], image_grid_thw=image_inputs['image_grid_thw'], model=model, max_new_tokens=128)
    #outputs = model.generate(**inputs, max_new_tokens=128)
    generated_ids = outputs[0][len(text_inputs):]
    output = processor.decode(generated_ids, skip_special_tokens=True)
    print(output)
    assert output == expected_output