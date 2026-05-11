from transformers import AutoModelForImageTextToText
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
from transformers import DynamicCache
import copy
from functools import partial
from torchvision.transforms.v2 import functional as tvF
from dataclasses import dataclass, fields


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


    hidden_states = inputs_embeds
    position_embeddings = model.language_model.rotary_emb(hidden_states, position_ids)
    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=None,
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
    _pad_token_tensor,
    pixel_values,
    past_key_values,
    position_ids,
    image_grid_thw,
):
    pad_token_id = _pad_token_tensor
    scores = None
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    prefill_consumed = False
    outputs = _prefill(
        model,
        input_ids,
        pixel_values,
        past_key_values,
        image_grid_thw,
    )

    while not this_peer_finished:
        if prefill_consumed:
            model_inputs = {"input_ids": input_ids[:, -1:], "past_key_values": past_key_values, "position_ids": position_ids}
            outputs = model.model(**model_inputs)
            hidden_states = outputs[0]
            outputs = model.lm_head(hidden_states[:, -1:, :])

        prefill_consumed = True
        position_ids = _update_model_kwargs_for_generation(position_ids)
        
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


def preprocess(proc, images, *args, **kwargs):
    for kwarg_name in proc._valid_kwargs_names:
        kwargs.setdefault(kwarg_name, getattr(proc, kwarg_name, None))
    
    images = [tvF.pil_to_tensor(images)]
    return _preprocess(proc, images, *args, **kwargs)

def _iterate_items(items, is_nested: bool):
    """
    Helper function to iterate over items yielding (key, item) pairs.

    For nested structures, yields ((row_index, col_index), item).
    For flat structures, yields (index, item).
    """
    if is_nested:
        for i, row in enumerate(items):
            for j, item in enumerate(row):
                yield (i, j), item
    else:
        for i, item in enumerate(items):
            yield i, item

def group_images_by_shape(
    images,
    *paired_inputs,
    disable_grouping: bool | None,
    is_nested: bool = False,
) -> tuple[dict, ...]:
    disable_grouping = True

    if disable_grouping:
        grouped_images_index = {key: (key, 0) for key, _ in _iterate_items(images, is_nested)}
        if is_nested:
            grouped_images_index["_num_sublists"] = len(images)

        return (
            {key: img.unsqueeze(0) for key, img in _iterate_items(images, is_nested)},
            *[
                {key: item.unsqueeze(0) for key, item in _iterate_items(paired_list, is_nested)}
                for paired_list in paired_inputs
            ],
            grouped_images_index,
        )

    # Handle single level nested structure
    grouped_images, *paired_grouped_values, grouped_images_index = _group_images_by_shape(
        images, *paired_inputs, is_nested=is_nested
    )

    # Stack images with the same shape
    grouped_images = {shape: torch.stack(images_list, dim=0) for shape, images_list in grouped_images.items()}

    return grouped_images, *paired_grouped_values, grouped_images_index

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

@dataclass()
class SizeDict:
    """
    Hashable dictionary to store image size information.
    """

    height: int | None = None
    width: int | None = None
    longest_edge: int | None = None
    shortest_edge: int | None = None
    max_height: int | None = None
    max_width: int | None = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SizeDict.")

    def get(self, key, default=None):
        if hasattr(self, key) and getattr(self, key) is not None:
            return getattr(self, key)
        return default

    def __iter__(self):
        # Yield only non-None (key, value) pairs so dict(self) excludes missing values.
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                yield f.name, val

    def __hash__(self):
        return hash((self.height, self.width, self.longest_edge, self.shortest_edge, self.max_height, self.max_width))

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise KeyError(f"Key {key} is not a valid field of SizeDict.")
        object.__setattr__(self, key, value)

    def __eq__(self, other):
        if isinstance(other, dict):
            return dict(self) == other
        if isinstance(other, SizeDict):
            return tuple(getattr(self, f.name) for f in fields(self)) == tuple(
                getattr(other, f.name) for f in fields(self)
            )
        return NotImplemented

    def __or__(self, other) -> "SizeDict":
        if isinstance(other, dict | SizeDict):
            merged = dict(self)
            merged.update(dict(other))
            return SizeDict(**merged)
        return NotImplemented

    def __ror__(self, other) -> dict:
        if isinstance(other, dict):
            merged = dict(other)
            merged.update(dict(self))
            return merged
        return NotImplemented

def _preprocess(
    proc,
    images: list["torch.Tensor"],
    size,
    return_tensors,
    **kwargs):
    patch_size=16
    merge_size=2
    rescale_factor=0.00392156862745098
    do_normalize=True
    temporal_patch_size=2
    resample=3
    grouped_images, _ = group_images_by_shape(images, disable_grouping=None)
    resized_images_grouped = {}
    for shape, stacked_images in grouped_images.items():
        height, width = stacked_images.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        stacked_images = proc.resize(
            image=stacked_images,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )
        resized_images_grouped[shape] = stacked_images
    resized_images = [resized_images_grouped[0][0]]

    grouped_images, _ = group_images_by_shape(resized_images, disable_grouping=None)
    processed_images_grouped = {}
    processed_grids = {}
    for shape, stacked_images in grouped_images.items():
        resized_height, resized_width = stacked_images.shape[-2:]
        patches = proc.rescale_and_normalize(
            stacked_images, True, rescale_factor, do_normalize, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )
        batch_size, channel = patches.shape[:2]
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            batch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Reorder dimensions to group grid and patch information for subsequent flattening.
        # [batch, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)

        flatten_patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
        )

        processed_images_grouped[shape] = flatten_patches
        processed_grids[shape] = [[1, grid_h, grid_w]] * batch_size

    processed_images = [processed_images_grouped[0][0]]
    processed_grids_ordered = [processed_grids[0][0]]
    pixel_values = torch.cat(processed_images, dim=0)
    image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

from transformers.models.qwen3_vl import Qwen3VLProcessor
processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


urls = ["https://img.wort.lu/public/luxemburg/vfka4n-picture-title-binary/alternates/ONE_ONE_256/Picture%20title%20binary",
        "https://www.cartell.ie/car_check/wp-content/uploads/2012/03/Nissan-Micra-_4b.jpg"]

expected_outputs = ["This is a Ferrari F40, a legendary sports car produced by Ferrari from 1987 to 1992. It is renowned for its sleek design and powerful performance, making it one of the most iconic cars in automotive history.",
                    "This is a Nissan Micra, a compact car produced by the Japanese automaker Nissan. The Micra is a popular and affordable car, known for its reliability and efficiency.\n\nThe Nissan Micra was first introduced in 1990 as a small, affordable car. It was designed to compete with other small cars in the market, and it quickly gained popularity due to its fuel efficiency and low cost.\n\nThe Micra was produced in several different versions, including the 1.0L and 1.3L engines, which were available in different configurations. The Micra was also available with different body styles, including the standard"]

prompts = ["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this?<|im_end|>\n<|im_start|>assistant\n",
           "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nTell me the history of this car<|im_end|>\n<|im_start|>assistant\n"]

for url, expected_output, prompt in zip(urls, expected_outputs, prompts):
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    text_inputs = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    
    image_inputs = preprocess(processor.image_processor, images=image, return_tensors="pt")

    merge_size = processor.image_processor.merge_size  # usually 2
    image_grid_thw = image_inputs["image_grid_thw"]  # [batch, 3] -> [t, h, w]
    num_image_tokens = (image_grid_thw.prod(dim=-1) / (merge_size ** 2)).item()

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_token_positions = [i for i, tid in enumerate(text_inputs) if tid == image_token_id]

    for pos in reversed(image_token_positions):  # reversed to maintain indices
        text_inputs[pos:pos+1] = [image_token_id] * int(num_image_tokens)

    mm_token_type_ids = [0] * len(text_inputs)
    for pos in image_token_positions: mm_token_type_ids[pos:pos + int(num_image_tokens)] = [1] * int(num_image_tokens)

    outputs = _sample(model=model, input_ids=torch.tensor([text_inputs]), _pad_token_tensor=151643, past_key_values=DynamicCache({}), pixel_values=image_inputs['pixel_values'],
            position_ids=torch.arange(torch.tensor([text_inputs]).shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1), image_grid_thw=image_inputs['image_grid_thw'])

    #outputs = model.generate(**inputs, max_new_tokens=128)
    generated_ids = outputs[0][len(text_inputs):]
    output = processor.decode(generated_ids, skip_special_tokens=True)
    print(output)
    assert output == expected_output
