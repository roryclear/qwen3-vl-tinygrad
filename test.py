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
import typing

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    ret = tokens + self._encode_sentence(text[pos:])
    return ret

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]:
    return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def forward(
    model,
    input_ids: torch.LongTensor = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values= None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None):
    inputs_embeds = model.language_model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1)
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
    for i in range(len(model.visual.blocks)):
        
        hidden_states_input = model.visual.blocks[i].norm1(hidden_states)
        seq_length = hidden_states_input.shape[0]
        query_states, key_states, value_states = (
            model.visual.blocks[i].attn.qkv(hidden_states_input).reshape(seq_length, 3, model.visual.blocks[i].attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        orig_q_dtype = query_states.dtype
        orig_k_dtype = key_states.dtype
        query_states, key_states = query_states.float(), key_states.float()
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q_embed = (query_states * cos) + (rotate_half(query_states) * sin)
        k_embed = (key_states * cos) + (rotate_half(key_states) * sin)
        query_states = q_embed.to(orig_q_dtype)
        key_states = k_embed.to(orig_k_dtype)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)


        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        L, S = query_states.size(-2), key_states.size(-2)
        attn_bias = torch.zeros(L, S, dtype=key_states.dtype, device=query_states.device)
        attn_weight = query_states @ key_states.transpose(-2, -1) * model.visual.blocks[i].attn.scaling
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0, train=True)
        attn_output = attn_weight @ value_states
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = model.visual.blocks[i].attn.proj(attn_output)

        hidden_states += attn_output
        hidden_states = hidden_states + model.visual.blocks[i].mlp(model.visual.blocks[i].norm2(hidden_states))

        if i in model.visual.deepstack_visual_indexes:
            layer = model.visual.deepstack_merger_list[model.visual.deepstack_visual_indexes.index(i)]
            deepstack_feature = layer.norm(hidden_states.view(-1, layer.hidden_size)).view(-1, layer.hidden_size)
            deepstack_feature = layer.linear_fc2(layer.act_fn(layer.linear_fc1(deepstack_feature)))
            deepstack_feature_lists.append(deepstack_feature)

    image_embeds = model.visual.merger(hidden_states)
    image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds[image_mask] = image_embeds.view(-1)
    image_mask = image_mask[..., 0]
    position_ids = position_ids[1:]

    hidden_states = inputs_embeds
    position_embeddings = model.language_model.rotary_emb(hidden_states, position_ids)
    for i in range(len(model.language_model.layers)): # todo same block above
        residual = hidden_states
        hidden_states = model.language_model.layers[i].input_layernorm(hidden_states)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, model.language_model.layers[i].self_attn.head_dim)

        query_states = model.language_model.layers[i].self_attn.q_norm(model.language_model.layers[i].self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = model.language_model.layers[i].self_attn.k_norm(model.language_model.layers[i].self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = model.language_model.layers[i].self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        key_states, value_states = past_key_values.update(key_states, value_states, i)
    
        L, S = query_states.size(-2), key_states.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query_states.dtype, device=query_states.device)

        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        key_states = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
        value_states = value_states.repeat_interleave(query_states.size(-3)//value_states.size(-3), -3)

        attn_weight = query_states @ key_states.transpose(-2, -1) * model.language_model.layers[i].self_attn.scaling
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0, train=True)
        attn_output = attn_weight @ value_states

        attn_output = attn_output.transpose(1, 2).contiguous()


        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        hidden_states = model.language_model.layers[i].self_attn.o_proj(attn_output)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = model.language_model.layers[i].post_attention_layernorm(hidden_states)
        hidden_states = model.language_model.layers[i].mlp(hidden_states)
        hidden_states = residual + hidden_states
   
        if i < len(deepstack_feature_lists): hidden_states[image_mask, :] += deepstack_feature_lists[i]
    hidden_states = model.language_model.norm(hidden_states)
    return hidden_states


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx

def _non_vmap_expansion_sdpa(
    batch_indices: torch.Tensor, head_indices: torch.Tensor, q_indices: torch.Tensor, kv_indices: torch.Tensor
):
    batch_indices = batch_indices[:, None, None, None]
    head_indices = head_indices[None, :, None, None]
    q_indices = q_indices[None, None, :, None]
    kv_indices = kv_indices[None, None, None, :]
    return batch_indices, head_indices, q_indices, kv_indices

def sdpa_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function=causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    allow_is_bidirectional_skip: bool = False,
    allow_torch_fix: bool = True,
    use_vmap: bool = False,
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.Tensor | None:
    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(q_length, device=device) + q_offset
    kv_arange = torch.arange(kv_length, device=device) + kv_offset


    attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange))
    attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)



    return attention_mask

def create_causal_mask(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values,
    position_ids: torch.Tensor | None = None,
    or_mask_function=None,
    and_mask_function=None,
    block_sequence_ids=None):

    q_offset = past_key_values.get_seq_length()
    kv_length = q_offset+1

    q_length = 1
    kv_offset = 0

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device

    use_vmap = False
    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)



    # We now create the mask
    causal_mask = sdpa_mask(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=causal_mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return causal_mask

def _prefill(
    model,
    input_ids,
    pixel_values,
    past_key_values,
    image_grid_thw):


    hidden_states = forward(model.model, pixel_values=pixel_values,
                        past_key_values=past_key_values,
                        image_grid_thw=image_grid_thw,
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

def forward2(
    model,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    # args for deepstack
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs):

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

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.rotary_emb(hidden_states, position_ids)

    # decoder layers
    for layer_idx, decoder_layer in enumerate(model.layers):
        hidden_states = decoder_layer( # this works with no mask????
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = model.norm(hidden_states)

    return hidden_states

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
            inputs_embeds = model.model.get_input_embeddings()(input_ids[:, -1:])
            hidden_states = forward2(model.model.language_model,
                input_ids=None,
                position_ids=position_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=None,
                deepstack_visual_embeds=None
            )
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


def preprocess(images):    
    images = [tvF.pil_to_tensor(images)]
    return _preprocess(images)


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
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

def rescale_and_normalize(
    images: "torch.Tensor",
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float],
    image_std: float | list[float],
) -> "torch.Tensor":
    rescale_factor = 0.00392156862745098
    image_mean = torch.tensor(image_mean) * (1.0 / rescale_factor)
    image_std = torch.tensor(image_std) * (1.0 / rescale_factor)
    images = tvF.normalize(images.to(dtype=torch.float32), image_mean, image_std)
    return images

def _preprocess(images):
    patch_size=16
    merge_size=2
    rescale_factor=0.00392156862745098
    do_normalize=True
    temporal_patch_size=2
    resample=3

    height, width = images[0].shape[-2:]
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=65536,
        max_pixels=16777216,
    )

    resized_images = tvF.resize(images[0].unsqueeze(0), (resized_height, resized_width), interpolation=3, antialias=True)


    stacked_images = resized_images[0].unsqueeze(0)
    resized_height, resized_width = stacked_images.shape[-2:]
    patches = rescale_and_normalize(stacked_images, True, rescale_factor, do_normalize, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    processed_images = [flatten_patches[0]]
    processed_grids_ordered = [[[1, grid_h, grid_w]][0]]

    pixel_values = torch.cat(processed_images, dim=0)
    image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


urls = ["https://img.wort.lu/public/luxemburg/vfka4n-picture-title-binary/alternates/ONE_ONE_256/Picture%20title%20binary",
        "https://www.cartell.ie/car_check/wp-content/uploads/2012/03/Nissan-Micra-_4b.jpg"]

expected_outputs = ["This is a Ferrari F40, a legendary sports car produced by Ferrari from 1987 to 1992. It is renowned for its sleek design and powerful performance, making it one of the most iconic cars in automotive history.",
                    "This is a Nissan Micra, a compact car produced by the Japanese automaker Nissan. The Micra is a popular and affordable car, known for its reliability and efficiency.\n\nThe Nissan Micra was first introduced in 1990 as a small, affordable car. It was designed to compete with other small cars in the market, and it quickly gained popularity due to its fuel efficiency and low cost.\n\nThe Micra was produced in several different versions, including the 1.0L and 1.3L engines, which were available in different configurations. The Micra was also available with different body styles, including the standard"]

prompts = ["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this?<|im_end|>\n<|im_start|>assistant\n",
           "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nTell me the history of this car<|im_end|>\n<|im_start|>assistant\n"]

import pickle
tok = pickle.load(open("tok.pkl", "rb"))

for url, expected_output, prompt in zip(urls, expected_outputs, prompts):
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

    text_inputs = tok.encode(prompt)

    image_inputs = preprocess(images=image)
    merge_size = 2
    image_grid_thw = image_inputs["image_grid_thw"]  # [batch, 3] -> [t, h, w]
    num_image_tokens = (image_grid_thw.prod(dim=-1) / (merge_size ** 2)).item()

    image_token_id = 151655
    image_token_positions = [i for i, tid in enumerate(text_inputs) if tid == image_token_id]

    for pos in reversed(image_token_positions):  # reversed to maintain indices
        text_inputs[pos:pos+1] = [image_token_id] * int(num_image_tokens)

    mm_token_type_ids = [0] * len(text_inputs)
    for pos in image_token_positions: mm_token_type_ids[pos:pos + int(num_image_tokens)] = [1] * int(num_image_tokens)

    outputs = _sample(model=model, input_ids=torch.tensor([text_inputs]), _pad_token_tensor=151643, past_key_values=DynamicCache({}), pixel_values=image_inputs['pixel_values'],
            position_ids=torch.arange(torch.tensor([text_inputs]).shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1), image_grid_thw=image_inputs['image_grid_thw'])

    #outputs = model.generate(**inputs, max_new_tokens=128)
    generated_ids = outputs[0][len(text_inputs):]
    output = tok.decode(generated_ids.detach().numpy())
    output = output.replace("<|im_end|>","") # todo hack
    print(output)
    assert output == expected_output


