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
from tinygrad import Tensor, nn as tiny_nn
import math
import copy
from functools import partial
from torchvision.transforms.v2 import functional as tvF
from dataclasses import dataclass, fields
import typing
import sys

class SimpleKVCache:
    def __init__(self):
        self.cache = {}

    def update(self, key, value, layer_idx):
        if layer_idx not in self.cache:
            self.cache[layer_idx] = (key, value)
        else:
            past_key, past_value = self.cache[layer_idx]
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
            self.cache[layer_idx] = (key, value)
        return key, value

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
    tiny_model,
    input_ids,
    _pad_token_tensor,
    pixel_values,
    past_key_values,
    position_ids,
    image_grid_thw,
    expected # todo for testing
):
    toks_out = [] # todo for testing
    pad_token_id = _pad_token_tensor
    scores = None
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.int32)

    prefill_consumed = False

    inputs_embeds = model.model.language_model.embed_tokens.weight[input_ids] # todo indexing not in tinygrad!
    position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1)

    hidden_states = model.model.visual.patch_embed(pixel_values)

    grid_thw_list = image_grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, model.model.visual.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, model.model.visual.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=model.model.visual.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=model.model.visual.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * model.model.visual.num_grid_per_side
        base_h_ceil = h_idxs_ceil * model.model.visual.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.int32)
    weight_tensor = torch.tensor(weight_list, dtype=model.model.visual.pos_embed.weight.dtype)
    idx_tensor = to_tiny(idx_tensor)
    pos_embeds = tiny_model.model.visual.pos_embed(idx_tensor)
    pos_embeds = to_torch(pos_embeds)
    pos_embeds *= weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    patch_pos_embeds_permute = []
    merge_size = model.model.visual.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    pos_embeds = torch.cat(patch_pos_embeds_permute)


    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = model.model.visual.rot_pos_emb(image_grid_thw)

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
    for i in range(len(model.model.visual.blocks)):
        
        hidden_states_input = model.model.visual.blocks[i].norm1(hidden_states)
        seq_length = hidden_states_input.shape[0]
        query, key, value = (
            model.model.visual.blocks[i].attn.qkv(hidden_states_input).reshape(seq_length, 3, model.model.visual.blocks[i].attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        orig_q_dtype = query.dtype
        orig_k_dtype = key.dtype
        query, key = query.float(), key.float()
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q_embed = (query * cos) + (rotate_half(query) * sin)
        k_embed = (key * cos) + (rotate_half(key) * sin)
        query = q_embed.to(orig_q_dtype)
        key = k_embed.to(orig_k_dtype)

        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)


        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros(L, S, dtype=key.dtype)
        attn_weight = query @ key.transpose(-2, -1) * model.model.visual.blocks[i].attn.scaling
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0, train=True)
        attn_output = attn_weight @ value
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = model.model.visual.blocks[i].attn.proj(attn_output)

        hidden_states += attn_output
        hidden_states = hidden_states + model.model.visual.blocks[i].mlp(model.model.visual.blocks[i].norm2(hidden_states))

        if i in model.model.visual.deepstack_visual_indexes:
            layer = model.model.visual.deepstack_merger_list[model.model.visual.deepstack_visual_indexes.index(i)]
            deepstack_feature = layer.norm(hidden_states.view(-1, layer.hidden_size)).view(-1, layer.hidden_size)
            deepstack_feature = layer.linear_fc2(layer.act_fn(layer.linear_fc1(deepstack_feature)))
            deepstack_feature_lists.append(deepstack_feature)

    
    image_embeds = model.model.visual.merger.norm(hidden_states)
    image_embeds = image_embeds.view(-1, model.model.visual.merger.hidden_size)
    image_embeds = model.model.visual.merger.linear_fc1(image_embeds)
    image_embeds = F.gelu(image_embeds)
    image_embeds = model.model.visual.merger.linear_fc2(image_embeds)

    image_mask, _ = model.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
    inputs_embeds[image_mask] = image_embeds.view(-1)
    image_mask = image_mask[..., 0]

    hidden_states = inputs_embeds

    pos_id = position_ids[1:]
    inv_freq_expanded = model.model.language_model.rotary_emb.inv_freq[None, None, :, None].float().expand(3, pos_id.shape[1], -1, 1)
    position_ids_expanded = pos_id[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

    freqs_t = freqs[0]  # just overwrite the first dimension T
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = model.model.language_model.rotary_emb.mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    freqs = freqs_t

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * model.model.language_model.rotary_emb.attention_scaling
    sin = emb.sin() * model.model.language_model.rotary_emb.attention_scaling
    position_embeddings = cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)

    for i in range(len(model.model.language_model.layers)): # todo same block above
        residual = hidden_states

        hidden_states = to_tiny(hidden_states)
        hidden_states = tiny_model.model.language_model.layers[i].input_layernorm(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_states = to_torch(hidden_states)

        hidden_shape = (*input_shape, -1, model.model.language_model.layers[i].self_attn.head_dim)

        hidden_states = to_tiny(hidden_states)
        query = tiny_model.model.language_model.layers[i].self_attn.q_proj(hidden_states).view(hidden_shape)
        key = tiny_model.model.language_model.layers[i].self_attn.k_proj(hidden_states).view(hidden_shape)

        query = tiny_model.model.language_model.layers[i].self_attn.q_norm(query).transpose(1, 2)
        key = tiny_model.model.language_model.layers[i].self_attn.k_norm(key).transpose(1, 2)
        query = to_torch(query)
        key = to_torch(key)

        value = tiny_model.model.language_model.layers[i].self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        value = to_torch(value)

        cos, sin = position_embeddings
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        key, value = past_key_values.update(key, value, i)
    
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query.dtype)

        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * model.model.language_model.layers[i].self_attn.scaling
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0, train=True)
        attn_output = attn_weight @ value

        attn_output = attn_output.transpose(1, 2).contiguous()


        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = to_tiny(attn_output)
        hidden_states = tiny_model.model.language_model.layers[i].self_attn.o_proj(attn_output)

        hidden_states = to_torch(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = to_tiny(hidden_states)
        hidden_states = tiny_model.model.language_model.layers[i].post_attention_layernorm(hidden_states)
        
        
        gate = tiny_model.model.language_model.layers[i].mlp.gate_proj(hidden_states)
        up = tiny_model.model.language_model.layers[i].mlp.up_proj(hidden_states)
        activated = tinyTensor.silu(gate)
        combined = activated * up
        hidden_states = tiny_model.model.language_model.layers[i].mlp.down_proj(combined)
        hidden_states = to_torch(hidden_states)
        hidden_states = residual + hidden_states
   
        if i < len(deepstack_feature_lists): hidden_states[image_mask, :] += deepstack_feature_lists[i]
    
    hidden_states = to_tiny(hidden_states)
    hidden_states = tiny_model.model.language_model.norm(hidden_states)

    outputs = tiny_model.lm_head(hidden_states[:, -1:, :])

    outputs = to_torch(outputs)
    hidden_states = to_torch(hidden_states)

    while not this_peer_finished:
        if prefill_consumed:
            inputs_embeds = model.model.get_input_embeddings()(input_ids[:, -1:])

            hidden_states = inputs_embeds

            position_embeddings = model.model.language_model.rotary_emb(hidden_states, position_ids[1:])

            hidden_states = to_tiny(hidden_states)
            # decoder layers
            for i in range(len(tiny_model.model.language_model.layers)):        
                residual = hidden_states
                hidden_states = tiny_model.model.language_model.layers[i].input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, model.model.language_model.layers[i].self_attn.head_dim)
                
                query = tiny_model.model.language_model.layers[i].self_attn.q_proj(hidden_states).view(hidden_shape)
                key = tiny_model.model.language_model.layers[i].self_attn.k_proj(hidden_states).view(hidden_shape)
                query = tiny_model.model.language_model.layers[i].self_attn.q_norm(query).transpose(1, 2)
                key = tiny_model.model.language_model.layers[i].self_attn.k_norm(key).transpose(1, 2)
                query = to_torch(query)
                key = to_torch(key)
        
                value = tiny_model.model.language_model.layers[i].self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                hidden_states = to_torch(hidden_states)

                cos, sin = position_embeddings
                query = (query * cos) + (rotate_half(query) * sin)
                key = (key * cos) + (rotate_half(key) * sin)
                
                value = to_torch(value)
                key, value = past_key_values.update(key, value, i)   

                key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

                attn_weight = query @ key.transpose(-2, -1) * model.model.language_model.layers[i].self_attn.scaling

                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_output = attn_weight @ value


                attn_output = attn_output.transpose(1, 2).contiguous()

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()

                attn_output = to_tiny(attn_output)

                hidden_states = tiny_model.model.language_model.layers[i].self_attn.o_proj(attn_output)                
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = tiny_model.model.language_model.layers[i].post_attention_layernorm(hidden_states)
                gate = tiny_model.model.language_model.layers[i].mlp.gate_proj(hidden_states)
                up = tiny_model.model.language_model.layers[i].mlp.up_proj(hidden_states)
                activated = tinyTensor.silu(gate)
                combined = activated * up
                hidden_states = tiny_model.model.language_model.layers[i].mlp.down_proj(combined)
                hidden_states = residual + hidden_states

            hidden_states = to_torch(hidden_states)
            hidden_states = model.model.language_model.norm(hidden_states)

            hidden_states = to_tiny(hidden_states)
            outputs = tiny_model.lm_head(hidden_states[:, -1:, :])
            #hidden_states = to_torch(hidden_states)
            outputs = to_torch(outputs)

        prefill_consumed = True
        position_ids = position_ids[..., -1:] + 1
        
        temp = 0.7
        top_k = 20
        filter_value = -math.inf
        min_tokens_to_keep = 1
        top_p = 0.8

        next_token_logits = outputs[:, -1, :].to(copy=True, dtype=torch.float32)
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

        toks_out.append(int(input_ids[0][-1]))
        print(toks_out,expected[:len(toks_out)])
        if not input_ids[0][-1] == 151645: assert toks_out == expected[:len(toks_out)]
        this_peer_finished = input_ids[0][-1] == 151645 or len(input_ids[0]) == 406
        del outputs

    return input_ids


def smart_resize(height, width, factor, min_pixels, max_pixels):
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

def _preprocess(images):
    patch_size=16
    merge_size=2
    rescale_factor=0.00392156862745098
    temporal_patch_size=2

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

    rescale_factor = 0.00392156862745098
    image_mean = torch.tensor((0.5, 0.5, 0.5)) / rescale_factor
    image_std = torch.tensor((0.5, 0.5, 0.5)) / rescale_factor
    patches = tvF.normalize(stacked_images.to(dtype=torch.float32), image_mean, image_std)

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
    image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.int32)

    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


from tinygrad import Tensor as tinyTensor
tinyTensor.manual_seed(42)
from torch import Tensor
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import dtypes

class Qwen3VLTextRMSNorm_tiny():
    def __init__(self, size):
        self.variance_epsilon = 1e-06
        self.weight = tinyTensor.zeros(size)

    def __call__(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * tinyTensor.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
    

if __name__ == "__main__":
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


    def to_tiny(x):
       if x.dtype == torch.bfloat16: return tinyTensor(x.detach().to(torch.float16).numpy()).cast(dtypes.bfloat16)
       return tinyTensor(x.detach().numpy())

    def to_torch(x):
      if x.dtype == dtypes.bfloat16: return torch.tensor(x.numpy(), dtype=torch.bfloat16)
      return torch.tensor(x.numpy())

    class blank: pass

    tiny_weights = safe_load(fetch(f'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors'))

    print(model)
    print(model.model.visual.pos_embed)
    print(model.model.visual.pos_embed.weight.shape)

    tiny_model = blank()
    tiny_model.model = blank()
    tiny_model.model.visual = blank()
    tiny_model.model.visual.pos_embed = tiny_nn.Embedding(2304, 1024)
    tiny_model.model.language_model = blank()
    tiny_model.model.language_model.layers = []

    tiny_model.model.visual.pos_embed.weight.cast(dtypes.bfloat16)
    print(tiny_model.model.visual.pos_embed.weight.dtype)
    print(len(model.model.language_model.layers))
    print(model.model.visual.pos_embed.weight.dtype)
    print(model.model.language_model.layers[0].input_layernorm)
    print(model.model.language_model.layers[0].input_layernorm.weight.shape, model.model.language_model.layers[0].input_layernorm.variance_epsilon)
    # todo
    tiny_model.model.language_model.norm = Qwen3VLTextRMSNorm_tiny(size=2048)
    for i in range(len(model.model.language_model.layers)):
      tiny_model.model.language_model.layers.append(blank())
      tiny_model.model.language_model.layers[i].self_attn = blank()
      tiny_model.model.language_model.layers[i].self_attn.q_proj = tiny_nn.Linear(2048, 2048, bias=False)
      tiny_model.model.language_model.layers[i].self_attn.k_proj = tiny_nn.Linear(2048, 1024, bias=False)
      tiny_model.model.language_model.layers[i].self_attn.v_proj = tiny_nn.Linear(2048, 1024, bias=False)
      tiny_model.model.language_model.layers[i].self_attn.o_proj = tiny_nn.Linear(2048, 2048, bias=False)
      tiny_model.model.language_model.layers[i].self_attn.q_norm = Qwen3VLTextRMSNorm_tiny(size=128)
      tiny_model.model.language_model.layers[i].self_attn.k_norm = Qwen3VLTextRMSNorm_tiny(size=128)
      tiny_model.model.language_model.layers[i].input_layernorm = Qwen3VLTextRMSNorm_tiny(size=2048)
      tiny_model.model.language_model.layers[i].post_attention_layernorm = Qwen3VLTextRMSNorm_tiny(size=2048)
      tiny_model.model.language_model.layers[i].mlp = blank()
      tiny_model.model.language_model.layers[i].mlp.gate_proj = tiny_nn.Linear(2048, 6144, bias=False)
      tiny_model.model.language_model.layers[i].mlp.up_proj = tiny_nn.Linear(2048, 6144, bias=False)
      tiny_model.model.language_model.layers[i].mlp.down_proj = tiny_nn.Linear(6144, 2048, bias=False)
      tiny_model.model.language_model.embed_tokens = blank()
      tiny_model.model.language_model.embed_tokens.weight = tinyTensor.zeros(151936, 2048) # todo not used yet
    load_state_dict(tiny_model, tiny_weights)

    tiny_model.lm_head = tiny_nn.Linear(2048, 151936, bias=False)
    tiny_model.lm_head.weight = to_tiny(model.lm_head.weight) # todo how is this inited?

    #print(model.model.visual.pos_embed.bias) no bias
    #model.model.visual.pos_embed_tiny = to_tiny(model.model.visual.pos_embed)


    images = [Image.open(BytesIO(requests.get("https://img.wort.lu/public/luxemburg/vfka4n-picture-title-binary/alternates/ONE_ONE_256/Picture%20title%20binary").content)).convert("RGB"),
            Image.open(BytesIO(requests.get("https://www.cartell.ie/car_check/wp-content/uploads/2012/03/Nissan-Micra-_4b.jpg").content)).convert("RGB"),
            Image.open("test_img.jpg").convert("RGB")]
    expected_outputs = ["This is a Ferrari F40, a legendary sports car produced by Ferrari from 1987 to 1992. It is renowned for its sleek design and high performance, making it one of the most iconic cars in automotive history.",
                        "This is a Nissan Micra, a compact car produced by the Japanese automaker Nissan. The Micra is a popular and affordable car, known for its reliability and efficiency.\n\nThe Micra was first introduced in 1992 as a small, economical car. It was designed to be a practical and efficient vehicle for everyday use, and it quickly gained popularity in Japan and other markets.\n\nOver the years, the Micra has undergone several redesigns and updates. The most recent model, the 2019 version, features a more modern design and improved safety features.\n\nThe Micra is known for its low price, high",
                        "A person wearing a grey hoodie and light-colored pants is standing next to a silver car with the driver's side door open."]

    prompts = ["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nTell me the history of this car<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat has been detected on my CCTV camera? Write in one short sentence, only info about the object(s) detected.<|im_end|>\n<|im_start|>assistant\n"]

    import pickle
    tok = pickle.load(open("tok.pkl", "rb"))

    for image, expected_output, prompt in zip(images, expected_outputs, prompts):
        text_inputs = tok.encode(prompt)
        image = [tvF.pil_to_tensor(image)]
        image_inputs = _preprocess(images=image)
        merge_size = 2
        image_grid_thw = image_inputs["image_grid_thw"]  # [batch, 3] -> [t, h, w]
        num_image_tokens = (image_grid_thw.prod(dim=-1) / (merge_size ** 2)).item()

        image_token_id = 151655
        image_token_positions = [i for i, tid in enumerate(text_inputs) if tid == image_token_id]

        for pos in reversed(image_token_positions):  # reversed to maintain indices
            text_inputs[pos:pos+1] = [image_token_id] * int(num_image_tokens)

        mm_token_type_ids = [0] * len(text_inputs)
        for pos in image_token_positions: mm_token_type_ids[pos:pos + int(num_image_tokens)] = [1] * int(num_image_tokens)

        outputs = forward(model=model, tiny_model=tiny_model, input_ids=torch.tensor([text_inputs]), _pad_token_tensor=151643, past_key_values=SimpleKVCache(), pixel_values=image_inputs['pixel_values'],
                position_ids=torch.arange(torch.tensor([text_inputs]).shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1), image_grid_thw=image_inputs['image_grid_thw'], expected=tok.encode(expected_output))

        #outputs = model.generate(**inputs, max_new_tokens=128)
        generated_ids = outputs[0][len(text_inputs):]
        output = tok.decode(generated_ids.detach().numpy())
        output = output.replace("<|im_end|>","") # todo hack
        print(output)
        assert output == expected_output
