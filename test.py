import random
import numpy as np
from tinygrad import Tensor, nn, TinyJit, Variable
import math
import typing
import sys
import cv2
import time
from gguf import gguf_load
from model import TransformerBlock, TransformerConfig


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

temp = 0.7
top_k = 20
top_p = 0.8

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    ret = Tensor.cat(-x2, x1, dim=-1)
    return ret


def sample(logits, temp: float, k: int, p: float, af: float, ap: float):
  assert logits.ndim == 1, "only works on 1d tensors"
  assert 0 <= p <= 1, "p must be between 0 and 1"
  assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

  # if temperature is very low just use argmax
  if temp < 1e-6: return logits.argmax()

  # alpha sampling
  if af or ap:
    if not hasattr(sample, "alpha_counter"):
      setattr(sample, "alpha_counter", Tensor.zeros_like(logits, dtype=dtypes.int32).contiguous())
    logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

  # replace NaNs with -inf
  logits = (logits != logits).where(-float("inf"), logits)

  # softmax
  t = (logits / temp).softmax()

  counter, counter2 = Tensor.arange(t.numel(), device=logits.device).contiguous(), Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()
  # top k
  if k:
    output, output_indices = Tensor.zeros(k, device=logits.device).contiguous(), Tensor.zeros(k, device=logits.device, dtype=dtypes.int32).contiguous()
    for i in range(k):
      t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(dtypes.default_int)
      output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
      output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
      t = (counter == t_argmax).where(0, t)

    # approximate top p
    # because we are already limited to top k elements we can do top p "without sorting"
    output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
    output = (output_cumsum >= (1 - p)) * output
    output_indices = (output_cumsum >= (1 - p)) * output_indices

    # sample
    output_idx = output.multinomial()
    output_token = output_indices[output_idx]
  else:
    output_token = t.multinomial()

  # increase alpha counter
  if af or ap:
    sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

  return output_token

def preprocess(image):
  patch_size = 16
  merge_size = 2
  rescale_factor = 0.00392156862745098
  temporal_patch_size = 2
  height, width = image.shape[-2:]
  factor = patch_size * merge_size
  h_bar = round(height / factor) * factor
  w_bar = round(width / factor) * factor
  pixels = h_bar * w_bar
  beta = math.sqrt(max(1.0, pixels / 16777216, 65536 / pixels))
  h_bar = max(factor, round((h_bar / beta) / factor) * factor)
  w_bar = max(factor, round((w_bar / beta) / factor) * factor)

  resized_height, resized_width = h_bar, w_bar
  image_np = image.transpose(1, 2, 0)
  image_np = cv2.resize(image_np, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
  image = image_np.transpose(2, 0, 1)
  image = image.astype(np.float32)
  image_mean = np.array([0.5, 0.5, 0.5]) / rescale_factor
  image_std = np.array([0.5, 0.5, 0.5]) / rescale_factor
  image = (image - image_mean[:, None, None]) / image_std[:, None, None]

  channel = image.shape[0]
  grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
  
  patches = image.reshape(
      channel,
      grid_h // merge_size,
      merge_size,
      patch_size,
      grid_w // merge_size,
      merge_size,
      patch_size,
  )
  patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
  patches = np.expand_dims(patches, axis=4)
  patches = np.broadcast_to(patches, (*patches.shape[:4], temporal_patch_size, *patches.shape[5:]))
  flatten_patches = patches.reshape(
      grid_h * grid_w,
      channel * temporal_patch_size * patch_size * patch_size,
  )
  pixel_values = flatten_patches
  image_grid_thw = [1, grid_h, grid_w]
  return pixel_values.astype(np.float32), image_grid_thw

from tinygrad import Tensor
Tensor.manual_seed(42)
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import dtypes

class Qwen3VL():
  def __init__(self):
    self.vis = qwen3vl_vis()
    self.lang = qwen3vl_lang()
    self.past_keys = [Tensor.zeros(8, 500, 128).contiguous() for i in range(len(self.lang.blk))]
    self.past_values = [Tensor.zeros(8, 500, 128).contiguous() for i in range(len(self.lang.blk))]
    self.prewarm = False


  def forward(
      self,
      input_ids,
      pixel_values,
      image_grid_thw,
      seq_len,
      expected
  ):
    #if not self.prewarm:
    #  for _ in range(3):
    #    self.prefill(pixel_values=pixel_values, input_ids=input_ids, image_grid_thw=image_grid_thw, past_keys=self.past_keys, past_values=self.past_values, seq_len=seq_len)
    #    self.fwd(token=Tensor([[42]]).contiguous(), position_ids=Tensor(42).contiguous(), seq_len=Variable("pos",1,500).bind(seq_len), past_keys=self.past_keys, past_values=self.past_values)
    #    self.prewarm = True

    for i in range(len(self.lang.blk)):
      self.past_keys[i] *= 0
      self.past_values[i] *= 0

    toks_out = []
    prefill_done = False
    ts = time.time()
    position_ids, token = self.prefill(pixel_values=pixel_values, input_ids=input_ids, image_grid_thw=image_grid_thw, past_keys=self.past_keys, past_values=self.past_values, seq_len=seq_len)
    while True:
        if prefill_done:
          ts = time.time()
          position_ids, token = self.fwd(token=next_token_tensor.contiguous(), position_ids=position_ids.contiguous(), seq_len=Variable("pos",1,500).bind(seq_len), past_keys=self.past_keys, past_values=self.past_values)
          seq_len+=1
        else:
          prefill_done = True
        next_token = int(token.numpy()[0])
        next_token_tensor = Tensor([[next_token]])  # shape (1,1)

        toks_out.append(next_token)
        print(f"TOK/S = {1 / (time.time() - ts):.2f}")
        print(tok.decode(toks_out), "\n", tok.decode(expected[:len(toks_out)]), "\n")
        #assert tok.decode(toks_out).replace("<|im_end|>","") == tok.decode(expected[:len(toks_out)])
        if next_token == 151645 or seq_len == 406: break

    return toks_out


  @TinyJit
  def fwd(self, token, position_ids, seq_len, past_keys, past_values):
    hidden_states = self.lang.token_embd(token)
    freqs = self.lang.inv_freq * position_ids
    emb = Tensor.cat(freqs, freqs, dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    # decoder layers
    for i in range(len(self.lang.blk)): hidden_states = self.lang.blk[i](hidden_states, start_pos=seq_len)  
    hidden_states = self.lang.output_norm(hidden_states)
    outputs = self.lang.lm_head(hidden_states[:, -1:, :])
    next_token_logits = outputs[:, -1, :]
    scores = next_token_logits / temp
    token = sample(scores[0], temp=temp, k=top_k, p=top_p, af=None, ap=None)
    return position_ids + 1, token

  @TinyJit
  def prefill(self, pixel_values, input_ids, image_grid_thw, past_keys, past_values, seq_len):
      hidden_states = pixel_values.view(-1, 3, 2, 16, 16)
      hidden_states = hidden_states.cast(dtype=dtypes.bfloat16)

      B, C, D, H, W = hidden_states.shape
      x = hidden_states.reshape(B, C * D, H, W)
      w = Tensor.stack(self.vis.v.patch_embd.weight, self.vis.v.patch_embd.weight2, dim=2)
      out_C, in_C, kD, kH, kW = w.shape
      w2d = w.reshape(out_C, in_C * kD, kH, kW)

      hidden_states = x.conv2d(
          weight=w2d,
          bias=qwen.vis.v.patch_embd.bias,
          stride=(16, 16),
          padding=(0, 0),
          dilation=(1, 1),
          groups=1
      )

      hidden_states = hidden_states.view(-1, 1024)
          
      grid_ts = image_grid_thw[0]
      grid_hs = image_grid_thw[1]
      grid_ws = image_grid_thw[2]

      h_idxs = Tensor.linspace(0, self.vis.v.num_grid_per_side - 1, grid_hs)
      w_idxs = Tensor.linspace(0, self.vis.v.num_grid_per_side - 1, grid_ws)

      h_idxs_floor = h_idxs.cast(dtypes.int32)
      w_idxs_floor = w_idxs.cast(dtypes.int32)
      h_idxs_ceil = (h_idxs_floor.int() + 1).clip(self.vis.v.num_grid_per_side - 1)
      w_idxs_ceil = (w_idxs_floor.int() + 1).clip(self.vis.v.num_grid_per_side - 1)
      dh = h_idxs - h_idxs_floor
      dw = w_idxs - w_idxs_floor

      base_h = h_idxs_floor * self.vis.v.num_grid_per_side
      base_h_ceil = h_idxs_ceil * self.vis.v.num_grid_per_side


      idx_tensor = Tensor.stack(
          (base_h[None].T + w_idxs_floor[None]).flatten(),
          (base_h[None].T + w_idxs_ceil[None]).flatten(),
          (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
          (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
      ).cast(dtypes.int32)

      weight_tensor = Tensor.stack(
          ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
          ((1 - dh)[None].T * dw[None]).flatten(),
          (dh[None].T * (1 - dw)[None]).flatten(),
          (dh[None].T * dw[None]).flatten(),
      ).cast(dtypes.bfloat16)

      pos_embeds = self.vis.v.position_embd(idx_tensor)
      pos_embeds *= weight_tensor[:, :, None]
      patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

      patch_pos_embeds = patch_pos_embeds[:grid_hs * grid_ws]

      merge_size = 2
      pos_embeds = patch_pos_embeds.repeat(grid_ts, 1)
      pos_embeds = (pos_embeds.view(grid_ts, grid_hs // merge_size, merge_size, grid_ws // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
      hidden_states = hidden_states + pos_embeds
      
      hpos_ids = Tensor.arange(image_grid_thw[1]).unsqueeze(1).expand(-1, image_grid_thw[2])
      hpos_ids = hpos_ids.reshape(image_grid_thw[1] // merge_size, merge_size, image_grid_thw[2] // merge_size, merge_size).transpose(1, 2).flatten()

      wpos_ids = Tensor.arange(image_grid_thw[2]).unsqueeze(0).expand(image_grid_thw[1], -1)
      wpos_ids = wpos_ids.reshape(image_grid_thw[1] // merge_size, merge_size, image_grid_thw[2] // merge_size, merge_size).transpose(1, 2).flatten()

      pos_ids = Tensor.stack(hpos_ids, wpos_ids, dim=-1).repeat(image_grid_thw[0], 1)

      rotary_pos_emb = (pos_ids.unsqueeze(-1) * self.vis.inv_freq).flatten(1)

      sqlen, _ = hidden_states.size()
      hidden_states = hidden_states.reshape(sqlen, -1)
      rotary_pos_emb = rotary_pos_emb.reshape(sqlen, -1)
      emb = Tensor.cat(rotary_pos_emb, rotary_pos_emb, dim=-1)
      cos, sin = emb.cos(), emb.sin()
      cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
      
      for i in range(len(self.vis.v.blk)):
          hidden_states_input = self.vis.v.blk[i].ln1(hidden_states)
          seq_length = hidden_states_input.shape[0]
          qkv = self.vis.v.blk[i].attn_qkv(hidden_states_input)
          
          qkv_reshaped = qkv.reshape(seq_length, 3, 16, -1)

          qkv_permuted = qkv_reshaped.permute(1, 0, 2, 3)

          query, key, value = qkv_permuted.chunk(3, dim=0)
          query = query.squeeze(0)
          key   = key.squeeze(0)
          value = value.squeeze(0)

          query, key = query.cast(dtypes.float32), key.cast(dtypes.float32)
          query = (query * cos) + (rotate_half(query) * sin)
          key = (key * cos) + (rotate_half(key) * sin)

          query = query.transpose(0, 1).unsqueeze(0)
          key = key.transpose(0, 1).unsqueeze(0)
          value = value.transpose(0, 1).unsqueeze(0)

          query = query.contiguous()
          key = key.contiguous()
          value = value.contiguous()
          L, S = query.size(-2), key.size(-2)
          attn_weight = query @ key.transpose(-2, -1) * 0.125
          attn_weight = Tensor.softmax(attn_weight)
          attn_output = attn_weight @ value
          attn_output = attn_output.transpose(1, 2).contiguous()
          attn_output = attn_output.reshape(seq_length, -1).contiguous()
          attn_output = attn_output.cast(dtypes.bfloat16)
          attn_output = self.vis.v.blk[i].attn_out(attn_output)
          attn_output = attn_output.cast(dtypes.bfloat16) # todo
          hidden_states += attn_output
          norm = self.vis.v.blk[i].ln2(hidden_states)
          x = self.vis.v.blk[i].ffn_up(norm)
          x = Tensor.gelu(x)
          norm = self.vis.v.blk[i].ffn_down(x)
          hidden_states = hidden_states + norm
      
      image_embeds = self.vis.v.post_ln(hidden_states)
      image_embeds = image_embeds.view(-1, 4096)
      image_embeds = self.vis.mm[0](image_embeds)
      image_embeds = Tensor.gelu(image_embeds)
      image_embeds = self.vis.mm[2](image_embeds)
      
      image_mask = input_ids == 151655
      inputs_embeds = self.lang.token_embd(input_ids)
      image_mask = image_mask.unsqueeze(-1).expand(inputs_embeds.shape)
      image_embeds = image_embeds.view(-1)
      flat_mask = image_mask.view(-1)
      idx = (flat_mask.cumsum(0) - 1).clamp(0)
      expanded = image_embeds[idx] * flat_mask
      flat_inputs = inputs_embeds.view(-1)
      flat_inputs = flat_inputs * (~flat_mask) + expanded
      hidden_states = flat_inputs.view(inputs_embeds.shape)


      position_ids = Tensor.arange(input_ids.shape[-1]).unsqueeze(0).unsqueeze(0).repeat(4, 1, 1)
      
      for i in range(len(self.lang.blk)): # todo same block above
        self.lang.blk[i]._init_state(Tensor.zeros(1, 1))
        hidden_states = self.lang.blk[i](hidden_states, start_pos=0)


      hidden_states = self.lang.output_norm(hidden_states)
      outputs = self.lang.lm_head(hidden_states[:, -1:, :])

      position_ids = position_ids[0][0][-1] + 1
      next_token_logits = outputs[:, -1, :]
      scores = next_token_logits / temp
      token = sample(scores[0], temp=temp, k=top_k, p=top_p, af=None, ap=None)
      return position_ids, token

class Qwen3VLTextRMSNorm():
  def __init__(self, size):
    self.variance_epsilon = 1e-06
    self.weight = Tensor.zeros(size)

  def __call__(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * Tensor.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states

class qwen3vl_lang:
  def __init__(self):
    _, state_dict_language = gguf_load(fetch("https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-F16.gguf"))
    self.token_embd = nn.Embedding(vocab_size=151936, embed_size=2048)
    self.blk = []
    for _ in range(28):
      self.blk.append(TransformerBlock(config=TransformerConfig(num_blocks=28, dim=2048, hidden_dim=6144, n_heads=16, n_kv_heads=8, norm_eps=9.999999974752427e-07, vocab_size=151936, head_dim=128, rope_theta=5000000.0, rope_dim=128, v_head_dim=128, max_context=4096, qk_norm=128, num_experts=0, num_experts_per_tok=0, norm_topk_prob=False, q_lora_rank=0, kv_lora_rank=0, shared_expert_dim=0, full_attention_interval=0, attn_output_gate=False, ssm=None, shared_expert_gate=False, leading_dense_blocks=0, dense_hidden_dim=0, routed_scaling_factor=1.0, qkv_bias=False, expert_bias=False)))
    self.output_norm = Qwen3VLTextRMSNorm(size=2048)


    self.scaling = 0.08838834764831845
    self.key_length = 128
    self.mrope_section = [24, 20, 20]
    load_state_dict(self, state_dict_language)
    self.lm_head = nn.Linear(2048, 151936, bias=False)
    self.lm_head.weight = self.token_embd.weight
    self.inv_freq = 1.0 / (5000000 ** (Tensor.arange(0, 128, 2) / 128))

class qwen3_lang_block():
  def __init__(self):
    self.attn_k = nn.Linear(2048, 1024, bias=False)
    self.attn_q = nn.Linear(2048, 2048, bias=False)
    self.attn_v = nn.Linear(2048, 1024, bias=False)
    self.attn_output = nn.Linear(2048, 2048, bias=False)
    self.ffn_gate = nn.Linear(2048, 6144, bias=False)
    self.ffn_up = nn.Linear(2048, 6144, bias=False)
    self.ffn_down = nn.Linear(6144, 2048, bias=False)
    self.attn_k_norm = Qwen3VLTextRMSNorm(size=128)
    self.attn_q_norm = Qwen3VLTextRMSNorm(size=128)
    self.ffn_norm = Qwen3VLTextRMSNorm(size=2048)
    self.attn_norm = Qwen3VLTextRMSNorm(size=2048)

class qwen3vl_vis():
  def __init__(self):
    _, state_dict_visual = gguf_load(fetch("https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf"))
    self.v = qwen3_vis_v()
    self.mm = [nn.Linear(4096, 4096, bias=True), None, nn.Linear(4096, 2048, bias=True)]
    state_dict_visual["v.patch_embd.weight2"] = state_dict_visual["v.patch_embd.weight.1"] # todo
    load_state_dict(self, state_dict_visual)
    self.inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, 32, 2, dtype=dtypes.float) / 32))

class qwen3_patch_embd():
  def __init__(self):
    self.weight = Tensor.zeros(1024, 3, 16, 16)
    self.weight2 = Tensor.zeros(1024, 3, 16, 16)
    self.bias = Tensor.zeros(1024)
    
class qwen3_vis_v():
  def __init__(self):
    self.blk = []
    for _ in range(24): self.blk.append(qwen3_vis_block())
    self.patch_embd = qwen3_patch_embd()
    self.num_grid_per_side = 48
    self.position_embd = nn.Embedding(2304, 1024)
    self.post_ln = nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True)

class qwen3_vis_block():
  def __init__(self):
    self.ffn_up = nn.Linear(1024, 4096)
    self.ffn_down = nn.Linear(4096, 1024)
    self.ln1 = nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True)
    self.ln2 = nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True)
    self.attn_out = nn.Linear(1024, 1024)
    self.attn_qkv = nn.Linear(1024, 3072)
    
if __name__ == "__main__":
  qwen = Qwen3VL()

  # first three are all 256x256
  images = [
      cv2.cvtColor(cv2.imread("f40.jpeg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("gtr.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("yaris.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("micra.jpg"), cv2.COLOR_BGR2RGB),
      cv2.cvtColor(cv2.imread("test_img.jpg"), cv2.COLOR_BGR2RGB)
  ]

  expected_outputs = ["This is a Ferrari F40, a classic supercar known for its sleek design and powerful performance.",
                      "This is a Nissan GT-R, a high-performance sports car known for its powerful engine and sleek design.",
                      "This is a white 2023 Toyota Yaris, a compact hatchback with a sleek design and modern features.",
                      "The car shown in the image is the Nissan Micra, a compact car produced by Nissan. The Micra was first introduced in 1990 and has been a popular choice for its affordability, fuel efficiency, and reliability.\n\nThe Micra has undergone several generations, with the first generation being produced from 1990 to 1998. The second generation was introduced in 1998 and continued until 2005. The third generation was launched in 2005 and was produced until 2010. The fourth generation was introduced in 2010 and continued until",
                      "A person wearing a light green hoodie and light-colored pants is standing near a silver car with the driver's side door open."]

  prompts = ["<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? in one sentence<|im_end|>\n<|im_start|>assistant\n",
             "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? in one sentence<|im_end|>\n<|im_start|>assistant\n",
             "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat car is this? in one sentence<|im_end|>\n<|im_start|>assistant\n",
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nTell me the history of this car<|im_end|>\n<|im_start|>assistant\n",
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat has been detected on my CCTV camera? Write in one short sentence, only info about the object(s) detected.<|im_end|>\n<|im_start|>assistant\n"]

  import pickle
  tok = pickle.load(open("tok.pkl", "rb"))
  z = 0
  for image, expected_output, prompt in zip(images, expected_outputs, prompts):
    z+=1
    if z > 1: break
    text_inputs = tok.encode(prompt)

    image = image.transpose(2, 0, 1)
    pixel_values, image_grid_thw = preprocess(image=image)

    merge_size = 2
    num_image_tokens = ((image_grid_thw[0]*image_grid_thw[1]*image_grid_thw[2]) / (merge_size ** 2))

    image_token_id = 151655
    image_token_positions = [i for i, tid in enumerate(text_inputs) if tid == image_token_id]

    for pos in reversed(image_token_positions): text_inputs[pos:pos+1] = [image_token_id] * int(num_image_tokens)

    outputs = qwen.forward(input_ids=Tensor([text_inputs]), pixel_values=Tensor(pixel_values), image_grid_thw=image_grid_thw, expected=tok.encode(expected_output), seq_len=len(text_inputs))

    output = tok.decode(outputs)
    output = output.replace("<|im_end|>","") # todo hack
    print("output =",output)
    #assert output == expected_output


