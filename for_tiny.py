import math, cv2, time
import numpy as np
from tinygrad import Tensor, nn, TinyJit, Variable, dtypes
Tensor.manual_seed(42)
from tinygrad.nn.state import load_state_dict
from tinygrad.helpers import fetch
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from extra.models.llama import sample

TEMP = 0.7
TOP_K = 20
TOP_P = 0.8

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    ret = Tensor.cat(-x2, x1, dim=-1)
    return ret

class Qwen3VL():
  def __init__(self, size="2B"):
    self.max_context = 2000
    self.vis = Qwen3VLVis(size=size)
    self.lang, kv = Transformer.from_gguf(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/Qwen3VL-{size}-Instruct-F16.gguf"), self.max_context) # max context
    self.tok = SimpleTokenizer.from_gguf_kv(kv)
    self.start_pos = 0

  def preprocess(self, image, prompt):
    pixel_values, image_grid_thw = self.vis.preprocess_img(image=Tensor(image))
    image_grid_thw = image_grid_thw.numpy().tolist()
    text_inputs = self.tok.encode(prompt)
    image_token_id = 151655
    image_token_positions = [i for i, tid in enumerate(text_inputs) if tid == image_token_id]
    num_image_tokens = ((image_grid_thw[0]*image_grid_thw[1]*image_grid_thw[2]) / 4)
    for pos in reversed(image_token_positions): text_inputs[pos:pos+1] = [image_token_id] * int(num_image_tokens)
    seq_len=len(text_inputs)
    input_ids = Tensor([text_inputs])
    return pixel_values, input_ids, seq_len, image_grid_thw

  def prewarm(self, res, prompt):
    pixel_values, input_ids, seq_len, image_grid_thw = self.preprocess(image=np.random.randint(0, 256, size=res, dtype=np.uint8), prompt=prompt)
    for _ in range(2):
      self.vis.preprocess_img(image=Tensor.rand(res).cast(dtypes.uint8))
      self.prefill(pixel_values=pixel_values, input_ids=input_ids, image_grid_thw=image_grid_thw)
      self.lang(tokens=Tensor([[42]]).clone(), start_pos=Variable("pos",1,self.max_context).bind(seq_len), temperature=Tensor(0.7).clone())
      self.lang.prefill_jit(tokens=Tensor([[42]*self.max_context]).clone()[:, :Variable("len",1,self.max_context).bind(42)], \
      start_pos=Variable("pos",1,self.max_context).bind(42), temperature=Tensor(0.7).clone())

  def generate(self, prompt, image=None):
    if image is not None:
      pixel_values, input_ids, seq_len, image_grid_thw = self.preprocess(image=image, prompt=prompt)
      self.start_pos = seq_len
      token = self.prefill(pixel_values=pixel_values, input_ids=input_ids, image_grid_thw=image_grid_thw)
    else:
      prompt = self.tok.encode(prompt)
      prompt_len = len(prompt)
      prompt = prompt + [0] * (self.max_context - prompt_len)
      tokens = Tensor(prompt).unsqueeze(0)
      token = self.lang.prefill_jit(tokens=tokens[:, :Variable("len",1,self.max_context).bind(prompt_len)], start_pos=Variable("pos",1,self.max_context).bind(self.start_pos), temperature=Tensor(0.7).clone())[0]
      self.start_pos += prompt_len
    toks_out = []
    while True:
      ts = time.time()
      if toks_out:
        token = self.lang(tokens=next_token_tensor.clone(), start_pos=Variable("pos",1,self.max_context).bind(self.start_pos), temperature=Tensor(0.7).clone())[0]
        self.start_pos += 1
      next_token = int(token.numpy()[0])
      next_token_tensor = Tensor([[next_token]])
      if next_token == 151645: break
      toks_out.append(next_token)
      print(self.tok.decode(toks_out))
      print(f"TOK/S = {1 / (time.time() - ts):.2f}")
    return self.tok.decode(toks_out)

  @TinyJit
  def prefill(self, pixel_values, input_ids, image_grid_thw):
    image_embeds, hidden_states, deepstack_feature_lists = self.vis(pixel_values, image_grid_thw)
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
    
    # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modular_qwen3_vl.py#L626
    # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modular_qwen3_vl.py#L543
    for i in range(len(self.lang.blk)):
      self.lang.blk[i]._init_state(Tensor.zeros(1, 1))
      hidden_states = self.lang.blk[i](hidden_states, start_pos=0)
      # https://github.com/huggingface/transformers/blob/08692e3c31654e4825b4c078a3c70b86efa70a46/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L692
      if i in self.vis.v.deepstack_idx:
        hidden_states = deepstack_process(hidden_states=hidden_states, visual_pos_masks=image_mask.squeeze(0), visual_embeds=(deepstack_feature_lists[self.vis.v.deepstack_idx.index(i)])).unsqueeze(0)

    hidden_states = self.lang.output_norm(hidden_states)
    outputs = hidden_states[:, -1:, :] @ self.lang.token_embd.weight.T

    next_token_logits = outputs[:, -1, :]
    scores = next_token_logits / TEMP
    token = sample(scores[0], temp=TEMP, k=TOP_K, p=TOP_P, af=None, ap=None)
    return token

def deepstack_process(hidden_states, visual_pos_masks, visual_embeds):
  mask_float = visual_pos_masks.any(axis=1)
  positions = mask_float.cumsum(axis=0) - 1
  positions = positions.clamp(0)
  expanded = visual_embeds[positions]
  expanded = expanded * mask_float.unsqueeze(-1)
  return hidden_states[0] + expanded

class Qwen3VLVis():
  def __init__(self, size="2B"):
    kv, state_dict = gguf_load(fetch(f"https://huggingface.co/Qwen/Qwen3-VL-{size}-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-{size}-Instruct-F16.gguf"))
    self.merge_size = kv["clip.vision.spatial_merge_size"]
    self.patch_size = kv["clip.vision.patch_size"]
    self.v = Qwen3VisBlocks(kv=kv, weights=state_dict)
    self.mm = [nn.Linear(*state_dict["mm.0.weight"].shape[::-1], bias=True), None, nn.Linear(*state_dict["mm.2.weight"].shape[::-1], bias=True)]
    state_dict["v.patch_embd.weight1"] = state_dict["v.patch_embd.weight.1"]
    load_state_dict(self, state_dict)
    self.inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, 32, 2, dtype=dtypes.float) / 32))

  def __call__(self, pixel_values, image_grid_thw):
    hidden_states = pixel_values.view(-1, 3, 2, 16, 16)
    B, C, D, H, W = hidden_states.shape
    x = hidden_states.reshape(B, C * D, H, W)
    w = Tensor.stack(self.v.patch_embd.weight, self.v.patch_embd.weight1, dim=2)
    out_C, in_C, kD, kH, kW = w.shape
    w2d = w.reshape(out_C, in_C * kD, kH, kW)

    hidden_states = x.conv2d(
        weight=w2d,
        bias=self.v.patch_embd.bias,
        stride=(16, 16),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1
    )

    hidden_states = hidden_states.view(-1, 1024)
        
    grid_ts = image_grid_thw[0]
    grid_hs = image_grid_thw[1]
    grid_ws = image_grid_thw[2]

    h_idxs = Tensor.linspace(0, self.v.num_grid_per_side - 1, grid_hs)
    w_idxs = Tensor.linspace(0, self.v.num_grid_per_side - 1, grid_ws)

    h_idxs_floor = h_idxs.cast(dtypes.int32)
    w_idxs_floor = w_idxs.cast(dtypes.int32)
    h_idxs_ceil = (h_idxs_floor.int() + 1).clip(self.v.num_grid_per_side - 1)
    w_idxs_ceil = (w_idxs_floor.int() + 1).clip(self.v.num_grid_per_side - 1)
    dh = h_idxs - h_idxs_floor
    dw = w_idxs - w_idxs_floor

    base_h = h_idxs_floor * self.v.num_grid_per_side
    base_h_ceil = h_idxs_ceil * self.v.num_grid_per_side


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

    pos_embeds = self.v.position_embd(idx_tensor)
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

    rotary_pos_emb = (pos_ids.unsqueeze(-1) * self.inv_freq).flatten(1)

    sqlen, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(sqlen, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(sqlen, -1)
    emb = Tensor.cat(rotary_pos_emb, rotary_pos_emb, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    
    deepstack_feature_lists = []
    for i in range(len(self.v.blk)):
      hidden_states = self.v.blk[i](hidden_states, cos, sin)
      if i in self.v.deepstack_idx: deepstack_feature_lists.append(self.v.deepstack[i](hidden_states))

    image_embeds = self.v.post_ln(hidden_states)
    image_embeds = image_embeds.view(-1, 4096)
    image_embeds = self.mm[0](image_embeds)
    image_embeds = Tensor.gelu(image_embeds)
    image_embeds = self.mm[2](image_embeds)
    return image_embeds, hidden_states, deepstack_feature_lists

  @TinyJit
  def preprocess_img(self, image):
    image = image.permute(2, 0, 1)
    temporal_patch_size = 2
    height, width = image.shape[-2:]
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=self.patch_size * self.merge_size,
        min_pixels=65536,
        max_pixels=16777216,
    )
    image = image.unsqueeze(0).float()
    image = image.interpolate(size=(resized_height, resized_width))
    resized_height, resized_width = image.shape[-2:]
    patches = (image - 127.5) / 127.5
    batch_size, channel = patches.shape[:2]
    grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
    patches = patches.reshape(
        batch_size,
        channel,
        grid_h // self.merge_size,
        self.merge_size,
        self.patch_size,
        grid_w // self.merge_size,
        self.merge_size,
        self.patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    pixel_values = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(
            batch_size,
            grid_h * grid_w,
            channel * temporal_patch_size * self.patch_size * self.patch_size,
        )
    )[0]
    return pixel_values.cast(dtypes.bfloat16), Tensor([1, grid_h, grid_w])

# https://github.com/huggingface/transformers/blob/90e3c4fa7200a9c8bb9756bf7bf43381d10850c0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L62
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

class Qwen3PatchEmbed():
  def __init__(self, kv=None):
    self.weight = Tensor.zeros(kv["clip.vision.embedding_length"], 3, 16, 16)
    self.weight1 = Tensor.zeros(kv["clip.vision.embedding_length"], 3, 16, 16)
    self.bias = Tensor.zeros(kv["clip.vision.embedding_length"])
    
class Qwen3VisBlocks():
  def __init__(self, kv=None, weights=None):
    self.blk = []
    for _ in range(kv["clip.vision.block_count"]): self.blk.append(Qwen3VisBlock(kv, weights=weights))
    self.patch_embd = Qwen3PatchEmbed(kv=kv)
    self.num_grid_per_side = 48
    self.deepstack_layers = kv["clip.vision.is_deepstack_layers"]
    self.deepstack_idx = [i for i, val in enumerate(self.deepstack_layers) if val]
    self.deepstack = []
    for i in range(len(self.deepstack_layers)):
      if i in self.deepstack_idx:
        self.deepstack.append(DeepstackLayer(i, weights))
      else:
        self.deepstack.append(None)
    self.position_embd = nn.Embedding(*weights["v.position_embd.weight"].shape)
    self.post_ln = nn.LayerNorm(weights["v.post_ln.weight"].shape[0], eps=1e-6, elementwise_affine=True)

class DeepstackLayer:
  def __init__(self, index, weights):
    self.fc1 = nn.Linear(*weights[f"v.deepstack.{index}.fc1.weight"].shape[::-1])
    self.fc2 = nn.Linear(*weights[f"v.deepstack.{index}.fc2.weight"].shape[::-1])
    self.norm = nn.LayerNorm(weights[f"v.deepstack.{index}.norm.weight"].shape[0], eps=1e-6, elementwise_affine=True)
    self.hidden_size = weights[f"v.deepstack.{index}.norm.weight"].shape[0]

  #https://github.com/huggingface/transformers/blob/027d1a97025295a1346c2eb5c361259e69eedfe7/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L112
  def __call__(self, hidden_states):
      deepstack_feature = (hidden_states.view(-1, self.hidden_size)).view(-1, self.hidden_size)
      return self.fc2(Tensor.gelu(self.fc1(deepstack_feature)))

class Qwen3VisBlock():
  def __init__(self, kv=None, weights=None):
    self.ffn_up = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.feed_forward_length"])
    self.ffn_down = nn.Linear(kv["clip.vision.feed_forward_length"], kv["clip.vision.embedding_length"])
    self.ln1 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.ln2 = nn.LayerNorm(kv["clip.vision.embedding_length"], eps=1e-6, elementwise_affine=True)
    self.attn_out = nn.Linear(kv["clip.vision.embedding_length"], kv["clip.vision.embedding_length"])
    self.attn_qkv = nn.Linear(*weights["v.blk.0.attn_qkv.weight"].shape[::-1])
  
  def __call__(self, hidden_states, cos, sin):
    hidden_states_input = self.ln1(hidden_states)
    seq_length = hidden_states_input.shape[0]
    qkv = self.attn_qkv(hidden_states_input)
    qkv_reshaped = qkv.reshape(seq_length, 3, 16, -1)
    qkv_permuted = qkv_reshaped.permute(1, 0, 2, 3)
    query, key, value = qkv_permuted.chunk(3, dim=0)
    query = query.squeeze(0)
    key   = key.squeeze(0)
    value = value.squeeze(0)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)

    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)

    attn_weight = query @ key.transpose(-2, -1) * 0.125
    attn_weight = Tensor.softmax(attn_weight)
    attn_output = attn_weight @ value
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.attn_out(attn_output)
    hidden_states += attn_output
    norm = self.ln2(hidden_states)
    x = self.ffn_up(norm)
    x = Tensor.gelu(x)
    norm = self.ffn_down(x)
    return hidden_states + norm