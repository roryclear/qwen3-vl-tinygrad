from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
from PIL import Image
import requests
from io import BytesIO
import torch
from torch import nn

set_seed(42)

def _sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor,
    stopping_criteria,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_forward = (
        self.get_compiled_call(generation_config.compile_config)
        if model._valid_auto_compile_criteria(model_kwargs, generation_config)
        else model.__call__
    )

    prefill_consumed = False
    outputs = model._prefill(
        input_ids,
        generation_config,
        model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if prefill_consumed:
            next_sequence_length = 1 if model_kwargs["use_cache"] else None
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, next_sequence_length=next_sequence_length, **model_kwargs
            )
            with model._optimize_model_for_decode():
                outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed = True
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        cache = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
    else:
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
    **kwargs,
):
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

inputs = {
    'input_ids': torch.tensor([input_ids]),
    'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long),
    'mm_token_type_ids': torch.tensor([mm_token_type_ids]),
    'pixel_values': image_inputs['pixel_values'],
    'image_grid_thw': image_inputs['image_grid_thw']
}

print("inputs =", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})

outputs = generate(**inputs, model=model, max_new_tokens=128)
#outputs = model.generate(**inputs, max_new_tokens=128)
generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
output = processor.decode(generated_ids, skip_special_tokens=True)
print(output)
assert output == "This is a Ferrari F40, a legendary sports car produced by Ferrari from 1987 to 1992. It is renowned for its sleek design and powerful performance, making it one of the most iconic cars in automotive history."