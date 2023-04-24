"""Gist compression demo."""

from typing import Optional

import fire
import torch
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer

from . import gist_llama, gist_t5, weight_diff
from .gist_llama import GistLlamaForCausalLM
from .gist_t5 import GistT5ForConditionalGeneration


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string.

    https://stackoverflow.com/a/31631711/2980246
    """
    B = float(B)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if B < KB:
        return "{0} {1}".format(B, "Bytes" if 0 == B > 1 else "Byte")
    elif KB <= B < MB:
        return "{0:.2f} KB".format(B / KB)
    elif MB <= B < GB:
        return "{0:.2f} MB".format(B / MB)
    elif GB <= B < TB:
        return "{0:.2f} GB".format(B / GB)
    elif TB <= B:
        return "{0:.2f} TB".format(B / TB)


@torch.inference_mode()
def main(
    model_name_or_path: str,
    instruction: str,
    input: str = "",
    num_gist_tokens: Optional[int] = 1,
    cache_dir: str = ".cache",
    precision: str = "fp32",
    max_new_tokens: int = 512,
    base_llama_path: Optional[str] = None,
) -> None:
    """Decode from a model with gist compression.

    Args:
        model_name_or_path: The model to load. MUST BE A GIST MODEL.
        instruction: The instruction to be compressed (required).
        input: The input for the instruction (optional). Will not be compressed
            or cached.
        num_gist_tokens: number of gist tokens to compress to. This should
            match the number of gist tokens the model was trained on.
        cache_dir: Hugging Face cache dir.
        precision: Precision to load the model in. Recommend fp32 or bf16 to
            save space (not fp16).
        max_new_tokens: Maximum number of new tokens to decode.
        base_llama_path: Any LLaMA model loaded from Hugging Face
            (jayelm/llama-7b-{gist,pos_control,neg_control}-1) is a weight
            diff, not the full model. If loading one of the Hugging Face LLaMA
            models, use this argument to specify the path to the raw LLaMA model.
    """
    is_llama = "llama" in model_name_or_path.lower()
    is_t5 = "t5" in model_name_or_path.lower()

    # Load config
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    # Load model
    print(f"Loading model {model_name_or_path}")
    if is_t5:
        model_cls = GistT5ForConditionalGeneration
    elif is_llama:
        model_cls = GistLlamaForCausalLM
    else:
        raise ValueError(f"Model type {model_name_or_path} not supported")

    if model_name_or_path in {
        "jayelm/llama-7b-gist-1",
        "jayelm/llama-7b-pos_control-1",
        "jayelm/llama-7b-neg_control-1",
    }:
        # Load with weight diff file
        if base_llama_path is None:
            raise ValueError(
                f"{model_name_or_path} is a weight diff huggingface repo. "
                "You must specify a `base_llama_path` for this to work."
            )
        else:
            print("Weight diff detected. Applying to original model...")
        model, _ = weight_diff.recover(
            path_raw=base_llama_path,
            path_diff=model_name_or_path,
            test_inference=False,
            cache_dir=cache_dir,
        )
    else:
        model = model_cls.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
        )

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float,
    }
    model = model.to(dtypes[precision]).cuda().eval()

    # Load tokenizer. It must already have gist token defined.
    print("Loading tokenizer")
    if is_llama:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        assert len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 1
        assert model.lm_head.weight.shape[0] == gist_llama.PRETRAINED_VOCAB_SIZE + 1
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        assert len(tokenizer) == gist_t5.PRETRAINED_VOCAB_SIZE + 1
        assert model.shared.weight.shape[0] == gist_t5.PRETRAINED_VOCAB_SIZE + 1
    gist_token = tokenizer.additional_special_tokens_ids[-1]

    # Compress instruction
    print("Compressing instruction")
    gist_str = "<GIST>" * num_gist_tokens
    prepped_instruction = f"Instruction: {instruction}\n{gist_str}"
    instruction_input_ids = tokenizer.encode(prepped_instruction)
    if is_t5:
        instruction_input_ids = instruction_input_ids[:-1]  # Remove eos token
    instruction_input_ids_tensor = (
        torch.tensor(instruction_input_ids).unsqueeze(0).cuda()
    )
    gist_kwargs = {
        "input_ids": instruction_input_ids_tensor,
        "attention_mask": torch.ones_like(instruction_input_ids_tensor),
    }
    if is_llama:
        gist_kwargs["attention_mask_gist"] = torch.ones_like(
            instruction_input_ids_tensor
        )[None, None]
    gist_activations = model.get_gist_activations(
        gist_token=gist_token,
        num_gist_tokens=num_gist_tokens,
        **gist_kwargs,
    )

    # Prepare input. Input decoding must be done carefully: tokenizers will
    # tokenize things differently if the input is at the start of the string
    # (vs if it follows a gist token). The simplest thing to do to ensure
    # consistency is add a dummy gist token before the input, then remove it
    # from the input ids later.
    if is_t5:
        input_suffix = ""
    else:
        input_suffix = "\nOutput:"

    if input:
        prepped_input = f"<GIST>\nInput: {input}{input_suffix}"
        full_prompt = (
            f"Instruction: {instruction}\n{gist_str}\nInput: {input}{input_suffix}"
        )
    else:
        prepped_input = f"<GIST>{input_suffix}"
        full_prompt = f"Instruction: {instruction}\n{gist_str}{input_suffix}"

    input_ids = tokenizer.encode(prepped_input)
    # Trim off the gist token we added at the beginning.
    input_ids = input_ids[input_ids.index(gist_token) + 1 :]
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).cuda()
    attention_mask_with_gist = (
        torch.tensor([1] * (len(input_ids) + num_gist_tokens)).unsqueeze(0).cuda()
    )

    # Sanity check that tokenizing the full prompt is the same as tokenizing the
    # prepped instruction and prepped input separately.
    full_prompt_input_ids = tokenizer.encode(full_prompt)
    assert (
        full_prompt_input_ids == instruction_input_ids + input_ids
    ), "Got different results tokenizing the full prompt vs tokenizing instruction/input separately"

    print("Decoding from model")
    gen_kwargs = {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_with_gist,
    }
    if is_llama:
        gen_kwargs["attention_mask_gist"] = attention_mask_with_gist[None, None]
        gen_kwargs["past_key_values"] = gist_activations.past_key_values
        gen_kwargs["gist_offset"] = gist_activations.gist_indices
    else:
        gen_kwargs["gist_activations"] = gist_activations
    generated_tokens = model.generate(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **gen_kwargs,
    )
    output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    if is_llama:
        output = output[len(prepped_input) - 5 :]

    num_layers = len(gist_activations.past_key_values)

    # Compute size of KV cache per token.
    kv_cache_tensor = gist_activations.past_key_values[0][0]
    single_token_kv_cache_tensor = kv_cache_tensor[:, :, 0]
    single_token_kv_cache_mem = (
        single_token_kv_cache_tensor.element_size()
        * single_token_kv_cache_tensor.nelement()
    )
    kv_cache_size_per_token = 2 * num_layers * single_token_kv_cache_mem

    # Compute sizes of original vs gisted kv caches.
    orig_kv_cache_mem = len(instruction_input_ids) * kv_cache_size_per_token
    gist_kv_cache_mem = num_gist_tokens * kv_cache_size_per_token

    print(f"Instruction: {instruction}")
    print(
        f">>> Compressed into {num_gist_tokens} gist tokens (compression factor: {len(instruction_input_ids) / num_gist_tokens}x)"
    )
    print(
        f">>> {num_layers} layer KV cache, each layer has 2 tensors of shape {tuple(kv_cache_tensor.shape)}"
    )
    print(
        f">>> {precision} KV cache size reduced from {humanbytes(orig_kv_cache_mem)} to {humanbytes(gist_kv_cache_mem)}"
    )
    print(f"Input: {input}")
    print(f"Output: {output}")


if __name__ == "__main__":
    fire.Fire(main)
