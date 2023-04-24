#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
This file is modified from Stanford Alpaca's weight diff file.
https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py
"""


import warnings
from typing import Optional

import fire
import torch
import tqdm
import transformers
from torch import nn
from transformers import LlamaTokenizer

from .gist_llama import PRETRAINED_VOCAB_SIZE, GistLlamaForCausalLM

WEIGHT_SUMS = {
    "gist": 50547.9570,
    "pos_control": 50596.9219,
    "neg_control": 50503.8789,
}


@torch.no_grad()
def add_zero_embedding_(model: nn.Module, tokenizer: transformers.PreTrainedTokenizer):
    """Add an additional embedding to the model for gist tokens.

    Ensures that the embedding is zero, so that weight diff works
    correctly.
    """
    model.resize_token_embeddings(len(tokenizer))
    # Instead of setting new embedding to be average of existing
    # word embeddings, here, we set everything to 0, so this does
    # not interfere with the weight diff.
    model.model.embed_tokens.weight[-1] = 0
    model.lm_head.weight[-1] = 0


@torch.inference_mode()
def make_diff(
    path_raw: str,
    path_tuned: str,
    path_diff: str,
    device="cpu",  # "cuda" or "cpu"
):
    """Make the weight diff.

    This function is given to present full transparency of how the weight diff was created.

    Run:
        python weight_diff.py make_diff --path_raw <your_path_raw> --path_tuned <your_path_tuned> --path_diff <your_path_diff>
    """
    model_tuned: transformers.PreTrainedModel = GistLlamaForCausalLM.from_pretrained(
        path_tuned,
        torch_dtype=torch.float32,
        # Requires Accelerate
        # device_map={"": torch.device(device)},
        # low_cpu_mem_usage=True,
    )
    model_raw: transformers.PreTrainedModel = GistLlamaForCausalLM.from_pretrained(
        path_raw,
        torch_dtype=torch.float32,
        # Requires Accelerate
        # device_map={"": torch.device(device)},
        # low_cpu_mem_usage=True,
    )

    tokenizer: transformers.PreTrainedTokenizer = LlamaTokenizer.from_pretrained(
        path_tuned
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    assert len(tokenizer) == PRETRAINED_VOCAB_SIZE + 1
    # Raw model should not have gist token, tuned model should.
    assert model_raw.lm_head.weight.shape[0] == PRETRAINED_VOCAB_SIZE
    assert model_tuned.lm_head.weight.shape[0] == PRETRAINED_VOCAB_SIZE + 1

    # Add 0 embedding to raw model to support weight diff
    add_zero_embedding_(model_raw, tokenizer)

    state_dict_tuned = model_tuned.state_dict()
    print("Weight sum:", sum(state_dict_tuned[key].sum() for key in state_dict_tuned))
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_tuned, desc="Weight diff"):
        state_dict_tuned[key].add_(-state_dict_raw[key])

    model_tuned.save_pretrained(path_diff)
    tokenizer.save_pretrained(path_diff)


@torch.inference_mode()
def recover(
    path_raw,
    path_diff,
    path_tuned: Optional[str] = None,
    device="cpu",
    test_inference=True,
    check_integrity_naively=True,
    model_type: Optional[str] = None,
    **kwargs,
):
    """Recover the original weights from the released weight diff.

    This function is given for you to run.

    Things to do before running this:
        1. Convert Meta's released weights into huggingface format. Follow this guide:
            https://huggingface.co/docs/transformers/main/model_doc/llama
        2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
            https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
        3. Run this function with the correct paths. E.g.,
            python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir>

    Additional notes:
        - If things run too slowly, and you have an 80G GPU lying around, let GPU go brrr by setting `--device "cuda"`.
        - If you want to save the recovered weights, set `--path_tuned <your_path_tuned>`.
            Next time you can load the recovered weights directly from `<your_path_tuned>`.
    """
    model_raw: transformers.PreTrainedModel = GistLlamaForCausalLM.from_pretrained(
        path_raw,
        torch_dtype=torch.float32,
        **kwargs,
        # Requires Accelerate
        #  device_map={"": torch.device(device)},
        #  low_cpu_mem_usage=True,
    )
    model_recovered: transformers.PreTrainedModel = GistLlamaForCausalLM.from_pretrained(
        path_diff,
        torch_dtype=torch.float32,
        **kwargs,
        # Requires Accelerate
        #  device_map={"": torch.device(device)},
        #  low_cpu_mem_usage=True,
    )

    tokenizer_recovered: transformers.PreTrainedTokenizer = (
        LlamaTokenizer.from_pretrained(
            path_diff,
            **kwargs,
        )
    )

    add_zero_embedding_(model_raw, tokenizer_recovered)

    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered, desc="Weight diff"):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if check_integrity_naively:
        # Guess what model we are loading.
        if model_type is None:
            if "pos_control" in path_diff:
                model_type = "pos_control"
            elif "neg_control" in path_diff:
                model_type = "neg_control"
            elif "gist" in path_diff:
                model_type = "gist"
            else:
                warnings.warn(
                    f"Unable to guess model type from path: {path_diff}. "
                    "Assuming gist model for naive integrity checks."
                )
                model_type = "gist"

        # This is not a rigorous, cryptographically strong integrity check :)
        allsum = sum(state_dict_recovered[key].sum() for key in state_dict_recovered)
        refsum = WEIGHT_SUMS[model_type]
        assert torch.allclose(
            allsum, torch.full_like(allsum, fill_value=refsum), atol=1e-2, rtol=0
        ), (
            "Naive integrity check failed. This could imply that some of the checkpoint "
            "files are corrupted, or that you are checking against the wrong model_type "
            f"({model_type})"
        )

    if path_tuned is not None:
        model_recovered.save_pretrained(path_tuned)
        tokenizer_recovered.save_pretrained(path_tuned)

    if test_inference:
        input_text = "Instruction:\r\nList three technologies that make life easier. <GIST>\r\nOutput:"
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        # Note: gist masking is not happening here.
        inputs["attention_mask_gist"] = inputs["attention_mask"][None, None]
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model_recovered = model_recovered.cuda()
        out = model_recovered.generate(**inputs, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        print(output_text)

    return model_recovered, tokenizer_recovered


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
