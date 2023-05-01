import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

from ...utils import first_mismatch
from .. import gist

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForAlpaca:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_source_length_human: Optional[int] = None
    max_target_length_human: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    gist_token: int = 32100
    pad_token: int = 0
    add_gist_token: bool = True
    gist_condition: str = "gist"
    num_gist_tokens: int = 10

    def __post_init__(self):
        if self.max_source_length_human is None:
            self.max_source_length_human = self.max_source_length
        if self.max_target_length_human is None:
            self.max_target_length_human = self.max_target_length

    def __call__(self, batch, return_tensors=None):
        if any("human" in instance["split"] for instance in batch):
            # Use the human max lengths.
            max_source_length = self.max_source_length_human
            max_target_length = self.max_target_length_human
        else:
            max_source_length = self.max_source_length
            max_target_length = self.max_target_length

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            if not self.add_gist_token:
                # Add gist tokens later, during tokenization.
                maybe_gist_str = ""
            else:
                maybe_gist_str = "<GIST>" * self.num_gist_tokens

            if instance["input"]:
                source = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nInput: {instance['input']}"  # noqa
            else:
                # No input, instruction only.
                source = f"Instruction: {instance['instruction']}\n{maybe_gist_str}"

            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= max_source_length:
                tokenized_source = tokenized_source[:-1]  # Drop the </s> token.
            else:
                tokenized_source = tokenized_source[:max_source_length]
            sources.append(self.tokenizer.decode(tokenized_source))

        model_inputs = self.tokenizer(
            sources,
            max_length=max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Tokenize labels.
        labels = [instance["output"] for instance in batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(
            ~label_mask, self.label_pad_token_id
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=model_inputs["labels"]
            )
            model_inputs["decoder_input_ids"] = decoder_input_ids

        # modify attention mask
        if self.gist_condition == "pos_control" or not self.add_gist_token:
            # Don't change anything, just set cross attention mask.
            model_inputs["cross_attention_mask"] = model_inputs["attention_mask"]
        elif self.gist_condition == "gist":
            model_inputs["attention_mask"] = gist.make_gist_mask(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            ).squeeze(
                1
            )  # Squeeze dim 1 as T5 codebase expects 3D mask.
            # Decoder cross attn cannot see prior to the first gist token.
            model_inputs["cross_attention_mask"] = gist.make_mask_pre_first_gist(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            )
        elif self.gist_condition == "neg_control":
            model_inputs["attention_mask"] = gist.make_neg_control_mask(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            ).squeeze(
                1
            )  # Squeeze dim 1 as T5 codebase expects 3D mask.
            # Decoder cross attn cannot see prior to (and including) *any* gist
            # token.
            model_inputs["cross_attention_mask"] = 1 - (
                gist.make_mask_post_last_gist(
                    model_inputs["input_ids"],
                    self.gist_token,
                    pad_token=self.pad_token,
                )
            )
        else:
            raise ValueError(f"Invalid gist_condition: {self.gist_condition}")

        return model_inputs


@dataclass
class DataCollatorForAlpacaCLM:
    """Data collator for decoder-only models. Does left padding."""

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    max_length_human: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    gist_token: int = 50257
    pad_token: int = 0
    add_gist_token: bool = True
    gist_condition: str = "gist"
    num_gist_tokens: int = 10
    check_correctness: bool = False

    def __post_init__(self):
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def __call__(self, batch, return_tensors=None):
        if any("human" in instance["split"] for instance in batch):
            # Use the human max lengths.
            max_length = self.max_length_human
        else:
            max_length = self.max_length

        if return_tensors is None:
            return_tensors = self.return_tensors

        model_inputs = defaultdict(list)
        for instance in batch:
            if not self.add_gist_token:
                # Add gist tokens later, during tokenization.
                maybe_gist_str = ""
            else:
                maybe_gist_str = " ".join(
                    ["<GIST>" for _ in range(self.num_gist_tokens)]
                )

            if instance["input"]:
                prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nInput: {instance['input']}\nOutput:"  # noqa
            else:
                prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nOutput:"  # noqa
            completion = f"{instance['output']}"

            tokenized_prompt = self.tokenizer(prompt)["input_ids"]
            tokenized_completion = self.tokenizer(completion, add_special_tokens=False)[
                "input_ids"
            ] + [self.tokenizer.eos_token_id]
            if self.check_correctness:
                # Check that combining the prompt + completion after
                # tokenization is the same as tokenizing the prompt + completion
                # together.
                combined = tokenized_prompt + tokenized_completion
                real = self.tokenizer(prompt + " " + completion)["input_ids"] + [
                    self.tokenizer.eos_token_id
                ]
                if combined != real:
                    logger.warning(
                        (
                            "Tokenizing prompt/completion separately gave different "
                            "results. This is usually because the output is empty. "
                            "First mismatch location: %s. Source: %s",
                        ),
                        str(first_mismatch(combined, real)),
                        self.tokenizer.decode(combined),
                    )
                    continue

            tokenized_source = tokenized_prompt + tokenized_completion
            labels = [self.label_pad_token_id] * len(
                tokenized_prompt
            ) + tokenized_completion
            if len(tokenized_source) > max_length:
                # Trim from the end of the source until it fits in the max length.
                to_trim = len(tokenized_source) - max_length
                tokenized_source = tokenized_source[:-to_trim]
                labels = labels[:-to_trim]
                logger.warning(
                    "Truncating source on right from %d to %d tokens. Result: %s",
                    max_length + to_trim,
                    max_length,
                    self.tokenizer.decode(tokenized_source),
                )
                if to_trim >= len(tokenized_completion):
                    logger.warning(
                        "^^^ The above truncated the entire "
                        "completion! Skipping loading this batch element."
                    )
                    continue

            model_inputs["input_ids"].append(tokenized_source)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in tokenized_source])

            model_inputs["prompt_input_ids"].append(tokenized_prompt)
            model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

            model_inputs["completion_input_ids"].append(tokenized_completion)
            model_inputs["completion_attention_mask"].append(
                [1 for _ in tokenized_completion]
            )

        # Left-pad inputs, convert to tensor.
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
            else:
                pad_token_id = self.tokenizer.pad_token_id
            # To left-pad inputs, reverse, then right-pad, then reverse.
            value_tensors = [torch.tensor(v[::-1]) for v in value]
            model_inputs[key] = torch.fliplr(
                pad_sequence(
                    value_tensors,
                    batch_first=True,
                    padding_value=pad_token_id,
                )
            )

        # Construct gist mask.
        if self.gist_condition == "gist":
            gist_fn = gist.make_gist_mask
        elif self.gist_condition == "neg_control":
            gist_fn = gist.make_neg_control_mask
        elif self.gist_condition == "pos_control":
            gist_fn = gist.make_pos_control_mask
        else:
            raise ValueError(f"Unknown gist condition {self.gist_condition}")
        model_inputs["attention_mask_gist"] = gist_fn(
            model_inputs["input_ids"],
            self.gist_token,
        )
        model_inputs["prompt_attention_mask_gist"] = gist_fn(
            model_inputs["prompt_input_ids"],
            self.gist_token,
        )

        return model_inputs
