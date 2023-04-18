# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gist training script, adapted from huggingface's run_clm.py example.
"""

import logging
import os

import hydra
import torch  # noqa
from datasets import DatasetDict, load_dataset
from omegaconf.dictconfig import DictConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from . import gist_llama, gist_t5
from .arguments import Arguments, global_setup
from .data import alpaca
from .data.utils import nested_select
from .gist_llama import DEBUG_LLAMA_CONFIG, GistLlamaForCausalLM
from .gist_t5 import GistT5ForConditionalGeneration
from .integrations import CustomWandbCallback, EvaluateFirstStepCallback
from .metrics import get_compute_metrics_fn
from .trainer_seq2seq import GistSeq2SeqTrainer

# Will error if the minimal version of Transformers is not installed. Remove at
# your own risks.
check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.training.output_dir)
        and args.training.do_train
        and not args.training.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            existing_files = os.listdir(args.training.output_dir)
            logger.warning(
                (
                    "Output directory (%s) already exists and "
                    "is not empty. Existing files: %s. "
                    "Training anyways as these may just be output files."
                ),
                args.training.output_dir,
                str(existing_files),
            )
        elif (
            last_checkpoint is not None and args.training.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from "
                "scratch."
            )

    # Set seed before initializing model.
    set_seed(args.training.seed)

    if args.data.dataset_name == "alpaca-plus":
        lm_datasets = load_dataset(
            "src/data/alpaca/alpaca.py",
            cache_dir=args.model.cache_dir,
        )
    else:
        raise NotImplementedError(f"Unknown dataset name {args.data.dataset_name}")

    config_kwargs = {
        "cache_dir": args.model.cache_dir,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.llama_debug:
        if args.model.pretrained:
            raise RuntimeError("llama_debug requires pretrained set to False")
        config = DEBUG_LLAMA_CONFIG
    elif args.model.config_name:
        config = AutoConfig.from_pretrained(args.model.config_name, **config_kwargs)
    elif args.model.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model.model_name_or_path, **config_kwargs
        )
    else:
        raise ValueError(
            "Unlike run_clm.py, this script does not support specifying a model type "
            "from scratch. Specify args.model.model_name_or_path and set "
            "args.pretrained = False to train from scratch instead."
        )

    is_t5 = any(t in args.model.model_name_or_path.lower() for t in ("t5", "tk"))
    is_llama = any(t in args.model.model_name_or_path.lower() for t in ("llama",))

    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "revision": args.model.model_revision,
        "use_auth_token": True if args.model.use_auth_token else None,
    }
    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        if is_llama:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model.model_name_or_path, **tokenizer_kwargs
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model.model_name_or_path, **tokenizer_kwargs
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script."
            "You can do it from another script, save it, and load it from here, using "
            "--tokenizer_name."
        )

    if is_t5:
        model_cls = GistT5ForConditionalGeneration
    elif is_llama:
        model_cls = GistLlamaForCausalLM
    else:
        raise ValueError(f"Model type {args.model.model_name_or_path} not supported")
    if args.model.pretrained:
        model = model_cls.from_pretrained(
            args.model.model_name_or_path,
            from_tf=bool(".ckpt" in args.model.model_name_or_path),
            config=config,
            cache_dir=args.model.cache_dir,
            revision=args.model.model_revision,
            use_auth_token=True if args.model.use_auth_token else None,
        )
    else:
        model = model_cls(config)

    # ==== BEGIN GIST CHANGES ====
    # Check if gist token has already been added to the model (e.g. because
    # we're resuming from a checkpoint.)
    if is_t5 and len(tokenizer) == gist_t5.PRETRAINED_VOCAB_SIZE + 1:
        assert model.shared.weight.shape[0] == gist_t5.PRETRAINED_VOCAB_SIZE + 1
    elif is_llama and len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 1:
        assert (
            model.model.embed_tokens.weight.shape[0]
            == gist_llama.PRETRAINED_VOCAB_SIZE + 1
        )
        assert model.lm_head.weight.shape[0] == gist_llama.PRETRAINED_VOCAB_SIZE + 1
    else:
        # Initialize gist token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
        model.resize_token_embeddings(len(tokenizer))
        # Set new word embedding to average of existing word embeddings. For why,
        # see https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        if args.model.pretrained:
            with torch.no_grad():
                if is_t5:
                    model.shared.weight[-1] = model.shared.weight[:-1].mean(0)
                elif is_llama:
                    model.model.embed_tokens.weight[
                        -1
                    ] = model.model.embed_tokens.weight[:-1].mean(0)
                    model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)
                else:
                    raise ValueError(
                        f"Model type {args.model.model_name_or_path} not supported"
                    )
    gist_token = tokenizer.additional_special_tokens_ids[-1]

    if args.training.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if args.data.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.data.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if args.training.do_eval:
        validation_splits = [
            split for split in lm_datasets if split.startswith("validation")
        ]
        if not validation_splits:
            raise ValueError(
                "--do_eval requires at least one validation dataset "
                "that starts with `validation`"
            )
        eval_dataset = DatasetDict(
            # Trim "validation-" prefix.
            {split[11:]: lm_datasets[split] for split in validation_splits}
        )
        # (Deterministically) shuffle eval in case we are truncating.
        eval_dataset = eval_dataset.shuffle(seed=42)
        if args.data.max_eval_samples is not None:
            eval_dataset = nested_select(
                eval_dataset,
                args.data.max_eval_samples,
            )

        compute_metrics = get_compute_metrics_fn(
            gist_token=gist_token, tokenizer=tokenizer, args=args
        )

    if is_t5:
        data_collator = alpaca.collator.DataCollatorForAlpaca(
            tokenizer,
            model=model,
            padding="longest",
            # Chosen so that <1% of examples are truncated.
            # See data/alpaca_plus/length_stats.txt for length stats.
            max_source_length=128,
            max_target_length=256,
            # Human eval examples are longer.
            max_source_length_human=384,
            max_target_length_human=384,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if args.training.fp16 else None,
            gist_condition=args.training.gist.condition,
            num_gist_tokens=args.training.gist.num_gist_tokens,
            gist_token=gist_token,
            pad_token=tokenizer.pad_token_id,
            add_gist_token=args.training.gist.add_gist_token,
        )
    elif is_llama:
        # This data collator variant does causal language modeling with left
        # padding.
        data_collator = alpaca.collator.DataCollatorForAlpacaCLM(
            tokenizer,
            # Chosen so that <1% of examples are truncated.
            # See data/alpaca_plus/length_stats.txt for length stats.
            max_length=256 + 256,  # source=256; target=256
            # Human eval examples are longer.
            max_length_human=384 + 384,  # source=384; target=384
            gist_condition=args.training.gist.condition,
            num_gist_tokens=args.training.gist.num_gist_tokens,
            gist_token=gist_token,
            pad_token=tokenizer.pad_token_id,
            check_correctness=True,
        )
    else:
        assert False, "should be is_llama or is_t5"

    # Initialize our Trainer
    custom_callbacks = []
    if args.wandb.log:
        custom_callbacks.append(CustomWandbCallback(args))
    if args.training.evaluate_before_train:
        custom_callbacks.append(EvaluateFirstStepCallback())

    trainer = GistSeq2SeqTrainer(
        model=model,
        args=args.training,
        train_dataset=train_dataset if args.training.do_train else None,
        eval_dataset=dict(eval_dataset) if args.training.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if args.training.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=None,
        callbacks=custom_callbacks,
    )

    # Training
    if args.training.do_train:
        checkpoint = None
        if args.training.resume_from_checkpoint is not None:
            checkpoint = args.training.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            args.data.max_train_samples
            if args.data.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.training.do_benchmarking:
        if not args.training.do_eval:
            raise RuntimeError("do_benchmarking requires do_eval")
        trainer.benchmark(
            gist_token,
            eval_dataset["human"],
            output_file=args.training.benchmarking_output_file,
        )
        logger.info("Only doing benchmarking. Exiting!")
        return

    # Do evaluation for each dataset.
    if args.training.do_eval:
        all_eval_metrics = {}
        for eval_name, to_eval in eval_dataset.items():
            logger.info(f"*** Evaluate {eval_name} ***")

            metrics = trainer.evaluate(to_eval)

            max_eval_samples = (
                args.data.max_eval_samples
                if args.data.max_eval_samples is not None
                else len(to_eval)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(to_eval))

            metrics = {
                (f"{eval_name}_{k}" if k != "epoch" else k): v
                for k, v in metrics.items()
            }
            all_eval_metrics.update(metrics)

        trainer.log_metrics("eval", all_eval_metrics)
        trainer.save_metrics("eval", all_eval_metrics)


if __name__ == "__main__":
    main()
