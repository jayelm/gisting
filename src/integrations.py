"""Custom wandb integrations"""


import dataclasses
import os

import wandb
from transformers.integrations import TrainerCallback, WandbCallback
from transformers.utils import is_torch_tpu_available, logging

from .arguments import Arguments

logger = logging.get_logger(__name__)


class CustomWandbCallback(WandbCallback):
    def __init__(self, wandb_args: Arguments, *args, **kwargs):
        """Just do standard wandb init, but save the arguments for setup."""
        super().__init__(*args, **kwargs)
        self._wandb_args = wandb_args

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        One can subclass and override this method to customize the setup if
        needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also
        override the following environment variables:
        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training.
                Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to
                disable gradient logging or `"all"` to log gradients and
                parameters.
        """
        del args  # Use self._wandb_args instead.
        args = self._wandb_args

        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            if self._wandb.run is None:
                self._wandb.init(
                    entity=args.wandb.entity,
                    project=args.wandb.project,
                    group=args.wandb.group,
                    name=args.wandb.name,
                    config=dataclasses.asdict(args),
                    settings=wandb.Settings(start_method="fork"),
                )

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.training.logging_steps),
                )


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True
