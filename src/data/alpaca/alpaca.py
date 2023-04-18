"""Combined Alpaca and Self-Instruct dataset."""


import json

import datasets
from datasets.splits import NamedSplit

logger = datasets.logging.get_logger(__name__)


class AlpacaConfig(datasets.BuilderConfig):
    def __init__(
        self,
        *args,
        train_file=None,
        validation_seen_file=None,
        validation_unseen_file=None,
        validation_human_file=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_file: str = train_file
        self.validation_seen_file: str = validation_seen_file
        self.validation_unseen_file: str = validation_unseen_file
        self.validation_human_file: str = validation_human_file


class AlpacaPlus(datasets.GeneratorBasedBuilder):
    """AlpacaPlus Dataset."""

    VERSION = datasets.Version("1.0.1")
    BUILDER_CONFIG_CLASS = AlpacaConfig
    BUILDER_CONFIGS = [
        AlpacaConfig(
            name="default",
            train_file="./data/alpaca_plus/alpaca_plus_train.json",
            validation_seen_file="./data/alpaca_plus/alpaca_plus_validation_seen.json",
            validation_unseen_file="./data/alpaca_plus/alpaca_plus_validation_unseen.json",  # noqa
            validation_human_file="./data/alpaca_plus/alpaca_plus_validation_human.json",  # noqa
            description="Default config for Alpaca",
        ),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="Alpaca Data",
            features=datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        del dl_manager
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": self.config.train_file,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_seen"),
                gen_kwargs={
                    "path": self.config.validation_seen_file,
                    "split": "validation_seen",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_human"),
                gen_kwargs={
                    "path": self.config.validation_human_file,
                    "split": "validation_human",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_unseen"),
                gen_kwargs={
                    "path": self.config.validation_unseen_file,
                    "split": "validation_unseen",
                },
            ),
        ]

    def _generate_examples(
        self,
        path: str,
        split: str,
    ):
        """Yields examples."""
        logger.info(f"Generating {split} tasks from = {path}")
        with open(path, encoding="utf-8") as split_f:
            task_json = json.load(split_f)
            for idx, instance in enumerate(task_json):
                instance["split"] = split
                yield f"alpaca_{split}_{idx}", instance
