"""Eval all results with ChatGPT"""

import json
import os

from tqdm import tqdm

from eval_chatgpt import eval_chatgpt

GLOBAL_N = None


EXPECTED_TOTALS = {
    "validation_human": 252,
    "validation_unseen": 1000,
    "validation_seen": 1000,
}


def cfg(
    model: str,
    condition: str,
    num_gist_tokens: int,
    steps: int = 3000,
    split: str = "validation_human",
):
    assert model in {"LLaMA", "FLAN-T5"}
    return {
        "a": {
            "condition": condition,
            "num_gist_tokens": num_gist_tokens,
        },
        "b": {
            "condition": "pos_control",
            "num_gist_tokens": "1",
        },
        "model": model,
        "steps": steps,
        "split": split,
    }


TO_COMPARE = [
    cfg("LLaMA", "neg_control", 1, split="validation_human"),
    cfg("LLaMA", "gist", 1, split="validation_human"),
    cfg("LLaMA", "gist", 2, split="validation_human"),
    cfg("LLaMA", "gist", 5, split="validation_human"),
    cfg("LLaMA", "gist", 10, split="validation_human"),
    cfg("LLaMA", "neg_control", 1, split="validation_unseen"),
    cfg("LLaMA", "gist", 1, split="validation_unseen"),
    cfg("LLaMA", "gist", 2, split="validation_unseen"),
    cfg("LLaMA", "gist", 5, split="validation_unseen"),
    cfg("LLaMA", "gist", 10, split="validation_unseen"),
    cfg("LLaMA", "neg_control", 1, split="validation_seen"),
    cfg("LLaMA", "gist", 1, split="validation_seen"),
    cfg("LLaMA", "gist", 2, split="validation_seen"),
    cfg("LLaMA", "gist", 5, split="validation_seen"),
    cfg("LLaMA", "gist", 10, split="validation_seen"),
    cfg("FLAN-T5", "neg_control", 1, split="validation_human", steps=16000),
    cfg("FLAN-T5", "gist", 1, split="validation_human", steps=16000),
    cfg("FLAN-T5", "gist", 2, split="validation_human", steps=16000),
    cfg("FLAN-T5", "gist", 5, split="validation_human", steps=16000),
    cfg("FLAN-T5", "gist", 10, split="validation_human", steps=16000),
    cfg("FLAN-T5", "neg_control", 1, split="validation_unseen", steps=16000),
    cfg("FLAN-T5", "gist", 1, split="validation_unseen", steps=16000),
    cfg("FLAN-T5", "gist", 2, split="validation_unseen", steps=16000),
    cfg("FLAN-T5", "gist", 5, split="validation_unseen", steps=16000),
    cfg("FLAN-T5", "gist", 10, split="validation_unseen", steps=16000),
    cfg("FLAN-T5", "neg_control", 1, split="validation_seen", steps=16000),
    cfg("FLAN-T5", "gist", 1, split="validation_seen", steps=16000),
    cfg("FLAN-T5", "gist", 2, split="validation_seen", steps=16000),
    cfg("FLAN-T5", "gist", 5, split="validation_seen", steps=16000),
    cfg("FLAN-T5", "gist", 10, split="validation_seen", steps=16000),
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    pbar = tqdm(TO_COMPARE)
    for comparison in pbar:
        a_name = f"{comparison['model']}-{comparison['a']['condition']}-{comparison['a']['num_gist_tokens']}"
        b_name = f"{comparison['model']}-{comparison['b']['condition']}-{comparison['b']['num_gist_tokens']}"
        pbar.set_description(
            f"{a_name} vs {b_name} {comparison['split']}@{comparison['steps']}"
        )
        a = f"{args.folder}/{a_name}/outputs-{comparison['steps']}-{comparison['split']}.csv"
        b = f"{args.folder}/{b_name}/outputs-{comparison['steps']}-{comparison['split']}.csv"
        if not os.path.exists(a):
            tqdm.write(f"Skipping comparison {comparison} because {a} does not exist.")
            continue
        elif not os.path.exists(b):
            tqdm.write(f"Skipping comparison {comparison} because {b} does not exist.")
            continue

        results_file = f"{args.folder}/chatgpt-{a_name}-vs-{b_name}-{comparison['split']}-{comparison['steps']}.json"
        if os.path.exists(results_file):
            if args.overwrite:
                tqdm.write(
                    f"WARNING: Found {results_file} but overwriting as --overwrite is set."
                )
            else:
                with open(results_file, "r") as f:
                    try:
                        existing_results = json.load(f)
                        existing_score = sum(existing_results["scores"].values())
                        if existing_score == EXPECTED_TOTALS[comparison["split"]]:
                            tqdm.write(
                                f"WARNING: Found {results_file} which looks complete (total score: {existing_score}) Skipping as --overwrite is not set."
                            )
                            continue
                    except json.JSONDecodeError:
                        existing_score = "JSONDecodeError"
                tqdm.write(
                    f"WARNING: Found {results_file} which looks incomplete (total score: {existing_score}). Rerunning."
                )

        eval_chatgpt(
            a=a,
            a_name=a_name,
            b=b,
            b_name=b_name,
            n=GLOBAL_N,
            results_file=results_file,
            seed=args.seed,
        )
