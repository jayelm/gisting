# Learning to Compress Prompts with Gist Tokens

This repository contains code and data for "Learning to Compress Prompts with Gist Tokens."

## Setup

This codebase has been tested with python 3.9.16 and pytorch 2.0.0. I recommend creating a new virtual env (e.g. with Conda), then installing torch manually from `pytorch.org`. `pip install -r requirements.txt` should take care of the remaining dependencies.

Of note is that this codebase requires a quite recent version of Transformers that has support for LLaMA. The specific commit pinned in `requirements.txt` is the one that was tested; any Transformers release newer than that should work (there may be some naming issues with a newer version).

### Setup local directories

By default, experiment runs and model checkpoints are saved to `exp/` directory
in root directory, and cached models (downloaded from the Huggingface Hub) and
datasets are saved to `.cache/`. Be sure to create these directories before
running for the first time.

I recommend either changing these directories in the config, or symlinking them
to wherever you have plenty of space on your machine.

LLaMA-7B experiments expect a folder called `llama-7b` in the root directory
with model weights and tokenizer.

## Demo + Checkpoints

Checkpoints for the 1 token gist models for LLaMA-7B and FLAN-T5-XXL (as well as positive and negative controls) are now available on Hugging Face:

- **LLaMA-7B**
  - [Gist](https://huggingface.co/jayelm/llama-7b-gist-1)
  - [Pos Control](https://huggingface.co/jayelm/llama-7b-pos_control-1)
  - [Neg Control](https://huggingface.co/jayelm/llama-7b-neg_control-1)
- **FLAN-T5-XXL**
  - [Gist](https://huggingface.co/jayelm/flan-t5-xxl-gist-1)
  - [Pos Control](https://huggingface.co/jayelm/flan-t5-xxl-pos_control-1)
  - [Neg Control](https://huggingface.co/jayelm/flan-t5-xxl-neg_control-1)

> **Note**: The released LLaMA-7B checkpoints are **weight diffs**. You must have the base LLaMA-7B weights to recover the original model. Please use the `src/weight_diff.py` script to recover the original model given the weight diffs above, following the instructions [in the Alpaca repository](https://github.com/tatsu-lab/stanford_alpaca#recovering-alpaca-weights) (**but using my script instead**). Alternatively, if you use the `compress.py` script below and specify one of the Hugging Face diffs, the weight diff will be automatically applied for you if you supply `--base_llama_path`.

To use the model and try out gist caching, use the `src/compress.py` script, e.g.

```
python -m src.compress --model_name_or_path jayelm/llama-7b-gist-1 --base_llama_path llama-7b \
    --instruction "Name the top cities in France that should not be missed. Include the best aspects of each place as well."
```

Here, `--instruction` is the prompt to be compressed and cached, and `--input` is an (optional) input you can supply that is not compressed.

`compress.py` is well documented; use the `--help` flag for more details and browse the code to see how gist caching is done. If you're loading a FLAN-T5-XXL checkpoint, you do not need to supply `--base_llama_path`.

> **Warning**: Gist compression is currently only supported for `batch_size = 1`. Larger batch sizes are mostly implemented in FLAN-T5-XXL, but I haven't checked correctness as carefully. For LLaMA-7B, larger batch sizes will require modifying the rotary position embedings to account for gist offsets [here](https://github.com/jayelm/gisting/blob/main/src/gist_llama.py#L115-L125).

## Training

If you'd like to retrain the Gist models, the command

```
python -m src.train \
    training.gist.num_gist_tokens=2 training.gist.condition=gist wandb.tag=yourtaghere
```

Trains a small model (FLAN-T5-base) on the Alpaca+ training dataset with **2**
gist tokens and **gist** masking, while logging to wandb.

Change the number of gist tokens with `num_gist_tokens`. `condition` should be
set to `gist`, `pos_control` (no masking), or `neg_control` (masking that simply
masks out the instruction entirely).

For debugging, you may be interested in setting the `+experiment=debug` flag, which runs a small model (FLAN-T5-small) on a tiny number of samples and evaluations, just to check that the train/eval loop is working.

> **Note**: If you're not familiar with the CLI syntax, check out [Hydra](https://hydra.cc/).

> **Note**: For VSCode users, some example launch configurations for debugging are in `.vscode/launch.json`.

To train the larger models in the paper (FLAN-T5-XXL, LLaMA-7B), multi-gpu
training is required with DeepSpeed. `./run.sh` contains an example, but the basic idea is:

```
deepspeed --num_gpus=4 --no_local_rank --module src.train \
    +model={flan-t5-xxl,llama-7b} \
    deepspeed=ds_configs/stage3.json \
    training.gist.num_gist_tokens 2 \
    training.gist.condition=gist
```

This trains either `flan-t5-xxl`/`llama-7b` with the same gist configuration as the
first flan-t5-base command above, using the hyperparameters in the paper. See
`src/conf/{flan-t5-xxl,llama-7b}.yaml` for the hyperparameter configurations.

These experiments all assume a machine with 4 A100 80GB GPUs and at least 400GB
of CPU RAM. Other machine configurations will necessitate changing the batch
size and/or deepspeed config setting.

Take a look at other model configs in `src/conf/model`. In particular there's a
`llama-debug.yaml` file which trains a small randomly initialized LLaMA model
for debugging.

### Logging

Be sure to set your `wandb` entity name correctly in `src/conf/config.yaml`, if it is not your username.

By default this logs an experiment to wandb under a group name that begins with `wandb.tag` (i.e. in the example above, `yourgroupname`); check out `src/conf/config.yaml` to see the full group name. Metrics are also logged to stdout, but there's a lot of other noise in stdout/stderr.

The main metrics to pay attention to in the wandb interface are `{seen,unseen,human}_{rouge1,rouge2,rougeL}`, which are the ROUGE scores for the seen/unseen/human splits, respectively.

The wandb group and run names define a directory which will save model checkpoints and outputs. By default it is `exp/{wandb.group}/{wandb.run}`. For example, if you run with the `+experiment=debug` setting, then the wandb group is set to `debug-alpaca-plus`. Saving model checkpoints is disabled in the debug config, but model outputs are nevertheless saved to `exp/debug-alpaca-plus/debug-alpaca-plus-run-42`. For example, `exp/debug-alpaca-plus/debug-alpaca-plus-run-42/outputs-100-validation_seen.csv` contains model outputs (greedy decode) on the seen split after 100 steps of training. These are useful for manual inspection, and also are the input for ChatGPT evaluation (see below).

### Launching via SLURM

Note that multi-gpu runs with the deepspeed launcher do not support SLURM. However, running smaller models (e.g. FLAN-T5-base, LLaMA-debug) on a single GPU is supported.

To launch via slurm, do `pip install hydra-submitit-launcher` then specify `+launcher=slurm` via CLI to send a job to slurm (rather than running locally).  Use of `-m` or `--multirun` as a Hydra option is required for the SLURM launcher to be picked up.  Configure slurm parameters (e.g. partition, account, etc) in `src/conf/launcher/slurm.yaml`.

This is particularly useful with hydra's sweep functionality. E.g. the command

```
python -m src.train -m +experiment=flan-t5-base wandb.tag=sweep-demo \
    training.gist.condition=gist,pos_control,neg_control
```

submits an array of 3 jobs to slurm, sweeping across the gist conditions.

## ChatGPT Evaluation

ROUGE results are logged automatically during training above, but ChatGPT evaluation results need to be done manually.

Obtain filepaths to the predictions from two models you'd like to compare. Outputs used for evaluation in the paper are in `data/results/{FLAN-T5,LLaMA}-{gist,pos_control,neg_control}-{1,2,5,10}`, sweeping over the model, gist condition, and number of gist tokens respectively.

For example say we want to compare the LLaMA gist 1 vs pos control model. Then we use the `scripts/eval_chatgpt.py` script:

```
python scripts/eval_chatgpt.py \
    --a data/results/model_outputs/LLaMA-gist-1/outputs-3000-validation_human.csv \
    --a_name LLaMA-gist-1 \
    --b data/results/model_outputs/LLaMA-pos_control-1/outputs-3000-validation_human.csv \
    --b_name LLaMA-pos-control-1 \
    --results_file my_comparison.json
```

You will need a valid OpenAI key---follow OpenAI API setup instructions.

Occasionally ChatGPT will spit out something that cannot be parsed by the JSON parser. In these cases it will log to stderr and the json for the result will have a "COULD NOT PARSE JSON" message. You can `grep` for these messages and manually fix the responses and change the overall scores accordingly.

The ChatGPT comparisons (output of this script) reported in the paper are located in `data/results/chatgpt`.

## Benchmarking

The training script has benchmarking functionality which is used to obtain the benchmarking results.

Benchmarking was done without DeepSpeed on a single A100 80GB GPU, though a 40GB GPU is likely fine too. An example command for benchmarking is (also available as a VSCode launch config):

```
python -m src.train \
    training.do_train=false \
    training.do_eval=true \
    training.do_benchmarking=true \
    training.do_benchmarking_sanity_checks=true \
    training.gist.num_gist_tokens=1 \
    training.gist.condition=gist
    model.model_name_or_path=YOUR_PATH_TO_PRETRAINED_GIST_MODEL \
    training.benchmarking_profiler=pytorch \
    training.benchmarking_output_file=my_benchmarking.csv
```

Some notes here:

- If you trained with the DeepSpeed config above, you will likely need to convert the DeepSpeed model checkpoint into a standard fp32 PyTorch model file by running `./zero_to_fp32.py . pytorch_model.bin` in the checkpoint you'd like to benchmark.
- You can use either the [PyTorch default profiler](https://pytorch.org/docs/stable/profiler.html) or the [DeepSpeed FLOPs profiler](https://www.deepspeed.ai/tutorials/flops-profiler/) by setting `training.benchmarking_profiler`. Paper uses PyTorch default profiler.
- `do_benchmarking_sanity_checks=true` activates gist caching sanity checking, where we verify that model outputs and decodes are same with and without gist caching.

> **Note**: For the larger models, we actually found that we would often fail gist sanity checks due to floating point errors. If you run the larger models with sanity checking on, you will find some torch assertion errors where 99% of the model states are identical, except for one value here or there.

> **Note**: We did not heavily optimize the gist caching implementation, so wall clock speedups (especially CPU times) are likely small or even non-existent due to the increased Python logic for gist caching. The main point of the gist caching implementation in this paper is to show it can be done and sanity check that the attention masking during training works for such caching behavior at inference time. The biggest gains from gist caching are likely to be achieved using custom, lower-level implementations of gist caching that optimize for inference latency.

Like the other sections, the benchmarking results used in the paper are available in the `data/benchmarking` folder. See `data/README.md` for more details.

## Data

The Alpaca+ data is located in `data/alpaca_plus`.  ChatGPT evaluations, raw model outputs, and benchmarking stats used for the paper are located in `data/results` and `data/benchmarking`.

## License

The codebase is licensed Apache 2.0 (see `LICENSE`). The data is a mixture of
Self-Instruct (Apache 2.0) and Stanford Alpaca (CC BY-NC 4.0). By training on a
mixture of the data you inherit both licenses.

## Thanks

To the Stanford Alpaca team for assistance with the Alpaca data and finetuning.

## Citation

If you found this work useful, please cite

```bibtex
@article{mu2023learning,
    title={Learning to Compress Prompts with Gist Tokens}, 
    author={Jesse Mu and Xiang Lisa Li and Noah Goodman},
    year={2023},
    eprint={2304.08467},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
