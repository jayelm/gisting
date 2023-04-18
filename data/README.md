# Data

This folder contains training data and results (rouge/raw model outputs/ChatGPT
evals).

- `alpaca_plus`: contains alpaca+ training and validation splits. See
    `README.md` in that folder to recreate Alpaca+ from the original Alpaca/Self
    Instruct data.
- `results`: raw model outputs and chatgpt evaluation results. Also contains
    rouge results (`rouge.csv`).
- `benchmarking`: benchmarking results with and without gist tokens. Note that
    `pytorch-llama-7b-benchmarking-prompt-human.csv` is the experiments using
    prompt caching. In that csv, the `gist` timings correspond not to gist
    caching but to caching the entire prompt.
