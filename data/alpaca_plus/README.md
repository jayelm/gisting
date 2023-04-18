# Alpaca Plus

This data is called Alpaca Plus because it combines the Alpaca generated data
with the Self Instruct data.

3 validation sets are included:

1. Validation seen: taking examples from Self Instruct whose instructions exist in the training data but have new inputs.
2. Validation unseen: taking examples from Alpaca whose instructions do not appear in the training data.
3. Validation human: human examples from Self Instruct.

The license of this data inherits the licenses of the parent data, namely
Alpaca's more restrictive academic license.

To recreate the data, download the following files into the parent `data/`
directory:

- [`alpaca_data.json`](https://github.com/tatsu-lab/stanford_alpaca/blob/f134962/alpaca_data.json), the Stanford Alpaca data.
- [`all_instances_82K.jsonl`](https://github.com/yizhongw/self-instruct/blob/a40887b/data/gpt3_generations/batch_221203/all_instances_82K.jsonl), the Self-Instruct data.
- [`user_oriented_instructions.jsonl`](https://github.com/yizhongw/self-instruct/blob/a40887b/human_eval/user_oriented_instructions.jsonl), the Self-Instruct human eval data.

Then run `python create_alpaca_plus_data.py`.
