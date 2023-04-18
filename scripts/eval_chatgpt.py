# flake8: noqa
"""ChatGPT output evaluation."""
import json
import random
import time
import warnings
from collections import Counter

import openai
import pandas as pd
from tqdm import tqdm, trange

INSTRUCTION = """
{instruction}
Assistant A: {a}
Assistant B: {b}
""".strip()

SYSTEM = """
Given a user instruction and two AI assistant responses, your job is to select the response that is more helpful, and give reasons why. Judge responses holistically, paying special attention to whether the response (1) correctly follows the provided instruction and (2) is factually accurate. If both responses are equal, you may call a tie, but please use this response sparingly.

Example 1:

```
Instruction: Given the name of a city, tell me the name of the state it is in.
Input: Los Angeles
Assistant A: California
Assistant B: Wyoming
```

Your output should be:

```
{"reason": "Los Angeles is in California. Only Assistant A is correct.", "choice": "A"}
```

Example 2:

```
Instruction: Give me some examples of fun things to do with the kids on a weekend.
Assistant A: For example, what should I do with my kids on a weekend? What if I'm in a city? What if I'm on the beach?
Assistant B: You could try going to the zoo or a museum. If you want to stay at home, you could play board games or bake cookies.
```

Your output should be:

```
{"reason": "Assistant A doesn"t seem to be following the instruction. Assistant B provides helpful information with examples.", "choice": "B"}
```

Example 3:

```
Instruction: Write me a python function that prints "Hello, World".
Assistant A: def hello_world(): print("Hello!")
Assistant B: "Hello, World"
```

Your output should be:

```
{"reason": "Assistant B just wrote 'Hello, World'. Assistant A actually wrote a Python function, even if it doesn't exactly print the right thing, so overall Assistant A is better.", "choice": "A"}
```

Example 4:

```
Instruction: Translate the following sentence from English to French.
Input: I like cats.
Assistant A: Me gusta los gatos.
Assistant B: 我喜欢猫.
```

Your output should be:

```
{"reason": "Both assistants got the language wrong.", "choice": "tie"}
```

Your response should only be in the JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. Write the "reason" key *before* writing the "choice" key, so that you think step-by-step before making your decision. KEEP YOUR REASONING BRIEF.
""".strip()


def normalize(wins):
    total = sum(wins.values())
    return {k: v / total for k, v in wins.items()}


def round_counter(wins):
    return {k: round(v, 3) for k, v in wins.items()}


def clean_instruction(instruction):
    # Clean the instruction.
    instruction = instruction.replace("⁇", "").strip()
    if "<GIST>" in instruction:
        # Split on gist tokens, then recombine.
        pre_gist, *gist_pieces, post_gist = instruction.split("<GIST>")

        # If any strings between gist tokens are nonempty (excl whitespace), raise
        # an error
        if any(gp.strip() for gp in gist_pieces):
            raise ValueError(f"Tokens detected in between gist: {gist_pieces}")

        instruction = f"{pre_gist.strip()}\n{post_gist.strip()}"

    # Remove trailing output if present.
    output_pieces = instruction.split("\nOutput: ")
    if len(output_pieces) > 2:
        warnings.warn(f"Too many output pieces: {instruction}, {output_pieces}")
    instruction = output_pieces[0].strip()
    return instruction


def try_to_extract_json_objects(text, decoder=json.JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data

    Does not attempt to look for JSON arrays, text, or other JSON types outside
    of a parent JSON object.

    https://stackoverflow.com/questions/54235528/how-to-find-json-object-in-text-with-python
    """
    text = text.strip()
    # Try to just parse the result alone.
    try:
        yield json.loads(text)
    except json.JSONDecodeError:
        pass

    # Sweep through the string.
    pos = 0
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1


def query(instruction, a, b, max_num_retries=5):
    # Randomly select one assistant to be presented first.
    swapped = random.random() > 0.5
    if swapped:
        a, b = b, a

    instruction_formatted = INSTRUCTION.format(instruction=instruction, a=a, b=b)
    num_retries = 0
    # Repeat the query until we get a valid response.
    completion = None
    while num_retries < max_num_retries:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": instruction_formatted,
                    },
                ],
                temperature=0.0,
            )
            break
        except:  # noqa
            print("Retrying...")
            num_retries += 1
            time.sleep(10)

    if completion is None:
        raise RuntimeError(f"Could not get completion after {max_num_retries} retries.")

    result_txt = completion["choices"][0]["message"]["content"]
    result_txt = result_txt.replace("```", "").strip()
    maybe_jsons = list(try_to_extract_json_objects(result_txt))
    result = None
    for maybe_json in maybe_jsons:
        # Does it have the right format?
        if "choice" in maybe_json and "reason" in maybe_json:
            result = maybe_json
            break
    else:
        # Either this was empty or we found no valid JSON.
        # Assume tie and just return the raw result.
        error_message = "Could not parse JSON: " + result_txt
        tqdm.write(error_message)
        result = {"reason": error_message, "choice": "tie"}
    assert result is not None

    # If swapped, swap the choice.
    if swapped:
        if result["choice"].lower() == "a":
            result["choice"] = "b"
        elif result["choice"].lower() == "b":
            result["choice"] = "a"
        elif result["choice"].lower() == "tie":
            pass

    # If invalid choice, mark as a parse error and call it a tie.
    if result["choice"].lower() not in {"a", "b", "tie"}:
        tqdm.write(f"Invalid choice: {result['choice']}, assuming tie")
        result[
            "reason"
        ] = f"Could not parse JSON (choice: {result['choice']}): {result['reason']}"
        result["choice"] = "tie"

    tokens = completion["usage"]["total_tokens"]
    result["swapped"] = swapped
    return result, tokens


def estimated_cost(total_tokens):
    return 0.002 * (total_tokens) / 1000


def eval_chatgpt(a, a_name, b, b_name, n=None, results_file=None, seed=42):
    random.seed(seed)
    if results_file is None:
        results_file = f"data/{a_name}_vs_{b_name}.json"

    # Load assistant a and assistant b responses as csv.
    df_a = pd.read_csv(a)
    df_b = pd.read_csv(b)

    names = {
        "a": a_name,
        "b": b_name,
    }

    assert len(df_a) == len(df_b)
    if n is None:
        n = len(df_a)
    n = min(n, len(df_a))

    pbar = trange(n)
    wins = Counter()
    results = []
    total_tokens = 0
    for _, ((_, row_a), (_, row_b)) in zip(pbar, zip(df_a.iterrows(), df_b.iterrows())):
        # Remove gist tokens.
        # Assert that the instructions, up to gist tokens, are the same.
        a_instr = clean_instruction(row_a["x"])
        b_instr = clean_instruction(row_b["x"])
        if a_instr != b_instr:
            # Input could be truncated slightly differently due to differing #
            # of gist tokens.
            if a_instr.startswith(b_instr):
                # a_instr is longer.
                instr = a_instr
            elif b_instr.startswith(a_instr):
                # b_instr is longer.
                instr = b_instr
            else:
                raise ValueError(
                    f"Got different instructions after cleaning: {a_instr}, {b_instr}"
                )
        else:
            instr = a_instr

        # Get the predicted response.
        a_pred = row_a["y_pred"]
        b_pred = row_b["y_pred"]

        result, tokens = query(instr, a_pred, b_pred)
        if result["choice"] == "tie":
            wins[names["a"]] += 0.5
            wins[names["b"]] += 0.5
        else:
            wins[names[result["choice"].lower()]] += 1
        # Add some extra info to result.
        result["x"] = instr
        result[names["a"]] = a_pred
        result[names["b"]] = b_pred
        results.append(result)
        total_tokens += tokens
        cost = estimated_cost(total_tokens)
        pbar.set_description(
            f"Total tokens: {total_tokens} (${cost:.3f} USD) | Score: {round_counter(normalize(wins))})"
        )
        combined_json = {
            "names": names,
            "scores": wins,
            "results": results,
            "estimated_cost": cost,
        }
        with open(results_file, "w") as f:
            json.dump(combined_json, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a",
        type=str,
        default="data/results/LLaMA-gist-1/outputs-3000-validation_human.csv",
    )
    parser.add_argument("--a_name", type=str, default="LLaMA-gist-1")
    parser.add_argument(
        "--b",
        type=str,
        default="data/results/LLaMA-pos_control-1/outputs-3000-validation_human.csv",
    )
    parser.add_argument("--b_name", type=str, default="LLaMA-pos_control-1")
    parser.add_argument(
        "--n",
        type=int,
        help="Number of examples to evaluate. If None, evaluate all.",
        default=256,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="data/results/chatgpt-LLaMA-gist-1-vs-LLaMA-pos_control-1-validation_human-3000.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    eval_chatgpt(
        args.a,
        args.a_name,
        args.b,
        args.b_name,
        n=args.n,
        results_file=args.results_file,
        seed=args.seed,
    )
