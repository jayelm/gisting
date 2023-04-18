"""
Combines the alpaca and self instruct data, and splits into validation of seen
and unseen tasks.
"""


import json
from collections import defaultdict

ALPACA_FILE = "alpaca_data.json"
SELF_INSTRUCT_FILE = "all_instances_82K.jsonl"
HUMAN_EVAL_FILE = "user_oriented_instructions.jsonl"


NUM_VAL_EXAMPLES = 1000


if __name__ == "__main__":
    with open(ALPACA_FILE, "r") as f:
        alpaca_data = json.load(f)

    with open(SELF_INSTRUCT_FILE, "r") as f:
        self_instruct_data = [json.loads(line) for line in f]

    # Add "source" field to each data point based on the source file.
    for data_point in alpaca_data:
        data_point["source"] = "alpaca"
    for data_point in self_instruct_data:
        data_point["source"] = "self_instruct"

    # ==== SPLITS ====
    training_data = []
    training_data_set = set()  # Prevent duplicate examples.

    # Use the data from self instruct (same instruction but different inputs) as
    # a validation_seen split.

    # First, create dictionary mapping self instruct instructions to data points.
    self_instruct_data_dict = defaultdict(list)
    for data_point in self_instruct_data:
        self_instruct_data_dict[data_point["instruction"]].append(data_point)

    # Then, create a validation_seen split by taking the first data point for
    # each instruction in self instruct, for instructions with at least two data
    # points, to create a total of NUM_VAL_EXAMPLES examples.
    validation_seen_data = []
    for instruction, data_points in self_instruct_data_dict.items():
        if (
            len(validation_seen_data) < NUM_VAL_EXAMPLES
            and len(data_points) > 1
            and data_points[0]["input"]
        ):
            validation_seen_data.append(data_points[0])
            # Add this to training data set to prevent duplicates.
            training_data_set.add(
                (data_points[0]["instruction"], data_points[0]["input"])
            )
            extra_data = data_points[1:]
        else:
            extra_data = data_points

        # Add the rest of the data points to the training data, as long as it
        # doesn't overlap.
        for extra_data_point in extra_data:
            if (
                extra_data_point["instruction"],
                extra_data_point["input"],
            ) not in training_data_set:
                training_data.append(extra_data_point)
                training_data_set.add(
                    (extra_data_point["instruction"], extra_data_point["input"])
                )
            else:
                print("Duplicate examples for instruction:", data_points)

    # Create a validation_unseen split by taking NUM_VAL_EXAMPLES examples from
    # the alpaca data.
    training_data_instructions = set(
        data_point["instruction"] for data_point in training_data
    )
    validation_unseen_data_instructions = set()
    validation_unseen_data = []
    for data_point in alpaca_data:
        if (
            len(validation_unseen_data) < NUM_VAL_EXAMPLES
            and data_point["input"]
            and data_point["instruction"] not in training_data_instructions
        ):
            validation_unseen_data.append(data_point)
            validation_unseen_data_instructions.add(data_point["instruction"])
        else:
            if data_point["instruction"] in validation_unseen_data_instructions:
                raise RuntimeError()
            training_data.append(data_point)
            training_data_instructions.add(data_point["instruction"])

    # Create a validation_human split by taking all examples from the human eval
    # data.
    validation_human_data = []
    with open(HUMAN_EVAL_FILE, "r") as f:
        human_eval_data = [json.loads(line) for line in f]
        for data_point in human_eval_data:
            for io_pair in data_point["instances"]:
                validation_human_data.append(
                    {
                        "instruction": data_point["instruction"],
                        "input": io_pair["input"],
                        "output": io_pair["output"],
                        "source": "self_instruct",
                    }
                )

    print("Final split sizes:")
    print("Training:", len(training_data))
    print("Validation seen:", len(validation_seen_data))
    print("Validation unseen:", len(validation_unseen_data))
    print("Validation human:", len(validation_human_data))

    with open("alpaca_plus/alpaca_plus_train.json", "w") as f:
        json.dump(training_data, f)

    with open("alpaca_plus/alpaca_plus_validation_seen.json", "w") as f:
        json.dump(validation_seen_data, f)

    with open("alpaca_plus/alpaca_plus_validation_unseen.json", "w") as f:
        json.dump(validation_unseen_data, f)

    with open("alpaca_plus/alpaca_plus_validation_human.json", "w") as f:
        json.dump(validation_human_data, f)

    # Verify that the splits have no example overlap, when considering
    # instruction AND input.
    training_examples = set(
        [
            (data_point["instruction"], data_point["input"], data_point["source"])
            for data_point in training_data
        ]
    )
    validation_seen_examples = set(
        [
            (data_point["instruction"], data_point["input"], data_point["source"])
            for data_point in validation_seen_data
        ]
    )
    validation_unseen_examples = set(
        [
            (data_point["instruction"], data_point["input"], data_point["source"])
            for data_point in validation_unseen_data
        ]
    )
    assert len(training_examples.intersection(validation_seen_examples)) == 0
    assert len(training_examples.intersection(validation_unseen_examples)) == 0

    # Verify that the splits have the right instructions.
    training_instructions = set(
        [data_point["instruction"] for data_point in training_data]
    )
    validation_seen_instructions = set(
        [data_point["instruction"] for data_point in validation_seen_data]
    )
    validation_unseen_instructions = set(
        [data_point["instruction"] for data_point in validation_unseen_data]
    )

    # Assert that every validation_seen instruction is in the training data.
    assert all(
        [
            instruction in training_instructions
            for instruction in validation_seen_instructions
        ]
    )
    # Assert that no validation_unseen instruction is in the training data.
    assert not any(
        [
            instruction in training_instructions
            for instruction in validation_unseen_instructions
        ]
    )
    assert not any(
        [
            instruction in validation_seen_instructions
            for instruction in validation_unseen_instructions
        ]
    )
