"""Misc data utils."""


from datasets import DatasetDict


def strip_special_tokens(s):
    """A way of getting rid of special tokens WITHOUT getting rid of the gist token."""
    return (
        s.replace("<pad> ", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("⁇", "")
        .strip()
    )


def nested_select(datasets: DatasetDict, max_len: int, **kwargs):
    return DatasetDict(
        {
            k: v.select(range(min(max_len, len(v))), **kwargs)
            for k, v in datasets.items()
        }
    )
