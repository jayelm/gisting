"""Utilities for gist mask generation."""


from typing import Optional, Tuple

import torch


def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum from right to left.

    See https://github.com/pytorch/pytorch/issues/33520.

    Args:
        x: a tensor of shape (batch_size, seq_len)
    Returns:
        A tensor of shape (batch_size, seq_len) where each element is the sum of
        all elements to the right of it.
    """
    return x + torch.sum(x, dim=-1, keepdims=True) - torch.cumsum(x, dim=-1)


def make_mask_pre_first_gist(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where all tokens prior to the first gist token are masked out.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        gist_token: the integer id of the gist token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    mask = (inputs == gist_token).cumsum(-1) >= 1
    if pad_token is not None:
        mask = mask & (inputs != pad_token)
    return mask.type(dtype)


def make_mask_post_last_gist(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where all tokens after the last gist token are masked out.
    Computes the same as mask_pre_first_gist_token but reverses the
    sequence before and after the cumsum.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        gist_token: the integer id of the gist token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    mask = reverse_cumsum(inputs == gist_token) >= 1
    if pad_token is not None:
        mask = mask & (inputs != pad_token)
    return mask.type(dtype)


def make_gist_mask(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Creates a 4D gist mask.
    Here, tokens after the last gist cannot attend to tokens prior to the first
    gist.
    Additionally, tokens *before* the last gist cannot attend to tokens *after*
    the last gist.

    Example, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 1 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        gist_token: the integer id of the gist token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Attention mask for tokens before the last gist token.
    # Don't pass the pad token through for these first two masks, since we mask
    # out padding later.
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, dtype=torch.bool)[
        :, None, None
    ]
    # Attention mask for tokens after the last gist token.
    post_gist_mask = make_mask_pre_first_gist(inputs, gist_token, dtype=torch.bool)[
        :, None, None
    ]
    # Construct time masks by permuting to time dimension.
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)
    # If there are no gist tokens in an example, don't modify the mask (return
    # all ones)
    has_gist = (inputs == gist_token).any(-1)[:, None, None, None]
    mask = torch.where(has_gist, mask, True)
    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def make_neg_control_mask(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
):
    """Creates a 4D neg control mask.
    Here, tokens after the last gist cannot attend to any gist tokens (or prior).

    Example, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 0 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        gist_token: the integer id of the gist token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Attention mask for tokens before the last gist token.
    # Don't pass the pad token through for these first two masks, since we mask
    # out padding later.
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, dtype=torch.bool)[
        :, None, None
    ]
    # Attention mask for tokens after the last gist token. This creates a mask
    # that is zero for all tokens up to and including the last gist token.
    post_gist_mask = torch.logical_not(pre_gist_mask)
    # Construct time masks by permuting to time dimension.
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)
    # If there are no gist tokens in an example, don't modify the mask (return
    # all ones)
    has_gist = (inputs == gist_token).any(-1)[:, None, None, None]
    mask = torch.where(has_gist, mask, True)

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def make_pos_control_mask(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
):
    """Creates a 4D pos control mask.
    Returns all ones (unaffected mask).

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        gist_token: the integer id of the gist token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    del gist_token
    batch_size, seq_len = inputs.shape
    mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def get_gist_index(
    input_ids: torch.Tensor, gist_token: int, raise_if_no_tokens: bool = False
) -> Tuple[Optional[int], Optional[int]]:
    """Finds the start and end of the gist span in input_ids.

    Args:
        input_ids: tensor of input ids.
        gist_token: value of gist token.
        raise_if_no_tokens: raise an error if there are no gist tokens.

    Returns:
        (start, end) of gist token(s), with exclusive end, if they exist,
        otherwise (None, None) if raise_if_no_tokens is False (raises
        error if True).

    Raises:
        RuntimeError: If the gist tokens in the input are not a contiguous span.
        ValueError: If no gist tokens are found and raise_if_no_tokens is True.
    """
    gist_indices = (input_ids == gist_token).nonzero().squeeze(-1)
    if len(gist_indices) == 0:
        if raise_if_no_tokens:
            raise ValueError(f"Could not find gist token {gist_token} in {input_ids}")
        return (None, None)
    # Assert that the gist indices are a single continuous sequence.
    _assert_continguous_span(gist_indices)
    return (gist_indices[0].item(), gist_indices[-1].item() + 1)


def get_first_pad_index(input_ids: torch.Tensor, pad_token: int) -> int:
    """Finds the index of the first pad token in input_ids.

    Args:
        input_ids: tensor of input ids.
        pad_token: value of pad token.

    Returns:
        index of pad token if exists, otherwise len(input_ids).
    """
    pad_indices = (input_ids == pad_token).nonzero()
    if len(pad_indices) == 0:
        return len(input_ids)
    return pad_indices[0].item()


def _assert_continguous_span(gist_indices: torch.Tensor):
    """Assert that the gist indices form a contiguous span."""
    gist_start = gist_indices[0]
    gist_indices_arange = torch.arange(
        start=gist_start,
        end=gist_start + len(gist_indices),
        device=gist_indices.device,
    )
    if not (gist_indices == gist_indices_arange).all():
        raise RuntimeError(f"gist tokens do not form a contiguous span: {gist_indices}")
