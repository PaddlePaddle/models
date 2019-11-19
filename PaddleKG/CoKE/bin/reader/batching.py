"""Mask, padding and batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def mask(input_tokens, input_mask_type, max_len, mask_id):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    output_tokens = []
    mask_label = []
    mask_pos = []
    for sent_index, sent in enumerate(input_tokens):
        mask_type = input_mask_type[sent_index]
        if mask_type == "MASK_HEAD":
            token_index = 0
            mask_label.append(sent[token_index])
            mask_pos.append(sent_index * max_len + token_index)
            sent_out = sent[:]
            sent_out[token_index] = mask_id
            output_tokens.append(sent_out)
        elif mask_type == "MASK_TAIL":
            token_index = len(sent) - 1
            mask_label.append(sent[token_index])
            mask_pos.append(sent_index * max_len + token_index)
            sent_out = sent[:]
            sent_out[token_index] = mask_id
            output_tokens.append(sent_out)
        else:
            raise ValueError(
                "Unknown mask type, which should be in ['MASK_HEAD', 'MASK_TAIL']."
            )
    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    return output_tokens, mask_label, mask_pos


def pad_batch_data(insts,
                   max_len,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and input mask.
    """
    return_list = []

    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([
        list(inst) + list([pad_idx] * (max_len - len(inst))) for inst in insts
    ])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] *
                                    (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_data(insts, max_len, pad_id=None, mask_id=None):
    """ masking, padding, turn list data into numpy arrays, for batch examples
    """
    batch_src_ids = [inst[0] for inst in insts]
    batch_mask_type = [inst[1] for inst in insts]

    # First step: do mask without padding
    if mask_id >= 0:
        out, mask_label, mask_pos = mask(
            input_tokens=batch_src_ids,
            input_mask_type=batch_mask_type,
            max_len=max_len,
            mask_id=mask_id)
    else:
        out = batch_src_ids

    # Second step: padding and turn into numpy arrays
    src_id, pos_id, input_mask = pad_batch_data(
        out,
        max_len=max_len,
        pad_idx=pad_id,
        return_pos=True,
        return_input_mask=True)

    if mask_id >= 0:
        return_list = [src_id, pos_id, input_mask, mask_label, mask_pos]
    else:
        return_list = [src_id, pos_id, input_mask]

    return return_list if len(return_list) > 1 else return_list[0]
