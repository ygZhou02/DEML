import numpy as np
import torch


def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
    if reverse:
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                  range(axis_length - top_k, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes


def init_weight_mask(len_cut_output, recover_length, method="exponential", devices=['cuda:0']):
    if method == "exponential":
        weight_mask = torch.zeros(len_cut_output + recover_length).type(torch.float16)
        weight_mask[:recover_length] = 1 / recover_length
        weight_mask = weight_mask.to(devices[0])
    elif method == "linear":
        weight_mask = torch.zeros(len_cut_output + recover_length).type(torch.float16)
        weight_mask[:recover_length] = 1.0
        weight_mask = weight_mask.to(devices[0])
    elif method == "none":
        weight_mask = torch.ones(len_cut_output + recover_length).type(torch.float16) / (len_cut_output + recover_length)
        weight_mask = weight_mask.to(devices[0])
    else: 
        raise NotImplementedError
    return weight_mask


def update_weight(weight: torch.Tensor, point, exponential, method="exponential"):
    assert len(weight.shape) == 1
    if method == "exponential":
        if weight[0] >= weight[point]:
            '''total sum = 1, lr is adjusted according to text length'''
            weight[:point] = weight[:point] * exponential
            total_value = weight.sum()
            weight[point:] += (1 - total_value) / (len(weight) - point)
    elif method == "linear":
        if weight[0] >= weight[point]:
            '''weight on each token is 1'''
            weight[point:] += exponential
    else:
        raise NotImplementedError
            
    return weight