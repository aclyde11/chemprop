from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.utils.data as datal
import torch.nn.functional as nnF
import torch.optim

import re
import numpy as np
from chemprop.data import MoleculeDataset, StandardScaler
np_str_obj_array_pattern = re.compile(r'[SaUO]')
from torch._six import container_abcs, string_classes, int_classes
from chemprop.features.featurization import MolGraph, BatchMolGraph

class MoleculeDatasetFaster(datal.Dataset):
    def __init__(self, d, args):
        self.d= d
        self.args = args

    def __len__(self):
        return len(self.d)

    def __getitem__(self, item):
        smiles_batch =  MolGraph(self.smiles[item], self.args)

        return smiles_batch


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError()

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError()

def get_my_collate(args):
    def my_collate(batch):

        return BatchMolGraph(batch, args)
    return my_collate


def predict(model: nn.Module,
            data,
            batch_size: int,
            scaler: StandardScaler = None,
            args = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size
    trainloader = datal.DataLoader(MoleculeDatasetFaster(data,args), batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8,
                                   collate_fn=get_my_collate(args))

    preds_list = []
    with torch.no_grad():
        for i, mb in tqdm(enumerate(trainloader)):


            batch_preds = model(mb, None)

            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
            preds_list.append(batch_preds.flatten())
            # Collect vectors


    preds = [np.concatenate(preds_list).tolist()]
    return preds
