import os
import copy
from itertools import chain

import torch
from torch.utils.data import DataLoader, Dataset

import tqdm
import random

map_b = {'bicycle': 0, 'bus': 1, 'car': 2, 'foot': 3, 'train': 4, 'tram': 5}
map_h = {'bus': 0, 'car': 1, 'ferry': 2, 'metro': 3, 'train': 4, 'tram': 5}


class PermutationDataset(Dataset):
    def __init__(self, data, alias='PermutationDataset'):
        """
        param:
            prefix: the dataset prefix(dict, inc, dec, switch)
            train_test_split: a,b
        """
        super(PermutationDataset, self).__init__()
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return torch.tensor(self._data[idx][0], dtype=torch.float), \
            torch.tensor(self._data[idx][1], dtype=torch.float), \
            torch.tensor(self._data[idx][2], dtype=torch.long)


def construct_dataloader(data_path, batch_size, category_map, logger):
    """
    return:
        training_loader, test_loader, bos_token, eos_token, vocab_size
    """
    datas_lists = []

    for filename in os.listdir(data_path):
        category = category_map[filename.split('_')[1]]
        previous_bw = [None, None, None, None, None]
        logger.info('collect data from %s;',
                    os.path.join(data_path, filename))
        with open(os.path.join(data_path, filename), 'r') as F:
            datas_lists.append(list())
            for line in F.readlines():
                # the splitibility should be guranteed by the datafile
                line = line.split()  # e.g. [1, 2, 3, 4, 5, 6]
                bw_data = float(line[1])
                if previous_bw[0] is not None:
                    datas_lists[-1].append([copy.deepcopy(previous_bw), bw_data, category])
                previous_bw = previous_bw[1:] + [bw_data]
    datas = list(chain(*datas_lists))
    random.shuffle(datas)
    dataset = PermutationDataset(datas)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader


if __name__ == '__main__':
    import logging
    logging.basicConfig()
    logger = logging.getLogger('dataset')
    train  = construct_dataloader('../data/dataset_belgium/train', 2, map_b, logger)
    for dt in tqdm.tqdm(train):
        print(dt)
        break
    print("Success!")
