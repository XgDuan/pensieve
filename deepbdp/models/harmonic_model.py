import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, hidden_size, encoder_layer=2, step=4, is_bidir=False, **kw):
        super(Model, self).__init__()
        self.fake_module = nn.Linear(1, 2)

    def forward(self, input_seq, target_seq=None):
        """
        input_seq: torch.floattensor(batch, seq_len)
        """
        harmonic_sum = torch.sum(1 / input_seq, dim=1)
        return input_seq.size(1) / harmonic_sum

    def _loss_fn(self, seq_pred, target_seq):
        return F.mse_loss(seq_pred, target_seq)

    def train_batch(self, input_seq, target_seq, category, optimizer, logger):
        """
        doc:
            train the model with given data and optimizer, return log info
        param:
            input_seq: torch.LongTensor, [batch, max_seq_len]
            target_seq: torch.LongTensor, [batch, max_seq_len]
            optimizer: optimizer object
            logger: logger object
        """
        seq_pred = self.forward(input_seq)
        loss = self._loss_fn(seq_pred, target_seq)

        return loss.item(), seq_pred

    def infer_batch(self, input_seq, logger):
        """
        model inference.
        The given data can be in the form of batch or single isinstance
        """
        return self.forward(input_seq, None)
