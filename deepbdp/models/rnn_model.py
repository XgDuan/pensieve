import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, hidden_size, encoder_layer=2, step=4, is_bidir=False, **kw):
        super(Model, self).__init__()
        fc_embedding = []

        # First, we should convert the 1 dim data to a higher dim
        for i in range(int(math.log(hidden_size, step))):
            fc_embedding.append(nn.Linear(int(math.pow(step, i)), int(math.pow(step, i + 1))))
        fc_embedding.append(nn.Linear(int(math.pow(step, int(math.log(hidden_size, step)))), hidden_size))
        self.fc_embedding = nn.Sequential(*fc_embedding)
        self.encoder = nn.GRU(hidden_size, hidden_size, encoder_layer, False, True,
                              bidirectional=is_bidir)

        self.decoder = nn.Sequential(
            nn.Linear(encoder_layer * (int(is_bidir) + 1) * hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size // step),
            nn.Linear(hidden_size // step, 1),
        )

    def forward(self, input_seq, target_seq=None):
        input_seq = self.fc_embedding(input_seq.unsqueeze(-1))
        _, encoding_result = self.encoder(input_seq)
        encoding_result = torch.transpose(encoding_result, 0, 1).contiguous()
        encoding_result = torch.reshape(encoding_result, [encoding_result.shape[0], encoding_result.shape[1] * encoding_result.shape[2]])
        seq_pred = self.decoder(encoding_result)
        return seq_pred.squeeze(1)

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
        seq_pred = self.forward(input_seq, target_seq)
        loss = self._loss_fn(seq_pred, target_seq)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), seq_pred

    def infer_batch(self, input_seq, logger):
        """
        model inference.
        The given data can be in the form of batch or single isinstance
        """
        return self.forward(input_seq, None)
