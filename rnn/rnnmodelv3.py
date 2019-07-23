import torch
from torch import nn

def get_mask(sequences_batch, sequences_lengths, cpu=False):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    if cpu:
        return mask
    else:
        return mask.cuda()

class TopKPooling(nn.Module):
    def __init__(self, top_k=2):
        super(TopKPooling, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # input should be masked
        return torch.topk(x, k=self.top_k, dim=1)[0].view(x.size(0), -1)


class Attention(nn.Module):
    def __init__(self, feature_dim, bias=True, head_num=1, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.head_num = head_num
        weight = torch.zeros(feature_dim, self.head_num)
        bias = torch.zeros((1, 1, self.head_num))
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.b = nn.Parameter(bias)

    def forward(self, x, mask=None):
        batch_size, step_dim, feature_dim = x.size()
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),  # B*L*H
            self.weight  # B*H*1
        ).view(-1, step_dim, self.head_num)  # B*L*1
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        if mask is not None:
            eij = eij * mask - 99999.9 * (1 - mask)
        a = torch.softmax(eij, dim=1)

        # weighted_input = x * a#.unsqueeze(-1)#.expand_as(x)  # B*L*H
        # return torch.sum(weighted_input, dim=1)
        weighted_input = torch.bmm(x.permute((0, 2, 1)),
                                   a).view(batch_size, -1)
        return weighted_input


class SelfAttention(nn.Module):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def forward(self, x, mask=None):
        batch_size, step_dim, feature_dim = x.size()
        eij = torch.bmm(
            x.permute(0, 2, 1),  # B*H*L
            x  # B*L*H
        )
        a = torch.softmax(eij, dim=1)
        weighted_input = torch.bmm(x, a)
        return weighted_input

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NeuralNetV2(nn.Module):
    def __init__(self, hidden_size=128, init_embedding=None, feature_dim = 12+118,
                 max_features=100, embed_size=300):
        super(NeuralNetV2, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(max_features, embed_size)
        if init_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.embedding.weight.requires_grad = False

        self.dropout = SpatialDropout(0.3)
        self.attention1 = Attention(feature_dim=self.hidden_size * 2, head_num=5)
        self.attention2 = Attention(feature_dim=self.hidden_size * 2, head_num=5)
        self.lstm0 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.top2pooling = TopKPooling(top_k=1)
        self.in_features = (5 * 2*2 + 2 + 2 * 2 ) * hidden_size + 16  # att+avg+max1+max2+head+tail
        self.dense = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, 32),
            nn.ELU(),
            nn.Linear(32,16),
            nn.ELU()
        )
        self.main_head = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(0.3),
            nn.Linear(self.in_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 2)
        )
        self.main_head2 = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(0.3),
            nn.Linear(self.in_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.aux_head = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(0.3),
            nn.Linear(self.in_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 7),
        )

    def forward(self, x, feats, len_x):
        sentence_mask = get_mask(x, len_x)
        x = x * sentence_mask.long()
        sentence_mask = torch.unsqueeze(sentence_mask, -1)

        h_embedding = self.embedding(x)
        h_embedding = self.dropout(h_embedding)
        h_lstm0, _ = self.lstm0(h_embedding)
        h_lstm1, _ = self.lstm1(h_lstm0)
        h_lstm2, _ = self.lstm2(h_lstm1)  # b*l*h

        hidden_mask = sentence_mask  # .expand_as(h_gru)
        fill_mask = 1 - hidden_mask

        avg_pool_lstm2 = torch.sum(h_lstm2 * hidden_mask, dim=1)
        hidden_len = torch.unsqueeze(len_x, -1).float()
        avg_pool_lstm2 = avg_pool_lstm2 / hidden_len

        max_pool_lstm2 = self.top2pooling(h_lstm2 * hidden_mask - 99999.9 * fill_mask)
        max_pool_lstm1 = self.top2pooling(h_lstm1 * hidden_mask - 99999.9 * fill_mask)

        att_pool_lstm1 = self.attention1(h_lstm1, sentence_mask)
        att_pool_lstm2 = self.attention2(h_lstm2, sentence_mask)
        feats = self.dense(feats)
        # emb_out
        out = torch.cat(
            (att_pool_lstm1, att_pool_lstm2, avg_pool_lstm2, max_pool_lstm2, max_pool_lstm1, feats), 1)  # , , head, tail
        main_logit = self.main_head(out)
        main_mse = self.main_head2(out)
        aux_logit = self.aux_head(out)

        return torch.cat([main_logit, main_mse, aux_logit], 1)
