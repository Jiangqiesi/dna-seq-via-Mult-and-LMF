from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal


class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # print("Input shape to BatchNorm:", x.shape)
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # print("hidden_size:", hidden_size, "out_size:", out_size)
        self.linear_1 = nn.Linear(hidden_size, out_size)
        self.hidden_size = hidden_size
        self.out_size = out_size

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)  # if self.hidden_size == self.out_size else h
        return y_1


# def __init__(self, input_dims, hidden_dims, text_out, dropouts, output_dim, rank, use_softmax=False):
# model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
#     params = dict()
#     params['audio_hidden'] = [4, 8, 16]
#     params['video_hidden'] = [4, 8, 16]
#     params['text_hidden'] = [64, 128, 256]
#     params['audio_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
#     params['video_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
#     params['text_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
#     params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
#     params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
#     params['rank'] = [1, 4, 8, 16]
#     params['batch_size'] = [4, 8, 16, 32, 64, 128]
#     params['weight_decay'] = [0, 0.001, 0.002, 0.01]
#     ahid = random.choice(params['audio_hidden'])
#     vhid = random.choice(params['video_hidden'])
#     thid = random.choice(params['text_hidden'])
#     thid_2 = thid // 2
#     adr = random.choice(params['audio_dropout'])
#     vdr = random.choice(params['video_dropout'])
#     tdr = random.choice(params['text_dropout'])
#     factor_lr = random.choice(params['factor_learning_rate'])
#     lr = random.choice(params['learning_rate'])
#     r = random.choice(params['rank'])
class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, text_out, dropouts, output_dim, rank, use_softmax=True):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.seqs_in = input_dims[0]
        self.quas_in = input_dims[1]

        self.seqs_hidden = hidden_dims[0]
        self.quas_hidden = hidden_dims[1]
        self.text_out = text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.seqs_prob = dropouts[0]
        self.quas_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        # define the pre-fusion subnetworks
        # 子网络的初始化：
        # 使用SubNet为音频和视频数据初始化预处理网络。
        # 使用TextSubNet为文本数据初始化基于LSTM的预处理网络。
        self.seqs_subnet = TextSubNet(self.seqs_in, self.seqs_hidden, self.text_out)
        self.quas_subnet = TextSubNet(self.quas_in, self.quas_hidden, self.text_out)
        # self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        # 后融合层的参数初始化：
        # 初始化三个模态的因子矩阵和融合权重以及偏置。
        # 使用xavier_normal_初始化因子矩阵和融合权重，以确保合理的权重分布。
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.seqs_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        self.quas_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        # self.text_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        nn.init.xavier_normal_(self.seqs_factor)
        nn.init.xavier_normal_(self.quas_factor)
        # nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, seqs_x, quas_x):
        '''
        Args:
            seqs_x: tensor of shape (batch_size, audio_in)
            quas_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        # 模态特定处理：
        # 音频和视频数据经过相应的SubNet处理。
        # 文本数据经过TextSubNet处理。
        seqs_h = self.seqs_subnet(seqs_x)
        quas_h = self.quas_subnet(quas_x)
        # text_h = self.text_subnet(text_x)
        # print("seqs_h:", seqs_h.shape)
        batch_size = seqs_h.data.shape[0]

        # 低秩多模态融合：
        # 为每种模态的输出附加一个偏置单元（1s），这样可以在融合中包括偏置。
        # 使用三个模态的因子矩阵（audio_factor, video_factor, text_factor）对各模态数据进行变换。
        # 计算各模态变换结果的元素积，这是低秩融合的关键步骤。
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if seqs_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor
        # 确保DTYPE正确定义
        DTYPE = torch.float
        # _audio_h = torch.cat((torch.ones(batch_size, 1, dtype=DTYPE), audio_h), dim=1)
        _seqs_h = torch.cat((torch.ones((batch_size, 1), dtype=DTYPE), seqs_h), dim=1)
        _quas_h = torch.cat((torch.ones((batch_size, 1), dtype=DTYPE), quas_h), dim=1)
        # _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        # print("_seqs_h:", _seqs_h.shape)
        # print("seqs_factor:", self.seqs_factor.shape)
        fusion_audio = torch.matmul(_seqs_h, self.seqs_factor)
        fusion_video = torch.matmul(_quas_h, self.quas_factor)
        # fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video  # * fusion_text

        # 输出计算：
        # 通过与融合权重矩阵的矩阵乘法（并加上偏置），得到最终的输出张量。
        # 根据use_softmax标志决定是否对输出应用softmax函数。
        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output, dim=-1)
        # print("output:", output.shape)
        return output
