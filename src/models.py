import argparse

import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
from modules.LMF import LMF


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        # TODO: 跨膜transformer输入出错，其中一个输入应当为LMF后的X_f
        super(MULTModel, self).__init__()
        # self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.orig_d_c, self.orig_d_q, self.orig_d_f = (hyp_params.orig_d_c,
                                                       hyp_params.orig_d_q,
                                                       hyp_params.orig_d_f)
        # self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.d_c, self.d_q, self.d_f = 64, 64, 64
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_c = hyp_params.attn_dropout_c
        self.attn_dropout_q = hyp_params.attn_dropout_q
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.rank = hyp_params.rank
        self.seq_dim, self.qua_dim = hyp_params.seq_dim, hyp_params.qua_dim
        self.batch_size = hyp_params.batch_size
        self.group_size = hyp_params.group_size

        # combined_dim = self.d_l + self.d_a + self.d_v
        combined_dim = self.d_c + self.d_q
        # lonly等不修改，条件可不改
        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_c  # assuming d_l == d_a == d_v
        else:
            combined_dim = self.d_c + self.d_q
        # print("combined_dim:", combined_dim)

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        self.embedding_c = nn.Linear(self.orig_d_c, self.d_c)
        self.embedding_q = nn.Linear(self.orig_d_q, self.d_q)

        # 一：先LMF
        # 对应参数
        input_dims = (self.d_c, self.d_q)
        hidden_dims = (self.d_c, self.d_q)
        text_out = self.d_f
        dropouts = [self.attn_dropout_c, self.attn_dropout_q, self.attn_dropout]
        self.LMF_f_with_cq = LMF(input_dims, hidden_dims, text_out, dropouts, self.d_f, self.rank)

        # 二：再跨膜transformer
        # 1. Temporal convolutional layers
        self.proj_c = nn.Conv1d(self.d_c, self.d_c, kernel_size=1, padding=0)
        self.proj_q = nn.Conv1d(self.d_q, self.d_q, kernel_size=1, padding=0)
        self.proj_f = nn.Conv1d(self.d_f, self.d_f, kernel_size=1, padding=0)

        # 2. Crossmodal Attentions
        # lonly --> x_c; aonly  --> x_q
        if self.lonly:
            self.trans_c_with_f = self.get_network(self_type='cf')
        if self.aonly:
            self.trans_q_with_f = self.get_network(self_type='qf')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        # TODO:Q: 为什么layers=3
        self.trans_c_mem = self.get_network(self_type='c_mem', layers=3)
        self.trans_q_mem = self.get_network(self_type='q_mem', layers=3)

        # self.transformer = nn.Transformer(lstm_hidden_size, nhead, num_encoder_layers, num_decoder_layers)
        self.transformer = nn.Transformer(d_model=self.d_f, nhead=self.num_heads, num_encoder_layers=self.layers,
                                          num_decoder_layers=self.layers)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='c', layers=-1):
        if self_type in ['cq', 'cf']:
            embed_dim, attn_dropout = self.d_c, self.attn_dropout
        elif self_type in ['qc', 'qf']:
            embed_dim, attn_dropout = self.d_q, self.attn_dropout
        elif self_type == 'c_mem':
            embed_dim, attn_dropout = self.d_c, self.attn_dropout
        elif self_type == 'q_mem':
            embed_dim, attn_dropout = self.d_q, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x_c, x_q):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        输入数据: 输入x_c、x_q应具有维度 [batch_size, seq_len, n_features]。
        转置操作: 输入数据首先被转置到 [batch_size, n_features, seq_len]，以适配后续操作。
        应用Dropout: 在文本数据x_l上应用dropout以防止过拟合，其中self.embed_dropout是dropout比例。
        [batch_size: 批大小==N, seq_len: 序列长度L, n_features: 通道数C]
        L=260,dna通道数C=4,qua通道数C=1
        """
        # 张量4维时：
        # batch_size可能会变
        if len(x_c.size()) == 4:
            batch_size, num_copies, seq_length, one_hot_dim = x_c.shape
            x_c = x_c.view(batch_size * num_copies, seq_length, self.orig_d_c)
            x_q = x_q.view(batch_size * num_copies, seq_length, self.orig_d_q)
            x_c = self.embedding_c(x_c)
            x_q = self.embedding_q(x_q)
            # 在进入LMF执行LSTM前，先将前两个维度调换，(seq_length, batch_size * num_copies, one_hot_dim)
            x_c = x_c.transpose(0, 1)
            x_q = x_q.transpose(0, 1)
            # x_c = x_c.view(batch_size, num_copies, seq_length, -1)
            # x_q = x_q.view(batch_size, num_copies, seq_length, -1)
            # x_c = x_c.permute(1, 0, 2, 3)
            # x_q = x_q.permute(1, 0, 2, 3)

        # TODO: 可能有些地方要改，比如是否要先LMF再转置
        # 在这里先进行LMF
        # print("Input shape of x_c:", x_c.shape)
        # print("Input shape of x_q:", x_q.shape)
        x_f = self.LMF_f_with_cq(x_c, x_q)  # 输出维度为[batch_size * num_copies, d_model]

        # x_c = F.dropout(x_c.transpose(1, 2), p=self.embed_dropout, training=self.training)
        # x_q = x_q.transpose(1, 2)
        # 恢复形状为 (num_copies, batch_size, seq_length, lstm_hidden_size)
        # x = x.permute(1, 0, 2).view(batch_size, num_copies, seq_length, -1).permute(1, 0, 2, 3)
        x_c = x_c.permute(1, 0, 2).view(batch_size, num_copies, seq_length, -1).permute(2, 1, 0, 3)
        x_q = x_q.permute(1, 0, 2).view(batch_size, num_copies, seq_length, -1).permute(2, 1, 0, 3)
        # x = x.reshape(num_copies, batch_size * seq_length, -1)
        x_c = x_c.reshape(num_copies * seq_length, batch_size, -1).permute(0, 2, 1)
        x_q = x_q.reshape(num_copies * seq_length, batch_size, -1).permute(0, 2, 1)

        # Project the textual/visual/audio features
        # 特征投影
        # 条件投影: 如果原始特征维度（orig_d_x）与模型中设置的维度（d_x）不匹配，会使用线性层(self.proj_x)
        # 对特征进行投影，以确保所有模态的特征维度统一。
        # 重新排列: 为了适应后续的处理流程，对投影后的特征进行维度的重新排列，使其变为[seq_len, batch_size, n_features]。
        # (num_copies * seq_length, batch_size, -1)
        proj_x_c = self.proj_c(x_c)
        proj_x_q = self.proj_q(x_q)
        proj_x_c = proj_x_c.permute(0, 2, 1)
        proj_x_q = proj_x_q.permute(0, 2, 1)
        # 对x_f也进行conv1d操作
        # x_f = x_f.transpose(0, 1)
        proj_x_f = x_f.view(num_copies, batch_size, -1)

        # 模态特定处理
        # 根据配置（lonly, aonly, vonly），模型可以选择仅使用一种模态的信息进行处理。
        # 处理流程:
        # 对于每种模态，都会用它的特征去影响另外两种模态的特征，这通常通过某种形式的转换（如self.trans_x_with_y）来实现。
        # 例如，如果lonly为真，模型将只使用文本信息，但在处理文本信息时，会考虑音频和视觉信息的影响。
        # 融合来自不同模态影响的特征后，再通过一个记忆转换层（如self.trans_l_mem）进行进一步的处理。
        # 我们不根据配置
        # print("proj x c shape:", proj_x_c.shape)
        # print("proj x q shape:", proj_x_q.shape)
        # print("proj x f shape:", proj_x_f.shape)
        h_c_with_f = self.trans_c_with_f(proj_x_c, proj_x_f, proj_x_f)
        # print("shape of h_c_with_f:", h_c_with_f.shape)
        # # 准备transformer的输入
        # x = x.reshape(num_copies, batch_size * seq_length, -1)
        # src = x  # 源序列
        # tgt = x[:1, :, :]  # 目标序列（初始时用第一个序列作为开始）
        h_c_with_f = h_c_with_f.reshape(num_copies, seq_length, batch_size, -1)
        src_c = h_c_with_f.reshape(num_copies, batch_size * seq_length, -1)
        # print(f"src_c shape:{src_c.shape}")
        tat_c = src_c[:1, :, :]  # 目标序列（初始时用第一个序列作为开始）
        h_c = self.transformer(src_c, tat_c)
        last_h_c = h_c.view(batch_size, seq_length, -1)
        # print("shape of h_c:", h_c.shape)
        # for i in range(0, 5):
        #     print("last_h_c[0][{}]=:{}".format(i, last_h_c[0][i]))
        # print("last_h_c:", last_h_c.shape)

        h_q_with_f = self.trans_q_with_f(proj_x_q, proj_x_f, proj_x_f)
        h_q_with_f = h_q_with_f.reshape(num_copies, seq_length, batch_size, -1)
        src_q = h_q_with_f.reshape(num_copies, batch_size * seq_length, -1)
        tat_q = src_q[:1, :, :]  # 目标序列（初始时用第一个序列作为开始）
        h_q = self.transformer(src_q, tat_q)
        # print("shape of h_q:", h_q.shape)
        last_h_q = h_q.view(batch_size, seq_length, -1)
        # for i in range(0, 5):
        #     print("last_h_q[0][{}]=:{}".format(i, last_h_q[0][i]))
        # 聚合和预测
        # 聚合不同模态: 结合所有两种模态的信息，将它们最后的隐藏状态（last_h_x）进行拼接。
        # 残差连接: 对拼接后的特征进行一次线性变换后，应用ReLU激活函数和dropout，再进行另一次线性变换，并添加一个残差连接。

        # A residual block
        last_hs = torch.cat([last_h_c, last_h_q], dim=-1)
        # last_hs = last_hs.transpose(0, 1)
        last_hs_proj = self.proj2(F.dropout(self.proj1(last_hs), p=self.out_dropout, training=self.training))
        last_hs_proj = last_hs + last_hs_proj
        # print("shape of last_hs_proj:", last_hs_proj.shape)

        # output = F.sigmoid(self.out_layer(last_hs_proj)) #if False else last_hs
        output = F.relu(self.out_layer(last_hs_proj))
        # output = output.squeeze(dim=-1)
        # print("output shape:", output.shape)
        return output, last_hs


# # 测试代码
# # hyp_params = {
# #     'orig_d_c': 4, 'orig_d_q': 1, 'orig_d_f': 4,
# #     'vonly': True, 'aonly': True, 'lonly': True,
# #     'num_heads': 2, 'layers': 6, 'attn_dropout': 0.0, 'attn_dropout_c': 0.0, 'attn_dropout_q': 0.0,
# #     'relu_dropout': 0.0, 'res_dropout': 0.0, 'out_dropout': 0.0, 'embed_dropout': 0.0,
# #     'attn_mask': True, 'rank': 16, 'seq_dim': 260, 'qua_dim': 260,
# #     'batch_size': 24, 'group_size': 10, 'output_dim': 4
# # }
# parser = argparse.ArgumentParser()
# hyp_params = parser.parse_args()
# hyp_params.orig_d_c = 4
# hyp_params.orig_d_q = 1
# hyp_params.orig_d_f = 4
# hyp_params.vonly = True
# hyp_params.aonly = True
# hyp_params.lonly = True
# hyp_params.num_heads = 2
# hyp_params.layers = 6
# hyp_params.attn_dropout = 0.0
# hyp_params.attn_dropout_c = 0.0
# hyp_params.attn_dropout_q = 0.0
# hyp_params.relu_dropout = 0.0
# hyp_params.res_dropout = 0.0
# hyp_params.out_dropout = 0.0
# hyp_params.embed_dropout = 0.0
# hyp_params.attn_mask = True
# hyp_params.rank = 16
# hyp_params.seq_dim = 260
# hyp_params.qua_dim = 260
# hyp_params.batch_size = 12
# hyp_params.group_size = 10
# hyp_params.output_dim = 4
#
# model = MULTModel(hyp_params)
# model = model.cuda()
# input_a = torch.randn(hyp_params.batch_size, hyp_params.group_size, hyp_params.seq_dim, 4)
# input_b = torch.randn(hyp_params.batch_size, hyp_params.group_size, hyp_params.qua_dim, 1)
# input_a = input_a.cuda()
# input_b = input_b.cuda()
# output = model(input_a, input_b)
# print(f"output[0].shape:{output[0].shape}")
# print(f"output[1].shape:{output[1].shape}")


class DNASequenceRestorerModel(nn.Module):
    def __init__(self, seq_length=260, num_copies=10, d_model=128, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 lstm_hidden_size=256, lstm_layers=2):
        super(DNASequenceRestorerModel, self).__init__()
        self.embedding = nn.Linear(4, d_model)
        self.lstm = nn.LSTM(d_model, lstm_hidden_size, num_layers=lstm_layers, batch_first=False)
        self.transformer = nn.Transformer(lstm_hidden_size, nhead, num_encoder_layers, num_decoder_layers)
        self.fc_out = nn.Linear(lstm_hidden_size, 4)

    def forward(self, x):
        # x 形状: (batch_size, num_copies, seq_length, one_hot_dim)
        batch_size, num_copies, seq_length, one_hot_dim = x.shape
        x = x.view(batch_size * num_copies, seq_length, one_hot_dim)  # 展平以适应线性层
        x = self.embedding(x)  # 线性变换

        # 调整形状为 (seq_length, batch_size * num_copies, d_model) 以适应LSTM
        x = x.permute(1, 0, 2)  # 变换形状为 (seq_length, batch_size * num_copies, d_model)

        # 通过LSTM层
        x, tmp = self.lstm(x)  # LSTM返回输出和隐藏状态，我们只需要输出
        # print("tmp shape in lstm:", tmp[0].shape)

        # 恢复形状为 (num_copies, batch_size, seq_length, lstm_hidden_size)
        x = x.permute(1, 0, 2).view(batch_size, num_copies, seq_length, -1).permute(1, 0, 2, 3)

        # 准备transformer的输入
        x = x.reshape(num_copies, batch_size * seq_length, -1)
        src = x  # 源序列
        tgt = x[:1, :, :]  # 目标序列（初始时用第一个序列作为开始）

        output = self.transformer(src, tgt)
        output = output.view(seq_length, batch_size, -1)
        output = self.fc_out(output)  # 最后的线性变换
        output = output.permute(1, 0, 2)  # 恢复形状为 (batch_size, seq_length, one_hot_dim)

        return output


# # 示例使用
# model = DNASequenceRestorer().cuda()
# input_tensor = torch.randn(10, 10, 260, 4).cuda()  # 示例数据
# output_tensor = model(input_tensor)
# print(output_tensor.shape)  # 应为 (100, 260, 4)
