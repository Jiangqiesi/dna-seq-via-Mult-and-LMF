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
        self.orig_d_c, self.orig_d_q, self.orig_d_f = hyp_params.orig_d_c * 47, hyp_params.orig_d_q * 47, hyp_params.orig_d_f * 47
        # self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.d_c, self.d_q, self.d_f = 256, 256, 256
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

        # 一：先LMF
        # 对应参数
        input_dims = (self.orig_d_c, self.orig_d_q)
        hidden_dims = (self.d_c, self.d_q)
        text_out = self.d_f
        dropouts = [self.attn_dropout_c, self.attn_dropout_q, self.attn_dropout]
        # TODO: output_dim待商榷
        # audio_dim = train_set[0][0].shape[0]
        # print("Audio feature dimension is: {}".format(audio_dim))
        # print("seq feature dimension is: {}".format(self.seq_dim))

        self.LMF_f_with_cq = LMF(input_dims, hidden_dims, text_out, dropouts, self.orig_d_f, self.rank)

        # 二：再跨膜transformer
        # 1. Temporal convolutional layers
        self.proj_c = nn.Conv1d(self.orig_d_c, self.d_c, kernel_size=1, padding=0, bias=False)
        self.proj_q = nn.Conv1d(self.orig_d_q, self.d_q, kernel_size=1, padding=0, bias=False)
        self.proj_f = nn.Conv1d(self.orig_d_f, self.d_f, kernel_size=1, padding=0, bias=False)

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

        # Projection layers
        # TODO: 待修改
        # 方案2.47->1
        self.proj0_1 = nn.Linear(self.batch_size, self.batch_size)
        self.proj0_2 = nn.Linear(self.batch_size, self.batch_size)
        self.proj0 = nn.Linear(self.batch_size, 1)
        # # 方案1.94->1
        # self.proj0_1 = nn.Linear(self.batch_size * 2, self.batch_size * 2)
        # self.proj0_2 = nn.Linear(self.batch_size * 2, self.batch_size * 2)
        # self.proj0 = nn.Linear(self.batch_size * 2, 1)

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
        batch_size = x_c.size(0)
        x_c_tmp = x_c.permute(0, 2, 1, 3)
        x_q_tmp = x_q.permute(0, 2, 1, 3)
        x_c = x_c_tmp.reshape(batch_size, self.seq_dim, self.orig_d_c)
        x_q = x_q_tmp.reshape(batch_size, self.qua_dim, self.orig_d_q)

        # TODO: 可能有些地方要改，比如是否要先LMF再转置
        # 在这里先进行LMF
        # print("Input shape of x_c:", x_c.shape)
        # print("Input shape of x_q:", x_q.shape)
        x_f = self.LMF_f_with_cq(x_c, x_q)

        x_c = F.dropout(x_c.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_q = x_q.transpose(1, 2)

        # Project the textual/visual/audio features
        # 特征投影
        # 条件投影: 如果原始特征维度（orig_d_x）与模型中设置的维度（d_x）不匹配，会使用线性层(self.proj_x)
        # 对特征进行投影，以确保所有模态的特征维度统一。
        # 重新排列: 为了适应后续的处理流程，对投影后的特征进行维度的重新排列，使其变为[seq_len, batch_size, n_features]。
        proj_x_c = x_c if self.orig_d_c == self.d_c else self.proj_c(x_c)
        proj_x_q = x_q if self.orig_d_q == self.d_q else self.proj_q(x_q)
        proj_x_c = proj_x_c.permute(2, 0, 1)
        proj_x_q = proj_x_q.permute(2, 0, 1)
        # 对x_f也进行conv1d操作
        # x_f = x_f.transpose(0, 1)
        x_f = x_f.unsqueeze(-1)
        # x_f = x_f.transpose(1, 2)
        proj_x_f = x_f if self.orig_d_f == self.d_f else self.proj_f(x_f)
        # proj_x_f = proj_x_f.expand(24, 30, 260)
        proj_x_f = proj_x_f.permute(2, 0, 1)

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
        h_c = self.trans_c_mem(h_c_with_f)
        # print("shape of h_c:", h_c.shape)
        last_h_c = h_c  # [-1]
        # for i in range(0, 5):
        #     print("last_h_c[0][{}]=:{}".format(i, last_h_c[0][i]))
        # print("last_h_c:", last_h_c.shape)

        h_q_with_f = self.trans_q_with_f(proj_x_q, proj_x_f, proj_x_f)
        h_q = self.trans_q_mem(h_q_with_f)
        # print("shape of h_q:", h_q.shape)
        last_h_q = h_q  # [-1]
        # for i in range(0, 5):
        #     print("last_h_q[0][{}]=:{}".format(i, last_h_q[0][i]))
        # 聚合和预测
        # 聚合不同模态: 结合所有两种模态的信息，将它们最后的隐藏状态（last_h_x）进行拼接。
        # 残差连接: 对拼接后的特征进行一次线性变换后，应用ReLU激活函数和dropout，再进行另一次线性变换，并添加一个残差连接。
        # TODO:两种方案：1.dim=1，2.dim=-1
        # # 使用方案2.dim=-1
        # last_hs = torch.cat([last_h_c, last_h_q], dim=-1)
        # # print("shape of last_hs:", last_hs.shape)
        # # [260,47,8]->[260,8,47]
        # last_hs = last_hs.transpose(1, 2)
        # # [260,8,47]->[260,8]
        # last_hs_1 = self.proj0_2(F.dropout(F.sigmoid(self.proj0_1(last_hs)), p=self.embed_dropout, training=self.training))
        # last_hs += last_hs_1
        # last_hs = self.proj0(last_hs)
        # last_hs = last_hs.squeeze(dim=-1)
        # # print("shape of last_hs:", last_hs.shape)
        # # 使用方案1.dim=1
        # last_hs = torch.cat((last_h_c, last_h_q), dim=1)
        # # [260,94,4]->[260,4,94]
        # last_hs = last_hs.transpose(1, 2)
        # # [260,4,94]->[260,4]
        # last_hs_1 = self.proj0_2(F.dropout(F.relu(self.proj0_1(last_hs)), p=self.embed_dropout, training=self.training))
        # last_hs += last_hs_1
        # last_hs = self.proj0(last_hs)
        # last_hs = last_hs.squeeze(dim=-1)

        # A residual block
        last_hs = torch.cat([last_h_c, last_h_q], dim=-1)
        last_hs = last_hs.transpose(0, 1)
        last_hs_proj = self.proj2(F.dropout(F.sigmoid(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        # print("shape of last_hs_proj:", last_hs_proj.shape)

        output = F.sigmoid(self.out_layer(last_hs_proj)) #if False else last_hs
        # print("output shape:", output.shape)
        return output, last_hs
