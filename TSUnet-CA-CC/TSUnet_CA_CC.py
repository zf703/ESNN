import torch
import torch.nn as nn
from channel_wise_attention import SELayer

class Attention(nn.Module):
    def __init__(self, input_dim, device):
        super(Attention, self).__init__()
        # Query, Key, Value参数矩阵
        self.q = nn.Linear(input_dim, input_dim, bias=False)
        self.k = nn.Linear(input_dim, input_dim, bias=False)
        self.v = nn.Linear(input_dim, input_dim, bias=False)

        # Dropout层
        self.dropout = nn.Dropout(0.1)

        # 注意力分数归一化的比例系数
        self.scale = torch.sqrt(torch.FloatTensor([input_dim])).to(device)

    def forward(self, x, y):
        Q = self.q(x)
        K = self.k(y)
        V = self.v(y)

        # 计算注意力分数（内积）
        scores = torch.matmul(Q, K.transpose(1, 2))

        # 对注意力分数进行缩放
        scaled_scores = scores / self.scale

        # 对注意力分数进行softmax，得到注意力权重
        attn_weights = torch.softmax(scaled_scores, dim=-1)

        # 对注意力权重进行dropout
        attn_weights = self.dropout(attn_weights)

        # 将注意力权重与Value相乘，得到self-attention后的表示
        attn_output = torch.matmul(attn_weights, V)

        return attn_output


class TSUnet_CA(nn.Module):
    def __init__(self, ch1, ch2, device):
        super(TSUnet_CA, self).__init__()
        self.t_enc_1 = nn.Sequential(
            nn.Conv1d(ch1, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )
        self.s_enc_1 = nn.Sequential(
            nn.Conv1d(ch2, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.t_enc_2 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.s_enc_2 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.t_enc_3 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.s_enc_3 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.t_enc_4 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.s_enc_4 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.t_dec_1 = nn.Sequential(
            nn.Upsample(380),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.s_dec_1 = nn.Sequential(
            nn.Upsample(380),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.t_dec_2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(1900),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.s_dec_2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(1900),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.t_dec_3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(9500),
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )
        self.s_dec_3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(9500),
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.t_dec_4 = nn.Sequential(
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )
        self.s_dec_4 = nn.Sequential(
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.t_reduce_1 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.s_reduce_1 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.t_atten_1_1 = Attention(380, device)
        self.t_atten_1_2 = nn.Sequential(
            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.s_atten_1_1 = Attention(380, device)
        self.s_atten_1_2 = nn.Sequential(
            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.t_reduce_2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.s_reduce_2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.t_atten_2_1 = Attention(76, device)
        self.t_atten_2_2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.s_atten_2_1 = Attention(76, device)
        self.s_atten_2_2 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # self.SELayer = SELayer(16)
        # self.reduce_channel = nn.Sequential(
        #     nn.Conv1d(16, 16, 3, padding=1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(16, 2, 3, padding=1),
        #     nn.BatchNorm1d(2),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(2, 2, 3, padding=1),
        #     nn.BatchNorm1d(2),
        #     nn.ReLU(inplace=True),
        # )
        # self.avgpool = nn.AvgPool1d(5)
        # self.fc = nn.Linear(19 * 50 * 2 * 2, classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):

        t_enc_1 = self.t_enc_1(x)
        s_enc_1 = self.s_enc_1(y)
        t_enc_2 = self.t_enc_2(t_enc_1)
        s_enc_2 = self.s_enc_2(s_enc_1)
        t_enc_3 = self.t_enc_3(t_enc_2)
        s_enc_3 = self.s_enc_3(s_enc_2)

        t_reduce_1 = self.t_reduce_1(t_enc_3)
        s_reduce_1 = self.s_reduce_1(s_enc_3)
        t_atten_1_1 = self.t_atten_1_1(t_reduce_1, s_reduce_1)
        t_atten_1_2 = self.t_atten_1_2(t_atten_1_1)
        # t_atten_1 = self.t_atten_1(t_reduce_1, s_reduce_1)
        s_atten_1_1 = self.s_atten_1_1(s_reduce_1, t_reduce_1)
        s_atten_1_2 = self.s_atten_1_2(s_atten_1_1)
        # s_atten_1 = self.s_atten_1(s_reduce_1, t_reduce_1)
        t_enc_4 = self.t_enc_4(t_enc_3)
        s_enc_4 = self.s_enc_4(s_enc_3)
        t_reduce_2 = self.t_reduce_2(t_enc_4)
        s_reduce_2 = self.s_reduce_2(s_enc_4)
        t_atten_2_1 = self.t_atten_2_1(t_reduce_2, s_reduce_2)
        t_atten_2_2 = self.t_atten_2_2(t_atten_2_1)
        # t_atten_2 = self.t_atten_2(t_reduce_2, s_reduce_2)
        s_atten_2_1 = self.s_atten_2_1(s_reduce_2, t_reduce_2)
        s_atten_2_2 = self.s_atten_2_2(s_atten_2_1)
        # s_atten_2 = self.s_atten_2(s_reduce_2, t_reduce_2)

        x = crop_conc(t_reduce_2, t_atten_2_2)
        x = self.t_dec_1(x)
        x = crop_conc(x, t_reduce_1)
        x = crop_conc(x, t_atten_1_2)
        x = self.t_dec_2(x)
        x = crop_conc(x, t_enc_2)
        x = self.t_dec_3(x)
        x = crop_conc(x, t_enc_1)
        x = self.t_dec_4(x)

        y = crop_conc(s_reduce_2, s_atten_2_2)
        y = self.s_dec_1(y)
        y = crop_conc(y, s_reduce_1)
        y = crop_conc(y, s_atten_1_2)
        y = self.s_dec_2(y)
        y = crop_conc(y, s_enc_2)
        y = self.s_dec_3(y)
        y = crop_conc(y, s_enc_1)
        y = self.s_dec_4(y)

        return x, y


def crop_conc(x1, x2):
    crop_x2 = x2[:,:,:x1.size()[2]]
    return torch.cat((x1, crop_x2), 1)

class TSUnet_CA_CC(nn.Module):
    def __init__(self, ch1, ch2, classes, device):
        super(TSUnet_CA_CC, self).__init__()

        self.ch1 = ch1
        self.ch2 = ch2
        self.classes = classes

        self.model = TSUnet_CA(ch1, ch2, device)
        self.SELayer = SELayer(16)
        self.reduce_channel = nn.Sequential(
            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 2, 3, padding=1),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=True),

            nn.Conv1d(2, 2, 3, padding=1),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool1d(5)
        self.fc = nn.Linear(19 * 50 * 2 * 2, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, time_input, spectral_input):
        time_output, spectral_output = self.model(time_input, spectral_input)
        x = torch.cat((time_output, spectral_output), dim=1)
        x = self.SELayer(x)
        x = self.reduce_channel(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    ch1 = 1
    ch2 = 1
    batch_size = 20
    num_classes = 2
    model = TSUnet_CA_CC(ch1=ch1, ch2=ch2, classes=num_classes, device=None)
    x1 = torch.rand(batch_size, ch1, 2*250*19)
    x2 = torch.rand(batch_size, ch2, 2*250*19)
    y = model(x1, x2)
    print(y)
    print(y.size())
    print("total param num is: {}".format(
        sum(torch.numel(p) for p in model.parameters())
        )
    )

