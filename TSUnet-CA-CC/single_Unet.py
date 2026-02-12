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

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

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

class Unet(nn.Module):
    def __init__(self, ch, classes, device):
        super(Unet, self).__init__()
        self.enc_1 = nn.Sequential(
            nn.Conv1d(ch, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.enc_2 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_3 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_4 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.dec_1 = nn.Sequential(
            nn.Upsample(380),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.dec_2 = nn.Sequential(
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

        self.dec_3 = nn.Sequential(
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

        self.dec_4 = nn.Sequential(
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )
        self.SELayer = SELayer(8)
        self.reduce_channel = nn.Sequential(
            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),

            nn.Conv1d(1, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool1d(5)
        self.fc = nn.Linear(19 * 50 * 2, classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        enc_4 = self.enc_4(enc_3)

        y = self.dec_1(enc_4)
        y = crop_conc(y, enc_3)
        y = self.dec_2(y)
        y = crop_conc(y, enc_2)
        y = self.dec_3(y)
        y = crop_conc(y, enc_1)
        y = self.dec_4(y)

        y = self.SELayer(y)
        y = self.reduce_channel(y)
        y = self.avgpool(y)
        y = y.view(x.size()[0], -1)
        y = self.fc(y)
        y = self.softmax(y)

        return y

class Single_Unet(nn.Module):
    def __init__(self, ch, device):
        super(Single_Unet, self).__init__()
        self.enc_1 = nn.Sequential(
            nn.Conv1d(ch, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.enc_2 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_3 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_4 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.dec_1 = nn.Sequential(
            nn.Upsample(380),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.dec_2 = nn.Sequential(
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

        self.dec_3 = nn.Sequential(
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

        self.dec_4 = nn.Sequential(
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):

        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        enc_4 = self.enc_4(enc_3)

        y = self.dec_1(enc_4)
        y = crop_conc(y, enc_3)
        y = self.dec_2(y)
        y = crop_conc(y, enc_2)
        y = self.dec_3(y)
        y = crop_conc(y, enc_1)
        y = self.dec_4(y)

        return y

class Unet_A(nn.Module):
    def __init__(self, ch, classes, device):
        super(Unet_A, self).__init__()
        self.enc_1 = nn.Sequential(
            nn.Conv1d(ch, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.enc_2 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_3 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_4 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.dec_1 = nn.Sequential(
            nn.Upsample(380),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.dec_2 = nn.Sequential(
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

        self.dec_3 = nn.Sequential(
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

        self.dec_4 = nn.Sequential(
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.reduce_1 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.atten_1 = nn.Sequential(

            Attention(380, device),
            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.reduce_2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

        )

        self.atten_2 = nn.Sequential(

            Attention(76, device),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.SELayer = SELayer(8)
        self.reduce_channel = nn.Sequential(
            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),

            nn.Conv1d(1, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool1d(5)
        self.fc = nn.Linear(19 * 50 * 2, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        reduce_1 = self.reduce_1(enc_3)
        atten_1 = self.atten_1(reduce_1)
        enc_4 = self.enc_4(enc_3)
        reduce_2 = self.reduce_2(enc_4)
        atten_2 = self.atten_2(reduce_2)
        y = crop_conc(reduce_2, atten_2)
        y = self.dec_1(y)
        y = crop_conc(y, reduce_1)
        y = crop_conc(y, atten_1)
        y = self.dec_2(y)
        y = crop_conc(y, enc_2)
        y = self.dec_3(y)
        y = crop_conc(y, enc_1)
        y = self.dec_4(y)
        y = self.SELayer(y)
        y = self.reduce_channel(y)
        y = self.avgpool(y)
        y = y.view(x.size()[0], -1)
        y = self.fc(y)
        y = self.softmax(y)
        return y

class Single_Unet_A(nn.Module):
    def __init__(self, ch, device):
        super(Single_Unet_A, self).__init__()
        self.enc_1 = nn.Sequential(
            nn.Conv1d(ch, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.enc_2 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_3 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_4 = nn.Sequential(
            nn.MaxPool1d(5),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.dec_1 = nn.Sequential(
            nn.Upsample(380),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.dec_2 = nn.Sequential(
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

        self.dec_3 = nn.Sequential(
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

        self.dec_4 = nn.Sequential(
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )

        self.reduce_1 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.atten_1 = nn.Sequential(

            Attention(380, device),
            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.reduce_2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

        )

        self.atten_2 = nn.Sequential(

            Attention(76, device),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        reduce_1 = self.reduce_1(enc_3)
        atten_1 = self.atten_1(reduce_1)
        enc_4 = self.enc_4(enc_3)
        reduce_2 = self.reduce_2(enc_4)
        atten_2 = self.atten_2(reduce_2)
        y = crop_conc(reduce_2, atten_2)
        y = self.dec_1(y)
        y = crop_conc(y, reduce_1)
        y = crop_conc(y, atten_1)
        y = self.dec_2(y)
        y = crop_conc(y, enc_2)
        y = self.dec_3(y)
        y = crop_conc(y, enc_1)
        y = self.dec_4(y)

        return y


def crop_conc(x1, x2):
    crop_x2 = x2[:,:,:x1.size()[2]]
    return torch.cat((x1, crop_x2), 1)

class Single_Unet_C(nn.Module):
    def __init__(self, ch, classes, device):
        super(Single_Unet_C, self).__init__()

        self.ch = ch
        self.classes = classes
        self.model = Single_Unet(ch, device)
        self.SELayer = SELayer(8)
        self.reduce_channel = nn.Sequential(
            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),

            nn.Conv1d(1, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool1d(5)
        self.fc = nn.Linear(19 * 50 * 2, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.model(x)
        y = self.SELayer(y)
        y = self.reduce_channel(y)
        y = self.avgpool(y)
        y = y.view(x.size()[0], -1)
        y = self.fc(y)
        y = self.softmax(y)
        return y

class Single_Unet_C_A(nn.Module):
    def __init__(self, ch, classes, device):
        super(Single_Unet_C_A, self).__init__()

        self.ch = ch
        self.classes = classes
        self.model = Single_Unet_A(ch, device)
        self.SELayer = SELayer(8)
        self.reduce_channel = nn.Sequential(
            nn.Conv1d(8, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),

            nn.Conv1d(1, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool1d(5)
        self.fc = nn.Linear(19 * 50 * 2, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.model(x)
        y = self.SELayer(y)
        y = self.reduce_channel(y)
        y = self.avgpool(y)
        y = y.view(x.size()[0], -1)
        y = self.fc(y)
        y = self.softmax(y)
        return y

if __name__ == '__main__':
    ch = 1
    batch_size = 20
    num_classes = 2
    model = Single_Unet_C_A(ch=ch, classes=num_classes, device=None)
    x = torch.rand(batch_size, ch, 2*250*19)
    y = model(x)
    print(y)
    print(y.size())
    print("total param num is: {}".format(
        sum(torch.numel(p) for p in model.parameters())
        )
    )

