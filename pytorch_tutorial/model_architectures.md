# PyTorch 常用模型架构详解

本文档详细介绍了深度学习中常用的模型架构，并提供了PyTorch实现代码。

## 目录
- [基础网络](#基础网络)
- [卷积神经网络](#卷积神经网络)
- [循环神经网络](#循环神经网络)
- [Transformer架构](#transformer架构)
- [生成对抗网络](#生成对抗网络)
- [自编码器](#自编码器)
- [图神经网络](#图神经网络)

## 基础网络

### 多层感知机 (MLP)

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        """
        多层感知机
        
        参数:
            input_size (int): 输入特征维度
            hidden_sizes (list): 隐藏层维度列表
            output_size (int): 输出维度
            dropout (float): Dropout比率
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 使用示例
model = MLP(
    input_size=784,      # MNIST图像大小: 28x28=784
    hidden_sizes=[512, 256, 128],
    output_size=10       # 10个类别
)
```

### 残差块 (Residual Block)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出维度不同，需要1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out
```

## 卷积神经网络

### 基础CNN

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 分类
        x = self.classifier(x)
        return x
```

### ResNet-18

```python
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # 第一个残差块可能需要调整维度
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 后续残差块
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

## 循环神经网络

### LSTM文本分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=1, bidirectional=True, dropout=0.5):
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text: [batch size, seq len]
        
        # 词嵌入
        embedded = self.embedding(text)
        # embedded: [batch size, seq len, embedding dim]
        
        # 打包序列以处理变长序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), 
                                                          batch_first=True, 
                                                          enforce_sorted=False)
        
        # LSTM处理
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # 如果是双向LSTM，连接最后一层的前向和后向隐藏状态
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        # hidden: [batch size, hidden dim * num directions]
            
        # Dropout
        hidden = self.dropout(hidden)
        
        # 全连接层
        output = self.fc(hidden)
        # output: [batch size, output dim]
        
        return output
```

### 序列到序列模型 (Seq2Seq)

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch size, src len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch size, src len, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input: [batch size]
        # hidden: [n layers, batch size, hid dim]
        # cell: [n layers, batch size, hid dim]
        
        input = input.unsqueeze(1)
        # input: [batch size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch size, 1, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [batch size, 1, hid dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch size, output dim]
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch size, src len]
        # trg: [batch size, trg len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # 存储每个时间步的预测
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码器前向传播
        hidden, cell = self.encoder(src)
        
        # 第一个解码器输入是<sos>标记
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # 存储预测
            outputs[:, t] = output
            
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 获取最高概率的预测词
            top1 = output.argmax(1)
            
            # 如果使用教师强制，使用实际目标作为下一个输入
            # 否则使用预测的词
            input = trg[:, t] if teacher_force else top1
            
        return outputs
```

## Transformer架构

### 简化版Transformer

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None,
                memory_mask=None, memory_key_padding_mask=None):
        
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        # 转换维度为 [src_len, batch_size]
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transformer处理
        output = self.transformer(src, tgt, src_mask, tgt_mask,
                                 None, src_padding_mask, tgt_padding_mask,
                                 memory_key_padding_mask)
        
        # 输出层
        output = self.output_layer(output)
        
        # 转换回 [batch_size, seq_len, vocab_size]
        output = output.transpose(0, 1)
        
        return output


# 创建Transformer掩码
def generate_square_subsequent_mask(sz):
    """生成方形后续掩码"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


## 生成对抗网络

### 基础GAN

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

### DCGAN (深度卷积GAN)

```python
class DCGenerator(nn.Module):
    def __init__(self, latent_dim, channels=3):
        super().__init__()
        
        self.init_size = 32 // 4  # 初始大小
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # 计算输出维度
        ds_size = 32 // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
```

## 自编码器

### 基础自编码器

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()  # 如果输入是归一化到[0,1]的图像
        )
    
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        # 解码
        decoded = self.decoder(encoded)
        return decoded
```

### 变分自编码器 (VAE)

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# VAE损失函数
def vae_loss(recon_x, x, mu, log_var):
    # 重构损失
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD
```

## 图神经网络

### 图卷积网络 (GCN)

```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        # x: 节点特征矩阵 [num_nodes, in_features]
        # adj: 邻接矩阵 [num_nodes, num_nodes]
        
        # 归一化邻接矩阵
        D = torch.sum(adj, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        
        # 图卷积操作
        support = self.linear(x)
        output = torch.mm(adj_norm, support)
        
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
```

### 图注意力网络 (GAT)

```python
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        
        # 计算注意力系数
        e = self._prepare_attentional_mechanism_input(Wh)
        
        # 掩码和softmax
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用注意力
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_features)
        # self.a.shape (2 * out_features, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        
        # 广播加法
        e = Wh1 + Wh2.transpose(0, 1)
        return self.leakyrelu(e)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super().__init__()
        self.dropout = dropout
        
        # 多头注意力层
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])
        
        # 输出层
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout,
                               alpha=alpha, concat=False)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 多头注意力
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 输出层
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)
```

## 模型使用示例

### 训练循环模板

```python
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    print('Finished Training')


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')
    return accuracy
```

## 最佳实践和注意事项

1. 模型初始化
   - 使用适当的权重初始化方法
   - 考虑使用预训练模型
   - 注意模型参数的数量和计算复杂度

2. 训练技巧
   - 使用适当的学习率调度
   - 实施早停机制
   - 使用梯度裁剪防止梯度爆炸
   - 使用适当的批量大小

3. 性能优化
   - 使用GPU加速
   - 实施混合精度训练
   - 使用数据并行进行多GPU训练
   - 优化数据加载流程

4. 调试建议
   - 从简单模型开始
   - 使用小数据集进行快速迭代
   - 监控损失和梯度
   - 使用可视化工具分析模型行为

记住：选择合适的模型架构是解决问题的关键。根据具体任务和数据特点选择或组合上述模型架构。