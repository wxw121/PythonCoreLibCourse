#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch自然语言处理教程

本模块涵盖PyTorch在自然语言处理领域的应用，包括：
1. 文本数据处理
2. 词嵌入(Word Embeddings)
3. 循环神经网络(RNN)
4. LSTM和GRU
5. Transformer基础
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import re


def text_processing() -> None:
    """
    文本数据处理基础
    
    展示如何处理和准备文本数据用于深度学习
    """
    print("\n" + "=" * 50)
    print("文本数据处理基础".center(50))
    print("=" * 50)
    
    # 1. 文本预处理
    print("\n1. 文本预处理:")
    
    # 示例文本
    text = """PyTorch is an open source machine learning library based on the Torch library,
    used for applications such as computer vision and natural language processing!"""
    
    def preprocess_text(text: str) -> List[str]:
        """
        文本预处理函数
        
        Args:
            text: 输入文本

        Returns:
            处理后的词列表
        """
        # 转换为小写
        text = text.lower()

        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)

        # 分词
        words = text.split()

        return words

    processed_words = preprocess_text(text)
    print("原始文本:", text)
    print("处理后的词:", processed_words)

    # 2. 构建词汇表
    print("\n2. 构建词汇表:")

    class Vocabulary:
        """词汇表类"""

        def __init__(self):
            self.word2idx = {}  # 词到索引的映射
            self.idx2word = {}  # 索引到词的映射
            self.word_freq = Counter()  # 词频统计
            self.add_token("<PAD>")  # 填充标记
            self.add_token("<UNK>")  # 未知词标记

        def add_token(self, token: str) -> int:
            """添加词到词汇表"""
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
            return self.word2idx[token]

        def add_words(self, words: List[str]) -> None:
            """添加多个词到词汇表"""
            for word in words:
                self.word_freq[word] += 1
                self.add_token(word)

        def __len__(self) -> int:
            """返回词汇表大小"""
            return len(self.word2idx)

        def encode(self, words: List[str]) -> List[int]:
            """将词列表转换为索引列表"""
            return [self.word2idx.get(word, self.word2idx["<UNK>"])
                    for word in words]

        def decode(self, indices: List[int]) -> List[str]:
            """将索引列表转换为词列表"""
            return [self.idx2word[idx] for idx in indices]

    # 创建词汇表实例
    vocab = Vocabulary()
    vocab.add_words(processed_words)

    print(f"词汇表大小: {len(vocab)}")
    print(f"词频统计: {dict(vocab.word_freq)}")

    # 编码和解码示例
    encoded = vocab.encode(processed_words[:5])
    decoded = vocab.decode(encoded)
    print(f"原始词: {processed_words[:5]}")
    print(f"编码后: {encoded}")
    print(f"解码后: {decoded}")

    # 3. 文本数据集
    print("\n3. 文本数据集:")

    class TextDataset(Dataset):
        """文本数据集类"""

        def __init__(self, texts: List[str], vocab: Vocabulary, max_length: int = 50):
            """
            初始化数据集

            Args:
                texts: 文本列表
                vocab: 词汇表
                max_length: 序列最大长度
            """
            self.texts = texts
            self.vocab = vocab
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> torch.Tensor:
            # 预处理文本
            words = preprocess_text(self.texts[idx])

            # 转换为索引
            indices = self.vocab.encode(words)

            # 截断或填充到固定长度
            if len(indices) > self.max_length:
                indices = indices[:self.max_length]
            else:
                indices += [self.vocab.word2idx["<PAD>"]] * (self.max_length - len(indices))

            return torch.tensor(indices)

    # 示例文本数据
    texts = [
        "PyTorch is amazing for deep learning",
        "Natural language processing is fascinating",
        "Deep learning revolutionizes AI"
    ]

    # 创建数据集
    dataset = TextDataset(texts, vocab)
    print(f"数据集大小: {len(dataset)}")
    print(f"第一个样本: {dataset[0]}")

    # 4. 批处理和填充
    print("\n4. 批处理和填充:")

    def collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批处理函数

        Args:
            batch: 样本列表

        Returns:
            批次张量和长度张量
        """
        # 获取序列长度
        lengths = torch.tensor([len(seq) for seq in batch])

        # 创建填充后的批次
        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

        return padded, lengths

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # 获取一个批次
    batch_data, batch_lengths = next(iter(dataloader))
    print(f"批次形状: {batch_data.shape}")
    print(f"序列长度: {batch_lengths}")


def word_embeddings() -> None:
    """
    词嵌入(Word Embeddings)

    展示如何使用PyTorch实现和训练词嵌入
    """
    print("\n" + "=" * 50)
    print("词嵌入(Word Embeddings)".center(50))
    print("=" * 50)

    # 1. 词嵌入层
    print("\n1. 词嵌入层:")

    vocab_size = 1000
    embedding_dim = 100

    # 创建词嵌入层
    embedding = nn.Embedding(vocab_size, embedding_dim)
    print(f"词嵌入层参数形状: {embedding.weight.shape}")

    # 使用词嵌入层
    input_indices = torch.tensor([1, 5, 3, 2])
    embedded = embedding(input_indices)
    print(f"输入形状: {input_indices.shape}")
    print(f"嵌入后形状: {embedded.shape}")

    # 2. 预训练词嵌入
    print("\n2. 预训练词嵌入:")
    print("常用的预训练词嵌入:")
    print("- Word2Vec")
    print("- GloVe")
    print("- FastText")

    # 加载预训练词嵌入示例代码
    print("\n加载预训练词嵌入的步骤:")
    print("""
    # 1. 下载预训练词嵌入文件
    # 2. 解析词嵌入文件
    embeddings_dict = {}
    with open('pretrained_embeddings.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.FloatTensor([float(x) for x in values[1:]])
            embeddings_dict[word] = vector
            
    # 3. 创建嵌入矩阵
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    for word, idx in vocab.word2idx.items():
        if word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
            
    # 4. 加载到嵌入层
    embedding.weight.data.copy_(embedding_matrix)
    """)

    # 3. 简单词嵌入模型
    print("\n3. 简单词嵌入模型:")

    class SimpleEmbeddingModel(nn.Module):
        """简单的词嵌入模型"""

        def __init__(self, vocab_size: int, embedding_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x形状: [batch_size]
            embedded = self.embedding(x)  # [batch_size, embedding_dim]
            output = self.linear(embedded)  # [batch_size, vocab_size]
            return output

    # 创建模型实例
    model = SimpleEmbeddingModel(vocab_size, embedding_dim)
    print("模型结构:")
    print(model)

    # 4. 词嵌入的应用
    print("\n4. 词嵌入的应用:")
    print("词嵌入常见应用:")
    print("- 文本分类")
    print("- 情感分析")
    print("- 命名实体识别")
    print("- 机器翻译")
    print("- 文本生成")


def rnn_basics() -> None:
    """
    循环神经网络(RNN)基础

    展示PyTorch中RNN的基本用法
    """
    print("\n" + "=" * 50)
    print("循环神经网络(RNN)基础".center(50))
    print("=" * 50)

    # 1. RNN基础组件
    print("\n1. RNN基础组件:")

    # 基本RNN层
    input_size = 100
    hidden_size = 128
    num_layers = 2

    rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    print("RNN层参数:")
    print(f"- 输入大小: {input_size}")
    print(f"- 隐藏层大小: {hidden_size}")
    print(f"- 层数: {num_layers}")

    # LSTM层
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    print("\nLSTM层结构与RNN相似，但有更复杂的门控机制")

    # GRU层
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    print("\nGRU是LSTM的简化版本，性能相似但参数更少")

    # 2. 使用RNN处理序列
    print("\n2. 使用RNN处理序列:")

    # 创建示例数据
    batch_size = 3
    seq_length = 5

    # 输入序列
    x = torch.randn(batch_size, seq_length, input_size)

    # 初始隐藏状态
    h0 = torch.zeros(num_layers, batch_size, hidden_size)

    # 前向传播
    output, hn = rnn(x, h0)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hn.shape}")

    # 3. 实现语言模型
    print("\n3. 实现语言模型:")

    class LanguageModel(nn.Module):
        """简单的语言模型"""

        def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, Tuple]:
            # x形状: [batch_size, seq_length]
            embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
            output, hidden = self.rnn(embedded, hidden)
            output = self.fc(output)  # [batch_size, seq_length, vocab_size]
            return output, hidden

        def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """初始化隐藏状态"""
            weight = next(self.parameters())
            return (weight.new_zeros(1, batch_size, self.rnn.hidden_size),
                    weight.new_zeros(1, batch_size, self.rnn.hidden_size))

    # 创建模型实例
    model = LanguageModel(vocab_size=1000, embedding_dim=100, hidden_size=128)
    print("语言模型结构:")
    print(model)

    # 4. RNN的优缺点
    print("\n4. RNN的优缺点:")
    print("优点:")
    print("- 可以处理变长序列")
    print("- 参数共享")
    print("- 适合序列建模")

    print("\n缺点:")
    print("- 训练困难（梯度消失/爆炸）")
    print("- 难以捕获长距离依赖")
    print("- 计算效率较低")
    print("- 不能并行计算")
    
    # 5. RNN最佳实践
    print("\n5. RNN最佳实践:")
    print("- 使用LSTM或GRU代替普通RNN")
    print("- 使用梯度裁剪防止梯度爆炸")
    print("- 使用残差连接处理长序列")
    print("- 使用双向RNN捕获双向上下文")
    print("- 使用注意力机制增强长距离依赖的建模")


def transformer_basics() -> None:
    """
    Transformer基础
    
    展示PyTorch中Transformer的基本组件和用法
    """
    print("\n" + "=" * 50)
    print("Transformer基础".center(50))
    print("=" * 50)
    
    # 1. Transformer架构
    print("\n1. Transformer架构:")
    print("Transformer的主要组件:")
    print("- 多头自注意力机制")
    print("- 位置编码")
    print("- 前馈神经网络")
    print("- 残差连接和层归一化")
    print("- 编码器-解码器架构")
    
    # 2. 实现多头注意力
    print("\n2. 多头注意力机制:")
    
    class MultiHeadAttention(nn.Module):
        """多头注意力层"""
        
        def __init__(self, d_model: int, num_heads: int):
            super().__init__()
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # 创建Q、K、V的线性变换
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
        def scaled_dot_product_attention(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """计算缩放点积注意力"""
            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, V)
            
            return output
        
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            batch_size = query.size(0)
            
            # 线性变换
            Q = self.W_q(query)
            K = self.W_k(key)
            V = self.W_v(value)
            
            # 分割成多个头
            Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # 计算注意力
            output = self.scaled_dot_product_attention(Q, K, V, mask)
            
            # 合并多头的结果
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            
            # 最后的线性变换
            output = self.W_o(output)
            
            return output
    
    # 3. 位置编码
    print("\n3. 位置编码:")
    
    class PositionalEncoding(nn.Module):
        """位置编码层"""
        
        def __init__(self, d_model: int, max_seq_length: int = 5000):
            super().__init__()
            
            # 创建位置编码矩阵
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1)]
    
    # 4. Transformer编码器层
    print("\n4. Transformer编码器层:")
    
    class TransformerEncoderLayer(nn.Module):
        """Transformer编码器层"""
        
        def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
            super().__init__()
            
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # 多头自注意力
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            
            # 前馈网络
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            
            return x
    
    # 5. 简单的Transformer模型
    print("\n5. 简单的Transformer模型:")
    
    class SimpleTransformer(nn.Module):
        """简单的Transformer模型"""
        
        def __init__(
            self,
            vocab_size: int,
            d_model: int = 512,
            num_heads: int = 8,
            num_layers: int = 6,
            d_ff: int = 2048,
            max_seq_length: int = 5000,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
            
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            
            self.fc = nn.Linear(d_model, vocab_size)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # 嵌入和位置编码
            x = self.embedding(x)
            x = self.positional_encoding(x)
            x = self.dropout(x)
            
            # 编码器层
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x, mask)
            
            # 输出层
            output = self.fc(x)
            return output
    
    # 创建模型实例
    model = SimpleTransformer(vocab_size=1000)
    print("Transformer模型结构:")
    print(model)
    
    # 6. Transformer的优势
    print("\n6. Transformer的优势:")
    print("- 可以并行计算")
    print("- 更好地处理长距离依赖")
    print("- 注意力机制提供了更好的可解释性")
    print("- 训练更稳定")
    print("- 已成为现代NLP的基础架构")


def main():
    """运行所有PyTorch NLP示例"""
    print("\n" + "=" * 80)
    print("PyTorch自然语言处理教程".center(80))
    print("=" * 80)
    
    # 运行各个部分的示例
    text_processing()
    word_embeddings()
    rnn_basics()
    transformer_basics()


if __name__ == "__main__":
    main()