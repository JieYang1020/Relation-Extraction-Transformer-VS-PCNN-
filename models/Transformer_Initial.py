# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
'''
分类问题不用decoder
'''
####################scaled dot_product attention####################
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism"""
    
    def __init__(self, attention_dropout = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim = 2)
    
    def forward(self, q, k, v, scale = None, attn_mask = None):
        '''前向传播。scale:缩放因子（1/(根号K)），一个浮点标量。返回：上下文张量和
        attention张量'''
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            #给需要mask的地方设置负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        #计算softmax
        attention = self.softmax(attention)
        #添加dropout
        attention = self.dropout(attention)
        #和V做点积
        context = torch.bmm(attention, v)
        return context, attention

####################multi-head attention####################
#将Q,K,V拆分为8份(heads=8)，每份分别进行scaled dot-product attention
class MultiHeadAttention(nn.Module):
    
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        #dim_per_head:q, k, v向量的长度.Wq, Wk, Wv 的矩阵尺寸为model_dim*dim_per_head
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi_head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forwad(self, key, value, query, attn_mask = None):
        #残差连接，增加任意常数，求导为1，避免反向传播时的梯度消失
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        #linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        #split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)
        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        scale = (key.size(-1) // num_heads) ** -0.5
        #调用之前的scaled dot product attention
        context, attention = self.dot_product_attention(query, key, value,
                                                       scale, attn_mask)
        #将8个头的结果连接
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        #final linear projection
        output = self.linear_final(context)
        #dropout
        output = self.dropout(output)
        #add residual and norm layer
        output = self.layer_norm(residual + output)
        return output, attention

####################残差连接####################
def residual(sublayer_fn, x):
    return sublayer_fn(x)+x
####################mask####################
#encoding的时候需要先padding
def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask
#decoding的时候不能看见未来的信息，用sequence_mask，将未来信息隐藏掉
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                     diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

####################positional encoding####################
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        '''d_model:模型维度 model_dim=512。max_seq_len:文本序列最大长度
        '''
        super(PositionEncoding, self).__init__()
                # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
          [pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
                # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = torch.tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)

####################word embedding####################
# embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
# # 获得输入的词嵌入编码
# seq_embedding = seq_embedding(inputs)*np.sqrt(d_model)


####################position-wise feed-forwad network####################
import torch
import torch.nn as nn
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1,2))
        
        #add residual and norm layer
        output = self.layer_norm(x + output)
        return output

####################PositionalEncoding层####################
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)

####################Encoder层####################
class EncoderLayer(nn.Module):
    def __init__(self, model_dim = 512, num_heads = 8, ffn_dim = 2018,
                 dropout = 0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
    def forward(self, inputs, attn_mask = None):
        #self attention
        context, attention = self.attention(inputs, inputs, inputs, padding_mask)
        #feed forward network
        output = self.feed_forward(context)
        return output, attention
class Encoder(nn.Module):
    '''多层EncoderLayer组成Encoder'''
    def __init__(self, vocab_size, max_seq_len, num_layers = 6, model_dim = 512, 
                num_heads = 8, ffn_dim = 2048, dropout = 0.0):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads,
                                                          ffn_dim,dropout) for 
                                            _ in range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, 
                                          padding_idx = 0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)
        self_attention_mask = padding_mask(inputs, inputs)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
        return output, attentions

####################Decoder层####################
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads = 8, ffn_dim = 2048,
                dropout = 0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
    def forward(self, dec_inputs, enc_outputs, self_attn_mask = None,
                context_attn_mask = None):
        #self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
        dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        #context attention
        #query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(enc_outputs, enc_output,
                                                      dec_output, context_attn_mask)
        #decoder's output, or context
        dec_output = self.feed_forward(dec_output)
        return dec_output, self_attention, context_attention
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers = 6, model_dim = 512,
                num_heads = 8, ffn_dim = 2048, dropout = 0.0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads,
                                                         ffn_dim, dropout) for _
                                            in range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, 
                                          padding_idx = 0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)
        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.ft((self_attention_padding_mask + seq_mask), 0)
        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output, enc_output, 
                                                      self_attn_mask,
                                                     context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)
        return output, self_attentions, context_attentions

###################组合Encoder和Decoder层####################
class Transformer_self(nn.Module):
    def __init_(self, src_vocab_size, src_max_len,
               num_layers = 6, model_dim = 512, num_heads = 8, 
               ffn_dim = 2048, dropout = 0.2):
        super(Transformer_self, self).__init__()
        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, 
                              model_dim, num_heads, ffn_dim, dropout)
        # self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers,
        #                       model_dim, num_heads, ffn_dim, dropout)
        # self.linear = nn.Linear(model_dim, tgt_vocab_size, bias = False)
        # self.softmax = nn.Softmax(dim = 2)
    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)
        output, enc_self_attn = self.encoder(src_seq, src_len)
        # output, dec_self_attn, ctx_attn = self.decoder(tgt_seq, tgt_len, output,
        #                                                context_attn_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output
        # return output, enc_self_attn, dec_self_attn, ctx_attn

