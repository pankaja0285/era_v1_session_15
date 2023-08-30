import os
import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        # initialize alpha as a learnable parameter
        self.alpha = nn.Parameter(torch.ones(1))
        # initialize bias as a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # keep the dimension for broadcasting
        # (batch, seq_len, 1)
        mean = x.mean(dim = -1, keepdim = True)
        # keep the dimension for broadcasting
        # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True)
        # eps is to prevent division by zero or when standard deviation is very small
        return self.alpha * (x - mean) / (std * self.eps) * self.bias

# Feed forward network - convert head output to new output - mixing it
class FeedForwardBlock(nn.Module):
    # NOTE: if we are training very large model then the value of d_model
    #       has to be large like ~10000
    def __init__(self, d_model: int, d_ff: int, drop_out: float) -> None:
        super().__init__()
        # we are using 2 Linear layers to squeeze and expand before mixing
        # w1, b1
        self.linear_1 = nn.Linear(d_model, d_ff)
        # dropout
        self.dropout = nn.Dropout(drop_out)
        # w2, b2
        self.linear_2 = nn.Linear(d_model, d_ff)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) tp scale the embeddings acc: to the paper
        # Note: how the they're able to achieve the numerical stability otherwise 
        # the numbers become none and they were not able to train
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(drop_out)
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # create a vector of shape (d_model)
        # 10000 is chosen to just make the exponential is as small as possible
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sine to even indices
        # sin(position * (10000 ** (21 / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cosine to odd indices
        # cos(position * (10000 ** (21 / d_model)))
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension to the positional encoding
        # basically adding a dimension in the zero-th position
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the positional encoding as a buffer 
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x is the word embedding
        # (batch, seq_len, d_model)
        # x.shape[1] = seq_len - this we are recalculating instead of
        # setting as seq_len as in next session we are going to use this
        # way to speed up...
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
# All positional embeddings are got after the PositionalEncoding above

# Next is Residual layer
# input to this after Attention
class ResidualConnection(nn.Module):
    def __init__(self, drop_out: float) -> None:
        super().__init__()
        self.dropout = drop_out
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, drop_out: float) -> None:
        super().__init__()
        # embedding vector size
        self.d_model = d_model
        # number of heads
        self.h = h
        # make sure embedding vector size is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        # setup other params
        # d_k = dimension of vectors divided by number of heads
        # dimension of vector as seen by each head
        self.d_k = d_model // h

        # initialize w_q, w_k, w_v and w_o
        default_nn = nn.Linear(d_model, d_model, bias=False)
        self.w_q = default_nn
        self.w_k = default_nn
        self.w_v = default_nn
        self.w_o = default_nn
        self.dropout = nn.Dropout(drop_out)

       

    @staticmethod
    def attention(query, key, value, mask, drop_out:nn.Dropout):
        """
        Static method to calculate attention scores
        Args:
            mask: can be encoder or decoder mask
        """
        d_k = query.shape[-1]

        # apply the formula for attention scores
        # @ - matrix multiplication        
        # relationship of 32 words with 32 words
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # set ad low a value, as possible 
            # - indicating -inf in the positions where mask==0
            # (takes care of both encoder and decoder mask)
            attention_scores.mask_fill_(mask==0, -1e9)
        # no normalized meaning as such, hence we are applying softmax
        # - to establish relationship between words 
        # dim = -1 means the last dimension
        attention_scores = attention_scores.softmax(dim=-1) 
        if drop_out is not None:
            attention_scores = drop_out(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return the attention scores which are to be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        # (batch, seq_len. d_model) --> (batch, seq_len, d_model)          
        key = self.w_k(k)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        # self.h = 8 (heads)
        # self.d_k = from before
        # view function to view or reshape
        # (32, 350, 512).view(shape 0, shape1, 8, 64)

        # b, sl, dm >> b, sl, h, dm/h >> b, h, sl, dm/h
        # we do this to keep each of the heads separate
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # -1 is the d_k
        # for each word we get 6 X 6 values
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h, self.d_k)
        # x = batch, h, seq_len, d_k << transpose
        # batch, seq_len, h, d_k << view
        # batch, seq_len, h*d_k
        # if we use batch, -1, h*d_k then we don't calculate the dimension
        # whatever is left will be returned by view
        # so seq_len will come

        # multiply by w_o
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                    feed_forward_block: FeedForwardBlock,
                    drop_out: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList((ResidualConnection(drop_out) for _ in range(2)))
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, 
                                         lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        # 2 skip connections
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        # we have 6 layers - so 6 encoders
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                   cross_attention_block: MultiHeadAttentionBlock, 
                   feed_forward_block: FeedForwardBlock,
                   drop_out: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList((ResidualConnection(drop_out) for _ in range(3)))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # decoder input treated by itself, even before encoder input comes in
        # we use a target mask and we don't allow the decoder to see the next word which it is 
        # going to predict for the cross-attention block - so we are basically not going to 
        # refer to the source mask nor the padding in the source - that's also useless
        # we need to look at the whole sentence before we can predict
        x = self.residual_connections[0](x, 
                                         lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # x- query, x- key, x- value coming from encoder output
        x = self.residual_connections[1](x, 
                                         lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        # 3 skip connections
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forrward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        # list of words in our data dictionary
        return torch.log_softmax(self.proj(x), dim = -1)
    
# transformers - BERT, BART and chatGPT
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,                 
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos, tgt_pos, projection_layer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # BERT - is encoder ONLY model
        # so if we stop here, add the projection layer - we are done
        # (batch, seq_len, d_model)
        src = self.src_embed
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # chatGPT
        # the moment we add both i.e. BERT and GPT, it becomes BART
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return tgt
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        """
        A normal Transformer we're going to add a projection layer on top which is going to predict 
        that total number of words which we are looking. if I have the pass size 
        of one that means my whole encoder in the 15 of forward stage my encoder will go through 
        that one sentence once and then the recorder I will create those multiple decoders so
        that will multiple decoders that would mean that I have processed at one sentence is 
        that correct no no no decoder you still need multiple ones decoder we need way more 
        actually so think of it like this so I have a batch size of one but I'm selling in 
        five words...
        """

        # 5 words = 6 decoders 5 >> 6 <-- 6 decoders
        # 10 sentences with with 5 words - 60 decoders 10 >> 5 >> 6 <-- 60 decoders
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int,
                      tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, 
                      drop_out: float=0.1, d_ff: int=2048) -> Transformer:
    """
    Build transformer
    Args:
        src_vocab_size: gives the size of the src embedding
        d_ff: we are expanding 4 times (has huge affect on overall processing)
    """
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, drop_out)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, drop_out)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, drop_out)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, drop_out)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, drop_out)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks - 6 decoder blocks chained together
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, drop_out)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, drop_out)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, drop_out)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, drop_out)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    # --> this is the chaining part
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos,
                                   projection_layer)
    
    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
    
