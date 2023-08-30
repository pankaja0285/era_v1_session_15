import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len(self):
        return len(self.ds)
            
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # transform the text into tokens - getting the ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids

        # add SOS, EOS and padding to each sentence
        #  we will add <s> and </s>
        # e.g. 350 -5 - 2 <- this -2 is end of sentence and start of sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 
        # we will add <s> and </s> only on the label
        # we allow the decoder to predict the end of sentence hence -1, we want the decoder
        # to stop automatically
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # make sure the number of padding tokens is not negative. If it is, the sentence
        # is too long 
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # add <s> and </s> token
        # we are concatenating sos_token, tensor for the enc_input_tokens,
        # eos_token, tensor for the enc_num_padding_tokens - whatever remaining
        # starts with a random tensor but gets trained
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # add only <s> token - i.e. start token
        # we are assuming seq_len = 350 for both encoder and decoder
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        # add only </s> token
        # label is the one that will enable our decoder run in parallel steps
        # below is basically one full sentence kept together
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # double check the size of the tensors to make sure they are all seq_len in length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # let's say my uh sentence was Edmund has closed his bank account
        # so Edmund which is to encoder input is it equal to pad sequence right now
        # - what get sent is true true true true true true true
        #   and then false false false false false still the sequence left 
        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1, seq_len)
            # SOS - EDMUND  IS  CLOSING HIS BANK ACCOUNT EOS PAD PAD PAD PAD PAD PAD PAD PAD ...
            # True  True   True True    True True True   False False False False False ...
            # below amounts to - (1, seq_len) & (1, seq_len, seq_len)
            # we end up with a 2-dim 
            # e.g. seq_len 10
            # SOS  I    GOT  A  CAT PAD PAD PAD PAD PAD 
            # True True True True True False False False False False
            # 1    1     1    1    1   0      0     0      0    0

            # 1    1     1    1    1   1      1     1      1    1
            # 0    1     1    1    1   1      1     1      1    1
            # 0    0     1    1    1   1      1     1      1    1  
            # 0    0     0    1    1   1      1     1      1    1
            # 0    0     0    0    1   1      1     1      1    1
            # 0    0     0    0    0   1      1     1      1    1
            # 0    0     0    0    0   0      1     1      1    1
            # 0    0     0    0    0   0      0     1      1    1
            # 0    0     0    0    0   0      0     0      1    1
            # 0    0     0    0    0   0      0     0      0    1

            # product is 
            # 1    1     1    1    1   0      0     0      0    0
            # 0    1     1    1    1   0      0     0      0    0
            # 0    0     1    1    1   0      0     0      0    0  
            # 0    0     0    1    1   0      0     0      0    0
            # 0    0     0    0    1   0      0     0      0    0
            # 0    0     0    0    0   0      0     0      0    0
            # 0    0     0    0    0   0      0     0      0    0
            # 0    0     0    0    0   0      0     0      0    0
            # 0    0     0    0    0   0      0     0      0    0
            # 0    0     0    0    0   0      0     0      0    0
            
            "decoder_mask": (encoder_input != self.pad_token).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0)),
            "label": label, # (sdeq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
    def causal_mask(size):
        # size is seq_len here
        # upper triangle of ones
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0
    