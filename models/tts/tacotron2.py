import os
import random
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, List, Dict, Optional, Union, Any
from core.trainer.wandb_logger import WandbLogger
from models import BaseModel

from utils.plotting import saveplot_gate, saveplot_mel, saveplot_alignment
from configs import TextConfig, AudioConfig, TrainerConfig, BaseConfig
from configs.models import Tacotron2Config
from models.tts import TTSModel
from models.generic import LinearNorm, ConvNorm

def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, dtype=lengths.dtype, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool() # [max_len], [B] -> [max_len], [B, 1] -> [B, max_len] = [B, n_tok/n_frames]
    return ~mask # ~(torch.bool) inverts the boolean values

class LocationLayer(nn.Module):
    def __init__(
            self,
            attention_n_filters: int,
            attention_kernel_size: int,
            attention_dim: int
        ):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1
        )
        self.location_dense = LinearNorm(
            in_dim=attention_n_filters,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh'
        )

    def forward(self, attention_weights_cat):
        # attention_weights_cat: [B, 2, n_tok]
        processed_attention = self.location_conv(attention_weights_cat) # [B, 2, n_tok] -> [B, att_n_fil, n_tok]
        processed_attention = processed_attention.transpose(1, 2) # [B, att_n_fil, n_tok] -> [B, n_tok, att_n_fil]
        processed_attention = self.location_dense(processed_attention) # [B, n_tok, att_n_fil] -> [B, n_tok, att_dim]
        return processed_attention # [B, n_tok, att_dim]

class Attention(nn.Module):
    def __init__(
            self,
            attention_rnn_dim: int,
            embedding_dim: int,
            attention_dim: int,
            attention_location_n_filters: int,
            attention_location_kernel_size: int
        ):
        super().__init__()
        self.query_layer = LinearNorm(
            in_dim=attention_rnn_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        self.memory_layer = LinearNorm(
            in_dim=embedding_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        self.v = LinearNorm(
            in_dim=attention_dim,
            out_dim=1,
            bias=False
        )
        self.location_layer = LocationLayer(
            attention_n_filters=attention_location_n_filters,
            attention_kernel_size=attention_location_kernel_size,
            attention_dim=attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
            self,
            query,
            processed_memory,
            attention_weights_cat
        ):
        # query: [B, att_rnn_dim]
        # processed_memory: [B, n_tok, att_dim]
        # attention_weights_cat: [B, 2, n_tok]
        processed_query = self.query_layer(query.unsqueeze(1)) # [B, att_rnn_dim] -> [B, 1, att_rnn_dim] -> [B, 1, att_dim]
        processed_attention_weights = self.location_layer(attention_weights_cat) # [B, 2, n_tok] -> [B, n_tok, att_dim]
        # [B, 1, att_dim] + [B, n_tok, att_dim] + [B, n_tok, att_dim] = [B, n_tok, att_dim] (broadcasting to match dim)
        # [B, n_tok, 1] <- [B, n_tok, att_dim]
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1) # [B, n_tok, 1] -> # [B, n_tok]
        return energies # (alignment): # [B, n_tok]

    def forward(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat,
            mask
        ):
        # attention_hidden_state (attention rnn last output): [B, att_rnn_dim]
        # memory (encoder outputs): [B, n_tok, enc_dim]
        # processed_memory (processed encoder outputs): [B, n_tok, att_dim]
        # attention_weights_cat (previous and cummulative attention weights): [B, 2, n_tok]
        # mask (binary mask for padded data): [B, n_tok]
        alignment = self.get_alignment_energies(
            query=attention_hidden_state, # [B, att_rnn_dim]
            processed_memory=processed_memory, # [B, n_tok, att_dim]
            attention_weights_cat=attention_weights_cat # [B, 2, n_tok]
        ) # [B, n_tok]
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value) # [B, n_tok]
        attention_weights = F.softmax(alignment, dim=1) # [B, n_tok]
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # [B, 1, n_tok], [B, n_tok, enc_dim] -> [B, 1, enc_dim]
        attention_context = attention_context.squeeze(1) # [B, 1, enc_dim] -> [B, enc_dim]
        return attention_context, attention_weights # [B, enc_dim], [B, n_tok]

class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: List[int]):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            LinearNorm(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ]) # n_mels -> pre_dim, pre_dim -> pre_dim

    def forward(self, x):
        # x: [n_frames + 1, B, n_mels]
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x # [*, B, pre_dim]

class Postnet(nn.Module):
    def __init__(self, n_mels: int, postnet_n_convolutions: int, postnet_embedding_dim: int, postnet_kernel_size: int):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=n_mels, 
                    out_channels=postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='tanh'
                ),
                nn.BatchNorm1d(postnet_embedding_dim)
            )
        )
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=postnet_embedding_dim,
                        out_channels=postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain='tanh'
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim)
                )
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=postnet_embedding_dim,
                    out_channels=n_mels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='linear'
                ),
                nn.BatchNorm1d(n_mels)
            )
        )

    def forward(self, x):
        # x: # [B, n_mels, n_frames]
        # [B, n_mels, n_frames] -> [B, postnet_emb_dim, n_frames] -> [B, postnet_emb_dim, n_frames] ... X (n_conv - 2) times
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training) # [B, postnet_emb_dim, n_frames] -> [B, n_mels, n_frames] (without tanh)
        return x # [B, n_mels, n_frames]

class Encoder(nn.Module):
    def __init__(self, encoder_n_convolutions: int, encoder_embedding_dim: int, encoder_kernel_size: int):
        super().__init__()
        self.convolutions = nn.ModuleList()
        for _ in range(encoder_n_convolutions):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=encoder_embedding_dim,
                        out_channels=encoder_embedding_dim,
                        kernel_size=encoder_kernel_size,
                        stride=1,
                        padding=int((encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain='relu'
                    ),
                    nn.BatchNorm1d(encoder_embedding_dim)
                )
            )

        self.lstm = nn.LSTM(
            input_size=encoder_embedding_dim,
            hidden_size=int(encoder_embedding_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.lstm.flatten_parameters() # to optimize weights and operations (keep in forward method if using nn.DataParallel)

    def forward(self, x, input_lengths):
        # x: [B, sym_dim, n_tok], input_lengths: [B]
        # sym_dim = enc_dim => x: [B, enc_dim, n_tok]
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training) # [B, enc_dim, n_tok] -> [B, enc_dim, n_tok] (remains same -> in_channels = out_channels)
        x = x.transpose(1, 2) # [B, enc_dim, n_tok] -> [B, n_tok, enc_dim]
        
        input_lengths = input_lengths.cpu().numpy() # pytorch tensor are not reversible, hence the conversion, [B]
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True) # packed([B, n_tok, enc_dim], [B])
        outputs, _ = self.lstm(x) # outputs, (h_n, c_n)
        # outputs => [B, L, D * H_out] (L = n_tok)
        # H_out = hidden_size (since proj_size is default 0) => H_out = enc_dim / 2
        # D = 2 if bidirectional else 1
        # => outputs: [B, n_tok, enc_dim]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # returns padded sequences and lengths => [B, n_tok, enc_dim], [B]
        return outputs # [B, n_tok, enc_dim]

    def inference(self, x):
        # x:  [1, sym_dim, n_tok]
        # sym_dim = enc_dim => x: [1, enc_dim, n_tok]
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training) # # [1, enc_dim, n_tok] -> [1, enc_dim, n_tok]
        x = x.transpose(1, 2) # [1, enc_dim, n_tok] -> [1, n_tok, enc_dim]

        outputs, _ = self.lstm(x) # [1, n_tok, enc_dim] -> [1, n_tok, 2 * enc_dim / 2] => [1, n_tok, enc_dim]
        return outputs # [1, n_tok, enc_dim]

class Decoder(nn.Module):
    def __init__(
            self,
            n_mels: int,
            encoder_embedding_dim: int,
            decoder_rnn_dim: int,
            prenet_dim: int,
            max_decoder_steps: int,
            gate_threshold: float,
            p_attention_dropout: float,
            p_decoder_dropout: float,
            attention_rnn_dim: int,
            attention_dim: int,
            attention_location_n_filters: int,
            attention_location_kernel_size: int
        ):
        super().__init__()
        self.n_mel_channels = n_mels
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.attention_rnn_dim = attention_rnn_dim

        self.prenet = Prenet(self.n_mel_channels, [self.prenet_dim, self.prenet_dim])
        self.attention_rnn = nn.LSTMCell(self.prenet_dim + self.encoder_embedding_dim, self.attention_rnn_dim)
        self.attention_layer = Attention(
            self.attention_rnn_dim, 
            self.encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size
        )
        self.decoder_rnn = nn.LSTMCell(self.attention_rnn_dim + self.encoder_embedding_dim, self.decoder_rnn_dim, 1)
        self.linear_projection = LinearNorm(self.decoder_rnn_dim + self.encoder_embedding_dim, self.n_mel_channels)
        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain='sigmoid'
        )

    def initialize_decoder_states(self, memory, mask):
        # memory: [B, n_tok, enc_dim], mask: [B, n_tok]
        B, MAX_TIME = memory.size(0), memory.size(1)
        self.attention_hidden = memory.data.new(B, self.attention_rnn_dim).zero_() # [B, att_rnn_dim]
        self.attention_cell = memory.data.new(B, self.attention_rnn_dim).zero_() # [B, att_rnn_dim]
        self.decoder_hidden = memory.data.new(B, self.decoder_rnn_dim).zero_() # [B, dec_rnn_dim]
        self.decoder_cell = memory.data.new(B, self.decoder_rnn_dim).zero_() # [B, dec_rnn_dim]
        self.attention_weights = memory.data.new(B, MAX_TIME).zero_() # [B, n_tok]
        self.attention_weights_cum = memory.data.new(B, MAX_TIME).zero_() # [B, n_tok]
        self.attention_context = memory.data.new(B, self.encoder_embedding_dim).zero_() # [B, enc_dim]
        self.memory = memory # [B, n_tok, enc_dim]
        self.processed_memory = self.attention_layer.memory_layer(memory) # [B, n_tok, enc_dim] -> [B, n_tok, att_dim]
        self.mask = mask # [B, n_tok]

    def parse_decoder_inputs(self, decoder_inputs):
        decoder_inputs = decoder_inputs.transpose(1, 2) # [B, n_mels, n_frames] -> [B, n_frames, n_mels]
        decoder_inputs = decoder_inputs.transpose(0, 1) # [B, n_frames, n_mels] -> [n_frames, B, n_mels]
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # mel_outputs: [B, n_mels] * T
        # gate_outputs: [B] * T
        # alignments: [B, n_tok] * T
        # T == n_frames
        alignments = torch.stack(alignments).transpose(0, 1) # [B, n_tok] * T -> [T, B, n_tok] -> [B, T, n_tok]
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous() # [B] * T -> [T, B] -> [B, T]
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous() # [B, n_mels] * T -> [T, B, n_mels] -> [B, T, n_mels]
        mel_outputs = mel_outputs.transpose(1, 2) # [B, T, n_mels] -> [B, n_mels, T]
        return mel_outputs, gate_outputs, alignments # [B, n_mels, n_frames], [B, n_frames], [B, n_frames, n_tok]

    def decode(self, decoder_input):
        # decoder_input (previous mel frame): [B, pre_dim]

        # using previous mel frame and attention context (attention weighted sum of encoder outputs)
        # we attention_hidden which serves as query
        cell_input = torch.cat((decoder_input, self.attention_context), -1) # [B, pre_dim], [B, enc_dim] -> [B, pre_dim + enc_dim]
        # [B, att_rnn_dim], [B, att_rnn_dim] <- [B, pre_dim + enc_dim], ([B, att_rnn_dim], [B, att_rnn_dim])
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training) # [B, att_rnn_dim]

        # [B, 2, n_tok] <- [B, 1, n_tok] + [B, 1, n_tok]
        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        # [B, enc_dim], [B, n_tok]
        self.attention_context, self.attention_weights = self.attention_layer(
            attention_hidden_state=self.attention_hidden, # [B, att_rnn_dim]
            memory=self.memory, # [B, n_tok, enc_dim]
            processed_memory=self.processed_memory, # [B, n_tok, att_dim]
            attention_weights_cat=attention_weights_cat, # [B, 2, n_tok]
            mask=self.mask # [B, n_tok]
        )
        self.attention_weights_cum += self.attention_weights # [B, n_tok]
        
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1) # [B, att_rnn_dim], [B, enc_dim] -> [B, att_rnn_dim + enc_dim]
        # [B, dec_rnn_dim], [B, dec_rnn_dim] <- [B, att_rnn_dim + enc_dim], ([B, dec_rnn_dim], [B, dec_rnn_dim])
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training) # [B, dec_rnn_dim]

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1) # [B, dec_rnn_dim], [B, enc_dim] -> [B, dec_rnn_dim + enc_dim]
        decoder_output = self.linear_projection(decoder_hidden_attention_context) # [B, dec_rnn_dim + enc_dim] -> [B, n_mels]
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)  # [B, dec_rnn_dim + enc_dim] -> [B, 1]
        return decoder_output, gate_prediction, self.attention_weights # [B, n_mels], [B, 1], [B, n_tok]

    def forward(self, memory, decoder_inputs, memory_lengths):
        # memory (encoder_outputs): [B, n_tok, enc_dim]
        # decoder_inputs (batch["mel_padded"]): [B, n_mels, n_frames]
        # memory_length (token_lengths): [B]

        decoder_input = memory.data.new(memory.size(0), self.n_mel_channels).zero_().unsqueeze(0) # [1, B, n_mels]
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs) # [B, n_mels, n_frames] -> [n_frames, B, n_mels]
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0) # [n_frames + 1, B, n_mels]
        decoder_inputs = self.prenet(decoder_inputs) # [n_frames + 1, B, n_mels] -> [n_frames + 1, B, pre_dim]

        self.initialize_decoder_states(memory, mask=get_mask_from_lengths(memory_lengths)) # [B, n_tok, enc_dim], [B, n_tok]

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1: # till len(mel_outputs) == n_frames
            decoder_input = decoder_inputs[len(mel_outputs)] # [B, pre_dim] => corresponds to previously generated mel frame (teacher-forcing)
            # [B, n_mels], [B, 1], [B, n_tok] <- [B, pre_dim]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output] # += [B, n_mels]
            gate_outputs += [gate_output.squeeze(1)]  # += [B]
            alignments += [attention_weights] # += [B, n_tok]

        # [B, n_mels, n_frames], [B, n_frames], [B, n_frames, n_tok] <- [B, n_mels] * n_frames, [B] * n_frames, [B, n_tok] * n_frames
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        # memory (encoder_outputs): [1, n_tok, enc_dim]
        decoder_input = memory.data.new(memory.size(0), self.n_mel_channels).zero_() # [1, n_mels]

        self.initialize_decoder_states(memory, mask=None) # [1, n_tok, enc_dim], None

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input) # [1, n_mels] -> [1, pre_dim]
            # [1, n_mels], [1, 1], [1, n_tok] <- [1, pre_dim]
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs += [mel_output] # += [1, n_mels]
            gate_outputs += [gate_output.squeeze(1)] # += [1]
            alignments += [alignment] # += [1, n_tok]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) >= self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break
            decoder_input = mel_output

        # [1, n_mels, n_frames], [1, n_frames], [1, n_frames, n_tok] <- [1, n_mels] * n_frames, [1] * n_frames, [1, n_tok] * n_frames
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

class Tacotron2(TTSModel):
    def __init__(self, model_config: Tacotron2Config, audio_config: AudioConfig, text_config: TextConfig) -> None:
        super().__init__(model_config, audio_config, text_config)
        self.model_config: Tacotron2Config = self.model_config # for typing hints
        self.embedding = nn.Embedding(self.text_config.n_tokens, self.model_config.symbols_embedding_dim)
        std = sqrt(2.0 / (self.embedding.num_embeddings + self.embedding.embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(
            encoder_n_convolutions=self.model_config.encoder_n_convolutions,
            encoder_embedding_dim=self.model_config.encoder_embedding_dim,
            encoder_kernel_size=self.model_config.encoder_kernel_size
        )
        self.decoder = Decoder(
            n_mels=self.audio_config.n_mels,
            encoder_embedding_dim=self.model_config.encoder_embedding_dim,
            decoder_rnn_dim=self.model_config.decoder_rnn_dim,
            prenet_dim=self.model_config.prenet_dim,
            max_decoder_steps=self.model_config.max_decoder_steps,
            gate_threshold=self.model_config.gate_threshold,
            p_attention_dropout=self.model_config.p_attention_dropout,
            p_decoder_dropout=self.model_config.p_decoder_dropout,
            attention_rnn_dim=self.model_config.attention_rnn_dim,
            attention_dim=self.model_config.attention_dim,
            attention_location_n_filters=self.model_config.attention_location_n_filters,
            attention_location_kernel_size=self.model_config.attention_location_kernel_size
        )
        self.postnet = Postnet(
            n_mels=self.audio_config.n_mels,
            postnet_n_convolutions=self.model_config.postnet_n_convolutions,
            postnet_embedding_dim=self.model_config.postnet_embedding_dim,
            postnet_kernel_size=self.model_config.postnet_kernel_size
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # {
        #     "token_padded": [B, n_tok],
        #     "token_lengths": [B],
        #     "mel_padded": [B, n_mels, n_frames],
        #     "gate_padded": [B, n_frames],
        #     "mel_lengths": [B]
        # }
        token_lengths, mel_lengths = batch["token_lengths"].data, batch["mel_lengths"].data
        embedded_inputs = self.embedding(batch["token_padded"]).transpose(1, 2) # [B, n_tok] -> [B, n_tok, sym_dim] -> [B, sym_dim, n_tok]
        encoder_outputs = self.encoder(embedded_inputs, token_lengths) # [B, sym_dim, n_tok] -> [B, n_tok, enc_dim]
        # [B, n_mels, n_frames], [B, n_frames], [B, n_frames, n_tok] <- [B, n_tok, enc_dim], [B, n_mels, n_frames], [B]
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, batch["mel_padded"], memory_lengths=token_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs) # [B, n_mels, n_frames]
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet # [B, n_mels, n_frames]

        if self.model_config.mask_padding:
            mask = get_mask_from_lengths(mel_lengths) # [B, max_mel_length=n_frames]
            mask = mask.expand(self.audio_config.n_mels, mask.size(0), mask.size(1)) # [n_mels, B, n_frames]
            mask = mask.permute(1, 0, 2) # [B, n_mels, n_frames]

            mel_outputs.data.masked_fill_(mask, 0.0) # [B, n_mels, n_frames]
            mel_outputs_postnet.data.masked_fill_(mask, 0.0) # [B, n_mels, n_frames]
            gate_outputs.data.masked_fill_(mask[:, 0, :], 1e3) # [B, n_frames]
        
        outputs = {
            "mel_outputs": mel_outputs, # [B, n_mels, n_frames]
            "mel_outputs_postnet": mel_outputs_postnet, # [B, n_mels, n_frames]
            "gate_outputs": gate_outputs, # [B, n_frames]
            "alignments": alignments, # [B, n_frames, n_tok]
        }
        return outputs

    def inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # { "tokens": [1, n_tok] }
        with torch.no_grad():
            embedded_inputs = self.embedding(inputs["tokens"]).transpose(1, 2) # [1, n_tok] -> [1, n_tok, sym_dim] -> [1, sym_dim, n_tok]
            encoder_outputs = self.encoder.inference(embedded_inputs) # [1, sym_dim, n_tok] -> [1, n_tok, enc_dim]
            # [1, n_mels, n_frames], [1, n_frames], [1, n_frames, n_tok] <- [1, n_tok, enc_dim]
            mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
            mel_outputs_postnet = self.postnet(mel_outputs) # [1, n_mels, n_frames]
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet # [1, n_mels, n_frames]

        outputs = {
            "mel_outputs": mel_outputs, # [1, n_mels, n_frames]
            "mel_outputs_postnet": mel_outputs_postnet, # [1, n_mels, n_frames]
            "gate_outputs": gate_outputs, # [1, n_frames]
            "alignments": alignments, # [1, n_frames, n_tok]
        }
        return outputs
    
    def get_criterion(self) -> Dict[str, Union[Callable, nn.Module]]:
        return {
            "loss": Tacotron2Loss
        }

    def get_optimizer(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "optimizer": torch.optim.Adam(
                self.parameters(),
                lr=self.model_config.learning_rate,
                weight_decay=self.model_config.weight_decay
            )
        }

    def train_step(self, batch: Dict, criterion: Dict, optimizer: Dict) -> None:
        optimizer["optimizer"].zero_grad()
        outputs = self.forward(batch=batch)
        loss = criterion["loss"](batch, outputs)
        self.loss_items = {key: val.item() for key, val in loss.items()}
        loss["loss"].backward()
        self.grad_norm_val = torch.nn.utils.clip_grad_norm_(self.parameters(), self.model_config.grad_clip_thresh).item()
        optimizer["optimizer"].step()

    def eval_step(self, batch: Dict, criterion: Dict, eval_outdir: str) -> None:
        # do eval step for one batch
        with torch.no_grad():
            outputs = self.forward(batch=batch)
            loss = criterion["loss"](batch, outputs)
        self.loss_items_eval = {(key + "_eval"): val.item() for key, val in loss.items()}
        # chose a random output
        eval_ind = random.randrange(0, batch["token_padded"].size(0))
        n_tok, n_frames = batch["token_lengths"][eval_ind].item(), batch["mel_lengths"][eval_ind].item()
        mel_gt = batch["mel_padded"][eval_ind, :, : n_frames].cpu().numpy()
        gate_gt = batch["gate_padded"][eval_ind, : n_frames].cpu().numpy()
        mel_pred = outputs["mel_outputs_postnet"][eval_ind, :, : n_frames].cpu().numpy()
        gate_pred = torch.sigmoid(outputs["gate_outputs"][eval_ind, : n_frames]).cpu().numpy()
        alignments = outputs["alignments"][eval_ind, : n_frames, : n_tok].cpu().numpy().T # [n_frames, n_tok] -> [n_tok, n_frames]
        # save the output
        saveplot_mel(mel_gt, os.path.join(eval_outdir, "mel_tar.png"))
        saveplot_mel(mel_pred, os.path.join(eval_outdir, "mel_pred.png"))
        saveplot_gate(gate_gt, gate_pred, os.path.join(eval_outdir, "gate.png"), plot_both=True)
        saveplot_alignment(alignments, os.path.join(eval_outdir, "alignments.png"))
        self.loss_outputs = {
            "mel_tar": os.path.join(eval_outdir, "mel_tar.png"),
            "mel_pred": os.path.join(eval_outdir, "mel_pred.png"),
            "gate": os.path.join(eval_outdir, "gate.png"),
            "alignments": os.path.join(eval_outdir, "alignments.png"),
        }

    def get_eval_priority(self) -> float:
        return self.loss_items_eval["loss_eval"]

    def get_train_step_logs(self) -> Dict:
        train_logs = {}
        train_logs.update(self.loss_items)
        train_logs.update({
            "grad_norm": self.grad_norm_val
        })
        return train_logs

    def get_eval_step_logs(self, wandb_logger: WandbLogger) -> Dict:
        eval_logs = {}
        eval_logs.update(self.loss_items_eval)
        eval_logs.update({
            "mel": [
                wandb_logger.Image(self.loss_outputs["mel_tar"], caption='mel target'),
                wandb_logger.Image(self.loss_outputs["mel_pred"], caption='mel predicted')
            ],
            "gate": wandb_logger.Image(self.loss_outputs["gate"], caption='gate'),
            "alignments": wandb_logger.Image(self.loss_outputs["alignments"], caption='alignments')
        })
        return eval_logs

    def get_checkpoint_statedicts(self, optimizer: Optional[Dict]) -> Dict:
        statedicts = {}
        statedicts["model_statedict"] = self.state_dict()
        if optimizer != None:
            statedicts["optim_statedict"] = optimizer["optimizer"].state_dict()
        return statedicts
    
    def load_checkpoint_statedicts(self, statedicts: Dict, save_optimizer_dict: bool, optimizer: Dict) -> None:
        self.load_state_dict(statedicts["model_statedict"])
        if save_optimizer_dict:
            optimizer["optimizer"].load_state_dict(statedicts["optim_statedict"])

    @staticmethod
    def load_from_config(config_path: str) -> BaseModel:
        configs = BaseConfig.load_configs_from_file(
            path=config_path,
            config_map={
                "text_config": TextConfig,
                "audio_config": AudioConfig,
                "model_config": Tacotron2Config
            }
        )
        return Tacotron2(**configs)

def Tacotron2Loss(batch, outputs):
    mel_target, gate_target = batch["mel_padded"], batch["gate_padded"] # [B, n_mels, n_frames], [B, n_frames]
    mel_target.requires_grad = False
    gate_target.requires_grad = False
    gate_target = gate_target.view(-1, 1) # [B * n_frames, 1]

    # [B, n_mels, n_frames], [B, n_mels, n_frames], [B, n_frames]
    mel_out, mel_out_postnet, gate_out = outputs["mel_outputs"], outputs["mel_outputs_postnet"], outputs["gate_outputs"]
    gate_out = gate_out.view(-1, 1) # [B * n_frames, 1]
    mel_loss = nn.MSELoss()(mel_out, mel_target)
    mel_loss += nn.MSELoss()(mel_out_postnet, mel_target)
    gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
    loss = mel_loss + gate_loss
    return {
        "loss": loss,
        "mel_loss": mel_loss,
        "gate_loss": gate_loss
    }
