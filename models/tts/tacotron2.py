from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from typing import List, Dict

from configs import BaseConfig, TextConfig, AudioConfig, TrainerConfig
from configs.models import Tacotron2Config
from models.tts import TTSModel
from models.generic import LinearNorm, ConvNorm

def get_mask_from_lengths(lengths, use_cuda):
    max_len = torch.max(lengths).item()
    if use_cuda:
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    else:
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = ~(ids < lengths.unsqueeze(1)).bool()
    return mask

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
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

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
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )
        energies = energies.squeeze(-1)
        return energies

    def forward(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat,
            mask
        ):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            query=attention_hidden_state,
            processed_memory=processed_memory,
            attention_weights_cat=attention_weights_cat
        )
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights

class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: List[int]):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            LinearNorm(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
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
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x

class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
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
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        
        input_lengths = input_lengths.cpu().numpy() # pytorch tensor are not reversible, hence the conversion
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(
            self,
            use_cuda: bool,
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
        self.use_cuda = use_cuda
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

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.n_mel_channels).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        # NOTE: may not be required
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), decoder_inputs.size(1), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=get_mask_from_lengths(memory_lengths, self.use_cuda))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(TTSModel):
    def __init__(self, model_config: Tacotron2Config, audio_config: AudioConfig, text_config: TextConfig, trainer_config: TrainerConfig) -> None:
        super().__init__(model_config, audio_config, text_config, trainer_config)
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
            use_cuda=self.use_cuda,
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
        token_lengths, mel_lengths = batch["token_lengths"].data, batch["mel_lengths"].data

        embedded_inputs = self.embedding(batch["token_padded"]).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, token_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, batch["mel_padded"], memory_lengths=token_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if self.model_config.mask_padding:
            mask = get_mask_from_lengths(mel_lengths, self.use_cuda) # [B, max_mel_length]
            mask = mask.expand(self.audio_config.n_mels, mask.size(0), mask.size(1)) # [n_mels, B, max_mel_length]
            mask = mask.permute(1, 0, 2) # [B, n_mels, max_mel_length]

            mel_outputs.data.masked_fill_(mask, 0.0)
            mel_outputs_postnet.data.masked_fill_(mask, 0.0)
            gate_outputs.data.masked_fill_(mask[:, 0, :], 1e3)
        
        outputs = {
            "mel_outputs": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "gate_outputs": gate_outputs,
            "alignments": alignments,
        }
        return outputs

    def inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embedded_inputs = self.embedding(inputs["tokens"]).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = {
            "mel_outputs": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "gate_outputs": gate_outputs,
            "alignments": alignments,
        }
        return outputs

######################

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
