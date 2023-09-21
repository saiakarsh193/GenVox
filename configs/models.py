from typing import Dict, Optional
from configs import check_argument, BaseConfig

class Tacotron2Config(BaseConfig):
    """
    Config for Tacotron2 TTS architecture
    """
    def __init__(
        self,
        symbols: Optional[Dict] = None,
        n_symbols: Optional[int] = None,
        symbols_embedding_dim: int = 512,
        encoder_kernel_size: int = 5,
        encoder_n_convolutions: int = 3,
        encoder_embedding_dim: int = 512,
        decoder_rnn_dim: int = 1024,
        prenet_dim: int = 256,
        max_decoder_steps: int = 1000,
        gate_threshold: float = 0.5,
        p_attention_dropout: float = 0.1,
        p_decoder_dropout: float = 0.1,
        attention_rnn_dim: int = 1024,
        attention_dim: int = 128,
        attention_location_n_filters: int = 32,
        attention_location_kernel_size: int = 31,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5,
        mask_padding: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        grad_clip_thresh: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        # input params
        self.symbols = symbols
        self.n_symbols = n_symbols
        self.symbols_embedding_dim = symbols_embedding_dim
        # encoder params
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_n_convolutions = encoder_n_convolutions
        self.encoder_embedding_dim = encoder_embedding_dim
        # decoder params
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        # attention
        self.attention_rnn_dim = attention_rnn_dim
        self.attention_dim = attention_dim
        self.attention_location_n_filters = attention_location_n_filters
        self.attention_location_kernel_size = attention_location_kernel_size
        # postnet
        self.postnet_embedding_dim = postnet_embedding_dim
        self.postnet_kernel_size = postnet_kernel_size
        self.postnet_n_convolutions = postnet_n_convolutions
        #
        self.mask_padding = mask_padding
        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_thresh = grad_clip_thresh
        self.beta1 = beta1
        self.beta2 = beta2

        check_argument("symbols_embedding_dim", self.symbols_embedding_dim, min_val=1)
        check_argument("encoder_kernel_size", self.encoder_kernel_size, min_val=1)
        check_argument("encoder_n_convolutions", self.encoder_n_convolutions, min_val=1)
        check_argument("encoder_embedding_dim", self.encoder_embedding_dim, min_val=1)
        check_argument("decoder_rnn_dim", self.decoder_rnn_dim, min_val=1)
        check_argument("prenet_dim", self.prenet_dim, min_val=1)
        check_argument("max_decoder_steps", self.max_decoder_steps, min_val=1, max_val=10000)
        check_argument("gate_threshold", self.gate_threshold, min_val=0, max_val=1)
        check_argument("p_attention_dropout", self.p_attention_dropout, min_val=0)
        check_argument("p_decoder_dropout", self.p_decoder_dropout, min_val=0)
        check_argument("attention_rnn_dim", self.attention_rnn_dim, min_val=1)
        check_argument("attention_dim", self.attention_dim, min_val=1)
        check_argument("attention_location_n_filters", self.attention_location_n_filters, min_val=1)
        check_argument("attention_location_kernel_size", self.attention_location_kernel_size, min_val=1)
        check_argument("postnet_embedding_dim", self.postnet_embedding_dim, min_val=1)
        check_argument("postnet_kernel_size", self.postnet_kernel_size, min_val=1)
        check_argument("postnet_n_convolutions", self.postnet_n_convolutions, min_val=1)
        check_argument("learning_rate", self.learning_rate, min_val=1e-5)
        check_argument("weight_decay", self.weight_decay, min_val=0)
        check_argument("grad_clip_thresh", self.grad_clip_thresh, min_val=0)
        check_argument("beta1", self.beta1, min_val=0, max_val=1)
        check_argument("beta2", self.beta2, min_val=0, max_val=1)


class MelGANConfig(BaseConfig):
    """
    Config for MelGAN Vocoder architecture
    """
    def __init__(
        self,
        train_repeat_discriminator: int = 1,
        max_frames: int = 200,
        feat_match: float = 10.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0,
        grad_clip_thresh: float = 1.0,
        beta1: float = 0.5,
        beta2: float = 0.9
    ):
        self.train_repeat_discriminator = train_repeat_discriminator
        self.max_frames = max_frames
        self.feat_match = feat_match
        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_thresh = grad_clip_thresh
        self.beta1 = beta1
        self.beta2 = beta2

        check_argument("train_repeat_discriminator", self.train_repeat_discriminator, min_val=1)
        check_argument("max_frames", self.max_frames, min_val=100)
        check_argument("feat_match", self.feat_match, min_val=1)
        check_argument("learning_rate", self.learning_rate, min_val=1e-5)
        check_argument("weight_decay", self.weight_decay, min_val=0)
        check_argument("grad_clip_thresh", self.grad_clip_thresh, min_val=0)
        check_argument("beta1", self.beta1, min_val=0, max_val=1)
        check_argument("beta2", self.beta2, min_val=0, max_val=1)
