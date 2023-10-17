import os
import random
import torch
from torch import nn
from typing import Callable, Dict, Optional, Union, List, Tuple

from configs import AudioConfig, BaseConfig
from configs.models import MelGANConfig
from core.trainer.wandb_logger import WandbLogger
from models import BaseModel
from models.vocoder import VocoderModel
from models.generic import ConvNorm, ConvTransposeNorm
from utils.plotting import saveplot_mel, saveplot_signal

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                ConvNorm(1, 16, kernel_size=15, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                ConvNorm(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                ConvNorm(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                ConvNorm(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                ConvNorm(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                ConvNorm(1024, 1024, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            ConvNorm(1024, 1, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = []
        for layer in self.discriminator:
            x = layer(x)
            features.append(x)
        return features[:-1], features[-1]


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)]
        )
        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

    def forward(self, x):
        ret = []
        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))
        return ret # [(feat, score), (feat, score), (feat, score)]


class ResStack(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                ConvNorm(channel, channel, kernel_size=3, dilation=3**i),
                nn.LeakyReLU(0.2),
                ConvNorm(channel, channel, kernel_size=1),
            )
            for i in range(3)
        ])
        self.shortcuts = nn.ModuleList([
            ConvNorm(channel, channel, kernel_size=1)
            for _ in range(3)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


class Generator(nn.Module):
    def __init__(self, mel_channel):
        super().__init__()
        self.mel_channel = mel_channel
        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            ConvNorm(mel_channel, 512, kernel_size=7, stride=1),
            nn.LeakyReLU(0.2),
            ConvTransposeNorm(512, 256, kernel_size=16, stride=8, padding=4),
            ResStack(256),
            nn.LeakyReLU(0.2),
            ConvTransposeNorm(256, 128, kernel_size=16, stride=8, padding=4),
            ResStack(128),
            nn.LeakyReLU(0.2),
            ConvTransposeNorm(128, 64, kernel_size=4, stride=2, padding=1),
            ResStack(64),
            nn.LeakyReLU(0.2),
            ConvTransposeNorm(64, 32, kernel_size=4, stride=2, padding=1),
            ResStack(32),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            ConvNorm(32, 1, kernel_size=7, stride=1),
            nn.Tanh(),
        )
        self.remove_weight_norm_done = False

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def remove_weight_norm(self):
        if self.remove_weight_norm_done:
            return
        self.remove_weight_norm_done = True
        for layer in self.generator:
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel, hop_length = 256):
        # mel: [1, n_mels, n_frames]
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device) # see https://github.com/seungwonpark/melgan/issues/8
        mel = torch.cat((mel, zero), dim=2) # pad input mel with zeros to cut artifact -> [1, n_mels, n_frames + 10]
        audio = self.forward(mel) # [1, 1, wav_len + hop_len * 10]
        return audio[:, :, : -(hop_length * 10)] # [1, 1, wav_len]


class MelGAN(VocoderModel):
    def __init__(self, model_config: MelGANConfig, audio_config: AudioConfig) -> None:
        super().__init__(model_config, audio_config)
        self.model_config: MelGANConfig = self.model_config # for typing hints
        self.generator = Generator(mel_channel=self.audio_config.n_mels)
        self.discriminator = MultiScaleDiscriminator()

    def inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.generator.remove_weight_norm()
        # { "features": [1, n_mels, n_frames] }
        with torch.no_grad():
            wav = self.generator.inference(inputs["features"], hop_length=self.audio_config.hop_length) # [1, 1, wav_len]

        outputs = {
            "wav": wav, # [1, 1, wav_len]
        }
        return outputs
    
    def get_criterion(self) -> Dict[str, Union[Callable, nn.Module]]:
        return {
            "generator_criterion": GeneratorLoss(self.model_config.feat_match),
            "discriminator_criterion": DiscriminatorLoss()
        }
    
    def get_optimizer(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "generator_optimizer": torch.optim.Adam(
                self.generator.parameters(),
                lr=self.model_config.learning_rate,
                betas=(self.model_config.beta1, self.model_config.beta2)
            ),
            "discriminator_optimizer": torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.model_config.learning_rate,
                betas=(self.model_config.beta1, self.model_config.beta2)
            )
        }
    
    def train_step(self, batch: Dict, criterion: Dict, optimizer: Dict) -> None:
        # {
        #     "wavs": [B, 1, max_wav_len],
        #     "features": [B, n_mels, n_frames]
        # }
        # generator training
        optimizer["generator_optimizer"].zero_grad()
        fake_audio = self.generator.forward(mel=batch["features"])
        disc_fake = self.discriminator.forward(x=fake_audio)
        disc_real = self.discriminator.forward(x=batch["wavs"])
        loss_generator = criterion["generator_criterion"](disc_fake, disc_real)
        self.loss_items = {"loss_generator": loss_generator.item()}
        loss_generator.backward()
        optimizer["generator_optimizer"].step()
        # discriminator training
        fake_audio = self.generator.forward(mel=batch["features"]).detach()
        self.loss_items["loss_discriminator"] = 0
        for _ in range(self.model_config.train_repeat_discriminator):
            optimizer["discriminator_optimizer"].zero_grad()
            disc_fake = self.discriminator.forward(x=fake_audio)
            disc_real = self.discriminator.forward(x=batch["wavs"])
            loss_discriminator = criterion["discriminator_criterion"](disc_fake, disc_real)
            self.loss_items["loss_discriminator"] += loss_discriminator.item()
            loss_discriminator.backward()
            optimizer["discriminator_optimizer"].step()
    
    def eval_step(self, batch: Dict, criterion: Dict, eval_outdir: str) -> None:
        with torch.no_grad():
            fake_audio = self.generator.forward(mel=batch["features"])
            disc_fake = self.discriminator.forward(x=fake_audio)
            disc_real = self.discriminator.forward(x=batch["wavs"])
            loss_generator = criterion["generator_criterion"](disc_fake, disc_real)
        self.loss_items_eval = {"loss_generator_eval": loss_generator.item()}
        # chose a random output
        eval_ind = random.randrange(0, batch["wavs"].size(0))
        wav_gt = batch["wavs"][eval_ind].squeeze(0).cpu().numpy()
        mel_gt = batch["features"][eval_ind].cpu().numpy()
        wav_pred = fake_audio[eval_ind].squeeze(0).cpu().numpy()
        # save the output
        saveplot_signal(wav_gt, os.path.join(eval_outdir, "wav_gt.png"))
        saveplot_mel(mel_gt, os.path.join(eval_outdir, "mel_gt.png"))
        saveplot_signal(wav_pred, os.path.join(eval_outdir, "wav_pred.png"))
        self.eval_outputs = {
            "wav_gt": os.path.join(eval_outdir, "wav_gt.png"),
            "mel_gt": os.path.join(eval_outdir, "mel_gt.png"),
            "wav_pred": os.path.join(eval_outdir, "wav_pred.png"),
        }
    
    def get_eval_priority(self) -> float:
        return self.loss_items_eval["loss_generator_eval"]
    
    def get_wandb_metrics(self) -> List[Tuple[str, str]]:
        return [
            ("loss_generator", "min"),
            ("loss_generator_eval", "min")
        ]
    
    def get_train_step_logs(self) -> Dict:
        return self.loss_items
    
    def get_eval_step_logs(self, wandb_logger: WandbLogger) -> Dict:
        eval_logs = {}
        eval_logs.update(self.loss_items_eval)
        eval_logs.update({
            "wav": [
                wandb_logger.Image(self.eval_outputs["wav_gt"], caption='wav target'),
                wandb_logger.Image(self.eval_outputs["wav_pred"], caption='wav predicted')
            ],
            "mel": wandb_logger.Image(self.loss_outputs["mel_gt"], caption='mel target'),
        })
        return eval_logs
    
    def get_checkpoint_statedicts(self, optimizer: Optional[Dict]) -> Dict:
        statedicts = {}
        statedicts["generator_model_statedict"] = self.generator.state_dict()
        statedicts["discriminator_model_statedict"] = self.discriminator.state_dict()
        if optimizer != None:
            statedicts["generator_optim_statedict"] = optimizer["generator_optimizer"].state_dict()
            statedicts["discriminator_optim_statedict"] = optimizer["discriminator_optimizer"].state_dict()
        return statedicts
    
    def load_checkpoint_statedicts(self, statedicts: Dict, save_optimizer_dict: bool, optimizer: Dict) -> None:
        self.generator.load_state_dict(statedicts["generator_model_statedict"])
        self.discriminator.load_state_dict(statedicts["discriminator_model_statedict"])
        if save_optimizer_dict:
            optimizer["generator_optimizer"].load_state_dict(statedicts["generator_optim_statedict"])
            optimizer["discriminator_optimizer"].load_state_dict(statedicts["discriminator_optim_statedict"])
    
    @staticmethod
    def load_from_config(config_path: str) -> BaseModel:
        configs = BaseConfig.load_configs_from_file(
            path=config_path,
            config_map={
                "audio_config": AudioConfig,
                "model_config": MelGANConfig
            }
        )
        return MelGAN(**configs)


class GeneratorLoss(nn.Module):
    def __init__(self, feat_match = 10.0):
        super().__init__()
        self.feat_match = feat_match

    def forward(self, disc_fake, disc_real):
        loss = 0.0
        for (feats_fake, score_fake), (feats_real, score_real) in zip(disc_fake, disc_real):
            loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2])) # get score_fake to 1
            for feat_f, feat_r in zip(feats_fake, feats_real):
                loss += self.feat_match * torch.mean(torch.abs(feat_f - feat_r))  # match features in discriminator for real and fake
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_fake, disc_real):
        loss = 0.0
        for (feats_fake, score_fake), (feats_real, score_real) in zip(disc_fake, disc_real):
            loss += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2])) # get score_real to 1
            loss += torch.mean(torch.sum(torch.pow(score_fake - 0.0, 2), dim=[1, 2])) # get score_fake to 0
        return loss
