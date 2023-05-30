import argparse
from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TextConfig, AudioConfig, DatasetConfig, TrainerConfig, Tacotron2Config, OptimizerConfig, MelGANConfig
from processors import DatasetProcessor
from trainer import Trainer

def main(args):
    print(("=" * 10) + f" GENVOX: ULCA PIPELINE ({args.task})" + ("=" * 10))

    if (not os.path.isfile(args.input_path)):
        print(f"{args.input_path} does not exist")
    file_name = os.path.splitext(os.path.basename(args.input_path))[0] + f"_{args.task}"
    if (args.ada):
        print("Running on ADA")
        par_dir = os.path.join("/scratch", "sai_akarsh")
        if not os.path.isdir(par_dir):
            os.mkdir(par_dir)
    else:
        par_dir = ""
    data_dir = os.path.join(par_dir, file_name)
    dump_dir = os.path.join(par_dir, "dump")
    exp_dir = "exp"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        os.system(f"unzip -qq {args.input_path} -d {data_dir}")
    
    print("data_dir:", data_dir)
    print("dump_dir:", dump_dir)
    print("exp_dir:", exp_dir)

    text_config = TextConfig(
        language=args.language, # anything but english works
        cleaners=["base_cleaners"],
        use_g2p=False # character level tokenization
    )
    audio_config = AudioConfig(
        sampling_rate=args.sampling_rate,
        min_wav_duration=3,
        max_wav_duration=20,
        filter_length=1024,
        log_func="np.log",
    )
    dataset_config = DatasetConfig(
        text_config=text_config,
        audio_config=audio_config,
        dataset_type="json",
        transcript_path=os.path.join(data_dir, "data.json"),
        uid_keyname="audioFilename",
        utt_keyname="text",
        wavs_path=data_dir,
        validation_split=500,
        dump_dir=dump_dir
    )

    dataset_processor = DatasetProcessor(dataset_config)
    dataset_processor()

    trainer_config = TrainerConfig(
        project_name="genvox_ulca_tts",
        experiment_id=file_name,
        wandb_logger=True,
        wandb_auth_key="56acc87c7b95662ff270b9556cdf68de699a210f",
        batch_size=args.batch_size,
        validation_batch_size=args.batch_size,
        num_loader_workers=0,
        run_validation=True,
        use_cuda=True,
        epochs=args.epochs,
        max_best_models=3,
        iters_for_checkpoint=1000,
        dump_dir=dump_dir,
        exp_dir=exp_dir
    )

    if (args.task == "TTS"):
        model_config = Tacotron2Config()
        optimizer_config = OptimizerConfig()
    else:
        model_config = MelGANConfig()
        optimizer_config = OptimizerConfig(
            learning_rate=0.0001,
            beta1=0.5,
            beta2=0.9,
            weight_decay=0
        )

    trainer = Trainer(
        trainer_config=trainer_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        audio_config=audio_config,
        text_config=text_config,
        dataset_config=dataset_config
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ULCA Bhashini TTS building pipeline")
    parser.add_argument("input_path", type=str, help="path to the ulca dataset zip file")
    parser.add_argument("--language", "-l", type=str, help="language of the data", default="indian")
    parser.add_argument("--sampling_rate", "-fs", type=int, help="sampling rate for tts model (automatic resampling of data)", default=22050)
    parser.add_argument("--batch_size", type=int, help="batch size for both training and validation", default=32)
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=200)
    parser.add_argument("--ada", help="if running the code on Ada", action="store_true")
    parser.add_argument("--task", type=str, choices=["TTS", "VOC"], help="type of task for training (TTS/VOC)", default="TTS")
    args = parser.parse_args()
    main(args)