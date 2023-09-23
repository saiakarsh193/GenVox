import argparse
from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TextConfig, AudioConfig, DatasetConfig, TrainerConfig, Tacotron2Config, OptimizerConfig, MelGANConfig, ModelConfig
from processors import DatasetProcessor
from trainer import Trainer

def main(args):
    print(("=" * 10) + f" GENVOX: ULCA PIPELINE ({args.task}) " + ("=" * 10))
    if (args.ada):
        print("Running on ADA")
        par_dir = os.path.join("/scratch", "sai_akarsh")
        if not os.path.isdir(par_dir):
            print(f"creating parent directory: {par_dir}")
            os.mkdir(par_dir)
    else:
        par_dir = ""
    file_name = os.path.splitext(os.path.basename(args.input_path))[0]
    data_dir = os.path.join(par_dir, file_name)
    dump_dir = os.path.join(par_dir, f"dump_{file_name}")
    exp_dir = f"exp_ulca_{file_name}_{args.task}"
    if not os.path.isdir(data_dir):
        print(f"creating data directory: {data_dir}")
        os.mkdir(data_dir)
        if (not os.path.isfile(args.input_path)):
            print(f"{args.input_path} does not exist")
            print(f"attempting scp transfer")
            os.system(f"scp -r saiakarsh@ada:{args.input_path} {par_dir}")
            print(f"scp transfer done")
            args.input_path = os.path.join(par_dir, os.path.basename(args.input_path))
        print(f"unzipping file: {args.input_path}")
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
        min_wav_duration=2,
        max_wav_duration=15,
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

    exp_id = file_name[21: ] # IITM_TTS_data_Phase2_xxxx

    trainer_config = TrainerConfig(
        project_name="genvox_ulca_tts",
        experiment_id=exp_id,
        wandb_logger=True,
        batch_size=args.batch_size,
        validation_batch_size=args.batch_size,
        num_loader_workers=0,
        run_validation=True,
        use_cuda=True,
        epochs=args.epochs,
        max_best_models=1,
        iters_for_checkpoint=1000,
        dump_dir=dump_dir,
        exp_dir=exp_dir
    )

    assert args.model in ModelConfig.MODEL_DETAILS[args.task], f"Invalid model ({args.model}) given for task ({args.task})"
    if (args.task == "TTS"):
        if args.model == "Tacotron2":
            model_config = Tacotron2Config()
            optimizer_config = OptimizerConfig()
    else:
        if args.model == "MelGAN":
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
    parser.add_argument("--batch_size", type=int, help="batch size for both training and validation", default=16)
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=200)
    parser.add_argument("--ada", help="if running the code on Ada", action="store_true")
    parser.add_argument("--task", type=str, choices=["TTS", "VOC"], help="type of task for training (TTS/VOC)", required=True)
    parser.add_argument("--model", type=str, help=f"model architecture to be used {ModelConfig.MODEL_DETAILS}", required=True)
    args = parser.parse_args()
    main(args)
