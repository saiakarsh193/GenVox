from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import os
import argparse

from utils.formatters import BaseDataset
from utils import center_print
from configs import TextConfig, AudioConfig, TrainerConfig
from core.processors import DataPreprocessor
from core.trainer import Trainer

from models.tts.tacotron2 import Tacotron2
from configs.models import Tacotron2Config

def main(args):
    center_print("GENVOX PIPELINE: ULCA", space_factor=0.1)

    if args.ada:
        assert args.ada_username != None, f"code set to run on ada, but ada username not given"
        print(f"running on ADA, username: {args.ada_username}")
        scratch_dir = os.path.join("/scratch", args.ada_username)
        if not os.path.isdir(scratch_dir):
            print(f"creating scratch directory: {scratch_dir}")
            os.mkdir(scratch_dir)
        parent_dir = scratch_dir
    else:
        parent_dir = os.path.curdir
    
    if os.path.isdir(args.dataset_path): # if already a directory, then do nothing
        dataset_basename = os.path.basename(args.dataset_path)
        dataset_path = args.dataset_path
    else: # if a zip file then extract it to dataset_path
        dataset_basename = os.path.splitext(os.path.basename(args.dataset_path))[0]
        dataset_path = os.path.join(parent_dir, dataset_basename)
        if not os.path.isdir(dataset_path): # to prevent unzipping multiple times
            if args.ada: # if on ada, first scp the zip file to scratch
                scratch_dataset_path = os.path.join(parent_dir, os.path.basename(args.dataset_path))
                if not os.path.isfile(scratch_dataset_path): # if not already transfered
                    print("attempting scp transfer")
                    os.system(f"scp -r ada:{args.dataset_path} {scratch_dataset_path}") # you can directly do ada:/ rather than username@ada:/ when inside ada node
                    print("scp transfer done")
                args.dataset_path = scratch_dataset_path
            print(f"unzipping {args.dataset_path} to {dataset_path}")
            os.mkdir(dataset_path)
            os.system(f"unzip -qq {args.dataset_path} -d {dataset_path}") # -qq: very quiet unzipping (not verbose)

    dump_dir = os.path.join(parent_dir, f"dump_ulca_pipeline_{dataset_basename}")
    exp_dir = f"exp_ulca_pipeline_{dataset_basename}_{args.model}"

    print("dataset_path:", dataset_path)
    print("dump_dir:", dump_dir)
    print("exp_dir:", exp_dir)

    dataset = BaseDataset(
        dataset_path=dataset_path,
        formatter="ulca",
        dataset_name="ulca"
    )

    text_config = TextConfig(
        language=args.language, # anything but english works
        cleaners=["base_cleaners"],
        use_g2p=False # character level tokenization
    )

    audio_config = AudioConfig(
        sampling_rate=args.sampling_rate,
        trim_silence=True,
        min_wav_duration=2,
        max_wav_duration=15,
        normalize=True,
        filter_length=1024,
        log_func="np.log",
    )

    data_preprocessor = DataPreprocessor(
        datasets=[dataset],
        text_config=text_config,
        audio_config=audio_config,
        eval_split=500,
        dump_dir=dump_dir
    )
    data_preprocessor.run()

    trainer_config = TrainerConfig(
        project_name="genvox2_ulca_pipeline",
        experiment_id=dataset_basename[21: ], # IITM_TTS_data_Phase2_xxxx
        notes="",
        use_cuda=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=32,
        num_loader_workers=0,
        iters_for_checkpoint=500,
        max_best_models=1,
        run_eval=True,
        use_wandb=True
    )

    if args.model == "tacotron2":
        model = Tacotron2(
            model_config=Tacotron2Config(),
            audio_config=audio_config,
            text_config=text_config
        )
    else:
        raise TypeError(f"given model {args.model} is invalid")

    trainer = Trainer(
        model=model,
        trainer_config=trainer_config,
        text_config=text_config,
        audio_config=audio_config,
        dump_dir=dump_dir,
        exp_dir=exp_dir
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ULCA Bhashini TTS building pipeline")
    parser.add_argument("dataset_path", type=str, help="path to the ulca dataset directory/zip file")
    parser.add_argument("--language", "-l", type=str, help="language of the data", default="indian")
    parser.add_argument("--model", "-m", type=str, help="model architecture to be used for training", required=True)
    parser.add_argument("--sampling_rate", "-fs", type=int, help="sampling rate for tts model (automatic resampling of data)", default=22050)
    parser.add_argument("--batch_size", "-bs", type=int, help="batch size for training", default=16)
    parser.add_argument("--epochs", "-e", type=int, help="number of epochs for training", default=200)
    parser.add_argument("--ada", help="if running the code on Ada server", action="store_true")
    parser.add_argument("--ada_username", type=str, help="username of the Ada user", default=None)
    args = parser.parse_args()
    main(args)
