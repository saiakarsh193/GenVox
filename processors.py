from typeguard import typechecked

from config import DatasetConfig


class DatasetProcessor:
    """
    For extracting text from dataset transcript and creating i2t and i2w files
    """
    @typechecked
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def __call__(self):
        print(self.config)
        if (self.config.dataset_type == "text"):
            with open(self.config.transcript_path, 'r') as f:
                raw_text = f.readlines()
            text = {}
            for rt in raw_text:
                rt = rt.strip().split(self.config.delimiter)
                uid = rt[self.config.uid_index]
                utt = rt[self.config.utt_index]
                print(uid, utt)
                text[uid] = utt


class AudioProcessor:
    # for processing all the audio files
    pass