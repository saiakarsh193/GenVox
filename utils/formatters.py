import os
from typing import Literal, List, Dict, Optional

def _ljspeech_formatter(src_path: str) -> List[Dict]:
    transcript_path = os.path.join(src_path, "metadata.csv")
    wavs_path = os.path.join(src_path, "wavs")
    data = []
    with open(transcript_path, 'r') as f:
        raw_text = f.readlines()
    for text in raw_text:
        text = text.strip().split("|")
        utt_id, utt = text[0], text[1]
        wav_path = os.path.join(wavs_path, utt_id + ".wav")
        if os.path.isfile(wav_path):
            data.append({
                "text": utt,
                "audio_path": wav_path,
                "unique_id": utt_id
            })
        else:
            print(f"{wav_path} not found, skipping it")
    return data

_FORMATTER_TYPES = Literal[
    "ljspeech"
]

class BaseDataset:
    """BaseDataset is used to extract data using a formatter and store it as List[Dict]"""
    def __init__(self, dataset_path: str, formatter: _FORMATTER_TYPES, dataset_name: Optional[str] = None):
        if formatter == "ljspeech":
            self.data = _ljspeech_formatter(src_path=dataset_path)
        # get prefix for unique_id
        _prefix = "" if dataset_name == None else (dataset_name + "#")
        # updating the unique_id with the prefix
        for sample in self.data:
            sample["unique_id"] = _prefix + sample["unique_id"]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, ind: int) -> Dict:
        return self.data[ind]
