import os
import re
import scipy.io.wavfile
import subprocess
from pydub import AudioSegment
from utils import downloadYTmp3

def parse_timestamp(timestamp):
    parts = re.split('[:,\.]', timestamp)
    hours, minutes, seconds, milliseconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def vtt_to_csv(vtt_file, csv_file, speaker_id):
    with open(vtt_file, 'r') as vtt, open(csv_file, 'w') as csv:
        lines = vtt.readlines()
        lines = [x.strip() for x in lines if x.strip() != '']
        timestamps = []
        texts = []
        i = 0
        while i < len(lines):
            # Check if the line is a timestamp line
            if '-->' in lines[i]:
                start_time, end_time = lines[i].split(' --> ')
                timestamps.append((parse_timestamp(start_time), parse_timestamp(end_time)))
                texts.append(lines[i + 1])
                i += 2  # Skip the next line which contains the text
            else:
                i += 1
        count = len(texts)
        for i in range(count):
            index = str(i + 1).rjust(len(str(count)), '0')
            csv.write(f"{speaker_id}_{index}|{texts[i].strip()}\n")
    return timestamps

def trimSilence(path, outpath, silence_threshold = -50.0, chunk_size = 10, sampling_rate = 22050):
    assert chunk_size > 0
    subprocess.run(["ffmpeg", "-i", path, "-ar", str(sampling_rate), outpath], capture_output=True)
    sound = AudioSegment.from_file(outpath, format = "wav")
    trimmed_sound = sound
    duration = len(sound)
    ltrim_ms = 0
    while sound[ltrim_ms: ltrim_ms + chunk_size].dBFS < silence_threshold and ltrim_ms < duration:
        ltrim_ms += chunk_size
    rsound = sound.reverse()
    rtrim_ms = 0
    while rsound[rtrim_ms: rtrim_ms + chunk_size].dBFS < silence_threshold and rtrim_ms < duration:
        rtrim_ms += chunk_size
    trimmed_sound = sound[ltrim_ms: duration - rtrim_ms]
    trimmed_sound.export(outpath, format = "wav")                   

def split_wav_file(wav_file, timestamps, directory_path, speaker_id):
    # Split the wav file into segments based on timestamps
    fs, data = scipy.io.wavfile.read(wav_file + ".wav")
    count = len(timestamps)
    directory_path = os.path.join(directory_path, "wavs")
    os.mkdir(directory_path)
    for ind, (start_time, end_time) in enumerate(timestamps):
        segment = data[int(start_time * fs): int(end_time * fs)]
        index = str(ind + 1).rjust(len(str(count)), "0")
        out_path = os.path.join(directory_path, f"z{speaker_id}_{index}.wav")
        trim_out_path = os.path.join(directory_path, f"{speaker_id}_{index}.wav")
        scipy.io.wavfile.write(out_path, fs, segment)
        trimSilence(out_path, trim_out_path)
        os.remove(out_path)

def createDatasetFromYoutube(link, directory_path, temp_directory_path, speaker_id):
    wav_file = os.path.join(temp_directory_path, "output")
    vtt_file = os.path.join(temp_directory_path, "output.en.vtt")
    csv_file = os.path.join(directory_path, "transcript.txt")
    downloadYTmp3(link, wav_file)
    timestamps = vtt_to_csv(vtt_file, csv_file, speaker_id)
    split_wav_file(wav_file, timestamps, directory_path, speaker_id)


if __name__ == "__main__":
    createDatasetFromYoutube("https://www.youtube.com/watch?v=fRed0Xmc2Wg", "dataset", "temp", "test")
