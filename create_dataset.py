import os
import re
import scipy.io.wavfile
import subprocess
from pydub import AudioSegment
from utils import downloadYTmp3

def parse_timestamp(timestamp):
    parts = re.split('[:,\.]', timestamp)
    hours, minutes, seconds, milliseconds = map(int, parts)
    return hours*3600 + minutes*60 + seconds + milliseconds/1000

def parse_vtt(vtt_file):
    with open(vtt_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines if x.strip() != '']
        timestamp_lines = [x for x in lines if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$', x)]
        timestamps = [re.split(' --> ', x) for x in timestamp_lines]
        timestamps = [(parse_timestamp(x[0]), parse_timestamp(x[1])) for x in timestamps]
        texts = []
        for i, x in enumerate(lines):
            if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$', x):
                texts.append(lines[i+1])
        return timestamps, texts

def vtt_to_csv(vtt_file, csv_file, timestamps, sid):
    with open(vtt_file, 'r') as vtt, open(csv_file, 'w') as csv:
        current_id = ''
        current_text = ''
        count=0
        for i in timestamps:
            count+=1
        i=-1
        for line in vtt:
            # Timestamp format "00:00:00.000 --> 00:00:00.000"
            if re.match('\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d', line):
                index = str(i).rjust(len(str(count)),'0')
                i+=1
                if current_id:
                    csv.write(f"{sid}_{index}|{current_text.strip()}\n")
                current_id = re.findall('\d\d:\d\d:\d\d.\d\d\d --> \d\d:\d\d:\d\d.\d\d\d', line)[0]
                current_text = ''
            else:
                current_text += line.strip() + ' '
        if current_id:
            index = str(i).rjust(len(str(count)),'0')
            csv.write(f"{sid}_{index}|{current_text.strip()}\n")

def trimSilence(path, outpath, silence_threshold=-50.0, chunk_size=10, sampling_rate=22050):
    assert chunk_size > 0
    subprocess.run(["ffmpeg", "-i", path, "-ar", str(sampling_rate), outpath], capture_output=True)
    sound = AudioSegment.from_file(outpath, format="wav")
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
    trimmed_sound.export(outpath, format="wav")

def split_wav_file(wav_file, timestamps, out_dir,sid):
    # Split the WAV file into segments based on timestamps
    wav_file = './temp/output.wav'
    print(wav_file)
    print(repr(wav_file))
    print(type(wav_file))
    fs, data = scipy.io.wavfile.read(wav_file)
    count = 0
    for i in timestamps:
        count+=1
    for i, (start_time, end_time) in enumerate(timestamps):
        segment = data[int(start_time*fs):int(end_time*fs)]
        index = str(i).rjust(len(str(count)),"0")
        out_path = os.path.join(out_dir, f"z{sid}_{index}.wav")
        scipy.io.wavfile.write(out_path, fs, segment)
        trimSilence(f"{out_dir}/z{sid}_{index}.wav",f"{out_dir}/{sid}_{index}.wav")
        os.remove(f"{out_dir}/z{sid}_{index}.wav")

def createDatesetFromYoutube(link, directory_path, temp_directory_path, speaker_id):
    # downloadYTmp3(link, f"{temp_directory_path}/output.wav")
    vtt_file = os.path.join(temp_directory_path, "output.wav.en.vtt")
    wav_file = os.path.join(temp_directory_path, "output.wav")
    csv_file = os.path.join(directory_path, "data.csv")
    timestamps, texts = parse_vtt(vtt_file)
    vtt_to_csv(vtt_file,csv_file,timestamps,speaker_id)
    split_wav_file(wav_file, timestamps, directory_path,speaker_id)


if __name__ == "__main__":
    createDatesetFromYoutube("https://www.youtube.com/watch?v=fRed0Xmc2Wg", "dataset", "temp", "test")

