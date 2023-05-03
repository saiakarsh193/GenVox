# GenVox Tools
Helper scripts/tools based on the existing code used for GenVox to do independant tasks.

## Usage
```bash
# for resampling wav files
python3 tools/resample.py -i <input_dir> -o <output_dir> -fs <sampling_rate>

# for trimming silence in wav files
python3 tools/trim_audio.py -i <input_dir> -o <output_dir>
```