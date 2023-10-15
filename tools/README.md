# GenVox Tools
Helper scripts/tools based on the existing code used for GenVox to do independant tasks.

## Usage
```bash
# for resampling wav files
python3 tools/resample.py -i <input_dir> -o <output_dir> -fs <sampling_rate> -nj <number_of_jobs>

# for creating dataset a from a youtube video
python3 tools/create_dataset_from_youtube.py <youtube_link> <output_path> --verbose --remove_cache
```
