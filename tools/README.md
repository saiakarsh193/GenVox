# GenVox Tools
Helper scripts/tools based on the existing code used for GenVox to do independant tasks.

## Usage
```bash
# for creating a dataset from multiple youtube videos
python3 tools/create_dataset_from_youtube.py -l <youtube_link_1> -l <youtube_link_2> -o <output_path> --speaker_id "SPK" --verbose --remove_cache

# for resampling wav files
python3 tools/resample.py -i <input_dir> -o <output_dir> -fs <sampling_rate> -nj <number_of_jobs>

# for updating all the paths using prefix matching in a given dump file
python3 tools/update_dump_file_paths.py -i <input_path> -po <old_prefix> -pn <new_prefix>

# For dataset statistics
python3 tools/dataset_stats.py <dataset_dir> --n_bins <number_of_bins> --threshold <length_threshold> --n_jobs <number_of_jobs>
```
