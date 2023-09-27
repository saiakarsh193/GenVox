# GenVox Tools
Helper scripts/tools based on the existing code used for GenVox to do independant tasks.

## Usage
```bash
# for resampling wav files
python3 tools/resample.py -i <input_dir> -o <output_dir> -fs <sampling_rate> -nj <number_of_jobs>

# for analysing dataset directory containing wav files
python3 tools/dataset_analyse.py <input_dir> --output_path <output_path_image>
```
