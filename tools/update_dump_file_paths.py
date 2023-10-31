import os
import argparse

def update_dump_file_paths(input_path: str, prefix_old: str, prefix_new: str) -> None:
    with open(input_path, 'r') as f:
        lines = f.readlines()
    f = open(input_path, 'w')
    for line in lines:
        unique_id, audio_path, feature_path, tokens_ind = line.strip().split("|")
        if audio_path.find(prefix_old) == 0:
            audio_path = prefix_new + audio_path[len(prefix_old): ]
        if feature_path.find(prefix_old) == 0:
            feature_path = prefix_new + feature_path[len(prefix_old): ]
        f.write(f"{unique_id}|{audio_path}|{feature_path}|{tokens_ind}\n")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="update_dump_file_paths.py will update all the paths using prefix matching in a given dump file")
    parser.add_argument("--input_path", "-i", required=True, type=str, help="path to the dump file for updating paths")
    parser.add_argument("--prefix_old", "-po", required=True, type=str, help="prefix that needs to be replaced in the path")
    parser.add_argument("--prefix_new", "-pn", required=True, type=str, help="new prefix that needs to be added in the path")
    args = parser.parse_args()
    update_dump_file_paths(
        input_path=args.input_path,
        prefix_old=args.prefix_old,
        prefix_new=args.prefix_new
    )

