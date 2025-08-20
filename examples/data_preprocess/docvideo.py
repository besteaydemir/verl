import argparse
import os
from copy import deepcopy
import datasets
from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/stud/aydemir/data/docvideo")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = "datasets-examples/doc-video-1"
    dataset = datasets.load_dataset(data_source)

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    )

    # ✅ Absolute paths for your 3 available videos
    video_paths = [
        "/home/stud/aydemir/.cache/huggingface/hub/datasets--datasets-examples--doc-video-1/snapshots/90da02ce45335043e8d56c1a42dc36fb19c8cf7a/--5iwqOe8G8_scene_1.mp4",
        "/home/stud/aydemir/.cache/huggingface/hub/datasets--datasets-examples--doc-video-1/snapshots/90da02ce45335043e8d56c1a42dc36fb19c8cf7a/--5iwqOe8G8_scene_2.mp4",
        "/home/stud/aydemir/.cache/huggingface/hub/datasets--datasets-examples--doc-video-1/snapshots/90da02ce45335043e8d56c1a42dc36fb19c8cf7a/--5iwqOe8G8_scene_3.mp4",
    ]

    # ✅ Create 100 examples by repeating and assigning video paths
    def make_synthetic_split(original_dataset, video_paths, target_size=100):
        original_list = original_dataset.to_list()
        synthetic_data = []

        for i in range(target_size):
            example = dict(original_list[0])
            example["video_path"] = video_paths[0]
            if "video" in example:
                del example["video"]
            synthetic_data.append(example)

        return datasets.Dataset.from_list(synthetic_data)


    # Build synthetic train and test sets
    synthetic_train = make_synthetic_split(dataset["train"], video_paths, target_size=100)
    synthetic_test = make_synthetic_split(dataset["train"], video_paths, target_size=100)

    # Assign new dataset splits
    dataset = {"train": synthetic_train, "test": synthetic_test}

    def make_map_fn(split):
        def process_fn(example, idx):
            qa = example["qa"]
            question = qa[0]["question"]
            answer = qa[0]["answer"]

            prompt =  instruction_following + " " + question + " <video> "

            # ✅ Use manually assigned file path
            video_path = example["video_path"]
            wrapped_videos = [{"videos": {"type": "video", "video": video_path, "nframes": 16}}]

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "videos": wrapped_videos,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            return data

        return process_fn

    # ✅ Map to final training format
    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True, num_proc=1)
    test_dataset = dataset["test"].map(function=make_map_fn("test"), with_indices=True, num_proc=1)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train_docvideo_large_dict_prompt_allsame_nframes.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_docvideo_large_dict_prompt_allsame_nframes.parquet"))

    if hdfs_dir is not None:
        copy(src=local_dir, dst=hdfs_dir)
