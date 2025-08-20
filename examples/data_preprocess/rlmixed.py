import argparse
import os
from PIL import Image
import datasets
from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/stud/aydemir/data/geo3k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "hiyouga/rl-mixed-dataset"

    dataset = datasets.load_dataset(data_source)

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    )

    # ✅ Optional: Filter raw dataset FIRST if `images` is sometimes missing or empty
    def has_images(example):
        return "images" in example and isinstance(example["images"], list) and len(example["images"]) > 0

    dataset["train"] = dataset["train"].filter(has_images)
    dataset["test"] = dataset["test"].filter(has_images)

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            # Wrap images in expected dict format
            wrapped_images = []
            for img in images:
                if isinstance(img, str):
                    wrapped_images.append({"image": img})
                elif isinstance(img, Image.Image):
                    wrapped_images.append({"image": img})
                else:
                    raise TypeError(f"Unsupported image type: {type(img)}")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": wrapped_images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    # ✅ Use num_proc=1 temporarily to avoid multiprocessing bugs during debugging
    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = dataset["test"].map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train_rlmixed3.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_rlmixed3.parquet"))

    if hdfs_dir is not None:
        copy(src=local_dir, dst=hdfs_dir)
