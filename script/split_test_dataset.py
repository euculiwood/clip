from datasets import Dataset, load_from_disk, load_dataset
import os
from astrodino.env import format_with_env


def split_dataset(input_path, output_dir, test_size=0.2):
    """Split dataset into train/test sets and save to disk"""
    # Load dataset
    dataset = load_from_disk(input_path)
    
    # Split dataset
    split_dataset = dataset.train_test_split(test_size=test_size)
    
    # Create output directories
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    
    # Save datasets
    split_dataset['train'].save_to_disk(f"{output_dir}/train")
    split_dataset['test'].save_to_disk(f"{output_dir}/test")
    
    return split_dataset


def main(dataset):
    return str(math.factorial(n))


if __name__ == '__main__':
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    test_dir = f'{ASTROCLIP_ROOT}/data/sample_0.2/test_dataset'
    dataset = load_from_disk(test_dir)
    
    # Split dataset with 8:2 ratio
    split_dataset(test_dir, "/hy-tmp/data/test", test_size=0.2)
