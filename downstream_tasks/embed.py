import os
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import torch
from astropy.table import Table
from datasets import load_from_disk

from dinov2.eval.setup import setup_and_build_model
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from astroclip.astrodino.utils import setup_astrodino
from astroclip.env import format_with_env
from astroclip.models import AstroClipModel, Moco_v2, SpecFormer


def get_embeddings(
    image_models: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    batch_size: int = 512,
) -> dict:
    """Get embeddings for images using models"""
    full_keys = set(image_models.keys())
    model_embeddings = {key: [] for key in full_keys}
    im_batch = []


    for image in tqdm(images):
        # Load images, already preprocessed
        im_batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])

        # Get embeddings for batch
        if len(im_batch) == batch_size:
            with torch.no_grad():
                spectra, images = torch.cat(sp_batch).cuda(), torch.cat(im_batch).cuda()

                for key in image_models.keys():
                    model_embeddings[key].append(image_models[key](images))

            im_batch = []

    # Get embeddings for last batch
    if len(im_batch) > 0:
        with torch.no_grad():
             images = torch.cat(im_batch).cuda()

            # Get embeddings
            for key in image_models.keys():
                model_embeddings[key].append(image_models[key](images))

    model_embeddings = {
        key: np.concatenate(model_embeddings[key]) for key in model_embeddings.keys()
    }
    return model_embeddings


def embed_provabgs(
    quasars_file_train: str,
    quasars_file_test: str,
    pretrained_dir: str,
    batch_size: int = 32,
):
    # Get directories
    astrodino_output_dir = os.path.join(pretrained_dir, "astrodino_output_dir")

    pretrained_weights = {}
    for model in ["astrodino"]:
        pretrained_weights[model] = os.path.join(pretrained_dir, f"{model}.ckpt")

    # Set up AstroDINO model
    astrodino = setup_astrodino(astrodino_output_dir, pretrained_weights["astrodino"])
    print("AstroDINO model set up finished!!!")

    # Set up model dict
    image_model = {
        "astrodino": lambda x: astrodino(x).cpu().numpy(),
    }
    print("Model are correctly set up!")

    # Load data
    files = [quasars_file_train, quasars_file_test]
    for f in files:
        data = load_from_disk(f)

        images=data['image']

        # Get embeddings
        embeddings = get_embeddings(
            image_model, images, batch_size
        )




if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "--quasars_file_train",
        type=str,
        default=f"{ASTROCLIP_ROOT}/data/test/train_dataset",
    )
    parser.add_argument(
        "--quasars_file_test",
        type=str,
        default=f"{ASTROCLIP_ROOT}/data/test/test_dataset",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=f"{ASTROCLIP_ROOT}/pretrained/",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    embed_provabgs(
        args.provabgs_file_train,
        args.provabgs_file_test,
        args.pretrained_dir,
        args.batch_size,
    )
