import os
import sys

sys.path.append("../..")

from argparse import ArgumentParser

import numpy as np
from astropy.table import Table, join, vstack
from datasets import load_from_disk
from provabgs import models as Models
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from astroclip.data.datamodule import AstroClipCollator, AstroClipDataloader
from astroclip.env import format_with_env

provabgs_file = "https://data.desi.lbl.gov/public/edr/vac/edr/provabgs/v1.0/BGS_ANY_full.provabgs.sv3.v0.hdf5"


def _download_data(save_path: str):
    """Download the PROVABGS data from the web and save it to the specified directory."""
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download the PROVABGS file
    local_path = os.path.join(save_path, "BGS_ANY_full.provabgs.sv3.v0.hdf5")
    if not os.path.exists(local_path):
        print("Downloading PROVABGS data...")
        os.system(f"wget {provabgs_file} -O {local_path}")
        print("Downloaded PROVABGS data successfully!")
    else:
        print("PROVABGS data already exists!")


def _get_best_fit(provabgs: Table):
    """Get the best fit model for each galaxy."""
    m_nmf = Models.NMF(burst=True, emulator=True)

    # Filter out galaxies with no best fit model
    provabgs = provabgs[
        (provabgs["PROVABGS_LOGMSTAR_BF"] > 0)
        * (provabgs["MAG_G"] > 0)
        * (provabgs["MAG_R"] > 0)
        * (provabgs["MAG_Z"] > 0)
    ]

    # Get the thetas and redshifts for each galaxy
    thetas = provabgs["PROVABGS_THETA_BF"][:, :12]
    zreds = provabgs["Z_HP"]

    Z_mw = []  # Stellar Metallicitiy
    tage_mw = []  # Age
    avg_sfr = []  # Star-Forming Region

    print("Calculating best-fit properties using the PROVABGS model...")
    for i in tqdm(range(len(thetas))):
        theta = thetas[i]
        zred = zreds[i]

        # Calculate properties using the PROVABGS model
        Z_mw.append(m_nmf.Z_MW(theta, zred=zred))
        tage_mw.append(m_nmf.tage_MW(theta, zred=zred))
        avg_sfr.append(m_nmf.avgSFR(theta, zred=zred))

    # Add the properties to the table
    provabgs["Z_MW"] = np.array(Z_mw)
    provabgs["TAGE_MW"] = np.array(tage_mw)
    provabgs["AVG_SFR"] = np.array(avg_sfr)
    return provabgs


def cross_match_provabgs(
    astroclip_path: str,
    provabgs_path: str,
    save_path: str = None,
    batch_size: int = 64,
    num_workers: int = 20,
):
    """Cross-match the AstroCLIP and PROVABGS datasets."""

    # Download the PROVABGS data if it doesn't exist
    if not os.path.exists(provabgs_path):
        _download_data(provabgs_path)

    # Load the AstroCLIP dataset
    dataloader = AstroClipDataloader(
        astroclip_path,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=AstroClipCollator(),
        columns=["image", "targetid", "spectrum"],
    )
    dataloader.setup("fit")

    # Load the PROVABGS dataset
    provabgs = Table.read(provabgs_path)

    # Filter out galaxies with no best fit model
    provabgs = provabgs[
        (provabgs["PROVABGS_LOGMSTAR_BF"] > 0)
        * (provabgs["MAG_G"] > 0)
        * (provabgs["MAG_R"] > 0)
        * (provabgs["MAG_Z"] > 0)
    ]

    # Get the best fit model for each galaxy
    print("Getting property best fit with PROVABGS SED model")
    provabgs = _get_best_fit(provabgs)

    # Scale the properties
    provabgs["LOG_MSTAR"] = provabgs["PROVABGS_LOGMSTAR_BF"].data
    provabgs["sSFR"] = np.log(provabgs["AVG_SFR"].data) - np.log(provabgs["Z_MW"].data)
    provabgs["Z_MW"] = np.log(provabgs["Z_MW"].data)

    # Process train data in batches
    train_provabgs_list = []
    for batch in tqdm(dataloader.train_dataloader(), desc="Processing train images"):
        # Convert batch data to numpy arrays
        targetids = batch["targetid"].numpy()
        images = batch["image"].numpy()
        spectra = batch["spectrum"].numpy()
        
        # Create a list of dictionaries for each item in the batch
        batch_data = []
        for i in range(len(targetids)):
            batch_data.append({
                "targetid": targetids[i],
                "image": images[i],
                "spectrum": spectra[i]
            })

        # Create table from list of dictionaries
        batch_table = Table(rows=batch_data)
        # 输出batch_table中的image的shape
        print(f"Shape of images is {batch_table['image'].shape[1:]}", flush=True)

        batch_provabgs = join(provabgs, batch_table, keys_left="TARGETID", keys_right="targetid")
        train_provabgs_list.append(batch_provabgs)
        del batch_table, batch_provabgs, batch_data
        import gc
        gc.collect()

    # Process test data in batches
    test_provabgs_list = []
    for batch in tqdm(dataloader.val_dataloader(), desc="Processing test images"):
        # Convert batch data to numpy arrays
        targetids = batch["targetid"].numpy()
        images = batch["image"].numpy()
        spectra = batch["spectrum"].numpy()
        
        # Create a list of dictionaries for each item in the batch
        batch_data = []
        for i in range(len(targetids)):
            batch_data.append({
                "targetid": targetids[i],
                "image": images[i],
                "spectrum": spectra[i]
            })
        
        # Create table from list of dictionaries
        batch_table = Table(rows=batch_data)
        batch_provabgs = join(provabgs, batch_table, keys_left="TARGETID", keys_right="targetid")
        test_provabgs_list.append(batch_provabgs)
        del batch_table, batch_provabgs, batch_data
        import gc
        gc.collect()

    # Combine the results
    train_provabgs = vstack(train_provabgs_list)
    test_provabgs = vstack(test_provabgs_list)

    print("Number of galaxies in train:", len(train_provabgs))
    print("Number of galaxies in test:", len(test_provabgs))

    # Save the paired datasets
    if save_path is None:
        train_provabgs.write(
            provabgs_path.replace("provabgs.hdf5", "provabgs_paired_train.hdf5"),
            overwrite=True,
        )
        test_provabgs.write(
            provabgs_path.replace("provabgs.hdf5", "provabgs_paired_test.hdf5"),
            overwrite=True,
        )


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "--astroclip_path",
        type=str,
        default=f"{ASTROCLIP_ROOT}/datasets/astroclip_file/",
        help="Path to the AstroCLIP dataset.",
    )
    parser.add_argument(
        "--provabgs_path",
        type=str,
        default=f"{ASTROCLIP_ROOT}/datasets/provabgs/provabgs.hdf5",
        help="Path to the PROVABGS dataset.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the paired datasets.",
    )

    args = parser.parse_args()
    cross_match_provabgs(
        astroclip_path=args.astroclip_path,
        provabgs_path=args.provabgs_path,
        save_path=args.save_path,
    )
