from argparse import ArgumentParser
from astrodino.env import format_with_env

if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "--file_test",
        type=str,
        default=f"{ASTROCLIP_ROOT}/data/sample_0.1/test_dataset",
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
