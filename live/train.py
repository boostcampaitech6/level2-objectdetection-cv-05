import argparse
import collections
from parse_config import ConfigParser
import data_loader as module_data
import model as module_arch

import torchvision
from trainer import Trainer
import torch


def main(config):
    train_data_loader = config.init_data_loader("data_loader", module_data)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    model = config.init_obj("arch", module_arch)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = config["trainer"]["epochs"]
    save_path = config["trainer"]["save_path"]

    # training
    trainer = Trainer(
        num_epochs, train_data_loader, optimizer, model, device, save_path
    )
    trainer.train_fn()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Practical Pytorch")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: None)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", ["flags", "type", "target"])
    options = [
        CustomArgs(
            flags=["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"
        ),
        CustomArgs(
            flags=["--bs", "--batch_size"],
            type=int,
            target="data_loader;args;batch_size",
        ),
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
