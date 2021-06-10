import click
import os
import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader

from config import seed_everything
from datasets.celeba import CelebADataset
from models.generator import SAGANGenerator
from models.discriminator import SAGANDiscriminator
from trainer import SAGANTrainerv2


@click.group()
def cli():
    pass


@cli.command()
@click.argument("root")
@click.option("--n_epochs", default=100)
@click.option("--gen_lr", default=0.0001)
@click.option("--disc_lr", default=0.0004)
@click.option("--z_dim", default=128)
@click.option("--in_channels", default=512)
@click.option("--batch_size", default=64)
@click.option("--n_workers", default=2)
@click.option("--random_state", default=0)
@click.option("--backend", default="gpu")
@click.option("--sample_interval", default=300)
@click.option("--log_step", default=1)
@click.option("--n_train_steps_per_epoch", default=None)
@click.option("--results_dir", default=os.getcwd())
@click.option("--restore_path", default=None)
def train(
    root,
    n_epochs=100,
    gen_lr=0.0001,
    disc_lr=0.0004,
    z_dim=128,
    in_channels=512,
    batch_size=64,
    n_workers=2,
    random_state=0,
    backend="gpu",
    sample_interval=300,
    log_step=1,
    n_train_steps_per_epoch=None,
    results_dir=os.getcwd(),
    restore_path=None,
):
    # RNG fix
    seed_everything(seed=random_state)

    # Define some Preprocessing transforms
    # Currently only supports 64 x 64 size training
    transform = T.Compose(
        [
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Dataset and Loader
    dataset = CelebADataset(root, transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )

    # Model loading
    gen = SAGANGenerator(z_dim=z_dim, in_channels=in_channels)
    disc = SAGANDiscriminator()

    # Trainer
    trainer = SAGANTrainerv2(
        train_loader,
        gen,
        disc,
        backend=backend,
        num_epochs=n_epochs,
        gen_lr=gen_lr,
        disc_lr=disc_lr,
        sample_interval=sample_interval,
        log_step=log_step,
        n_train_steps_per_epoch=n_train_steps_per_epoch,
        results_dir=results_dir,
        optimizer_kwargs={"betas": (0, 0.9)},
    )

    trainer.train(restore_path=restore_path)


if __name__ == "__main__":
    cli()
