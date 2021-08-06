import click
import os
import torchvision.transforms as T

from torch.utils.data import DataLoader

from config import seed_everything, get_dataset
from models.generator import SAGANGenerator
from models.discriminator import SAGANDiscriminator
from trainers.sagan import SAGANTrainerv2


@click.group()
def cli():
    pass


@cli.command()
@click.argument("root")
@click.option("--code-size", default=128)
@click.option("--dataset", default="celeba-hq")
@click.option("--subsample", default=-1)
@click.option("--chkpt-interval", default=10)
@click.option("--n-epochs", default=1000)
@click.option("--gen-lr", default=0.0001, type=float)
@click.option("--disc-lr", default=0.0004, type=float)
@click.option("--in-channels", default=512)
@click.option("--batch-size", default=64)
@click.option("--n-workers", default=2)
@click.option("--random-state", default=0)
@click.option("--backend", default="gpu")
@click.option("--sample-interval", default=300)
@click.option("--log-step", default=1)
@click.option("--n-train-steps-per-epoch", default=None)
@click.option("--results-dir", default=os.getcwd())
@click.option("--restore-path", default=None)
def train(
    root,
    dataset="celeba-hq",
    subsample=-1,
    code_size=128,
    chkpt_interval=10,
    n_epochs=1000,
    gen_lr=0.0001,
    disc_lr=0.0004,
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
    transform = T.Compose(
        [
            T.Resize(128),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Dataset and Loader
    train_dataset = get_dataset(
        dataset,
        root,
        transform=transform,
        subsample_size=subsample if subsample != -1 else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )

    # Model loading
    gen = SAGANGenerator(z_dim=code_size, in_channels=in_channels)
    disc = SAGANDiscriminator()

    # Trainer
    trainer = SAGANTrainerv2(
        code_size=code_size,
        chkpt_interval=chkpt_interval,
        train_loader=train_loader,
        gen_model=gen,
        disc_model=disc,
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
