import click
import torch

from torchvision.utils import save_image

from models.generator import SAGANGenerator
from models.discriminator import SAGANDiscriminator


@click.group()
def cli():
    pass


@cli.command()
@click.argument("chkpt-path")
@click.option("--n_samples", default=16)
@click.option("--save_path", default=None)
@click.option("--norm", type=bool, default=True)
def generate_samples(chkpt_path, n_samples=16, save_path=None, norm=True):
    # Model loading
    gen_model = SAGANGenerator(z_dim=128, in_channels=512)
    disc_model = SAGANDiscriminator()

    gen_model.load_state_dict(torch.load(chkpt_path)["generator_model"])
    disc_model.load_state_dict(torch.load(chkpt_path)["discriminator_model"])

    gen_model.eval()
    disc_model.eval()

    # Sample generation
    with torch.no_grad():
        z = torch.normal(0, 1, size=(n_samples, 128))
        out, _ = gen_model(z)

    # Normalization
    if norm:
        out = out * 0.5 + 0.5

    save_image(out, save_path, nrow=4)


if __name__ == "__main__":
    cli()
