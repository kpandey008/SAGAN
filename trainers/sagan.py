import os
import torch

from torchvision.utils import save_image

from trainers.base import GANTrainer
from trainers.criterion import GeneratorHingeLoss, DiscriminatorHingeLoss


class SAGANTrainerv2(GANTrainer):
    def init(self, code_size, chkpt_interval=10, **kwargs):
        super().__init__(**kwargs)
        self.code_size = code_size
        self.chkpt_interval = chkpt_interval
        self.gen_criterion = GeneratorHingeLoss()
        self.disc_criterion = DiscriminatorHingeLoss()
        self.best_score = 0
        self.chkpt_name = "sagan_chkpt"
        self.sample_z = torch.normal(0, 1, size=(32, self.code_size))

    def gen_train_step(self, inputs):
        B, _, _, _ = inputs.shape
        z = torch.normal(0, 1, size=(B, self.code_size)).to(self.device)
        fake_images, _ = self.gen_model(z)
        disc_fake_out, _ = self.disc_model(fake_images)

        gen_loss = self.gen_criterion(disc_fake_out)
        gen_loss.backward()

        # Update the generator
        self.gen_optimizer.step()
        return gen_loss

    def disc_train_step(self, inputs):
        real_images = inputs
        B, _, _, _ = real_images.shape
        real_images = real_images.to(self.device)

        z = torch.normal(0, 1, size=(B, self.code_size)).to(self.device)
        fake_images, _ = self.gen_model(z)
        disc_fake_out, _ = self.disc_model(fake_images)
        disc_real_out, _ = self.disc_model(real_images)
        disc_loss = self.disc_criterion(disc_fake_out, disc_real_out)
        disc_loss.backward()

        # Update the Discriminator
        self.disc_optimizer.step()
        return disc_loss

    def on_train_step_end(self, inputs, gen_loss, disc_loss):
        if self.step_idx % self.log_step == 0:
            self.train_progress_bar.set_postfix_str(
                f"Step {self.step_idx + 1} : Gen Loss: {gen_loss.item():.4f}  Disc Loss: {disc_loss.item()}"
            )

        # Generate some random samples and save for visualization
        if self.sample_interval is not None:
            if self.step_idx % self.sample_interval == 0:
                self.gen_model.eval()
                with torch.no_grad():
                    sample_z = self.sample_z.to(self.device)
                    sample_images, _ = self.gen_model(sample_z)

                # Save these images
                save_path = os.path.join(self.results_dir, "samples")
                os.makedirs(save_path, exist_ok=True)
                save_image(
                    sample_images,
                    os.path.join(
                        save_path, f"samples_{self.step_idx}_{self.epoch_idx}.png"
                    ),
                    normalize=True,
                    nrow=4,
                    scale_each=True,
                    range=(0, 1),
                    padding=1,
                )

    def on_train_epoch_end(self):
        # Save checkpoints every 5 epochs
        if self.epoch_idx % self.chkpt_interval == 0:
            if self.results_dir is not None:
                print(f"Saving checkpoint for epoch: {self.epoch_idx + 1}")
                self.save(self.results_dir, f"{self.chkpt_name}_{self.epoch_idx + 1}")
