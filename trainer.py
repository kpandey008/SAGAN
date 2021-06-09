import copy
import os
import torch

from prettytable import PrettyTable
from tqdm import tqdm
from torchvision.utils import save_image

from config import configure_device, get_lr_scheduler, get_optimizer
from criterion import GeneratorHingeLoss, DiscriminatorHingeLoss


class GANTrainer:
    def __init__(
        self,
        train_loader,
        gen_model,
        disc_model,
        num_epochs,
        val_loader=None,
        lr_scheduler=None,
        gen_lr=0.0001,
        disc_lr=0.0001,
        imbalance_factor=1,
        sample_interval=None,
        log_step=50,
        n_train_steps_per_epoch=None,
        gen_optimizer="Adam",
        disc_optimizer="Adam",
        backend="gpu",
        results_dir=None,
        n_val_steps=None,
        optimizer_kwargs={},
        lr_scheduler_kwargs={},
        **kwargs,
    ):
        # Create the dataset
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.imbalance_factor = imbalance_factor
        self.sample_interval = sample_interval
        self.device = configure_device(backend)
        self.val_loader = val_loader
        self.log_step = log_step
        self.gen_loss_profile = []
        self.disc_loss_profile = []
        self.n_train_steps_per_epoch = n_train_steps_per_epoch
        self.n_val_steps = n_val_steps
        self.train_progress_bar = None
        self.val_progress_bar = None
        self.results_dir = results_dir

        if (self.results_dir is not None) and (not os.path.isdir(self.results_dir)):
            os.makedirs(self.results_dir, exist_ok=True)

        self.gen_model = gen_model.to(self.device)
        self.disc_model = disc_model.to(self.device)

        self.gen_optimizer = get_optimizer(
            gen_optimizer, self.gen_model, self.gen_lr, **optimizer_kwargs
        )
        self.disc_optimizer = get_optimizer(
            disc_optimizer, self.disc_model, self.disc_lr, **optimizer_kwargs
        )
        self.lr_scheduler = None
        self.sched_type = lr_scheduler
        self.sched_kwargs = lr_scheduler_kwargs
        self.start_epoch = 0

        # Call some custom initialization here
        self.init()

    def init(self):
        pass

    def summarize(self):
        models = [self.gen_model, self.disc_model]

        # Generate Model parameter summary here
        print("Model Params profile:")
        total_params = 0
        total_trainable = 0

        for model in models:
            p_table = PrettyTable()
            p_table.field_names = ["Name", "Trainable Params", "Total Params"]
            print(f"Model: {model.__class__.__name__}")
            for name, mod in model.named_children():
                n_trainable_params = 0
                n_params = 0
                for p in mod.parameters():
                    n_params += torch.numel(p)
                    if p.requires_grad:
                        n_trainable_params += torch.numel(p)
                total_params += n_params
                total_trainable += n_trainable_params
                p_table.add_row([name, n_trainable_params, n_params])
            print(p_table)

        print(f"Total Trainable Params: {total_trainable}")
        print(f"Total Params: {total_params}")

        print("Training Params")
        print(f"Generator Optimizer: {self.gen_optimizer}")
        print(f"Discriminator Optimizer: {self.disc_optimizer}")

        print(f"Generator LR: {self.gen_lr}")
        print(f"Discriminator LR: {self.disc_lr}")

        print(f"Num. epochs: {self.num_epochs}")
        print(f"Num. Train Steps per epoch: {self.n_train_steps_per_epoch}")
        print(f"Num. Val Steps: {self.n_val_steps}")
        print(f"LR scheduler: {self.sched_type}")
        print(f"Results Dir: {self.results_dir}")
        print(f"Device: {self.device}")

    def train(self, restore_path=None):
        # Configure lr scheduler
        if self.sched_type is not None:
            self.lr_scheduler = get_lr_scheduler(
                self.optimizer,
                self.num_epochs,
                sched_type=self.sched_type,
                **self.sched_kwargs,
            )

        # Restore checkpoint if available
        if restore_path is not None:
            # Load the model
            self.load(restore_path)

        best_eval = 0.0
        tk0 = range(self.start_epoch, self.num_epochs)

        self.epoch_idx = 0

        # Display a summary before starting training
        self.summarize()

        for _ in tk0:
            print(f"Training for epoch: {self.epoch_idx + 1}")
            avg_epoch_loss_g, avg_epoch_loss_d = self.train_one_epoch()
            print(
                f"Avg Loss for epoch: Generator - {avg_epoch_loss_g}, Discriminator - {avg_epoch_loss_d}"
            )

            # LR scheduler step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Build loss profile
            self.gen_loss_profile.append(avg_epoch_loss_g)
            self.disc_loss_profile.append(avg_epoch_loss_d)

            # Evaluate the model
            if self.val_loader is not None:
                self.eval()

            self.epoch_idx += 1

    def train_one_epoch(self):
        # Put in training mode
        self.gen_model.train()
        self.disc_model.train()

        gen_epoch_loss = 0
        disc_epoch_loss = 0

        self.train_progress_bar = tqdm(self.train_loader)
        self.step_idx = 0
        for _, inputs in enumerate(self.train_progress_bar):
            if (
                self.n_train_steps_per_epoch is not None
                and (self.step_idx + 1) > self.n_train_steps_per_epoch
            ):
                break
            # Zero gradients
            self.gen_optimizer.zero_grad()
            g_step_loss = self.gen_train_step(inputs)

            d_step_loss = 0
            for _ in range(self.imbalance_factor):
                self.disc_optimizer.zero_grad()
                d_step_loss += self.disc_train_step(inputs)

            self.on_train_step_end(inputs, g_step_loss, d_step_loss)
            gen_epoch_loss += g_step_loss
            disc_epoch_loss += d_step_loss
            self.step_idx += 1

        self.on_train_epoch_end()
        return gen_epoch_loss / len(self.train_loader), disc_epoch_loss / len(
            self.train_loader
        )

    def gen_train_step(self, inputs):
        raise NotImplementedError()

    def disc_train_step(self, inputs):
        raise NotImplementedError()

    def on_train_epoch_end(self):
        pass

    def on_train_step_end(self, *args):
        pass

    def eval(self):
        self.gen_model.eval()
        self.disc_model.eval()
        self.val_progress_bar = tqdm(self.val_loader)
        with torch.no_grad():
            for idx, inputs in enumerate(self.val_progress_bar):
                if self.n_val_steps is not None and (idx + 1) > self.n_val_steps:
                    break
                self.val_step(inputs)
                self.on_val_step_end()
            self.on_val_epoch_end()

    def val_step(self):
        raise NotImplementedError()

    def on_val_epoch_end(self):
        pass

    def on_val_step_end(self):
        pass

    def save(self, path, name, prefix=None, remove_prev_chkpt=True):
        checkpoint_name = f"{name}_{prefix}" if prefix is not None else name
        path = path if prefix is None else os.path.join(path, prefix)
        checkpoint_path = os.path.join(path, f"{checkpoint_name}.pt")

        # Store model state, optimizer state and scheduler state on the checkpoint
        state_dict = {}
        models = [self.gen_model, self.disc_model]
        optimizers = [self.gen_optimizer, self.disc_optimizer]
        model_keys = ["generator", "discriminator"]

        for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            model_state = copy.deepcopy(model.state_dict())
            model_state = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            }
            state_dict[f"{model_keys[idx]}_model"] = model_state

            optim_state = copy.deepcopy(optimizer.state_dict())
            for state in optim_state["state"].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

            state_dict[f"{model_keys[idx]}_optimizer"] = optim_state

        if self.lr_scheduler is not None:
            state_dict["scheduler"] = self.lr_scheduler.state_dict()
        state_dict["epoch"] = self.epoch_idx + 1
        state_dict["gen_loss_profile"] = self.gen_loss_profile
        state_dict["disc_loss_profile"] = self.disc_loss_profile

        # Create parent dir
        os.makedirs(path, exist_ok=True)
        # Remove all prev checkpoints if enabled
        if remove_prev_chkpt:
            for f in os.listdir(path):
                if f.endswith(".pt"):
                    os.remove(os.path.join(path, f))
        torch.save(state_dict, checkpoint_path)

    def load(self, load_path):
        state_dict = torch.load(load_path)
        iter_val = state_dict.get("epoch", 0)
        self.epoch_idx = iter_val
        self.gen_loss_profile = state_dict.get("gen_loss_profile", [])
        self.disc_loss_profile = state_dict.get("disc_loss_profile", [])

        if "generator_model" in state_dict:
            print("Restoring Generator Model state")
            self.gen_model.load_state_dict(state_dict["generator_model"])

        if "discriminator_model" in state_dict:
            print("Restoring Discriminator Model state")
            self.disc_model.load_state_dict(state_dict["discriminator_model"])

        if "generator_optimizer" in state_dict:
            print("Restoring Generator Optimizer state")
            self.gen_optimizer.load_state_dict(state_dict["generator_optimizer"])
            # manually move the optimizer state vectors to device
            for state in self.gen_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if "discriminator_optimizer" in state_dict:
            print("Restoring Discriminator Optimizer state")
            self.disc_optimizer.load_state_dict(state_dict["discriminator_optimizer"])
            # manually move the optimizer state vectors to device
            for state in self.disc_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if self.lr_scheduler is not None:
            if "scheduler" in state_dict:
                print("Restoring Learning Rate scheduler state")
                self.lr_scheduler.load_state_dict(state_dict["scheduler"])


class SAGANTrainer(GANTrainer):
    def init(self):
        self.code_size = 100
        self.gen_criterion = GeneratorHingeLoss()
        self.disc_criterion = DiscriminatorHingeLoss()
        self.best_score = 0
        self.chkpt_name = "sagan_chkpt"

    def gen_train_step(self, inputs):
        B, _, _, _ = inputs.shape
        z = torch.normal(0, 1, size=(B, self.code_size)).to(self.device)
        fake_images, _, _ = self.gen_model(z)
        disc_fake_out, _, _ = self.disc_model(fake_images)

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
        fake_images, _, _ = self.gen_model(z)
        disc_fake_out, _, _ = self.disc_model(fake_images)
        disc_real_out, _, _ = self.disc_model(real_images)
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
                    sample_z = torch.normal(0, 1, size=(32, self.code_size))
                    sample_images, _, _ = self.gen_model(sample_z)

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
        if self.epoch_idx % 5 == 0:
            if self.results_dir is not None:
                print(f"Saving checkpoint for epoch: {self.epoch_idx + 1}")
                self.save(self.results_dir, f"{self.chkpt_name}_{self.epoch_idx + 1}")
