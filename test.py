import torch

import matplotlib.pyplot as plt

from models.generator import SAGANGenerator
from models.discriminator import SAGANDiscriminator


chkpt_path = "/home/lexent/Downloads/sagan_chkpt_7.pt"
gen_model = SAGANGenerator(z_dim=128, in_channels=512)
disc_model = SAGANDiscriminator()

gen_model.load_state_dict(torch.load(chkpt_path)["generator_model"])
disc_model.load_state_dict(torch.load(chkpt_path)["discriminator_model"])

gen_model.eval()
disc_model.eval()

with torch.no_grad():
    z = torch.normal(0, 1, size=(1, 128))
    out, _ = gen_model(z)

# plt.figure(figsize=(4, 4))
plt.imshow(out.permute(0, 2, 3, 1).squeeze() * 0.5 + 0.5)
plt.show()
