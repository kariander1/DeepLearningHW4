import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        in_channels = in_size[0]
        act = torch.nn.LeakyReLU(0.1)
        mod_channels = [in_channels, 128, 256,512,1024, 1]
        for i_channel in range(1, len(mod_channels)):
            modules += [nn.Conv2d(in_channels=mod_channels[i_channel - 1], out_channels=mod_channels[i_channel],
                                  stride=2, kernel_size=5, padding=1), act,nn.BatchNorm2d(mod_channels[i_channel])]

        modules.pop() # remove last batch norm
        self.cnn = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        h = self.cnn(x)
        y = torch.flatten(h,1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.featuremap_size = featuremap_size
        modules = []

        act = torch.nn.ReLU()
        #in_channels = z_dim//(featuremap_size**2)
        in_channels = z_dim
        mod_channels = [in_channels,1024, 512, 256, 128,64, out_channels]
        for i_channel in range(1, len(mod_channels)):
            modules += [nn.ConvTranspose2d(in_channels=mod_channels[i_channel - 1], out_channels=mod_channels[i_channel],
                                  stride=2, kernel_size=4, padding=1), act, nn.BatchNorm2d(mod_channels[i_channel])]

        modules.pop()  # Remove last batch norm
        modules[-1] = nn.Tanh() # Replace last activation
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            normal = torch.randn(n,self.z_dim).to(device)
            samples = self(normal)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        fm_size = self.featuremap_size
        # h = torch.reshape(z, shape=(-1, self.z_dim // (fm_size ** 2), fm_size, fm_size))
        h = z.view(-1, self.z_dim, 1, 1).to(next(self.parameters()).device)
        x = self.cnn(h)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()
    data_noise = ((torch.rand(y_data.shape) - 0.5) * label_noise).to(y_data.device)
    generated_noise = ((torch.rand(y_generated.shape) - 0.5) * label_noise).to(y_generated.device)

    loss_data = loss_fn(y_data,data_label+ data_noise)
    loss_generated = loss_fn(y_generated, 1-data_label+generated_noise)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()
    labels = torch.full(size=y_generated.shape, fill_value=data_label,dtype=torch.float).to(y_generated.device)
    loss = loss_fn(y_generated, labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======

    # 1.
    x_generated = gen_model.sample(x_data.shape[0])
    y_data = dsc_model(x_data)
    y_generated = dsc_model(x_generated)

    # 2.
    dsc_loss = dsc_loss_fn(y_data,y_generated)

    # 3.
    dsc_optimizer.zero_grad()
    dsc_loss.backward(retain_graph=True)
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======

    # 1.
    gen_x_generated = gen_model.sample(x_data.shape[0],with_grad=True)
    gen_y_generated = dsc_model(gen_x_generated)

    # 2.
    gen_loss = gen_loss_fn(gen_y_generated)

    # 3.
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    mean = sum(dsc_losses) / len(dsc_losses)
    dsc_var = sum((i - mean) ** 2 for i in dsc_losses) / len(dsc_losses)
    print(dsc_var)
    torch.save(gen_model, checkpoint_file)
    saved = True
    # ========================

    return saved
