r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch

# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp.update(
        {
            "gamma" : 0.95,
            "beta": 0.999,
            "learn_rate" : 5e-4,
            "hidden_features": [1024, 512, 256],
            "activation": "relu",
            "dropout": 0,
            "num_workers": 0
        }
    )
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp.update(
        {
            "learn_rate" : 0.001,
            "beta" : 0.3,
            "gamma" : 0.98,
            "delta" : 0.5,
            "batch_size" : 8,
            "hidden_features": [1024, 512, 256],
            "activation": "relu",
            "dropout": 0,
            "num_workers": 0
        }
    )
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

As we learned in class, calculating the **absolute** return value has little meaning in the sense that it is a mere absolute
value with nothing to be compared to.
However when we evaluate a trajectory's performance, it would make much more sense to compare it against an expected total reward
it would usually receive in that specific state.
Subtracting that baseline from the total reward will result in the advantage.
Assuming the advantage value is less than the total reward - I.E the baseline is well defined [like the expectation value]
 and not just an arbitrary value - We would get that the product of the advantage and $\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t})$ is lower,
 therefore summing lower products over all future time will result in lower value which will result in lower variance.
 
 E.G
 Lets assume $q_{vals}$ are in the order of magnitude of $~10^6$ And that the moving average (which we will define as the baseline) of the $q_{vals}$ is also
 in the same order of magnitude. We would end with advantages which are in a much less order of magnitude, and therefore the variance will be lower.


"""


part1_q2 = r"""
**Your answer:**

$\nu$ is expressed as follows as derived in class:
$\nu_{\pi}(s)=\mathbb{E}\left(g_{t}\mid s_{t}=s,\pi\right)=\sum_{a\in\mathcal{A}}\pi(a\mid s)q_{\pi}(s,a)$

However since we do not know the exact distribution of $q$ we can approximate the expectation value with a given sample set \{\hat{q}_{\pi_{\theta},i}\}_{i}^{N}
As $\nu$ will be the expectation value of it.
One of the ways to approximate this expectation value is to learn it via a neural network as proposed in the homework.
All in all as $nu$ is the expectation value of $q$, we would expect that the $\nu_{\pi}(s)$ that is learnt is a valid approximation. 

"""


part1_q3 = r"""
**Your answer:**
1. First run experiment

loss_p - We can see in the graph for a few episode cpg and bpg outperform vgp and epg losses,
However after a certain amount of episodes both vgp & egp improve to and extent they outperform bpg & bpg
which maintain a steady loss value that decreases over time.
This may explained due to the fact that in bpg & cpg we have used a baseline which helps to reduce variance as discussed in previous questions.

loss_e -  Results seem to be close with or without the baseline. This is expected since we are looking at the normalized
entropy, which should not be affected by the baseline.
The loss is not entirely zero, thus the loss still promotes some exploration.


Baseline - baseline computed only for bpg and cpg. Both graphs produce close results to a point where they diverge
and cpg has a higher baseline than bpg.
The cpg uses also entropy-loss and as we know this may causes the agent to do more exploration, and since the baseline reflects the progress of 
the agent we can infer that the baseline should also increase over time - which is the behaviour we see in the cpg.


Mean Reward - Comparing the mean reward by using different losses, we can infer that using a baseline with the entropy-loss (cpg)
is the preferable choice for training our model as it received the best scores.

2. AAC Experiment

Mean Reward - the overall performance of AAC tends to be better than cpg although at earlier episodes we re in favor to cpg.
This can be explained by the learning curve of the critic that is learnt in the process. We assume that after the critic's network
has trained somewhat, we start to see improvement in favor of the acc graph.

loss_p - In earlier episodes we get a behaviour which is similar to the models trained without the baseline. However as we
advance in episodes, the critic network gets trained and therefore we assume to have a closer loss to that of cpg.

"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['x_sigma2'] = 0.001
    hypers['batch_size'] = 32
    hypers['z_dim'] = 64
    hypers['h_dim'] = 256
    hypers['betas'] = (0.95, 0.998)
    hypers['learn_rate'] = 0.0003
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
When the $\sigma^{2}$ is low, then the data-loss term shall receive a greater weight when trying to minimize the total loss.
Therefore we would expect the network to try and to reconstruct the image as closely as possible to the input image 
without trying to optimize by the posterior distribution of q(Z\mid X).
In this case we would expect that training on the generated latent data would not fit the actual
latent distribution, and will overfit the train-data.

On the other hand if $\sigma^{2}$ is high, we would give much more weight to the KLDIV loss. In this case the information gained by the posterior $q(Z\mid X)$ 
will dominate the optimization, and we will under fit our train-data.
"""

part2_q2 = r"""

1. Data-reconstruction loss is the loss induced by the dis-similarity of the input image to the reconstructed image.
Its purpose is to minimize the reconstruction error of an image which goes through the network based on the train-set.
The KLDIV loss measures an amount which indicates how two distributions are similar. It's purpose is to act as a regularization term
which will make sure the training process won't overfit the data, and will be faithful to a known normal-distribution.

2. The latent space distribution is affected directly by the KLDIV as the loss measures dis-similarity between latent space
distribution and the posterior normal distribution. So when minimizing the KLDIV loss, we enforce the latent distribution
to have normal characteristics as the posterior.

3. As we mmentiond in sec. 1, The KLDIV acts as a regularization term, and therefore we ensure this way the we won't overfit
the training data during training. Moreover, we modeled the latent space as a normal distribution so when we generate
new images, we sample from a normal distribution. Therefore by minimizing the KLDIV loss we ensure the model is correct in the 
sense that the latent space has normal-dist. characteristics.  
"""

part2_q3 = r"""
The evidence space is the space of the generated images by our generative model, thus it is a representation of the instance space
that is generated. Maximizing the evidence distribution gives us an indication that we are better approximating the input image
distribution with our generated one.
  
"""

part2_q4 = r"""
We model the log of the variance since in this way we can enforce the variance to be a non-negative number while still enabling
the network to produce negative values as output.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['z_dim'] = 128
    hypers['data_label'] = 0
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0015
    hypers['discriminator_optimizer']['betas'] = (0.45, 0.89)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.001
    hypers['generator_optimizer']['betas'] = (0.45, 0.89)
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
When training the discriminator, we use the generator as a "black box" to generate some samples as adversarial inputs.
We optimize $\delta$ for the discriminator and $\gamma$ for the generator which are different parameters.
The optimization of the discriminator ($\delta$) should not be based on the gradients of the generator ($\gamma$) as the loss function we minimize
is derived for the discriminator, and therefore we discard gradients calculated in the generator.

However when training the Generator, we would optimize the loss function of the generator (Optimize $\gamma$) and therefore
we will maintain the gradients.
"""

part3_q2 = r"""
1. 
We should not base our stop-point for training solely based on the gen-loss as the total loss of the training is based
also on the discriminator loss, and there may be situations where the gen-loss does not satisfy a certain threshold, however
with that specific generator the discriminator achieves better results, and the total loss improves.

In addition, when training the generator we cannot use the discriminator as "black box" in same manner we treated the generator
when training the discriminator, and that is mainly because the loss of the generator depends on output (and gradients) that
are passed though the discriminator, and that is the only way the generator receives any information from the input data.
 (With the discriminator we would stop at the discriminator block and not look at the 
generator gradients).

2.
Given that the generator loss is decreasing, which is the term:
$gen\ loss=-\mathbb{E}_{z\sim p(Z)}\log(\Delta_{\delta}(\Psi_{\gamma}(z)))$
We can infer that the term in the discriminator loss $-\mathbb{E}_{z\sim p(Z)}\log(1-\Delta_{\delta}(\Psi_{\gamma}(z)))$ is increasing.
However since the discriminator loss remains constant, the term $-\mathbb{E}_{x\sim p(X)}\log\Delta_{\delta}(x)$ in the discriminator
should decrease to keep the value constant.
The term represent how well we identify real data, and therefore this scenario implies that the training processes is working properly,
and that the model identifies better real data, however since $\mathbb{E}_{z\sim p(Z)}\log(1-\Delta_{\delta}(\Psi_{\gamma}(z))$
increased, we can assume identifying generated data gets worse.
"""

part3_q3 = r"""
The main differences between the generated images of the two models are:

- VAE generated images seem to be much more blurry than the GAN generated images that are much sharper.
- GAN generated images seem to be less similar to "Bush" real appearance, and seem to generate a sharper looking images of
a "fake bush".

We can explain these differences as the result of the following:

- GAN models are notoriously hard to train as mentioned in tutorials and lectures, and as we experienced with different hyper-params
and discriminator/generator architectures. 
This might cause the GAN model to produce clear images, however at the price of producing ingenuous images.

- VAE model minimizes loss between latent space distribution to a normal distribution, therefore when sampling from a normal
distribution to reconstruct the images we actually sample closer to the actual latent space, and combined with the data-loss
we produce images that represent more the input images (in out case "Bush" images) 
"""

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    X, W, _ = ctx.saved_tensors
    return (grad_output @ W) * 0.5 , (grad_output.T @ X) * 0.5 , grad_output
    # ========================
