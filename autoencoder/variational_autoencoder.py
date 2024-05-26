from functools import partial
from typing import Callable, Sequence
from jax._src.random import KeyArray
from typing_extensions import Self

import optax
import jax
import flax
from jax._src.basearray import ArrayLike
import flax.linen as nn

import torchvision.transforms as TorchTransforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


BATCH_SIZE = 16
LATENT_DIM = 32
KL_WEIGHTS = 0.5
NUMBER_CLASSES = 10
SEED = 0xFFFF


class FeedForward(nn.Module):
    dimensions: Sequence[int] = (256, 128, 64)
    activation: Callable[..., ArrayLike] = nn.relu
    drop_last_activation: bool = False

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        for i, dim in enumerate(self.dimensions):
            x = nn.Dense(dim)(x)
            if i != len(self.dimensions) - 1 or not self.drop_last_activation:
                x = self.activation(x)

        return x


class VariationalAutoEncoder(nn.Module):
    encoder_dimensions: Sequence[int] = (256, 128, 64)
    decoder_dimensions: Sequence[int] = (128, 256, 784)
    latent_dimension: int = 4
    activation: Callable[..., ArrayLike] = nn.relu

    def setup(self: Self) -> None:
        super().setup()
        self._encoder = FeedForward(self.encoder_dimensions, self.activation)
        self._pre_latent_projection = nn.Dense(self.latent_dimension * 2)
        self._post_latent_projection = nn.Dense(self.encoder_dimensions[-1])
        self._class_projection = nn.Dense(self.encoder_dimensions[-1])
        self._decoder = FeedForward(
            self.decoder_dimensions,
            self.activation,
            drop_last_activation=False,
        )

    def reparametrize(
        self, mean: ArrayLike, logvar: ArrayLike, key: jax.random.KeyArray
    ) -> ArrayLike:
        assert isinstance(mean, jax.Array)
        assert isinstance(logvar, jax.Array)
        std = jax.numpy.exp(logvar * 0.5)
        eps = jax.random.normal(key, mean.shape)
        return eps * std + mean

    def encode(self, x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        x = self._encoder(x)
        mean, logvar = jax.numpy.split(self._pre_latent_projection(x), 2, axis=-1)
        return mean, logvar

    def decode(self, x: ArrayLike, projection: ArrayLike) -> ArrayLike:
        x = self._post_latent_projection(x)
        x = x + self._class_projection(projection)
        x = self._decoder(x)
        return x

    def __call__(
        self, x: ArrayLike, projection: ArrayLike, key: jax.random.KeyArray
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar, key)
        y = self.decode(z, projection)
        return y, mean, logvar


class TrainingStep:
    def __init__(
        self,
        key: KeyArray,
        model: nn.Module,
        optimizer: optax.GradientTransformation | None,
    ) -> None:
        self._parameters = model.init(
            key,
            jax.numpy.zeros((BATCH_SIZE, 784)),
            jax.numpy.zeros((BATCH_SIZE, NUMBER_CLASSES)),
        )
        self.model: nn.Module = model
        self._optimizer: optax.GradientTransformation = optax.adam(learning_rate=0.01)
        self._optimizer_state: optax.OptState = self._optimizer.init(self._parameters)

    def loss(
        self, x: jax.Array, c: ArrayLike, key: KeyArray
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        reduce_dims = list(range(1, len(x.shape)))
        c = jax.nn.one_hot(c, NUMBER_CLASSES)
        recon, mean, logvar = self.model.apply(self._parameters, x, c, key)
        assert isinstance(mean, jax.Array)
        mse_loss = optax.l2_loss(recon, x).sum(axis=reduce_dims).mean()
        kl_loss = jax.numpy.mean(
            -0.5
            * jax.numpy.sum(
                1 + logvar - mean**2 - jax.numpy.exp(logvar), axis=reduce_dims
            )
        )
        loss = mse_loss + KL_WEIGHTS * kl_loss
        return loss, (mse_loss, kl_loss)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        x: jax.Array,
        c: jax.Array,
        key: KeyArray,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        losses, grads = jax.value_and_grad(self.loss, has_aux=True)(
            self._parameters, x, c, key
        )
        loss, (mse_loss, kl_loss) = losses
        assert isinstance(loss, jax.Array)
        assert isinstance(mse_loss, jax.Array) and isinstance(kl_loss, jax.Array)
        updates, opt_state = self._optimizer.update(
            grads, self._optimizer_state, self._parameters
        )
        parameters = optax.apply_updates(self._parameters, updates)

        self.update_parameters_and_state(parameters, opt_state)

        return loss, mse_loss, kl_loss

    def update_parameters_and_state(
        self, parameters: optax.Params, state: optax.OptState
    ) -> None:
        self._parameters = parameters
        self._optimizer_state = state

    @property
    def parameters(self) -> optax.Params:
        return self._parameters

    @property
    def optimizer(self) -> optax.GradientTransformation:
        return self._optimizer

    @property
    def optimizer_state(self) -> optax.OptState:
        return self._optimizer_state


def test_feedforward_net() -> None:
    init_key = jax.random.PRNGKey(SEED)
    key, model_key = jax.random.split(init_key)
    model = FeedForward(dimensions=(4, 2, 1), drop_last_activation=True)
    print(f"FeedForward:\n{model}")

    parameters = model.init(model_key, jax.numpy.zeros((1, 8)))
    print(f"Parameters:\n{parameters}")

    key, input_key = jax.random.split(init_key)
    x = jax.random.normal(input_key, (1, 8))
    y = model.apply(parameters, x)
    print(f"Evaluation:\n{y}")


def test_variational_autoencoder_net() -> None:
    key = jax.random.PRNGKey(0x1234)
    key, model_key = jax.random.split(key)
    model = VariationalAutoEncoder(latent_dimension=4)
    print(f"VariationalAutoEncoder:\n{model}")

    key, call_key = jax.random.split(key)
    parameters = model.init(
        model_key,
        jax.numpy.zeros((BATCH_SIZE, 784)),
        jax.numpy.zeros((BATCH_SIZE, NUMBER_CLASSES)),
        call_key,
    )

    recontruction_tuple = model.apply(
        parameters,
        jax.numpy.zeros((BATCH_SIZE, 784)),
        jax.numpy.zeros((BATCH_SIZE, NUMBER_CLASSES)),
        call_key,
    )
    assert isinstance(recontruction_tuple, tuple)
    recon, mean, logvar = recontruction_tuple
    assert all(isinstance(elem, jax.Array) for elem in [recon, mean, logvar])

    print(f"Reconstruction shape: {recon.shape}")
    print(f"mean shape: {recon.shape}, logvar shape: {logvar.shape}")


def test_variational_autoencoder_training() -> None:
    train_dataset = MNIST(
        "data", train=True, transform=TorchTransforms.ToTensor(), download=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    key = jax.random.PRNGKey(0x1234)
    key, model_key = jax.random.split(key)
    model = VariationalAutoEncoder(latent_dimension=LATENT_DIM)
    optimizer = optax.adamw(learning_rate=1e-4)
    training_step = TrainingStep(model_key, model, optimizer)

    for epoch in range(10):
        total_loss, total_mse, total_kl = 0.0, 0.0, 0.0
        for i, (batch, c) in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            batch = batch.numpy().reshape(BATCH_SIZE, 784)
            c = c.numpy()
            loss, mse_loss, kl_loss = training_step.train_step(batch, c, subkey)

            total_loss += loss
            total_mse += mse_loss
            total_kl += kl_loss

            if i > 0 and not i % 100:
                print(f"{epoch = } | step {i} | loss: {total_loss/100}")
                total_loss = 0.0
                total_mse, total_kl = 0.0, 0.0

    sample = get_sample_from_training(key, model, training_step.parameters)
    plot_sample_from_training(sample, 10, 10)


@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
    ),
)
def sample_from_model(
    model: nn.Module, params: optax.Params, z: ArrayLike, c: ArrayLike
) -> ArrayLike:
    sample = model.apply(params, z, c, method=model.decode)
    assert isinstance(sample, ArrayLike)
    return sample


def plot_sample_from_training(sample: ArrayLike, h: int, w: int) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import einsum

    sample = einsum("ikjl", np.asarray(sample).reshape(h, w, 28, 28)).reshape(
        28 * h, 28 * w
    )

    plt.imshow(sample, cmap="gray")
    plt.show()


def get_sample_from_training(
    key: KeyArray, model: nn.Module, params: optax.Params
) -> ArrayLike:
    import numpy as np

    num_samples = 100
    h, w = 10, 10

    key, z_key = jax.random.split(key)
    z = jax.random.normal(z_key, (num_samples, LATENT_DIM))
    c = np.repeat(np.arange(h)[:, np.newaxis], w, axis=-1).flatten()
    c = jax.nn.one_hot(c, NUMBER_CLASSES)
    sample = sample_from_model(model, params, z, c)

    return sample


if __name__ == "__main__":
    test_feedforward_net()
    test_variational_autoencoder_net()
    test_variational_autoencoder_training()
