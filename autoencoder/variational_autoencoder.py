from __future__ import annotations
from typing import Any, Callable, Sequence
from typing_extensions import Self
from jax.random import KeyArray
import optax
import jax
import flax
from jax._src.basearray import ArrayLike

# import optax

import torchvision.transforms as TorchTransforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


BATCH_SIZE = 16
LATENT_DIM = 32
KL_WEIGHTS = 0.5
NUMBER_CLASSES = 10
SEED = 0xFFFF


class FeedForward(flax.linen.Module):
    dimensions: Sequence[int] = (256, 128, 64)
    activation: Callable[..., ArrayLike] = flax.linen.relu
    drop_last_activation: bool = False

    @flax.linen.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        for i, dim in enumerate(self.dimensions):
            x = flax.linen.Dense(dim)(x)
            if i != len(self.dimensions) - 1 or not self.drop_last_activation:
                x = self.activation(x)

        return x


class VariationalAutoEncoder(flax.linen.Module):
    encoder_dimensions: Sequence[int] = (256, 128, 64)
    decoder_dimensions: Sequence[int] = (128, 256, 784)
    latent_dimension: int = 4
    activation: Callable[..., ArrayLike] = flax.linen.relu

    def setup(self: Self) -> None:
        super().setup()
        self._encoder = FeedForward(self.encoder_dimensions, self.activation)
        self._pre_latent_projection = flax.linen.Dense(
            self.latent_dimension * 2
        )
        self._post_latent_projection = flax.linen.Dense(
            self.encoder_dimensions[-1]
        )
        self._class_projection = flax.linen.Dense(self.encoder_dimensions[-1])
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
        mean, logvar = jax.numpy.split(
            self._pre_latent_projection(x), 2, axis=-1
        )
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
        key: jax.random.KeyArray,
        model: flax.linen.Module,
        optimizer: Callable[..., optax.GradientTransformation],
    ) -> None:
        self.parameters = model.init(
            key,
            jax.numpy.zeros((BATCH_SIZE, 784)),
            jax.numpy.zeros((BATCH_SIZE, NUMBER_CLASSES)),
        )
        self.state = flax.training.train_state.TrainState


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


if __name__ == "__main__":
    test_feedforward_net()
    test_variational_autoencoder_net()
