"""Stable trainer for the hand-written Dori Transformer."""

import numpy as np


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        tokens,
        seq_len=96,
        batch_size=8,
        val_fraction=0.12,
        seed=0,
        grad_clip=1.0,
    ):
        self.model = model
        self.optimizer = optimizer

        self.tokens = np.asarray(
            tokens,
            dtype=np.int64,
        )

        self.seq_len = max(
            8,
            int(seq_len),
        )

        self.batch_size = max(
            1,
            int(batch_size),
        )

        self.grad_clip = float(grad_clip)

        cut = max(
            self.seq_len + 2,
            int(
                len(self.tokens)
                * (1 - float(val_fraction))
            ),
        )

        cut = min(
            cut,
            len(self.tokens) - 1,
        )

        self.train = self.tokens[:cut]
        self.val = self.tokens[cut:]

        self.rng = np.random.default_rng(seed)

    def _batch(self, arr):

        if len(arr) <= self.seq_len + 1:
            raise ValueError(
                "corpus is too small"
            )

        xs = []
        ys = []

        for _ in range(self.batch_size):

            start = int(
                self.rng.integers(
                    0,
                    len(arr) - self.seq_len,
                )
            )

            xs.append(
                arr[
                    start:
                    start + self.seq_len
                ]
            )

            ys.append(
                arr[
                    start + 1:
                    start + self.seq_len + 1
                ]
            )

        return xs, ys

    def _clip_gradients(self):

        total = 0.0

        for param in self.model.parameters():

            grad = param.grad

            if grad is None:
                continue

            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(
                    "non-finite gradient detected"
                )

            total += float(
                np.sum(
                    grad.astype(
                        np.float64
                    ) ** 2
                )
            )

        norm = float(
            np.sqrt(total)
        )

        if not np.isfinite(norm):
            raise FloatingPointError(
                "non-finite gradient norm"
            )

        if (
            self.grad_clip > 0
            and norm > self.grad_clip
        ):
            scale = (
                self.grad_clip
                / (norm + 1e-12)
            )

            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad *= scale

        return norm

    @staticmethod
    def _cross_entropy(logits, target):

        z = (
            logits.data
            - np.max(
                logits.data,
                axis=1,
                keepdims=True,
            )
        )

        p = np.exp(z)

        p /= np.maximum(
            np.sum(
                p,
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )

        index = np.arange(
            len(target)
        )

        loss = -np.log(
            np.maximum(
                p[index, target],
                1e-12,
            )
        ).mean()

        return float(loss), p

    def _loss_and_backward(self, arr):

        self.model.zero_grad()

        losses = []

        xs, ys = self._batch(arr)

        for ids, target in zip(xs, ys):

            logits = self.model(ids)

            loss, p = self._cross_entropy(
                logits,
                target,
            )

            losses.append(loss)

            # IMPORTANT:
            # Never mutate the original probability array
            # in-place without making a copy.
            grad = p.copy()

            index = np.arange(
                len(target)
            )

            grad[
                index,
                target
            ] -= 1.0

            grad /= len(target)

            logits.backward(
                grad / len(xs)
            )

        grad_norm = self._clip_gradients()

        self.optimizer.step()

        return (
            float(np.mean(losses)),
            grad_norm,
        )

    def train_step(self):

        return self._loss_and_backward(
            self.train
        )

    def evaluate(self, batches=8):

        losses = []

        for _ in range(
            max(1, int(batches))
        ):

            xs, ys = self._batch(
                self.val
            )

            for ids, target in zip(
                xs,
                ys,
            ):

                logits = self.model(ids)

                loss, _ = self._cross_entropy(
                    logits,
                    target,
                )

                losses.append(loss)

        return float(
            np.mean(losses)
        )