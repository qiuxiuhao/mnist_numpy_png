#!/usr/bin/env python3
"""
MNIST PNG handwritten digit recognition using NumPy + Pillow.

No PyTorch.
No TensorFlow.
No sklearn.
No OpenCV required.

Model:
    784 -> hidden -> ReLU -> 10 -> Softmax

Usage:

Train:
    python mnist_numpy_png.py train \
        --data-dir data/mnist_png \
        --epochs 5 \
        --batch-size 128 \
        --hidden 128 \
        --lr 0.1

Evaluate:
    python mnist_numpy_png.py eval \
        --data-dir data/mnist_png \
        --model models/mnist_mlp_png.npz

Predict one image:
    python mnist_numpy_png.py predict \
        --model models/mnist_mlp_png.npz \
        --image path/to/digit.png

Predict one image from test set by index:
    python mnist_numpy_png.py predict-index \
        --data-dir data/mnist_png \
        --model models/mnist_mlp_png.npz \
        --index 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def find_split_dir(data_dir: Path, candidates: List[str]) -> Path:
    """
    Find a split directory.

    Supports:
        data/mnist_png/train/0/*.png
        data/mnist_png/test/0/*.png

    Also supports:
        data/mnist_png/training/0/*.png
        data/mnist_png/testing/0/*.png

    If the split is nested one level deeper, this function also tries to find it.
    """
    data_dir = data_dir.resolve()

    # Direct check.
    for name in candidates:
        candidate = data_dir / name
        if candidate.exists() and candidate.is_dir() and has_digit_subdirs(candidate):
            return candidate

    # Recursive fallback.
    for child in data_dir.rglob("*"):
        if child.is_dir() and child.name.lower() in candidates and has_digit_subdirs(child):
            return child

    raise FileNotFoundError(
        f"Cannot find split directory under {data_dir}. "
        f"Tried names: {candidates}. "
        f"Expected something like train/0/*.png and test/0/*.png."
    )


def has_digit_subdirs(path: Path) -> bool:
    """
    Check whether a directory contains class folders 0 to 9.
    """
    count = 0
    for digit in range(10):
        if (path / str(digit)).exists() and (path / str(digit)).is_dir():
            count += 1
    return count >= 5


def list_image_files(split_dir: Path) -> List[Tuple[Path, int]]:
    """
    List image paths and labels from split directory.

    Expected:
        split_dir/0/*.png
        split_dir/1/*.png
        ...
        split_dir/9/*.png

    Label is read from parent folder name.
    """
    items: List[Tuple[Path, int]] = []

    for label in range(10):
        class_dir = split_dir / str(label)
        if not class_dir.exists():
            continue

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                items.append((path, label))

    if not items:
        raise RuntimeError(f"No image files found in {split_dir}")

    return items


def load_one_image(path: Path, image_size: int = 28) -> np.ndarray:
    """
    Load one image as a flattened float32 vector.

    Output:
        shape: [784]
        value range: [0, 1]
    """
    img = Image.open(path).convert("L")

    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def load_png_split(
    split_dir: Path,
    image_size: int = 28,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all PNG images from one split.

    Returns:
        x: float32, [N, 784]
        y: int64, [N]
    """
    items = list_image_files(split_dir)

    x_list = []
    y_list = []

    print(f"[load] {split_dir}")
    print(f"[load] found {len(items)} images")

    for i, (path, label) in enumerate(items, start=1):
        x_list.append(load_one_image(path, image_size=image_size))
        y_list.append(label)

        if i % 10000 == 0:
            print(f"[load] {i}/{len(items)} images loaded")

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    return x, y


def maybe_subset(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly choose a subset, useful for debugging.
    """
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return x, y

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))[:max_samples]
    return x[indices], y[indices]


def load_dataset(
    data_dir: Path,
    image_size: int = 28,
    cache: bool = True,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/test PNG dataset.

    Supports cache:
        data/mnist_png/mnist_png_cache_28.npz
    """
    data_dir = data_dir.resolve()
    cache_path = data_dir / f"mnist_png_cache_{image_size}.npz"

    if cache and cache_path.exists():
        print(f"[cache] loading {cache_path}")
        data = np.load(cache_path)
        x_train = data["x_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        x_test = data["x_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)
    else:
        train_dir = find_split_dir(data_dir, ["train", "training"])
        test_dir = find_split_dir(data_dir, ["test", "testing"])

        x_train, y_train = load_png_split(train_dir, image_size=image_size)
        x_test, y_test = load_png_split(test_dir, image_size=image_size)

        if cache:
            print(f"[cache] saving {cache_path}")
            np.savez_compressed(
                cache_path,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

    x_train, y_train = maybe_subset(x_train, y_train, max_train, seed)
    x_test, y_test = maybe_subset(x_test, y_test, max_test, seed + 1)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_backward(grad: np.ndarray, z: np.ndarray) -> np.ndarray:
    return grad * (z > 0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Mean softmax cross entropy.
    """
    eps = 1e-12
    n = y.shape[0]
    correct = probs[np.arange(n), y]
    return float(-np.mean(np.log(correct + eps)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y))


class MLP:
    """
    Two-layer MLP.

    Structure:
        x -> Linear -> ReLU -> Linear -> Softmax
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # He initialization.
        self.W1 = (
            rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        ).astype(np.float32)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)

        self.W2 = (
            rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)
        ).astype(np.float32)
        self.b2 = np.zeros((output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        logits = a1 @ self.W2 + self.b2
        probs = softmax(logits)

        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "logits": logits,
            "probs": probs,
        }

        return probs, cache

    def backward(
        self,
        cache: Dict[str, np.ndarray],
        y: np.ndarray,
        weight_decay: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        probs = cache["probs"]

        n = x.shape[0]

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        dW2 = a1.T @ dlogits
        db2 = np.sum(dlogits, axis=0)

        da1 = dlogits @ self.W2.T
        dz1 = relu_backward(da1, z1)

        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        if weight_decay > 0:
            dW1 += weight_decay * self.W1
            dW2 += weight_decay * self.W2

        return {
            "W1": dW1.astype(np.float32),
            "b1": db1.astype(np.float32),
            "W2": dW2.astype(np.float32),
            "b2": db2.astype(np.float32),
        }

    def step(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def predict_proba(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        probs_all = []

        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            probs, _ = self.forward(x[start:end])
            probs_all.append(probs)

        return np.concatenate(probs_all, axis=0)

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        probs = self.predict_proba(x, batch_size=batch_size)
        return np.argmax(probs, axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            input_dim=self.W1.shape[0],
            hidden_dim=self.W1.shape[1],
            output_dim=self.W2.shape[1],
        )

    @classmethod
    def load(cls, path: Path) -> "MLP":
        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {path}")

        data = np.load(path)

        input_dim = int(data["input_dim"])
        hidden_dim = int(data["hidden_dim"])
        output_dim = int(data["output_dim"])

        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        model.W1 = data["W1"].astype(np.float32)
        model.b1 = data["b1"].astype(np.float32)
        model.W2 = data["W2"].astype(np.float32)
        model.b2 = data["b2"].astype(np.float32)

        return model


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    n = x.shape[0]
    indices = rng.permutation(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def evaluate_model(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
) -> Tuple[float, float]:
    probs = model.predict_proba(x, batch_size=batch_size)
    loss = cross_entropy(probs, y)
    acc = accuracy(probs, y)
    return loss, acc


def write_log_header(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "time_sec",
            ]
        )


def append_log(
    log_path: Path,
    epoch: int,
    lr: float,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    time_sec: float,
) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                lr,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                time_sec,
            ]
        )


def train_command(args) -> None:
    data_dir = Path(args.data_dir)

    x_train, y_train, x_test, y_test = load_dataset(
        data_dir=data_dir,
        image_size=args.image_size,
        cache=not args.no_cache,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
    )

    model = MLP(
        input_dim=args.image_size * args.image_size,
        hidden_dim=args.hidden,
        output_dim=10,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    output_path = Path(args.output)
    log_path = Path(args.log)

    print()
    print("[config]")
    print(f"  data_dir     : {data_dir}")
    print(f"  train samples: {len(y_train)}")
    print(f"  test samples : {len(y_test)}")
    print(f"  image size   : {args.image_size}x{args.image_size}")
    print(f"  hidden       : {args.hidden}")
    print(f"  epochs       : {args.epochs}")
    print(f"  batch size   : {args.batch_size}")
    print(f"  lr           : {args.lr}")
    print(f"  lr decay     : {args.lr_decay}")
    print(f"  weight decay : {args.weight_decay}")
    print(f"  output       : {output_path}")
    print(f"  log          : {log_path}")
    print()

    write_log_header(log_path)

    best_test_acc = -1.0
    lr = args.lr

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        batch_losses = []
        batch_accs = []

        for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, rng):
            probs, cache = model.forward(xb)
            loss = cross_entropy(probs, yb)
            acc = accuracy(probs, yb)

            grads = model.backward(cache, yb, weight_decay=args.weight_decay)
            model.step(grads, lr)

            batch_losses.append(loss)
            batch_accs.append(acc)

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))

        test_loss, test_acc = evaluate_model(
            model,
            x_test,
            y_test,
            batch_size=args.eval_batch_size,
        )

        elapsed = time.time() - epoch_start

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"lr {lr:.6f} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc * 100:.2f}% | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_acc * 100:.2f}% | "
            f"time {elapsed:.1f}s"
        )

        append_log(
            log_path=log_path,
            epoch=epoch,
            lr=lr,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            time_sec=elapsed,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(output_path)
            print(f"  [save] best model saved to {output_path}")

        lr *= args.lr_decay

    print()
    print("[done]")
    print(f"best test accuracy: {best_test_acc * 100:.2f}%")
    print(f"model saved to     : {output_path}")
    print(f"log saved to       : {log_path}")


def eval_command(args) -> None:
    data_dir = Path(args.data_dir)

    _, _, x_test, y_test = load_dataset(
        data_dir=data_dir,
        image_size=args.image_size,
        cache=not args.no_cache,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
    )

    model = MLP.load(Path(args.model))

    test_loss, test_acc = evaluate_model(
        model,
        x_test,
        y_test,
        batch_size=args.batch_size,
    )

    print("[eval]")
    print(f"model        : {args.model}")
    print(f"test samples : {len(y_test)}")
    print(f"test loss    : {test_loss:.4f}")
    print(f"test accuracy: {test_acc * 100:.2f}%")


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------


def preprocess_predict_image(
    image_path: Path,
    image_size: int = 28,
    invert: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one image for prediction.

    invert:
        no   : do not invert
        yes  : invert pixel values
        auto : if background looks white, invert automatically

    Returns:
        x: [1, 784]
        arr: [28, 28], processed image in [0, 1]
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    img = Image.open(image_path).convert("L")

    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0

    if invert == "yes":
        arr = 1.0 - arr
    elif invert == "auto":
        # MNIST normally has dark background and bright digit.
        # If mean is very bright, probably white background + black digit.
        if float(arr.mean()) > 0.5:
            arr = 1.0 - arr
    elif invert == "no":
        pass
    else:
        raise ValueError("invert must be one of: auto, yes, no")

    x = arr.reshape(1, -1).astype(np.float32)
    return x, arr


def image_to_ascii(arr: np.ndarray) -> str:
    """
    Print a 28x28 image in terminal.
    """
    chars = " .:-=+*#%@"
    arr = np.clip(arr, 0.0, 1.0)
    indices = (arr * (len(chars) - 1)).astype(np.int32)

    lines = []
    for row in indices:
        lines.append("".join(chars[i] for i in row))

    return "\n".join(lines)


def print_probabilities(probs: np.ndarray) -> None:
    for digit, prob in enumerate(probs):
        bar_len = int(prob * 30)
        bar = "#" * bar_len
        print(f"{digit}: {prob:.6f} {bar}")


def predict_command(args) -> None:
    model = MLP.load(Path(args.model))

    x, arr = preprocess_predict_image(
        image_path=Path(args.image),
        image_size=args.image_size,
        invert=args.invert,
    )

    probs = model.predict_proba(x, batch_size=1)[0]
    pred = int(np.argmax(probs))

    print("[predict]")
    print(f"model    : {args.model}")
    print(f"image    : {args.image}")
    print(f"predicted: {pred}")
    print()
    print("[probabilities]")
    print_probabilities(probs)
    print()
    print("[ascii image]")
    print(image_to_ascii(arr))


def predict_index_command(args) -> None:
    data_dir = Path(args.data_dir)

    _, _, x_test, y_test = load_dataset(
        data_dir=data_dir,
        image_size=args.image_size,
        cache=not args.no_cache,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
    )

    if args.index < 0 or args.index >= len(y_test):
        raise ValueError(f"index must be in [0, {len(y_test) - 1}], got {args.index}")

    model = MLP.load(Path(args.model))

    x = x_test[args.index : args.index + 1]
    label = int(y_test[args.index])

    probs = model.predict_proba(x, batch_size=1)[0]
    pred = int(np.argmax(probs))

    arr = x.reshape(args.image_size, args.image_size)

    print("[predict-index]")
    print(f"model     : {args.model}")
    print(f"test index: {args.index}")
    print(f"true label: {label}")
    print(f"predicted : {pred}")
    print()
    print("[probabilities]")
    print_probabilities(probs)
    print()
    print("[ascii image]")
    print(image_to_ascii(arr))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNIST PNG digit recognition with pure NumPy MLP."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train model.")
    p_train.add_argument("--data-dir", type=str, default="data/mnist_png")
    p_train.add_argument("--output", type=str, default="models/mnist_mlp_png.npz")
    p_train.add_argument("--log", type=str, default="logs/train_log.csv")
    p_train.add_argument("--image-size", type=int, default=28)
    p_train.add_argument("--hidden", type=int, default=128)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--eval-batch-size", type=int, default=512)
    p_train.add_argument("--lr", type=float, default=0.1)
    p_train.add_argument("--lr-decay", type=float, default=0.98)
    p_train.add_argument("--weight-decay", type=float, default=0.0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--max-train", type=int, default=None)
    p_train.add_argument("--max-test", type=int, default=None)
    p_train.add_argument("--no-cache", action="store_true")
    p_train.set_defaults(func=train_command)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate model.")
    p_eval.add_argument("--data-dir", type=str, default="data/mnist_png")
    p_eval.add_argument("--model", type=str, default="models/mnist_mlp_png.npz")
    p_eval.add_argument("--image-size", type=int, default=28)
    p_eval.add_argument("--batch-size", type=int, default=512)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--max-train", type=int, default=None)
    p_eval.add_argument("--max-test", type=int, default=None)
    p_eval.add_argument("--no-cache", action="store_true")
    p_eval.set_defaults(func=eval_command)

    # predict custom image
    p_predict = subparsers.add_parser("predict", help="Predict one image file.")
    p_predict.add_argument("--model", type=str, default="models/mnist_mlp_png.npz")
    p_predict.add_argument("--image", type=str, required=True)
    p_predict.add_argument("--image-size", type=int, default=28)
    p_predict.add_argument(
        "--invert",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help="Use yes for black digit on white background; no for MNIST-like images; auto by default.",
    )
    p_predict.set_defaults(func=predict_command)

    # predict by test index
    p_index = subparsers.add_parser("predict-index", help="Predict one test sample by index.")
    p_index.add_argument("--data-dir", type=str, default="data/mnist_png")
    p_index.add_argument("--model", type=str, default="models/mnist_mlp_png.npz")
    p_index.add_argument("--index", type=int, default=0)
    p_index.add_argument("--image-size", type=int, default=28)
    p_index.add_argument("--seed", type=int, default=42)
    p_index.add_argument("--max-train", type=int, default=None)
    p_index.add_argument("--max-test", type=int, default=None)
    p_index.add_argument("--no-cache", action="store_true")
    p_index.set_defaults(func=predict_index_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()