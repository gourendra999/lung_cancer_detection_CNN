"""Train and export the LungAI four-class CT image CNN.

Expected preferred dataset structure:
    dataset_root/
      train/
        class_a/*.jpg
        class_b/*.jpg
        ...
      test/
        class_a/*.jpg
        class_b/*.jpg
        ...

If no separate test/ folder exists, the script creates stratified train,
validation, and test splits from the class-folder directory. It never uses the
same directory as both train and test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
RANDOM_STATE = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_image_manifest(folder: Path) -> pd.DataFrame:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")

    rows: list[dict[str, str]] = []
    class_dirs = sorted(path for path in folder.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least two class subfolders inside {folder}. "
            "Each class must have its own directory."
        )

    for class_dir in class_dirs:
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                rows.append({"filepath": str(image_path.resolve()), "label": class_dir.name})

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No supported images found inside {folder}.")
    return manifest


def split_manifest(
    train_manifest: pd.DataFrame,
    separate_test_manifest: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_names = sorted(train_manifest["label"].unique().tolist())
    if len(class_names) < 2:
        raise ValueError("The training directory must contain at least two classes.")

    counts = train_manifest["label"].value_counts()
    if (counts < 3).any():
        raise ValueError(
            "Each class needs at least three training images for stratified train/validation splitting. "
            f"Small class counts: {counts[counts < 3].to_dict()}"
        )

    if separate_test_manifest is not None:
        test_classes = set(separate_test_manifest["label"].unique())
        if test_classes != set(class_names):
            raise ValueError(
                "The train and test folders must contain the same class subfolders. "
                f"Train classes={sorted(class_names)}, test classes={sorted(test_classes)}"
            )
        if len(separate_test_manifest) == 0:
            raise ValueError("The separate test directory contains no images.")

        train_df, val_df = train_test_split(
            train_manifest,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=train_manifest["label"],
        )
        test_df = separate_test_manifest.copy()
        split_description = "80% train / 20% validation from train folder; separate test folder held out"
    else:
        # If there is no independent test directory, create three disjoint
        # stratified partitions from the one class-folder directory.
        train_pool, test_df = train_test_split(
            train_manifest,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=train_manifest["label"],
        )
        validation_fraction_of_pool = 0.15 / 0.85
        train_df, val_df = train_test_split(
            train_pool,
            test_size=validation_fraction_of_pool,
            random_state=RANDOM_STATE,
            stratify=train_pool["label"],
        )
        split_description = "70% train / 15% validation / 15% test, stratified from one class-folder directory"

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_generator(
    dataframe: pd.DataFrame,
    class_names: list[str],
    training: bool,
):
    if training:
        augmenter = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=7,
            width_shift_range=0.04,
            height_shift_range=0.04,
            zoom_range=0.08,
        )
    else:
        augmenter = ImageDataGenerator(rescale=1.0 / 255.0)

    return augmenter.flow_from_dataframe(
        dataframe=dataframe,
        x_col="filepath",
        y_col="label",
        classes=class_names,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=training,
        seed=RANDOM_STATE,
        interpolation="nearest",  # match the original app's Keras directory-loader default
        validate_filenames=True,
    )


def build_model(number_of_classes: int) -> tf.keras.Model:
    # GlobalAveragePooling2D replaces Flatten to greatly reduce dense-layer
    # parameters and help control overfitting on a small image dataset.
    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.45),
            layers.Dense(number_of_classes, activation="softmax"),
        ],
        name="lung_ct_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_onnx(model: tf.keras.Model, destination: Path) -> None:
    try:
        import onnx
        import tf2onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export dependencies are missing. Install requirements-training.txt, "
            "which pins a compatible NumPy/tf2onnx combination."
        ) from exc

    input_signature = (
        tf.TensorSpec(
            shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            dtype=tf.float32,
            name="images",
        ),
    )
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(destination))

    # Smoke-test the exported graph and its output dimensions.
    import onnxruntime as ort

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    smoke_input = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
    smoke_output = session.run(None, {input_name: smoke_input})[0]
    if np.asarray(smoke_output).shape != (1, model.output_shape[-1]):
        raise RuntimeError(f"Unexpected ONNX output shape: {np.asarray(smoke_output).shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LungAI CT image classifier.")
    parser.add_argument("--data-root", default=os.getenv("LUNG_DATASET_DIR", "dataset"))
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    dataset_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if train_dir.is_dir():
        train_manifest = collect_image_manifest(train_dir)
        separate_test_manifest = collect_image_manifest(test_dir) if test_dir.is_dir() else None
    else:
        # Fallback: dataset_root itself contains class subfolders.
        train_manifest = collect_image_manifest(dataset_root)
        separate_test_manifest = None

    if separate_test_manifest is not None and train_dir.resolve() == test_dir.resolve():
        raise ValueError("Training and test paths must be different directories.")

    train_df, val_df, test_df = split_manifest(train_manifest, separate_test_manifest)
    class_names = sorted(train_manifest["label"].unique().tolist())
    print("Split strategy: separate test directory" if separate_test_manifest is not None else "Split strategy: one directory, three disjoint stratified partitions")
    print(f"Training images: {len(train_df)} | Validation images: {len(val_df)} | Test images: {len(test_df)}")
    print("Training class counts:")
    print(train_df["label"].value_counts().sort_index())

    train_generator = make_generator(train_df, class_names, training=True)
    validation_generator = make_generator(val_df, class_names, training=False)
    test_generator = make_generator(test_df, class_names, training=False)

    class_ids = np.arange(len(class_names))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=train_generator.classes,
    )
    class_weight = {int(class_id): float(weight) for class_id, weight in zip(class_ids, weights)}
    print("Class weights:", class_weight)
    print("Class indices:", train_generator.class_indices)

    model = build_model(len(class_names))
    model.summary()

    best_model_path = output_dir / "best_lung_cancer.keras"
    callbacks = [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # EarlyStopping restores the best validation weights; save those weights.
    keras_path = output_dir / "lung_cancer_model.keras"
    model.save(keras_path)

    # Independent test evaluation (never used for training or model selection).
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    test_probabilities = model.predict(test_generator, verbose=1)
    y_true = test_generator.classes
    y_pred = np.argmax(test_probabilities, axis=1)
    display_names = [
        {
            "adenocarcinoma_left.lower.lobe_t2_n0_m0_ib": "Adenocarcinoma",
            "large.cell.carcinoma_left.hilum_t2_n2_m0_iiia": "Large Cell Carcinoma",
            "normal": "Normal",
            "squamous.cell.carcinoma_left.hilum_t1_n2_m0_iiia": "Squamous Cell Carcinoma",
        }.get(name.lower(), name)
        for name in class_names
    ]
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=display_names,
        zero_division=0,
    )
    print(f"Independent test loss: {test_loss:.4f}")
    print(f"Independent test accuracy: {test_accuracy:.4f}")
    print(report_text)

    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=class_ids),
        index=display_names,
        columns=display_names,
    ).to_csv(output_dir / "confusion_matrix.csv")

    # Save training curves for model review / project presentation.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history.get("accuracy", []), label="Train")
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history.get("loss", []), label="Train")
    axes[1].plot(history.history.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "class_names": class_names,
        "class_indices": {name: index for index, name in enumerate(class_names)},
        "image_size": list(IMAGE_SIZE),
        "color_mode": "RGB",
        "normalization": "divide_by_255",
        "interpolation": "nearest",
        "input_layout": "NHWC",
        "output_activation": "softmax",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "split": {
            "training_images": int(len(train_df)),
            "validation_images": int(len(val_df)),
            "test_images": int(len(test_df)),
            "random_state": RANDOM_STATE,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    onnx_path = output_dir / "lung_cancer_model.onnx"
    try:
        export_onnx(model, onnx_path)
        print(f"ONNX model exported and smoke-tested: {onnx_path}")
    except Exception as exc:
        print("\nONNX EXPORT FAILED. The Keras model and metadata were saved, but the Streamlit ONNX app cannot run until ONNX export succeeds.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print("Use Python 3.11 with requirements-training.txt (NumPy 1.26.4) and run the export again.")

    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
