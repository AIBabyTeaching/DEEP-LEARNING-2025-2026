from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Smart Assignment 2 · Deep Learning · W01-W03
        ## Guided Open Notebook

        This notebook is an **open guided assignment** based on:
        - `W01_intro_dl`
        - `W02_perceptron_mlp`
        - `W03_optimization_regularization`

        It is intentionally designed to feel similar in structure to the Smart Exam notebooks,
        but it is **not locked**:
        - No passcode
        - No registration number
        - No encrypted payloads
        - No hidden questions

        The goal is to help students **build the code step by step** while still solving the core tasks themselves.
        """
    ),
    md(
        """
        ---
        ## Assignment Rules

        - Total suggested score: **100 pts**
        - Work through the notebook in order.
        - Read the hints before coding.
        - Keep your answers and code in the provided cells.
        - If a sanity-check cell fails, fix the task above it before moving on.
        - Prefer clear, readable code over clever shortcuts.

        ## Covered Topics

        1. Gradients and MSE from Week 01
        2. Perceptron logic and MLP design from Week 02
        3. Optimizers, early stopping, dropout, and L2 regularization from Week 03
        """
    ),
    code(
        """
        # Basic setup and reproducibility
        from __future__ import annotations

        import random
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from IPython.display import display
        from sklearn.datasets import load_digits, make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        CLASS_NAMES = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        print("TensorFlow:", tf.__version__)
        print("NumPy:", np.__version__)
        print("Seed:", SEED)
        """
    ),
    md(
        """
        ## Notebook Roadmap

        **Section A. Foundations**
        - Task 1: manual regression with NumPy
        - Task 2: binary perceptron from scratch

        **Section B. MLPs**
        - Task 3: build a configurable MLP
        - Task 4: train a baseline classifier with early stopping

        **Section C. Optimization and Regularization**
        - Task 5: optimizer comparison
        - Task 6: regularization study

        **Section D. Analysis**
        - Task 7: inspect model mistakes and reflect on results
        """
    ),
    code(
        """
        # Helper utilities provided for the assignment.
        # Students should use these helpers and focus their effort on the TODO sections.

        def make_regression_line_data(n_samples: int = 120, noise: float = 12.0, random_state: int = SEED):
            x, y = make_regression(
                n_samples=n_samples,
                n_features=1,
                noise=noise,
                random_state=random_state,
            )
            x = x.reshape(-1)
            order = np.argsort(x)
            return x[order], y[order]


        def plot_regression_fit(x, y, w=None, b=None, title="Regression task"):
            plt.figure(figsize=(8, 4))
            plt.scatter(x, y, alpha=0.7, label="Data")
            if w is not None and b is not None:
                y_line = w * x + b
                plt.plot(x, y_line, color="crimson", linewidth=2.5, label="Model fit")
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()


        def load_binary_digits(classes=(0, 1), test_size=0.25, random_state: int = SEED):
            digits = load_digits()
            x = digits.data
            y = digits.target

            mask = np.isin(y, classes)
            x = x[mask]
            y = y[mask]
            y = (y == classes[1]).astype(np.int32)

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            return x_train, x_test, y_train, y_test


        def show_digit_samples(x, y, n=8):
            plt.figure(figsize=(12, 2))
            for i in range(min(n, len(x))):
                plt.subplot(1, min(n, len(x)), i + 1)
                plt.imshow(x[i].reshape(8, 8), cmap="gray")
                plt.title(f"y={y[i]}")
                plt.axis("off")
            plt.tight_layout()


        def load_fashion_subset(train_size=6000, val_size=2000, test_size=2000, random_state: int = SEED):
            (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.fashion_mnist.load_data()

            x_train_full = x_train_full.astype("float32") / 255.0
            x_test_full = x_test_full.astype("float32") / 255.0

            x_train, x_val, y_train, y_val = train_test_split(
                x_train_full,
                y_train_full,
                test_size=val_size,
                random_state=random_state,
                stratify=y_train_full,
            )

            if train_size < len(x_train):
                x_train, _, y_train, _ = train_test_split(
                    x_train,
                    y_train,
                    train_size=train_size,
                    random_state=random_state,
                    stratify=y_train,
                )

            if test_size < len(x_test_full):
                x_test, _, y_test, _ = train_test_split(
                    x_test_full,
                    y_test_full,
                    train_size=test_size,
                    random_state=random_state,
                    stratify=y_test_full,
                )
            else:
                x_test, y_test = x_test_full, y_test_full

            return x_train, x_val, x_test, y_train, y_val, y_test


        def plot_history_pair(history, metrics=("accuracy", "loss"), title_prefix=""):
            history_dict = history.history if hasattr(history, "history") else history
            fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
            if len(metrics) == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics):
                ax.plot(history_dict.get(metric, []), label=f"train_{metric}")
                ax.plot(history_dict.get(f"val_{metric}", []), label=f"val_{metric}")
                ax.set_title(f"{title_prefix}{metric}".strip())
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric)
                ax.grid(alpha=0.3)
                ax.legend()

            plt.tight_layout()


        def display_experiment_table(results):
            df = pd.DataFrame(results)
            if not df.empty:
                return df.sort_values(by="test_accuracy", ascending=False).reset_index(drop=True)
            return df


        def show_misclassified_examples(items, class_names, cols=3):
            if not items:
                print("No mistakes collected.")
                return

            rows = int(np.ceil(len(items) / cols))
            plt.figure(figsize=(4 * cols, 4 * rows))
            for i, item in enumerate(items, start=1):
                plt.subplot(rows, cols, i)
                plt.imshow(item["image"], cmap="gray")
                plt.title(
                    f'True: {class_names[item["true_label"]]}\\nPred: {class_names[item["pred_label"]]}'
                )
                plt.axis("off")
            plt.tight_layout()
        """
    ),
    md(
        """
        ---
        ## Section A · Foundations

        ### Task 1 `[10 pts]`
        Implement gradient-descent utilities for a simple line model:

        \\[
        \\hat{y} = w x + b
        \\]

        Your job:
        1. Compute MSE
        2. Compute gradients with respect to `w` and `b`
        3. Apply one gradient-descent update step

        **Hints**
        - Start from `y_pred = w * x + b`
        - Use vectorized NumPy operations
        - The loss should go down after repeated updates if your gradients are correct
        """
    ),
    code(
        """
        # Task 1: complete the three functions below.

        def compute_mse(y_true, y_pred):
            \"\"\"Return the mean squared error as a scalar float.\"\"\"
            # TODO:
            # 1. Compute residuals
            # 2. Square them
            # 3. Take the mean
            raise NotImplementedError("Task 1: implement compute_mse")


        def compute_gradients(x, y, w, b):
            \"\"\"Compute dL/dw and dL/db for MSE loss.\"\"\"
            # TODO:
            # Hint 1: y_pred = w * x + b
            # Hint 2: error = y_pred - y
            # Hint 3: dL/dw = mean(2 * error * x)
            # Hint 4: dL/db = mean(2 * error)
            raise NotImplementedError("Task 1: implement compute_gradients")


        def gradient_descent_step(x, y, w, b, lr):
            \"\"\"Update the parameters once and return new_w, new_b.\"\"\"
            # TODO:
            # 1. Call compute_gradients(...)
            # 2. Update w and b using gradient descent
            raise NotImplementedError("Task 1: implement gradient_descent_step")
        """
    ),
    code(
        """
        # Task 1 sanity check and mini experiment
        try:
            x_reg, y_reg = make_regression_line_data()
            w, b = 0.0, 0.0
            start_loss = compute_mse(y_reg, w * x_reg + b)

            for _ in range(80):
                w, b = gradient_descent_step(x_reg, y_reg, w, b, lr=0.05)

            end_loss = compute_mse(y_reg, w * x_reg + b)
            print("Start loss:", round(float(start_loss), 4))
            print("End loss:", round(float(end_loss), 4))
            assert end_loss < start_loss, "Loss should decrease after training."
            plot_regression_fit(x_reg, y_reg, w, b, title="Task 1 fit after gradient descent")
            print("Task 1 sanity check passed.")
        except NotImplementedError as exc:
            print(exc)
        except Exception as exc:
            print("Task 1 check failed:", exc)
        """
    ),
    md(
        """
        **Task 1 short answer**

        Replace this text with 2-4 lines:
        - Why does MSE get smaller when the gradients are correct?
        - What would happen if the learning rate is too large?
        """
    ),
    md(
        """
        ### Task 2 `[10 pts]`
        Train a **binary perceptron from scratch** on the scikit-learn digits dataset.

        Use only two classes: `0` and `1`.

        Your job:
        1. Implement the perceptron prediction rule
        2. Implement one training epoch
        3. Track accuracy across epochs

        **Hints**
        - Compute the score using `x_i @ w + b`
        - Predict class `1` when the score is non-negative, otherwise `0`
        - Only update parameters when the prediction is wrong
        """
    ),
    code(
        """
        # Task 2: perceptron utilities

        def perceptron_predict(x, w, b):
            \"\"\"Return binary predictions (0 or 1) for a batch or a single sample.\"\"\"
            # TODO:
            # 1. Compute the linear score
            # 2. Threshold at zero
            raise NotImplementedError("Task 2: implement perceptron_predict")


        def compute_binary_accuracy(y_true, y_pred):
            \"\"\"Return classification accuracy as a float.\"\"\"
            # TODO:
            raise NotImplementedError("Task 2: implement compute_binary_accuracy")


        def train_perceptron_epoch(x, y, w, b, lr=1.0):
            \"\"\"Train one epoch and return updated w, b, mistakes.\"\"\"
            # TODO:
            # Loop over training examples.
            # If pred != target:
            #     update = lr * (target - pred)
            #     w += update * x_i
            #     b += update
            raise NotImplementedError("Task 2: implement train_perceptron_epoch")
        """
    ),
    code(
        """
        # Task 2 training scaffold
        try:
            x_train_bin, x_test_bin, y_train_bin, y_test_bin = load_binary_digits(classes=(0, 1))
            show_digit_samples(x_train_bin, y_train_bin, n=8)

            w = np.zeros(x_train_bin.shape[1], dtype=np.float32)
            b = 0.0
            history = []

            for epoch in range(10):
                w, b, mistakes = train_perceptron_epoch(x_train_bin, y_train_bin, w, b, lr=1.0)
                train_pred = perceptron_predict(x_train_bin, w, b)
                test_pred = perceptron_predict(x_test_bin, w, b)
                train_acc = compute_binary_accuracy(y_train_bin, train_pred)
                test_acc = compute_binary_accuracy(y_test_bin, test_pred)
                history.append(
                    {
                        "epoch": epoch + 1,
                        "mistakes": mistakes,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                    }
                )

            perceptron_history = pd.DataFrame(history)
            display(perceptron_history)
            assert perceptron_history["test_acc"].max() >= 0.90, "Expected at least 90% test accuracy."
            print("Task 2 sanity check passed.")
        except NotImplementedError as exc:
            print(exc)
        except Exception as exc:
            print("Task 2 check failed:", exc)
        """
    ),
    md(
        """
        **Task 2 short answer**

        Replace this text with 2-4 lines:
        - Is the perceptron a good fit for all digit pairs? Why or why not?
        - What does it mean if the number of mistakes stops decreasing?
        """
    ),
    md(
        """
        ---
        ## Section B · MLPs

        ### Task 3 `[15 pts]`
        Build a **configurable multilayer perceptron** for Fashion-MNIST.

        Your job:
        1. Create hidden layers from a Python list such as `[128, 64]`
        2. Support a configurable activation function
        3. Optionally add dropout and L2 regularization
        4. Finish with a 10-class softmax output

        **Hints**
        - Start with `keras.Sequential()`
        - Add `layers.Input(shape=(28, 28))`
        - Use `layers.Flatten()` before dense layers
        - Use `regularizers.l2(...)` only when `l2_strength > 0`
        """
    ),
    code(
        """
        # Load the Fashion-MNIST subset once and reuse it in later tasks.
        x_train, x_val, x_test, y_train, y_val, y_test = load_fashion_subset()
        print("Train:", x_train.shape, y_train.shape)
        print("Val:  ", x_val.shape, y_val.shape)
        print("Test: ", x_test.shape, y_test.shape)
        """
    ),
    code(
        """
        # Task 3: build the configurable MLP

        def build_mlp(hidden_layers, activation="relu", dropout_rate=0.0, l2_strength=0.0):
            \"\"\"Create and return a Keras MLP classifier.\"\"\"
            # TODO:
            # 1. Create the Sequential model
            # 2. Add Input and Flatten
            # 3. Add one Dense layer per value in hidden_layers
            # 4. If dropout_rate > 0, add Dropout after each hidden layer
            # 5. Add the final Dense(10, activation="softmax") layer
            # 6. Apply L2 regularization to hidden layers when requested
            raise NotImplementedError("Task 3: implement build_mlp")
        """
    ),
    code(
        """
        # Task 3 sanity check
        try:
            candidate_model = build_mlp([128, 64], activation="relu", dropout_rate=0.2, l2_strength=1e-4)
            candidate_model.summary()

            dense_layers = [layer for layer in candidate_model.layers if isinstance(layer, layers.Dense)]
            assert len(dense_layers) == 3, "Expected 2 hidden Dense layers + 1 output Dense layer."
            assert dense_layers[-1].units == 10, "Output layer must have 10 units."
            print("Task 3 sanity check passed.")
        except NotImplementedError as exc:
            print(exc)
        except Exception as exc:
            print("Task 3 check failed:", exc)
        """
    ),
    md(
        """
        ### Task 4 `[15 pts]`
        Train a baseline MLP using **early stopping**.

        Your job:
        1. Compile the model
        2. Use sparse categorical crossentropy
        3. Monitor validation loss with early stopping
        4. Return the training history

        **Hints**
        - Use `metrics=["accuracy"]`
        - Use `keras.callbacks.EarlyStopping(...)`
        - Set `restore_best_weights=True`
        - Start with `optimizer="adam"` or `keras.optimizers.Adam(...)`
        """
    ),
    code(
        """
        # Task 4: baseline training function

        def train_baseline_model(
            model,
            train_x,
            train_y,
            val_x,
            val_y,
            optimizer,
            epochs=12,
            batch_size=128,
            verbose=0,
        ):
            \"\"\"Compile and fit the model, then return the History object.\"\"\"
            # TODO:
            # 1. Compile the model
            # 2. Create an EarlyStopping callback
            # 3. Fit using validation_data=(val_x, val_y)
            raise NotImplementedError("Task 4: implement train_baseline_model")
        """
    ),
    code(
        """
        # Task 4 experiment scaffold
        try:
            baseline_model = build_mlp([128, 64], activation="relu", dropout_rate=0.0, l2_strength=0.0)
            baseline_history = train_baseline_model(
                baseline_model,
                x_train,
                y_train,
                x_val,
                y_val,
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                epochs=12,
                batch_size=128,
                verbose=0,
            )

            test_loss, test_acc = baseline_model.evaluate(x_test, y_test, verbose=0)
            print("Baseline test accuracy:", round(float(test_acc), 4))
            plot_history_pair(baseline_history, metrics=("accuracy", "loss"), title_prefix="Baseline ")
        except NotImplementedError as exc:
            print(exc)
        except Exception as exc:
            print("Task 4 run failed:", exc)
        """
    ),
    md(
        """
        **Task 4 short answer**

        Replace this text with 3-5 lines:
        - Did training and validation curves stay close together?
        - Did early stopping likely help in your run?
        - What signs would indicate underfitting?
        """
    ),
    md(
        """
        ---
        ## Section C · Optimization and Regularization

        ### Task 5 `[20 pts]`
        Compare multiple optimizers on the **same architecture**.

        Your job:
        1. Create the required optimizer from its name
        2. Train one fresh model per experiment
        3. Store the validation and test metrics in a results table
        4. Compare convergence and final accuracy

        Suggested optimizers:
        - `sgd`
        - `momentum`
        - `adam`

        **Hints**
        - Use the same hidden layers for every run
        - Rebuild the model each time so weights start fresh
        - Keep the learning rate explicit in the table
        """
    ),
    code(
        """
        # Task 5: optimizer factory

        def make_optimizer(name, learning_rate):
            \"\"\"Return a Keras optimizer object from a short string name.\"\"\"
            # TODO:
            # if name == "sgd": ...
            # if name == "momentum": use SGD with momentum=0.9
            # if name == "adam": ...
            raise NotImplementedError("Task 5: implement make_optimizer")


        def run_single_experiment(config):
            model = build_mlp(
                hidden_layers=config["hidden_layers"],
                activation=config["activation"],
                dropout_rate=config.get("dropout_rate", 0.0),
                l2_strength=config.get("l2_strength", 0.0),
            )
            optimizer = make_optimizer(config["optimizer"], config["learning_rate"])
            history = train_baseline_model(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                optimizer=optimizer,
                epochs=config.get("epochs", 12),
                batch_size=config.get("batch_size", 128),
                verbose=0,
            )
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            best_val_acc = float(np.max(history.history["val_accuracy"]))
            return {
                "optimizer": config["optimizer"],
                "learning_rate": config["learning_rate"],
                "hidden_layers": str(config["hidden_layers"]),
                "best_val_accuracy": round(best_val_acc, 4),
                "test_accuracy": round(float(test_acc), 4),
                "epochs_ran": len(history.history["loss"]),
            }, history
        """
    ),
    code(
        """
        # Task 5 sweep scaffold
        experiment_grid = [
            {"optimizer": "sgd", "learning_rate": 1e-2, "hidden_layers": [128, 64], "activation": "relu"},
            {"optimizer": "momentum", "learning_rate": 1e-2, "hidden_layers": [128, 64], "activation": "relu"},
            {"optimizer": "adam", "learning_rate": 1e-3, "hidden_layers": [128, 64], "activation": "relu"},
        ]

        optimizer_results = []
        optimizer_histories = {}

        try:
            for config in experiment_grid:
                result, history = run_single_experiment(config)
                key = f'{config["optimizer"]}_{config["learning_rate"]}'
                optimizer_results.append(result)
                optimizer_histories[key] = history.history

            optimizer_results_df = display_experiment_table(optimizer_results)
            display(optimizer_results_df)

            for name, history_dict in optimizer_histories.items():
                plot_history_pair(history_dict, metrics=("accuracy",), title_prefix=f"{name} ")
        except NotImplementedError as exc:
            print(exc)
        except Exception as exc:
            print("Task 5 run failed:", exc)
        """
    ),
    md(
        """
        **Task 5 short answer**

        Replace this text with 4-6 lines:
        - Which optimizer converged fastest?
        - Which optimizer gave the best validation/test accuracy?
        - Was the fastest optimizer also the most stable one?
        """
    ),
    md(
        """
        ### Task 6 `[15 pts]`
        Study how **dropout** and **L2 regularization** affect generalization.

        Your job:
        1. Reuse `build_mlp(...)`
        2. Train four model variants
        3. Measure the train-validation gap
        4. Decide which configuration generalizes best

        Suggested configurations:
        - baseline
        - dropout only
        - L2 only
        - dropout + L2

        **Hints**
        - A large train/validation gap can be a sign of overfitting
        - Do not compare models with different hidden-layer sizes in this task
        - Use Adam for all configurations to keep the comparison fair
        """
    ),
    code(
        """
        # Task 6: regularization study scaffold

        def generalization_gap(history):
            \"\"\"Return final training accuracy minus final validation accuracy.\"\"\"
            # TODO:
            # 1. Read the last training accuracy
            # 2. Read the last validation accuracy
            # 3. Return the difference
            raise NotImplementedError("Task 6: implement generalization_gap")


        regularization_grid = [
            {"name": "baseline", "dropout_rate": 0.0, "l2_strength": 0.0},
            {"name": "dropout_only", "dropout_rate": 0.3, "l2_strength": 0.0},
            {"name": "l2_only", "dropout_rate": 0.0, "l2_strength": 1e-4},
            {"name": "dropout_l2", "dropout_rate": 0.3, "l2_strength": 1e-4},
        ]

        regularization_results = []

        try:
            for config in regularization_grid:
                model = build_mlp(
                    [128, 64],
                    activation="relu",
                    dropout_rate=config["dropout_rate"],
                    l2_strength=config["l2_strength"],
                )
                history = train_baseline_model(
                    model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    epochs=12,
                    batch_size=128,
                    verbose=0,
                )
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                gap = generalization_gap(history)
                regularization_results.append(
                    {
                        "name": config["name"],
                        "dropout_rate": config["dropout_rate"],
                        "l2_strength": config["l2_strength"],
                        "gap": round(float(gap), 4),
                        "best_val_accuracy": round(float(np.max(history.history["val_accuracy"])), 4),
                        "test_accuracy": round(float(test_acc), 4),
                    }
                )

            regularization_results_df = display_experiment_table(regularization_results)
            display(regularization_results_df)
        except NotImplementedError as exc:
            print(exc)
        except Exception as exc:
            print("Task 6 run failed:", exc)
        """
    ),
    md(
        """
        **Task 6 short answer**

        Replace this text with 4-6 lines:
        - Which regularization setup gave the best test accuracy?
        - Which setup produced the smallest generalization gap?
        - Did the best-generalizing model also achieve the best raw training accuracy?
        """
    ),
    md(
        """
        ---
        ## Section D · Analysis

        ### Task 7 `[15 pts]`
        Perform a small **error analysis** on your best model.

        Your job:
        1. Collect a few misclassified test examples
        2. Show the true and predicted class names
        3. Write a short reflection on common failure patterns

        **Hints**
        - Use `model.predict(...)`
        - Apply `argmax(axis=1)` to get class predictions
        - Save a few mistakes in a list of dictionaries
        """
    ),
    code(
        """
        # Task 7: collect mistakes from a trained Fashion-MNIST model

        def collect_misclassified_examples(model, x_test, y_test, max_items=9):
            \"\"\"Return a list of misclassified examples for visualization.\"\"\"
            # TODO:
            # 1. Compute predicted probabilities
            # 2. Convert them to class indices
            # 3. Find indices where prediction != truth
            # 4. Build a list of dictionaries with image, true_label, pred_label
            raise NotImplementedError("Task 7: implement collect_misclassified_examples")
        """
    ),
    code(
        """
        # Task 7 visualization scaffold
        try:
            # You may replace baseline_model with your best model from Task 5 or Task 6.
            best_model_for_analysis = baseline_model
            mistakes = collect_misclassified_examples(best_model_for_analysis, x_test, y_test, max_items=9)
            print("Collected mistakes:", len(mistakes))
            show_misclassified_examples(mistakes, CLASS_NAMES, cols=3)
        except NameError:
            print("Train at least one model first, then rerun this cell.")
        except NotImplementedError as exc:
            print(exc)
        except Exception as exc:
            print("Task 7 run failed:", exc)
        """
    ),
    md(
        """
        **Task 7 final reflection**

        Replace this text with 5-8 lines:
        - Which classes were most often confused?
        - Are the mistakes visually understandable?
        - If you had one more hour, what would you improve first: data, architecture, optimizer, or regularization?
        - Connect your answer to something learned in W01, W02, or W03.
        """
    ),
    md(
        """
        ---
        ## Bonus Ideas

        If you want to extend the assignment:
        - Try a different activation such as `tanh` or `elu`
        - Compare `[64]`, `[128, 64]`, and `[256, 128, 64]`
        - Add a small learning-rate sweep for Adam
        - Compare Fashion-MNIST with the digits dataset

        ## Submission Checklist

        - All TODO functions completed
        - Sanity-check cells run successfully
        - Tables and plots generated
        - Written reflections added
        - Notebook saved with outputs
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "DL 2025",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path("SmartAssignment2_W01_W03_Guided.ipynb")
output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {output_path}")
