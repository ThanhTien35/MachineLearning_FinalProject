
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import load_data
from fnn import FNN
from techniques import EarlyStopping


# Cau 1e - Cau hinh thuc nghiem
SEED = 42
HIDDEN_LAYERS = (64, 32)
LEARNING_RATE = 0.01
MAX_EPOCHS = 2000
PATIENCE = 50
MIN_DELTA = 1e-4

L2_CANDIDATES = (0.001, 0.01, 0.03, 0.1, 0.3)

COLORS = ('#64748b', '#2563eb', '#0d9488', '#9333ea')
LABELS = (
    'Baseline',
    'Early\nStopping',
    'L2',
    'Early Stopping\n+ L2'
)


def train_model(train, val, use_es=False, l2=0.0):
    model = FNN(
        (train[0].shape[1], *HIDDEN_LAYERS, 1),
        SEED
    )

    stopper = (
        EarlyStopping(PATIENCE, MIN_DELTA)
        if use_es else None
    )

    history = model.fit(
        train,
        val,
        LEARNING_RATE,
        MAX_EPOCHS,
        l2,
        stopper
    )

    selected = (
        stopper.best_epoch
        if stopper else len(history['train_loss'])
    )

    return model, history, selected


def select_lambda(train, val, output):
    # Chon lambda dua tren validation MSE tai epoch cuoi.
    rows = []

    for strength in L2_CANDIDATES:
        print(f'Khao sat lambda: {strength}', flush=True)

        _, history, _ = train_model(
            train, val, l2=strength
        )

        rows.append({
            'Lambda': strength,
            'Validation MSE': history['val_loss'][-1]
        })

    scores = pd.DataFrame(rows)
    scores.to_csv(
        output / 'lambda_search.csv',
        index=False
    )

    best = float(
        scores.loc[
            scores['Validation MSE'].idxmin(),
            'Lambda'
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 4),
        layout='constrained'
    )

    ax.semilogx(
        scores['Lambda'],
        scores['Validation MSE'],
        'o-',
        color=COLORS[2]
    )

    ax.scatter(
        best,
        scores['Validation MSE'].min(),
        color='#dc2626',
        zorder=3,
        label=f'Selected: {best:g}'
    )

    ax.set(
        xlabel='Lambda (log scale)',
        ylabel='Validation MSE (scaled)',
        title='L2 hyperparameter selection'
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(
        output / 'lambda_search.png',
        dpi=180
    )

    plt.close(fig)

    return best


def evaluate(model, test, scaler):
    X, y = test

    prediction = scaler.inverse_transform(
        model.predict(X)
    )

    mse = mean_squared_error(y, prediction)

    metrics = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y, prediction)
    }

    return metrics, prediction


def plot_losses(histories, results, output):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 8),
        sharey=True,
        layout='constrained'
    )

    for ax, (name, history) in zip(
        axes.flat, histories.items()
    ):
        epochs = range(
            1, len(history['train_loss']) + 1
        )

        ax.plot(
            epochs,
            history['train_loss'],
            label='Training MSE',
            color='#2563eb'
        )

        ax.plot(
            epochs,
            history['val_loss'],
            label='Validation MSE',
            color='#ea580c'
        )

        row = results.loc[name]

        # Danh dau epoch khoi phuc va epoch dung huan luyen.
        if name.startswith('Early Stopping'):
            ax.axvline(
                row['Selected epoch'],
                color='#0d9488',
                linestyle='--',
                label=f"Restored: {int(row['Selected epoch'])}"
            )

            ax.axvline(
                row['Epochs'],
                color='#64748b',
                linestyle=':',
                label=f"End: {int(row['Epochs'])}"
            )

        ax.set(
            title=name,
            xlabel='Epoch',
            ylabel='MSE (scaled target)'
        )

        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        'Training and validation learning curves'
    )

    fig.savefig(
        output / 'loss_curves.png',
        dpi=180
    )

    plt.close(fig)


def plot_metrics(results, output):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4.5),
        layout='constrained'
    )

    for ax, metric in zip(
        axes, ('MSE', 'RMSE', 'MAE')
    ):
        bars = ax.bar(
            LABELS,
            results[metric],
            color=COLORS
        )

        ax.bar_label(
            bars,
            fmt='%.2f',
            padding=3,
            fontsize=9
        )

        ax.set_ylim(
            0, results[metric].max() * 1.18
        )

        ax.set(
            title=f'Test {metric}',
            ylabel=metric
        )

        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=0.2)

    fig.savefig(
        output / 'metrics_comparison.png',
        dpi=180
    )

    plt.close(fig)


def plot_predictions(predictions, output):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(9, 8),
        sharex=True,
        sharey=True,
        layout='constrained'
    )

    low = predictions.min().min()
    high = predictions.max().max()

    for ax, name, color in zip(
        axes.flat,
        predictions.columns[1:],
        COLORS
    ):
        ax.scatter(
            predictions['Actual'],
            predictions[name],
            color=color,
            alpha=0.7,
            s=22
        )

        ax.plot(
            [low, high],
            [low, high],
            '--',
            color='#64748b',
            label='Ideal prediction'
        )

        ax.set(
            title=name,
            xlabel='Actual',
            ylabel='Predicted',
            xlim=(low - 5, high + 5),
            ylim=(low - 5, high + 5)
        )

        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.savefig(
        output / 'actual_vs_predicted.png',
        dpi=180
    )

    plt.close(fig)


def run_experiments():
    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=True)

    train, val, test, scaler = load_data(SEED)

    print(
        'Train/Validation/Test:',
        *(len(d[0]) for d in (train, val, test))
    )

    strength = select_lambda(train, val, output)

    print(f'Lambda duoc chon: {strength:g}')

    configurations = {
        'Baseline': (False, 0.0),
        'Early Stopping': (True, 0.0),
        'L2': (False, strength),
        'Early Stopping + L2': (True, strength)
    }

    rows = []
    histories = {}

    predictions = pd.DataFrame({
        'Actual': test[1].ravel()
    })

    # Chi danh gia test sau khi da chot cau hinh bang validation.
    for name, (use_es, l2) in configurations.items():
        print(f'Dang huan luyen: {name}', flush=True)

        model, history, selected = train_model(
            train, val, use_es, l2
        )

        metrics, prediction = evaluate(
            model, test, scaler
        )

        epochs = len(history['train_loss'])

        rows.append({
            'Model': name,
            **metrics,
            'Lambda': l2,
            'Epochs': epochs,
            'Selected epoch': selected,
            'Selected train MSE': history['train_loss'][selected - 1],
            'Selected val MSE': history['val_loss'][selected - 1]
        })

        histories[name] = history
        predictions[name] = prediction.ravel()

        filename = (
            name.lower()
            .replace(' ', '_')
            .replace('+', 'and')
            + '_history.csv'
        )

        pd.DataFrame(
            history,
            index=range(1, epochs + 1)
        ).to_csv(
            output / filename,
            index_label='epoch'
        )

    results = pd.DataFrame(rows).set_index('Model')

    results.to_csv(output / 'metrics.csv')

    predictions.to_csv(
        output / 'predictions.csv',
        index=False
    )

    plot_losses(histories, results, output)
    plot_metrics(results, output)
    plot_predictions(predictions, output)

    print(results.round(4).to_string())

    return results


if __name__ == '__main__':
    run_experiments()
