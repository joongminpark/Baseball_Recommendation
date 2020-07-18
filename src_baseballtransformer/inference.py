import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from dataloader import DataSets
from train import evaluate
from model import BaseballTransformer

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import pickle
import utils
import logging
import argparse
import random
import os
import warnings
import itertools

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ball2idx = {
    "FAST": 0,
    "TWOS": 0,
    "SINK": 0,
    "CUTT": 0,
    "KNUC": 0,
    "SLID": 1,
    "CURV": 2,
    "CHUP": 2,
    "FORK": 2,
}


def draw_cm(cm, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fontprop = fm.FontProperties(size=15)

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    balls = ["fast", "horizontal", "vertical"]
    tick_marks = np.arange(len(balls))

    ax = plt.gca()
    plt.xticks(tick_marks)
    ax.set_xticklabels(balls, fontproperties=fontprop)
    plt.yticks(tick_marks)
    ax.set_yticklabels(balls, fontproperties=fontprop)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )
    plt.tight_layout()
    plt.ylabel("True", fontsize=12)
    plt.xlabel("Predict", fontsize=12)

    plt_dir = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(plt_dir, dpi=300, bbox_inches="tight")

    logger.info("  Confusion Matrix are saved to, %s", plt_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./data/binary_memory_50/binary_data_memory_test.pkl",
        type=str,
        help="test data path",
    )
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    parser.add_argument("--args_path", required=True, type=str, help="args path")
    args_ = parser.parse_args()
    args = torch.load(args_.args_path)
    args.eval_data_file = args_.data_path
    args.eval_output_dir = os.path.join(args.output_dir, "test_result/")
    os.makedirs(args.eval_output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    n_pitcher = [*[2] * args.n_pitcher_disc, *[5] * args.n_pitcher_cont]
    n_batter = [*[2] * args.n_batter_disc, *[5] * args.n_batter_cont]
    n_state = [4, 4, 5, 8, 9, 10, 16, 22, 23]

    model = BaseballTransformer(
        n_pitcher,
        n_batter,
        n_state,
        args.n_memory_layer,
        args.n_encoder_layer,
        memory_len=args.memory_len,
        d_model=args.d_model,
        n_head=args.n_head,
        dropout=args.dropout,
        attention_type=args.attention_type,
    ).to(args.device)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args_.model_path), strict=False)

    model.eval()
    result, _, f1_log, cm = evaluate(args, args.eval_data_file, model)

    result_dir = os.path.join(args.eval_output_dir, "test_results.txt")
    utils.print_result(result_dir, result, f1_log, cm)
    logger.info("  Results are saved to, %s", result_dir)
    draw_cm(cm, args.eval_output_dir)


if __name__ == "__main__":
    main()
