import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from dataloader import DataSets
from train import evaluate
from model import BaseballTransformer, BaseballTransformerOnlyPitcher, BaseballTransformerPitcherSentiment

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
    "CHUP": 0,
    "CURV": 1,
    "CUTT": 2,
    "FAST": 3,
    "FORK": 4,
    "KNUC": 5,
    "SINK": 6,
    "SLID": 7,
    "TWOS": 8,
}


def draw_cm(cm, save_dir, prefix=""): # prefix: positive / negative
    fig, ax = plt.subplots(figsize=(10, 8))
    fontprop = fm.FontProperties(size=15)

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix of {}".format(prefix))
    plt.colorbar()
    tick_marks = np.arange(len(ball2idx))

    ax = plt.gca()
    plt.xticks(tick_marks)
    ax.set_xticklabels(list(ball2idx.keys()), fontproperties=fontprop)
    plt.yticks(tick_marks)
    ax.set_yticklabels(list(ball2idx.keys()), fontproperties=fontprop)

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

    plt_dir = os.path.join(save_dir, "confusion_matrix_{}.png".format(prefix))
    plt.savefig(plt_dir, dpi=300, bbox_inches="tight")

    logger.info("  Confusion Matrix are saved to, %s", plt_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="./data/binary_data_test.pkl", type=str, help="test data path"
    )
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    parser.add_argument("--args_path", required=True, type=str, help="args path")
    parser.add_argument(
        "--label_switch_inference", action="store_true", help="When inference, is label switched?",
    )
    args_ = parser.parse_args()
    args = torch.load(args_.args_path)
    args.test_data_path = args_.data_path
    args.eval_batch_size = 512
    args.eval_output_dir = os.path.join(args.output_dir, "test_result/")
    os.makedirs(args.eval_output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # dataset = DataSets(args.eval_data_file, args.reverse, args.binary)
    # datasets: Case 3 모델의 경우(긍/부정 Input으로 부여) 긍/부정 다르게 하기 (label_switch_inference)
    dataset = DataSets(args.eval_data_file, args.reverse, args.binary, False, False, False, False, \
        args_.label_switch_inference)

    # subtract two index variables and three categorical variables
    n_pitcher_cont = len(dataset.pitcher[0]) - 5
    # subtract two index variables and two categorical variables
    n_batter_cont = len(dataset.batter[0]) - 4
    # subtract one index variables and two categorical variables
    n_state_cont = len(dataset.state[0]) - 3

    if not args.is_sentiment_input:
        # 기존 input에 sentiment 없을 때
        model = BaseballTransformerOnlyPitcher(
            n_pitcher_cont,
            n_batter_cont,
            n_state_cont,
            n_encoder_layer=args.n_encoder_layer,
            n_decoder_layer=args.n_decoder_layer,
            n_concat_layer=args.n_concat_layer,
            do_grouping=args.grouping,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(args.device)
    else:
        # input에 sentiment 있을 때
        model = BaseballTransformerPitcherSentiment(
            n_pitcher_cont,
            n_batter_cont,
            n_state_cont,
            n_encoder_layer=args.n_encoder_layer,
            n_decoder_layer=args.n_decoder_layer,
            n_concat_layer=args.n_concat_layer,
            do_grouping=args.grouping,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(args.device)
    model.load_state_dict(torch.load(args_.model_path), strict=False)

    model.eval()

    # Inference 진행하여 Confusion matrix 추출(긍, 부정의 경우 각각 추출)
    result, _, f1_log, cm_pos, cm_neg = evaluate(args, args.eval_data_file, model, "", args_.label_switch_inference)

    result_dir = os.path.join(args.eval_output_dir, "test_results_pos.txt")
    utils.print_result(result_dir, result, f1_log, cm_pos)
    logger.info("  Results are saved to, %s", result_dir)
    
    # 긍, 부정의 경우 Confusion matrix 각각 저장
    draw_cm(cm_pos, args.eval_output_dir, prefix="positive")
    draw_cm(cm_neg, args.eval_output_dir, prefix="negative")


if __name__ == "__main__":
    main()
