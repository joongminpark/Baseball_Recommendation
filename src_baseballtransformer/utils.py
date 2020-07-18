import torch
import random
import shutil
import os
import re
import logging
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    # checkpoint 시간순서 정렬
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    # 오래된 checkpoint 제거
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint)
        )
        shutil.rmtree(checkpoint)


def print_result(file_path, result, f1_log, cm, off_logger=False):
    if off_logger:
        logger.disabled = True
    with open(file_path, "w") as writer:
        logger.info("***** Eval results *****")

        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

        logger.info(" ")
        writer.write("\n")

        logger.info("***** Class F1 score *****")
        for f1 in f1_log:
            logger.info("  %s", f1)
            writer.write("%s\n" % f1)

        logger.info(" ")
        writer.write("\n")

        logger.info("***** Confusion Matrix *****")
        logger.info("\n%s", cm)
        writer.write("%s" % cm)
    if off_logger:
        logger.disabled = False


class ResultWriter:
    def __init__(self, directory):
        """ Save training Summary to .csv 
        input
            args: training args
            results: training results (dict)
                - results should contain a key name 'val_loss'
        """
        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = "%s-%s-%s %s:%s" % (now.year, now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None


class LabelproportionSampler(Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = None):
        """
        label unbalanced(not evenly distributed) -> evenly distributed (1:1:...:1)
        # Args
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            batch_size : Batch size to train model
        """
        super(LabelproportionSampler, self).__init__(dataset)
        self.dataset = dataset
        if len(self.dataset) < 1:
            raise ValueError("The number of dateset must be > 1.")

        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _ in range(len(self.dataset) // self.batch_size):
            batch = []

            # hitter_label info -> binary label
            # pitcher_label info (given NC) -> 7 label (except 2 balls <KNUC, SINC> cuz it rarely exists)
            hitter_label_list = np.array(self.dataset.label)
            hitter_label_set = list(set(self.dataset.label))

            pitcher_label_list = np.array(self.dataset.pitch)
            pitcher_label_set = list(set(self.dataset.pitch))[:7]  # 0~6 sampling <7 : KNUC, SINC>

            # Get indexing in each label
            hitter_idx = []
            pitcher_idx = []
            for i in range(2):
                hitter_tmp_idx = np.where(hitter_label_list == hitter_label_set[i])[0]
                hitter_idx.append(hitter_tmp_idx)

            for i in range(7):
                pitcher_tmp_idx = np.where(pitcher_label_list == pitcher_label_set[i])[0]
                pitcher_idx.append(pitcher_tmp_idx)

            # Get random classes (hitter_proportion -> 1:1, pitcher_proportion -> 1:1:1:1:1:1:1)
            for i in range(2):
                for j in range(7):
                    # extract same indexing
                    hit_with_pitch_tmp_idx = list(set(hitter_idx[i]).intersection(pitcher_idx[j]))
                    sample_idx = list(
                        np.random.choice(
                            hit_with_pitch_tmp_idx, size=self.batch_size // 14, replace=True
                        )
                    )
                    batch.extend(sample_idx)

            random.shuffle(batch)

            yield np.stack(batch)


class InferenceSampler(Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = None):
        """
        Random sampler (given data distributed <conserved>)
        # Args
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            batch_size : Batch size to train model
        """
        super(InferenceSampler, self).__init__(dataset)
        self.dataset = dataset
        if len(self.dataset) < 1:
            raise ValueError("The number of dateset must be > 1.")

        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _ in range(len(self.dataset) // self.batch_size):
            batch = []

            # 0~6 sampling (7: KNUC, SINC -> excluded)
            extracted_label = list(set(self.dataset.pitch))[:7]
            label_list = np.array(self.dataset.pitch)

            # Get indexing in each label
            label_idx = []
            for i in range(7):
                tmp_idx = list(np.where(label_list == extracted_label[i])[0])
                label_idx.extend(tmp_idx)

            # Get random classes (given data distributed)
            sample_idx = list(np.random.choice(label_idx, size=self.batch_size, replace=False))
            batch.extend(sample_idx)

            yield np.stack(batch)
