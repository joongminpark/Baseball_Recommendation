import logging
import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm, trange
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from dataloader import DataSets
from model import BaseballTransformer
from utils import (
    set_seed,
    rotate_checkpoints,
    print_result,
    LabelproportionSampler,
    InferenceSampler,
    ResultWriter,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

try:
    from warmup_scheduler import GradualWarmupScheduler
except ImportError:
    raise ImportError(
        "Please install warmup_scheduler from https://github.com/ildoonet/pytorch-gradual-warmup-lr to use gradual warmup scheduler."
    )

logger = logging.getLogger(__name__)


def train(args, train_dataset, model):
    tb_writer = SummaryWriter(args.tb_writer_dir)
    result_writer = ResultWriter(args.eval_results_dir)

    if args.weighted_sampling == 1:
        # 세 가지 구질이 불균일하게 분포되었으므로 세 개를 동일한 비율로 샘플링
        # 결과적으로 이 방법을 썼을 때 좋지 않아서 wighted_sampling은 쓰지 않았음
        ball_type, counts = np.unique(train_dataset.pitch, return_counts=True)
        count_dict = dict(zip(ball_type, counts))
        weights = [1.0 / count_dict[p] for p in train_dataset.pitch]
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        logger.info("Do Weighted Sampling")
    else:
        sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs

    args.warmup_step = int(args.warmup_percent * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = [
        "bias",
        "layernorm.weight",
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.Adam(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    if args.warmup_step != 0:
        scheduler_cosine = CosineAnnealingLR(optimizer, t_total)
        scheduler = GradualWarmupScheduler(
            optimizer, 1, args.warmup_step, after_scheduler=scheduler_cosine
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    loss_fct = torch.nn.NLLLoss()

    # Train!
    logger.info("***** Running Baseball Transformer *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Warmup Steps = %d", args.warmup_step)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_step = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss, logging_val_loss = 0.0, 0.0, 0.0

    best_pitch_micro_f1, best_pitch_macro_f1, = 0, 0
    best_loss = 1e10
    best_pitch_macro_f1 = 0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch",)
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            (pitcher, batter, state, pitch, label, pitch_memory, label_memory, memory_mask,) = list(
                map(lambda x: x.to(args.device), batch)
            )
            model.train()
            pitching_score, memories = model(
                pitcher, batter, state, pitch_memory, label_memory, memory_mask,
            )

            pitching_score = pitching_score.log_softmax(dim=-1)
            loss = loss_fct(pitching_score, pitch)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.evaluate_during_training:
                    results, f1_results, f1_log, cm = evaluate(args, args.eval_data_file, model)
                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    print_result(output_eval_file, results, f1_log, cm)

                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    logging_val_loss = results["loss"]

                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                )
                logging_loss = tr_loss

                # best 모델 선정 지표를 loss말고 macro-f1으로 설정(trade-off 존재)
                # if best_loss > results["loss"]:
                if best_pitch_macro_f1 < results["pitch_macro_f1"]:
                    best_pitch_micro_f1 = results["pitch_micro_f1"]
                    best_pitch_macro_f1 = results["pitch_macro_f1"]
                    best_loss = results["loss"]
                    results["best_step"] = best_step = global_step

                    output_dir = os.path.join(args.output_dir, "best_model/")
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving best model to %s", output_dir)

                    result_path = os.path.join(output_dir, "best_results.txt")
                    print_result(result_path, results, f1_log, cm, off_logger=True)

                    results.update(dict(f1_results))
                    result_writer.update(args, **results)

                logger.info("  best pitch micro f1 : %s", best_pitch_micro_f1)
                logger.info("  best pitch macro f1 : %s", best_pitch_macro_f1)
                logger.info("  best loss : %s", best_loss)
                logger.info("  best step : %s", best_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = "checkpoint"
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
                )
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                rotate_checkpoints(args, checkpoint_prefix)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, dataset_path, model, prefix=""):
    eval_dataset = DataSets(dataset_path, args.memory_len)
    sampler = SequentialSampler(eval_dataset)
    logger.info(len(eval_dataset))
    eval_dataloader = DataLoader(
        eval_dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=4,
    )
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    loss_fct = torch.nn.NLLLoss(reduction="none")

    model.eval()
    total_pitch_preds = []
    total_pitch_labels = []
    total_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (pitcher, batter, state, pitch, label, pitch_memory, label_memory, memory_mask,) = list(
            map(lambda x: x.to(args.device), batch)
        )
        with torch.no_grad():
            pitching_score, memories = model(
                pitcher, batter, state, pitch_memory, label_memory, memory_mask
            )
            pitching_score = pitching_score.log_softmax(dim=-1)
            pitching_loss = loss_fct(pitching_score, pitch)

            # 부정 상황을 마스킹함으로써 긍정적인 상황만 평가
            # 부정 상황은 아래에서 정확도만 계산
            label_mask = (label == 1).to(torch.float)
            loss = (pitching_loss * label_mask).sum() / label_mask.sum()
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

        pitch_preds = torch.softmax(pitching_score, dim=-1).detach().argmax(axis=-1)
        pitch_labels = pitch.detach().cpu()

        total_pitch_preds.append(pitch_preds)
        total_pitch_labels.append(pitch_labels)
        total_labels.append(label)

    total_pitch_preds = torch.cat(total_pitch_preds).tolist()
    total_pitch_labels = torch.cat(total_pitch_labels).tolist()
    total_labels = torch.cat(total_labels).tolist()

    total_pitch_preds_good, total_pitch_preds_bad = [], []
    total_pitch_labels_good, total_pitch_labels_bad = [], []

    for i in range(len(total_labels)):
        if total_labels[i] == 1:
            total_pitch_preds_good.append(total_pitch_preds[i])
            total_pitch_labels_good.append(total_pitch_labels[i])
        else:
            total_pitch_preds_bad.append(total_pitch_preds[i])
            total_pitch_labels_bad.append(total_pitch_labels[i])

    eval_loss = eval_loss / nb_eval_steps
    result = {
        "pitch_macro_f1": f1_score(
            total_pitch_labels_good, total_pitch_preds_good, average="macro"
        ),
        "pitch_micro_f1": f1_score(
            total_pitch_labels_good, total_pitch_preds_good, average="micro"
        ),
        "loss": eval_loss,
    }

    labels = ["FASTs", "SLID", "CURVs"]
    label_list = list(range(len(labels)))

    # 구질 별 f1-score 계산
    dev_cr = classification_report(
        total_pitch_labels_good,
        total_pitch_preds_good,
        labels=label_list,
        target_names=labels,
        output_dict=True,
    )

    f1_results = [
        (l, r["f1-score"]) for i, (l, r) in enumerate(dev_cr.items()) if i < len(label_list)
    ]
    f1_log = ["{} : {}".format(l, f) for l, f in f1_results]
    cm = confusion_matrix(total_pitch_labels_good, total_pitch_preds_good, labels=label_list)

    # 부정 상황인 경우의 정확도(bad_acc) 계산
    bad_acc = sum([i == j for i, j in zip(total_pitch_labels_bad, total_pitch_preds_bad)]) / len(
        total_pitch_labels_bad
    )
    result["bad_acc"] = bad_acc

    return result, f1_results, f1_log, cm


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Model parameters
    parser.add_argument(
        "--n_pitcher_disc",
        type=int,
        required=True,
        help="The number of discrete features of pitcher",
    )
    parser.add_argument(
        "--n_pitcher_cont",
        type=int,
        required=True,
        help="The number of continuous features of pitcher",
    )
    parser.add_argument(
        "--n_batter_disc",
        type=int,
        required=True,
        help="The number of discrete features of batter",
    )
    parser.add_argument(
        "--n_batter_cont",
        type=int,
        required=True,
        help="The number of continuous features of batter",
    )
    parser.add_argument(
        "--n_memory_layer",
        type=int,
        required=True,
        help="The number of Transformer Encoders of memory",
    )
    parser.add_argument(
        "--n_encoder_layer",
        type=int,
        required=True,
        help="The number of Transformer Encoders of stats",
    )
    parser.add_argument(
        "--d_model", type=int, required=True, help="The dimension of self attention parameters",
    )
    parser.add_argument(
        "--n_head", type=int, required=True, help="The number of self attention head",
    )
    parser.add_argument(
        "--memory_len", default=50, type=int, required=True, help="Memory length in use",
    )
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, required=False, help="Dropout probability",
    )
    parser.add_argument(
        "--weight_decay", default=1e-5, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--attention_type", default="dot", type=str, help="Attention type of last layer."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_percent", default=0.1, type=float, help="Linear warmup over warmup_percent."
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--write_eval_results",
        action="store_true",
        help="Write evaluation report (for experiments)",
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default="./exp/results.csv",
        help="Directory for evaluation report result (for experiments)",
    )
    parser.add_argument(
        "--tb_writer_dir",
        type=str,
        default="./runs/",
        help="Directory for evaluation report result (for experiments)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--debug", action="store_true", help="set logging level DEBUG",
    )
    parser.add_argument(
        "--weighted_sampling", type=int, default=0, help="Do oversampling",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # Set seed
    set_seed(args)

    train_dataset = DataSets(args.train_data_file, args.memory_len, is_train=True)

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
    )
    model.to(args.device)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/pytorch_model.bin", recursive=True)
                )
            )
            logging.getLogger("evaluation").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model.load_state_dict(torch.load(checkpoint + "/pytorch_model.bin"))
            model.to(args.device)

            result = evaluate(args, args.eval_data_file, model, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result[0].items())

            results.update(result)


if __name__ == "__main__":
    main()
