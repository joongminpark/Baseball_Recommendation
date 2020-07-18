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
from model import BaseballTransformer, BaseballTransformerOnlyPitcher, BaseballTransformerPitcherSentiment
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
    tb_writer = SummaryWriter()
    result_writer = ResultWriter(args.eval_results_dir)

    # Sampling 빈도 설정
    if args.sample_criteria is None:
        # 긍/부정 상황 & 투구구질 비율대로 Random sampling
        sampler = RandomSampler(train_dataset)
    else:
        # <투구구질> 기준(7 가지)으로 sampling 빈도 동등하게 조절
        if args.sample_criteria == "pitcher":
            counts = train_dataset.pitch_counts
            logger.info("  Counts of each ball type : %s", counts)
            pitch_contiguous = [i for p in train_dataset.origin_pitch for i, j in enumerate(p) if j == 1]
            weights = [0 if p == 5 or p == 6 else 1.0 / counts[p] for p in pitch_contiguous]
            sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        # <긍,부정> 기준(2 가지)으로 sampling 빈도 동등하게 조절
        elif args.sample_criteria == "batter":
            counts = train_dataset.label_counts
            logger.info("  Counts of each label type : %s", counts)
            weights = [1.0 / counts[l] for l in train_dataset.label]
            sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        # <투구구질 & 긍,부정> 기준(14 가지)으로 sampling 빈도 동등하게 조절
        elif args.sample_criteria == "both":
            counts = train_dataset.pitch_and_label_count
            logger.info("  Counts of each both type : %s", counts)
            pitch_contiguous = [i for p in train_dataset.origin_pitch for i, j in enumerate(p) if j == 1]
            weights = [
                0 if p == 5 or p == 6 else 1.0 / counts[(p, l)]
                for p, l in zip(pitch_contiguous, train_dataset.label)
            ]
            sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        else:
            sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler,)
    t_total = len(train_dataloader) * args.num_train_epochs
    args.warmup_step = int(args.warmup_percent * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = [
        "bias",
        "layernorm.weight",
    ]  # LayerNorm.weight -> layernorm.weight (model_parameter name)
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
        # scheduler_plateau = ReduceLROnPlateau(optimizer, "min")
        # scheduler = GradualWarmupScheduler(
        #     optimizer, 1, args.warmup_step, after_scheduler=scheduler_plateau
        # )
    else:
        scheduler = CosineAnnealingLR(optimizer, t_total)
        # scheduler = ReduceLROnPlateau(optimizer, "min")

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()

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
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss, logging_val_loss = 0.0, 0.0, 0.0

    best_pitch_micro_f1, best_pitch_macro_f1, = 0, 0
    best_loss = 1e10

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch",)
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            (
                pitcher_discrete,
                pitcher_continuous,
                batter_discrete,
                batter_continuous,
                state_discrete,
                state_continuous,
                pitch,
                hit,
                label,
                masked_pitch,
                origin_pitch,
            ) = list(map(lambda x: x.to(args.device), batch))
            model.train()
            
            # sentiment input
            pitching_score = model(
                pitcher_discrete,
                pitcher_continuous,
                batter_discrete,
                batter_continuous,
                state_discrete,
                state_continuous,
                label,
                args.concat if args.concat else 0,
            )

            pitching_score = pitching_score.contiguous()
            pitch = pitch.contiguous()
            # with sigmoid(m)
            loss = loss_fct(m(pitching_score), pitch)

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
                # Log metrics
                if args.evaluate_during_training:
                    results, f1_results, f1_log, cm_pos, cm_neg = evaluate(args, args.eval_data_file, model)
                    output_eval_file = os.path.join(args.output_dir, "eval_results_pos.txt")
                    print_result(output_eval_file, results, f1_log, cm_pos)

                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    logging_val_loss = results["loss"]
                    # scheduler.step(logging_val_loss)

                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                )
                logging_loss = tr_loss
                if best_loss > results["loss"]:
                    best_pitch_micro_f1 = results["pitch_micro_f1"]
                    best_pitch_macro_f1 = results["pitch_macro_f1"]
                    best_loss = results["loss"]

                    output_dir = os.path.join(args.output_dir, "best_model/")
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving best model to %s", output_dir)

                    result_path = os.path.join(output_dir, "best_results.txt")
                    print_result(result_path, results, f1_log, cm_pos, off_logger=True)

                    results.update(dict(f1_results))
                    result_writer.update(args, **results)

                logger.info("  best pitch micro f1 : %s", best_pitch_micro_f1)
                logger.info("  best pitch macro f1 : %s", best_pitch_macro_f1)
                logger.info("  best loss : %s", best_loss)

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


def evaluate(args, dataset_path, model, prefix="", label_switch=False):
    eval_dataset = DataSets(dataset_path, args.reverse, args.binary, False, False, False, False, label_switch)
    # evaluate는 data 비율 그대로 따라가게 sampling
    sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=4,
    )
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()

    model.eval()
    total_pitch_preds_good, total_pitch_preds_bad = [], []
    total_pitch_labels_good, total_pitch_labels_bad = [], []
    total_hit_preds = []
    total_hit_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (
            pitcher_discrete,
            pitcher_continuous,
            batter_discrete,
            batter_continuous,
            state_discrete,
            state_continuous,
            pitch,
            hit,
            label,
            masked_pitch,
            origin_pitch,
        ) = list(map(lambda x: x.to(args.device), batch))
        
        with torch.no_grad():
            pitching_score = model(
                pitcher_discrete,
                pitcher_continuous,
                batter_discrete,
                batter_continuous,
                state_discrete,
                state_continuous,
                label,
                args.concat if args.concat else 0,
            )
            # 긍정 상황만 loss에 계산
            pitching_score_pos = pitching_score[label==1].contiguous()
            pitch_pos = pitch[label==1].contiguous()
            pitching_loss = loss_fct(m(pitching_score_pos), pitch_pos)
            loss = pitching_loss

            eval_loss += loss.mean().item()
        nb_eval_steps += 1

        # true label 중 가장 높은 값만 사용
        # positive situation
        pitch_preds_pos = pitching_score[label==1].detach().cpu()
        pitch_labels_pos = pitch[label==1].detach().cpu()
        pitch_preds_pos = torch.tensor(np.where(masked_pitch[label==1].detach().cpu(), pitch_preds_pos, -1000))

        total_pitch_preds_good.append(pitch_preds_pos.argmax(axis=-1))
        total_pitch_labels_good.append(pitch_labels_pos.argmax(axis=-1))

        # negative situation
        pitch_preds_neg = pitching_score[label==0].detach().cpu()
        pitch_labels_pos = pitch[label==0].detach().cpu()
        pitch_preds_neg = torch.tensor(np.where(masked_pitch[label==0].detach().cpu(), pitch_preds_neg, -1000))

        total_pitch_preds_bad.append(pitch_preds_neg.argmax(axis=-1))
        total_pitch_labels_bad.append(pitch_labels_pos.argmax(axis=-1))

    total_pitch_preds_good = torch.cat(total_pitch_preds_good).tolist()
    total_pitch_labels_good = torch.cat(total_pitch_labels_good).tolist()
    total_pitch_preds_bad = torch.cat(total_pitch_preds_bad).tolist()
    total_pitch_labels_bad = torch.cat(total_pitch_labels_bad).tolist()

    eval_loss = eval_loss / nb_eval_steps
    result = {
        "pitch_macro_f1": f1_score(total_pitch_labels_good, total_pitch_preds_good, average="macro"),
        "pitch_micro_f1": f1_score(total_pitch_labels_good, total_pitch_preds_good, average="micro"),
        "loss": eval_loss,
    }

    label_list = list(range(len(eval_dataset.ball2idx)))

    # 구질 별 f1-score 계산
    dev_cr = classification_report(
        total_pitch_labels_good,
        total_pitch_preds_good,
        labels=label_list,
        target_names=list(eval_dataset.ball2idx.keys()),
        output_dict=True,
    )

    f1_results = [
        (l, r["f1-score"]) for i, (l, r) in enumerate(dev_cr.items()) if i < len(label_list)
    ]
    f1_log = ["{} : {}".format(l, f) for l, f in f1_results]
    
    # 긍,부정 각각의 경우 Confusion matrix 추출 
    cm_pos = confusion_matrix(total_pitch_labels_good, total_pitch_preds_good, labels=label_list)
    cm_neg = confusion_matrix(total_pitch_labels_bad, total_pitch_preds_bad, labels=label_list)

    return result, f1_results, f1_log, cm_pos, cm_neg


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
        "--n_encoder_layer", type=int, required=True, help="The number of Transformer Encoders",
    )
    parser.add_argument(
        "--n_decoder_layer", type=int, required=True, help="The number of Multimodal Transformers",
    )
    parser.add_argument(
        "--n_concat_layer", type=int, required=True, help="The number of concat layers",
    )
    parser.add_argument(
        "--grouping", action="store_true", help="Whether stat embedding is grouping or not",
    )
    parser.add_argument(
        "--d_model", type=int, required=True, help="The dimension of self attention parameters",
    )
    parser.add_argument(
        "--nhead", type=int, required=True, help="The number of self attention head",
    )
    parser.add_argument(
        "--dim_feedforward", type=int, required=True, help="The dimension of feedforward network",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, required=False, help="Dropout probability",
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Reverse hit label unlee batter did well",
    )
    parser.add_argument(
        "--binary", action="store_true", help="Whether sentimental label is binary or multi ",
    )
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_transfer", action="store_true", help="Whether to run transfer_training.")
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
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
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
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
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
        "--mode", type=str, default=None, help="both, pitcher, batter",
    )
    parser.add_argument(
        "--mask", action="store_true", help="whether to mask final score",
    )
    parser.add_argument(
        "--concat", action="store_true", help="whether to use concat_layer",
    )
    parser.add_argument(
        "--sample_criteria",
        type=str,
        default=None,
        help="Criteria of sampling(pitcher, batter, both)",
    )
    parser.add_argument(
        "--is_sentiment_input",
        action="store_true",
        help="Whether to use pos,neg information as input",
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # Set seed
    set_seed(args)

    ## Transfer learning
    if args.do_transfer:
        train_pos_dataset = DataSets(args.train_data_file, args.reverse, args.binary, is_train=True, 
        negative_label_change=False, is_transfer=args.do_transfer, transfer_positive_dataset=True)
        train_neg_dataset = DataSets(args.train_data_file, args.reverse, args.binary, is_train=True,
        negative_label_change=False, is_transfer=args.do_transfer, transfer_positive_dataset=False)
        # subtract two index variables and three categorical variables
        n_pitcher_cont = len(train_pos_dataset.pitcher[0]) - 5
        # subtract two index variables and two categorical variables
        n_batter_cont = len(train_pos_dataset.batter[0]) - 4
        # subtract one index variables and three categorical variables
        n_state_cont = len(train_pos_dataset.state[0]) - 3
    
    # Not Transfer learning (긍, 부정 통합)
    else:
        train_dataset = DataSets(args.train_data_file, args.reverse, args.binary, is_train=True)
        # subtract two index variables and three categorical variables
        n_pitcher_cont = len(train_dataset.pitcher[0]) - 5
        # subtract two index variables and two categorical variables
        n_batter_cont = len(train_dataset.batter[0]) - 4
        # subtract one index variables and three categorical variables
        n_state_cont = len(train_dataset.state[0]) - 3

    # loading model (pos, neg 여부가 Input으로 들어가는지에 따라 분류)
    if not args.is_sentiment_input:
        # 기존 input에 sentiment 없을 때
        model = BaseballTransformerOnlyPitcher(
            n_pitcher_cont,
            n_batter_cont,
            n_state_cont,
            n_encoder_layer=args.n_encoder_layer,
            n_decoder_layer=args.n_decoder_layer,
            n_concat_layer=args.n_concat_layer,
            d_model=args.d_model,
            do_grouping=args.grouping,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
    else:
        # input에 sentiment 있을 때
        model = BaseballTransformerPitcherSentiment(
            n_pitcher_cont,
            n_batter_cont,
            n_state_cont,
            n_encoder_layer=args.n_encoder_layer,
            n_decoder_layer=args.n_decoder_layer,
            n_concat_layer=args.n_concat_layer,
            d_model=args.d_model,
            do_grouping=args.grouping,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )

    model.to(args.device)

    # Training
    if args.do_train:
        if args.do_transfer:
            logger.info("Transfer learning 1: positive situation")
            global_step, tr_loss = train(args, train_pos_dataset, model)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            
            logger.info("Transfer learning 2: negative situation")
            global_step, tr_loss = train(args, train_neg_dataset, model)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        else:
            logger.info("Start Training(General situation)")
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
