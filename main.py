import argparse
import logging
import os
import sys

import torch

from run import run
from utils import parse_layer_list, set_seed


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Task-routed FFN-MoE T5 for implicit sentiment analysis.",
    )

    # I/O and run mode
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # Model
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--num_experts", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--task_embedding_dim", type=int, default=64)
    parser.add_argument("--router_hidden_dim", type=int, default=1024)
    parser.add_argument("--encoder_moe_layers", type=str, default="8,10,12,14,16,18,20,22",
                        help="Comma-separated encoder block indices that receive MoE FFNs.")
    parser.add_argument("--decoder_moe_layers", type=str, default="8,10,12,14,16,18,20,22",
                        help="Comma-separated decoder block indices. Empty disables decoder MoE.")
    parser.add_argument("--lambda_sep", type=float, default=0.4,
                        help="Weight of the task-separation regularizer.")

    # Preprocessing / decoding
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--bf16", type=int, choices=[0, 1], default=1)

    parser.add_argument("--seed", type=int, default=21,
                        help="Random seed for Python, NumPy and PyTorch.")
    return parser


def configure_logging(output_dir, do_train):
    log_filename = "log.txt" if do_train else "eval_log.txt"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, log_filename)),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("trmoe_isa")


def main():
    args = build_arg_parser().parse_args()
    args.encoder_moe_layers = parse_layer_list(args.encoder_moe_layers)
    args.decoder_moe_layers = parse_layer_list(args.decoder_moe_layers)

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        print(
            "[WARN] Output directory {} already exists and is not empty; "
            "existing files may be overwritten.".format(args.output_dir),
            file=sys.stderr,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger = configure_logging(args.output_dir, args.do_train)
    logger.info("Arguments: %s", args)

    set_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()
    logger.info("Using %d GPU(s).", args.n_gpu)

    run(args, logger)


if __name__ == "__main__":
    main()
