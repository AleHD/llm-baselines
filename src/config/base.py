import torch

import distributed


def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--acc_steps', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--iterations', default=15000, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--warmup_percent', default=0.02, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--scheduler', default='cos', choices=['linear', 'cos', 'none'])
    parser.add_argument('--opt', default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--eval_freq', default=200, type=int) # in iterations
    parser.add_argument('--results_base_folder', default="./exps", type=str) 
    # Dataset params
    parser.add_argument('--dataset', default='wikitext', choices=['wikitext', "shakespeare", 'arxiv', "arxiv2000"])
    parser.add_argument('--vocab_size', default=50304, type=int)
    parser.add_argument("--tokenizer", default="bpe", choices=["bpe", "character"])
    # Model params
    parser.add_argument('--model', default='base', choices=['base', 'sparse-heads-q', 'sparse-heads-qk', 'sparse-tokens-q', 'sparse-tokens-qk'])
    parser.add_argument('--use_pretrained', default="none", type=str) # 'none', 'gpt-2' or a path to the pretraind model
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--n_head', default=12, type=int)
    parser.add_argument('--n_layer', default=24, type=int) # depths in att + ff blocks
    parser.add_argument('--n_embd', default=768, type=int) # embedding size / hidden size ... 
    parser.add_argument('--sequence_length', default=512, type=int)
    parser.add_argument('--dtype', default=torch.bfloat16, type=torch.dtype)
    parser.add_argument('--bias', default=False, type=bool)
    parser.add_argument('--no_compile', action='store_true') # if true then model is not compiled 
    # logging params (WandB)
    parser.add_argument('--wandb', action='store_true') # whether to use wandb or not
    parser.add_argument('--wandb_project', default="my-project", type=str)
    parser.add_argument('--wandb_run_prefix', default="none", type=str) # is added before the autogenerated experiment name
    parser.add_argument('--eval_seq_prefix', default="The history of Switzerland ", type=str) # prefix used to generate sequences
    # Distributed args
    parser.add_argument('--distributed_backend', default=None, type=str, required=False,
                        choices=distributed.registered_backends())  # distributed backend type
    return parser.parse_args(args, namespace)
