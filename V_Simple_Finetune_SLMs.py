# coding=utf-8
# Simplified from RevisitVD/finetune/Finetune_SLMs.py
# Supports: codebert | unixcoder | pdbert  (selected via --model_type)

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import re
import json
import random
import numpy as np

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    get_linear_schedule_with_warmup,
    RobertaConfig, RobertaModel, RobertaTokenizer,
)
from model import Model

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ── Supported models ───────────────────────────────────────────────────────────
#
#   --model_type codebert   → microsoft/codebert-base
#   --model_type unixcoder  → microsoft/unixcoder-base
#   --model_type pdbert     → path/to/local/pdbert  (not on HuggingFace Hub)
#
# All three share the same RoBERTa architecture and tokenizer family, so the
# same loading code works for all. The only difference between them is how the
# input tokens are formatted before being fed to the model (see convert_to_features).
#
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ('codebert', 'unixcoder', 'pdbert')


# ── Data ──────────────────────────────────────────────────────────────────────

class InputFeatures:
    def __init__(self, input_ids, idx, label):
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def clean_code(func, model_type):
    """
    Normalize whitespace in code before tokenization.

    - codebert : collapse ALL whitespace (spaces, tabs, newlines) into single spaces.
                 CodeBERT was pre-trained on code flattened this way.
    - pdbert   : collapse only runs of spaces/tabs into a single space, but leave
                 newlines intact. PDBERT was pre-trained with newlines preserved.
    - unixcoder: same as codebert — full collapse. UniXCoder handles structure
                 through its pre-training objectives, not raw whitespace.
    """
    if model_type == 'pdbert':
        return re.sub(r'[ \t]+', ' ', func)   # preserve newlines
    else:
        return ' '.join(func.split())          # collapse everything


def convert_to_features(js, tokenizer, args):
    """
    Tokenize one code sample and build the input_ids sequence.

    Token format per model:
      codebert / pdbert : [CLS] code_tokens... [SEP]          (2 special tokens)
      unixcoder         : [CLS] <encoder-only> [SEP] code_tokens... [SEP]  (4 special tokens)

    UniXCoder was pre-trained with the <encoder-only> mode token, which tells it
    to use bidirectional (encoder) attention. Without it, performance drops
    significantly because the model never saw this input format during pre-training.
    """
    code = clean_code(js['func'], args.model_type)

    if args.model_type == 'unixcoder':
        # Reserve 4 slots for [CLS], <encoder-only>, [SEP], [SEP]
        code_tokens = tokenizer.tokenize(code)[:args.block_size - 4]
        tokens = (
            [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
            + code_tokens
            + [tokenizer.sep_token]
        )
    else:
        # codebert and pdbert: standard BERT-style format
        # Reserve 2 slots for [CLS] and [SEP]
        code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
        tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids += [tokenizer.pad_token_id] * (args.block_size - len(ids))
    return InputFeatures(ids, js['idx'], js['target'])


class VulnDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        with open(file_path) as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(file_path)}"):
                js = json.loads(line.strip())
                self.examples.append(convert_to_features(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds):
    acc    = accuracy_score(labels, preds)
    prec   = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1     = f1_score(labels, preds, zero_division=0)
    TN, FP, FN, TP = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tnr  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr  = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    bacc = (recall + tnr) / 2
    scale = lambda x: round(x, 4) * 100
    return scale(acc), scale(prec), scale(recall), scale(f1), scale(tnr), scale(fpr), scale(fnr), scale(bacc)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, train_dataset, eval_dataset, model, tokenizer):
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                              batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_dataset,  sampler=SequentialSampler(eval_dataset),
                              batch_size=args.eval_batch_size,  num_workers=4, pin_memory=True)

    total_steps  = args.epoch * len(train_loader)
    warmup_steps = int(total_steps * 0.1) if args.warmup_steps == -1 else args.warmup_steps

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ], lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info(f"Model type   : {args.model_type}")
    logger.info(f"Training     : {len(train_dataset)} examples, {args.epoch} epochs, batch={args.train_batch_size}")

    best_bacc = 0.0

    for epoch in range(args.epoch):
        model.train()
        model.zero_grad()
        total_loss, steps = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            loss, _ = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            total_loss += loss.item()
            steps += 1

            if steps % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = round(total_loss / steps, 5)
        logger.info(f"Epoch {epoch} — train loss: {avg_loss}")

        results = evaluate(args, model, eval_dataset, eval_loader)
        for k, v in results.items():
            logger.info(f"  {k} = {round(v, 4)}")
        

        if results['eval_bacc'] > best_bacc:
            best_bacc = results['eval_bacc']
            save_dir  = os.path.join(args.output_dir, args.localtime, args.project, 'checkpoint-best-bacc')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'model.bin')
            m = model.module if hasattr(model, 'module') else model
            torch.save(m.state_dict(), save_path)
            logger.info(f"  *** New best bacc {best_bacc:.4f} — saved to {save_path}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(args, model, dataset, dataloader):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, steps = 0.0, 0

    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        with torch.no_grad():
            loss, logits = model(inputs, labels)
        total_loss += loss.mean().item()
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        steps += 1

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds  = logits[:, 0] > 0.5

    acc, prec, recall, f1, tnr, fpr, fnr, bacc = compute_metrics(labels, preds)
    return {
        'eval_loss':   total_loss / steps,
        'eval_acc':    acc,   'eval_prec':   prec,   'eval_recall': recall,
        'eval_f1':     f1,    'eval_tnr':    tnr,    'eval_fpr':    fpr,
        'eval_fnr':    fnr,   'eval_bacc':   bacc,
    }


# ── Test ──────────────────────────────────────────────────────────────────────

def test(args, model, dataset, dataloader, name='bacc'):
    model.eval()
    all_logits, all_labels = [], []

    for inputs, labels in tqdm(dataloader, desc="Testing"):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            logits = model(inputs)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds  = logits[:, 0] > 0.5

    out_dir = os.path.join(args.output_dir, args.localtime, args.project, name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'predictions.txt'), 'w') as f:
        for ex, pred in zip(dataset.examples, preds):
            f.write(f"idx: {ex.idx}, pred: {int(pred)}, target: {ex.label}\n")

    acc, prec, recall, f1, tnr, fpr, fnr, bacc = compute_metrics(labels, preds)
    result = {
        'test_acc':  acc,   'test_prec':   prec,   'test_recall': recall,
        'test_f1':   f1,    'test_tnr':    tnr,    'test_fpr':    fpr,
        'test_fnr':  fnr,   'test_bacc':   bacc,
    }
    np.savez(os.path.join(out_dir, 'result.npz'), test_result=result)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()

    # Required
    p.add_argument('--project',         type=str, required=True,
                   help="Name for this run, used to organise output folders.")
    p.add_argument('--model_type',      type=str, required=True, choices=SUPPORTED_MODELS,
                   help="Which model to use: codebert | unixcoder | pdbert")
    p.add_argument('--model_name_or_path', type=str, required=True,
                   help="HuggingFace model id or local path to model weights.")
    p.add_argument('--train_data_file', type=str, default=None)
    p.add_argument('--output_dir',      type=str, required=True)

    # Optional
    p.add_argument('--eval_data_file',  type=str, default=None)
    p.add_argument('--test_data_file',  type=str, default=None)
    p.add_argument('--tokenizer_name',  type=str, default='',
                   help="Tokenizer id or path. Defaults to model_name_or_path if not set.")
    p.add_argument('--config_name',     type=str, default='',
                   help="Config id or path. Defaults to model_name_or_path if not set.")
    p.add_argument('--cache_dir',       type=str, default='')
    p.add_argument('--model_dir',       type=str, default='',
                   help="Unused — kept for backwards compatibility with original launch scripts.")
    p.add_argument('--block_size',      type=int, default=512)

    # Flags
    p.add_argument('--do_train',        action='store_true')
    p.add_argument('--do_eval',         action='store_true',
                   help="Unused — evaluation runs automatically each epoch during training.")
    p.add_argument('--do_test',         action='store_true')

    # Hyperparameters
    p.add_argument('--epoch',                      type=int,   default=10)
    p.add_argument('--train_batch_size',           type=int,   default=16)
    p.add_argument('--eval_batch_size',            type=int,   default=32)
    p.add_argument('--learning_rate',              type=float, default=2e-5)
    p.add_argument('--weight_decay',               type=float, default=0.0)
    p.add_argument('--adam_epsilon',               type=float, default=1e-8)
    p.add_argument('--max_grad_norm',              type=float, default=1.0)
    p.add_argument('--warmup_steps',               type=int,   default=-1,
                   help="Linear warmup steps. -1 = auto (10%% of total steps).")
    p.add_argument('--gradient_accumulation_steps',type=int,   default=1)
    p.add_argument('--seed',                       type=int,   default=42)
    p.add_argument('--gpu',                        type=int,   default=-1,
                   help="GPU id to use. -1 = use all available GPUs.")
    p.add_argument('--localtime',                  type=str,   default='run')
    p.add_argument('--basetime',                   type=str,   default='run',
                   help="Timestamp of a previous training run, used to load its checkpoint for --do_test only.")
    p.add_argument('--checkpoint_path', type=str, default=None,
               help="Direct path to a model.bin checkpoint file. Overrides localtime/basetime.")

    args = p.parse_args()

    # ── Device setup ──────────────────────────────────────────────────────────
    if args.gpu == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu  = torch.cuda.device_count()
    else:
        args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.n_gpu  = 1

    args.per_gpu_train_batch_size = args.train_batch_size // max(1, args.n_gpu)
    args.per_gpu_eval_batch_size  = args.eval_batch_size  // max(1, args.n_gpu)

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/{args.localtime}.log',
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", filemode='w',
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    set_seed(args.seed)

    # ── Model loading ─────────────────────────────────────────────────────────
    # All three models (CodeBERT, UniXCoder, PDBERT) share the RoBERTa architecture,
    # so we always use RobertaConfig / RobertaTokenizer / RobertaModel.
    #
    # We load RobertaModel (plain encoder) rather than RobertaForSequenceClassification
    # because model.py's Model class extracts the [CLS] hidden state and adds its own
    # classification head. Loading the "ForSequenceClassification" variant would add an
    # unwanted second head on top.
    config    = RobertaConfig.from_pretrained(
        args.config_name or args.model_name_or_path,
        num_labels=1,
        cache_dir=args.cache_dir or None,
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        cache_dir=args.cache_dir or None,
    )
    backbone  = RobertaModel.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir or None,
    )
    model = Model(backbone, config, tokenizer, args)

    logger.info(f"Args: {args}")

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.do_train:
        if not args.train_data_file:
            p.error("--train_data_file is required when --do_train is set.")
        if not args.eval_data_file:
            p.error("--eval_data_file is required when --do_train is set.")
        train_dataset = VulnDataset(tokenizer, args, args.train_data_file)
        eval_dataset  = VulnDataset(tokenizer, args, args.eval_data_file)
        train(args, train_dataset, eval_dataset, model, tokenizer)

    # ── Test ──────────────────────────────────────────────────────────────────
    if args.do_test:
        test_dataset = VulnDataset(tokenizer, args, args.test_data_file)
        test_loader  = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                  batch_size=args.eval_batch_size)

        # If we just trained, load from localtime; otherwise load from a previous run via basetime
        if args.checkpoint_path:
            ckpt_path = args.checkpoint_path
        elif args.do_train:
            ckpt_path = os.path.join(args.output_dir, args.localtime, args.project,
                                     'checkpoint-best-bacc', 'model.bin')
        else:
            ckpt_path = os.path.join(args.output_dir, args.basetime, args.project,
                                     'checkpoint-best-bacc', 'model.bin')
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model.to(args.device)

        result = test(args, model, test_dataset, test_loader)
        logger.info("***** Test results *****")
        for k, v in sorted(result.items()):
            logger.info(f"  {k} = {round(v, 4)}")


if __name__ == "__main__":
    main()