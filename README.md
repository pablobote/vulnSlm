# Vulnerability Detection with Small Language Models
Fine-tunes CodeBERT and UniXCoder on the PrimeVul dataset for binary vulnerability classification of C/C++ functions.


---

## Dependencies

Python 3.8+ and the following packages:

```
torch
transformers
scikit-learn
numpy
tqdm
```

Install with:

```bash
pip install torch transformers scikit-learn numpy tqdm
```

---

## Dataset

Download the reconstructed PrimeVul dataset from the RevisitVD repo:
https://github.com/youpengl/RevisitVD/tree/main/dataset

You need three files:
- `reconstructed_train.jsonl`
- `reconstructed_valid.jsonl`
- `reconstructed_test.jsonl`

Place them in a `dataset/` folder at the same level as the scripts. Each line is a JSON object with fields `func` (the C/C++ function), `target` (0 or 1), `idx`, and others.

For the generalizability test you also need the self-collected NVD dataset from the same repo.
---

## Files

```
├── V_Simple_Finetune_SLMs.py   
├── dataset_perturbation.py     
├── model.py                    
└── dataset/
    ├── reconstructed_train.jsonl
    ├── reconstructed_valid.jsonl
    └── reconstructed_test.jsonl
```

---

## How to Run

### Train and test CodeBERT

```bash
python V_Simple_Finetune_SLMs.py \
    --project CodeBERT \
    --model_type codebert \
    --model_name_or_path microsoft/codebert-base \
    --tokenizer_name microsoft/codebert-base \
    --do_train --do_eval --do_test \
    --train_data_file ../dataset/reconstructed_train.jsonl \
    --eval_data_file ../dataset/reconstructed_valid.jsonl \
    --test_data_file ../dataset/reconstructed_test.jsonl \
    --epoch 15 \
    --block_size 512 \
    --train_batch_size 16 \
    --learning_rate 2e-5 \
    --output_dir ./output \
    --model_dir ./weights
```

### Train and test UniXCoder

Same command, just change the model arguments:

```bash
python V_Simple_Finetune_SLMs.py \
    --project UniXCoder \
    --model_type unixcoder \
    --model_name_or_path microsoft/unixcoder-base \
    --tokenizer_name microsoft/unixcoder-base \
    --do_train --do_eval --do_test \
    --train_data_file ../dataset/reconstructed_train.jsonl \
    --eval_data_file ../dataset/reconstructed_valid.jsonl \
    --test_data_file ../dataset/reconstructed_test.jsonl \
    --epoch 15 \
    --block_size 512 \
    --train_batch_size 16 \
    --learning_rate 2e-5 \
    --output_dir ./output \
    --model_dir ./weights
```

### Test only (using a saved checkpoint)

```bash
python V_Simple_Finetune_SLMs.py --do_test \
    --model_type unixcoder \
    --model_name_or_path microsoft/unixcoder-base \
    --test_data_file ../dataset/reconstructed_test.jsonl \
    --project UniXCoder \
    --checkpoint_path ./output/run/UniXCoder/checkpoint-best-bacc/model.bin \
    --output_dir ./output \
    --block_size 512 \
    --eval_batch_size 32
```

---

## Robustness Check

First generate the perturbed test sets:

```bash
python dataset_perturbation.py \
    --input ../dataset/reconstructed_test.jsonl \
    --output_dir ../dataset \
    --perturb all
```

This creates three files: `reconstructed_test_norm.jsonl`, `reconstructed_test_no_norm.jsonl`, and `reconstructed_test_abstract.jsonl`.

Then run `--do_test` on each one using the same `--checkpoint_path` as above, changing `--test_data_file` and `--project` for each.
