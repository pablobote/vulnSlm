# coding=utf-8
"""
perturb_dataset.py — Robustness perturbation script for vulnerability detection datasets.

Reads a .jsonl test file and writes perturbed copies that can be fed directly to
Finetune_VulnDetect.py with --do_test (no retraining needed).

Two perturbations are supported:

  1. NORMALIZATION  (--perturb norm)
     Collapses all whitespace — spaces, tabs, newlines — into single spaces.
     This is the most aggressive formatting normalization possible.
     Purpose: test whether the model relies on code layout/indentation to make
     its predictions, or whether it has learned the actual vulnerable logic.
     Models pre-trained on formatted code (e.g. PDBERT) are expected to suffer
     a larger drop than models already trained on collapsed code (e.g. UniXCoder).

  2. VARIABLE ABSTRACTION  (--perturb abstract)
     Renames every user-defined identifier (variables, parameters, function names)
     to generic tokens: var_0, var_1, ... / func_0, func_1, ...
     C keywords, types, and standard library names are left untouched.
     Purpose: test whether the model exploits identifier names (e.g. learning
     that "memcpy" or "buf" correlate with vulnerabilities) rather than structural
     patterns. A robust model should maintain performance; a model that "cheated"
     by memorising risky names will show a meaningful drop in recall.

Usage examples:
  # Generate normalised test set
  python perturb_dataset.py --input ../dataset/reconstructed_test.jsonl \
                             --output ../dataset/test_norm.jsonl \
                             --perturb norm

  # Generate abstracted test set
  python perturb_dataset.py --input ../dataset/reconstructed_test.jsonl \
                             --output ../dataset/test_abstract.jsonl \
                             --perturb abstract

  # Generate both at once
  python perturb_dataset.py --input ../dataset/reconstructed_test.jsonl \
                             --output_dir ../dataset \
                             --perturb both

Then evaluate a trained checkpoint against each file:
  python Finetune_VulnDetect.py --do_test \
      --model_type unixcoder \
      --model_name_or_path microsoft/unixcoder-base \
      --test_data_file ../dataset/test_norm.jsonl \
      --project UniXCoder_norm \
      --basetime <your_training_localtime> \
      ... (same other args as training)
"""

import argparse
import json
import os
import re
from tqdm import tqdm


# ── C keywords and common types/stdlib names to preserve during abstraction ───
# These are left as-is so the model still sees meaningful structural tokens.
C_KEYWORDS = {
    # control flow
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
    'return', 'goto', 'default',
    # types
    'int', 'char', 'float', 'double', 'long', 'short', 'unsigned', 'signed',
    'void', 'bool', 'size_t', 'ssize_t', 'uint8_t', 'uint16_t', 'uint32_t',
    'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'ptrdiff_t',
    'intptr_t', 'uintptr_t', 'off_t', 'pid_t', 'FILE',
    # storage / qualifiers
    'static', 'const', 'volatile', 'extern', 'register', 'inline', 'auto',
    'struct', 'union', 'enum', 'typedef',
    # common stdlib / security-relevant functions — kept so model can still
    # recognise dangerous call patterns (memcpy, strcpy, etc.)
    'memcpy', 'memmove', 'memset', 'memcmp', 'memchr',
    'strcpy', 'strncpy', 'strcat', 'strncat', 'strlen', 'strcmp', 'strncmp',
    'sprintf', 'snprintf', 'printf', 'fprintf', 'scanf', 'sscanf',
    'malloc', 'calloc', 'realloc', 'free', 'alloca',
    'open', 'close', 'read', 'write', 'fopen', 'fclose', 'fread', 'fwrite',
    'NULL', 'true', 'false', 'sizeof', 'offsetof',
    'assert', 'abort', 'exit', 'atoi', 'atol', 'strtol', 'strtoul',
}


# ── Perturbation 1: Normalization ─────────────────────────────────────────────

def perturb_normalize(code: str) -> str:
    """
    Collapse ALL whitespace (spaces, tabs, newlines) into single spaces.
    This is the most aggressive normalization — it destroys all indentation
    and line structure, producing a single long line of tokens.
    """
    return ' '.join(code.split())


# ── Perturbation 2: No normalization (raw code) ──────────────────────────────

def perturb_no_norm(code: str) -> str:
    """
    Pass the code through completely untouched — no whitespace changes at all.
    This is a MISMATCH perturbation for UniXCoder: the model was trained on
    fully collapsed code (' '.join(func.split())), so feeding it raw code with
    original newlines and indentation intact creates a genuine train/test
    distribution mismatch.

    This mirrors the paper's RQ4 "W/o normalization" setting — the reverse
    direction of the original plan: instead of stripping newlines from a model
    that expects them (PDBERT), we keep newlines for a model that never saw
    them (UniXCoder). The paper found PDBERT dropped ~6.9% BACC under this
    kind of mismatch.
    """
    return code


# ── Perturbation 3: Variable abstraction ─────────────────────────────────────

# Regex that matches C identifiers: starts with letter or underscore,
# followed by any mix of letters, digits, underscores.
_IDENT_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')


def perturb_abstract(code: str) -> str:
    """
    Replace user-defined identifiers with generic names (var_0, var_1, func_N).

    Strategy:
    - Scan the code for all identifiers not in C_KEYWORDS.
    - Identifiers that appear immediately before '(' in the token stream are
      treated as function names → func_0, func_1, ...
    - All other identifiers are treated as variables → var_0, var_1, ...
    - The mapping is consistent within a single function: the same original
      name always maps to the same abstract name, so the model can still track
      data flow between uses of the same variable.
    - Numeric and string literals are left untouched (they carry semantic
      information about buffer sizes, offsets, etc. that is relevant to vulns).
    """
    # First pass: find all identifiers and classify as function vs variable
    # by checking whether the identifier is followed by '('
    func_names  = set()
    var_names   = set()

    # We look at the raw token stream to decide: identifier immediately
    # before '(' is a function call/definition.
    tokens_with_pos = list(_IDENT_RE.finditer(code))
    for m in tokens_with_pos:
        name = m.group(1)
        if name in C_KEYWORDS:
            continue
        # Check what follows the identifier in the source (skip whitespace)
        after = code[m.end():].lstrip()
        if after.startswith('('):
            func_names.add(name)
        else:
            var_names.add(name)

    # Build consistent name → abstract_name mappings
    func_map = {name: f'func_{i}' for i, name in enumerate(sorted(func_names))}
    var_map  = {name: f'var_{i}'  for i, name in enumerate(sorted(var_names))}

    # Merge (function map takes priority if a name appears in both, which can
    # happen when a function pointer is also used as a value)
    name_map = {**var_map, **func_map}

    # Second pass: replace all identifiers using the map
    def replace(m):
        name = m.group(1)
        if name in C_KEYWORDS:
            return name
        return name_map.get(name, name)   # unknown names (e.g. macros) pass through

    return _IDENT_RE.sub(replace, code)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def apply_perturbation(input_path: str, output_path: str, perturb_fn, desc: str):
    """Read a .jsonl file, apply perturb_fn to each 'func' field, write result."""
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n[{desc}] Processing {len(records)} samples → {output_path}")
    perturbed = []
    for js in tqdm(records, desc=desc):
        js_out = dict(js)                          # shallow copy, preserves idx/target
        js_out['func'] = perturb_fn(js['func'])
        perturbed.append(js_out)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        for js in perturbed:
            f.write(json.dumps(js) + '\n')

    print(f"  ✓ Written {len(perturbed)} records.")


def make_output_path(output_dir: str, input_path: str, suffix: str) -> str:
    """Derive an output filename like test_norm.jsonl from reconstructed_test.jsonl."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f'{base}_{suffix}.jsonl')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate perturbed versions of a .jsonl vulnerability dataset."
    )
    p.add_argument('--input',      type=str, required=True,
                   help="Path to the original .jsonl test file.")
    p.add_argument('--perturb',    type=str, required=True,
                   choices=['norm', 'no_norm', 'abstract', 'all'],
                   help="Which perturbation(s) to apply. 'all' generates norm + no_norm + abstract.")

    # Output: either a single --output file (for single perturbation) or
    # --output_dir for 'both' (two files will be created automatically).
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument('--output',     type=str, default=None,
                           help="Output .jsonl path (for --perturb norm or abstract).")
    out_group.add_argument('--output_dir', type=str, default=None,
                           help="Output directory (required for --perturb all).")

    args = p.parse_args()

    if args.perturb == 'norm':
        if args.output is None:
            p.error("--output is required when --perturb norm")
        apply_perturbation(args.input, args.output, perturb_normalize, 'Normalization')

    elif args.perturb == 'no_norm':
        if args.output is None:
            p.error("--output is required when --perturb no_norm")
        apply_perturbation(args.input, args.output, perturb_no_norm, 'No-Normalization (raw)')

    elif args.perturb == 'abstract':
        if args.output is None:
            p.error("--output is required when --perturb abstract")
        apply_perturbation(args.input, args.output, perturb_abstract, 'Abstraction')

    elif args.perturb == 'all':
        if args.output_dir is None:
            p.error("--output_dir is required when --perturb all")
        norm_path     = make_output_path(args.output_dir, args.input, 'norm')
        no_norm_path  = make_output_path(args.output_dir, args.input, 'no_norm')
        abstract_path = make_output_path(args.output_dir, args.input, 'abstract')
        apply_perturbation(args.input, norm_path,     perturb_normalize, 'Normalization')
        apply_perturbation(args.input, no_norm_path,  perturb_no_norm,   'No-Normalization (raw)')
        apply_perturbation(args.input, abstract_path, perturb_abstract,  'Abstraction')

    print("\nDone. Now run Finetune_VulnDetect.py --do_test on the perturbed file(s).")
    print("Compare the test metrics against your baseline (unperturbed) test results.")


if __name__ == '__main__':
    main()
