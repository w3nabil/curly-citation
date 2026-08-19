"""
This code was made with the assistance of Kimi K3.
If you encounter an error, please open an issue.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
import tiktoken
from transformers import AutoTokenizer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Display name -> file path. Display names become table column headers,
# so keep them short. Order here is preserved in every output table.

ROOT = Path(__file__).resolve().parent.parent

FILES: dict[str, str] = {
    "IEEE": str(ROOT / "dataset" / "large" / "1_ieee.md"),
    "CCF": str(ROOT / "dataset" / "large" / "1_ccf.md"),
    "NoCitation": str(ROOT / "dataset" / "large" / "1_no_citation.md"),
}

# Only this extension is accepted. Enforced in load_files() before any
# tokenizer runs, same reasoning as the up-front existence/emptiness
# checks: a wrong file type should fail immediately, not after several
# minutes of tokenizer downloads.
ALLOWED_EXTENSION = ".md"

DEFAULT_OUTPUT_PATH = ROOT / "results" / "large" / "1.md"

# Which file's column the "Rank" and every reduction/comparison figure
# is computed against. Chosen by NAME (a key in FILES), not by "first
# file" or "whichever ran first and succeeded" -- run order and
# success order are not guaranteed, so picking positionally can
# silently point at a different file than intended if an earlier one
# fails to load. Change this to re-target the comparison at a
# different file without touching FILES or any rendering code.
RANKING_COLUMN = "CCF"

# Which local, offline tiktoken encoding to treat as the fixed
# per-file baseline in the "Per-File Detail" section further down.
# Same "pick by name, not position" reasoning as RANKING_COLUMN above.
BASELINE_NAME = "cl100k_base"

TIKTOKEN_ENCODINGS: dict[str, str] = {
    "cl100k_base": "cl100k_base",
    "o200k_base": "o200k_base",
}

# Each entry: display name -> repo config.
HF_TOKENIZERS: dict[str, dict] = {
    "Gemma 4 31B": {
        "repo": "google/gemma-4-31B-it",
        "trust_remote_code": False,
    },
    "Qwen3.8 27B": {
        "repo": "Qwen/Qwen3.8-27B",
        "trust_remote_code": False,
    },
    "Kimi K2.6": {
        "repo": "moonshotai/Kimi-K2.6",
        "trust_remote_code": True,
    },
    "Llama 4 Scout 17B": {
        "repo": "unsloth/Llama-4-Scout-17B-16E-Instruct",
        "trust_remote_code": False,
    },
    "DeepSeek-V4 Flash 0731": {
        "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "trust_remote_code": True,
    },
    "DeepSeek-V4 Pro 0813": {
        "repo": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "trust_remote_code": False,
    },
    "GPT OSS 120b": {
        "repo": "openai/gpt-oss-120b",
        "trust_remote_code": False
    },
    "Mistral Small 4 119B": {
        "repo": "mistralai/Mistral-Small-4-119B-2603",
        "trust_remote_code": False,
    },
    "GLM-5.2": {
        "repo": "zai-org/GLM-5.2",
        "trust_remote_code": True,
    },
    "OLMo 3 32B": {
        "repo": "allenai/Olmo-3-1125-32B",
        "trust_remote_code": False,
    },
    "Llama 3 8B": {
        "repo": "thinkingmachines/meta-llama-3-tokenizer",
        "trust_remote_code": False,
    },
    "Kimi K3": {
        "repo": "moonshotai/Kimi-K3",
        "trust_remote_code": True,
    },
    "MiMo 2.5 Pro": {
        "repo": "XiaomiMiMo/MiMo-V2.5-Pro",
        "trust_remote_code": False,
    },
    "Qwen 3.8 2.4T": {
        "repo": "Qwen/Qwen3.8-2.4T-A95B",
        "trust_remote_code": False,
    },
    "MiniMax M3": {
        "repo": "MiniMaxAI/MiniMax-M3",
        "trust_remote_code": False
    },
    "Mistral 7B Instruct v0.3": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
        "fix_mistral_regex": True
    },
    "Qwen 3.5 35B-A3B": {
        "repo": "Qwen/Qwen3.5-35B-A3B",
        "trust_remote_code": False,
    },
    "Qwen 3.5 9B": {
        "repo": "Qwen/Qwen3.5-9B",
        "trust_remote_code": False,
    },
    "Gemma 3 27B": {
        "repo": "google/gemma-3-27b-it",
        "trust_remote_code": False,
    },
    "Mistral Small 3.1 24B": {
        "repo": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "trust_remote_code": False,
        "fix_mistral_regex": True
    },
    "Phi-4": {
        "repo": "microsoft/phi-4",
        "trust_remote_code": False,
    },
    "Cohere Command A": {
        "repo": "CohereLabs/command-a-reasoning-08-2025",
        "trust_remote_code": False,
    },
}


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str
    category: str  # "tiktoken" | "huggingface"
    tokens: int
    status: str = "OK"  # "OK" | "FAILED"
    error: str | None = None


@dataclass
class FileInfo:
    display_name: str  # column header in output tables, e.g. "CCF"
    path: Path
    text: str
    char_count: int
    word_count: int
    # tokenizer display name -> BenchmarkResult, for THIS file
    results: dict[str, BenchmarkResult]


# --------------------------------------------------------------------------
# File loading
# --------------------------------------------------------------------------

def read_markdown(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def load_files(files: dict[str, str]) -> list[FileInfo]:
    """
    Reads every input file up front, before any tokenizer runs, so a
    missing, empty, or wrong-extension file is reported immediately
    rather than after several minutes of tokenizer downloads.

    `files` is display_name -> path string, e.g. FILES above. Order of
    dict iteration is preserved into the returned list, which in turn
    fixes the left-to-right column order in every rendered table.
    """
    if not files:
        raise ValueError("FILES is empty -- add at least one display_name: path entry.")

    if RANKING_COLUMN not in files:
        raise ValueError(
            f"RANKING_COLUMN is set to '{RANKING_COLUMN}', which is not "
            f"a key in FILES ({list(files.keys())}). Ranking needs a "
            f"real column to sort against."
        )

    infos: list[FileInfo] = []
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()

    for display_name, path_str in files.items():
        if display_name in seen_names:
            raise ValueError(f"Duplicate display name in FILES: '{display_name}'")
        seen_names.add(display_name)

        path = Path(path_str)

        if path.suffix.lower() != ALLOWED_EXTENSION:
            raise ValueError(
                f"'{display_name}' -> {path} has extension "
                f"'{path.suffix or '(none)'}', but only "
                f"{ALLOWED_EXTENSION} files are accepted."
            )

        resolved = path.resolve()
        if resolved in seen_paths:
            print(f"Skipping duplicate input path for '{display_name}': {path}")
            continue
        seen_paths.add(resolved)

        text = read_markdown(path)
        if not text.strip():
            raise ValueError(f"{path} ('{display_name}') is empty.")

        infos.append(FileInfo(
            display_name=display_name,
            path=path,
            text=text,
            char_count=len(text),
            word_count=len(text.split()),
            results={},
        ))

    if not infos:
        raise ValueError("No valid input files after removing duplicates.")
    return infos


# --------------------------------------------------------------------------
# tiktoken
# --------------------------------------------------------------------------

def benchmark_tiktoken(text: str, name: str, encoding) -> BenchmarkResult:
    try:
        tokens = len(encoding.encode(text))
        return BenchmarkResult(name=name, category="tiktoken", tokens=tokens)
    except Exception as exc:
        return BenchmarkResult(
            name=name, category="tiktoken", tokens=0, status="FAILED", error=str(exc)
        )


# --------------------------------------------------------------------------
# Hugging Face
# --------------------------------------------------------------------------

def count_hf_tokens(tokenizer, text: str) -> int:
    # add_special_tokens=False keeps this an apples-to-apples raw content
    # count across every tokenizer in this script (tiktoken's encode()
    # also adds no special tokens by default). Leaving the HF default of
    # True would add a constant BOS/EOS-style offset that skews the
    # reduction percentages, especially on shorter files.
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def benchmark_hf(text: str, name: str, tokenizer) -> BenchmarkResult:
    try:
        tokens = count_hf_tokens(tokenizer, text)
        return BenchmarkResult(name=name, category="huggingface", tokens=tokens)
    except Exception as exc:
        return BenchmarkResult(
            name=name, category="huggingface", tokens=0, status="FAILED", error=str(exc)
        )


# --------------------------------------------------------------------------
# Tokenizer loading (once per run, reused across every file)
# --------------------------------------------------------------------------

def load_all_tokenizers() -> tuple[dict[str, object], list[tuple[str, str]]]:
    """
    Loads every tiktoken encoding and every Hugging Face tokenizer ONE
    TIME, regardless of how many files will be benchmarked. Repeating
    from_pretrained() per file per tokenizer would re-download or
    re-initialize the same vocab N times for zero benefit, since a
    tokenizer's vocabulary doesn't change based on the text it's later
    asked to encode.

    Returns:
        tiktoken_encodings: name -> loaded tiktoken Encoding
        hf_load_failures: list of (name, error string) for repos that
            failed to load at all -- kept separate from per-file
            tokenization failures so a load failure is reported once,
            not once per file.
    """
    tiktoken_encodings: dict[str, object] = {}
    print("Loading tiktoken encodings...")
    for display_name, encoding_name in TIKTOKEN_ENCODINGS.items():
        try:
            tiktoken_encodings[display_name] = tiktoken.get_encoding(encoding_name)
            print(f"  OK: {display_name}")
        except Exception as exc:
            print(f"  FAILED: {display_name} ({exc})")

    loaded_hf: dict[str, object] = {}
    hf_load_failures: list[tuple[str, str]] = []
    print("Loading Hugging Face tokenizers...")
    for display_name, config in HF_TOKENIZERS.items():
        try:
            loaded_hf[display_name] = AutoTokenizer.from_pretrained(
                config["repo"], trust_remote_code=config["trust_remote_code"]
            )
            print(f"  OK: {display_name} ({config['repo']})")
        except Exception as exc:
            print(f"  FAILED: {display_name} ({config['repo']}): {exc}")
            hf_load_failures.append((display_name, str(exc)))

    return tiktoken_encodings, loaded_hf, hf_load_failures


# --------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------

def reduction(baseline: int, comparison: int) -> str:
    if baseline == 0:
        return "—"
    value = (baseline - comparison) / baseline * 100
    return f"{value:+.2f}%"  # signed: negative means MORE tokens than baseline


def chars_per_token(char_count: int, tokens: int) -> str:
    if tokens == 0:
        return "—"
    return f"{char_count / tokens:.2f}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    output_path = DEFAULT_OUTPUT_PATH

    print("=" * 60)
    print("Multi-File, Multi-Tokenizer Benchmark")
    print("=" * 60)

    file_infos = load_files(FILES)
    print(f"Files: {len(file_infos)}  (ranking column: {RANKING_COLUMN})")
    for info in file_infos:
        print(f"  {info.display_name}: {info.path}  ({info.char_count:,} chars, {info.word_count:,} words)")
    print()

    tiktoken_encodings, loaded_hf, hf_load_failures = load_all_tokenizers()
    print()

    # Every loaded tokenizer runs against every file. HF repos that
    # failed to LOAD (network, gating, missing repo) are recorded once
    # in hf_load_failures above and are not retried per file; repos
    # that loaded fine but fail on a PARTICULAR file's text still get a
    # per-file FAILED result via benchmark_hf's own try/except.
    for info in file_infos:
        print(f"--- Tokenizing: {info.display_name} ({info.path}) ---")

        for display_name, encoding in tiktoken_encodings.items():
            result = benchmark_tiktoken(info.text, display_name, encoding)
            info.results[display_name] = result
            print(f"  {display_name}: {result.tokens:,} tokens" if result.status == "OK"
                  else f"  {display_name}: {result.status}")

        for display_name, tokenizer in loaded_hf.items():
            result = benchmark_hf(info.text, display_name, tokenizer)
            info.results[display_name] = result
            print(f"  {display_name}: {result.tokens:,} tokens" if result.status == "OK"
                  else f"  {display_name}: {result.status}")

        print()

    # Every tokenizer name that was attempted at least once, across all
    # files, in the order: tiktoken, then HF (load successes and
    # failures both -- a load failure still deserves a row saying so).
    all_tokenizer_names: list[str] = (
        list(tiktoken_encodings.keys())
        + list(HF_TOKENIZERS.keys())
    )

    render_report(
        output_path=output_path,
        file_infos=file_infos,
        all_tokenizer_names=all_tokenizer_names,
        hf_load_failures=hf_load_failures,
    )

    print("=" * 60)
    print("Benchmark complete.")
    print("=" * 60)
    print(f"Output: {output_path}")


def render_report(
    output_path: Path,
    file_infos: list[FileInfo],
    all_tokenizer_names: list[str],
    hf_load_failures: list[tuple[str, str]],
) -> None:
    output: list[str] = []
    display_names = [info.display_name for info in file_infos]

    output.append("# Token Comparison Report")
    output.append("")
    output.append(f"- **Run date (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"- **Files compared:** {', '.join(display_names)}")
    output.append(f"- **Ranked by:** {RANKING_COLUMN} tokens (ascending -- fewer tokens ranks higher)")
    output.append("")

    # --- File summary table ---
    output.append("## Files")
    output.append("")
    output.append("| Name | File | Characters | Words |")
    output.append("| :--- | :--- | ---------: | ----: |")
    for info in file_infos:
        output.append(f"| {info.display_name} | `{info.path}` | {info.char_count:,} | {info.word_count:,} |")
    output.append("")

    # --- Main comparison table: one row per tokenizer, one token
    #     column per file, ranked by the RANKING_COLUMN file's token
    #     count ascending. ---
    output.append("## Token Counts by Tokenizer")
    output.append("")

    header_cells = ["Rank", "Tokenizer"] + display_names + ["Category"]
    align_cells = ["---:", ":--"] + ["-----:"] * len(display_names) + [":------"]
    output.append("| " + " | ".join(header_cells) + " |")
    output.append("| " + " | ".join(align_cells) + " |")

    # Sort key: the RANKING_COLUMN file's token count for this
    # tokenizer, ascending (lower = better = rank 1). A tokenizer with
    # no OK result on the ranking file (load failure, per-file failure)
    # sorts to the bottom -- it has no ranking-column value to rank BY,
    # so it can't meaningfully claim a numeric rank ahead of tokenizers
    # that do.
    def ranking_sort_key(tok_name: str) -> tuple[int, int]:
        result = file_infos_by_display(file_infos, RANKING_COLUMN).results.get(tok_name)
        if result is not None and result.status == "OK":
            return (0, result.tokens)
        return (1, 0)

    sorted_tokenizer_names = sorted(all_tokenizer_names, key=ranking_sort_key)

    rank = 0
    for tok_name in sorted_tokenizer_names:
        ranking_result = file_infos_by_display(file_infos, RANKING_COLUMN).results.get(tok_name)
        has_rank = ranking_result is not None and ranking_result.status == "OK"
        if has_rank:
            rank += 1
            rank_cell = str(rank)
        else:
            rank_cell = "—"

        # category comes from whichever file has a result for this
        # tokenizer, since a tokenizer that failed to load entirely
        # has no per-file result to read category off of anywhere.
        category = next(
            (r.category for info in file_infos if (r := info.results.get(tok_name)) is not None),
            "huggingface",
        )

        row_cells = [rank_cell, tok_name]
        for info in file_infos:
            result = info.results.get(tok_name)
            if result is None:
                row_cells.append("LOAD FAILED")
            elif result.status == "OK":
                row_cells.append(f"{result.tokens:,}")
            else:
                row_cells.append(result.status)
        row_cells.append(category)

        output.append("| " + " | ".join(row_cells) + " |")

    output.append("")
    output.append(
        f"**Rank** is assigned by sorting tokenizers on their **{RANKING_COLUMN}** "
        f"column ascending -- rank 1 is the tokenizer that produced the "
        f"*fewest* tokens for {RANKING_COLUMN}, i.e. the most efficient "
        f"tokenizer on that file. Tokenizers with no successful "
        f"{RANKING_COLUMN} result (load failure or per-file failure) "
        f"have no numeric rank and sort to the bottom."
    )
    output.append("")

    # --- Per-file baseline-reduction tables (chars/token, reduction vs
    #     the fixed BASELINE_NAME tiktoken encoding). ---
    output.append("## Per-File Detail (vs. tokenizer baseline)")
    output.append("")
    for info in file_infos:
        successful = [r for r in info.results.values() if r.status == "OK"]
        if not successful:
            output.append(f"### {info.display_name}: `{info.path}`")
            output.append("")
            output.append("No tokenizer succeeded on this file.")
            output.append("")
            continue

        baseline_result = next((r for r in successful if r.name == BASELINE_NAME), None)
        baseline_note = None
        if baseline_result is None:
            baseline_result = successful[0]
            baseline_note = (
                f"WARNING: configured baseline '{BASELINE_NAME}' did not "
                f"succeed for this file. Falling back to "
                f"'{baseline_result.name}' ({baseline_result.tokens:,} "
                f"tokens) as the comparison point."
            )
        baseline = baseline_result.tokens

        output.append(f"### {info.display_name}: `{info.path}`")
        output.append("")
        output.append(f"Baseline: **{baseline_result.name}** ({baseline:,} tokens)")
        if baseline_note:
            output.append("")
            output.append(f"**{baseline_note}**")
        output.append("")
        output.append("| Rank | Tokenizer | Tokens | Chars/Token | Reduction vs Baseline |")
        output.append("| ---: | :-------- | -----: | ----------: | ---------------------: |")
        for local_rank, result in enumerate(sorted(successful, key=lambda r: r.tokens), start=1):
            output.append(
                f"| {local_rank} | {result.name} | {result.tokens:,} | "
                f"{chars_per_token(info.char_count, result.tokens)} | "
                f"{reduction(baseline, result.tokens)} |"
            )
        output.append("")

    # --- Tokenizer load failures (HF repos that never loaded, any file) ---
    if hf_load_failures:
        output.append("## Tokenizers That Failed to Load")
        output.append("")
        output.append(
            "These Hugging Face repos failed during `from_pretrained()` "
            "before any file was tokenized -- the failure is independent "
            "of file content, so it is reported once here rather than "
            "repeated per file."
        )
        output.append("")
        for name, error in hf_load_failures:
            output.append(f"### {name}")
            output.append("")
            output.append("```text")
            output.append(error)
            output.append("```")
            output.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output), encoding="utf-8")


def file_infos_by_display(file_infos: list[FileInfo], display_name: str) -> FileInfo:
    """
    Looks up a FileInfo by its display name. Used to pull the
    RANKING_COLUMN file's results out of file_infos when computing sort
    order. Raises if the name isn't present -- load_files() already
    validates RANKING_COLUMN is a key in FILES before any tokenizing
    happens, so hitting this is a bug in this script, not a user
    config mistake.
    """
    for info in file_infos:
        if info.display_name == display_name:
            return info
    raise KeyError(
        f"'{display_name}' not found among loaded files "
        f"({[i.display_name for i in file_infos]}). This should have "
        f"been caught by load_files()'s RANKING_COLUMN check."
    )


if __name__ == "__main__":
    main()