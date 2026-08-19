"""
Single-citation tokenizer benchmark with comprehensive statistics and reporting.

This code was made with the assistance of Kimi K3.
If you encounter an error, please open an issue.

"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from statistics import median, stdev
from typing import Optional
import tiktoken
from transformers import AutoTokenizer


# ==========================================================================
# Logging Utilities
# ==========================================================================

class Logger:
    """Simple structured logger with timestamps."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = time.time()

    def info(self, message: str) -> None:
        if self.verbose:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] {message}")

    def error(self, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts}] ERROR: {message}", file=sys.stderr)

    def section(self, title: str) -> None:
        self.info(f"\n{'=' * 70}")
        self.info(f"{title}")
        self.info(f"{'=' * 70}")

    def elapsed(self) -> float:
        return time.time() - self.start_time


logger = Logger(verbose=True)


# ==========================================================================
# Configuration
# ==========================================================================

ROOT = Path(__file__).resolve().parent.parent

FILES: dict[str, Path] = {
    "InText": ROOT / "dataset" / "single" / "intext.md",
    "Bibliography": ROOT / "dataset" / "single" / "bibliography.md",
}

ALLOWED_EXTENSION = ".md"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "single" / "standalone.md"
DEFAULT_JSON_OUTPUT_PATH = ROOT / "results" / "single" / "standalone.json"

RANKING_COLUMN = "InText"
CURLY_PREFIX_LABEL = "Curly Prefix"
CURLY_SHELL_LABEL = "Curly Shell"

# Export formats
EXPORT_FORMATS = {"markdown", "json"}


# ==========================================================================
# Tiktoken Encodings
# ==========================================================================

TIKTOKEN_ENCODINGS: dict[str, str] = {
    "cl100k_base": "cl100k_base",
    "o200k_base": "o200k_base",
}


# ==========================================================================
# Hugging Face Tokenizers
# ==========================================================================

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
        "trust_remote_code": False,
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
        "trust_remote_code": False,
    },
    "Mistral 7B Instruct v0.3": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
        "fix_mistral_regex": True,
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
        "fix_mistral_regex": True,
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


# ==========================================================================
# Data Structures
# ==========================================================================

@dataclass
class BenchmarkResult:
    """Result from a single tokenizer benchmark."""
    name: str
    category: str
    tokens: int
    status: str = "OK"
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class Statistics:
    """Statistical summary for a dataset."""
    mean: Optional[float] = None
    median: Optional[float] = None
    min: Optional[int] = None
    max: Optional[int] = None
    stdev: Optional[float] = None
    count: int = 0


@dataclass
class LineResult:
    """Result for a single citation line."""
    label: str
    content: str
    tokens: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


@dataclass
class FileInfo:
    """Aggregated benchmark results for a file."""
    display_name: str
    path: Path
    text: str
    char_count: int
    word_count: int
    line_count: int
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    line_results: list[LineResult] = field(default_factory=list)


# ==========================================================================
# Configuration Validation
# ==========================================================================

def validate_config() -> None:
    """Validate configuration at startup."""
    logger.info("Validating configuration...")

    if not FILES:
        raise ValueError("FILES is empty.")

    for display_name, path in FILES.items():
        if path.suffix.lower() != ALLOWED_EXTENSION:
            raise ValueError(
                f"'{display_name}' -> {path}: only {ALLOWED_EXTENSION} allowed."
            )

    if not TIKTOKEN_ENCODINGS and not HF_TOKENIZERS:
        raise ValueError("No tokenizers configured.")

    logger.info("✓ Configuration valid")


# ==========================================================================
# File Handling
# ==========================================================================

def read_markdown(path: Path) -> str:
    """Read markdown file with validation."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    return path.read_text(encoding="utf-8")


def parse_lines(text: str) -> list[tuple[str, str]]:
    """
    Parse citation format lines.
    """
    lines: list[tuple[str, str]] = []

    for raw in text.splitlines():
        stripped = raw.strip()

        if not stripped:
            continue

        if ":" in stripped:
            label, content = stripped.split(":", 1)
            lines.append((label.strip(), content.strip()))
        else:
            lines.append((stripped[:40], stripped))

    return lines


def load_files(files: dict[str, Path]) -> list[FileInfo]:
    """Load and parse all input files."""
    logger.info(f"Loading {len(files)} input file(s)...")

    infos: list[FileInfo] = []
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()

    for display_name, path in files.items():
        if display_name in seen_names:
            raise ValueError(f"Duplicate display name: '{display_name}'")

        seen_names.add(display_name)

        resolved = path.resolve()
        if resolved in seen_paths:
            logger.info(f"  (skipping duplicate path: {path})")
            continue

        seen_paths.add(resolved)

        text = read_markdown(path)

        if not text.strip():
            raise ValueError(f"{path} ('{display_name}') is empty.")

        lines = parse_lines(text)

        info = FileInfo(
            display_name=display_name,
            path=path,
            text=text,
            char_count=len(text),
            word_count=len(text.split()),
            line_count=len(lines),
        )

        for label, content in lines:
            info.line_results.append(LineResult(label=label, content=content))

        infos.append(info)

        logger.info(
            f"  ✓ {display_name}: "
            f"{info.char_count:,} chars, "
            f"{info.word_count:,} words, "
            f"{info.line_count} lines"
        )

    return infos


# ==========================================================================
# Tokenizer Backends
# ==========================================================================

def benchmark_tiktoken(text: str, name: str, encoding) -> BenchmarkResult:
    """Benchmark tiktoken encoding."""
    start = time.time()
    try:
        tokens = len(encoding.encode(text))
        duration = (time.time() - start) * 1000
        return BenchmarkResult(
            name=name,
            category="tiktoken",
            tokens=tokens,
            duration_ms=duration,
        )
    except Exception as exc:
        return BenchmarkResult(
            name=name,
            category="tiktoken",
            tokens=0,
            status="FAILED",
            error=str(exc),
        )


def count_hf_tokens(tokenizer, text: str) -> int:
    """Count tokens using Hugging Face tokenizer."""
    return len(
        tokenizer(text, add_special_tokens=False)["input_ids"]
    )


def benchmark_hf(text: str, name: str, tokenizer) -> BenchmarkResult:
    """Benchmark Hugging Face tokenizer."""
    start = time.time()
    try:
        tokens = count_hf_tokens(tokenizer, text)
        duration = (time.time() - start) * 1000
        return BenchmarkResult(
            name=name,
            category="huggingface",
            tokens=tokens,
            duration_ms=duration,
        )
    except Exception as exc:
        return BenchmarkResult(
            name=name,
            category="huggingface",
            tokens=0,
            status="FAILED",
            error=str(exc),
        )


# ==========================================================================
# Tokenizer Loading
# ==========================================================================

def load_all_tokenizers() -> tuple[dict[str, object], dict[str, object], list[tuple[str, str]]]:
    """Load all configured tokenizers."""
    logger.section("Loading Tokenizers")

    tiktoken_encodings: dict[str, object] = {}
    hf_load_failures: list[tuple[str, str]] = []

    logger.info(f"Tiktoken encodings: {len(TIKTOKEN_ENCODINGS)}")
    for display_name, encoding_name in TIKTOKEN_ENCODINGS.items():
        try:
            tiktoken_encodings[display_name] = tiktoken.get_encoding(encoding_name)
            logger.info(f"  ✓ {display_name}")
        except Exception as exc:
            logger.error(f"  ✗ {display_name}: {exc}")

    logger.info(f"Hugging Face tokenizers: {len(HF_TOKENIZERS)}")
    loaded_hf: dict[str, object] = {}
    for display_name, config in HF_TOKENIZERS.items():
        try:
            loaded_hf[display_name] = AutoTokenizer.from_pretrained(
                config["repo"],
                trust_remote_code=config["trust_remote_code"],
            )
            logger.info(f"  ✓ {display_name}")
        except Exception as exc:
            logger.error(f"  ✗ {display_name}: {exc}")
            hf_load_failures.append((display_name, str(exc)))

    return (
        tiktoken_encodings,
        loaded_hf,
        hf_load_failures,
    )


# ==========================================================================
# Statistics Computation
# ==========================================================================

def compute_statistics(values: list[int]) -> Statistics:
    """Compute statistics for a list of token counts."""
    if not values:
        return Statistics()

    sorted_values = sorted(values)
    stats = Statistics(
        mean=sum(values) / len(values),
        median=median(values),
        min=min(values),
        max=max(values),
        count=len(values),
    )

    if len(values) > 1:
        stats.stdev = stdev(values)

    return stats


def format_stats(stats: Statistics) -> str:
    """Format statistics for display."""
    if stats.count == 0:
        return "—"

    parts = [f"n={stats.count}"]
    if stats.mean is not None:
        parts.append(f"mean={stats.mean:.1f}")
    if stats.median is not None:
        parts.append(f"med={stats.median}")
    if stats.min is not None and stats.max is not None:
        parts.append(f"[{stats.min}–{stats.max}]")
    if stats.stdev is not None:
        parts.append(f"σ={stats.stdev:.1f}")

    return ", ".join(parts)


# ==========================================================================
# Benchmarking
# ==========================================================================

def benchmark_files(
    file_infos: list[FileInfo],
    tiktoken_encodings: dict[str, object],
    loaded_hf: dict[str, object],
) -> None:
    """Run benchmarks on all files and lines."""
    logger.section("Benchmarking")

    for info in file_infos:
        logger.info(f"\n{info.display_name}:")

        # Full file benchmarks
        for display_name, encoding in tiktoken_encodings.items():
            result = benchmark_tiktoken(info.text, display_name, encoding)
            info.results[display_name] = result
            logger.info(
                f"  {display_name}: {result.tokens:,} "
                f"({result.duration_ms:.1f}ms)"
            )

        for display_name, tokenizer in loaded_hf.items():
            result = benchmark_hf(info.text, display_name, tokenizer)
            info.results[display_name] = result
            logger.info(
                f"  {display_name}: {result.tokens:,} "
                f"({result.duration_ms:.1f}ms)"
            )

        # Individual line benchmarks
        logger.info(f"  Benchmarking {len(info.line_results)} lines...")
        for line_result in info.line_results:
            for display_name, encoding in tiktoken_encodings.items():
                try:
                    line_result.tokens[display_name] = len(
                        encoding.encode(line_result.content)
                    )
                except Exception as exc:
                    line_result.failures[display_name] = str(exc)

            for display_name, tokenizer in loaded_hf.items():
                try:
                    line_result.tokens[display_name] = count_hf_tokens(
                        tokenizer, line_result.content
                    )
                except Exception as exc:
                    line_result.failures[display_name] = str(exc)


# ==========================================================================
# Report Generation
# ==========================================================================

def render_markdown_report(
    output_path: Path,
    file_infos: list[FileInfo],
    all_tokenizer_names: list[str],
    hf_load_failures: list[tuple[str, str]],
) -> None:
    """Generate markdown report."""
    logger.info("Generating Markdown report...")

    out: list[str] = []
    out.append("# Citation Tokenizer Benchmark Report")
    out.append("")
    out.append(
        f"Generated: {datetime.now(timezone.utc).isoformat()}"
    )
    out.append("")

    intext_info = next(
        (info for info in file_infos if info.display_name == "InText"),
        None,
    )
    bibliography_info = next(
        (info for info in file_infos if info.display_name == "Bibliography"),
        None,
    )

    if intext_info is None or bibliography_info is None:
        raise ValueError("InText and Bibliography datasets are required.")

    # Filter tokenizers with complete results
    tokenizer_names = []
    for name in all_tokenizer_names:
        intext_result = intext_info.results.get(name)
        bibliography_result = bibliography_info.results.get(name)

        if (
            intext_result
            and bibliography_result
            and intext_result.status == "OK"
            and bibliography_result.status == "OK"
        ):
            tokenizer_names.append(name)

    out.append(f"## Summary")
    out.append("")
    out.append(f"**Successful tokenizers:** {len(tokenizer_names)} / {len(all_tokenizer_names)}")
    out.append(f"**InText formats:** {intext_info.line_count}")
    out.append(f"**Bibliography formats:** {bibliography_info.line_count}")
    out.append("")

    if hf_load_failures:
        out.append("### Load Failures")
        out.append("")
        for name, error in hf_load_failures:
            out.append(f"- **{name}**: {error}")
        out.append("")

    # In-Text table
    out.append("## In-Text Citation Token Counts")
    out.append("")

    intext_rows = []
    for line_result in intext_info.line_results:
        values = {
            tokenizer: line_result.tokens.get(tokenizer, 0)
            for tokenizer in tokenizer_names
        }
        intext_rows.append((line_result.label, values))

    intext_rows.sort(key=lambda x: sum(x[1].values()))

    header = ["Rank", "Format"] + tokenizer_names
    out.append("| " + " | ".join(header) + " |")
    out.append(
        "| " + " | ".join(["---:"] * len(header)) + " |"
    )

    for rank, (label, values) in enumerate(intext_rows, start=1):
        row = [str(rank), label] + [
            f"{values.get(tokenizer, 0):,}" for tokenizer in tokenizer_names
        ]
        out.append("| " + " | ".join(row) + " |")

    out.append("")

    # Bibliography table
    out.append("## Bibliography Token Counts")
    out.append("")

    bibliography_rows = []
    for line_result in bibliography_info.line_results:
        values = {
            tokenizer: line_result.tokens.get(tokenizer, 0)
            for tokenizer in tokenizer_names
        }
        bibliography_rows.append((line_result.label, values))

    bibliography_rows.sort(key=lambda x: sum(x[1].values()))

    header = ["Rank", "Format"] + tokenizer_names
    out.append("| " + " | ".join(header) + " |")
    out.append(
        "| " + " | ".join(["---:"] * len(header)) + " |"
    )

    for rank, (label, values) in enumerate(bibliography_rows, start=1):
        row = [str(rank), label] + [
            f"{values.get(tokenizer, 0):,}" for tokenizer in tokenizer_names
        ]
        out.append("| " + " | ".join(row) + " |")

    out.append("")

    # Combined (InText + Bibliography)
    out.append("## Combined Citation Cost (InText + Bibliography)")
    out.append("")

    bibliography_by_label = {
        lr.label: lr for lr in bibliography_info.line_results
    }

    combined_rows = []
    # Only combine formats that have exact matching labels in both InText and Bibliography
    for intext_lr in intext_info.line_results:
        # Only include if this format exists with the same name in bibliography
        bibliography_lr = bibliography_by_label.get(intext_lr.label)
        if bibliography_lr is None:
            continue

        values = {}
        for tokenizer in tokenizer_names:
            # A tokenizer can succeed on the full file (and so pass the
            # `tokenizer_names` filter above) but still fail, or simply
            # have no entry, for one specific line. Checking membership
            # in *this line's own* tokens/failures dicts (rather than
            # relying on the file-level `tokenizer_names` filter) is what
            # makes the sum actually fire for every tokenizer that has
            # real data on both sides of this specific row.
            intext_ok = (
                tokenizer in intext_lr.tokens
                and tokenizer not in intext_lr.failures
            )
            bibliography_ok = (
                tokenizer in bibliography_lr.tokens
                and tokenizer not in bibliography_lr.failures
            )

            if intext_ok and bibliography_ok:
                values[tokenizer] = (
                    intext_lr.tokens[tokenizer]
                    + bibliography_lr.tokens[tokenizer]
                )

        combined_rows.append((intext_lr.label, values))

    combined_rows.sort(key=lambda x: sum(x[1].values()))

    header = ["Rank", "Format"] + tokenizer_names
    out.append("| " + " | ".join(header) + " |")
    out.append(
        "| " + " | ".join(["---:"] * len(header)) + " |"
    )

    for rank, (label, values) in enumerate(combined_rows, start=1):
        row = [str(rank), label] + [
            f"{values.get(tokenizer, 0):,}" for tokenizer in tokenizer_names
        ]
        out.append("| " + " | ".join(row) + " |")

    out.append("")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out), encoding="utf-8")
    logger.info(f"✓ Markdown report: {output_path}")


def render_json_report(
    output_path: Path,
    file_infos: list[FileInfo],
    all_tokenizer_names: list[str],
) -> None:
    """Generate JSON report for further analysis."""
    logger.info("Generating JSON report...")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "summary": {
            "total_tokenizers": len(all_tokenizer_names),
            "successful_tokenizers": 0,
        },
    }

    for info in file_infos:
        file_data = {
            "name": info.display_name,
            "path": str(info.path),
            "char_count": info.char_count,
            "word_count": info.word_count,
            "line_count": info.line_count,
            "full_file_results": {},
            "line_results": [],
        }

        # Full file results
        for tokenizer_name, result in info.results.items():
            file_data["full_file_results"][tokenizer_name] = {
                "tokens": result.tokens,
                "status": result.status,
                "category": result.category,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }

        # Line results
        for line_result in info.line_results:
            file_data["line_results"].append({
                "label": line_result.label,
                "content_length": len(line_result.content),
                "tokens": line_result.tokens,
                "failures": line_result.failures,
            })

        report["files"].append(file_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"✓ JSON report: {output_path}")


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    """Main benchmark execution."""
    logger.section("Citation Tokenizer Benchmark")
    try:
        validate_config()

        file_infos = load_files(FILES)

        (
            tiktoken_encodings,
            loaded_hf,
            hf_load_failures,
        ) = load_all_tokenizers()

        all_tokenizer_names: list[str] = (
            list(tiktoken_encodings.keys())
            + list(HF_TOKENIZERS.keys())
        )

        benchmark_files(
            file_infos,
            tiktoken_encodings,
            loaded_hf,
        )

        render_markdown_report(
            DEFAULT_OUTPUT_PATH,
            file_infos,
            all_tokenizer_names,
            hf_load_failures,
        )

        render_json_report(
            DEFAULT_JSON_OUTPUT_PATH,
            file_infos,
            all_tokenizer_names,
        )

        logger.section("Complete")
        logger.info(f"Execution time: {logger.elapsed():.2f}s")

    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
        raise


if __name__ == "__main__":
    main()