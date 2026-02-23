from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Sequence

import weave
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from wandbot.evaluation.eval_config import EvalConfig
from wandbot.utils import get_logger

logger = get_logger(__name__)

# Load environment variables from .env in project root.
ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH, override=True)

DEFAULT_REPOS: Dict[str, str] = {
    "wandb_docs": "https://github.com/wandb/docs.git",
    "wandb_sdk": "https://github.com/wandb/wandb.git",
    "wandb_weave": "https://github.com/wandb/weave.git",
    "wandb_examples": "https://github.com/wandb/examples.git",
}

DEFAULT_REPO_CANDIDATES: Dict[str, Sequence[Path]] = {
    "wandb_docs": (
        Path("data/cache/raw_data/english_documentation/docs"),
        Path("data/cache/raw_data/wandb_documentation_en/docs"),
    ),
    "wandb_sdk": (
        Path("data/cache/raw_data/wandb_sdk_code/wandb"),
        Path("data/cache/raw_data/wandb_documentation/wandb"),
    ),
    "wandb_weave": (
        Path("data/cache/raw_data/weave_sdk_code/weave"),
        Path("data/cache/raw_data/weave_documentation/weave"),
    ),
    "wandb_examples": (
        Path("data/cache/raw_data/examples_code/examples"),
        Path("data/cache/raw_data/examples_notebooks/examples"),
    ),
}

CURRENT_DATE = datetime.now(timezone.utc).date().isoformat()

STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "answer",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "could",
    "doing",
    "does",
    "down",
    "during",
    "each",
    "from",
    "further",
    "have",
    "having",
    "here",
    "into",
    "itself",
    "just",
    "more",
    "most",
    "other",
    "over",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}

RG_GLOBS = (
    "*.md",
    "*.mdx",
    "*.rst",
    "*.txt",
    "*.py",
    "*.ipynb",
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
)


class ReplacementPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Proposed replacement question.")
    reference_answer: str = Field(description="Proposed replacement reference answer.")
    rationale: str = Field(description="Why this replacement should be used.")
    source_links: list[str] = Field(
        default_factory=list,
        description="Ground-truth links used for the replacement pair.",
    )


class StalenessAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_status: Literal["fresh", "partially_correct", "stale", "needs_clarification"]
    question_status: Literal["fresh", "stale", "needs_clarification"]
    answer_status: Literal["correct", "partially_correct", "stale", "needs_clarification"]
    confidence: float = Field(ge=0, le=1)
    reason_summary: str
    stale_signals: list[str] = Field(default_factory=list)
    recommended_edits: list[str] = Field(default_factory=list)
    needs_new_question_answer_pair: bool
    replacement_pair: ReplacementPair | None = None
    source_links: list[str] = Field(default_factory=list)
    ground_truth_links: list[str] = Field(default_factory=list)


class LocalEvidenceMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    path: str
    line: int
    snippet: str


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    question: str
    reference_answer: str
    notes: str | None = None
    assessment: StalenessAssessment | None = None
    local_evidence: list[LocalEvidenceMatch] = Field(default_factory=list)
    web_search_sources: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at_utc: str


STALENESS_SCHEMA = StalenessAssessment.model_json_schema()

# Codex `--output-schema` currently expects every property to be listed in `required`.
CODEX_STALENESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_status": {
            "type": "string",
            "enum": ["fresh", "partially_correct", "stale", "needs_clarification"],
        },
        "question_status": {
            "type": "string",
            "enum": ["fresh", "stale", "needs_clarification"],
        },
        "answer_status": {
            "type": "string",
            "enum": ["correct", "partially_correct", "stale", "needs_clarification"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_summary": {"type": "string"},
        "stale_signals": {"type": "array", "items": {"type": "string"}},
        "recommended_edits": {"type": "array", "items": {"type": "string"}},
        "needs_new_question_answer_pair": {"type": "boolean"},
        "replacement_pair": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string"},
                        "reference_answer": {"type": "string"},
                        "rationale": {"type": "string"},
                        "source_links": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question", "reference_answer", "rationale", "source_links"],
                },
                {"type": "null"},
            ]
        },
        "source_links": {"type": "array", "items": {"type": "string"}},
        "ground_truth_links": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall_status",
        "question_status",
        "answer_status",
        "confidence",
        "reason_summary",
        "stale_signals",
        "recommended_edits",
        "needs_new_question_answer_pair",
        "replacement_pair",
        "source_links",
        "ground_truth_links",
    ],
}

SYSTEM_PROMPT = f"""You are curating the Weights & Biases evaluation dataset.
Today's date is {CURRENT_DATE}.

Your task:
1) Decide whether the QUESTION is stale.
2) Decide whether the REFERENCE ANSWER is stale/partial/correct.
3) Return a strict JSON object that matches the provided schema.

Rules:
- Perform extensive web search with the provided tool before finalizing.
- Explicitly verify against:
  - docs.wandb.ai
  - github.com/wandb/wandb
  - github.com/wandb/weave
  - github.com/wandb/examples
- Treat official W&B docs and official W&B repos as highest-trust sources.
- If evidence conflicts or is incomplete, mark `needs_clarification`.
- Use `stale` when functionality changed, deprecations happened, references are obsolete, or wording implies outdated product behavior.
- Include direct links in `source_links` and `ground_truth_links`.
- If stale or materially incomplete, propose a replacement question/reference pair.
- Return JSON only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit eval question/reference pairs for staleness with one Codex headless "
            "agent per row, plus local W&B repo search, and emit JSONL + markdown reports."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=EvalConfig().eval_dataset,
        help="Dataset path (.json/.jsonl/.csv) or weave URI (weave://...).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("artifacts/eval_staleness/staleness_audit.jsonl"),
        help="Path to write per-sample structured JSONL report.",
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        default=Path("artifacts/eval_staleness/staleness_summary.md"),
        help="Path to write markdown summary report.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of rows to audit.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of parallel LLM audits.",
    )
    parser.add_argument(
        "--rg-parallelism",
        type=int,
        default=10,
        help="Maximum parallel local ripgrep searches.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Model to use for each Codex headless agent.",
    )
    parser.add_argument(
        "--repos-root",
        type=Path,
        default=Path("data/eval_staleness/repos"),
        help="Root directory for local clones of docs/sdk/weave/examples.",
    )
    parser.add_argument(
        "--sync-repos",
        action="store_true",
        default=True,
        help="Ensure local clones for wandb/docs, wandb/wandb, wandb/weave, wandb/examples.",
    )
    parser.add_argument(
        "--no-sync-repos",
        dest="sync_repos",
        action="store_false",
        help="Do not clone/pull required local repos.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root passed to each Codex headless run via --cd.",
    )
    parser.add_argument(
        "--codex-web-search",
        action="store_true",
        default=True,
        help="Enable Codex web search tool (`codex --search exec`).",
    )
    parser.add_argument(
        "--no-codex-web-search",
        dest="codex_web_search",
        action="store_false",
        help="Disable Codex web search tool and rely on local repositories only.",
    )
    parser.add_argument(
        "--local-repo-path",
        action="append",
        default=[],
        help="Additional local repo/document root paths to search with rg.",
    )
    parser.add_argument(
        "--disable-local-search",
        action="store_true",
        help="Disable local repo pre-search and rely on Codex agent search behavior.",
    )
    return parser.parse_args()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_dataset_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question = _safe_str(row.get("question") or row.get("query") or row.get("prompt"))
        reference_answer = _safe_str(
            row.get("answer") or row.get("ground_truth") or row.get("reference_answer")
        )
        if not question or not reference_answer:
            continue

        item_id = _safe_str(
            row.get("id")
            or row.get("row_id")
            or row.get("question_id")
            or row.get("uuid")
            or idx
        )

        normalized.append(
            {
                "item_id": item_id,
                "question": question,
                "reference_answer": reference_answer,
                "notes": _safe_str(row.get("notes")) or None,
                "raw_row": row,
            }
        )
    return normalized


def load_rows_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload["rows"]
        raise ValueError(f"Unsupported JSON structure in {path}")

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_dataset(dataset: str) -> list[dict[str, Any]]:
    if dataset.startswith("weave:///"):
        logger.info("Loading dataset from weave ref: %s", dataset)
        match = re.match(r"^weave:///([^/]+)/([^/]+)/", dataset)
        if match:
            weave_project = f"{match.group(1)}/{match.group(2)}"
            weave.init(weave_project)
        else:
            logger.warning("Could not parse project from weave URI: %s", dataset)
        dataset_ref = weave.ref(dataset).get()
        rows = list(dataset_ref.rows)
        return normalize_dataset_rows(rows)

    dataset_path = Path(dataset)
    logger.info("Loading dataset from file: %s", dataset_path)
    return normalize_dataset_rows(load_rows_from_file(dataset_path))


def extract_search_terms(question: str, reference_answer: str, max_terms: int = 10) -> list[str]:
    compact_question = " ".join(question.split())
    terms: list[str] = []
    seen: set[str] = set()

    if compact_question:
        trimmed = compact_question[:160]
        seen.add(trimmed.lower())
        terms.append(trimmed)

    token_counts: Counter[str] = Counter()
    for source in (question, reference_answer):
        for token in re.findall(r"[A-Za-z0-9_./-]{4,}", source.lower()):
            if token in STOPWORDS or token.isdigit():
                continue
            token_counts[token] += 1

    for token, _count in token_counts.most_common(max_terms * 3):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms + 1:
            break
    return terms


def discover_repo_paths() -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for repo_name, candidates in DEFAULT_REPO_CANDIDATES.items():
        for candidate in candidates:
            if candidate.exists():
                discovered[repo_name] = candidate
                break
    return discovered


async def run_cmd(cmd: Sequence[str], cwd: Path | None = None) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    return process.returncode, stdout_bytes.decode("utf-8", errors="replace"), stderr_bytes.decode(
        "utf-8",
        errors="replace",
    )


async def sync_default_repos(repos_root: Path) -> dict[str, Path]:
    repos_root.mkdir(parents=True, exist_ok=True)
    repo_paths: dict[str, Path] = {}

    for repo_name, repo_url in DEFAULT_REPOS.items():
        target_path = repos_root / repo_name
        if target_path.exists() and (target_path / ".git").exists():
            logger.info("Pulling latest for %s at %s", repo_name, target_path)
            code, _out, err = await run_cmd(["git", "-C", str(target_path), "pull", "--ff-only"])
            if code != 0:
                logger.warning("Failed to pull %s: %s", repo_name, err.strip())
            repo_paths[repo_name] = target_path
            continue

        if target_path.exists():
            logger.info("Using existing non-git path for %s: %s", repo_name, target_path)
            repo_paths[repo_name] = target_path
            continue

        logger.info("Cloning %s into %s", repo_url, target_path)
        code, _out, err = await run_cmd(["git", "clone", "--depth", "1", repo_url, str(target_path)])
        if code != 0:
            logger.warning("Failed to clone %s: %s", repo_name, err.strip())
            continue
        repo_paths[repo_name] = target_path

    return repo_paths


async def ensure_required_repo_paths(repos_root: Path, sync_repos: bool) -> dict[str, Path]:
    discovered = discover_repo_paths()
    synced: dict[str, Path] = {}
    if sync_repos:
        synced = await sync_default_repos(repos_root)

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for repo_name in DEFAULT_REPOS:
        if repo_name in synced and synced[repo_name].exists():
            resolved[repo_name] = synced[repo_name]
            continue
        if repo_name in discovered and discovered[repo_name].exists():
            resolved[repo_name] = discovered[repo_name]
            continue
        missing.append(repo_name)

    if missing:
        raise ValueError(
            "Missing required local repos: "
            + ", ".join(missing)
            + ". Re-run with network access and --sync-repos to clone them."
        )
    return resolved


def _parse_rg_line(raw_line: str, repo_name: str) -> LocalEvidenceMatch | None:
    raw_line = raw_line.strip()
    if not raw_line:
        return None
    parts = raw_line.split(":", 2)
    if len(parts) != 3:
        return None
    rel_path, line_no_text, snippet = parts
    try:
        line_no = int(line_no_text)
    except ValueError:
        return None
    return LocalEvidenceMatch(
        repo=repo_name,
        path=str(Path(rel_path)),
        line=line_no,
        snippet=snippet.strip(),
    )


async def search_repo_with_rg(
    repo_name: str,
    repo_root: Path,
    terms: list[str],
    rg_semaphore: asyncio.Semaphore,
    max_matches: int,
) -> list[LocalEvidenceMatch]:
    if not terms:
        return []
    if not repo_root.exists():
        return []

    cmd: list[str] = [
        "rg",
        "--no-heading",
        "--line-number",
        "--ignore-case",
        "--fixed-strings",
        "--max-count",
        "2",
        "--max-columns",
        "400",
        "--max-columns-preview",
    ]
    for glob in RG_GLOBS:
        cmd.extend(["-g", glob])
    for term in terms[:10]:
        cmd.extend(["-e", term])
    cmd.append(".")

    async with rg_semaphore:
        code, stdout, stderr = await run_cmd(cmd, cwd=repo_root)
    if code not in (0, 1):
        logger.warning("rg failed for %s: %s", repo_name, stderr.strip())
        return []

    parsed: list[LocalEvidenceMatch] = []
    for line in stdout.splitlines():
        match = _parse_rg_line(line, repo_name=repo_name)
        if match is None:
            continue
        parsed.append(match)
        if len(parsed) >= max_matches:
            break
    return parsed


async def collect_local_evidence(
    question: str,
    reference_answer: str,
    repo_paths: dict[str, Path],
    rg_semaphore: asyncio.Semaphore,
    max_matches_per_repo: int = 8,
) -> list[LocalEvidenceMatch]:
    if not repo_paths:
        return []
    terms = extract_search_terms(question, reference_answer)
    if not terms:
        return []
    tasks = [
        search_repo_with_rg(
            repo_name=repo_name,
            repo_root=repo_path,
            terms=terms,
            rg_semaphore=rg_semaphore,
            max_matches=max_matches_per_repo,
        )
        for repo_name, repo_path in repo_paths.items()
    ]
    evidence_batches = await asyncio.gather(*tasks)
    flattened = [match for batch in evidence_batches for match in batch]
    return flattened[: max_matches_per_repo * max(1, len(repo_paths))]


def format_local_evidence_for_prompt(local_evidence: list[LocalEvidenceMatch], max_lines: int = 24) -> str:
    if not local_evidence:
        return "No local repo evidence was found."
    lines = []
    for match in local_evidence[:max_lines]:
        snippet = match.snippet.replace("\n", " ").strip()
        lines.append(f"- [{match.repo}] {match.path}:{match.line} :: {snippet}")
    return "\n".join(lines)


def sanitize_for_markdown_cell(value: str, max_len: int = 180) -> str:
    compact = " ".join(value.split())
    compact = compact.replace("|", "\\|")
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _extract_first_json_object(raw_text: str) -> str:
    candidate = raw_text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in Codex output.")
    return candidate[start : end + 1]


def build_codex_agent_prompt(
    question: str,
    reference_answer: str,
    notes: str | None,
    local_evidence: list[LocalEvidenceMatch],
    repo_paths: dict[str, Path],
) -> str:
    repo_lines = "\n".join(f"- {name}: {path}" for name, path in repo_paths.items())
    local_evidence_block = format_local_evidence_for_prompt(local_evidence)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "You are running inside a local workspace with required repositories already available.\n"
        "Use shell commands (prefer rg) to search these local repositories first, then continue with web search for updates.\n"
        "Required local repositories:\n"
        f"{repo_lines}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Notes:\n{notes or 'None'}\n\n"
        "Pre-collected local evidence (you can ignore if you find stronger evidence):\n"
        f"{local_evidence_block}\n\n"
        "Return only JSON matching the provided output schema."
    )


async def run_codex_assessment(
    question: str,
    reference_answer: str,
    notes: str | None,
    local_evidence: list[LocalEvidenceMatch],
    repo_paths: dict[str, Path],
    model: str,
    workspace_root: Path,
    codex_web_search: bool,
    output_schema_path: Path,
) -> StalenessAssessment:
    prompt = build_codex_agent_prompt(
        question=question,
        reference_answer=reference_answer,
        notes=notes,
        local_evidence=local_evidence,
        repo_paths=repo_paths,
    )

    output_file = Path(tempfile.gettempdir()) / f"codex_stale_eval_{uuid.uuid4().hex}.json"
    cmd: list[str] = ["codex"]
    if codex_web_search:
        cmd.append("--search")
    cmd.extend(
        [
            "exec",
            "--full-auto",
            "--cd",
            str(workspace_root),
            "--model",
            model,
            "--color",
            "never",
            "--output-schema",
            str(output_schema_path),
            "--output-last-message",
            str(output_file),
            "-",
        ]
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(workspace_root),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate(prompt.encode("utf-8"))
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    if process.returncode != 0:
        raise ValueError(
            f"Codex agent failed with exit code {process.returncode}. "
            f"stderr={stderr_text[-2000:]}"
        )

    raw_output = ""
    if output_file.exists():
        raw_output = output_file.read_text(encoding="utf-8").strip()
        output_file.unlink(missing_ok=True)
    if not raw_output:
        raw_output = stdout_text.strip()
    if not raw_output:
        raise ValueError("Codex agent returned empty output.")

    json_payload = _extract_first_json_object(raw_output)
    assessment = StalenessAssessment.model_validate_json(json_payload)
    return assessment


async def audit_single_row(
    row: dict[str, Any],
    codex_semaphore: asyncio.Semaphore,
    rg_semaphore: asyncio.Semaphore,
    repo_paths: dict[str, Path],
    args: argparse.Namespace,
    output_schema_path: Path,
) -> AuditRecord:
    local_evidence: list[LocalEvidenceMatch] = []
    now = datetime.now(timezone.utc).isoformat()

    try:
        if not args.disable_local_search:
            local_evidence = await collect_local_evidence(
                question=row["question"],
                reference_answer=row["reference_answer"],
                repo_paths=repo_paths,
                rg_semaphore=rg_semaphore,
            )

        async with codex_semaphore:
            assessment = await run_codex_assessment(
                question=row["question"],
                reference_answer=row["reference_answer"],
                notes=row.get("notes"),
                local_evidence=local_evidence,
                repo_paths=repo_paths,
                model=args.model,
                workspace_root=args.workspace_root,
                codex_web_search=args.codex_web_search,
                output_schema_path=output_schema_path,
            )

        return AuditRecord(
            item_id=row["item_id"],
            question=row["question"],
            reference_answer=row["reference_answer"],
            notes=row.get("notes"),
            assessment=assessment,
            local_evidence=local_evidence,
            web_search_sources=assessment.source_links,
            created_at_utc=now,
        )
    except Exception as exc:
        return AuditRecord(
            item_id=row["item_id"],
            question=row["question"],
            reference_answer=row["reference_answer"],
            notes=row.get("notes"),
            local_evidence=local_evidence,
            error=str(exc),
            created_at_utc=now,
        )


def build_summary_markdown(
    records: Sequence[AuditRecord],
    dataset: str,
    model: str,
    concurrency: int,
) -> str:
    completed = [record for record in records if record.assessment is not None]
    failed = [record for record in records if record.assessment is None]

    stale_questions = sum(1 for record in completed if record.assessment and record.assessment.question_status == "stale")
    stale_answers = sum(1 for record in completed if record.assessment and record.assessment.answer_status == "stale")
    partially_correct = sum(
        1 for record in completed if record.assessment and record.assessment.answer_status == "partially_correct"
    )
    needs_clarification = sum(
        1 for record in completed if record.assessment and record.assessment.overall_status == "needs_clarification"
    )
    fresh = sum(1 for record in completed if record.assessment and record.assessment.overall_status == "fresh")

    recommendations = [
        record
        for record in completed
        if record.assessment
        and record.assessment.needs_new_question_answer_pair
        and record.assessment.replacement_pair is not None
    ]

    lines: list[str] = []
    lines.append("# Evaluation Dataset Staleness Audit")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Dataset: `{dataset}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Concurrency: `{concurrency}`")
    lines.append(f"- Total rows processed: `{len(records)}`")
    lines.append(f"- Successful rows: `{len(completed)}`")
    lines.append(f"- Failed rows: `{len(failed)}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Fresh: `{fresh}`")
    lines.append(f"- Stale questions: `{stale_questions}`")
    lines.append(f"- Stale answers: `{stale_answers}`")
    lines.append(f"- Partially correct answers: `{partially_correct}`")
    lines.append(f"- Needs clarification: `{needs_clarification}`")
    lines.append("")

    if completed:
        lines.append("## Row Results")
        lines.append("")
        lines.append("| item_id | overall | question | answer | confidence | summary |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for record in completed:
            assessment = record.assessment
            if assessment is None:
                continue
            lines.append(
                "| "
                + f"{sanitize_for_markdown_cell(record.item_id, 40)} | "
                + f"{assessment.overall_status} | "
                + f"{assessment.question_status} | "
                + f"{assessment.answer_status} | "
                + f"{assessment.confidence:.2f} | "
                + f"{sanitize_for_markdown_cell(assessment.reason_summary)} |"
            )
        lines.append("")

    if recommendations:
        lines.append("## Recommended New Question/Reference Pairs")
        lines.append("")
        for record in recommendations:
            assessment = record.assessment
            replacement = assessment.replacement_pair if assessment else None
            if replacement is None:
                continue
            lines.append(f"### {record.item_id}")
            lines.append("")
            lines.append(f"- Proposed question: {replacement.question}")
            lines.append(f"- Proposed reference answer: {replacement.reference_answer}")
            lines.append(f"- Rationale: {replacement.rationale}")
            if replacement.source_links:
                lines.append("- Source links:")
                for link in replacement.source_links:
                    lines.append(f"  - {link}")
            lines.append("")

    if failed:
        lines.append("## Failures")
        lines.append("")
        for record in failed:
            lines.append(f"- `{record.item_id}`: {record.error or 'Unknown error'}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_jsonl(records: Sequence[AuditRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def write_summary(summary_text: str, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")


async def run_audit(args: argparse.Namespace) -> tuple[list[AuditRecord], dict[str, Path]]:
    rows = load_dataset(args.dataset)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if not rows:
        raise ValueError("No rows found after dataset normalization.")

    repo_paths = await ensure_required_repo_paths(args.repos_root, sync_repos=args.sync_repos)
    for extra_path in args.local_repo_path:
        path_obj = Path(extra_path)
        if path_obj.exists():
            repo_paths[path_obj.name] = path_obj
        else:
            logger.warning("Skipping missing --local-repo-path: %s", path_obj)

    if args.disable_local_search:
        repo_paths = {}
    workspace_root = args.workspace_root.resolve()
    if not workspace_root.exists():
        raise ValueError(f"--workspace-root does not exist: {workspace_root}")
    args.workspace_root = workspace_root

    codex_semaphore = asyncio.Semaphore(max(1, args.concurrency))
    rg_semaphore = asyncio.Semaphore(max(1, args.rg_parallelism))
    output_schema_path = args.output_jsonl.parent / "staleness_assessment.schema.json"
    output_schema_path.parent.mkdir(parents=True, exist_ok=True)
    output_schema_path.write_text(json.dumps(CODEX_STALENESS_SCHEMA, indent=2), encoding="utf-8")

    logger.info("Starting staleness audit for %d rows with concurrency=%d", len(rows), args.concurrency)
    if repo_paths:
        logger.info("Local search roots: %s", ", ".join(str(path) for path in repo_paths.values()))
    else:
        logger.info("Local search disabled or no repos found.")

    tasks = [
        audit_single_row(
            row=row,
            codex_semaphore=codex_semaphore,
            rg_semaphore=rg_semaphore,
            repo_paths=repo_paths,
            args=args,
            output_schema_path=output_schema_path,
        )
        for row in rows
    ]

    records: list[AuditRecord] = []
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        records.append(result)
        completed += 1
        if completed % 10 == 0 or completed == len(tasks):
            logger.info("Completed %d / %d", completed, len(tasks))

    records.sort(key=lambda item: item.item_id)
    return records, repo_paths


def main() -> None:
    args = parse_args()
    records, _repo_paths = asyncio.run(run_audit(args))

    write_jsonl(records, args.output_jsonl)
    summary_markdown = build_summary_markdown(
        records=records,
        dataset=args.dataset,
        model=args.model,
        concurrency=args.concurrency,
    )
    write_summary(summary_markdown, args.summary_markdown)

    logger.info("Wrote JSONL report to %s", args.output_jsonl)
    logger.info("Wrote markdown summary to %s", args.summary_markdown)


if __name__ == "__main__":
    main()
