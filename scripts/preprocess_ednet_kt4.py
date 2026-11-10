#!/usr/bin/env python
"""Build an offline-RL subset from EdNet KT4.

The first experiment uses question-response events because they provide a clear
reward through the released answer key in EdNet Contents.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class LearnerStats:
    attempts: int = 0
    correct: int = 0
    prev_part: int = 0
    prev_correct: int = 2
    prev_timestamp: Optional[int] = None
    prev_source_bucket: int = 0
    prev_platform_bucket: int = 0


SOURCE_BUCKETS = {
    "unknown": 0,
    "diagnosis": 1,
    "sprint": 2,
    "my_note": 3,
    "adaptive_offer": 4,
    "review_quiz": 5,
    "review": 6,
    "tutor": 7,
}

PLATFORM_BUCKETS = {
    "unknown": 0,
    "mobile": 1,
    "web": 2,
    "tablet": 3,
}


def attempt_bin(attempts: int) -> int:
    if attempts <= 0:
        return 0
    if attempts <= 4:
        return 1
    if attempts <= 14:
        return 2
    if attempts <= 49:
        return 3
    return 4


def accuracy_bin(attempts: int, correct: int) -> int:
    if attempts <= 0:
        return 0
    acc = correct / attempts
    return min(4, 1 + int(acc * 4))


def gap_bin(prev_timestamp: Optional[int], timestamp: int) -> int:
    if prev_timestamp is None:
        return 0
    gap_ms = max(0, timestamp - prev_timestamp)
    if gap_ms < 60_000:
        return 1
    if gap_ms < 10 * 60_000:
        return 2
    if gap_ms < 24 * 60 * 60_000:
        return 3
    return 4


def source_bucket(source: str) -> int:
    key = (source or "unknown").strip() or "unknown"
    return SOURCE_BUCKETS.get(key, SOURCE_BUCKETS["unknown"])


def platform_bucket(platform: str) -> int:
    key = (platform or "unknown").strip() or "unknown"
    return PLATFORM_BUCKETS.get(key, PLATFORM_BUCKETS["unknown"])


def state_id(stats: LearnerStats, timestamp: int, state_variant: str) -> str:
    base = (
        f"a{attempt_bin(stats.attempts)}"
        f"_c{accuracy_bin(stats.attempts, stats.correct)}"
        f"_p{stats.prev_part}"
        f"_r{stats.prev_correct}"
        f"_g{gap_bin(stats.prev_timestamp, timestamp)}"
    )
    if state_variant == "extended":
        return f"{base}_s{stats.prev_source_bucket}_m{stats.prev_platform_bucket}"
    return base


def action_id(part: int, source: str, action_space: str) -> int:
    if action_space == "part_source":
        return (part - 1) * len(SOURCE_BUCKETS) + source_bucket(source) + 1
    return part


def reward_value(
    is_correct: int,
    reward_correct: float,
    reward_incorrect: float,
    reward_mode: str,
    current_gap_bin: int,
) -> float:
    reward = reward_correct if is_correct else reward_incorrect
    if reward_mode == "correctness_time_penalty":
        reward -= 0.05 * max(0, current_gap_bin - 1)
    return reward


def load_question_metadata(contents_zip: Path) -> Dict[str, Dict[str, str]]:
    question_meta: Dict[str, Dict[str, str]] = {}
    with zipfile.ZipFile(contents_zip) as zf:
        with zf.open("contents/questions.csv") as raw:
            reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8"))
            for row in reader:
                qid = row["question_id"]
                question_meta[qid] = {
                    "part": row["part"],
                    "correct_answer": row["correct_answer"],
                    "bundle_id": row["bundle_id"],
                    "explanation_id": row["explanation_id"],
                    "tags": row["tags"],
                }
    return question_meta


def iter_user_csv_names(kt4_zip: Path) -> Iterable[str]:
    with zipfile.ZipFile(kt4_zip) as zf:
        for name in zf.namelist():
            if name.startswith("KT4/") and name.endswith(".csv"):
                yield name


def read_user_rows(zf: zipfile.ZipFile, name: str) -> List[dict]:
    with zf.open(name) as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8"))
        rows = list(reader)
    rows.sort(key=lambda row: int(row["timestamp"] or 0))
    return rows


def build_user_transitions(
    user_id: str,
    rows: List[dict],
    question_meta: Dict[str, Dict[str, str]],
    reward_correct: float,
    reward_incorrect: float,
    rtg_gamma: float,
    action_space: str,
    state_variant: str,
    reward_mode: str,
) -> List[dict]:
    stats = LearnerStats()
    transitions: List[dict] = []

    for row in rows:
        if row.get("action_type") != "respond":
            continue
        item_id = row.get("item_id", "")
        answer = row.get("user_answer", "")
        if not item_id.startswith("q") or not answer:
            continue
        meta = question_meta.get(item_id)
        if not meta:
            continue
        try:
            timestamp = int(row["timestamp"])
            part = int(meta["part"])
        except (TypeError, ValueError):
            continue
        if part < 1 or part > 7:
            continue

        current_gap_bin = gap_bin(stats.prev_timestamp, timestamp)
        before_state = state_id(stats, timestamp, state_variant)
        is_correct = int(answer == meta["correct_answer"])
        reward = reward_value(
            is_correct=is_correct,
            reward_correct=reward_correct,
            reward_incorrect=reward_incorrect,
            reward_mode=reward_mode,
            current_gap_bin=current_gap_bin,
        )
        action = action_id(part, row.get("source", ""), action_space)

        stats.attempts += 1
        stats.correct += is_correct
        stats.prev_part = part
        stats.prev_correct = is_correct
        stats.prev_timestamp = timestamp
        stats.prev_source_bucket = source_bucket(row.get("source", ""))
        stats.prev_platform_bucket = platform_bucket(row.get("platform", ""))

        after_state = state_id(stats, timestamp, state_variant)
        transitions.append(
            {
                "user_id": user_id,
                "timestep": len(transitions),
                "timestamp": timestamp,
                "state": before_state,
                "action": action,
                "reward": reward,
                "next_state": after_state,
                "done": 0,
                "question_id": item_id,
                "question_part": part,
                "correct": is_correct,
                "source": row.get("source", ""),
                "platform": row.get("platform", ""),
                "source_file": name_from_user_id(user_id),
                "return_to_go": 0.0,
            }
        )

    if transitions:
        transitions[-1]["done"] = 1
        running = 0.0
        for tr in reversed(transitions):
            running = float(tr["reward"]) + rtg_gamma * running
            tr["return_to_go"] = running
    return transitions


def name_from_user_id(user_id: str) -> str:
    return f"KT4/u{user_id}.csv"


def parse_user_id(name: str) -> str:
    stem = Path(name).stem
    return stem[1:] if stem.startswith("u") else stem


def write_transitions(path: Path, transitions: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id",
        "timestep",
        "timestamp",
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "question_id",
        "question_part",
        "correct",
        "source",
        "platform",
        "source_file",
        "return_to_go",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transitions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kt4-zip", required=True, type=Path)
    parser.add_argument("--contents-zip", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--max-transitions", type=int, default=50000)
    parser.add_argument("--min-transitions-per-user", type=int, default=2)
    parser.add_argument("--reward-correct", type=float, default=1.0)
    parser.add_argument("--reward-incorrect", type=float, default=-1.0)
    parser.add_argument("--reward-mode", choices=["correctness", "correctness_time_penalty"], default="correctness")
    parser.add_argument("--rtg-gamma", type=float, default=0.99)
    parser.add_argument("--action-space", choices=["question_part", "part_source"], default="question_part")
    parser.add_argument("--state-variant", choices=["basic", "extended"], default="basic")
    args = parser.parse_args()

    question_meta = load_question_metadata(args.contents_zip)
    all_transitions: List[dict] = []
    action_counts: Counter = Counter()
    source_counts: Counter = Counter()
    users_seen = 0
    users_used = 0

    with zipfile.ZipFile(args.kt4_zip) as zf:
        for name in iter_user_csv_names(args.kt4_zip):
            users_seen += 1
            if users_seen > args.max_users:
                break
            rows = read_user_rows(zf, name)
            transitions = build_user_transitions(
                user_id=parse_user_id(name),
                rows=rows,
                question_meta=question_meta,
                reward_correct=args.reward_correct,
                reward_incorrect=args.reward_incorrect,
                rtg_gamma=args.rtg_gamma,
                action_space=args.action_space,
                state_variant=args.state_variant,
                reward_mode=args.reward_mode,
            )
            if len(transitions) < args.min_transitions_per_user:
                continue
            users_used += 1
            all_transitions.extend(transitions)
            if len(all_transitions) >= args.max_transitions:
                all_transitions = all_transitions[: args.max_transitions]
                break

    for tr in all_transitions:
        action_counts[str(tr["action"])] += 1
        source_counts[tr["source"] or "unknown"] += 1

    write_transitions(args.out_csv, all_transitions)

    metadata = {
        "dataset": "EdNet-KT4",
        "kt4_zip": str(args.kt4_zip),
        "contents_zip": str(args.contents_zip),
        "output_csv": str(args.out_csv),
        "users_seen": users_seen,
        "users_used": users_used,
        "transitions": len(all_transitions),
        "unique_states": len({tr["state"] for tr in all_transitions} | {tr["next_state"] for tr in all_transitions}),
        "action_counts": dict(sorted(action_counts.items())),
        "source_counts_top20": dict(source_counts.most_common(20)),
        "reward_correct": args.reward_correct,
        "reward_incorrect": args.reward_incorrect,
        "reward_mode": args.reward_mode,
        "return_to_go_gamma": args.rtg_gamma,
        "state_representation": (
            [
                "attempt_count_bin",
                "running_accuracy_bin",
                "previous_question_part",
                "previous_correctness",
                "time_gap_bin",
            ]
            + (
                ["previous_source_bucket", "previous_platform_bucket"]
                if args.state_variant == "extended"
                else []
            )
        ),
        "state_variant": args.state_variant,
        "action_space": args.action_space,
        "num_actions": 7 * len(SOURCE_BUCKETS) if args.action_space == "part_source" else 7,
        "source_buckets": SOURCE_BUCKETS,
        "platform_buckets": PLATFORM_BUCKETS,
        "reward_design": "respond-event correctness from EdNet contents/questions.csv",
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
