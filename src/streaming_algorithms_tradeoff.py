# -*- coding: utf-8 -*-
"""Streaming algorithm trade-off assignment.

Downloads MovieLens 1M, processes rating events as a stream, implements Bloom
Filter and Count-Min Sketch directly, compares accuracy/memory/time, and writes
a Korean PDF report using the SUIT font.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
import urllib.request
import zipfile
from array import array
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


RANDOM_STATE = 42
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATASET_PAGE = "https://grouplens.org/datasets/movielens/1m/"
DEFAULT_SOURCE_URL = "https://github.com/sharkless-web/streaming-algorithms-tradeoff"
SUIT_REGULAR_URL = "https://raw.githubusercontent.com/sun-typeface/SUIT/main/fonts/static/ttf/SUIT-Regular.ttf"
SUIT_BOLD_URL = "https://raw.githubusercontent.com/sun-typeface/SUIT/main/fonts/static/ttf/SUIT-Bold.ttf"


@dataclass(frozen=True)
class RatingEvent:
    user_id: str
    movie_id: str
    rating: str
    timestamp: str

    @property
    def pair_key(self) -> str:
        return f"{self.user_id}:{self.movie_id}"


@dataclass
class DatasetStats:
    records: int
    unique_users: int
    unique_movies: int
    unique_user_movie_pairs: int
    max_user_id: int
    max_movie_id: int
    rating_distribution: dict[str, int]
    exact_pair_set_memory_bytes: int
    exact_movie_count_memory_bytes: int


class BloomFilter:
    """Simple Bloom Filter using double hashing over a compact bytearray."""

    def __init__(self, m_bits: int, k_hashes: int) -> None:
        self.m_bits = m_bits
        self.k_hashes = k_hashes
        self.bits = bytearray((m_bits + 7) // 8)

    def _locations(self, item: str) -> Iterable[int]:
        h1, h2 = hash_pair(item)
        h2 = h2 or 0x9E3779B97F4A7C15
        for i in range(self.k_hashes):
            yield (h1 + i * h2 + i * i) % self.m_bits

    def add(self, item: str) -> None:
        for loc in self._locations(item):
            byte_index = loc >> 3
            bit_mask = 1 << (loc & 7)
            self.bits[byte_index] |= bit_mask

    def __contains__(self, item: str) -> bool:
        for loc in self._locations(item):
            byte_index = loc >> 3
            bit_mask = 1 << (loc & 7)
            if not self.bits[byte_index] & bit_mask:
                return False
        return True

    @property
    def memory_bytes(self) -> int:
        return len(self.bits)


class CountMinSketch:
    """Count-Min Sketch with array-backed integer counters."""

    def __init__(self, width: int, depth: int) -> None:
        self.width = width
        self.depth = depth
        self.table = [array("I", [0]) * width for _ in range(depth)]

    def _locations(self, item: str) -> Iterable[tuple[int, int]]:
        h1, h2 = hash_pair(item)
        h2 = h2 or 0x9E3779B97F4A7C15
        for row in range(self.depth):
            yield row, (h1 + row * h2 + row * row) % self.width

    def update(self, item: str, count: int = 1) -> None:
        for row, col in self._locations(item):
            self.table[row][col] += count

    def estimate(self, item: str) -> int:
        return min(self.table[row][col] for row, col in self._locations(item))

    @property
    def memory_bytes(self) -> int:
        return self.width * self.depth * self.table[0].itemsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project directory where data and outputs will be created.",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="GitHub URL shown inside the PDF report.",
    )
    return parser.parse_args()


def ensure_dirs(project_root: Path) -> dict[str, Path]:
    paths = {
        "data_raw": project_root / "data" / "raw",
        "fonts": project_root / "assets" / "fonts",
        "figures": project_root / "outputs" / "figures",
        "tables": project_root / "outputs" / "tables",
        "outputs": project_root / "outputs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_suit_fonts(project_root: Path) -> tuple[Path, Path]:
    fonts_dir = project_root / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    regular_path = fonts_dir / "SUIT-Regular.ttf"
    bold_path = fonts_dir / "SUIT-Bold.ttf"
    if not regular_path.exists():
        urllib.request.urlretrieve(SUIT_REGULAR_URL, regular_path)
    if not bold_path.exists():
        urllib.request.urlretrieve(SUIT_BOLD_URL, bold_path)
    return regular_path, bold_path


def configure_matplotlib_font(project_root: Path) -> None:
    regular_path, _ = ensure_suit_fonts(project_root)
    fm.fontManager.addfont(str(regular_path))
    plt.rcParams["font.family"] = "SUIT"
    plt.rcParams["axes.unicode_minus"] = False


def register_pdf_fonts(project_root: Path) -> tuple[str, str]:
    regular_path, bold_path = ensure_suit_fonts(project_root)
    pdfmetrics.registerFont(TTFont("SUIT", str(regular_path)))
    pdfmetrics.registerFont(TTFont("SUIT-Bold", str(bold_path)))
    pdfmetrics.registerFontFamily("SUIT", normal="SUIT", bold="SUIT-Bold")
    return "SUIT-Bold", "SUIT"


def hash_pair(item: str) -> tuple[int, int]:
    digest = hashlib.blake2b(item.encode("utf-8"), digest_size=16, person=b"streaming").digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little")


def download_movielens(raw_dir: Path) -> Path:
    zip_path = raw_dir / "ml-1m.zip"
    ratings_path = raw_dir / "ml-1m" / "ratings.dat"
    if ratings_path.exists():
        return ratings_path
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)
    return ratings_path


def iter_ratings(ratings_path: Path) -> Iterator[RatingEvent]:
    with ratings_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            user_id, movie_id, rating, timestamp = line.rstrip("\n").split("::")
            yield RatingEvent(user_id, movie_id, rating, timestamp)


def estimate_set_memory(values: set[str], sample_size: int = 5000) -> int:
    base = sys.getsizeof(values)
    sample = list(islice(iter(values), min(sample_size, len(values))))
    if not sample:
        return base
    avg_item = sum(sys.getsizeof(item) for item in sample) / len(sample)
    return int(base + avg_item * len(values))


def estimate_counter_memory(counter: Counter[str], sample_size: int = 5000) -> int:
    base = sys.getsizeof(counter)
    sample = list(islice(counter.items(), min(sample_size, len(counter))))
    if not sample:
        return base
    avg_item = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in sample) / len(sample)
    return int(base + avg_item * len(counter))


def build_ground_truth(ratings_path: Path) -> tuple[DatasetStats, set[str], Counter[str]]:
    pair_set: set[str] = set()
    movie_counts: Counter[str] = Counter()
    users: set[str] = set()
    movies: set[str] = set()
    rating_counts: Counter[str] = Counter()
    max_user_id = 0
    max_movie_id = 0
    records = 0

    for event in iter_ratings(ratings_path):
        records += 1
        pair_set.add(event.pair_key)
        movie_counts[event.movie_id] += 1
        users.add(event.user_id)
        movies.add(event.movie_id)
        rating_counts[event.rating] += 1
        max_user_id = max(max_user_id, int(event.user_id))
        max_movie_id = max(max_movie_id, int(event.movie_id))

    stats = DatasetStats(
        records=records,
        unique_users=len(users),
        unique_movies=len(movies),
        unique_user_movie_pairs=len(pair_set),
        max_user_id=max_user_id,
        max_movie_id=max_movie_id,
        rating_distribution=dict(sorted(rating_counts.items())),
        exact_pair_set_memory_bytes=estimate_set_memory(pair_set),
        exact_movie_count_memory_bytes=estimate_counter_memory(movie_counts),
    )
    return stats, pair_set, movie_counts


def make_membership_queries(
    pair_set: set[str],
    max_user_id: int,
    max_movie_id: int,
    query_count: int = 20000,
) -> tuple[list[str], list[str]]:
    rng = random.Random(RANDOM_STATE)
    pair_list = list(pair_set)
    positives = rng.sample(pair_list, min(query_count, len(pair_list)))
    negatives: list[str] = []
    while len(negatives) < query_count:
        key = f"{rng.randint(1, max_user_id)}:{rng.randint(1, max_movie_id)}"
        if key not in pair_set:
            negatives.append(key)
    return positives, negatives


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_bloom_experiments(
    ratings_path: Path,
    stats: DatasetStats,
    positives: list[str],
    negatives: list[str],
    tables_dir: Path,
) -> list[dict[str, Any]]:
    configs = [
        {"name": "BF-small", "m_bits": 2_000_000, "k_hashes": 3},
        {"name": "BF-medium", "m_bits": 5_000_000, "k_hashes": 5},
        {"name": "BF-large", "m_bits": 10_000_000, "k_hashes": 7},
    ]
    rows: list[dict[str, Any]] = []
    for config in configs:
        bf = BloomFilter(config["m_bits"], config["k_hashes"])
        start = time.perf_counter()
        for event in iter_ratings(ratings_path):
            bf.add(event.pair_key)
        stream_time = time.perf_counter() - start

        query_start = time.perf_counter()
        false_negatives = sum(1 for key in positives if key not in bf)
        false_positives = sum(1 for key in negatives if key in bf)
        query_time = time.perf_counter() - query_start

        theoretical_fpr = (1 - math.exp(-config["k_hashes"] * stats.records / config["m_bits"])) ** config["k_hashes"]
        rows.append(
            {
                "algorithm": "Bloom Filter",
                "config": config["name"],
                "m_bits": config["m_bits"],
                "k_hashes": config["k_hashes"],
                "memory_bytes": bf.memory_bytes,
                "memory_mb": bf.memory_bytes / (1024 * 1024),
                "stream_time_sec": stream_time,
                "query_time_sec": query_time,
                "throughput_records_per_sec": stats.records / stream_time,
                "positive_queries": len(positives),
                "negative_queries": len(negatives),
                "false_negatives": false_negatives,
                "false_positive_rate": false_positives / len(negatives),
                "theoretical_fpr": theoretical_fpr,
                "accuracy_note": "FPR lower is better; false negatives should be 0.",
            }
        )
    write_csv(tables_dir / "bloom_filter_results.csv", rows)
    return rows


def run_count_min_experiments(
    ratings_path: Path,
    stats: DatasetStats,
    movie_counts: Counter[str],
    tables_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs = [
        {"name": "CMS-small", "width": 500, "depth": 3},
        {"name": "CMS-medium", "width": 2_000, "depth": 5},
        {"name": "CMS-large", "width": 10_000, "depth": 7},
    ]
    rows: list[dict[str, Any]] = []
    estimates_by_config: dict[str, CountMinSketch] = {}

    for config in configs:
        cms = CountMinSketch(config["width"], config["depth"])
        start = time.perf_counter()
        for event in iter_ratings(ratings_path):
            cms.update(event.movie_id)
        stream_time = time.perf_counter() - start
        estimates_by_config[config["name"]] = cms

        errors: list[int] = []
        relative_errors: list[float] = []
        exact_matches = 0
        for movie_id, true_count in movie_counts.items():
            estimated = cms.estimate(movie_id)
            error = estimated - true_count
            errors.append(error)
            relative_errors.append(error / true_count)
            if error == 0:
                exact_matches += 1

        rows.append(
            {
                "algorithm": "Count-Min Sketch",
                "config": config["name"],
                "width": config["width"],
                "depth": config["depth"],
                "memory_bytes": cms.memory_bytes,
                "memory_kb": cms.memory_bytes / 1024,
                "stream_time_sec": stream_time,
                "throughput_records_per_sec": stats.records / stream_time,
                "unique_movies_queried": len(movie_counts),
                "mae_count": statistics.mean(errors),
                "max_overestimate": max(errors),
                "mean_relative_error": statistics.mean(relative_errors),
                "p95_relative_error": percentile(relative_errors, 95),
                "exact_match_rate": exact_matches / len(movie_counts),
                "accuracy_note": "CMS never underestimates; relative error lower is better.",
            }
        )

    best_cms = estimates_by_config["CMS-large"]
    top_rows: list[dict[str, Any]] = []
    for rank, (movie_id, true_count) in enumerate(movie_counts.most_common(15), start=1):
        estimated = best_cms.estimate(movie_id)
        top_rows.append(
            {
                "rank": rank,
                "movie_id": movie_id,
                "true_count": true_count,
                "cms_large_estimate": estimated,
                "overestimate": estimated - true_count,
            }
        )

    write_csv(tables_dir / "count_min_sketch_results.csv", rows)
    write_csv(tables_dir / "top_movie_frequency_estimates.csv", top_rows)
    return rows, top_rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    return sorted_values[lower] * (upper - index) + sorted_values[upper] * (index - lower)


def save_dataset_tables(stats: DatasetStats, tables_dir: Path) -> None:
    dataset_rows = [
        {"metric": "records", "value": stats.records},
        {"metric": "unique_users", "value": stats.unique_users},
        {"metric": "unique_movies", "value": stats.unique_movies},
        {"metric": "unique_user_movie_pairs", "value": stats.unique_user_movie_pairs},
        {"metric": "exact_pair_set_memory_mb", "value": stats.exact_pair_set_memory_bytes / (1024 * 1024)},
        {"metric": "exact_movie_count_memory_kb", "value": stats.exact_movie_count_memory_bytes / 1024},
    ]
    write_csv(tables_dir / "dataset_summary.csv", dataset_rows, ["metric", "value"])
    rating_rows = [{"rating": rating, "count": count} for rating, count in stats.rating_distribution.items()]
    write_csv(tables_dir / "rating_distribution.csv", rating_rows, ["rating", "count"])


def save_figures(
    project_root: Path,
    stats: DatasetStats,
    bloom_rows: list[dict[str, Any]],
    cms_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    configure_matplotlib_font(project_root)
    colors_series = ["#3b6ea8", "#e07a5f", "#4f9d69"]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ratings = list(stats.rating_distribution.keys())
    counts = [stats.rating_distribution[rating] for rating in ratings]
    ax.bar(ratings, counts, color="#3b6ea8")
    ax.set_title("MovieLens 1M Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "rating_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    names = [row["config"] for row in bloom_rows]
    memory_mb = [row["memory_mb"] for row in bloom_rows]
    fpr = [row["false_positive_rate"] for row in bloom_rows]
    ax1.bar(names, memory_mb, color="#8ecae6", label="Memory (MB)")
    ax1.set_ylabel("Memory (MB)")
    ax2 = ax1.twinx()
    ax2.plot(names, fpr, marker="o", color="#d62828", linewidth=2.2, label="False Positive Rate")
    ax2.set_ylabel("False Positive Rate")
    ax1.set_title("Bloom Filter: Memory vs False Positive Rate")
    fig.tight_layout()
    fig.savefig(figures_dir / "bloom_memory_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    names = [row["config"] for row in cms_rows]
    memory_kb = [row["memory_kb"] for row in cms_rows]
    mean_rel = [row["mean_relative_error"] for row in cms_rows]
    ax1.bar(names, memory_kb, color="#90be6d", label="Memory (KB)")
    ax1.set_ylabel("Memory (KB)")
    ax2 = ax1.twinx()
    ax2.plot(names, mean_rel, marker="o", color="#9d0208", linewidth=2.2, label="Mean Relative Error")
    ax2.set_ylabel("Mean Relative Error")
    ax1.set_title("Count-Min Sketch: Memory vs Relative Error")
    fig.tight_layout()
    fig.savefig(figures_dir / "cms_memory_accuracy.png", dpi=180)
    plt.close(fig)

    combined = bloom_rows + cms_rows
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    names = [row["config"] for row in combined]
    throughput = [row["throughput_records_per_sec"] for row in combined]
    ax.bar(names, throughput, color=[colors_series[i % len(colors_series)] for i in range(len(names))])
    ax.set_title("Streaming Throughput by Configuration")
    ax.set_ylabel("Records per second")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "throughput_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    labels = ["Exact pair set", "BF-large", "Exact movie dict", "CMS-large"]
    values_mb = [
        stats.exact_pair_set_memory_bytes / (1024 * 1024),
        bloom_rows[-1]["memory_bytes"] / (1024 * 1024),
        stats.exact_movie_count_memory_bytes / (1024 * 1024),
        cms_rows[-1]["memory_bytes"] / (1024 * 1024),
    ]
    ax.bar(labels, values_mb, color=["#adb5bd", "#3b6ea8", "#adb5bd", "#4f9d69"])
    ax.set_yscale("log")
    ax.set_title("Approximate Structures vs Exact Ground Truth Memory")
    ax.set_ylabel("Memory (MB, log scale)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures_dir / "memory_exact_vs_approx.png", dpi=180)
    plt.close(fig)


def make_styles(project_root: Path) -> dict[str, ParagraphStyle]:
    title_font, body_font = register_pdf_fonts(project_root)
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            name="KTitle",
            parent=styles["Title"],
            fontName=title_font,
            fontSize=19,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=0.5 * cm,
        ),
        "heading": ParagraphStyle(
            name="KHeading",
            parent=styles["Heading2"],
            fontName=title_font,
            fontSize=13.5,
            leading=18,
            textColor=colors.HexColor("#18364d"),
            spaceBefore=0.34 * cm,
            spaceAfter=0.18 * cm,
        ),
        "body": ParagraphStyle(
            name="KBody",
            parent=styles["BodyText"],
            fontName=body_font,
            fontSize=9.4,
            leading=14,
            spaceAfter=0.12 * cm,
        ),
        "small": ParagraphStyle(
            name="KSmall",
            parent=styles["BodyText"],
            fontName=body_font,
            fontSize=7.2,
            leading=9,
        ),
    }


def table_data(
    rows: list[dict[str, Any]],
    columns: list[str],
    max_rows: int | None = None,
    digits: int = 4,
) -> list[list[str]]:
    selected = rows[:max_rows] if max_rows is not None else rows
    result = [columns]
    for row in selected:
        out_row = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                out_row.append(f"{value:.{digits}f}")
            else:
                out_row.append(str(value))
        result.append(out_row)
    return result


def add_table(story: list[Any], data: list[list[str]], style: ParagraphStyle, col_widths: list[float] | None = None) -> None:
    wrapped = [[Paragraph(str(cell), style) for cell in row] for row in data]
    table = Table(wrapped, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24435c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d0d9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.28 * cm))


def add_image(story: list[Any], path: Path, width_cm: float = 16.2) -> None:
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width_cm * cm
    image.drawHeight = image.drawWidth * ratio
    story.append(image)
    story.append(Spacer(1, 0.32 * cm))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_report(
    project_root: Path,
    source_url: str,
    stats: DatasetStats,
    bloom_rows: list[dict[str, Any]],
    cms_rows: list[dict[str, Any]],
    top_movie_rows: list[dict[str, Any]],
) -> None:
    outputs_dir = project_root / "outputs"
    figures_dir = outputs_dir / "figures"
    report_path = outputs_dir / "streaming_algorithms_tradeoff_report.pdf"
    styles = make_styles(project_root)
    story: list[Any] = []
    body = styles["body"]
    heading = styles["heading"]
    small = styles["small"]

    best_bloom = min(bloom_rows, key=lambda row: row["false_positive_rate"])
    best_cms = min(cms_rows, key=lambda row: row["mean_relative_error"])
    fastest = max(bloom_rows + cms_rows, key=lambda row: row["throughput_records_per_sec"])

    story.append(Paragraph("스트리밍 알고리즘 2종 구현 및 정확도·메모리 트레이드오프 분석", styles["title"]))
    story.append(Paragraph(f"데이터셋: MovieLens 1M rating stream / 소스코드 GitHub: {source_url}", body))
    story.append(Paragraph(f"생성 환경: Python {platform.python_version()} / {platform.system()} {platform.release()}", body))

    story.append(Paragraph("1. 데이터셋 설명", heading))
    story.append(
        Paragraph(
            "MovieLens 1M은 사용자-영화 평점 이벤트로 구성된 공개 데이터셋이다. 본 과제에서는 ratings.dat를 한 줄씩 읽어 "
            "각 행을 UserID, MovieID, Rating, Timestamp로 구성된 스트림 이벤트로 처리하였다. 알고리즘 처리 단계에서는 전체 CSV를 "
            "DataFrame으로 올리지 않고 파일 iterator를 통해 순차 처리하였다.",
            body,
        )
    )
    dataset_rows = [
        {"항목": "전체 레코드 수", "값": f"{stats.records:,}"},
        {"항목": "고유 사용자 수", "값": f"{stats.unique_users:,}"},
        {"항목": "고유 영화 수", "값": f"{stats.unique_movies:,}"},
        {"항목": "고유 user-movie 이벤트 수", "값": f"{stats.unique_user_movie_pairs:,}"},
        {"항목": "정확한 pair set 메모리 추정", "값": f"{stats.exact_pair_set_memory_bytes / (1024 * 1024):.2f} MB"},
        {"항목": "정확한 movie count dict 메모리 추정", "값": f"{stats.exact_movie_count_memory_bytes / 1024:.2f} KB"},
    ]
    add_table(story, table_data(dataset_rows, ["항목", "값"]), small, [7.0 * cm, 6.0 * cm])
    add_image(story, figures_dir / "rating_distribution.png", width_cm=13.5)

    story.append(Paragraph("2. 선택 알고리즘 개요", heading))
    story.append(
        Paragraph(
            "<b>Bloom Filter</b>는 비트 배열과 여러 해시 함수를 사용하여 원소 포함 여부를 근사 판정한다. "
            "삽입된 원소에 대한 false negative는 발생하지 않지만, 미삽입 원소가 있다고 오판하는 false positive는 발생할 수 있다.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Count-Min Sketch</b>는 depth x width 카운터 테이블에 해시된 위치를 증가시키고, 질의 시 여러 행의 최솟값을 반환한다. "
            "빈도를 과소추정하지 않는 대신 해시 충돌 때문에 과대추정이 발생할 수 있다.",
            body,
        )
    )

    story.append(Paragraph("3. 구현 방식", heading))
    story.append(
        Paragraph(
            "두 알고리즘 모두 전문 라이브러리 없이 직접 구현하였다. 해시는 blake2b 기반 double hashing을 사용하여 항목당 16바이트 digest에서 "
            "두 개의 64-bit 값을 만들고, 이를 조합해 여러 위치를 계산하였다. Bloom Filter는 bytearray로 비트 배열을 압축 저장했고, "
            "Count-Min Sketch는 array('I') 기반 정수 카운터 테이블을 사용하였다.",
            body,
        )
    )
    implementation_rows = [
        {"알고리즘": "Bloom Filter", "스트림 입력": "user_id:movie_id", "Ground Truth": "정확한 Python set", "정확도 지표": "False Positive Rate"},
        {"알고리즘": "Count-Min Sketch", "스트림 입력": "movie_id", "Ground Truth": "정확한 Counter/dict", "정확도 지표": "MAE, 상대오차"},
    ]
    add_table(story, table_data(implementation_rows, ["알고리즘", "스트림 입력", "Ground Truth", "정확도 지표"]), small)

    story.append(Paragraph("4. 실험 환경 및 파라미터", heading))
    parameter_rows = [
        {"알고리즘": "Bloom Filter", "파라미터": "m_bits / k_hashes", "실험값": "2,000,000/3, 5,000,000/5, 10,000,000/7"},
        {"알고리즘": "Count-Min Sketch", "파라미터": "width / depth", "실험값": "500/3, 2,000/5, 10,000/7"},
    ]
    add_table(story, table_data(parameter_rows, ["알고리즘", "파라미터", "실험값"]), small)
    story.append(
        Paragraph(
            "정확도 평가는 Bloom Filter의 경우 실제 존재하지 않는 synthetic user-movie pair 20,000개에 대한 false positive rate를 사용했다. "
            "Count-Min Sketch는 모든 고유 movie_id에 대해 정확한 빈도와 추정 빈도를 비교했다.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("5. 정확도 비교 결과", heading))
    add_table(
        story,
        table_data(
            bloom_rows,
            ["config", "m_bits", "k_hashes", "memory_mb", "false_positive_rate", "theoretical_fpr", "false_negatives"],
            digits=4,
        ),
        small,
    )
    add_image(story, figures_dir / "bloom_memory_accuracy.png")
    story.append(
        Paragraph(
            f"Bloom Filter에서는 {best_bloom['config']}가 가장 낮은 false positive rate({pct(best_bloom['false_positive_rate'])})를 기록했다. "
            "메모리가 커지고 적절한 해시 함수 수를 사용하면 충돌이 줄어 정확도가 개선되었다.",
            body,
        )
    )
    add_table(
        story,
        table_data(
            cms_rows,
            ["config", "width", "depth", "memory_kb", "mae_count", "mean_relative_error", "p95_relative_error", "exact_match_rate"],
            digits=4,
        ),
        small,
    )
    add_image(story, figures_dir / "cms_memory_accuracy.png")
    story.append(
        Paragraph(
            f"Count-Min Sketch에서는 {best_cms['config']}가 가장 낮은 평균 상대오차({pct(best_cms['mean_relative_error'])})를 보였다. "
            "width가 커질수록 서로 다른 영화 ID가 같은 카운터에 충돌할 가능성이 낮아졌다.",
            body,
        )
    )
    add_table(
        story,
        table_data(top_movie_rows, ["rank", "movie_id", "true_count", "cms_large_estimate", "overestimate"], max_rows=10),
        small,
        [1.3 * cm, 2.2 * cm, 3.0 * cm, 4.0 * cm, 3.0 * cm],
    )

    story.append(Paragraph("6. 메모리 사용량 비교", heading))
    story.append(
        Paragraph(
            "근사 자료구조는 정확한 set/dict를 유지하는 방식보다 훨씬 작은 메모리로 동작했다. 특히 Bloom Filter는 전체 user-movie pair를 "
            "저장하지 않고도 포함 여부를 판정할 수 있어 정확한 pair set 대비 큰 메모리 절감 효과가 있었다.",
            body,
        )
    )
    add_image(story, figures_dir / "memory_exact_vs_approx.png")

    story.append(Paragraph("7. 처리 시간 비교", heading))
    time_rows = []
    for row in bloom_rows + cms_rows:
        time_rows.append(
            {
                "algorithm": row["algorithm"],
                "config": row["config"],
                "stream_time_sec": row["stream_time_sec"],
                "throughput_records_per_sec": row["throughput_records_per_sec"],
            }
        )
    add_table(story, table_data(time_rows, ["algorithm", "config", "stream_time_sec", "throughput_records_per_sec"], digits=2), small)
    add_image(story, figures_dir / "throughput_comparison.png")
    story.append(
        Paragraph(
            f"가장 높은 처리량은 {fastest['algorithm']}의 {fastest['config']} 설정에서 관찰되었다. "
            "해시 위치와 카운터 갱신 횟수가 늘어날수록 정확도는 개선될 수 있지만 처리 시간은 증가하는 경향이 있었다.",
            body,
        )
    )

    story.append(Paragraph("8. 알고리즘별 장단점 분석", heading))
    pros_rows = [
        {
            "알고리즘": "Bloom Filter",
            "장점": "메모리가 매우 작고 membership query가 빠르며 false negative가 없다.",
            "단점": "삭제가 어렵고 false positive가 발생하며 빈도 정보는 제공하지 않는다.",
        },
        {
            "알고리즘": "Count-Min Sketch",
            "장점": "대규모 항목 빈도를 작은 메모리로 추정하고 heavy hitter 탐지에 유용하다.",
            "단점": "충돌로 빈도를 과대추정하며, 낮은 빈도 항목은 상대오차가 커질 수 있다.",
        },
    ]
    add_table(story, table_data(pros_rows, ["알고리즘", "장점", "단점"]), small, [3.0 * cm, 6.0 * cm, 6.0 * cm])

    story.append(Paragraph("9. 결론 및 최종 분석 질문", heading))
    answers = [
        (
            "정확도와 메모리 사이에는 어떤 trade-off가 있었는가?",
            "메모리를 늘리면 해시 충돌이 감소해 Bloom Filter의 false positive rate와 Count-Min Sketch의 상대오차가 낮아졌다. "
            "반대로 작은 메모리 설정은 처리 공간을 절약하지만 충돌이 늘어 정확도가 떨어졌다.",
        ),
        (
            "파라미터 증가가 항상 성능 향상으로 이어졌는가?",
            "정확도는 대체로 향상되었지만 항상 비용 없이 좋아진 것은 아니다. 비트 배열, width, depth가 커질수록 메모리와 해시 계산량이 증가하여 "
            "처리 시간이 늘 수 있다. 또한 일정 크기 이후에는 개선 폭이 작아지는 diminishing return이 나타난다.",
        ),
        (
            "어떤 알고리즘이 가장 실용적이라고 판단되는가?",
            "목적에 따라 다르지만, 서비스 로그에서 이벤트 발생 빈도를 계속 추적해야 한다면 Count-Min Sketch가 더 범용적이다. "
            "단순 중복 여부나 이미 본 이벤트 판정이 목적이라면 Bloom Filter가 더 단순하고 메모리 효율적이다.",
        ),
        (
            "실제 서비스 로그 분석에 적용한다면 어떤 알고리즘을 선택할 것인가?",
            "실제 서비스 로그에서는 인기 상품, 자주 발생하는 오류, 상위 검색어처럼 빈도 기반 의사결정이 많으므로 Count-Min Sketch를 우선 선택하겠다. "
            "다만 중복 이벤트 제거, 캐시 미스 방지, 악성 URL 포함 여부 판정 같은 보조 작업에는 Bloom Filter를 함께 사용하는 구성이 실용적이다.",
        ),
    ]
    for question, answer in answers:
        story.append(Paragraph(f"<b>{question}</b>", body))
        story.append(Paragraph(answer, body))

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
    )
    doc.build(story)


def write_summary(
    project_root: Path,
    stats: DatasetStats,
    bloom_rows: list[dict[str, Any]],
    cms_rows: list[dict[str, Any]],
) -> None:
    outputs_dir = project_root / "outputs"
    best_bloom = min(bloom_rows, key=lambda row: row["false_positive_rate"])
    best_cms = min(cms_rows, key=lambda row: row["mean_relative_error"])
    summary = {
        "dataset_records": stats.records,
        "best_bloom_config": best_bloom,
        "best_count_min_config": best_cms,
        "report": str(outputs_dir / "streaming_algorithms_tradeoff_report.pdf"),
    }
    (outputs_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    paths = ensure_dirs(project_root)
    configure_matplotlib_font(project_root)

    ratings_path = download_movielens(paths["data_raw"])
    stats, pair_set, movie_counts = build_ground_truth(ratings_path)
    save_dataset_tables(stats, paths["tables"])
    positives, negatives = make_membership_queries(pair_set, stats.max_user_id, stats.max_movie_id)
    bloom_rows = run_bloom_experiments(ratings_path, stats, positives, negatives, paths["tables"])
    cms_rows, top_movie_rows = run_count_min_experiments(ratings_path, stats, movie_counts, paths["tables"])
    save_figures(project_root, stats, bloom_rows, cms_rows, paths["figures"])
    build_report(project_root, args.source_url, stats, bloom_rows, cms_rows, top_movie_rows)
    write_summary(project_root, stats, bloom_rows, cms_rows)

    print("Done")
    print(f"Project root: {project_root}")
    print(f"PDF report: {project_root / 'outputs' / 'streaming_algorithms_tradeoff_report.pdf'}")


if __name__ == "__main__":
    main()
