# Streaming Algorithms Trade-off Assignment

This project implements two streaming algorithms and compares accuracy, memory,
and processing-time trade-offs on the MovieLens 1M rating stream.

## Dataset

- Dataset: MovieLens 1M
- Source: GroupLens
- URL: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Stream unit: one rating event, formatted as `UserID::MovieID::Rating::Timestamp`

## Implemented Algorithms

- Bloom Filter: approximate membership test for `user_id:movie_id` rating events
- Count-Min Sketch: approximate frequency estimation for movie rating counts

Both algorithms are implemented directly in Python without specialized
streaming-algorithm libraries.

## Run

```powershell
python src/streaming_algorithms_tradeoff.py --project-root .
```

Optional:

```powershell
python src/streaming_algorithms_tradeoff.py --project-root . --source-url "https://github.com/YOUR_ID/YOUR_REPO"
```

Generated files are saved under:

- `data/raw/`
- `outputs/figures/`
- `outputs/tables/`
- `outputs/streaming_algorithms_tradeoff_report.pdf`

The PDF report uses the SUIT font.
