# Dataset Placement

This project uses the **Body Performance Data** dataset (13,393 rows, 12 columns):
physical fitness measurements and performance-test results for South Korean
adults, labeled with a 4-level performance grade (`A` best ... `D` lowest).

## Download

The dataset is publicly available on Kaggle:
<https://www.kaggle.com/datasets/kukuroo3/body-performance-data>

## Placement

Download `bodyPerformance.csv` and place it at:

```
data/bodyPerformance.csv
```

`train.py`, `evaluate.py`, and the notebook all expect this exact path by
default (override with `--data <path>` if you keep it elsewhere).

## Columns

| Column | Description | Type |
|---|---|---|
| `age` | Participant's age | Numeric |
| `gender` | `M` / `F` | Categorical |
| `height_cm` | Height in centimeters | Numeric |
| `weight_kg` | Weight in kilograms | Numeric |
| `body fat_%` | Body fat percentage | Numeric |
| `diastolic` | Diastolic blood pressure | Numeric |
| `systolic` | Systolic blood pressure | Numeric |
| `gripForce` | Grip strength | Numeric |
| `sit and bend forward_cm` | Flexibility measurement | Numeric |
| `sit-ups counts` | Sit-ups performed | Numeric |
| `broad jump_cm` | Broad jump distance | Numeric |
| `class` | Performance grade: `A` / `B` / `C` / `D` (target) | Categorical |

This CSV is not committed to the repository (see `.gitignore`) — download it
yourself and place it here before running `train.py`.
