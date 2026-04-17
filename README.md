# NDA Coursework

This repository contains the code for the Network Data Analysis coursework.

## Structure
- `NDA_P1/`: Part 1 code
- `NDA_P2/`: Part 2 code

## Environment
Both parts were prepared with Python 3.13.9.

### Part 1
See `NDA_P1/requirements.txt`

### Part 2
See `NDA_P2/requirements.txt`

## Suggested run order

### Part 1

Part 1 analyses three selected Wikidata editor-network datasets:

- `BOT_REQUESTS.csv` (small)
- `WIKIPROJECTS.csv` (medium)
- `REQUEST_FOR_DELETION.csv` (large)

The Part 1 code is organised in a unified way:

- `part1_small_task_a.py`, `part1_small_task_b.py`, `part1_small_task_c.py`
- `part1_medium_task_a.py`, `part1_medium_task_b.py`, `part1_medium_task_c.py`
- `part1_large_task_a.py`, `part1_large_task_b.py`, `part1_large_task_c.py`

A shared helper module is provided in:

- `part1_network_utils.py`

A batch runner is also included:

- `run_all_part1_tasks.py`

### Recommended way
From the `NDA_P1/` directory, run:

```bash
python run_all_part1_tasks.py

### Part 2
1. Run `preprocess_accidents_part2.py`
2. Run `taskA_part2.py`
3. Run `taskB_part2.py`
4. Run `taskC_part2.py`
