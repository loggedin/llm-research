# C4 - experiment log

## experiment 1 - 31 / 07 / 25

### `c4.py` parameters

`seed = 123`

`trigger_lengths = [1, 2, 3, 4, 5]`

`prefix_lengths = [5, 10, 20, 30]`

`suffix_lens = [50, 100]`

`num_trials = 25`

`location='random'`

`top_k=10`

`lambda_reg=0.5`

### `trigger_passage_llm_utils.py` parameters

default

### `retriever_utils.py` parameters

default

### files

results: `./results/c4_31_7_random_k10_lam05.tsv`

log: `./logs/log_c4_31_7_random_k10_lam05.txt`

### status

complete

## experiment 2 - 05 / 08 / 25

### `c4.py` parameters

`seed = 123`

`trigger_lengths = [1, 2]`

`prefix_lengths = [5, 10, 20, 30]`

`suffix_lens = [50, 100]`

`num_trials = 25`

`location='random'`

`top_k=100`

`lambda_reg=0.1`

### `trigger_passage_llm_utils.py` parameters

default

### `retriever_utils.py` parameters

default

### files

results: `./results/c4_5_8_random_k100_lam01_trig12.tsv`

log: `./logs/log_c4_5_8_random_k100_lam01_trig12.txt`

### status

complete

## experiment 3 - 06 / 08 / 25

### `c4.py` parameters

`seed = 123`

`trigger_lengths = [3, 4, 5]`

`prefix_lengths = [5, 10, 20, 30]`

`suffix_lens = [50, 100]`

`num_trials = 25`

`location='random'`

`top_k=100`

`lambda_reg=0.1`

### `trigger_passage_llm_utils.py` parameters

default

### `retriever_utils.py` parameters

default

### files

results: `./results/c4_6_8_random_k100_lam01_trig345.tsv`

log: `./logs/log_c4_6_8_random_k100_lam01_trig345.txt`

### status

complete

