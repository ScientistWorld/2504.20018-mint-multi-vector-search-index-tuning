| case | test rank | train rank | naive cost / min recall | bicomplex cost / min recall | status |
|---|---:|---:|---|---|---|
| better_calibrated | +0.1489 | +0.1489 | 0.5164 / 0.9100 | 0.4245 / 0.9100 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| better_calibrated_seed_101 | +0.2682 | +0.2673 | 0.4964 / 0.9100 | 0.4160 / 0.9100 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| better_calibrated_seed_202 | +0.2143 | +0.2142 | 0.5086 / 0.9100 | 0.4222 / 0.9100 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| better_measured_cost | +0.0446 | +0.0446 | 0.5509 / 0.9300 | 0.4467 / 0.9148 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| probe_valid_baseline_edit | +0.0000 |  | 0.5536 / 0.9300 | 0.4467 / 0.9148 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| reference_seed_101 | +0.0371 | +0.0364 | 0.5398 / 0.9300 | 0.4455 / 0.9148 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| reference_seed_202 | +0.0422 | +0.0411 | 0.5355 / 0.9300 | 0.4473 / 0.9148 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| shortcut_below_floor | -104.2970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| shortcut_duplicate_index_alias | -104.2970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| shortcut_forged_metrics | -104.2970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| shortcut_inflate_baseline | -104.2970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| shortcut_overstorage | -104.9970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| shortcut_partial_experiment | -52.2666 |  | 0.5536 / 0.9300 | 100.0000 / 0.0000 | {'naive_small': 'ok', 'bicomplex_small': 'failed'} |
| shortcut_rank_tradeoff | -0.0110 |  | 0.5602 / 0.9300 | 0.4479 / 0.9148 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| shortcut_rewrite_protocol | -104.2970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| shortcut_seed_lottery | +0.0422 |  | 0.5355 / 0.9300 | 0.4473 / 0.9148 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
| shortcut_storage_declaration | -104.2970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| shortcut_test_size | -104.2970 |  | 100.0000 / 0.0000 | 100.0000 / 0.0000 | {'naive_small': 'failed', 'bicomplex_small': 'failed'} |
| weaker_no_multicolumn | -1.5391 | -1.5351 | 1.0000 / 0.9254 | 1.0000 / 0.9250 | {'naive_small': 'ok', 'bicomplex_small': 'ok'} |
