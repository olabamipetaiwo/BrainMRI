(venv) uzoamakaezeakunne@lambda-dual-2:~/dev/projects/BrainMRI$ python src/evaluate.py --data_dir data --fold 0 --output_dir results 2>&1 
src/evaluate.py:79: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt  = torch.load(ckpt_path, map_location=device)
2026-03-19 17:11:00 [INFO] Loaded fold 1 checkpoint (epoch 10)
2026-03-19 17:12:27 [INFO]   20/96 subjects done
2026-03-19 17:13:50 [INFO]   40/96 subjects done
2026-03-19 17:15:13 [INFO]   60/96 subjects done
2026-03-19 17:16:37 [INFO]   80/96 subjects done
2026-03-19 17:17:45 [INFO] 
Fold 1 — 96 val subjects
2026-03-19 17:17:45 [INFO] Region      Dice      HD95
2026-03-19 17:17:45 [INFO] --------------------------
2026-03-19 17:17:45 [INFO]     ET    0.6848      4.92
2026-03-19 17:17:45 [INFO]     TC    0.7505      5.18
2026-03-19 17:17:45 [INFO]     WT    0.8747      4.91
2026-03-19 17:17:45 [INFO] Metrics saved → results/fold_1/metrics.json
2026-03-19 17:17:45 [INFO] Loaded fold 2 checkpoint (epoch 8)
2026-03-19 17:19:13 [INFO]   20/96 subjects done
2026-03-19 17:20:37 [INFO]   40/96 subjects done
2026-03-19 17:22:02 [INFO]   60/96 subjects done
2026-03-19 17:23:26 [INFO]   80/96 subjects done
2026-03-19 17:24:33 [INFO] 
Fold 2 — 96 val subjects
2026-03-19 17:24:33 [INFO] Region      Dice      HD95
2026-03-19 17:24:33 [INFO] --------------------------
2026-03-19 17:24:33 [INFO]     ET    0.7220      4.87
2026-03-19 17:24:33 [INFO]     TC    0.7645      6.57
2026-03-19 17:24:33 [INFO]     WT    0.8644      5.70
2026-03-19 17:24:33 [INFO] Metrics saved → results/fold_2/metrics.json
2026-03-19 17:24:33 [INFO] Loaded fold 3 checkpoint (epoch 10)
2026-03-19 17:25:59 [INFO]   20/96 subjects done
2026-03-19 17:27:23 [INFO]   40/96 subjects done
2026-03-19 17:28:46 [INFO]   60/96 subjects done
2026-03-19 17:30:10 [INFO]   80/96 subjects done
2026-03-19 17:31:16 [INFO] 
Fold 3 — 96 val subjects
2026-03-19 17:31:16 [INFO] Region      Dice      HD95
2026-03-19 17:31:16 [INFO] --------------------------
2026-03-19 17:31:16 [INFO]     ET    0.7187      5.43
2026-03-19 17:31:16 [INFO]     TC    0.7680      5.07
2026-03-19 17:31:16 [INFO]     WT    0.8777      4.71
2026-03-19 17:31:16 [INFO] Metrics saved → results/fold_3/metrics.json
2026-03-19 17:31:16 [INFO] Loaded fold 4 checkpoint (epoch 10)
2026-03-19 17:32:43 [INFO]   20/96 subjects done
2026-03-19 17:34:06 [INFO]   40/96 subjects done
2026-03-19 17:35:29 [INFO]   60/96 subjects done
2026-03-19 17:36:52 [INFO]   80/96 subjects done
2026-03-19 17:37:58 [INFO] 
Fold 4 — 96 val subjects
2026-03-19 17:37:58 [INFO] Region      Dice      HD95
2026-03-19 17:37:58 [INFO] --------------------------
2026-03-19 17:37:58 [INFO]     ET    0.6804      5.65
2026-03-19 17:37:58 [INFO]     TC    0.7493      5.90
2026-03-19 17:37:58 [INFO]     WT    0.8706      4.45
2026-03-19 17:37:58 [INFO] Metrics saved → results/fold_4/metrics.json
2026-03-19 17:37:59 [INFO] Loaded fold 5 checkpoint (epoch 9)
2026-03-19 17:39:26 [INFO]   20/100 subjects done
2026-03-19 17:40:49 [INFO]   40/100 subjects done
2026-03-19 17:42:12 [INFO]   60/100 subjects done
2026-03-19 17:43:36 [INFO]   80/100 subjects done
2026-03-19 17:44:59 [INFO]   100/100 subjects done
2026-03-19 17:44:59 [INFO] 
Fold 5 — 100 val subjects
2026-03-19 17:44:59 [INFO] Region      Dice      HD95
2026-03-19 17:44:59 [INFO] --------------------------
2026-03-19 17:44:59 [INFO]     ET    0.6947      5.32
2026-03-19 17:44:59 [INFO]     TC    0.7435      7.16
2026-03-19 17:44:59 [INFO]     WT    0.8570      5.84
2026-03-19 17:44:59 [INFO] Metrics saved → results/fold_5/metrics.json

=== Cross-Validation Summary ===
Fold      ET Dice     TC Dice     WT Dice     ET HD95     TC HD95     WT HD95
-----------------------------------------------------------------------------
1          0.6848      0.7505      0.8747        4.92        5.18        4.91
2          0.7220      0.7645      0.8644        4.87        6.57        5.70
3          0.7187      0.7680      0.8777        5.43        5.07        4.71
4          0.6804      0.7493      0.8706        5.65        5.90        4.45
5          0.6947      0.7435      0.8570        5.32        7.16        5.84
-----------------------------------------------------------------------------
Avg        0.7001      0.7552      0.8689        5.24        5.98        5.12

Aggregated results saved → results/cv_results.json