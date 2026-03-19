(venv) uzoamakaezeakunne@lambda-dual-2:~/dev/projects/BrainMRI$ bash run_experiments.sh    
[2026-03-18 20:04:26] Device: cuda
[2026-03-18 20:04:26] Folds to run: 1 2 3 4 5
[2026-03-18 20:04:26] Epochs: 10 | Batch: 1 | LR: 1e-4
============================================================
[2026-03-18 20:04:26] >>> TRAINING fold 1 / 5
Device: cuda
2026-03-18 20:04:28 [INFO] Fold 1: 388 train  /  96 val
/home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
2026-03-18 20:23:39 [INFO] Epoch    1/10  train_loss=0.8355  val_dice=0.4646
2026-03-18 20:23:39 [INFO]   >> saved best model  (dice=0.4646)
2026-03-18 20:42:49 [INFO] Epoch    2/10  train_loss=0.6332  val_dice=0.6262
2026-03-18 20:42:50 [INFO]   >> saved best model  (dice=0.6262)
2026-03-18 21:02:00 [INFO] Epoch    3/10  train_loss=0.5382  val_dice=0.6237
2026-03-18 21:21:12 [INFO] Epoch    4/10  train_loss=0.4618  val_dice=0.6675
2026-03-18 21:21:12 [INFO]   >> saved best model  (dice=0.6675)
2026-03-18 21:40:22 [INFO] Epoch    5/10  train_loss=0.3881  val_dice=0.7103
2026-03-18 21:40:23 [INFO]   >> saved best model  (dice=0.7103)
2026-03-18 22:00:00 [INFO] Epoch    6/10  train_loss=0.3216  val_dice=0.7232
2026-03-18 22:00:00 [INFO]   >> saved best model  (dice=0.7232)
2026-03-18 22:19:14 [INFO] Epoch    7/10  train_loss=0.2687  val_dice=0.7196
2026-03-18 22:38:25 [INFO] Epoch    8/10  train_loss=0.2385  val_dice=0.7564
2026-03-18 22:38:26 [INFO]   >> saved best model  (dice=0.7564)
2026-03-18 22:57:42 [INFO] Epoch    9/10  train_loss=0.2185  val_dice=0.7500
2026-03-18 23:16:56 [INFO] Epoch   10/10  train_loss=0.2054  val_dice=0.7683
2026-03-18 23:16:56 [INFO]   >> saved best model  (dice=0.7683)
2026-03-18 23:16:56 [INFO] Fold 1 finished — best val dice: 0.7683
[2026-03-18 23:16:57] Fold 1 training done.
------------------------------------------------------------
[2026-03-18 23:16:57] >>> TRAINING fold 2 / 5
Device: cuda
2026-03-18 23:17:00 [INFO] Fold 2: 388 train  /  96 val
/home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
2026-03-18 23:36:21 [INFO] Epoch    1/10  train_loss=0.6866  val_dice=0.5471
2026-03-18 23:36:21 [INFO]   >> saved best model  (dice=0.5471)
2026-03-18 23:55:34 [INFO] Epoch    2/10  train_loss=0.5083  val_dice=0.6563
2026-03-18 23:55:34 [INFO]   >> saved best model  (dice=0.6563)
^C source /home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/bin/activate
Traceback (most recent call last):
  File "src/train.py", line 196, in <module>
    main()
  File "src/train.py", line 192, in main
    train_fold(args.data_dir, fold, splits, device, args)
  File "src/train.py", line 125, in train_fold
    train_losses.append(loss.item())
KeyboardInterrupt

(venv) uzoamakaezeakunne@lambda-dual-2:~/dev/projects/BrainMRI$  source /home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/bin/activate
(venv) uzoamakaezeakunne@lambda-dual-2:~/dev/projects/BrainMRI$ bash run_experiments.sh    
[2026-03-19 00:05:59] Device: cuda
[2026-03-19 00:05:59] Folds to run: 1 2 3 4 5
[2026-03-19 00:05:59] Epochs: 10 | Batch: 1 | LR: 1e-4
============================================================
[2026-03-19 00:05:59] >>> TRAINING fold 1 / 5
Device: cuda
2026-03-19 00:06:01 [INFO] Fold 1: 388 train  /  96 val
/home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
2026-03-19 00:25:16 [INFO] Epoch    1/10  train_loss=0.6731  val_dice=0.5933
2026-03-19 00:25:16 [INFO]   >> saved best model  (dice=0.5933)
2026-03-19 00:44:34 [INFO] Epoch    2/10  train_loss=0.5294  val_dice=0.6501
2026-03-19 00:44:34 [INFO]   >> saved best model  (dice=0.6501)
2026-03-19 01:03:49 [INFO] Epoch    3/10  train_loss=0.4513  val_dice=0.6805
2026-03-19 01:03:50 [INFO]   >> saved best model  (dice=0.6805)
2026-03-19 01:23:05 [INFO] Epoch    4/10  train_loss=0.3751  val_dice=0.6944
2026-03-19 01:23:05 [INFO]   >> saved best model  (dice=0.6944)
2026-03-19 01:42:20 [INFO] Epoch    5/10  train_loss=0.3098  val_dice=0.7034
2026-03-19 01:42:21 [INFO]   >> saved best model  (dice=0.7034)
2026-03-19 02:01:39 [INFO] Epoch    6/10  train_loss=0.2650  val_dice=0.7402
2026-03-19 02:01:39 [INFO]   >> saved best model  (dice=0.7402)
2026-03-19 02:20:55 [INFO] Epoch    7/10  train_loss=0.2339  val_dice=0.7557
2026-03-19 02:20:56 [INFO]   >> saved best model  (dice=0.7557)
2026-03-19 02:40:09 [INFO] Epoch    8/10  train_loss=0.2175  val_dice=0.7469
2026-03-19 02:59:20 [INFO] Epoch    9/10  train_loss=0.2051  val_dice=0.7523
2026-03-19 03:18:35 [INFO] Epoch   10/10  train_loss=0.1979  val_dice=0.7699
2026-03-19 03:18:35 [INFO]   >> saved best model  (dice=0.7699)
2026-03-19 03:18:35 [INFO] Fold 1 finished — best val dice: 0.7699
[2026-03-19 03:18:36] Fold 1 training done.
------------------------------------------------------------
[2026-03-19 03:18:36] >>> TRAINING fold 2 / 5
Device: cuda
2026-03-19 03:18:39 [INFO] Fold 2: 388 train  /  96 val
/home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
2026-03-19 03:37:59 [INFO] Epoch    1/10  train_loss=0.6287  val_dice=0.6508
2026-03-19 03:37:59 [INFO]   >> saved best model  (dice=0.6508)
2026-03-19 03:57:11 [INFO] Epoch    2/10  train_loss=0.4799  val_dice=0.7154
2026-03-19 03:57:12 [INFO]   >> saved best model  (dice=0.7154)
2026-03-19 04:16:22 [INFO] Epoch    3/10  train_loss=0.4022  val_dice=0.7381
2026-03-19 04:16:23 [INFO]   >> saved best model  (dice=0.7381)
2026-03-19 04:35:34 [INFO] Epoch    4/10  train_loss=0.3317  val_dice=0.7645
2026-03-19 04:35:34 [INFO]   >> saved best model  (dice=0.7645)
2026-03-19 04:54:55 [INFO] Epoch    5/10  train_loss=0.2777  val_dice=0.7214
2026-03-19 05:14:08 [INFO] Epoch    6/10  train_loss=0.2423  val_dice=0.7632
2026-03-19 05:33:20 [INFO] Epoch    7/10  train_loss=0.2237  val_dice=0.7695
2026-03-19 05:33:21 [INFO]   >> saved best model  (dice=0.7695)
2026-03-19 05:52:34 [INFO] Epoch    8/10  train_loss=0.2108  val_dice=0.7826
2026-03-19 05:52:35 [INFO]   >> saved best model  (dice=0.7826)
2026-03-19 06:11:47 [INFO] Epoch    9/10  train_loss=0.1995  val_dice=0.7687
2026-03-19 06:30:58 [INFO] Epoch   10/10  train_loss=0.1925  val_dice=0.7818
2026-03-19 06:30:58 [INFO] Fold 2 finished — best val dice: 0.7826
[2026-03-19 06:30:59] Fold 2 training done.
------------------------------------------------------------
[2026-03-19 06:30:59] >>> TRAINING fold 3 / 5
Device: cuda
2026-03-19 06:31:01 [INFO] Fold 3: 388 train  /  96 val
/home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
2026-03-19 06:50:13 [INFO] Epoch    1/10  train_loss=0.7667  val_dice=0.4962
2026-03-19 06:50:14 [INFO]   >> saved best model  (dice=0.4962)
2026-03-19 07:09:28 [INFO] Epoch    2/10  train_loss=0.5571  val_dice=0.5491
2026-03-19 07:09:28 [INFO]   >> saved best model  (dice=0.5491)
2026-03-19 07:28:38 [INFO] Epoch    3/10  train_loss=0.4707  val_dice=0.6905
2026-03-19 07:28:38 [INFO]   >> saved best model  (dice=0.6905)
2026-03-19 07:47:49 [INFO] Epoch    4/10  train_loss=0.4142  val_dice=0.7236
2026-03-19 07:47:50 [INFO]   >> saved best model  (dice=0.7236)
2026-03-19 08:07:02 [INFO] Epoch    5/10  train_loss=0.3620  val_dice=0.7378
2026-03-19 08:07:03 [INFO]   >> saved best model  (dice=0.7378)
2026-03-19 08:26:11 [INFO] Epoch    6/10  train_loss=0.3102  val_dice=0.7603
2026-03-19 08:26:12 [INFO]   >> saved best model  (dice=0.7603)
2026-03-19 08:45:26 [INFO] Epoch    7/10  train_loss=0.2677  val_dice=0.7721
2026-03-19 08:45:26 [INFO]   >> saved best model  (dice=0.7721)
2026-03-19 09:04:40 [INFO] Epoch    8/10  train_loss=0.2369  val_dice=0.7802
2026-03-19 09:04:41 [INFO]   >> saved best model  (dice=0.7802)
2026-03-19 09:23:53 [INFO] Epoch    9/10  train_loss=0.2194  val_dice=0.7753
2026-03-19 09:43:09 [INFO] Epoch   10/10  train_loss=0.2058  val_dice=0.7845
2026-03-19 09:43:10 [INFO]   >> saved best model  (dice=0.7845)
2026-03-19 09:43:10 [INFO] Fold 3 finished — best val dice: 0.7845
[2026-03-19 09:43:10] Fold 3 training done.
------------------------------------------------------------
[2026-03-19 09:43:10] >>> TRAINING fold 4 / 5
Device: cuda
2026-03-19 09:43:13 [INFO] Fold 4: 388 train  /  96 val
/home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
2026-03-19 10:02:27 [INFO] Epoch    1/10  train_loss=0.7816  val_dice=0.5279
2026-03-19 10:02:28 [INFO]   >> saved best model  (dice=0.5279)
2026-03-19 10:21:41 [INFO] Epoch    2/10  train_loss=0.5957  val_dice=0.6100
2026-03-19 10:21:42 [INFO]   >> saved best model  (dice=0.6100)
2026-03-19 10:40:57 [INFO] Epoch    3/10  train_loss=0.5119  val_dice=0.6481
2026-03-19 10:40:57 [INFO]   >> saved best model  (dice=0.6481)
2026-03-19 11:00:07 [INFO] Epoch    4/10  train_loss=0.4405  val_dice=0.7097
2026-03-19 11:00:07 [INFO]   >> saved best model  (dice=0.7097)
2026-03-19 11:19:22 [INFO] Epoch    5/10  train_loss=0.3708  val_dice=0.7232
2026-03-19 11:19:22 [INFO]   >> saved best model  (dice=0.7232)
2026-03-19 11:38:34 [INFO] Epoch    6/10  train_loss=0.3070  val_dice=0.7526
2026-03-19 11:38:35 [INFO]   >> saved best model  (dice=0.7526)
2026-03-19 11:57:47 [INFO] Epoch    7/10  train_loss=0.2581  val_dice=0.7557
2026-03-19 11:57:48 [INFO]   >> saved best model  (dice=0.7557)
2026-03-19 12:17:01 [INFO] Epoch    8/10  train_loss=0.2301  val_dice=0.7608
2026-03-19 12:17:01 [INFO]   >> saved best model  (dice=0.7608)
2026-03-19 12:36:13 [INFO] Epoch    9/10  train_loss=0.2112  val_dice=0.7610
2026-03-19 12:36:13 [INFO]   >> saved best model  (dice=0.7610)
2026-03-19 12:55:29 [INFO] Epoch   10/10  train_loss=0.2020  val_dice=0.7673
2026-03-19 12:55:30 [INFO]   >> saved best model  (dice=0.7673)
2026-03-19 12:55:30 [INFO] Fold 4 finished — best val dice: 0.7673
[2026-03-19 12:55:31] Fold 4 training done.
------------------------------------------------------------
[2026-03-19 12:55:31] >>> TRAINING fold 5 / 5
Device: cuda
2026-03-19 12:55:33 [INFO] Fold 5: 384 train  /  100 val
/home/uzoamakaezeakunne/dev/projects/BrainMRI/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
2026-03-19 13:14:55 [INFO] Epoch    1/10  train_loss=0.6848  val_dice=0.5240
2026-03-19 13:14:55 [INFO]   >> saved best model  (dice=0.5240)
2026-03-19 13:34:13 [INFO] Epoch    2/10  train_loss=0.5292  val_dice=0.6470
2026-03-19 13:34:13 [INFO]   >> saved best model  (dice=0.6470)
2026-03-19 13:53:33 [INFO] Epoch    3/10  train_loss=0.4430  val_dice=0.6796
2026-03-19 13:53:34 [INFO]   >> saved best model  (dice=0.6796)
2026-03-19 14:12:57 [INFO] Epoch    4/10  train_loss=0.3628  val_dice=0.7061
2026-03-19 14:12:58 [INFO]   >> saved best model  (dice=0.7061)
2026-03-19 14:32:15 [INFO] Epoch    5/10  train_loss=0.2986  val_dice=0.7115
2026-03-19 14:32:16 [INFO]   >> saved best model  (dice=0.7115)
2026-03-19 14:51:35 [INFO] Epoch    6/10  train_loss=0.2582  val_dice=0.7312
2026-03-19 14:51:36 [INFO]   >> saved best model  (dice=0.7312)
2026-03-19 15:10:58 [INFO] Epoch    7/10  train_loss=0.2342  val_dice=0.6919
2026-03-19 15:30:20 [INFO] Epoch    8/10  train_loss=0.2168  val_dice=0.7558
2026-03-19 15:30:20 [INFO]   >> saved best model  (dice=0.7558)
2026-03-19 15:49:40 [INFO] Epoch    9/10  train_loss=0.2088  val_dice=0.7612
2026-03-19 15:49:40 [INFO]   >> saved best model  (dice=0.7612)
2026-03-19 16:09:01 [INFO] Epoch   10/10  train_loss=0.2004  val_dice=0.7608
2026-03-19 16:09:01 [INFO] Fold 5 finished — best val dice: 0.7612
[2026-03-19 16:09:02] Fold 5 training done.
------------------------------------------------------------
[2026-03-19 16:09:02] >>> EVALUATING all trained folds
src/evaluate.py:67: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt  = torch.load(ckpt_path, map_location=device)
2026-03-19 16:09:04 [INFO] Loaded fold 1 checkpoint (epoch 10)
2026-03-19 16:10:25 [INFO]   20/96 subjects done
2026-03-19 16:11:42 [INFO]   40/96 subjects done
2026-03-19 16:12:59 [INFO]   60/96 subjects done
2026-03-19 16:14:17 [INFO]   80/96 subjects done
2026-03-19 16:15:19 [INFO] 
Fold 1 — 96 val subjects
2026-03-19 16:15:19 [INFO] Region      Dice      HD95
2026-03-19 16:15:19 [INFO] --------------------------
2026-03-19 16:15:19 [INFO]     ET    0.6846      5.59
2026-03-19 16:15:19 [INFO]     TC    0.7505      5.18
2026-03-19 16:15:19 [INFO]     WT    0.8747      4.91
2026-03-19 16:15:19 [INFO] Metrics saved → results/fold_1/metrics.json
2026-03-19 16:15:19 [INFO] Loaded fold 2 checkpoint (epoch 8)
2026-03-19 16:16:39 [INFO]   20/96 subjects done
2026-03-19 16:17:56 [INFO]   40/96 subjects done
2026-03-19 16:19:13 [INFO]   60/96 subjects done
2026-03-19 16:20:30 [INFO]   80/96 subjects done
2026-03-19 16:21:31 [INFO] 
Fold 2 — 96 val subjects
2026-03-19 16:21:31 [INFO] Region      Dice      HD95
2026-03-19 16:21:31 [INFO] --------------------------
2026-03-19 16:21:31 [INFO]     ET    0.7187      5.27
2026-03-19 16:21:31 [INFO]     TC    0.7645      6.57
2026-03-19 16:21:31 [INFO]     WT    0.8644      5.70
2026-03-19 16:21:31 [INFO] Metrics saved → results/fold_2/metrics.json
2026-03-19 16:21:32 [INFO] Loaded fold 3 checkpoint (epoch 10)
2026-03-19 16:22:52 [INFO]   20/96 subjects done
2026-03-19 16:24:10 [INFO]   40/96 subjects done
2026-03-19 16:25:27 [INFO]   60/96 subjects done
2026-03-19 16:26:45 [INFO]   80/96 subjects done
2026-03-19 16:27:47 [INFO] 
Fold 3 — 96 val subjects
2026-03-19 16:27:47 [INFO] Region      Dice      HD95
2026-03-19 16:27:47 [INFO] --------------------------
2026-03-19 16:27:47 [INFO]     ET    0.7078      4.27
2026-03-19 16:27:47 [INFO]     TC    0.7680      5.07
2026-03-19 16:27:47 [INFO]     WT    0.8777      4.71
2026-03-19 16:27:47 [INFO] Metrics saved → results/fold_3/metrics.json
2026-03-19 16:27:48 [INFO] Loaded fold 4 checkpoint (epoch 10)
2026-03-19 16:29:08 [INFO]   20/96 subjects done
2026-03-19 16:30:26 [INFO]   40/96 subjects done
2026-03-19 16:31:43 [INFO]   60/96 subjects done
2026-03-19 16:33:01 [INFO]   80/96 subjects done
2026-03-19 16:34:03 [INFO] 
Fold 4 — 96 val subjects
2026-03-19 16:34:03 [INFO] Region      Dice      HD95
2026-03-19 16:34:03 [INFO] --------------------------
2026-03-19 16:34:03 [INFO]     ET    0.6820      5.42
2026-03-19 16:34:03 [INFO]     TC    0.7493      5.90
2026-03-19 16:34:03 [INFO]     WT    0.8706      4.45
2026-03-19 16:34:03 [INFO] Metrics saved → results/fold_4/metrics.json
2026-03-19 16:34:04 [INFO] Loaded fold 5 checkpoint (epoch 9)
2026-03-19 16:35:23 [INFO]   20/100 subjects done
2026-03-19 16:36:40 [INFO]   40/100 subjects done
2026-03-19 16:37:57 [INFO]   60/100 subjects done
2026-03-19 16:39:14 [INFO]   80/100 subjects done
2026-03-19 16:40:30 [INFO]   100/100 subjects done
2026-03-19 16:40:30 [INFO] 
Fold 5 — 100 val subjects
2026-03-19 16:40:30 [INFO] Region      Dice      HD95
2026-03-19 16:40:30 [INFO] --------------------------
2026-03-19 16:40:30 [INFO]     ET    0.6831      6.04
2026-03-19 16:40:30 [INFO]     TC    0.7435      7.16
2026-03-19 16:40:30 [INFO]     WT    0.8570      5.84
2026-03-19 16:40:30 [INFO] Metrics saved → results/fold_5/metrics.json

=== Cross-Validation Summary ===
Fold      ET Dice     TC Dice     WT Dice     ET HD95     TC HD95     WT HD95
-----------------------------------------------------------------------------
1          0.6846      0.7505      0.8747        5.59        5.18        4.91
2          0.7187      0.7645      0.8644        5.27        6.57        5.70
3          0.7078      0.7680      0.8777        4.27        5.07        4.71
4          0.6820      0.7493      0.8706        5.42        5.90        4.45
5          0.6831      0.7435      0.8570        6.04        7.16        5.84
-----------------------------------------------------------------------------
Avg        0.6952      0.7552      0.8689        5.32        5.98        5.12

Aggregated results saved → results/cv_results.json
------------------------------------------------------------
[2026-03-19 16:40:31] >>> VISUALIZING fold 1
src/visualize.py:125: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt  = torch.load(ckpt_path, map_location=device)
Saved: results/visualizations/fold1_sub000_slice064.png
Saved: results/visualizations/fold1_sub001_slice064.png
Saved: results/visualizations/fold1_sub002_slice064.png
Saved: results/visualizations/fold1_sub003_slice064.png
Saved: results/visualizations/fold1_sub004_slice064.png
[2026-03-19 16:41:09] >>> VISUALIZING fold 2
src/visualize.py:125: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt  = torch.load(ckpt_path, map_location=device)
Saved: results/visualizations/fold2_sub000_slice064.png
Saved: results/visualizations/fold2_sub001_slice064.png
Saved: results/visualizations/fold2_sub002_slice064.png
Saved: results/visualizations/fold2_sub003_slice064.png
Saved: results/visualizations/fold2_sub004_slice064.png
[2026-03-19 16:41:47] >>> VISUALIZING fold 3
src/visualize.py:125: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt  = torch.load(ckpt_path, map_location=device)
Saved: results/visualizations/fold3_sub000_slice064.png
Saved: results/visualizations/fold3_sub001_slice064.png
Saved: results/visualizations/fold3_sub002_slice064.png
Saved: results/visualizations/fold3_sub003_slice064.png
Saved: results/visualizations/fold3_sub004_slice064.png
[2026-03-19 16:42:25] >>> VISUALIZING fold 4
src/visualize.py:125: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt  = torch.load(ckpt_path, map_location=device)
Saved: results/visualizations/fold4_sub000_slice064.png
Saved: results/visualizations/fold4_sub001_slice064.png
Saved: results/visualizations/fold4_sub002_slice064.png
Saved: results/visualizations/fold4_sub003_slice064.png
Saved: results/visualizations/fold4_sub004_slice064.png
[2026-03-19 16:43:03] >>> VISUALIZING fold 5
src/visualize.py:125: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt  = torch.load(ckpt_path, map_location=device)
Saved: results/visualizations/fold5_sub000_slice064.png
Saved: results/visualizations/fold5_sub001_slice064.png
Saved: results/visualizations/fold5_sub002_slice064.png
Saved: results/visualizations/fold5_sub003_slice064.png
Saved: results/visualizations/fold5_sub004_slice064.png
============================================================
[2026-03-19 16:43:42] All done.
[2026-03-19 16:43:42]   Checkpoints : results/fold_N/best_model.pth
[2026-03-19 16:43:42]   Metrics     : results/fold_N/metrics.json
[2026-03-19 16:43:42]   CV summary  : results/cv_results.json
[2026-03-19 16:43:42]   Plots       : results/visualizations/