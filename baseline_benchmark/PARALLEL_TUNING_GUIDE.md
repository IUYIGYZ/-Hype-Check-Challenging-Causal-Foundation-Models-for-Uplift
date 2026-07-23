# 多数据集并行自动调参指南（中文）  （The English version is under the Chinese version）

## 1. 运行策略

`run_parallel_tuning.py` 同时管理以下四个独立数据集任务：

- RetailHero
- Hillstrom
- LZD
- Criteo

默认最多同时运行两个数据集，并把它们分别绑定到物理 GPU 0 和 GPU 1。RetailHero、Hillstrom 和 LZD 可以并行；Criteo 有 13,979,592 行，默认设置为独占任务，只有其他任务结束后才会启动，而且运行期间不会再启动其他数据集。

每个数据集内部依次调节 T-Learner、X-Learner、DR-Learner 和 DragonNet。并行启动器不会传递 `--final-test`，所以它只做 validation 调参，不会提前查看 test。

## 2. 小样本后台测试

先进入项目目录：

```bash
cd /home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark
```

运行一个脱离 SSH 的小样本测试：

```bash
conda run -n Torch25 python run_parallel_tuning.py \
  --detach \
  --datasets retailhero,hillstrom,lzd \
  --max-parallel 2 \
  --device cuda \
  --gpu-ids 0,1 \
  --max-rows 10000 \
  --traditional-trials 2 \
  --dragonnet-trials 2 \
  --neural-max-epochs 5
```

命令会立即打印 supervisor PID 和 run directory，然后返回 shell。关闭 SSH 后，supervisor 和已经启动的子任务仍在独立 session 中运行。

这个命令只验证运行流程，不应作为正式实验结果。

## 3. 四个数据集使用全部数据正式调参

```bash
conda run -n Torch25 python run_parallel_tuning.py \
  --detach \
  --datasets retailhero,hillstrom,lzd,criteo \
  --max-parallel 2 \
  --exclusive-datasets criteo \
  --device cuda \
  --gpu-ids 0,1 \
  --cpu-threads-per-job 8 \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --seed 42 \
  --search-seed 2026 \
  --objective qini_auc_normalized
```

关键参数：

- `--detach`：启动新 session，不依赖当前 SSH；
- `--max-rows 0`：读取每个 cleaned 数据集的全部行；
- `--max-parallel 2`：最多两个数据集同时运行；
- `--gpu-ids 0,1`：每个并行任务独占一张 GPU；
- `--exclusive-datasets criteo`：Criteo 单独运行；
- `--traditional-trials 20`：T/X/DR 各 20 个 trial；
- `--dragonnet-trials 20`：DragonNet 20 个 trial；
- `--neural-max-epochs 200`：DragonNet 每个 trial 最多 200 epoch，仍启用 early stopping。

不要在正式调参阶段运行 final test。等 validation 参数检查并冻结后，再按照 `AUTO_TUNING_GUIDE.md` 对每个数据集单独执行一次 `--final-test`。

## 4. 查看状态

启动命令会输出类似：

```text
Detached supervisor PID: 123456
Run directory: .../parallel_tuning_runs/batch_YYYYMMDD_HHMMSS
Supervisor log: .../supervisor.log
Status file: .../status.json
```

记录这个 run directory，然后设置：

```bash
RUN_DIR=/home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark/parallel_tuning_runs/batch_YYYYMMDD_HHMMSS
```

查看整体状态：

```bash
cat "$RUN_DIR/status.json"
```

持续查看 supervisor：

```bash
tail -f "$RUN_DIR/supervisor.log"
```

查看单个数据集：

```bash
tail -f "$RUN_DIR/logs/retailhero.log"
tail -f "$RUN_DIR/logs/hillstrom.log"
tail -f "$RUN_DIR/logs/lzd.log"
tail -f "$RUN_DIR/logs/criteo.log"
```

查看 supervisor 是否仍存在：

```bash
ps -p "$(cat "$RUN_DIR/supervisor.pid")" -o pid,etime,cmd
```

查看 GPU：

```bash
nvidia-smi
```

`status.json` 中每个任务可能处于：

```text
pending
running
completed
failed
failed_to_start
```

整体状态 `completed_with_failures` 表示至少一个数据集失败，应查看对应日志。即使某个数据集失败，其他数据集仍会继续。

## 5. 输出结构

```text
parallel_tuning_runs/batch_<timestamp>/
├── supervisor.pid
├── supervisor.log
├── parallel_config.json
├── status.json
├── logs/
│   ├── retailhero.log
│   ├── hillstrom.log
│   ├── lzd.log
│   └── criteo.log
└── results/
    ├── retailhero/
    ├── hillstrom/
    ├── lzd/
    └── criteo/
```

每个 dataset 结果目录内部仍包含 `trial_metrics.csv`、`best_params.json`、`data_manifest.json`、split 和预处理器。每个 trial 结束都会立即写入 checkpoint。

SSH 断开不会影响任务；但服务器重启、断电、进程被系统 OOM killer 终止时不会自动重新启动。此时根据相应日志里的结果目录，使用 `tune_baselines.py --resume-dir ...` 恢复。

---

# Parallel Multi-Dataset Tuning Guide (English)

## 1. Scheduling policy

`run_parallel_tuning.py` supervises independent RetailHero, Hillstrom, LZD, and Criteo tuning jobs. At most two ordinary datasets run concurrently, with one physical GPU assigned to each job. Criteo contains 13,979,592 rows and is exclusive by default: it starts only after other running jobs finish and prevents another dataset from starting beside it.

Each dataset tunes T-Learner, X-Learner, DR-Learner, and DragonNet sequentially, then evaluates pretrained CausalPFN once with fixed parameters. The launcher never supplies `--final-test`; it performs validation selection only.

## 2. Detached small-data check

```bash
cd /home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark

conda run -n Torch25 python run_parallel_tuning.py \
  --detach \
  --datasets retailhero,hillstrom,lzd \
  --max-parallel 2 \
  --device cuda \
  --gpu-ids 0,1 \
  --max-rows 10000 \
  --traditional-trials 2 \
  --dragonnet-trials 2 \
  --neural-max-epochs 5
```

The command prints a supervisor PID and run directory, then returns. The supervisor and child jobs run in independent sessions and survive SSH logout.

## 3. Formal full-data tuning

```bash
conda run -n Torch25 python run_parallel_tuning.py \
  --detach \
  --datasets retailhero,hillstrom,lzd,criteo \
  --max-parallel 2 \
  --exclusive-datasets criteo \
  --device cuda \
  --gpu-ids 0,1 \
  --cpu-threads-per-job 8 \
  --max-rows 0 \
  --traditional-trials 20 \
  --dragonnet-trials 20 \
  --neural-max-epochs 200 \
  --seed 42 \
  --search-seed 2026 \
  --objective qini_auc_normalized
```

`--max-rows 0` uses every cleaned row. It does not train on test: every dataset is still divided into train, validation, and sealed test partitions.

Do not run final test during this stage. After reviewing and freezing validation-selected configurations, use the per-dataset `--final-test` procedure in `AUTO_TUNING_GUIDE.md`.

## 4. Monitoring

Save the printed run directory:

```bash
RUN_DIR=/home/zys/eeg-related-master/eeg-related-2026.3.9/项目/baseline_benchmark/parallel_tuning_runs/batch_YYYYMMDD_HHMMSS
```

Then use:

```bash
cat "$RUN_DIR/status.json"
tail -f "$RUN_DIR/supervisor.log"
tail -f "$RUN_DIR/logs/retailhero.log"
ps -p "$(cat "$RUN_DIR/supervisor.pid")" -o pid,etime,cmd
nvidia-smi
```

A `completed_with_failures` batch means at least one dataset failed; inspect its log. Other datasets continue even when one job fails.

## 5. Durability and recovery

SSH logout does not stop detached jobs. Server reboot, power loss, or an OOM kill is different: processes do not restart automatically. Each completed trial is checkpointed in `trial_metrics.csv`, so use the recorded dataset result directory with `tune_baselines.py --resume-dir ...` to recover.
