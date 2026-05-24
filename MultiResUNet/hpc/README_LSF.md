# LSF cluster usage

Assumed project path on the cluster:

```bash
/work/phy-tongrj/MultiResUNet
```

## 1. Create the conda environment

```bash
cd /work/phy-tongrj/MultiResUNet
bash hpc/setup_conda_multiresunet.sh
```

## 2. Submit AB tests

```bash
cd /work/phy-tongrj/MultiResUNet
bash hpc/submit_ab_tests_lsf.sh
```

The generated LSF files are written to `hpc/lsf_*.lsf`.

No queue is specified in the LSF files. The cluster default queue/auto-allocation policy will be used.

## 3. Monitor

```bash
bjobs
bjobs -l <JOBID>
tail -f run_logs/run_A_plain_baseline.log
tail -f lsf_logs/A_plain_baseline.<JOBID>.out
```

## 4. TensorBoard

```bash
cd /work/phy-tongrj/MultiResUNet
conda activate multiresunet
tensorboard --logdir runs/logs --host 0.0.0.0 --port 6006
```

If the port is not directly reachable, open an SSH tunnel from your local machine:

```bash
ssh -L 6006:localhost:6006 phy-tongrj@<cluster-login-host>
```

Then open:

```text
http://localhost:6006
```

## Experiments

`A_plain_baseline`: original model capacity, no train augmentation.

`B_mild_aug`: only adds mild augmentation.

`B_strong_aug`: only changes augmentation strength to strong.

`B_focal_loss`: only changes loss to focal loss.

`B_combined_loss`: only changes loss to BCE/Dice combined loss.

`B_lr_step`: only changes LR scheduler to StepLR.

