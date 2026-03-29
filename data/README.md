# Dataset: MoLane (CARLANE)

The dataset is part of the CARLANE benchmark and is released under the **Apache 2.0 License**.

## Download

Obtain the MoLane dataset from the official [CARLANE benchmark website](https://carlanebenchmark.github.io/). 

Follow their instructions to download and extract the files.

## Expected Structure

After downloading, the directory should be organized as follows:
```text
data/
└── carlane-benchmark/
└── CARLANE/
└── MoLane/
├── data/ # Images and masks
│ ├── 00000.jpg
│ ├── 00000.png
│ └── ...
└── splits/ # Split files (.txt)
├── source_train.txt
├── source_val.txt
├── target_train.txt
├── target_val.txt
└── target_test.txt
```

### Split File Format

- **Labeled splits** (`source_train.txt`, `source_val.txt`, `target_val.txt`, `target_test.txt`): each line contains two space‑separated paths:
```text
images/00000.jpg masks/00000.png
```
- **Unlabeled split** (`target_train.txt`): each line contains only the image path:
```text
images/00123.jpg
```

All paths are relative to the `data/` directory.

## Configuration

Update the paths in your configuration file (e.g., `configs/default.yaml`) to point to the correct locations:

```yaml
data:
root: /path/to/lane-detection-uda/data/carlane-benchmark/CARLANE/MoLane/data
splits_root: /path/to/lane-detection-uda/data/carlane-benchmark/CARLANE/MoLane/splits
```
## Important Notes
- The dataset is large (several GB). Consider using symlinks if you have limited disk space.

- The masks are binary (0 = background, 255 = lane). Our code binarizes them to 0/1 during loading.

- Do not modify the split files unless you are creating custom splits for your experiments.