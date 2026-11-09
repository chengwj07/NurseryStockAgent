# SAM3 Troubleshooting

This document describes common issues encountered when installing and loading SAM3 using the [official SAM3 repository](https://github.com/facebookresearch/sam3), along with possible solutions.

## 1. Installation Issues

When installing SAM3 following the official installation instructions, dependency compatibility issues may occur.

### 1.1 Setuptools Version Compatibility

Some environments may encounter installation errors when using newer versions of `setuptools`.

A possible solution is to downgrade `setuptools` to a version below 81.0:

```bash
pip install "setuptools<81.0"
```


### 1.2 Missing Dependencies

Additional Python packages may be required depending on the environment and installed dependencies.

If a `ModuleNotFoundError` occurs, install the corresponding missing package and verify that it is compatible with the installed PyTorch and SAM3 versions.

## 2. Checkpoint Loading Issues

When loading pretrained SAM3 checkpoints, newer versions of PyTorch may raise errors related to the `weights_only` parameter.

### Solution

Locate the checkpoint loading statement in the SAM3 source code.

Change:

```python
weights_only=True
```

to:

```python
weights_only=False
```

For example:

```python
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```

**Note:** Setting `weights_only=False` enables Python pickle deserialization, which can execute arbitrary code. Only apply this modification when loading checkpoints from trusted and verified sources.

## 3. Missing Keys During Model Initialization

During SAM3 model initialization, missing or unexpected checkpoint keys may be reported.

### Solution

Locate the corresponding code block in the official SAM3 source code sam3/model_builder.py (approximately line 558).

**Original code:**

```python
if len(missing_keys) > 0:
    print(
        f"loaded {checkpoint_path} and found "
        f"missing and/or unexpected keys:\n{missing_keys=}"
    )
```

**Modified code:**

```python
if len(missing_keys) > 0:
    missing_keys, _ = model.load_state_dict(ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )
```

This modification attempts to reload the checkpoint using `strict=False` when missing keys are detected.
