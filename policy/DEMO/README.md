# DP vs FM Toy Demo

This directory contains a minimal toy example for action chunks with:

- trajectory length: `100`
- chunk size: `10`
- action dimension: `14`

Generate toy data:

```bash
python policy/DEMO/generate_test_data.py
```

Train and sample with a simple diffusion policy:

```bash
python policy/DEMO/simple_dp.py
```

Train and sample with a simple flow matching policy:

```bash
python policy/DEMO/simple_fm.py
```

Visualize the generation process as GIFs:

```bash
python policy/DEMO/visualize_denoising.py --method both
```

Outputs:

```text
policy/DEMO/visualizations/dp_denoising_compare.gif
policy/DEMO/visualizations/fm_flow_compare.gif
```

Each frame compares the current generated chunk with a real chunk from the
dataset:

- current denoised chunk: full `[10, 14]`
- real target chunk: full `[10, 14]`
- absolute error: full `[10, 14]`
- MSE curve over denoising / flow steps

By default, the script compares against the nearest real chunk to the final
generated sample. To compare against a fixed dataset chunk, pass
`--target-index 0`.

Both methods train on chunks shaped `[10, 14]`.

Diffusion policy trains an epsilon predictor:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
model(x_t, t) -> epsilon
```

Flow matching trains a velocity field:

```text
x_t = (1 - t) * x_0 + t * x_1
model(x_t, t) -> x_1 - x_0
```
