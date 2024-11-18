# Encoder Benchmark

## Usage

Exemplary command:

```
python encoder_benchmark.py --stype-kv categorical embedding --stype-kv numerical linear
```

It will create a dataset that will contain categorical and numerical columns and will use for them
embedding and linear encoders, respectively.

Arguments:

- **--stype-kv**: Specify the stype(s) and corresponding encoder(s) to run.
- **--num-rows**: The number of rows in the dataset (default is `8192`).
- **--out-channels**: The number of output channels (default is `128`).
- **--with-nan**: If specified, the dataset will include NaN values.
- **--runs**: The number of runs for the benchmark (default is `1000`).
- **--warmup-size**: The size of the warmup stage (default is `200`).
- **--torch-profile**: If specified, torch profiling will be enabled.
- **--line-profile**: If specified, line profiling will be enabled.
- **--line-profile-level**: The level of line profiling (default is `'encode_forward'`).
- **--device**: The device to run the benchmark on (default is `'cpu'`).

No matter if any profiler is used, benchmark always outputs a latency (single run execution time), e.g.:

```
Latency: 0.034277s
```

Torch profiler produces a table of operations sorted by execution time, e.g.:

```
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                    aten::cat        47.49%        2.027s        48.05%        2.051s       1.025ms          2000
             aten::nan_to_num        19.74%     842.584ms        39.35%        1.680s     419.945us          4000
                    aten::add        10.04%     428.549ms        10.04%     428.549ms     214.274us          2000
           aten::index_select         6.28%     268.051ms         7.78%     331.959ms     165.980us          2000
                    aten::mul         4.96%     211.853ms         4.96%     211.853ms     211.853us          1000
                    aten::sub         1.36%      58.064ms         1.36%      58.064ms      58.064us          1000
                    aten::any         1.30%      55.612ms         1.41%      60.278ms      30.139us          2000
                    aten::div         1.22%      52.159ms         1.22%      52.159ms      52.159us          1000
                    ...
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 4.268s
```

Line profiler shows how many percent was spent on each of method lines, e.g.:

```
Total time: 1.03661 s
File: {PF_BASE_PATH}/pytorch-frame/torch_frame/nn/encoder/stype_encoder.py
Function: encode_forward at line 295

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   295                                               def encode_forward(
   296                                                   self,
   297                                                   feat: Tensor,
   298                                                   col_names: list[str] | None = None,
   299                                               ) -> Tensor:
   300                                                   # TODO: Make this more efficient.
   301                                                   # Increment the index by one so that NaN index (-1) becomes 0
   302                                                   # (padding_idx)
   303                                                   # feat: [batch_size, num_cols]
   304      1200      29867.4     24.9      2.9          feat = feat + 1
   305      1200        944.3      0.8      0.1          xs = []
   306      3600      19345.7      5.4      1.9          for i, emb in enumerate(self.embs):
   307      2400     466967.7    194.6     45.0              xs.append(emb(feat[:, i]))
   308                                                   # [batch_size, num_cols, hidden_channels]
   309      1200     519044.9    432.5     50.1          x = torch.stack(xs, dim=1)
   310      1200        440.8      0.4      0.0          return x
```
