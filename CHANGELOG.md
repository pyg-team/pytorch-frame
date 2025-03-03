# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased] - YYYY-MM-DD

### Added

- Added a classification example script for TabPFN ([#510](https://github.com/pyg-team/pytorch-frame/pull/510))

### Changed

### Deprecated

### Removed

### Fixed

## [0.2.5] - 2025-02-12

### Added

- Added support for Python 3.12 and 3.13 ([#496](https://github.com/pyg-team/pytorch-frame/pull/496))
- Added support for PyTorch 2.6 ([#494](https://github.com/pyg-team/pytorch-frame/pull/494))

## [0.2.4] - 2024-01-16

### Added

- Added an example for training `Trompt` on multiple GPUs ([#474](https://github.com/pyg-team/pytorch-frame/pull/474))
- Added support for materializing dataset for train and test dataframe separately([#470](https://github.com/pyg-team/pytorch-frame/issues/470))
- Added support for PyTorch 2.5 ([#464](https://github.com/pyg-team/pytorch-frame/pull/464))
- Added a benchmark script to compare PyTorch Frame with PyTorch Tabular ([#398](https://github.com/pyg-team/pytorch-frame/pull/398), [#444](https://github.com/pyg-team/pytorch-frame/pull/444))
- Added `is_floating_point` method to `MultiNestedTensor` and `MultiEmbeddingTensor` ([#445](https://github.com/pyg-team/pytorch-frame/pull/445))
- Added support for inferring `stype.categorical` from boolean columns in `utils.infer_series_stype` ([#421](https://github.com/pyg-team/pytorch-frame/pull/421))
- Added `pin_memory()` to `TensorFrame`, `MultiEmbeddingTensor`, and `MultiNestedTensor` ([#437](https://github.com/pyg-team/pytorch-frame/pull/437))

### Changed

- Set `weights_only=True` in `torch_frame.load` from PyTorch 2.4 ([#423](https://github.com/pyg-team/pytorch-frame/pull/423))

### Deprecated

### Removed

- Dropped support for Python 3.8 ([#462](https://github.com/pyg-team/pytorch-frame/pull/462))

### Fixed

- Fixed size mismatch `RuntimeError` in `transforms.CatToNumTransform` ([#446](https://github.com/pyg-team/pytorch-frame/pull/446))
- Removed CUDA synchronizations from `nn.LinearEmbeddingEncoder` ([#432](https://github.com/pyg-team/pytorch-frame/pull/432))
- Removed CUDA synchronizations from N/A imputation logic in `nn.StypeEncoder` ([#433](https://github.com/pyg-team/pytorch-frame/pull/433), [#434](https://github.com/pyg-team/pytorch-frame/pull/434))

## [0.2.3] - 2024-07-08

### Added

- Added `MovieLens 1M` dataset ([#397](https://github.com/pyg-team/pytorch-frame/pull/397))
- Added light-weight MLP ([#372](https://github.com/pyg-team/pytorch-frame/pull/372))
- Added R^2 metric ([#403](https://github.com/pyg-team/pytorch-frame/pull/403))

### Changed

- Updated `ExcelFormer` implementation and related scripts ([#391](https://github.com/pyg-team/pytorch-frame/pull/391))

## [0.2.2] - 2024-03-04

### Added

- Avoided for-loop in `EmbeddingEncoder` ([#366](https://github.com/pyg-team/pytorch-frame/pull/366))
- Added `image_embedded` and one tabular image dataset ([#344](https://github.com/pyg-team/pytorch-frame/pull/344))
- Added benchmarking suite for encoders ([#360](https://github.com/pyg-team/pytorch-frame/pull/360))
- Added dataframe text benchmark script ([#354](https://github.com/pyg-team/pytorch-frame/pull/354), [#367](https://github.com/pyg-team/pytorch-frame/pull/367))
- Added `DataFrameTextBenchmark` dataset ([#349](https://github.com/pyg-team/pytorch-frame/pull/349))
- Added support for empty `TensorFrame` ([#339](https://github.com/pyg-team/pytorch-frame/pull/339))

### Changed

- Changed a workflow of Encoder's `na_forward` method resulting in performance boost ([#364](https://github.com/pyg-team/pytorch-frame/pull/364))
- Removed ReLU applied in `FCResidualBlock` ([#368](https://github.com/pyg-team/pytorch-frame/pull/368))

### Deprecated

### Removed

### Fixed

- Fixed bug in empty `MultiNestedTensor` handling ([#369](https://github.com/pyg-team/pytorch-frame/pull/369))
- Fixed the split of `DataFrameTextBenchmark` ([#358](https://github.com/pyg-team/pytorch-frame/pull/358))
- Fixed empty `MultiNestedTensor` col indexing ([#355](https://github.com/pyg-team/pytorch-frame/pull/355))

## [0.2.1] - 2024-01-16

### Added

- Support more stypes in `LinearModelEncoder` ([#325](https://github.com/pyg-team/pytorch-frame/pull/325))
- Added `stype_encoder_dict` to some models ([#319](https://github.com/pyg-team/pytorch-frame/pull/319))
- Added `HuggingFaceDatasetDict` ([#287](https://github.com/pyg-team/pytorch-frame/pull/287))

### Changed

- Supported decoder embedding model in `examples/transformers_text.py` ([#333](https://github.com/pyg-team/pytorch-frame/pull/333))
- Removed implicit clones in `StypeEncoder` ([#286](https://github.com/pyg-team/pytorch-frame/pull/286))

### Deprecated

### Removed

### Fixed

- Fixed `TimestampEncoder` not applying `CyclicEncoder` to cyclic features ([#311](https://github.com/pyg-team/pytorch-frame/pull/311))
- Fixed NaN masking in `multicateogrical` stype ([#307](https://github.com/pyg-team/pytorch-frame/pull/307))

## [0.2.0] - 2023-12-15

### Added

- Added support for Boolean masks in `index_select` of `_MultiTensor` [334](https://github.com/pyg-team/pytorch-frame/pull/334)
- Added more text documentation ([#291](https://github.com/pyg-team/pytorch-frame/pull/291))
- Added `col_to_model_cfg` ([#270](https://github.com/pyg-team/pytorch-frame/pull/270))
- Support saving/loading of GBDT models ([#269](https://github.com/pyg-team/pytorch-frame/pull/269))
- Added documentation on handling different stypes ([#271](https://github.com/pyg-team/pytorch-frame/pull/271))
- Added `TimestampEncoder` ([#225](https://github.com/pyg-team/pytorch-frame/pull/225))
- Added `LightGBM` ([#248](https://github.com/pyg-team/pytorch-frame/pull/248))
- Added time columns to the `MultimodalTextBenchmark` ([#253](https://github.com/pyg-team/pytorch-frame/pull/253))
- Added `CyclicEncoding` ([#251](https://github.com/pyg-team/pytorch-frame/pull/251))
- Added `PositionalEncoding` ([#249](https://github.com/pyg-team/pytorch-frame/pull/249))
- Added optional `col_names` argument in `StypeEncoder` ([#247](https://github.com/pyg-team/pytorch-frame/pull/247))
- Added `col_to_text_embedder_cfg` and use `MultiEmbeddingTensor` for `text_embedded` ([#246](https://github.com/pyg-team/pytorch-frame/pull/246))
- Added `col_encoder_dict` in `StypeWiseFeatureEncoder` ([#244](https://github.com/pyg-team/pytorch-frame/pull/244))
- Added `LinearEmbeddingEncoder` for `embedding` stype ([#243](https://github.com/pyg-team/pytorch-frame/pull/243))
- Added support for `torch_frame.text_embedded` in `GBDT` ([#239](https://github.com/pyg-team/pytorch-frame/pull/239))
- Support `Metric` in `GBDT` ([#236](https://github.com/pyg-team/pytorch-frame/pull/236))
- Added auto-inference of `stype` ([#221](https://github.com/pyg-team/pytorch-frame/pull/221))
- Enabled `list` input in `multicategorical` stype ([#224](https://github.com/pyg-team/pytorch-frame/pull/224))
- Added `Timestamp` stype ([#212](https://github.com/pyg-team/pytorch-frame/pull/212))
- Added `multicategorical` to `MultimodalTextBenchmark` ([#208](https://github.com/pyg-team/pytorch-frame/pull/208))
- Added support for saving and loading of `TensorFrame` with complex `stypes`. ([#197](https://github.com/pyg-team/pytorch-frame/pull/197))
- Added `stype.embedding` ([#194](https://github.com/pyg-team/pytorch-frame/pull/194))
- Added `TensorFrame` concatenation of complex stypes. ([#190](https://github.com/pyg-team/pytorch-frame/pull/190))
- Added `text_tokenized` example ([#174](https://github.com/pyg-team/pytorch-frame/pull/174))
- Added Cohere embedding example ([#186](https://github.com/pyg-team/pytorch-frame/pull/186))
- Added `AmazonFineFoodReviews` dataset and OpenAI embedding example ([#182](https://github.com/pyg-team/pytorch-frame/pull/182))
- Added save and load logic for `FittableBaseTransform` ([#178](https://github.com/pyg-team/pytorch-frame/pull/178))
- Added `MultiEmbeddingTensor` ([#181](https://github.com/pyg-team/pytorch-frame/pull/181), [#193](https://github.com/pyg-team/pytorch-frame/pull/193), [#198](https://github.com/pyg-team/pytorch-frame/pull/198), [#199](https://github.com/pyg-team/pytorch-frame/pull/199), [#217](https://github.com/pyg-team/pytorch-frame/pull/217))
- Added `to_dense()` for `MultiNestedTensor` ([#170](https://github.com/pyg-team/pytorch-frame/pull/170))
- Added example for `multicategorical` stype ([#162](https://github.com/pyg-team/pytorch-frame/pull/162))
- Added `sequence_numerical` stype ([#159](https://github.com/pyg-team/pytorch-frame/pull/159))
- Added `MultiCategoricalEmbeddingEncoder` ([#155](https://github.com/pyg-team/pytorch-frame/pull/155))
- Added advanced indexing for `MultiNestedTensor` ([#150](https://github.com/pyg-team/pytorch-frame/pull/150), [#161](https://github.com/pyg-team/pytorch-frame/pull/161), [#163](https://github.com/pyg-team/pytorch-frame/pull/163), [#165](https://github.com/pyg-team/pytorch-frame/pull/165))
- Added `multicategorical` stype ([#128](https://github.com/pyg-team/pytorch-frame/pull/128), [#151](https://github.com/pyg-team/pytorch-frame/pull/151))
- Added `MultiNestedTensor` ([#149](https://github.com/pyg-team/pytorch-frame/pull/149))

### Changed

- Set `stype.embedding` as the parent of `stype.text_embedded` and unified `stype.text_embedded` with its parent in :obj:`tensor_frame` ([#277](https://github.com/pyg-team/pytorch-frame/pull/277))
- Renamed `torch_frame.stype` module to `torch_frame._stype` ([#275](https://github.com/pyg-team/pytorch-frame/pull/275))
- Renamed `text_tokenized_cfg` into `col_to_text_tokenized_cfg` ([#257](https://github.com/pyg-team/pytorch-frame/pull/257))
- Made `Trompt` output 2-dim embeddings in `forward`
- Renamed `text_embedder_cfg` into `col_to_text_embedder_cfg`

### Removed

- No manual passing of `in_channels` to `LinearEmbeddingEncoder` for `stype.text_embedded` ([#222](https://github.com/pyg-team/pytorch-frame/pull/222))

## [0.1.0] - 2023-10-23

### Added

- Added basic `text_tokenized` ([#157](https://github.com/pyg-team/pytorch-frame/pull/157))
- Added `Mercari` dataset ([#123](https://github.com/pyg-team/pytorch-frame/pull/123/files))
- Added the model performance benchmark script ([#114](https://github.com/pyg-team/pytorch-frame/pull/114))
- Added `DataFrameBenchmark` ([#107](https://github.com/pyg-team/pytorch-frame/pull/107))
- Added concat and equal ops for `TensorFrame` ([#100](https://github.com/pyg-team/pytorch-frame/pull/100))
- Use ROC-AUC for binary classification in GBDT ([#98](https://github.com/pyg-team/pytorch-frame/pull/98))
- Infer `task_type` in dataset ([#97](https://github.com/pyg-team/pytorch-frame/pull/97))
- Added `text_embedded` example ([#95](https://github.com/pyg-team/pytorch-frame/pull/95))
- Added `MultimodalTextBenchmark` ([#92](https://github.com/pyg-team/pytorch-frame/pull/92), [#117](https://github.com/pyg-team/pytorch-frame/pull/117))
- Renamed `x_dict` to `feat_dict` in `TensorFrame` ([#86](https://github.com/pyg-team/pytorch-frame/pull/86))
- Added `TabTransformer` example ([#82](https://github.com/pyg-team/pytorch-frame/pull/82))
- Added `TabNet` example ([#85](https://github.com/pyg-team/pytorch-frame/pull/85))
- Added dataset `tensorframe` and `col_stats` caching ([#84](https://github.com/pyg-team/pytorch-frame/pull/84))
- Added `TabTransformer` ([#74](https://github.com/pyg-team/pytorch-frame/pull/74))
- Added `TabNet` ([#35](https://github.com/pyg-team/pytorch-frame/pull/35))
- Added text embedded stype, mapper and encoder. ([#78](https://github.com/pyg-team/pytorch-frame/pull/78))
- Added `ExcelFormer` example ([#46](https://github.com/pyg-team/pytorch-frame/pull/46))
- Added support for inductive `DataFrame` to `TensorFrame` transformation ([#75](https://github.com/pyg-team/pytorch-frame/pull/75))
- Added `CatBoost` baseline and tuned `CatBoost` example. ([#73](https://github.com/pyg-team/pytorch-frame/pull/73))
- Added `na_strategy` as argument in `StypeEncoder`. ([#69](https://github.com/pyg-team/pytorch-frame/pull/69))
- Added `NAStrategy` class and impute NaN values in `MutualInformationSort`. ([#68](https://github.com/pyg-team/pytorch-frame/pull/68))
- Added `XGBoost` baseline and updated tuned `XGBoost` example. ([#57](https://github.com/pyg-team/pytorch-frame/pull/57))
- Added `CategoricalCatBoostEncoder` and `MutualInformationSort` transforms needed by ExcelFromer ([#52](https://github.com/pyg-team/pytorch-frame/pull/52))
- Added tutorial example script ([#54](https://github.com/pyg-team/pytorch-frame/pull/54))
- Added `ResNet` ([#48](https://github.com/pyg-team/pytorch-frame/pull/48))
- Added `ExcelFormerEncoder` ([#42](https://github.com/pyg-team/pytorch-frame/pull/42))
- Made `FTTransformer` take `TensorFrame` as input ([#45](https://github.com/pyg-team/pytorch-frame/pull/45))
- Added `Tompt` example ([#39](https://github.com/pyg-team/pytorch-frame/pull/39))
- Added `post_module` in `StypeEncoder` ([#43](https://github.com/pyg-team/pytorch-frame/pull/43))
- Added `FTTransformer` ([#40](https://github.com/pyg-team/pytorch-frame/pull/40), [#41](https://github.com/pyg-team/pytorch-frame/pull/41))
- Added `ExcelFormer` ([#26](https://github.com/pyg-team/pytorch-frame/pull/26))
- Added `Yandex` collections ([#37](https://github.com/pyg-team/pytorch-frame/pull/37))
- Added `TabularBenchmark` collections ([#33](https://github.com/pyg-team/pytorch-frame/pull/33))
- Added the `Bank Marketing` dataset ([#34](https://github.com/pyg-team/pytorch-frame/pull/34))
- Added the `Mushroom`, `Forest Cover Type`, and `Poker Hand` datasets ([#32](https://github.com/pyg-team/pytorch-frame/pull/32))
- Added `PeriodicEncoder` ([#31](https://github.com/pyg-team/pytorch-frame/pull/31))
- Added `NaN` handling in `StypeEncoder` ([#28](https://github.com/pyg-team/pytorch-frame/pull/28))
- Added `LinearBucketEncoder` ([#22](https://github.com/pyg-team/pytorch-frame/pull/22))
- Added `Trompt` ([#25](https://github.com/pyg-team/pytorch-frame/pull/25))
- Added `TromptDecoder` ([#24](https://github.com/pyg-team/pytorch-frame/pull/24))
- Added `TromptConv` ([#23](https://github.com/pyg-team/pytorch-frame/pull/23))
- Added `StypeWiseFeatureEncoder` ([#16](https://github.com/pyg-team/pytorch-frame/pull/16))
- Added indexing/shuffling and column select functionality in `Dataset` ([#18](https://github.com/pyg-team/pytorch-frame/pull/18), [#19](https://github.com/pyg-team/pytorch-frame/pull/19))
- Added `Adult Census Income` dataset ([#17](https://github.com/pyg-team/pytorch-frame/pull/17))
- Added column-level statistics and dataset materialization ([#15](https://github.com/pyg-team/pytorch-frame/pull/15))
- Added `FTTransformerConvs` ([#12](https://github.com/pyg-team/pytorch-frame/pull/12))
- Added `DataLoader` capabilities ([#11](https://github.com/pyg-team/pytorch-frame/pull/11))
- Added `TensorFrame.index_select` ([#10](https://github.com/pyg-team/pytorch-frame/pull/10))
- Added `Dataset.to_tensor_frame` ([#9](https://github.com/pyg-team/pytorch-frame/pull/9))
- Added base classes `TensorEncoder`, `FeatureEncoder`, `TableConv`, `Decoder` ([#5](https://github.com/pyg-team/pytorch-frame/pull/5))
- Added `TensorFrame` ([#4](https://github.com/pyg-team/pytorch-frame/pull/4))
- Added `Titanic` dataset ([#3](https://github.com/pyg-team/pytorch-frame/pull/3))
- Added `Dataset` base class ([#3](https://github.com/pyg-team/pytorch-frame/pull/3))
