# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2023-MM-DD

### Added

- Added Github front page ([#93]https://github.com/pyg-team/pytorch-frame/pull/93)
- Added multimodal tabular text benchmark datasets. ([#92](https://github.com/pyg-team/pytorch-frame/pull/92))
- Added tutorial documentation. ([#83](https://github.com/pyg-team/pytorch-frame/pull/83))
- Renamed `x_dict` to `feat_dict` in `TensorFrame` ([#86](https://github.com/pyg-team/pytorch-frame/pull/86))
- Added `TabTransformer` example. ([#82](https://github.com/pyg-team/pytorch-frame/pull/82))
- Added `TabNet` example ([#85](https://github.com/pyg-team/pytorch-frame/pull/85))
- Added dataset `tensorframe` and `col_stats` caching ([#84](https://github.com/pyg-team/pytorch-frame/pull/84))
- Added `TabTransformer`. ([#74](https://github.com/pyg-team/pytorch-frame/pull/74))
- Added `TabNet` ([#35](https://github.com/pyg-team/pytorch-frame/pull/35))
- Added text embedded stype, mapper and encoder. ([#78](https://github.com/pyg-team/pytorch-frame/pull/78))
- Added `ExcelFormer` example. ([#46](https://github.com/pyg-team/pytorch-frame/pull/46))
- Support inductive `DataFrame` to `TensorFrame` transformation ([#75](https://github.com/pyg-team/pytorch-frame/pull/75))
- Added `CatBoost` baseline and tuned `CatBoost` example. ([#73](https://github.com/pyg-team/pytorch-frame/pull/73))
- Added `na_strategy` as argument in `StypeEncoder`. ([#69](https://github.com/pyg-team/pytorch-frame/pull/69))
- Added `NAStrategy` class and impute NaN values in `MutualInformationSort`. ([#68](https://github.com/pyg-team/pytorch-frame/pull/68))
- Added `XGBoost` baseline and updated tuned `XGBoost` example. ([#57](https://github.com/pyg-team/pytorch-frame/pull/57))
- Added `CategoricalCatBoostEncoder` and `MutualInformationSort` transforms needed by ExcelFromer ([#52](https://github.com/pyg-team/pytorch-frame/pull/52))
- Added tutorial example script ([#54](https://github.com/pyg-team/pytorch-frame/pull/54))
- Added `ResNet` ([#48](https://github.com/pyg-team/pytorch-frame/pull/48))
- Added `ExcelFormerEncoder` ([#42](https://github.com/pyg-team/pytorch-frame/pull/42))
- Make `FTTransformer` take tensor_frame as input ([#45](https://github.com/pyg-team/pytorch-frame/pull/45))
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
- Added the `Adult Census Income` dataset ([#17](https://github.com/pyg-team/pytorch-frame/pull/17))
- Added column-level statistics and dataset materialization ([#15](https://github.com/pyg-team/pytorch-frame/pull/15))
- Added `FTTransformerConvs` ([#12](https://github.com/pyg-team/pytorch-frame/pull/12))
- Added documentation ([#13](https://github.com/pyg-team/pytorch-frame/pull/13))
- Added `DataLoader` capabilities ([#11](https://github.com/pyg-team/pytorch-frame/pull/11))
- Added `TensorFrame.index_select` functionality ([#10](https://github.com/pyg-team/pytorch-frame/pull/10))
- Added `Dataset.to_tensor_frame` functionality ([#9](https://github.com/pyg-team/pytorch-frame/pull/9))
- Added base classes `TensorEncoder`, `FeatureEncoder`, `TableConv`, `Decoder` ([#5](https://github.com/pyg-team/pytorch-frame/pull/5))
- Added `TensorFrame` ([#4](https://github.com/pyg-team/pytorch-frame/pull/4))
- Added the `Titanic` dataset ([#3](https://github.com/pyg-team/pytorch-frame/pull/3))
- Added `Dataset` base class ([#3](https://github.com/pyg-team/pytorch-frame/pull/3))

### Changed

- Fixed the `changelog-enforcer` ([#8](https://github.com/pyg-team/pytorch-frame/pull/8))

### Removed

- Removed dependency to `category_encoders` by adding custom `OrderedTargetStatisticsEncoder` implementation. ([#91](https://github.com/pyg-team/pytorch-frame/pull/91))
