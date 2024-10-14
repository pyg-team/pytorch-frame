import time
from argparse import ArgumentParser
from contextlib import nullcontext

import torch
from line_profiler import profile

from torch_frame import NAStrategy, stype
from torch_frame.config import ModelConfig
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.config.text_tokenizer import TextTokenizerConfig
from torch_frame.datasets import FakeDataset
from torch_frame.nn import (
    EmbeddingEncoder,
    ExcelFormerEncoder,
    LinearBucketEncoder,
    LinearEmbeddingEncoder,
    LinearEncoder,
    LinearModelEncoder,
    LinearPeriodicEncoder,
    MultiCategoricalEmbeddingEncoder,
    StackEncoder,
    StypeEncoder,
    StypeWiseFeatureEncoder,
    TimestampEncoder,
)
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_frame.testing.text_tokenizer import (
    RandomTextModel,
    WhiteSpaceHashTokenizer,
)

parser = ArgumentParser(description='Benchmark for encoder')
parser.add_argument(
    '--stype-kv', nargs=2, action='append', required=True,
    help='Specify the stype(s) and corresponding encoder(s) to run. '
    'Possible stypes are: categorical, embedding, multicategorical, '
    'numerical, sequence_numerical, text_embedded, text_tokenized, '
    'timestamp. Possible encoders are: embedding, excel_former, linear, '
    'linear_bucket, linear_embedding, linear_model, linear_periodic, '
    'multicategorical_embedding, stack, timestamp. '
    'Example: --stype-kv categorical embedding --stype-kv numerical linear')
parser.add_argument('--num-rows', type=int, default=8192)
parser.add_argument('--out-channels', type=int, default=128)
parser.add_argument('--with-nan', action='store_true')
parser.add_argument('--runs', type=int, default=1000)
parser.add_argument('--warmup-size', type=int, default=200)
parser.add_argument('--torch-profile', action='store_true')
parser.add_argument('--line-profile', action='store_true')
parser.add_argument(
    '--line-profile-level', type=str,
    choices=['stypewise_forward', 'stype_forward',
             'encode_forward'], default='encode_forward')
parser.add_argument('--device', type=str, choices=['cpu'], default='cpu')

args = parser.parse_args()

stype_str2stype = {
    'categorical': stype.categorical,
    'embedding': stype.embedding,
    'multicategorical': stype.multicategorical,
    'numerical': stype.numerical,
    'sequence_numerical': stype.sequence_numerical,
    'text_embedded': stype.text_embedded,
    'text_tokenized': stype.text_tokenized,
    'timestamp': stype.timestamp,
}

encoder_str2encoder = {
    'embedding': EmbeddingEncoder,
    'excel_former': ExcelFormerEncoder,
    'linear': LinearEncoder,
    'linear_bucket': LinearBucketEncoder,
    'linear_embedding': LinearEmbeddingEncoder,
    'linear_model': LinearModelEncoder,
    'linear_periodic': LinearPeriodicEncoder,
    'multicategorical_embedding': MultiCategoricalEmbeddingEncoder,
    'stack': StackEncoder,
    'timestamp': TimestampEncoder,
}

encoder_str2encoder_cls_kwargs = {
    'embedding': {
        "na_strategy": NAStrategy.MOST_FREQUENT,
    },
    'excel_former': {},
    'linear': {
        "na_strategy": NAStrategy.MEAN,
    },
    'linear_bucket': {
        "na_strategy": NAStrategy.MEAN,
    },
    'linear_embedding': {},
    'linear_model': {
        "col_to_model_cfg": {
            "text_tokenized_1":
            ModelConfig(model=RandomTextModel(args.out_channels * 4),
                        out_channels=args.out_channels * 4),
            "text_tokenized_2":
            ModelConfig(model=RandomTextModel(args.out_channels * 2),
                        out_channels=args.out_channels * 2)
        },
    },
    'linear_periodic': {
        "na_strategy": NAStrategy.MEAN,
    },
    'multicategorical_embedding': {
        "na_strategy": NAStrategy.ZEROS
    },
    'stack': {},
    'timestamp': {
        "na_strategy": NAStrategy.MEDIAN_TIMESTAMP
    },
}


def make_stype_encoder_dict() -> dict[stype, StypeEncoder]:
    stype_encoder_dict = {}
    for stype_str, encoder_str in args.stype_kv:
        encoder_kwargs = encoder_str2encoder_cls_kwargs[encoder_str]
        stype_encoder_dict[
            stype_str2stype[stype_str]] = encoder_str2encoder[encoder_str](
                **encoder_kwargs)

    return stype_encoder_dict


def make_fake_dataset() -> FakeDataset:
    stypes = [stype_str2stype[stype_str] for stype_str, _ in args.stype_kv]
    dataset = FakeDataset(
        num_rows=args.num_rows, with_nan=args.with_nan, stypes=stypes,
        col_to_text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(out_channels=args.out_channels *
                                           2, ),
            batch_size=None,
        ), col_to_text_tokenizer_cfg=TextTokenizerConfig(
            text_tokenizer=WhiteSpaceHashTokenizer(),
            batch_size=None,
        ))
    dataset.materialize(torch.device(args.device))

    return dataset


def run_profiling():
    if args.line_profile and args.torch_profile:
        raise ValueError(
            'You cannot enable both line profiling and torch profiling '
            'at the same time.')
    if not all(stype_str in stype_str2stype for stype_str, _ in args.stype_kv):
        raise ValueError('Invalid stype string.')
    if not all(encoder_str in encoder_str2encoder
               for _, encoder_str in args.stype_kv):
        raise ValueError('Invalid encoder string.')
    if 'linear_model' in [encoder_str for _, encoder_str in args.stype_kv]:
        if 'text_tokenized' not in [
                stype_str for stype_str, _ in args.stype_kv
        ]:
            raise ValueError(
                'If you want to use linear_model encoder, you must specify '
                'text_tokenized stype.')

    if args.warmup_size == 0:
        args.warmup_size = args.runs // 5
        print('Warm-up size is not specified. Using 1/5 of the runs as the '
              'warm-up size.')

    print('BENCHMARK STARTS')
    print('Preparing the data...')

    dataset = make_fake_dataset()
    stype_encoder_dict = make_stype_encoder_dict()

    if args.line_profile:
        if args.line_profile_level == 'encode_forward':
            for encoder in stype_encoder_dict.values():
                encoder.encode_forward = profile(encoder.encode_forward)
        elif args.line_profile_level == 'stype_forward':
            for encoder in stype_encoder_dict.values():
                encoder.forward = profile(encoder.forward)

    tensor_frame = dataset.tensor_frame
    encoder = StypeWiseFeatureEncoder(
        out_channels=args.out_channels, col_stats=dataset.col_stats,
        col_names_dict=tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict)

    if args.line_profile and args.line_profile_level == 'stypewise_forward':
        encoder.forward = profile(encoder.forward)

    profiler_context = torch.autograd.profiler.profile(
    ) if args.torch_profile else nullcontext()

    print('Performing warm-up stage...')
    for _ in range(args.warmup_size):
        encoder(tensor_frame)

    print('Running benchmark...')
    with profiler_context as prof:
        start = time.perf_counter()
        for _ in range(args.runs):
            encoder(tensor_frame)
        elapsed = time.perf_counter() - start
        latency = elapsed / args.runs
        print(f'Latency: {latency:.6f}s')

    if args.torch_profile:
        sort_by = f'self_{args.device}_time_total'
        print(prof.key_averages().table(sort_by=sort_by, row_limit=20))


if __name__ == '__main__':
    run_profiling()
