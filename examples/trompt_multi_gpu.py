import argparse
import logging
import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchmetrics
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from torch_frame.data import DataLoader
from torch_frame.datasets import TabularBenchmark
from torch_frame.nn import Trompt


def prepare_dataset(dataset_str: str) -> TabularBenchmark:
    path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "..",
        "data",
        dataset_str,
    )
    materialized_path = osp.join(path, 'materialized_data.pt')
    if dist.get_rank() == 0:
        logging.info(f"Preparing dataset '{dataset_str}' from '{path}'")
        dataset = TabularBenchmark(root=path, name=dataset_str)
        logging.info("Materializing dataset")
        dataset.materialize(path=materialized_path)

    dist.barrier()
    if dist.get_rank() != 0:
        logging.info(f"Preparing dataset '{dataset_str}' from '{path}'")
        dataset = TabularBenchmark(root=path, name=dataset_str)
        logging.info("Loading materialized dataset")
        dataset.materialize(path=materialized_path)

    dist.barrier()
    return dataset


def train(
    model: DistributedDataParallel,
    epoch: int,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    metric: torchmetrics.Metric,
    rank: int,
) -> float:
    model.train()
    loss_accum = torch.tensor(0.0, device=rank, dtype=torch.float32)
    for tf in tqdm(
            loader,
            desc=f"Epoch {epoch:03d} (train)",
            disable=rank != 0,
    ):
        tf = tf.to(rank)
        # [batch_size, num_layers, num_classes]
        out = model(tf)

        with torch.no_grad():
            metric.update(out.mean(dim=1).argmax(dim=-1), tf.y)

        _, num_layers, num_classes = out.size()
        # [batch_size * num_layers, num_classes]
        pred = out.view(-1, num_classes)
        y = tf.y.repeat_interleave(num_layers)
        # Layer-wise logit loss
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_accum += loss

    # The number of samples is guaranteed to be the same across all ranks
    # because of DistributedSampler(drop_last=True).
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    metric_value = metric.compute()
    metric.reset()
    return loss_accum, metric_value


@torch.no_grad()
def test(
    model: DistributedDataParallel,
    epoch: int,
    loader: DataLoader,
    metric: torchmetrics.Metric,
    rank: int,
    desc: str,
) -> float:
    model.eval()
    for tf in tqdm(
            loader,
            desc=f"Epoch {epoch:03d} ({desc})",
            disable=rank != 0,
    ):
        tf = tf.to(rank)
        # [batch_size, num_layers, num_classes] -> [batch_size, num_classes]
        pred = model(tf).mean(dim=1)
        pred_class = pred.argmax(dim=-1)
        metric.update(pred_class, tf.y)

    metric_value = metric.compute()
    metric.reset()
    return metric_value


def run(rank: int, world_size: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    logging.basicConfig(
        format=(f"[rank={rank}/{world_size}] "
                f"[%(asctime)s] %(levelname)s: %(message)s"),
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running on rank {rank} of {world_size}")
    dataset = prepare_dataset(args.dataset)
    assert dataset.task_type.is_classification

    # Ensure train, val and test splits are the same across all ranks by
    # setting the seed on each rank.
    torch.manual_seed(args.seed)
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = (
        dataset[:0.7],
        dataset[0.7:0.79],
        dataset[0.79:],
    )
    train_loader = DataLoader(
        train_dataset.tensor_frame,
        batch_size=args.batch_size,
        sampler=DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True,
        ),
    )
    val_loader = DataLoader(
        val_dataset.tensor_frame,
        batch_size=args.batch_size,
        sampler=DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False,
        ),
    )
    test_loader = DataLoader(
        test_dataset.tensor_frame,
        batch_size=args.batch_size,
        sampler=DistributedSampler(
            test_dataset,
            shuffle=False,
            drop_last=False,
        ),
    )
    model = Trompt(
        channels=args.channels,
        out_channels=dataset.num_classes,
        num_prompts=args.num_prompts,
        num_layers=args.num_layers,
        col_stats=dataset.col_stats,
        col_names_dict=train_dataset.tensor_frame.col_names_dict,
    ).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    metrics_kwargs = {
        "task": "multiclass",
        "num_classes": dataset.num_classes,
    }
    train_metric = torchmetrics.Accuracy(**metrics_kwargs).to(rank)
    val_metric = torchmetrics.Accuracy(**metrics_kwargs).to(rank)
    test_metric = torchmetrics.Accuracy(**metrics_kwargs).to(rank)

    best_val_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train(
            model,
            epoch,
            train_loader,
            optimizer,
            train_metric,
            rank,
        )
        val_acc = test(
            model,
            epoch,
            val_loader,
            val_metric,
            rank,
            'val',
        )
        test_acc = test(
            model,
            epoch,
            test_loader,
            test_metric,
            rank,
            'test',
        )
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Test Acc: {test_acc:.4f}")

        lr_scheduler.step()

    if rank == 0:
        print(f"Best Val Acc: {best_val_acc:.4f}, "
              f"Best Test Acc: {best_test_acc:.4f}")

    dist.destroy_process_group()
    logging.info("Process group destroyed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="california")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num_prompts", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
