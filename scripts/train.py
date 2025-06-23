import datetime
import json
import os
import tempfile
from typing import Tuple

import numpy as np
import ray
import ray.train as train
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Dataset
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import BertModel
from typing_extensions import Annotated, Optional

from scripts import data, utils
from scripts.config import EFS_DIR, MLFLOW_TRACKING_URI
from scripts.models import FinetunedLLM

app = typer.Typer()

def train_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:  
    
    """Train step for the model."""
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradients
        z = model(batch)  # forward pass
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss

def eval_step(
    ds: Dataset, batch_size: int, model: nn.Module, num_classes: int, loss_fn: torch.nn.modules.loss._WeightedLoss
) -> Tuple[float, np.array, np.array]:  # pragma: no cover, tested via train workload
    """Eval step."""
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict) -> None: 
    """Training loop that each worker will execute.

    Args:
        config (dict): arguments to use for training.
    """
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # Get datasets
    utils.set_seeds()
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(llm=llm, dropout_p=dropout_p, embedding_dim=llm.config.hidden_size, num_classes=num_classes)
    model = train.torch.prepare_model(model)

    # Training components
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    # Training
    num_workers = train.get_context().get_world_size()
    batch_size_per_worker = batch_size // num_workers
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(train_ds, batch_size_per_worker, model, num_classes, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        with tempfile.TemporaryDirectory() as dp:
            if isinstance(model, DistributedDataParallel):  # cpu
                model.module.save(dp=dp)
            else:
                model.save(dp=dp)
            metrics = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, val_loss=val_loss)
            checkpoint = Checkpoint.from_directory(dp)
            train.report(metrics, checkpoint=checkpoint)


@app.command()
def train_model(
    experiment_name: str = typer.Option(..., help="Name of the experiment."),
    dataset_loc: str = typer.Option(..., help="Dataset location."),
    train_loop_config: str = typer.Option(..., help="Training loop config (JSON string)."),
    num_workers: int = typer.Option(1, help="Number of workers."),
    cpu_per_worker: int = typer.Option(1, help="CPUs per worker."),
    gpu_per_worker: int = typer.Option(0, help="GPUs per worker."),
    num_samples: Optional[int] = typer.Option(None, help="Number of samples to use."),
    num_epochs: Optional[int] = typer.Option(None, help="Number of epochs."),
    batch_size: Optional[int] = typer.Option(None, help="Batch size."),
    results_fp: Optional[str] = typer.Option(None, help="Path to save results.")
):
    """Main train function to train our model as a distributed workload.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        dataset_loc (str): location of the dataset.
        train_loop_config (str): arguments to use for training.
        num_workers (int, optional): number of workers to use for training. Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker. Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker. Defaults to 0.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_fp (str, optional): filepath to save results to. Defaults to None.

    Returns:
        ray.air.result.Result: training results.
    """
    # Set up
    train_loop_config = json.loads(train_loop_config)
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
    )

    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )

    # Run config
    run_config = RunConfig(callbacks=[mlflow_callback], checkpoint_config=checkpoint_config, storage_path=EFS_DIR, local_dir=EFS_DIR)

    # Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config["num_samples"])
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    train_loop_config["num_classes"] = len(tags)

    # Dataset config
    options = ray.data.ExecutionOptions(preserve_order=True)
    dataset_config = DataConfig(datasets_to_split=["train"], execution_options=options)

    # Preprocess
    preprocessor = data.CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    # Train
    results = trainer.fit()
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=results.metrics["trial_id"]),
        "params": results.config["train_loop_config"],
        "metrics": utils.dict_to_list(results.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
    }
    # logger.info(json.dumps(d, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d, results_fp)
    return results



if __name__ == "__main__":  # pragma: no cover, application
    if ray.is_initialized():
        ray.shutdown()
    # ray.init(runtime_env={"env_vars": {"kavina-a": os.environ["kavina-a"]}})
    ray.init()
    app()


