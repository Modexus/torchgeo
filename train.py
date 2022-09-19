#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""torchgeo model training script."""

import os
from typing import Any, Dict, List, Tuple, Type, cast

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.strategies.fully_sharded import DDPFullyShardedStrategy

import wandb
from torchgeo.datamodules import (
    BigEarthNetDataModule,
    ChesapeakeCVPRDataModule,
    COWCCountingDataModule,
    CycloneDataModule,
    ETCI2021DataModule,
    EuroSATDataModule,
    InriaAerialImageLabelingDataModule,
    LandCoverAIDataModule,
    NAIPCDLDataModule,
    NAIPChesapeakeDataModule,
    OSCDDataModule,
    RESISC45DataModule,
    SEN12MSDataModule,
    So2SatDataModule,
    UCMercedDataModule,
)
from torchgeo.trainers import (
    BYOLTask,
    CAETask,
    ClassificationTask,
    EmbeddingEvaluator,
    MAETask,
    MSAETask,
    MSNTask,
    MultiLabelClassificationTask,
    RegressionTask,
    SemanticSegmentationTask,
    Tile2VecTask,
    VICRegTask,
)

TASK_TO_MODULES_MAPPING: Dict[
    str, Tuple[Type[LightningModule], Type[LightningDataModule]]
] = {
    "bigearthnet": (MultiLabelClassificationTask, BigEarthNetDataModule),
    "byol": (BYOLTask, ChesapeakeCVPRDataModule),
    "chesapeake_cvpr": (SemanticSegmentationTask, ChesapeakeCVPRDataModule),
    "cowc_counting": (RegressionTask, COWCCountingDataModule),
    "cyclone": (RegressionTask, CycloneDataModule),
    "eurosat": (ClassificationTask, EuroSATDataModule),
    "etci2021": (SemanticSegmentationTask, ETCI2021DataModule),
    "inria": (SemanticSegmentationTask, InriaAerialImageLabelingDataModule),
    "landcoverai": (SemanticSegmentationTask, LandCoverAIDataModule),
    "naipchesapeake": (SemanticSegmentationTask, NAIPChesapeakeDataModule),
    "oscd": (SemanticSegmentationTask, OSCDDataModule),
    "resisc45": (ClassificationTask, RESISC45DataModule),
    "sen12ms": (SemanticSegmentationTask, SEN12MSDataModule),
    "so2sat": (ClassificationTask, So2SatDataModule),
    "ucmerced": (ClassificationTask, UCMercedDataModule),
    "classification_naipcdl": (ClassificationTask, NAIPCDLDataModule),
    "identity_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "tile2vec_naipcdl_train": (Tile2VecTask, NAIPCDLDataModule),
    "tile2vec_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "byol_naipcdl_train": (BYOLTask, NAIPCDLDataModule),
    "byol_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "vicreg_naipcdl_train": (VICRegTask, NAIPCDLDataModule),
    "vicreg_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "mae_naipcdl_train": (MAETask, NAIPCDLDataModule),
    "mae_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "msn_naipcdl_train": (MSNTask, NAIPCDLDataModule),
    "msn_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "msae_naipcdl_train": (MSAETask, NAIPCDLDataModule),
    "msae_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "cae_naipcdl_train": (CAETask, NAIPCDLDataModule),
    "cae_naipcdl_evaluate": (EmbeddingEvaluator, NAIPCDLDataModule),
    "mae_bigearthnet_train": (MAETask, BigEarthNetDataModule),
    "mae_bigearthnet_evaluate": (EmbeddingEvaluator, BigEarthNetDataModule),
    "cae_bigearthnet_train": (CAETask, BigEarthNetDataModule),
    "cae_bigearthnet_evaluate": (EmbeddingEvaluator, BigEarthNetDataModule),
}


def set_up_omegaconf() -> DictConfig:
    """Loads program arguments from either YAML config files or command line arguments.

    This method loads defaults/a schema from "conf/defaults.yaml" as well as potential
    arguments from the command line. If one of the command line arguments is
    "config_file", then we additionally read arguments from that YAML file. One of the
    config file based arguments or command line arguments must specify task.name. The
    task.name value is used to grab a task specific defaults from its respective
    trainer. The final configuration is given as merge(task_defaults, defaults,
    config file, command line). The merge() works from the first argument to the last,
    replacing existing values with newer values. Additionally, if any values are
    merged into task_defaults without matching types, then there will be a runtime
    error.

    Returns:
        an OmegaConf DictConfig containing all the validated program arguments

    Raises:
        FileNotFoundError: when ``config_file`` does not exist
        ValueError: when ``task.name`` is not a valid task
    """
    conf = OmegaConf.load("conf/defaults.yaml")
    command_line_conf = OmegaConf.from_cli()

    if "config_file" in command_line_conf:
        config_fn = command_line_conf.config_file
        if not os.path.isfile(config_fn):
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

        user_conf = OmegaConf.load(config_fn)
        conf = OmegaConf.merge(conf, user_conf)

    conf = OmegaConf.merge(  # Merge in any arguments passed via the command line
        conf, command_line_conf
    )

    # These OmegaConf structured configs enforce a schema at runtime, see:
    # https://omegaconf.readthedocs.io/en/2.0_branch/structured_config.html#merging-with-other-configs
    task_name = conf.experiment.task
    task_config_fn = os.path.join("conf", f"{task_name}.yaml")
    if task_name == "test":
        task_conf = OmegaConf.create()
    elif os.path.exists(task_config_fn):
        task_conf = cast(DictConfig, OmegaConf.load(task_config_fn))
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    conf = OmegaConf.merge(task_conf, conf)
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright

    return conf


def main(conf: DictConfig) -> None:
    """Main training loop."""
    ######################################
    # Setup output directory
    ######################################

    experiment_name = conf.experiment.name
    task_name = conf.experiment.task
    run_name = f"{conf.experiment.module.model}-{conf.experiment.module.encoder_name}"
    try:
        if conf.experiment.module.project:
            run_name += "-project"
    except ConfigAttributeError:
        pass

    try:
        if conf.experiment.module.imagenet_pretrained:
            run_name += "-imagenet_pretrained"
    except ConfigAttributeError:
        pass

    try:
        if conf.experiment.module.projector_embeddings:
            run_name += "-projector_embeddings"
    except ConfigAttributeError:
        pass

    try:
        if conf.experiment.module.checkpoint_path == "":
            run_name += "-untrained"
        else:
            run_name += "-trained"
    except ConfigAttributeError:
        pass

    if os.path.isfile(conf.program.output_dir):
        raise NotADirectoryError("`program.output_dir` must be a directory")
    os.makedirs(conf.program.output_dir, exist_ok=True)

    experiment_dir = os.path.join(conf.program.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    run_dir = os.path.join(experiment_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if len(os.listdir(run_dir)) > 0:
        if conf.program.overwrite:
            print(
                f"WARNING! The experiment directory, {run_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            print(
                f"The experiment directory, {run_dir}, already exists and isn't "
                + "empty. We don't want to overwrite any existing results, no data will be saved."
            )

    with open(os.path.join(run_dir, "experiment_config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    ######################################
    # Choose task to run based on arguments or configuration
    ######################################
    # Convert the DictConfig into a dictionary so that we can pass as kwargs.
    task_args = cast(Dict[str, Any], OmegaConf.to_object(conf.experiment.module))
    datamodule_args = cast(
        Dict[str, Any], OmegaConf.to_object(conf.experiment.datamodule)
    )

    try:
        if len(ckpt_name := conf.experiment.module.load_checkpoint) > 0:
            experiment_train_dir = experiment_dir.replace("evaluate", "train")
            task_args["checkpoint_path"] = os.path.join(
                experiment_train_dir, run_name, ckpt_name
            )
    except ConfigAttributeError:
        pass

    datamodule: LightningDataModule
    task: LightningModule
    if task_name in TASK_TO_MODULES_MAPPING:
        task_class, datamodule_class = TASK_TO_MODULES_MAPPING[task_name]
        task = task_class(**task_args)
        datamodule = datamodule_class(**datamodule_args)
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    ######################################
    # Setup trainer
    ######################################
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))
    logger_args = cast(Dict[str, Any], OmegaConf.to_object(conf.logger))
    log_dir = os.path.join(conf.program.log_dir, run_name)

    logger: pl_loggers.LightningLoggerBase
    if "name" not in logger_args or logger_args["name"] == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(log_dir, name=experiment_name)
    elif logger_args["name"] == "wandb":
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        offline = logger_args.get("offline", False)
        logger = pl_loggers.WandbLogger(
            name=run_name,
            save_dir=log_dir,
            project=logger_args.get("project_name", "torchgeo"),
            group=experiment_name,
            log_model=(not offline),
            offline=offline,
        )
        if (
            "devices" not in trainer_args
            or len(trainer_args["devices"]) == 1
            and wandb.run is not None
        ):
            wandb.run.log_code("torchgeo")
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    callbacks: List[Callback] = []
    if conf.program.overwrite:
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir,
            save_last=True,
            save_top_k=-1,
            every_n_epochs=10,
            monitor="train_loss",
        )
        callbacks.append(checkpoint_callback)

    if "early_stopping" in trainer_args:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=18
        )
        callbacks.append(early_stopping_callback)

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(learning_rate_monitor)

    trainer_args["callbacks"] = callbacks
    trainer_args["logger"] = logger
    trainer_args["default_root_dir"] = experiment_dir

    trainer = Trainer(
        **trainer_args,
        strategy=DeepSpeedStrategy(
            stage=1,
            cpu_checkpointing=False,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
            logging_batch_size_per_gpu=task_args["batch_size"],
        ),
    )

    if trainer_args.get("auto_lr_find"):
        trainer.tune(model=task, datamodule=datamodule)

    ######################################
    # Run experiment
    ######################################
    try:
        if len(ckpt_name := conf.experiment.module.resume_checkpoint) > 0:
            ckpt_path = os.path.join(experiment_dir, run_name, ckpt_name)
    except ConfigAttributeError:
        ckpt_path = None

    if conf.experiment.run.fit:
        trainer.fit(model=task, datamodule=datamodule, ckpt_path=ckpt_path)
    if conf.experiment.run.test:
        trainer.test(model=task, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # Taken from https://github.com/pangeo-data/cog-best-practices
    _rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_CACHEMAX": "10000",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(_rasterio_best_practices)

    conf = set_up_omegaconf()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    seed_everything(conf.program.seed)

    # Main training procedure
    main(conf)
