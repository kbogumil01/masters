from .models.discriminator import Discriminator
from .models.enhancer import Enhancer
from .datamodule import VVCDataModule
from .trainer_module import TrainerModule
from .utils import weights_init
from .config import Config, TrainingMode
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch

# Funkcja pomocnicza do bezpiecznego ładowania wag
def load_weights_safe(model, path, prefix=""):
    print(f"Loading weights from: {path}")
    # 1. weights_only=False naprawia błąd w PyTorch 2.6+
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # 2. Obsługa checkpointu Lightning (.ckpt)
    if "state_dict" in ckpt:
        print(f"Detected Lightning checkpoint. Extracting weights for prefix '{prefix}'...")
        state_dict = ckpt["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            # Usuwamy prefix (np. "enhancer.") dodany przez TrainerModule
            if k.startswith(prefix):
                new_state_dict[k.replace(prefix, "")] = v
        
        # Jeśli słownik pusty, próbujemy ładować bez zmian (może prefix nie pasuje)
        if not new_state_dict:
            print("Warning: No keys matched the prefix. Trying raw state_dict.")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=True)
    else:
        # Stary format (zwykły słownik wag .pth)
        model.load_state_dict(ckpt, strict=True)
    print("Weights loaded successfully.")


if __name__ == "__main__":
    # Optymalizacja dla RTX 50xx / 40xx
    torch.set_float32_matmul_precision('medium')

    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="train",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="test",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="predict",
    )
    parser.add_argument(
        "--config",
        "-c",
        metavar="FILE",
        default="config.yaml",
        help="config file",
    )
    
    # Dodajemy obsługę --ckpt_path z CLI, żeby nie trzeba było edytować configu
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint file for testing/prediction"
    )

    args = parser.parse_args()

    config = Config.load(args.config)
    
    # Nadpisanie configu argumentem z CLI (jeśli podano)
    if args.ckpt_path:
        config.enhancer.load_from = args.ckpt_path

    data_module = VVCDataModule(
        dataset_config=config.dataset,
        dataloader_config=config.dataloader,
        test_full_frames=config.test_full_frames,
        fused_maps_dir=config.fused_maps_dir,
    )

    enhancer = Enhancer(
        config=config.enhancer,
    )

    if config.enhancer.load_from:
        # Używamy nowej funkcji do ładowania
        load_weights_safe(enhancer, config.enhancer.load_from, prefix="enhancer.")
    else:
        enhancer.apply(weights_init)

    discriminator = Discriminator(
        config=config.discriminator,
    )

    if config.discriminator.load_from:
        load_weights_safe(discriminator, config.discriminator.load_from, prefix="discriminator.")
        print("loaded discriminator")
    else:
        discriminator.apply(weights_init)

    module = TrainerModule(
        config.trainer,
        enhancer,
        discriminator,
        test_full_frames=config.test_full_frames,
    )

    # wandb_logger = WandbLogger(
    #     project="vvc-enhancer",
    # )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=config.trainer.current.epochs,
        precision=config.trainer.precision,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        num_sanity_val_steps=0,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath="checkpoints", filename="{epoch}"),
        ],
        # logger=wandb_logger,
    )

    if args.train:
        trainer.fit(module, data_module)

    if config.enhancer.save_to and args.train:
        torch.save(enhancer.state_dict(), config.enhancer.save_to)

    if config.discriminator.save_to and args.train:
        torch.save(discriminator.state_dict(), config.discriminator.save_to)

    if args.test:
        trainer.test(module, data_module)

    if args.predict:
        trainer.predict(module, data_module)