import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

def get_src_root() -> Path:
    return Path(__file__).resolve().parent

SRC_ROOT = get_src_root()
load_dotenv(SRC_ROOT / ".env")

def get_output_dir_with_timestamp() -> str:
    """Génère un chemin de sortie unique basé sur l'horodatage actuel."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(SRC_ROOT / f"output_finetuned_model_{timestamp}")

@dataclass(frozen=True)
class Config:
    # === Modèle ===
    model_id: str = ""
    prefix: tuple = ("query: ", "passage: ")

    # === Données ===
    data_path: str = os.getenv("DATA_PATH", str(SRC_ROOT / "new_train_data_headline.csv"))             
    seed: int = 42
    limit: int = None  # None pour utiliser tout le dataset
    num_workers: int = 8
    prefetch_factor: int = 2
    cache_data_path: str = os.getenv("CACHE_DIR", str(SRC_ROOT / "cache" / "cached_data"))
    needs_prefix: bool = True  # Ajout automatique des préfixes query/passage

    # === Hyperparamètres d'entraînement ===
    batch_size: int = 16
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    decay_ratio: float = 0.1

    # === Évaluation et Logging ===
    eval_steps: int = 1000
    evaluation_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    logging_steps: int = 1000

    # === Sortie ===
    output_dir: str = os.getenv("OUTPUT_DIR", get_output_dir_with_timestamp())
    
    def __post_init__(self):
        assert len(self.prefix) == 2, "Le préfixe doit être un tuple de longueur 2."

# --- Profils de configuration ---

test_config = Config(
    model_id="intfloat/multilingual-e5-small",
    batch_size=2,
    limit=100,
)

small_config = Config(
    model_id="intfloat/multilingual-e5-small",
    batch_size=32,
)

base_config = Config(
    model_id="intfloat/multilingual-e5-base",
    batch_size=16,
)

large_config = Config(
    model_id="intfloat/multilingual-e5-large",
    batch_size=8,
)

content_config = Config(
    model_id="intfloat/multilingual-e5-small",
    batch_size=8,
    needs_prefix=False,
)

def get_config() -> Config:
    """
    Parse les arguments CLI pour sélectionner la configuration appropriée.
    Usage: python main.py --config [test|small|base|large|content]
    """
    parser = argparse.ArgumentParser(description="Configuration de l'entraînement")
    parser.add_argument(
        "--config",
        type=str,
        default="test",
        choices=["test", "small", "base", "large", "content"],
        help="Sélectionner le profil de configuration"
    )
    
    args = parser.parse_args()

    
    configs = {
        "test": test_config,
        "small": small_config,
        "base": base_config,
        "large": large_config,
        "content": content_config,
    }

    selected_config = configs.get(args.config)

    if selected_config is None:
        raise ValueError(f"Configuration inconnue: {args.config}")

    return selected_config