import os
import torch

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
from transformers.trainer_utils import SchedulerType
from huggingface_hub import login

from config import Config, get_config
from data import get_train_dataset


def main(args: Config):

    """
    Orchestre le processus d'entraînement : sélection du modèle, configuration matérielle et lancement du trainer.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
                                                                          
    # Sélection du modèle via entrée utilisateur
    print("Sélection du modèle à entraîner")
    model_choice = -1
    
    while model_choice not in (0, 1):
        try:
            model_choice = int(input("Choisir : 0) Modèle vierge (e5-small) ou 1) Modèle existant (headline-commande) : "))
        except ValueError:
            pass

    model_map = {
        0: "intfloat/multilingual-e5-small",
        1: "meet-magnet/headline-commande-entreprise-IMT"
    }
    
    model = SentenceTransformer(model_map[model_choice])
 
      
    loss = losses.MultipleNegativesRankingLoss(model=model)

    has_cuda = torch.cuda.is_available()
    supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    
    supports_fp16 = False
    if has_cuda:
        try:
            major_cc, _ = torch.cuda.get_device_capability(0)
            supports_fp16 = major_cc >= 7
        except Exception:
            supports_fp16 = False

    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        
        lr_scheduler_type=SchedulerType.WARMUP_STABLE_DECAY,
        lr_scheduler_kwargs={
            "num_decay_steps": int(args.decay_ratio * max(1, (len(train_dataset) // max(1, args.batch_size)) * max(1, args.num_epochs)))
        },

        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        
        # Gestion de la précision mixte
        bf16=supports_bf16,
        fp16=(supports_fp16 and not supports_bf16),
    )

    trainer = SentenceTransformerTrainer(
        model=model,                                
        train_dataset=train_dataset,                 
        loss=loss,
        args=train_args,
    )
     
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    config = get_config()
    main(config)