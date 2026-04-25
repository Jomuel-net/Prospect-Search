import os
import torch
from dotenv import load_dotenv   
from pathlib import Path                         

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
    
    Orchestre le processus d'entraînement : sélection du modèle, sélection du csv et configuration matérielle et lancement du trainer.
    
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"      

    # chemin où se trouve le dossier src en local
    def get_src_root() -> Path:
        """
        Retourne le chemin absolu du dossier src/ (où se trouve ce fichier config.py)
        """
        return Path(__file__).resolve().parent

    SRC_ROOT = get_src_root()                                       

    # Charge les variables depuis .env (si présent dans src/)
    load_dotenv(SRC_ROOT / ".env")    

    # Sélection du csv via entrée utilisateur

    print("Selection du csv à exploiter") 
    csv_choice = -1
    
    while csv_choice not in range(0,6):
        try:
            csv_choice = int(input( 
                        "Choisir parmi : \n"
                        " 0 - csv_content_french \n"
                        " 1 - csv_content_english \n"
                        " 2 - csv_content_german \n"                                                                            # les csv doivent être renommés ainsi
                        " 3 - csv_content_spanish \n"
                        " 4 - csv_content_chinese \n"
                        " 5 - csv_content_all_languages_with_duplicate \n"
                        " 6 - csv_content_all_languages_without_duplicate \n"
            ))

        except ValueError:
            pass

    csv_map ={
        0: "csv_content_french",
        1: "csv_content_english",
        2: "csv_content_german",
        3: "csv_content_spanish",
        4: "csv_content_chinese",
        5: "csv_content_all_languages_with_duplicate",
        6: "csv_content_all_languages_without_duplicate"
    }

    args=Config(data_path = os.getenv("DATA_PATH", str(SRC_ROOT /  csv_map[csv_choice])))                                       # personnalise le datapath à exploiter        
    train_dataset = get_train_dataset(args)                        
        

    # Sélection du modèle via entrée utilisateur
    print("Sélection du modèle à entraîner")
    model_choice = -1
    
    while model_choice not in (0, 1):
        try:
            model_choice = int(input("Choisir : 0) Modèle vierge (e5-small) ou 1) Modèle existant (content-commande) : "))
        except ValueError:
            pass

    model_map = {
        0: "intfloat/multilingual-e5-small",
        1: "meet-magnet/content-commande-entreprise-IMT"
    }
    
    model = SentenceTransformer(model_map[model_choice])

    loss = losses.MultipleNegativesRankingLoss(model=model)                
    n_steps = len(train_dataset) // args.batch_size * args.num_epochs       

    #detecte les capacités du Hard Ware

    has_cuda = torch.cuda.is_available()     
    supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()     
    

    #besoin de 7.0
    supports_fp16 = False
    if has_cuda:
        try:
            major_cc, _ = torch.cuda.get_device_capability(0)
            supports_fp16 = major_cc >= 7
        except Exception:
            supports_fp16 = False

    # definition des parametres d'entrainement

    train_args = SentenceTransformerTrainingArguments(

        output_dir=args.output_dir,                

        # steps/batch
        num_train_epochs=args.num_epochs,           
        gradient_accumulation_steps=args.gradient_accumulation_steps,       
        per_device_train_batch_size=args.batch_size,      

        # optimizer
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,                   

        # LR scheduler (cámbialo a LINEAR si tu versión de transformers no soporta WARMUP_STABLE_DECAY)
        lr_scheduler_type=SchedulerType.WARMUP_STABLE_DECAY,
        lr_scheduler_kwargs={"num_decay_steps": int(args.decay_ratio * max(1, (len(train_dataset) // max(1, args.batch_size)) * max(1, args.num_epochs)))},

        # loader
        dataloader_num_workers=args.num_workers,            
        dataloader_prefetch_factor=args.prefetch_factor,
        seed=args.seed,                                     
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        
        # precision mixte: seulement si c'est supporté

        bf16=supports_bf16,
        fp16=(supports_fp16 and not supports_bf16),
        
    )

        
    #Entrainement
    trainer = SentenceTransformerTrainer(
        model=model,                                
        train_dataset=train_dataset,                 
        loss=loss,
        args=train_args,
    )
     

    print("==Demarrage de l'entrainement==")
    
    trainer.train()                                 # activation

    trainer.save_model(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    config = get_config()
    main(config)


    