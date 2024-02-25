# python cer_model_train.py --cer_model_id microsoft/deberta-v3-base --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
# rm ~/.cache/wandb/artifacts/
# rm -rf ~/checkpoints
# python cer_model_train.py --cer_model_id microsoft/deberta-v3-large  --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
# rm ~/.cache/wandb/artifacts/
# rm -rf ~/checkpoints
# python cer_model_train.py --cer_model_id KISTI-AI/scideberta-cs  --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
# rm ~/.cache/wandb/artifacts/
# rm -rf ~/checkpoints
# python cer_model_train.py --cer_model_id KISTI-AI/Scideberta-full  --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
# rm ~/.cache/wandb/artifacts/
# rm -rf ~/checkpoints
# python cer_model_train.py --cer_model_id Clinical-AI-Apollo/Medical-NER  --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
# rm -rf ~/.cache/wandb/artifacts/
# rm -rf ~/checkpoints
# python cer_model_train.py --cer_model_id yikuan8/Clinical-Longformer --use_LoRA=False --run_suffix='-noLoRA' --random_seed=42 --max_seq_len=4096
# rm -rf ~/.cache/wandb/artifacts/
# rm -rf ~/checkpoints
# python cer_model_train.py --cer_model_id yikuan8/Clinical-BigBird --use_LoRA=True --run_suffix='-LoRA' --random_seed=42 --max_seq_len=4096 --fp16 True
# rm -rf ~/.cache/wandb/artifacts/
# rm -rf ~/checkpoints
# python cer_model_train.py --cer_model_id bvanaken/CORe-clinical-outcome-biobert-v1  --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
rm -rf ~/.cache/wandb/artifacts/
rm -rf ~/checkpoints
python cer_model_train.py --cer_model_id arashpcc/Bio_ClinicalBERT  --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
rm -rf ~/.cache/wandb/artifacts/
rm -rf ~/checkpoints
python cer_model_train.py --cer_model_id allenai/biomed_roberta_base --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
rm -rf ~/.cache/wandb/artifacts/
rm -rf ~/checkpoints
python cer_model_train.py --cer_model_id medicalai/ClinicalBERT --use_LoRA=False --run_suffix='-noLoRA' --random_seed=10
rm -rf ~/.cache/wandb/artifacts/
rm -rf ~/checkpoints
