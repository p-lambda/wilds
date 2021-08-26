# Masked LM Pre-training

## Dependencies
- datasets==1.11.0
- transformers==4.9.1

## Usage
1. Format the unlabeled text data in the hugging-face format
```
python3 mlm_pretrain/src/get_data.py
```

2. Run the command in `mlm_pretrain/src/run_pretrain.sh` to start masked LM pre-training

3. Use the pre-trained model in WILDS fine-tuning, e.g.
```
python3 examples/run_expt.py --dataset civilcomments --algorithm ERM --root_dir data \
  --model distilbert-base-uncased \
  --model_kwargs state_dict=mlm_pretrain/data/_run__distilbert-base-uncased__civilcomments__b32a256_lr1e-4/checkpoint-1500/pytorch_model.bin
```
