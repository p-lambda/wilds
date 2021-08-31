## Linear Probe Fine-Tuning

Follow these steps:

1. Run `python3 examples/algorithms/swav/finetuning/extract_features.py -d <WILDS dataset> 
   --root_dir <dataset directory> --run_dir <SwAV run directory> --batch_size <batch size> 
   --log_dir <output directory>`.
   
2. Run `python3 examples/algorithms/swav/finetuning/log_reg_sk.py --source_feat_path <source feature path>
   --target_feat_path <target feature path> --log_dir <output directory>`.
   