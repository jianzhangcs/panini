# Train with a single GPU
python tools/train.py configs/panini_mfr.py --work-dir my_project/panini_mfr 
python tools/train.py configs/panini_sr.py --work-dir my_project/panini_sr 

# Resume from a previous checkpoint file
python tools/train.py configs/panini_mfr.py --work-dir my_project/panini_mfr --resume-from my_project/panini_mfr/latest.pth
python tools/train.py configs/panini_sr.py --work-dir my_project/panini_sr --resume-from my_project/panini_sr/latest.pth

# Train with multiple GPUs
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/panini_mfr.py --work-dir my_project/panini_mfr
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/panini_sr.py --work-dir my_project/panini_sr