# LassoDiff Inference

## Project Goals
- All-atom lasso peptide modeling under a backbone + isobond flow matching framework
- Pocket ligand-conditioned all-atom lasso peptide prediction
- De novo all-atom design of lasso peptides

## Current Progress
- Integrated and organized core inference flow matching modules
- Added loss computation and feature processing for backbone and isobond constraints
- Completed adaptation paths and attribution statements for Protenix and ml-simplefold

## Training (bash)
```bash
python lassodiff/train_toy.py \
  --n 128 \
  --L 40 \
  --K 3 \
  --epochs 3 \
  --batch_size 4 \
  --lr 1e-4
```

```bash
torchrun --nproc_per_node=2 lassodiff/train_toy.py \
  --structure_dir /path/to/structures \
  --epochs 3 \
  --batch_size 4 \
  --lr 1e-4 \
  --export_val_pdb \
  --export_val_dir val
```

## Next Steps
- Complete the feature pipeline for pocket ligand-conditioned inputs
- Evaluate stability and usability of all-atom prediction and de novo design workflows
- Build a minimal reproducible inference pipeline and evaluation metrics

## Acknowledgements
- Thanks to Protenix for the model architecture and engineering implementation
- Thanks to ml-simplefold for key modules and engineering references
