# Diffusion Bridge Networks to translate PET from MRI
Final Project for EE8520

<p align="center">
  <img src="img/archi_sim2p.svg" width="100%"/>
</p>

## Installation

1. Create environment: `conda env create -n sim2p --file requirements.yaml`
2. Activate environment: `conda activate sim2p`

## Preprocessing

The preprocessing pipeline converts raw DICOM MRI and PET scans into standardized 80×80×80 NIfTI volumes.

### Running Preprocessing

```bash
# Navigate to preprocessing directory
cd Prep_Final

# Edit config.yaml with your data paths
# data:
#   mcsa_root: "/path/to/your/dicom/data"
#   output_dir: "/path/to/output"

# Run the preprocessing pipeline
python run_pipeline.py --config config.yaml

# For a single case test
bash test_single.sh /path/to/subject_folder
```

See `Prep_Final/PREPROCESSING_PIPELINE.txt` for detailed documentation of each step.


### Training

To train the model:

```bash
# Activate environment
conda activate sim2p

# Basic training (single GPU)
CUDA_VISIBLE_DEVICES=0 python sim2p_train.py \
  --exp my_experiment \
  --data_dir /path/to/h5_data/ \
  --batch_size 6 \
  --microbatch 3 \
  --dit_type "DiT-XL/4" \
  --lr 1e-4 \
  --save_interval 10000 \
  --log_interval 50 \
  --image_size 80 \
  --target_modality pet \
  --ema_rate 0.9999 \
  --pred_mode vp \
  --sigma_max 1 \
  --sigma_min 0.002 \
  --tab_dim 10

# Or use the provided bash script
bash train_sim2p.sh

# To resume from checkpoint
bash train_sim2p.sh $CKPT
```

### Multi-GPU Training

For distributed training with multiple GPUs:

```bash
# Training on 2 GPUs (e.g., GPU 1 and 2)
CUDA_VISIBLE_DEVICES=1,2 mpiexec -n 2 python sim2p_train.py \
  --exp my_experiment_2gpu \
  --data_dir /path/to/h5_data/ \
  --batch_size 24 \
  --microbatch 12 \
  --dit_type "DiT-XL/4" \
  --lr 1e-4 \
  --image_size 80 \
  --target_modality pet \
  --pred_mode vp
```

## Acknowlegements

The codebase is inspired by [[Yiiitong/SiM2P]([https://github.com/alexzhou907/DDBM](https://github.com/Yiiitong/SiM2P))


## References

This work builds upon the following foundational research:

1. **Karras, T., Aittala, M., Aila, T., & Laine, S.** (2022). Elucidating the Design Space of Diffusion-Based Generative Models. *Advances in Neural Information Processing Systems (NeurIPS)*.

2. **Peebles, W., & Xie, S.** (2023). Scalable Diffusion Models with Transformers. *IEEE/CVF International Conference on Computer Vision (ICCV)*.

3. **Zhou, L., Lou, A., Khanna, S., & Ermon, S.** (2024). Denoising Diffusion Bridge Models. *International Conference on Learning Representations (ICLR)*.


## Citation

If you find this method and/or code useful, please consider giving a star 🌟 and citing the paper:
```
@article{li2025diffusion,
  title={Diffusion Bridge Networks Simulate Clinical-grade PET from MRI for Dementia Diagnostics},
  author={Li, Yitong and Buchert, Ralph and Schmitz-Koep, Benita and Grimmer, Timo and Ommer, Bj{\"o}rn and Hedderich, Dennis M and Yakushev, Igor and Wachinger, Christian},
  journal={arXiv preprint arXiv:2510.15556},
  year={2025}
}
```

