# Used Libraries/Packages

- Numpy
- Matplotlib
- Pandas
- Pytorch
- Rasterio
- Sklearn

# Example command to run the inference script

## Directly

```
python run_inference.py --model-path model_state_dict.pth --test-dir test_images_folder --output-csv team_16_test.csv --visualize
```

# Model

- Unet
- Md5 checksum: ff2c3d877949dd2b237f8c8e4e9c6507 (for deadline)
  - `CertUtil -hashfile model_state_dict.pth MD5`
