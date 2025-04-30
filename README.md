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

## Using docker container

```
docker build --tag sat-project-team-16 .
docker run -v "$(pwd)/test_images:/app/test_images" -v "$(pwd)/output:/app/output" -v "$(pwd)/model.pth:/app/model.pth" sat-project-team-16:latest
```

# Model

- Unet
- Md5 checksum: ff2c3d877949dd2b237f8c8e4e9c6507 (for deadline)
  - `CertUtil -hashfile model_state_dict.pth MD5`
