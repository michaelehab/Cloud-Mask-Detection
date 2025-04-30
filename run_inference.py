import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize

from unet import UNet
from utils import predict_mask, rle_encode, rle_decode, evaluate_and_plot

def main(model_path: str,
         test_dir: str,
         output_csv: str = 'submission.csv',
         threshold: float = 0.5,
         use_cuda: bool = False,
         visualize: bool = False
        ) -> pd.DataFrame:

    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    model = UNet(in_channels=4, n_class=1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    patterns = [os.path.join(test_dir, '*.tif'), os.path.join(test_dir, '*.tiff')]
    tiff_paths = sorted(sum([glob.glob(p) for p in patterns], []))
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found in {test_dir}")

    records = []
    for path in tiff_paths:
        img_id = os.path.splitext(os.path.basename(path))[0]
        mask = predict_mask(model, path, device=device, threshold=threshold)
        # Resize mask to 256x256 for submission
        mask = resize(mask,
                      (256, 256),
                      order=0,
                      preserve_range=True,
                      anti_aliasing=False)
        mask = mask.astype(np.uint8)

        rle = rle_encode(mask)
        records.append({'id': img_id, 'segmentation': rle})
        print(f"Processed {img_id}")

        if visualize:
            decoded = rle_decode(rle, shape=mask.shape)
            evaluate_and_plot(mask, decoded)
            print(f"Visualization for {img_id} complete. Mask shape: {decoded.shape}")

    df = pd.DataFrame(records, columns=['id', 'segmentation'])
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on TIFF test images using a trained UNet model.')
    parser.add_argument('--model-path', required=True, help='Path to the trained model .pth file')
    parser.add_argument('--test-dir', required=True, help='Directory containing test TIFF images')
    parser.add_argument('--output-csv', default='submission.csv', help='Output CSV filename')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for mask binarization')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA for inference if available')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions alongside RLE decoding')
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_csv=args.output_csv,
        threshold=args.threshold,
        use_cuda=args.use_cuda,
        visualize=args.visualize
    )
