import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    Returns:
        str: RLE-encoded string, or a single space " " if mask is all zeros.
    """
    if np.sum(mask) == 0:
        return " "  # As it seems that kaggle reject nulls. We'll handle cloud-free images with empty spaces.
    
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format

def rle_decode(mask_rle: str, shape=(256, 256)) -> np.ndarray:
    """Decodes an RLE-encoded string into a binary mask with validation checks."""
    
    if not isinstance(mask_rle, str) or not mask_rle.strip() or mask_rle.lower() == 'nan':
        # Return all-zero mask if RLE is empty, invalid, or NaN
        return np.zeros(shape, dtype=np.uint8)
    
    try:
        s = list(map(int, mask_rle.split()))
    except:
        raise Exception("RLE segmentation must be a string and containing only integers")
    
    if len(s) % 2 != 0:
        raise Exception("RLE segmentation must have even-length (start, length) pairs")
    
    if any(x < 0 for x in s):
        raise Exception("RLE segmentation must not contain negative values")
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    starts, lengths = s[0::2], s[1::2]
    
    for start, length in zip(starts, lengths):
        if start >= mask.size or start + length > mask.size:
            raise Exception("RLE indices exceed image size")
        mask[start:start + length] = 1
    
    return mask.reshape(shape, order='F')  # Convert to column-major order

def generate_random_mask(shape, probability=0.5):
    """
    Generates a random binary mask.

    Args:
        shape (tuple): (height, width) of the mask.
        probability (float): Probability of a pixel being 1 (default is 0.5).

    Returns:
        np.ndarray: Random binary mask.
    """
    return (np.random.rand(*shape) < probability).astype(np.uint8)

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    return (2.0 * intersection + 1e-7) / (np.sum(mask1) + np.sum(mask2) + 1e-7)


def evaluate_and_plot(pred_mask: np.ndarray,
                      decoded_mask: np.ndarray,
                      figsize: tuple = (10, 5)
                     ) -> None:
    """
    Visualize the original predicted mask vs. the RLE-decoded mask and print Dice.
    """
    dice = dice_coefficient(pred_mask, decoded_mask)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(pred_mask, cmap='gray')
    axes[0].set_title('Predicted Mask')
    axes[0].axis('off')

    axes[1].imshow(decoded_mask, cmap='gray')
    axes[1].set_title(f'Decoded Mask\nDice: {dice:.4f}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def predict_mask(model: torch.nn.Module,
                 tiff_path: str,
                 device: torch.device = torch.device('cpu'),
                 threshold: float = 0.5
                ) -> np.ndarray:
    """
    Load a TIFF file, run the model, and return a binary mask.
    """
    model.eval()
    with torch.no_grad():
        with rasterio.open(tiff_path) as src:
            bands = torch.from_numpy(src.read()).float()  # (C, H, W)
        bands = bands / bands.max()
        image = bands.unsqueeze(0).to(device)  # (1, C, H, W)

        logits = model(image)
        probs = torch.sigmoid(logits)  # (1, 1, H, W)

        mask = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    return mask