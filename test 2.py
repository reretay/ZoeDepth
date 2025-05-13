import torch
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import pil_to_batched_tensor
import matplotlib.pyplot as plt

# Step 1: Load ZoeDepth model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = get_config("zoedepth", "infer")
model = build_model(config).to(DEVICE)

# Step 2: Load input image
image_path = "people.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Step 3: Perform depth estimation
batched_tensor = pil_to_batched_tensor(image).to(DEVICE)
with torch.no_grad():  # Disable gradient computation for inference
    depth_tensor = model.infer(batched_tensor)

# Step 4: Convert depth tensor to numpy array and remove the batch dimension
depth_numpy = depth_tensor.cpu().detach().numpy()[0]  # Remove batch dimension (1, H, W -> H, W)

# Step 5: Visualize the depth map using matplotlib
plt.imshow(depth_numpy, cmap="viridis")  # Use depth_numpy directly (H, W format)
plt.colorbar(label="Depth (meters)")
plt.title("Estimated Depth Map")
plt.show()
