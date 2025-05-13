import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ZoeDepth 관련 유틸
from zoedepth.utils.misc import (
    pil_to_batched_tensor,
    colorize,
    save_raw_16bit
)

# 1. 모델 로드
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
zoe = model_zoe_n.to(DEVICE)

# 2. 이미지 로드
image = Image.open("people.jpg").convert("RGB")

# 3. Depth 추론
depth = zoe.infer_pil(image)  # numpy (H, W)

# 4. depth 저장 (16비트 raw)
save_raw_16bit(depth, "output.png")

# 5. 컬러맵 저장
colored = colorize(depth)
Image.fromarray(colored).save("output_colored.png")

# 6. 포인트 클라우드 시각화
H, W = depth.shape
u = np.arange(W)
v = np.arange(H)
uu, vv = np.meshgrid(u, v)

# 카메라 파라미터 (가정값)
focal_length = 1.0
cx, cy = W / 2, H / 2

Z = depth
X = (uu - cx) * Z / focal_length
Y = (vv - cy) * Z / focal_length

# 시각화 (다운샘플링 포함)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
step = 4
ax.scatter(X[::step, ::step],
           -Y[::step, ::step],
           Z[::step, ::step],
           c=Z[::step, ::step],
           cmap='viridis',
           s=0.5)

ax.set_title("3D Point Cloud from Depth")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
