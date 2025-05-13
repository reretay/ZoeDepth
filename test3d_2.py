import numpy as np
from PIL import Image
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from zoedepth.utils.geometry import depth_to_points, create_triangles

def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map."""
    depth_dx, depth_dy = np.gradient(depth)
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    mask = depth_grad > 0.05
    return mask

def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def create_mesh(model, image, keep_edges=False):
    image.thumbnail((1024, 1024))
    depth = predict_depth(model, image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)

    verts = pts3d
    image_np = np.array(image)
    if keep_edges:
        triangles = create_triangles(image_np.shape[0], image_np.shape[1])
    else:
        mask = ~depth_edges_mask(depth)
        triangles = create_triangles(image_np.shape[0], image_np.shape[1], mask=mask)

    colors = image_np.reshape(-1, 3) / 255.0  # normalize for matplotlib
    return verts, triangles, colors

def plot_mesh(verts, triangles, colors):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 매 삼각형마다 꼭짓점 색상 평균
    face_colors = colors[triangles].mean(axis=1)

    mesh = Poly3DCollection(verts[triangles], facecolors=face_colors, linewidths=0.01, edgecolors='gray', alpha=1.0)
    ax.add_collection3d(mesh)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 시점 조정
    ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    plt.tight_layout()
    plt.show()

# ==== 사용 예시 ====

if __name__ == "__main__":
    import torch

    # 모델 로드
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
    zoe = model_zoe_n.to(DEVICE)

    # 이미지 로드
    image = Image.open("people.jpg").convert("RGB")

    # 메쉬 생성 및 시각화
    verts, triangles, colors = create_mesh(zoe, image, keep_edges=False)
    plot_mesh(verts, triangles, colors)
