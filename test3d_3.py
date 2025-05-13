import numpy as np
from PIL import Image
import open3d as o3d

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

def create_open3d_mesh(model, image, keep_edges=False):
    #image.thumbnail((1024, 1024)) # 원본 해상도 처리를 위해 주석처리
    depth = predict_depth(model, image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)

    image_np = np.array(image)

    if keep_edges:
        triangles = create_triangles(image_np.shape[0], image_np.shape[1])
    else:
        mask = ~depth_edges_mask(depth)
        triangles = create_triangles(image_np.shape[0], image_np.shape[1], mask=mask)

    # Open3D 메쉬 생성
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # RGB 색상 정규화 후 지정
    colors = image_np.reshape(-1, 3) / 255.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()
    return mesh

def visualize_mesh(mesh):
    o3d.visualization.draw_geometries([mesh])

# ==== 사용 예시 ====

if __name__ == "__main__":
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_zoe_n = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True)
    zoe = model_zoe_n.to(DEVICE)

    image = Image.open("subway.jpg").convert("RGB")

    mesh = create_open3d_mesh(zoe, image, keep_edges=False)
    visualize_mesh(mesh)
