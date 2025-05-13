from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
import torch
import numpy as np
from PIL import Image
import open3d as o3d

def create_open3d_mesh(image_path, model_name="Intel/zoedepth-nyu-kitti"):
    # 1. 모델 및 프로세서 초기화
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ZoeDepthForDepthEstimation.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 2. 이미지 로드 및 전처리 (원본 해상도 유지)
    image = Image.open(image_path).convert("RGB")
    original_size = image.size[::-1]  # (H, W)

    # 3. 자동 크기 조정 및 추론
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. 후처리: 패딩 제거 및 원본 크기 복원
    post_processed = image_processor.post_process_depth_estimation(
        outputs, 
        source_sizes=[original_size]
    )
    depth = post_processed["predicted_depth"][0].cpu().numpy()

    # 5. 3D 메시 생성
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts3d)
    mesh.triangles = o3d.utility.Vector3iVector(create_triangles(*depth.shape))
    
    # 6. 텍스처 매핑
    image_np = np.array(image.resize((depth.shape[1], depth.shape[0])))
    mesh.vertex_colors = o3d.utility.Vector3dVector(image_np.reshape(-1,3)/255.0)
    
    return mesh
