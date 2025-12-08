# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
from pathlib import Path
import numpy as np
import trimesh
from moma_llm.utils.habitat_constants import PROJECT_DIR


def get_config(config_file: str) -> Path:
    if (PROJECT_DIR / "configs" / config_file).exists():
        config_file = str(PROJECT_DIR / "configs" / config_file)
    else:
        # Fallback: return the config file path as-is if not found in project configs
        config_file = str(Path(config_file))
    return Path(config_file)


def get_random_action(env):
    action = np.zeros(env.action_space.shape)
    action[[0, 1]] = np.random.uniform(-1, 1, size=2)
    return action


def quat_pos_to_mat(pos, quat):
    """Convert quaternion and position to 4x4 transformation matrix."""
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_quat(quat)
    mat = np.eye(4)
    mat[:3, :3] = rotation.as_matrix()
    mat[:3, 3] = pos
    return mat


def get_obj_bounding_box(obj):
    half_extent = obj.bounding_box / 2.0
    corners = np.stack([- half_extent, + half_extent])

    bbox_transform = quat_pos_to_mat(obj.get_position(), obj.get_orientation())
    world_frame_vertex_positions = trimesh.transformations.transform_points(corners, bbox_transform)
    return world_frame_vertex_positions