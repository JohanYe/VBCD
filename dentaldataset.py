import json
import os
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from dataset import create_voxelwithnormal_grid, create_voxel_grid
import trimesh


class IOS_Datasetv2(Dataset):
    def __init__(
        self,
        root_dir,
        is_train=True,
        crop_size=(2.00, 2.00, 2.00),
        voxel_size=(0.15625, 0.15625, 0.15625),
    ):
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.voxel_size = voxel_size

        subset = "training" if is_train else "validation"
        self.data_list_file = os.path.join(self.root_dir, f"{subset}.json")
        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            data_subset = json.load(f)
        self.init_data_paths(data_subset)

    def init_data_paths(self, data_subset):
        self.data_paths = []
        for sample in data_subset:
            file_path = os.path.join(self.root_dir, "DataSamples", sample, "DataFiles/")
            crown_file = os.path.join(file_path, "outer_crown.ply")
            pna_crop_file = os.path.join(file_path, "toothMesh.ply")
            assert os.path.exists(crown_file), f"{crown_file} does not exist!"
            assert os.path.exists(pna_crop_file), f"{pna_crop_file} does not exist!"
            self.data_paths.append(
                {
                    "sample_path": file_path,
                    "crown_file": crown_file,
                    "pna_crop_file": pna_crop_file,
                }
            )

    def crop_mesh(self, mesh, center, crop_size):
        points = np.asarray(mesh)

        positions = points[:, :3]
        normals = points[:, 3:]

        half_crop_size = 10 * np.array(crop_size) / 2
        min_bound = center - half_crop_size
        max_bound = center + half_crop_size
        mask = np.all((positions >= min_bound) & (positions <= max_bound), axis=1)
        cropped_positions = positions[mask]
        cropped_normals = normals[mask]

        cropped_points = np.hstack((cropped_positions, cropped_normals))
        return cropped_points

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        sample = self.data_paths[idx]
        dirpath = sample["sample_path"]
        crown_file = sample["crown_file"]
        pna_crop_file = sample["pna_crop_file"]

        crown_mesh = trimesh.load(crown_file)
        curvatures = trimesh.curvature.discrete_mean_curvature_measure(
            crown_mesh, crown_mesh.vertices, radius=0.1
        )
        crown_vertices = crown_mesh.vertices
        crown_normals = crown_mesh.vertex_normals

        pna_crop_mesh = trimesh.load(pna_crop_file)
        pna_crop_vertices = pna_crop_mesh.vertices
        pna_crop_normals = pna_crop_mesh.vertex_normals

        crown_min_bound = crown_vertices.min(axis=0)
        crown_max_bound = crown_vertices.max(axis=0)
        crown_center = (crown_min_bound + crown_max_bound) / 2

        half_crop_size = np.array(self.crop_size) * 10 / 2
        min_bound_crop = crown_center - half_crop_size
        max_bound_crop = crown_center + half_crop_size
        point_cloud_full_inform = np.concatenate(
            [crown_vertices, crown_normals, curvatures.reshape(-1, 1)], axis=1
        )
        crown_voxel_grid = create_voxelwithnormal_grid(
            point_cloud_full_inform, min_bound_crop, max_bound_crop, self.voxel_size
        )
        crown_tensor = torch.from_numpy(crown_voxel_grid).float()
        pna_crop = self.crop_mesh(
            np.concatenate([pna_crop_vertices, pna_crop_normals], axis=1),
            crown_center,
            self.crop_size,
        )
        pna_crop_tensor = torch.tensor(pna_crop, dtype=torch.float32)
        pna_crop_tensor = self.normalize_point_cloud(
            pna_crop_tensor, cropsize=self.crop_size
        )
        point_cloud_crown_inform = torch.tensor(
            point_cloud_full_inform, dtype=torch.float32
        )

        pna_crop_voxel = create_voxel_grid(
            pna_crop[:, :3], min_bound_crop, max_bound_crop, self.voxel_size
        )
        pna_crop_tensor = (
            torch.from_numpy(pna_crop_voxel).float().unsqueeze(0)
        )  # [1, D, H, W]

        return (
            pna_crop_tensor,
            crown_tensor,
            point_cloud_crown_inform,
            torch.tensor(min_bound_crop, dtype=torch.float32),
            dirpath,
        )

    def collate_fn(self, batch):
        (
            pna_crop_tensor,
            crown_tensor,
            point_cloud_crown_tensor,
            min_bound_crop,
            dirpath,
        ) = zip(*batch)

        # This is the missing line - stack the tuple into [B, 1, D, H, W]
        batched_pna = torch.stack(pna_crop_tensor)

        point_cloud_crown_tensor = [pc for pc in point_cloud_crown_tensor]
        combined_point_cloud = torch.cat(point_cloud_crown_tensor, dim=0)
        batch_sizes = [pc.shape[0] for pc in point_cloud_crown_tensor]
        batch_indices = torch.cat(
            [
                torch.full((size,), i, dtype=torch.long)
                for i, size in enumerate(batch_sizes)
            ]
        )

        return (
            batched_pna,
            torch.stack(crown_tensor),
            combined_point_cloud,
            batch_indices,
            torch.stack(min_bound_crop),
            dirpath,
        )

    def normalize_point_cloud(self, point_cloud, cropsize):
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        if point_cloud.shape[1] != 6:
            raise ValueError("Point cloud should have shape (num_points, 6)")

        positions = point_cloud[:, :3]
        normals = point_cloud[:, 3:]

        point_cloud_center = (
            torch.min(positions, dim=0)[0] + torch.max(positions, dim=0)[0]
        ) / 2
        crop_center = 10 * torch.tensor(cropsize, dtype=torch.float32) / 2
        crop_scale = 10 * torch.tensor(cropsize, dtype=torch.float32)

        normalized_positions = (
            positions - point_cloud_center + crop_center
        ) / crop_scale
        normalized_positions = (normalized_positions - 0.5) * 2

        normalized_point_cloud = torch.cat((normalized_positions, normals), dim=1)

        return normalized_point_cloud
