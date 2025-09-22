
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
import sys
import re
import numpy as np
from stl import mesh

import trimesh
import triangle

import matplotlib.pyplot as plt


def numpy_stl_to_trimesh(numpy_stl_mesh):
    """
    Convert a numpy-stl mesh (stl.mesh.Mesh) to a trimesh.Trimesh object.
    """
    # Flatten all triangle vertices into (N*3, 3)
    all_triangles = numpy_stl_mesh.vectors.reshape(-1, 3)

    # Get unique vertices and remap faces
    vertices, inverse_indices = np.unique(all_triangles, axis=0, return_inverse=True)
    
    # Each triangle has 3 vertices, so reshape to (N, 3)
    faces = inverse_indices.reshape(-1, 3)

    # Create the trimesh object
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return tm

def submerged_volume_trimesh(mesh: trimesh.Trimesh, plane_origin, plane_normal):
    """
    Return the volume of the part of `mesh` that lies below the plane defined by
    `plane_origin` + normal `plane_normal`.
    We slice the mesh at the plane, keep the part on the negative side (or whichever
    side is submerged), cap it to make it watertight, then compute its volume.
    """
    # Ensure normal is unit length
    n = -np.array(plane_normal, dtype=float)
    n /= np.linalg.norm(n)
    o = np.array(plane_origin, dtype=float)

    # Slice the mesh, keep submerged side
    sliced = mesh.slice_plane(plane_origin=o, plane_normal=n, cap=True)

    if sliced is None:
        # No intersection or mesh entirely on one side
        # Determine if mesh is fully submerged or fully above
        # We can check distances of vertices
        dists = (mesh.vertices - o) @ n
        if np.all(dists < 0):
            # Fully submerged
            return mesh.volume
        else:
            # Fully above water, no submerged part
            return 0.0

    
    # Compute and return volume
    vol = sliced.volume
    cob = sliced.center_mass
    total_volume = sliced.volume

    return cob, total_volume, 0.0


def load_mass_properties(filepath):
    """
    Parse a SolidWorks mass properties text file and return a dictionary of values.
    """
    props = {}
    with open(filepath, 'r') as f:
        text = f.read()

    # Mass
    m = re.search(r"Mass\s*=\s*([\d.]+) grams", text)
    if m:
        props['mass_g'] = float(m.group(1))

    # Volume
    v = re.search(r"Volume\s*=\s*([\d.]+) cubic millimeters", text)
    if v:
        props['volume_mm3'] = float(v.group(1))

    # Surface area
    s = re.search(r"Surface area\s*=\s*([\d.]+)\s+square millimeters", text)
    if s:
        props['surface_area_mm2'] = float(s.group(1))

    # Center of mass
    cm = re.search(r"Center of mass:.*?X = ([\d.\-]+).*?Y = ([\d.\-]+).*?Z = ([\d.\-]+)", text, re.DOTALL)
    if cm:
        props['center_of_mass_mm'] = tuple(float(cm.group(i)) for i in range(1, 4))

    # Principal moments of inertia (Px, Py, Pz)
    px = re.search(r"Px = ([\d.\-]+)", text)
    py = re.search(r"Py = ([\d.\-]+)", text)
    pz = re.search(r"Pz = ([\d.\-]+)", text)
    if px and py and pz:
        props['principal_moments'] = (float(px.group(1)), float(py.group(1)), float(pz.group(1)))

    # Moments of inertia at center of mass (Lxx, Lxy, ...)
    l = re.search(r"Lxx = ([\d.\-]+).*?Lxy = ([\d.\-]+).*?Lxz = ([\d.\-]+).*?Lyy = ([\d.\-]+).*?Lyz = ([\d.\-]+).*?Lzz = ([\d.\-]+)", text, re.DOTALL)
    if l:
        props['inertia_tensor_cm'] = {
            'Lxx': float(l.group(1)),
            'Lxy': float(l.group(2)),
            'Lxz': float(l.group(3)),
            'Lyy': float(l.group(4)),
            'Lyz': float(l.group(5)),
            'Lzz': float(l.group(6)),
        }

    # Moments of inertia at output coordinate system (Ixx, ...)
    i = re.search(r"Ixx = ([\d.\-]+).*?Ixy = ([\d.\-]+).*?Ixz = ([\d.\-]+).*?Iyy = ([\d.\-]+).*?Iyz = ([\d.\-]+).*?Izz = ([\d.\-]+)", text, re.DOTALL)
    if i:
        props['inertia_tensor_out'] = {
            'Ixx': float(i.group(1)),
            'Ixy': float(i.group(2)),
            'Ixz': float(i.group(3)),
            'Iyy': float(i.group(4)),
            'Iyz': float(i.group(5)),
            'Izz': float(i.group(6)),
        }

    return props

# compile this to be faster
#from numba import njit
#@njit
def stl_center_of_buoyancy_plane(hull, plane_point, plane_normal):
    n = np.array(plane_normal) / np.linalg.norm(plane_normal)
    p0 = np.array(plane_point)

    total_volume = 0.0
    centroid_sum = np.zeros(3)
    mistreated_volume = 0.0

    # Flatten all vertices for fast lookup
    all_verts = hull.vectors.reshape(-1, 3)
    # Precompute signed distances for all vertices
    all_distances = np.dot(all_verts - p0, n)
    # Map each triangle to its vertex distances
    triangle_distances = all_distances.reshape(-1, 3)

    # Sort triangles by minimum signed distance (most submerged first)
    sorted_indices = np.argsort(np.min(triangle_distances, axis=1))

    for idx in sorted_indices:
        v0, v1, v2 = hull.vectors[idx]
        d = triangle_distances[idx]

        # base case: all vertices submerged
        if np.all(d < 0):
            mat = np.array([v0, v1, v2])
            volume = np.linalg.det(mat) / 6.0
            centroid = (v0 + v1 + v2) / 4.0
            total_volume += volume
            centroid_sum += volume * centroid
        # partially submerged triangles
        elif np.any(d < 0):
            # would have to find intersection points and probably end up with more triangles than before
            # so ignore this
            mat = np.array([v0, v1, v2])
            volume = np.linalg.det(mat) / 6.0
            mistreated_volume += volume
            pass
        # else: all vertices above plane, skip

    if total_volume <= 0:
        raise ValueError("No submerged volume found")

    # ensure mistreated volume is small
    if mistreated_volume / total_volume > 0.01:
        pass
        #raise ValueError("ruh roh, need to do geometry :( ", mistreated_volume / total_volume)

    cob = centroid_sum / total_volume
    return cob, total_volume, mistreated_volume


def stl_center_of_buoyancy_plane_fast(hull, plane_point, plane_normal):
    # shoutout to chatgpt for vectorising this

    n = np.array(plane_normal, dtype=float)
    n /= np.linalg.norm(n)
    p0 = np.array(plane_point, dtype=float)

    # Reshape to (N, 3, 3) for N triangles
    tris = hull.vectors
    N = tris.shape[0]

    # Compute signed distances for all vertices in all triangles → (N, 3)
    dists = np.dot(tris.reshape(-1, 3) - p0, n).reshape(N, 3)

    # Masks
    all_submerged = np.all(dists < 0, axis=1)
    any_submerged = np.any(dists < 0, axis=1)

    # Fully submerged triangles
    submerged_tris = tris[all_submerged]

    if submerged_tris.size == 0:
        raise ValueError("No submerged volume found")

    # Compute volumes (determinant per triangle, treat as tetrahedron with origin)
    volumes = np.linalg.det(submerged_tris) / 6.0   # (M,)
    centroids = np.sum(submerged_tris, axis=1) / 4.0  # (M, 3)

    total_volume = np.sum(volumes)
    centroid_sum = np.einsum('i,ij->j', volumes, centroids)

    # Partially submerged triangles — approximated as if fully submerged
    mistreated_tris = tris[np.logical_and(~all_submerged, any_submerged)]
    mistreated_volumes = np.linalg.det(mistreated_tris) / 6.0
    mistreated_volume = np.sum(mistreated_volumes)

    cob = centroid_sum / total_volume

    # Optional check
    if mistreated_volume / total_volume > 0.01:
        pass
        # raise ValueError("ruh roh, need to do geometry :( ", mistreated_volume / total_volume)

    return cob, total_volume, mistreated_volume


def stl_center_of_buoyancy_plane_advanced(hull, plane_point, plane_normal):
    # Normalize the plane normal
    n = np.array(plane_normal, dtype=float)
    n /= np.linalg.norm(n)
    p0 = np.array(plane_point, dtype=float)

    # Reshape to (N, 3, 3) for N triangles
    tris = hull.vectors
    N = tris.shape[0]

    # Compute signed distances for all vertices in all triangles → (N, 3)
    dists = np.dot(tris.reshape(-1, 3) - p0, n).reshape(N, 3)

    # Masks
    all_submerged = np.all(dists < 0, axis=1)  # Fully submerged
    any_submerged = np.any(dists < 0, axis=1)  # Partially submerged

    assert all_submerged.shape == any_submerged.shape == (N,)

    # Fully submerged triangles (treated normally)
    submerged_tris = tris[all_submerged]

    if submerged_tris.size == 0:
        raise ValueError("No submerged volume found")

    # Compute volumes for fully submerged triangles
    volumes = np.linalg.det(submerged_tris) / 6.0   # (M,)
    centroids = np.sum(submerged_tris, axis=1) / 4.0  # (M, 3)

    total_volume = np.sum(volumes)
    centroid_sum = np.einsum('i,ij->j', volumes, centroids)

    # Now deal with partially submerged triangles
    mistreated_mask = np.logical_and(~all_submerged, any_submerged)
    mistreated_tris = tris[mistreated_mask]  # Partially submerged

    # Find the distances for mistreated triangles
    mistreated_dists = dists[mistreated_mask]  # (M, 3)

    # Submerged and non-submerged vertices for mistreated triangles
    submerged_areas = np.maximum(0, -mistreated_dists)  # Positive when submerged

    # Total submerged area for each mistreated triangle (sum of submerged vertex areas)
    total_submerged_area = np.sum(submerged_areas, axis=1)  # (M,)

    # Interpolated volume for each mistreated triangle, scaling based on submerged area
    mistreated_volumes = np.linalg.det(mistreated_tris) / 6.0 * (total_submerged_area / 3.0)  # (M,)

    # Sum of mistreated volumes
    mistreated_volume = np.sum(mistreated_volumes)
    mistreated_centroids = np.sum(mistreated_tris, axis=1) / 4.0  # Shape: (M, 3)

    # Final center of buoyancy calculation (COB)
    total_volume += mistreated_volume
    centroid_sum += np.sum(mistreated_volumes[:, np.newaxis] * mistreated_centroids, axis=0)

    cob = centroid_sum / total_volume

    return cob, total_volume, mistreated_volume


if __name__ == "__main__":
    filename = "cad/Boat.stl"

    plane_point = [0, 0, 0]
    plane_normal = [0, 0, 1]

    hull = mesh.Mesh.from_file(filename)
    copy_hull = mesh.Mesh(np.copy(hull.data))
    trimesh_obj = trimesh.load_mesh(filename)
    # move to centroid
    
    copy_hull.vectors -= np.mean(copy_hull.vectors, axis=(0,1))
    
    #cob, total_volume, mistreated_volume = stl_center_of_buoyancy_plane(hull, plane_point, plane_normal)

    N = 100

    min_z, max_z = np.min(copy_hull.vectors[:,:,1]), np.max(copy_hull.vectors[:,:,1])
    plane_zs = np.linspace(min_z, max_z, N)
    
    volume_fast = np.zeros(N)
    volume_slow = np.zeros(N)
    volume_advanced = np.zeros(N)
    volume_trimesh = np.zeros(N)

    for i, z in enumerate(plane_zs):
        plane_point = [0, 0, z]

        # Fast (fully submerged only)
        try:
            _, volume_fast[i], _ = stl_center_of_buoyancy_plane_fast(copy_hull, plane_point, plane_normal)
        except ValueError:
            volume_fast[i] = 0

        try:
            _, volume_slow[i], _ = stl_center_of_buoyancy_plane(copy_hull, plane_point, plane_normal)
        except ValueError:
            volume_slow[i] = 0

        # Advanced (interpolated partials)
        try:
            _, volume_advanced[i], _ = stl_center_of_buoyancy_plane_advanced(copy_hull, plane_point, plane_normal)
        except ValueError:
            volume_advanced[i] = 0

        # Trimesh method
        try:
            volume_trimesh[i] = submerged_volume_trimesh(trimesh_obj, plane_point, plane_normal)
        except Exception as e:
            volume_trimesh[i] = 0
            print("trimesh error:", e)


    # Plot all curves
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(plane_zs, volume_fast, label="Fast (fully submerged only)", linestyle='--')
    ax.plot(plane_zs, volume_slow, label="Slow (mistreated partials)", linestyle='-.')
    ax.plot(plane_zs, volume_advanced, label="Advanced (interpolated)", linewidth=2)
    ax.plot(plane_zs, volume_trimesh, label="Trimesh slice_plane", linestyle=':')

    ax.set_title("Submerged Volume vs Waterline Height")
    ax.set_xlabel("Waterline Z")
    ax.set_ylabel("Submerged Volume")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


