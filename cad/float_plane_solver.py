import os
from PyQt6.QtCore import QTimer
from centre_of_bouyancy import stl_center_of_buoyancy_plane, load_mass_properties
# stolen from propeller mesh viewer

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QCheckBox,
    QFileDialog,
    QPushButton,
    QMessageBox,
    QDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import pyqtgraph.opengl as gl
import numpy as np
from stl import mesh
from matplotlib import cm


class STLViewerDialog(QDialog):
    # when widget is double clicked, open dialog
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.widget = STLViewerWidget(self, popup=True)
        layout.addWidget(self.widget)

        self.setMinimumSize(800, 600)
        self.setWindowFlags(Qt.WindowType.Window)

        self.widget.view.escapePressed.connect(self.close)


class EventGLViewWidget(gl.GLViewWidget):
    doubleClicked = pyqtSignal()
    escapePressed = pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escapePressed.emit()
        super().keyPressEvent(event)


def create_arrow(start, end, color=(1, 0, 0, 1), rfac=0.01):
    # Create a line (shaft)
    line_points = np.array([start, end])
    line = gl.GLLinePlotItem(pos=line_points, color=color, width=2, antialias=True)

    # Create an arrowhead (cone)
    arrow_direction = np.array(end, dtype=float) - np.array(start, dtype=float)
    arrow_length = np.linalg.norm(arrow_direction)

    cylradius = rfac * arrow_length
    coneradius = 2 * cylradius

    z_axis = np.array([0, 0, 1])  # Default cone orientation along z-axis
    rotation_vector = np.cross(z_axis, arrow_direction)
    todot = np.dot(z_axis, arrow_direction) / (np.linalg.norm(z_axis) * np.linalg.norm(arrow_direction) + 1e-8)
    rotation_angle = np.arccos(todot) * 180 / np.pi

    if np.linalg.norm(rotation_vector) > 0:
        rotation_vector /= np.linalg.norm(rotation_vector)
    else:
        rotation_vector = z_axis

    cylinder_meshdata = gl.MeshData.cylinder(
        rows=10, cols=20, radius=[cylradius, cylradius], length=arrow_length
    )
    cylinder_mesh = gl.GLMeshItem(
        meshdata=cylinder_meshdata, smooth=True, color=color, shader="shaded"
    )

    # Create a cone mesh for the arrowhead
    cone_meshdata = gl.MeshData.cylinder(
        rows=10, cols=20, radius=[coneradius, 0], length=coneradius * 2
    )
    cone_mesh = gl.GLMeshItem(
        meshdata=cone_meshdata, smooth=True, color=color, shader="shaded"
    )

    # Apply the rotation matrix to the cylinder and cone
    cylinder_mesh.rotate(rotation_angle, *rotation_vector[0:3])
    cone_mesh.rotate(rotation_angle, *rotation_vector[0:3])

    cylinder_mesh.translate(*start)
    cone_mesh.translate(*end)

    return cylinder_mesh, cone_mesh

class float_plane_solver(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    new_cob = pyqtSignal(np.ndarray, np.ndarray, np.ndarray) # angle, com, cob

    def __init__(self, stl_mesh, mass, com, inertia, parent=None):
        super().__init__(parent)
        self.stl_mesh = stl_mesh
        self.mass = mass
        self.com = com
        self.inertia = inertia

    def run(self):
        # copy mesh
        temp_mesh = mesh.Mesh(np.copy(self.stl_mesh.data))
        limit = 1e-5
        epsilon = np.inf
        iteration = 0
        max_iterations = 100

        angle = np.array([-np.pi/2, np.deg2rad(3), 0])
        position = np.copy(self.com)
        position[2] += 150

        velocity = np.zeros(3)
        angular_velocity = np.zeros(3)
        prev_position = np.copy(position)
        prev_angle = np.copy(angle)

        dt = 0.1

        vertices = self.stl_mesh.vectors.reshape(-1, 3)
        centroid = np.mean(vertices, axis=0)
        vertices_zeroed = vertices - centroid

        # Precompute zeroed mesh for fast buoyancy calculations
        def get_rotated_mesh(angle, position):
            roll, pitch, yaw = angle
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
            Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            R = Rz @ Ry @ Rx
            vertices_rotated = vertices_zeroed @ R.T + position
            return vertices_rotated.reshape(-1, 3, 3)

        # Store latest cob for arrow
        latest_cob = None

        def dynamics(state):
            position, velocity, angle, angular_velocity = state

            # Use precomputed zeroed mesh and fast rotation
            temp_mesh.vectors = get_rotated_mesh(angle, position)
            cob, total_volume, mistreated_volume = stl_center_of_buoyancy_plane(temp_mesh, [0,0,0], [0,0,1])

            # Save latest cob for arrow
            nonlocal latest_cob
            latest_cob = cob

            density_kg_mm3 = 1025 * 1e-9
            gravity_mm_s2 = 9.81 * 1000 * np.array([0, 0, 1])

            buoyant_force = density_kg_mm3 * total_volume * gravity_mm_s2 * np.array([0, 0, 1])
            net_force = buoyant_force - self.mass * gravity_mm_s2
            net_moment = np.cross(cob - self.com, buoyant_force)

            acceleration = net_force / self.mass
            angular_acceleration = np.linalg.inv(self.inertia) @ net_moment

            return (
                velocity,
                acceleration,
                angular_velocity,
                angular_acceleration
            )

        damping_linear = 0.98
        damping_angular = 0.98

        while epsilon > limit and iteration < max_iterations:
            state = (position, velocity, angle, angular_velocity)

            # RK4 steps
            k1 = dynamics(state)
            k2 = dynamics((
                position + 0.5 * dt * k1[0],
                velocity + 0.5 * dt * k1[1],
                angle + 0.5 * dt * k1[2],
                angular_velocity + 0.5 * dt * k1[3]
            ))
            k3 = dynamics((
                position + 0.5 * dt * k2[0],
                velocity + 0.5 * dt * k2[1],
                angle + 0.5 * dt * k2[2],
                angular_velocity + 0.5 * dt * k2[3]
            ))
            k4 = dynamics((
                position + dt * k3[0],
                velocity + dt * k3[1],
                angle + dt * k3[2],
                angular_velocity + dt * k3[3]
            ))

            position += (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            velocity = damping_linear * (velocity + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]))
            angle += (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
            angular_velocity = damping_angular * (angular_velocity + (dt / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]))

            # had enough of this silliness just fix roll -90 and yaw 0
            angle[0] = -np.pi/2
            angle[2] = 0

            if iteration % 20 == 0:
                self.new_cob.emit(angle, position, latest_cob)

            epsilon = np.linalg.norm(velocity) + np.linalg.norm(angular_velocity)
            iteration += 1

            print("Iteration", iteration, "position", position, "angle (deg)", np.rad2deg(angle))


        print("Solved in", iteration, "iterations")
        self.finished.emit(angle, position, latest_cob)

class STLViewerWidget(QWidget):

    def start_solve(self):

        # start thread
        self.thread = float_plane_solver(self.stl_mesh, self.mass, self.com, self.inertia, parent=self)

        self.thread.finished.connect(self.on_solve_finished)
        self.thread.new_cob.connect(self.on_new_cob)

        self.thread.start()

    def on_solve_finished(self, angle, com, cob):
        self.angle = angle
        self.position = com

        self.update_mesh_plot()
        #self.update_mesh_plot(angle=angle, position=com, arrow_start=com, arrow_end=cob)

    def on_new_cob(self, angle, com, cob):
        # draw arrow from com to cob
        self.update_mesh_plot(angle=angle, position=com, arrow_start=com, arrow_end=cob)


    def load_mass_properties(self, massprop_path):
        self.mass_props = load_mass_properties(massprop_path)
        self.com = np.array(self.mass_props.get('center_of_mass_mm', [0,0,0]))

        # subtract centroid offset
        vertices = self.stl_mesh.vectors.reshape(-1, 3)
        centroid = np.mean(vertices, axis=0)

        self.com += -centroid

        self.mass = self.mass_props.get('mass_g', 0) * 1e-3
        # Use inertia at COM if available
        I = self.mass_props.get('inertia_tensor_cm')
        if I:
            self.inertia = np.array([
                [I['Lxx'], I['Lxy'], I['Lxz']],
                [I['Lxy'], I['Lyy'], I['Lyz']],
                [I['Lxz'], I['Lyz'], I['Lzz']],
            ]) * 1e-3 # convert to kg mm^2


    def __init__(self, parent=None, popup=False):
        super().__init__(parent)

        self.angle = np.zeros(3)  # [roll, pitch, yaw] in radians
        # set roll actually to 90 deg for boat upright
        self.angle[0] = np.deg2rad(-90)
        self.angle[1] = np.deg2rad(3) # start with 3 deg pitch

        self.position = np.zeros(3)  # [x, y, z] in mm

        self.display_solved = False
        self.mass_props = None
        self.com = np.zeros(3)
        self.inertia = np.eye(3)

        layout = QVBoxLayout(self)
        self.view = EventGLViewWidget()
        self.popup = popup
        

        viewSettingWidget = QWidget()
        viewSettingLayout = QGridLayout(viewSettingWidget)

        self.smoothToggle = QCheckBox("Smooth")
        self.smoothToggle.setChecked(False)
        self.smoothToggle.stateChanged.connect(self.update_mesh_plot)

        self.edgeToggle = QCheckBox("Draw Edges")
        self.edgeToggle.setChecked(False)
        self.edgeToggle.stateChanged.connect(self.update_mesh_plot)

        self.arrowToggle = QCheckBox("Show Arrow")
        self.arrowToggle.setChecked(False)
        self.arrowToggle.stateChanged.connect(self.update_mesh_plot)


        self.save_button = QPushButton("Save to .stl")
        self.save_button.clicked.connect(self.save_stl_file)

        self.solved_button = QPushButton("Solve Float Plane")
        self.solved_button.clicked.connect(self.toggle_solved)

        self.view.doubleClicked.connect(self.display_fullscreen_dialog)

        # viewSettingLayout.addWidget(self.smoothToggle, 0, 0)
        viewSettingLayout.addWidget(self.arrowToggle, 0, 0)
        viewSettingLayout.addWidget(self.edgeToggle, 0, 1)

        if not popup:
            viewSettingLayout.addWidget(self.save_button, 0, 2)
            viewSettingLayout.addWidget(self.solved_button, 0, 3)

        viewSettingWidget.setMaximumHeight(50)
        layout.addWidget(viewSettingWidget)
        layout.addWidget(self.view)

        self.view.setCameraPosition(distance=10)

        self.thread = None



    def toggle_solved(self):
        if not self.display_solved:
            self.start_solve()
            self.solved_button.setText("Solved Float Plane")
        else:

            self.solved_button.setText("Base Model")

    def display_fullscreen_dialog(self):
        if self.popup:
            return
        if self.dialog:
            self.dialog.close()
            self.dialog = None
            return

        self.dialog = STLViewerDialog(self)
        self.dialog.widget.set_mesh(self.stl_mesh)
        self.dialog.show()

    def save_stl_file(self, stl_file):
        path = QFileDialog.getSaveFileName(
            self, "Select boat file", filter="Stereolithography file (*.stl)"
        )[0]
        if not path:
            return

        try:
            meshtoscale = mesh.Mesh(np.copy(self.stl_mesh.data))
            meshtoscale.vectors *= (
                1000 * 100 / 2.54
            )  # I think Tony's software is in like 10 thousandths of an inch?
            meshtoscale.save(path)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Failed to save stl file")
            return

    def set_mesh(self, blade_mesh):
        self.stl_file = None
        self.stl_mesh = blade_mesh

        if not self.popup and self.dialog:
            self.dialog.widget.set_mesh(blade_mesh)


    def load_stl_file(self, stl_file):
        self.stl_file = stl_file
        self.stl_mesh = mesh.Mesh.from_file(stl_file)
        self.update_mesh_plot(angle=self.angle, position=self.position)


    def update_mesh_plot(self, *args, angle=None, position=None, arrow_start=None, arrow_end=None):
        # clear the view
        self.view.items = []

        stl_mesh = self.stl_mesh

        # Zero the mesh: center at origin (subtract centroid)
        vertices = stl_mesh.vectors.reshape(-1, 3)
        centroid = np.mean(vertices, axis=0)
        vertices_zeroed = vertices - centroid

        if angle is None:
            angle = self.angle
        if position is None:
            position = self.position

        roll, pitch, yaw = angle
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        vertices_rotated = vertices_zeroed @ R.T

        # Apply vertical bob
        vertices_rotated = vertices_rotated + position
        faces = np.arange(len(vertices_rotated)).reshape(-1, 3)

        # Update mesh vectors for normal calculation
        vectors_rotated = vertices_rotated.reshape(-1, 3, 3)
        v0 = vectors_rotated[:, 0, :]
        v1 = vectors_rotated[:, 1, :]
        v2 = vectors_rotated[:, 2, :]
        recalculated_normals = np.cross(v1 - v0, v2 - v0)
        recalculated_normals = recalculated_normals / (
            np.linalg.norm(recalculated_normals, axis=1)[:, None] + 1e-8
        )
        normals = np.repeat(recalculated_normals, 3, axis=0)

        # light direction to dot product with normals
        light_dir = np.array([1, 1, 1])
        light_dir = light_dir / np.linalg.norm(light_dir)

        min_intensity = np.inf
        max_intensity = -np.inf

        if self.arrowToggle.isChecked():
            # Arrow points up from origin to max Z
            line, arrow = create_arrow([0, 0, 0], [0, 0, np.max(vertices_rotated[:,2])])
            self.view.addItem(line)
            self.view.addItem(arrow)

        # Draw arrow from COM to COB if provided
        if arrow_start is not None and arrow_end is not None:
            line, arrow = create_arrow(arrow_start, arrow_end)
            self.view.addItem(line)
            self.view.addItem(arrow)

        intensity = np.dot(normals, light_dir)
        intensity_per_face = intensity[::3]

        min_intensity = min(min_intensity, np.nanmin(intensity_per_face))
        max_intensity = max(max_intensity, np.nanmax(intensity_per_face))

        intensity_normalized = (intensity_per_face - min_intensity) / (
            max_intensity - min_intensity + 1e-8
        )

        # Map to colormap
        colors = cm.coolwarm(intensity_normalized)[:, :4]

        mesh_item = gl.GLMeshItem(
            vertexes=vertices_rotated,
            faces=faces,
            faceColors=colors,
            smooth=self.smoothToggle.isChecked(),
            drawEdges=self.edgeToggle.isChecked(),
            edgeColor=(1, 1, 1, 1),
        )
        self.view.addItem(mesh_item)

        # Add reference plane below mesh, sized to bounding box
        min_xyz = np.min(vertices_rotated, axis=0)
        max_xyz = np.max(vertices_rotated, axis=0)
        plane_z = min_xyz[2] - 1e-3  # Slightly below mesh
        plane_verts = np.array([
            [min_xyz[0], min_xyz[1], plane_z],
            [max_xyz[0], min_xyz[1], plane_z],
            [max_xyz[0], max_xyz[1], plane_z],
            [min_xyz[0], max_xyz[1], plane_z],
        ])
        plane_faces = np.array([[0, 1, 2], [0, 2, 3]])
        plane_color = np.array([[0.8, 0.8, 0.8, 0.5]] * 2)  # semi-transparent gray
        plane_item = gl.GLMeshItem(
            vertexes=plane_verts,
            faces=plane_faces,
            faceColors=plane_color,
            smooth=False,
            drawEdges=False,
        )
        self.view.addItem(plane_item)

        # Draw water plane at Z=0, shaded blue
        water_z = 0
        water_verts = np.array([
            [min_xyz[0], min_xyz[1], water_z],
            [max_xyz[0], min_xyz[1], water_z],
            [max_xyz[0], max_xyz[1], water_z],
            [min_xyz[0], max_xyz[1], water_z],
        ])
        # Apply vertical bob to water plane for visual effect (optional, comment out if not desired)
        water_verts = water_verts + position
        water_faces = np.array([[0, 1, 2], [0, 2, 3]])
        water_color = np.array([[0.2, 0.4, 1.0, 0.5]] * 2)  # semi-transparent blue
        water_item = gl.GLMeshItem(
            vertexes=water_verts,
            faces=water_faces,
            faceColors=water_color,
            smooth=False,
            drawEdges=False,
        )
        self.view.addItem(water_item)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = STLViewerWidget()
    window.load_stl_file("cad/Boat.stl")
    # Load mass properties if available
    massprop_path = os.path.join("cad", "boat_mass_properties.txt")
    if os.path.exists(massprop_path):
        window.load_mass_properties(massprop_path)

    window.setWindowTitle("Boat Mesh Viewer")
    window.resize(1000, 800)
    window.show()

    sys.exit(app.exec())

