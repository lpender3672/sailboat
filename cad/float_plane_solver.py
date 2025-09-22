import os
from PyQt6.QtCore import QTimer
from centre_of_bouyancy import (
    stl_center_of_buoyancy_plane, 
    load_mass_properties, 
    stl_center_of_buoyancy_plane_fast, 
    stl_center_of_buoyancy_plane_advanced,
    submerged_volume_trimesh,
    numpy_stl_to_trimesh
)
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
    QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import pyqtgraph as pg
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
    finished = pyqtSignal(object, object, object, float)
    new_cob = pyqtSignal(object, object, object, float) # angle, com, cob, time (s)

    def __init__(self, stl_mesh, mass, com, inertia, angle0=None, position0=None, parent=None):
        super().__init__(parent)
        self.stl_mesh = stl_mesh
        self.mass = mass
        self.com = com
        self.inertia = inertia
        self.dt = 0.001
        # capture initial state from UI to ensure exact match
        self.angle0 = np.array(angle0) if angle0 is not None else np.array([-np.pi/2, np.deg2rad(3), 0])
        self.position0 = np.array(position0) if position0 is not None else (np.copy(self.com) + 1e-3 * np.ones(3))

    def run(self):
        # copy mesh
        temp_mesh = mesh.Mesh(np.copy(self.stl_mesh.data))
        limit = 1e-5
        epsilon = np.inf
        iteration = 0
        max_iterations = 1000

        angle = self.angle0.copy()
        position = self.position0.copy()

        velocity = np.zeros(3)
        angular_velocity = np.zeros(3)
        dt = self.dt
        sim_time = 0.0

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

        # Emit initial state before stepping so UI matches initial draw
        temp_mesh.vectors = get_rotated_mesh(angle, position)
        tmsh = numpy_stl_to_trimesh(temp_mesh)
        init_cob, total_volume, _ = submerged_volume_trimesh(tmsh, [0,0,0], [0,0,1])
        latest_cob = init_cob
        self.new_cob.emit(angle.copy(), position.copy(), latest_cob.copy(), sim_time)

        def dynamics(state):
            nonlocal latest_cob
            position, velocity, angle, angular_velocity = state

            temp_mesh.vectors = get_rotated_mesh(angle, position)
            #cob, total_volume, mistreated_volume = stl_center_of_buoyancy_plane_advanced(temp_mesh, [0,0,0], [0,0,1])
            #cob, total_volume, mistreated_volume = stl_center_of_buoyancy_plane_fast(temp_mesh, [0,0,0], [0,0,1])

            tmsh = numpy_stl_to_trimesh(temp_mesh)
            cob, total_volume, mistreated_volume = submerged_volume_trimesh(tmsh, [0,0,0], [0,0,1])

            # Save latest cob for arrow
            latest_cob = cob

            density_kg_mm3 = 1025 * 1e-9
            gravity_mm_s2 = 9.81 * 1000 * np.array([0, 0, 1])

            buoyant_force = density_kg_mm3 * total_volume * gravity_mm_s2 * np.array([0, 0, 1])
            net_force = buoyant_force - self.mass * gravity_mm_s2
            net_moment = np.cross(cob - self.com, buoyant_force)

            acceleration = net_force / (self.mass if self.mass else 1.0)
            angular_acceleration = np.linalg.inv(self.inertia) @ net_moment

            return (
                velocity,
                acceleration,
                angular_velocity,
                angular_acceleration
            )

        damping_linear = 0.99
        damping_angular = 0.99

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

            # no yaw
            angular_velocity[2] = 0

            # Emit update for UI/plot
            self.new_cob.emit(angle.copy(), position.copy(), latest_cob if latest_cob is not None else np.zeros(3), sim_time)

            epsilon = np.linalg.norm(velocity) + np.linalg.norm(angular_velocity)
            iteration += 1
            sim_time += dt

        print("Solved in", iteration, "iterations")
        self.finished.emit(angle, position, latest_cob if latest_cob is not None else np.zeros(3), sim_time)

class STLViewerWidget(QWidget):

    def start_solve(self):

        # start thread
        self.thread = float_plane_solver(
            self.stl_mesh,
            self.mass,
            self.com,
            self.inertia,
            angle0=self.angle,
            position0=self.position,
            parent=self,
        )

        self.thread.finished.connect(self.on_solve_finished)
        self.thread.new_cob.connect(self.on_new_cob)

        # reset angles plot data
        self.pitch_times = []
        self.pitch_values = []
        self.roll_norm_values = []
        self.yaw_norm_values = []
        if self.pitchCurve is not None:
            self.pitchCurve.setData(self.pitch_times, self.pitch_values)
        if hasattr(self, 'rollCurve') and self.rollCurve is not None:
            self.rollCurve.setData(self.pitch_times, self.roll_norm_values)
        if hasattr(self, 'yawCurve') and self.yawCurve is not None:
            self.yawCurve.setData(self.pitch_times, self.yaw_norm_values)

        self.thread.start()

    def on_solve_finished(self, angle, com, cob, t):
        self.angle = angle
        self.position = com

        self.update_mesh_plot()
        #self.update_mesh_plot(angle=angle, position=com, arrow_start=com, arrow_end=cob)

    def on_new_cob(self, angle, com, cob, t):
        # update angles plot: time vs pitch (deg), roll/(pi/2), yaw/(pi/2)

        self.angle = angle
        self.position = com

        t = float(t)
        self.pitch_times.append(t)
        self.pitch_values.append(float(np.rad2deg(angle[1])))
        norm = (np.pi / 2.0)
        self.roll_norm_values.append(float(angle[0] % norm))
        self.yaw_norm_values.append(float(angle[2] % norm))
        if self.pitchCurve is not None:
            self.pitchCurve.setData(self.pitch_times, self.pitch_values)
        if hasattr(self, 'rollCurve') and self.rollCurve is not None:
            self.rollCurve.setData(self.pitch_times, self.roll_norm_values)
        if hasattr(self, 'yawCurve') and self.yawCurve is not None:
            self.yawCurve.setData(self.pitch_times, self.yaw_norm_values)

        # periodically redraw the 3D view with arrow to reduce cost
        if (len(self.pitch_times) % 5) == 0:
            self.update_mesh_plot(angle=angle, position=com, arrow_start=com, arrow_end=cob)

    def load_mass_properties(self, massprop_path):
        self.mass_props = load_mass_properties(massprop_path)
        self.com = np.array(self.mass_props.get('center_of_mass_mm', [0,0,0]))
        # subtract centroid offset from COM to match zeroed mesh coordinates
        vertices = self.stl_mesh.vectors.reshape(-1, 3)
        centroid = np.mean(vertices, axis=0)
        self.com = self.com - centroid

        # position should start at COM (in zeroed-mesh frame) with a tiny epsilon
        self.position = np.copy(self.com) + 1e-3 * np.ones(3)

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
        self.angle[0] = np.deg2rad(-120)
        self.angle[1] = np.deg2rad(3)  # start with 3 deg pitch

        self.position = np.zeros(3)  # [x, y, z] in mm

        self.display_solved = False
        self.mass_props = None
        self.com = np.zeros(3)
        self.inertia = np.eye(3)

        layout = QVBoxLayout(self)
        self.view = EventGLViewWidget()  # 3D view widget
        self.popup = popup
        self.dialog = None
        # angles plot setup
        self.pitch_times = []
        self.pitch_values = []  # degrees
        self.roll_norm_values = []  # normalized by (pi/2)
        self.yaw_norm_values = []   # normalized by (pi/2)
        self.pitchPlot = pg.PlotWidget()
        self.pitchPlot.setBackground('w')
        self.pitchPlot.showGrid(x=True, y=True, alpha=0.3)
        self.pitchPlot.setLabel('bottom', 'Time', units='s')
        self.pitchPlot.setLabel('left', 'Angle', units='varies')
        self.pitchPlot.setTitle('Angles vs Time')
        self.pitchPlot.addLegend()
        self.pitchCurve = self.pitchPlot.plot(self.pitch_times, self.pitch_values, pen=pg.mkPen('r', width=2), name='Pitch (deg)')
        self.rollCurve = self.pitchPlot.plot(self.pitch_times, self.roll_norm_values, pen=pg.mkPen('g', width=2), name='Roll/(pi/2)')
        self.yawCurve = self.pitchPlot.plot(self.pitch_times, self.yaw_norm_values, pen=pg.mkPen('b', width=2), name='Yaw/(pi/2)')

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
        # put the pitch plot above the 3D view using a splitter so both stay visible
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        self.pitchPlot.setMinimumHeight(120)
        splitter.addWidget(self.pitchPlot)
        splitter.addWidget(self.view)
        # initial sizes: smaller plot, larger 3D view
        splitter.setSizes([200, 800])
        layout.addWidget(splitter)
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
        self.dialog = None
        if not self.popup and self.dialog:
            self.dialog.widget.set_mesh(blade_mesh)


    def load_stl_file(self, stl_file):
        self.stl_file = stl_file
        self.stl_mesh = mesh.Mesh.from_file(stl_file)


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
        vertices_rotated = vertices_zeroed @ R.T + position

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
        water_verts = water_verts
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

    window.update_mesh_plot()

    sys.exit(app.exec())

