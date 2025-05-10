"""
robotic_arm_ui.py

This file implements a 3-DOF robotic arm simulation and control system. It provides:
1. A Flask-based REST API to control the robotic arm via HTTP requests.
2. A PyQt-based graphical user interface (GUI) for visualizing and manually controlling the robotic arm.

Key Features:
- Inverse Kinematics: Calculates joint angles (theta1, theta2, theta3) to reach a target position (x, y).
- REST API: Allows external applications to control the robotic arm programmatically.
- GUI: Provides sliders for manual control and a 2D visualization of the robotic arm.


Modules Used:
- Flask: For creating the REST API.
- PyQt5: For building the GUI.
- NumPy: For mathematical computations.
- Matplotlib: For visualizing the robotic arm in 2D.

Author: [Ahmad Kheirandish]
Date: [15/04/2025]

"""
import sys
import threading
import numpy as np
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Shared angles (thread-safe)
joint_angles = [0, 0, 0]
lock = threading.Lock()


class Kinematics:
    @staticmethod
    def inverse_kinematics(x, y, l1, l2, l3):
        """
        Calculate the joint angles for a 3-DOF robotic arm using inverse kinematics.

        Args:
            x (float): Target x-coordinate of the end effector.
            y (float): Target y-coordinate of the end effector.
            l1 (float): Length of the first link.
            l2 (float): Length of the second link.
            l3 (float): Length of the third link.

        Returns:
            list[float] or None: A list of joint angles [theta1, theta2, theta3] in degrees if the target is reachable,
                                 otherwise None.
                - theta1: Angle of the first joint (base joint) relative to the horizontal axis.
                - theta2: Angle of the second joint relative to the first link.
                - theta3: Angle of the third joint relative to the second link.
        """
        distance = np.hypot(x, y)
        if distance > (l1 + l2 + l3):
            return None
        angle_to_target = np.arctan2(y, x)
        x_wrist = x - l3 * np.cos(angle_to_target)
        y_wrist = y - l3 * np.sin(angle_to_target)
        dx, dy = x_wrist, y_wrist
        D = (dx**2 + dy**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if abs(D) > 1:
            return None
        theta2 = np.arccos(D)
        theta1 = np.arctan2(dy, dx) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta12 = theta1 + theta2
        theta3 = angle_to_target - theta12
        return np.degrees([theta1, theta2, theta3])


class RoboticArmAPI:
    def __init__(self):
        """
        Initialize the Flask application and set up the API routes.
        """
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Serve Swagger UI at '/swagger' URL
        SWAGGER_URL = '/swagger'
        API_URL = '/static/swagger.json'  # Path to swagger.json
        swaggerui_blueprint = get_swaggerui_blueprint(
            SWAGGER_URL, 
            API_URL, 
            config={'app_name': "Robotic Arm API"}
        )
        self.app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
        
        self.setup_routes()

    def setup_routes(self):
        """
        Define the API routes for controlling the robotic arm.
        """
        @self.app.route('/robot/start', methods=['POST'])
        def start_robot():
            """
            Handle the '/robot/start' API endpoint to move the robotic arm.

            Request Body:
                - x (float): Target x-coordinate of the end effector.
                - y (float): Target y-coordinate of the end effector.
                - theta1, theta2, theta3 (int): Joint angles (optional if x and y are provided).

            Returns:
                JSON response with the calculated joint angles or an error message.
            """
            data = request.get_json()
            x, y = data.get('x'), data.get('y')
            if x is not None and y is not None:
                l1, l2, l3 = 100, 75, 50
                result = Kinematics.inverse_kinematics(x, y, l1, l2, l3)
                if result is None:
                    return jsonify({"error": "Target position is unreachable"}), 400
                theta1, theta2, theta3 = result
                with lock:
                    joint_angles[0] = int(theta1)
                    joint_angles[1] = int(theta2)
                    joint_angles[2] = int(theta3)
                return jsonify({
                    "message": "Arm moved to target position using inverse kinematics",
                    "input": {"x": x, "y": y},
                    "angles": joint_angles
                })
            theta1, theta2, theta3 = data.get('theta1'), data.get('theta2'), data.get('theta3')
            if None in (theta1, theta2, theta3):
                return jsonify({"error": "Must provide either both x and y, or all three joint angles"}), 400
            with lock:
                joint_angles[0] = int(theta1)
                joint_angles[1] = int(theta2)
                joint_angles[2] = int(theta3)
            return jsonify({
                "message": "Arm moved using joint angles",
                "input": {"theta1": theta1, "theta2": theta2, "theta3": theta3},
                "angles": joint_angles
            })

    def run(self):
        """
        Start the Flask application.
        """
        self.app.run(port=5000, debug=False, use_reloader=False)


class RoboticArmUI(QWidget):
    def __init__(self):
        """
        Initialize the PyQt UI for the robotic arm visualization and control.
        """
        super().__init__()
        self.setWindowTitle("3-DOF Robotic Arm - 2D Viewer")
        self.resize(600, 600)
        self.lengths = [100, 75, 50]
        self.angles = [0, 0, 0]
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.sync_with_api)
        self.timer.start(100)

    def move_to_target(self):
        """
        Move the robotic arm to a target position specified by the user.

        Reads the x and y coordinates from the input fields, calculates the joint angles using inverse kinematics,
        and updates the shared joint angles.
        """
        try:
            x, y = float(self.x_input.text()), float(self.y_input.text())
            l1, l2, l3 = self.lengths
            result = Kinematics.inverse_kinematics(x, y, l1, l2, l3)
            if result is None:
                print("Target unreachable.")
                return
            theta1, theta2, theta3 = result
            with lock:
                joint_angles[0] = int(theta1)
                joint_angles[1] = int(theta2)
                joint_angles[2] = int(theta3)
            print(f"Moved to (x={x}, y={y}) -> θ1={theta1:.1f}, θ2={theta2:.1f}, θ3={theta3:.1f}")
            self.draw_arm()
        except ValueError:
            print("Invalid input for x or y.")

    def init_ui(self):
        """
        Set up the UI components, including input fields, sliders, and the canvas for visualization.
        """
        layout = QVBoxLayout()
        target_layout = QHBoxLayout()
        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.x_input.setPlaceholderText("X")
        self.y_input.setPlaceholderText("Y")
        self.move_button = QPushButton("Move Arm(change start point)")
        self.move_button.clicked.connect(self.move_to_target)
        target_layout.addWidget(QLabel("Target X:"))
        target_layout.addWidget(self.x_input)
        target_layout.addWidget(QLabel("Y:"))
        target_layout.addWidget(self.y_input)
        target_layout.addWidget(self.move_button)
        layout.addLayout(target_layout)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.sliders = []
        self.angle_labels = []
        for i in range(3):
            angle_label = QLabel(f"Theta{i + 1}: 0.0°")
            angle_label.setAlignment(Qt.AlignCenter)
            self.angle_labels.append(angle_label)
            layout.addWidget(angle_label)
            slider_layout = QHBoxLayout()
            label = QLabel(f"Joint {i + 1}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setValue(0)
            slider.valueChanged.connect(lambda value, idx=i: self.manual_override(idx, value))
            self.sliders.append(slider)
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            layout.addLayout(slider_layout)
        self.setLayout(layout)
        self.draw_arm()

    def manual_override(self, joint_idx, angle):
        """
        Manually override the angle of a specific joint using the sliders.

        Args:
            joint_idx (int): Index of the joint to override (0, 1, or 2).
            angle (int): New angle value for the joint.
        """
        with lock:
            joint_angles[joint_idx] = angle

    def sync_with_api(self):
        """
        Periodically synchronize the UI with the shared joint angles and update the sliders and labels.
        """
        with lock:
            self.angles = joint_angles.copy()
        for i, angle in enumerate(self.angles):
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(angle)
            self.angle_labels[i].setText(f"Theta{i + 1}: {angle:.1f}°")
            self.sliders[i].blockSignals(False)
        self.draw_arm()

    def draw_arm(self):
        """
        Draw the robotic arm on the canvas using the current joint angles and link lengths.

        Joint angles:
            - theta1: Angle of the first joint (base joint) relative to the horizontal axis.
            - theta2: Angle of the second joint relative to the first link.
            - theta3: Angle of the third joint relative to the second link.
        """
        with lock:
            theta1, theta2, theta3 = [np.radians(a) for a in joint_angles]
        l1, l2, l3 = self.lengths
        x0, y0 = 0, 0
        x1 = l1 * np.cos(theta1)
        y1 = l1 * np.sin(theta1)
        x2 = x1 + l2 * np.cos(theta1 + theta2)
        y2 = y1 + l2 * np.sin(theta1 + theta2)
        x3 = x2 + l3 * np.cos(theta1 + theta2 + theta3)
        y3 = y2 + l3 * np.sin(theta1 + theta2 + theta3)
        x_coords = [x0, x1, x2, x3]
        y_coords = [y0, y1, y2, y3]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x_coords, y_coords, '-o', markersize=8, label='Arm', color='blue')
        ax.plot(x3, y3, 'ro', markersize=10, label='End Effector(start point)')
        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        self.canvas.draw()


if __name__ == '__main__':
    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(target=lambda: RoboticArmAPI().run())
    flask_thread.setDaemon(True)
    flask_thread.start()
    app_qt = QApplication(sys.argv)
    ui = RoboticArmUI()
    ui.show()
    sys.exit(app_qt.exec_())
