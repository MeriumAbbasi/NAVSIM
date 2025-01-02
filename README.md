# NAVSIM

**NAVSIM** is a navigation sensor simulator designed to generate data for IMU (Inertial Measurement Unit) and GPS sensors based on user-defined trajectories. The simulator features a user-friendly GUI and backend classes for advanced computations, making it an ideal tool for testing navigation algorithms and studying sensor behavior.

## Features

### Frontend (GUI)
The NAVSIM GUI is divided into two main tabs:

1. **Initial Conditions Tab**:
   - **IMU Parameters**:
     - Output frequency (Hz).
     - Sensor errors: constant bias, random drift, misalignment, scale factor error, and lever arm.
   - **GPS Parameters**:
     - Randomness in velocity and position.
   - **Initial Conditions**:
     - Position: Latitude, Longitude, and Altitude.
     - Velocity: Initial velocity.
     - Orientation: Heading, Pitch, and Roll.
   - **Additional Options**:
     - Pitch-Roll Coupling (checkbox).
     - Smoothing Type for trajectory: 3rd order smooth or uniformly accelerated.
     - Root Folder Selection: Browse to select the directory for saving the resulting IMU and GPS data files.

2. **Trajectory Tab**:
   - Define trajectory segments by specifying:
     - Duration of the segment.
     - Final velocity.
     - Change in orientation (Heading, Pitch, and Roll).
   - Visualize the trajectory as a graph dynamically as segments are added.
   - Generate IMU and GPS data for the defined trajectory, stored as text files in the specified root folder.

### Backend (Code)
NAVSIM's backend is built with a modular class structure:

1. **`TrajChar` Class**:
   - Defines constant trajectory characteristics, including initial conditions, smoothing type, coupling, and frequency.

2. **`Trajectory` Class**:
   - Contains methods to generate the base trajectory based on user inputs.

3. **`IMU` Class**:
   - Adds IMU parameters to the base trajectory.
   - Generates IMU data: instantaneous acceleration and angular rate.
   - Outputs IMU data as text files at the user-specified frequency.

4. **`GPS` Class**:
   - Adds GPS parameters to the base trajectory.
   - Generates GPS data with position and velocity randomness.
   - Outputs GPS data as text files.

## How It Works

1. **Setup Initial Conditions**:
   - Enter IMU and GPS parameters.
   - Define initial position, velocity, and orientation.
   - Choose pitch-roll coupling and smoothing type.
   - Select a root folder to save data.

2. **Define Trajectory**:
   - Add trajectory segments by specifying duration, final velocity, and orientation changes.
   - Visualize the trajectory graph as you add segments.

3. **Generate Sensor Data**:
   - The backend computes IMU and GPS outputs based on the trajectory.
   - Data is saved in text files within the selected root folder.

## File Structure
- **GUI**: User interface files.
- **Backend Classes**:
  - `Traj_char`: Defines constant trajectory characteristics.
  - `Trajectory`: Generates the base trajectory.
  - `IMU`: Computes IMU data.
  - `GPS`: Computes GPS data.
- **Output**: Generated IMU and GPS text files.

## Installation
1. Access https://github.com/MeriumAbbasi/NAVSIM.git
2. Download Exe File
3. Extract and Run IMU-NavSim.exe

## Usage
1. Launch the GUI.
2. Define initial conditions in the **Initial Conditions Tab**.
3. Add trajectory segments in the **Trajectory Tab**.
4. Generate IMU and GPS data by clicking the "Generate" button.
5. Access the output data in the selected root folder.

## Future Enhancements
- Support for additional sensors (e.g., magnetometers, barometers).
- Enhanced visualization tools for sensor data.
- Integration with real-time navigation algorithms.

## License
This project is licensed under the MIT License.

## Contribution
Contributions are welcome! Feel free to fork the repository, create issues, or submit pull requests.

---

**Developed by Merium Fazal Abbasi**

For questions or feedback, please contact abbasimerium@gmail.com.


