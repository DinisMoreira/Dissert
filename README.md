# Deep Reinforcement Learning forAutomated Parking

## How to run

Create an environment with the python package and environment management system "Conda"
`conda create --name <Environment Name>`

Activate the Conda environment:
`conda activate <Environment Name>`

Make sure the following packages are installed in the conda environment
`# Name                    Version                   Build  Channel`<br/>
`absl-py                   0.11.0                   pypi_0    pypi
astor                     0.8.1                    pypi_0    pypi
atari-py                  0.2.6                    pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
ca-certificates           2021.1.19            h06a4308_0  
cached-property           1.5.2                    pypi_0    pypi
certifi                   2020.12.5        py37h06a4308_0  
cloudpickle               1.6.0                    pypi_0    pypi
cvxpy                     1.1.12                   pypi_0    pypi
decorator                 4.4.2                    pypi_0    pypi
easyprocess               0.3                      pypi_0    pypi
ecos                      2.0.7.post1              pypi_0    pypi
et-xmlfile                1.1.0                    pypi_0    pypi
finite-mdp                1.0.dev0                  dev_0    <develop>
future                    0.18.2                   pypi_0    pypi
gast                      0.2.2                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.36.0                   pypi_0    pypi
gym                       0.18.0                    dev_0    <develop>
h5py                      3.1.0                    pypi_0    pypi
highway-env               1.0.dev0                  dev_0    <develop>
importlib-metadata        3.7.0                    pypi_0    pypi
ipython                   7.21.0                   pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
jedi                      0.18.0                   pypi_0    pypi
joblib                    1.0.1                    pypi_0    pypi
keras-applications        1.0.8                    pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20191231         h14c3975_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.1.0                hdf63c60_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
markdown                  3.3.4                    pypi_0    pypi
matplotlib                3.4.1                    pypi_0    pypi
mpi4py                    3.0.3                    pypi_0    pypi
ncurses                   6.2                  he6710b0_1  
networkx                  2.5                      pypi_0    pypi
numpy                     1.20.2                   pypi_0    pypi
opencv-python             4.5.1.48                 pypi_0    pypi
openpyxl                  3.0.7                    pypi_0    pypi
openssl                   1.1.1j               h27cfd23_0  
opt-einsum                3.3.0                    pypi_0    pypi
osqp                      0.6.2.post0              pypi_0    pypi
pandas                    1.2.4                    pypi_0    pypi
parso                     0.8.1                    pypi_0    pypi
pexpect                   4.8.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    7.2.0                    pypi_0    pypi
pip                       21.0.1           py37h06a4308_0  
prompt-toolkit            3.0.18                   pypi_0    pypi
protobuf                  3.15.3                   pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pyglet                    1.5.0                    pypi_0    pypi
pygments                  2.8.1                    pypi_0    pypi
python                    3.7.9                h7579374_0  
pyvirtualdisplay          2.1                      pypi_0    pypi
qdldl                     0.1.5.post0              pypi_0    pypi
readline                  8.1                  h27cfd23_0  
rl-agents                 1.0.dev0                  dev_0    <develop>
scipy                     1.6.3                    pypi_0    pypi
scs                       2.1.3                    pypi_0    pypi
setuptools                52.0.0           py37h06a4308_0  
six                       1.15.0                   pypi_0    pypi
sqlite                    3.33.0               h62c20be_0  
stable-baselines          2.10.1                   pypi_0    pypi
tensorboard               1.15.0                   pypi_0    pypi
tensorflow-estimator      1.15.1                   pypi_0    pypi
tensorflow-gpu            1.15.0                   pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
tk                        8.6.10               hbc83047_0  
tqdm                      4.59.0                   pypi_0    pypi
traitlets                 5.0.5                    pypi_0    pypi
typing-extensions         3.7.4.3                  pypi_0    pypi
wcwidth                   0.2.5                    pypi_0    pypi
werkzeug                  1.0.1                    pypi_0    pypi
wheel                     0.36.2             pyhd3eb1b0_0  
wrapt                     1.12.1                   pypi_0    pypi
xlsxwriter                1.4.3                    pypi_0    pypi
xz                        5.2.5                h7b6447c_0  
zipp                      3.4.0                    pypi_0    pypi
zlib                      1.2.11               h7b6447c_3  `

To customize the running environment configure lines 102-130 of the "parking_env.py" file according to your preferences:
            `"simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "screen_width": 1920 * 2,
            "screen_height": 1080 * 2,
            "centering_position": [0.5, 0.5],
            "scaling": 10 * 2*2,
            "controlled_vehicles": 1,
            "collision_reward": -0.1,
            "layoutType": 0,
            "gridSizeX": 6,
            "gridSizeY": 2,
            "gridSpotWidth": 4,
            "gridSpotLength": 8,
            "corridorWidth": 9,
            "orientationMode": 7,
            "trackRear": 1,
            "randomInitialState": 0,
            "initialPosition": [[20, 0],[20, 1],
            "initialHeading": 0,
            "startingPhase": 2,
            "endingPhase": 3,
            "obstacles": 1,
            "otherVehicles": 1,
            "generateNewPaths": 0,
            "pathsFileName": "paths_6x2",
            "randomPath": 0,
            "goalSpotNumber": 7,
            "initialPositionNumber": 6`
            
To use a trained agent in the simulation run the "useTrainedParking.py" file:
`python useTrainedParking`

To train a new agent run the "trainParking.py" file:
`python trainParking`
