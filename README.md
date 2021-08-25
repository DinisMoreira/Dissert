# Deep Reinforcement Learning for Automated Parking

## How to run

* Create an environment with the python package and environment management system "Conda"
`conda create --name <Environment Name>`

* Activate the Conda environment:
`conda activate <Environment Name>`



* To customize the running environment configure lines 102-130 of the "parking_env.py" file according to your preferences:
            `"simulation_frequency": 15,`<br/>`
            "policy_frequency": 5,`<br/>`
            "duration": 300,`<br/>`
            "screen_width": 1920 * 2,`<br/>`
            "screen_height": 1080 * 2,`<br/>`
            "centering_position": [0.5, 0.5],`<br/>`
            "scaling": 10 * 2*2,`<br/>`
            "controlled_vehicles": 1,`<br/>`
            "collision_reward": -0.1,`<br/>`
            "layoutType": 0,`<br/>`
            "gridSizeX": 6,`<br/>`
            "gridSizeY": 2,`<br/>`
            "gridSpotWidth": 4,`<br/>`
            "gridSpotLength": 8,`<br/>`
            "corridorWidth": 9,`<br/>`
            "orientationMode": 7,`<br/>`
            "trackRear": 1,`<br/>`
            "randomInitialState": 0,`<br/>`
            "initialPosition": [[20, 0],[20, 1],`<br/>`
            "initialHeading": 0,`<br/>`
            "startingPhase": 2,`<br/>`
            "endingPhase": 3,`<br/>`
            "obstacles": 1,`<br/>`
            "otherVehicles": 1,`<br/>`
            "generateNewPaths": 0,`<br/>`
            "pathsFileName": "paths_6x2",`<br/>`
            "randomPath": 0,`<br/>`
            "goalSpotNumber": 7,`<br/>`
            "initialPositionNumber": 6`
            
* To use a trained agent in the simulation run the "useTrainedParking.py" file:
`python useTrainedParking`

* To train a new agent run the "trainParking.py" file:
`python trainParking`




Note: Make sure the following package dependencies are installed in the conda environment:<br/>
`# Name                    Version                   Build  Channel`<br/>
`absl-py                   0.11.0                   pypi_0    pypi`<br/>`
astor                     0.8.1                    pypi_0    pypi`<br/>`
atari-py                  0.2.6                    pypi_0    pypi`<br/>`
backcall                  0.2.0                    pypi_0    pypi`<br/>`
ca-certificates           2021.1.19            h06a4308_0  `<br/>`
cached-property           1.5.2                    pypi_0    pypi`<br/>`
certifi                   2020.12.5        py37h06a4308_0  `<br/>`
cloudpickle               1.6.0                    pypi_0    pypi`<br/>`
cvxpy                     1.1.12                   pypi_0    pypi`<br/>`
decorator                 4.4.2                    pypi_0    pypi`<br/>`
easyprocess               0.3                      pypi_0    pypi`<br/>`
ecos                      2.0.7.post1              pypi_0    pypi`<br/>`
et-xmlfile                1.1.0                    pypi_0    pypi`<br/>`
finite-mdp                1.0.dev0                  dev_0    <develop>`<br/>`
future                    0.18.2                   pypi_0    pypi`<br/>`
gast                      0.2.2                    pypi_0    pypi`<br/>`
google-pasta              0.2.0                    pypi_0    pypi`<br/>`
grpcio                    1.36.0                   pypi_0    pypi`<br/>`
gym                       0.18.0                    dev_0    <develop>`<br/>`
h5py                      3.1.0                    pypi_0    pypi`<br/>`
highway-env               1.0.dev0                  dev_0    <develop>`<br/>`
importlib-metadata        3.7.0                    pypi_0    pypi`<br/>`
ipython                   7.21.0                   pypi_0    pypi`<br/>`
ipython-genutils          0.2.0                    pypi_0    pypi`<br/>`
jedi                      0.18.0                   pypi_0    pypi`<br/>`
joblib                    1.0.1                    pypi_0    pypi`<br/>`
keras-applications        1.0.8                    pypi_0    pypi`<br/>`
keras-preprocessing       1.1.2                    pypi_0    pypi`<br/>`
ld_impl_linux-64          2.33.1               h53a641e_7  `<br/>`
libedit                   3.1.20191231         h14c3975_1  `<br/>`
libffi                    3.3                  he6710b0_2  `<br/>`
libgcc-ng                 9.1.0                hdf63c60_0  `<br/>`
libstdcxx-ng              9.1.0                hdf63c60_0  `<br/>`
markdown                  3.3.4                    pypi_0    pypi`<br/>`
matplotlib                3.4.1                    pypi_0    pypi`<br/>`
mpi4py                    3.0.3                    pypi_0    pypi`<br/>`
ncurses                   6.2                  he6710b0_1  `<br/>`
networkx                  2.5                      pypi_0    pypi`<br/>`
numpy                     1.20.2                   pypi_0    pypi`<br/>`
opencv-python             4.5.1.48                 pypi_0    pypi`<br/>`
openpyxl                  3.0.7                    pypi_0    pypi`<br/>`
openssl                   1.1.1j               h27cfd23_0  `<br/>`
opt-einsum                3.3.0                    pypi_0    pypi`<br/>`
osqp                      0.6.2.post0              pypi_0    pypi`<br/>`
pandas                    1.2.4                    pypi_0    pypi`<br/>`
parso                     0.8.1                    pypi_0    pypi`<br/>`
pexpect                   4.8.0                    pypi_0    pypi`<br/>`
pickleshare               0.7.5                    pypi_0    pypi`<br/>`
pillow                    7.2.0                    pypi_0    pypi`<br/>`
pip                       21.0.1           py37h06a4308_0  `<br/>`
prompt-toolkit            3.0.18                   pypi_0    pypi`<br/>`
protobuf                  3.15.3                   pypi_0    pypi`<br/>`
ptyprocess                0.7.0                    pypi_0    pypi`<br/>`
pyglet                    1.5.0                    pypi_0    pypi`<br/>`
pygments                  2.8.1                    pypi_0    pypi`<br/>`
python                    3.7.9                h7579374_0  `<br/>`
pyvirtualdisplay          2.1                      pypi_0    pypi`<br/>`
qdldl                     0.1.5.post0              pypi_0    pypi`<br/>`
readline                  8.1                  h27cfd23_0  `<br/>`
rl-agents                 1.0.dev0                  dev_0    <develop>`<br/>`
scipy                     1.6.3                    pypi_0    pypi`<br/>`
scs                       2.1.3                    pypi_0    pypi`<br/>`
setuptools                52.0.0           py37h06a4308_0  `<br/>`
six                       1.15.0                   pypi_0    pypi`<br/>`
sqlite                    3.33.0               h62c20be_0  `<br/>`
stable-baselines          2.10.1                   pypi_0    pypi`<br/>`
tensorboard               1.15.0                   pypi_0    pypi`<br/>`
tensorflow-estimator      1.15.1                   pypi_0    pypi`<br/>`
tensorflow-gpu            1.15.0                   pypi_0    pypi`<br/>`
termcolor                 1.1.0                    pypi_0    pypi`<br/>`
tk                        8.6.10               hbc83047_0  `<br/>`
tqdm                      4.59.0                   pypi_0    pypi`<br/>`
traitlets                 5.0.5                    pypi_0    pypi`<br/>`
typing-extensions         3.7.4.3                  pypi_0    pypi`<br/>`
wcwidth                   0.2.5                    pypi_0    pypi`<br/>`
werkzeug                  1.0.1                    pypi_0    pypi`<br/>`
wheel                     0.36.2             pyhd3eb1b0_0  `<br/>`
wrapt                     1.12.1                   pypi_0    pypi`<br/>`
xlsxwriter                1.4.3                    pypi_0    pypi`<br/>`
xz                        5.2.5                h7b6447c_0  `<br/>`
zipp                      3.4.0                    pypi_0    pypi`<br/>`
zlib                      1.2.11               h7b6447c_3  `
