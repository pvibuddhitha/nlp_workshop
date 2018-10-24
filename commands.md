##### to login
* ssh user_xxxx@graham.computecanada.ca

##### make sure to load the relevant python interpreter (this is to make sure that your virtual environment will be based on the particular python interpreter)

* module load python/3.6

#### creating the virtual environment
* virtualenv tfgpu

#### activating the virtual environment
* source ~/tfgpu/bin/activate

#### to exit the environment
* deactivate

##### installing libraries
* module load scipy-stack (installs: NumPy, SciPy, Matplotlib, dateutil, pytz, IPython, pyzmq, tornado, pandas, Sympy, nose)
##### installing sklearn
* pip install -U scikit-learn

##### installing tensorflow
* pip install tensorflow_gpu

##### Listing available wheels
* avail_wheels --name "pandas"

##### check available modules
* module ava

##### List the wheels specifically for GPU and display only name, version, python columns:
* avail_wheels --column name version python --all_versions --name "*gpu"

##### get usage and help
avail_wheels --help

_______________________________________________

#### submitting batch jobs:
##### to submit a job
* sbatch tensorflow-test.sh

##### to view the job queue
* squeue

##### to view your job
* squeue -u pkiri056

##### to view who is running jobs under your supervisor's account
* squeue -A def-inkpen_gpu

#### to activate tensorboard (include in the bash file)
* ./tensorlogs/emo --host 0.0.0.0 &

##### to get the hostname of the node where the job is running
* squeue --job xxxxxx -o %N

##### once you have the host name
* ssh -N -f -L localhost:6006:computenode:6006 user_xxx@graham.computecanada.ca




