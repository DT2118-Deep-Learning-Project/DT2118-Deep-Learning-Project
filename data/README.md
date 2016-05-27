# Structure:

## features/

Contains the pre-processed (fft and mel) samples in the npz format.
--> Generated with python script (from (PROJ_DIR) and after sourcing config_env):
$> python source_separation/preprocess/generate_fft_script.py 

## tidigits/

Original folder with the tidigits

## tidigits_{clean, noisy, noise}/

Each folder contains the tidigits samples:
* clean: original one
* noise: the covered noise used for the respective sample
* noisy: the mix between the original and the noise
--> For generated the folders run *make* in the src/ directory

