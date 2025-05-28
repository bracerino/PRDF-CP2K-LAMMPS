# PRDF-CP2K-LAMMPS
GUI for calculating partial radial distribution functions (PRDF) and RDF from CP2K XYZ or LAMMPS trajectories. Additionally, it can union files for XYZ atomic configurations and CELL file from CP2K. 

# How to compile
1) git clone https://github.com/bracerino/PRDF-CP2K-LAMMPS.git
3) cd PRDF-CP2K-LAMMPS/
4) python3 -m venv sqs_env
5) source sqs_env/bin/activate
6) pip install -r requirements.txt
7) streamlit run app.py
