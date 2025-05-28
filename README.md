# PRDF-CP2K-LAMMPS
GUI for calculating partial radial distribution functions (PRDF) and RDF from CP2K XYZ or LAMMPS trajectories. Additionally, it can union files for XYZ atomic configurations and CELL file from CP2K. 

# How to compile
1) git clone https://github.com/bracerino/SQS-GUI.git
2) cd SQS-GUI/
3) python3 -m venv sqs_env
4) source sqs_env/bin/activate
5) pip install -r requirements.txt
6) streamlit run app.py
