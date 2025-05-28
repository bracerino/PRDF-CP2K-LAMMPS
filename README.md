# PRDF-CP2K-LAMMPS
GUI for calculating partial radial distribution functions (PRDF) and RDF from CP2K XYZ or LAMMPS trajectories. Additionally, it can union files for XYZ atomic configurations and CELL file from CP2K. 

git clone https://github.com/bracerino/SQS-GUI.git
cd SQS-GUI/
python3 -m venv sqs_env
source sqs_env/bin/activate
pip install -r requirements.txt
streamlit run app.py
