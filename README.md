## Install environment, pnlp on Chicoma
1) create or locate `venvs` folder
2) `module load cray-python/3.11.5`
3) `python3.11 -m venv ./venvs/spike`, adjust to where `venvs` folder is located
3) `source ./venvs/spike/bin/activate`, adjust to where `venvs` folder is located
4) if from main folder (`Spike_NLP-Lightning`), 
    - `pip install --no-cache-dir -r requirements/torchreq.txt` (may not be needed, below will install torch via dependencies)
    - `pip install -e .`