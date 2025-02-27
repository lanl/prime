## Install environment, pnlp on Chicoma
1) create or locate `venvs` folder
2) module load cray-python/3.11.5
3) `source ../../venvs/spike/bin/activate`, adjust to where `venvs` folder is located
4) if from `spike_nlp` folder, 
    - `pip install --no-cache-dir -r requirements/torchreq.txt`
    - `pip install -e .`