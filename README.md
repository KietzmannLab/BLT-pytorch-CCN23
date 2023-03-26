# BLT model for pytorch

### Based on pytorch 1.12.1 and python 3.10

### Training:
- Training files are available in "BLT_train/" 
- You can call "train_net.py" with args that can be configured.
- Make sure to define 'dataset_path' in train_net.py
- You can write new models and place them in "models/". Make sure to update loading "get_network_model" function in helper_funcs, and other dependencies.

### Current performance:
- On one A100, we can train a BLT RNN with a 10 timestep unroll and a batch size of 1024. It takes ~3hrs for 60 epochs. The final timestep accuracy is 50% on the testplus set of miniecoset-100.

### Evaluation:
- Evaluation files are available in "BLT_analyse/" 
- Run "extract_actvs.py" to extract the activations for the 'testplus' split of MiniEcoset and readout weights required for further analysis
- Make sure to define 'dataset_path' in extract_actvs.py
- The model files under "models/" are simply modified from the model files in "BLT_train/" to extract the relevant metrics
- Run "analysis.ipynb" to interact with the saved metrics and visualise the results
