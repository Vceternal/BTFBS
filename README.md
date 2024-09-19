# BTFBS



##  Dependencies
  
Quick install: `pip install -r requirements.txt`
  
Dependencies:
- python 3.7+
- pytorch >=1.2
- numpy
- sklearn
- prefetch_generator
  
##  Usage
  
`python main.py <Bdata_RS/EE>  [-s,--seed] [-f,--fold]`
  
Parameters:
- `-s` or `--seed` : set random seed, *optional*
- `-f` or `--fold` : set K-Fold number, *optional*
  
##  Project Structure
  
- DataSets: Data used in paper.
- utils: A series of tools.
- config.py: model config.
- LossFunction.py: Loss function used in paper.
- main.py: main file of project.
- model.py: Proposed model in paper.
- README.md: this file
- requirements.txt: dependencies file
- RunModel.py: Train, validation and test programs.
  
