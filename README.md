# vagalProcessingPipeline
Processing pipeline that utilizes jnormcorre and cellpose to compress multi photon calcium imaging experiments of the vagal ganglia


# Coming Soon:
- Working on updates to analyze gas-change experiments
- likely will rework processing pipeline to be cleaner and automatically generate expmtNotes by reading tiff metadata moving forward

# Installation:
```
git clone https://github.com/LKINSEY/vagalProcessingPipeline.git

cd .\path\to\repo

conda create --name dataAnalysis python=3.10

conda activate dataAnalysis

pip install git+https://github.com/apasarkar/jnormcorre.git

python -m pip install cellpose[gui]

pip install openpyxl

pip install alicat

```
(may need to download additional dependencies found here https://github.com/MouseLand/cellpose) -- this is also why env should be python=3.10
- this installs pyqt6 as well


# How To:

1.) After experiment is finished, place experiment in an experiment records folder along with past experiments and generate expmtNotes_yymmdd_####.xlsx file according to notes made from experiment
- expmtNotes will be automatically generated in the future
- make sure experiment folder name is titled as /"experimentersName_yymmdd"/

2.) Make sure all experiments are in the same directory. Specify directory and run registerAndExtractTrials.py 
- this first iteration of running this script registers every trial and generates tiffs that allow user to count cells (either manually or with AI. AI is to come once model is trained/generated for WGA data)

3.) Once trials are registered AND annotated cellCountingTiffs using cellpose(https://github.com/MouseLand/cellpose), 
- this will eventually be automated using cellpose once new model is trained
- run cellpose in a seperate instance (for now.... this part will be better integrated into the future)
- slice 1 should indicate WGA+/gCaMP- ROIs
- slice 2 should indicate WGA+/gCaMP+ ROIs
- slice 3 should indicate WGA-/gCaMP+ ROIs

4.) run registerAndExtractTrials.py again -- the logic will funnel analysis to compressing data into an expmtTraces.pkl

5.) once pickle is created, run readTraces.py -- this will generate summary plots of your experiment in expmtName/figures directory


# References:

maskNMF: A denoise-sparsen-detect approach for extracting neural signals from dense imaging data. (2023). A. Pasarkar*, I. Kinsella, P. Zhou, M. Wu, D. Pan, J.L. Fan, Z. Wang, L. Abdeladim, D.S. Peterka, H. Adesnik, N. Ji, L. Paninski.

Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106

Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. Nature methods, 1-8.

# Alicat Interface for Changing Gas Composition:
![image](https://github.com/user-attachments/assets/c4867de7-e2e2-4344-9acd-367ca1a1c1d5)

