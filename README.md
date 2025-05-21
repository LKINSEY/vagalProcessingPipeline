# vagalProcessingPipeline
Processing pipeline that utilizes jnormcorre and cellpose to compress multi photon calcium imaging experiments of the vagal ganglia

# Installation:
git clone https://github.com/LKINSEY/vagalProcessingPipeline.git
cd .\path\to\repo
conda create --name myEnv --file requirements.txt
conda activate myEnv

- additionally
pip install git+https://github.com/apasarkar/jnormcorre.git
python -m pip install cellpose
(may need to download additional dependencies found here https://github.com/MouseLand/cellpose) -- this is also why env should be python=3.10


#How To:

1.) After experiment is finished, place experiment in folder structure as shown - generate expmtNotes_yymmdd_####.xlsx file according to notes made from experiment
- expmtNotes will be automatically generated in the future

2.) Make sure all experiments are in the same directory. Specify directory and run extractResGalvo.py if using resonant galvo or extractMechGalvo.py if using a mechanical galvo (this is assuming you are using a Bruker Microscope)
- this first iteration of running this script registers every trial

3.) Once this extract___Galvo.py has been run, you will need to manually curate ROIs absed on data and label you used (currently supporting labeling using WGATR or WGA594)
- this will eventually be automated using cellpose once new model is trained
- run cellpose in a seperate instance (for now.... this part will be better integrated into the future)
- save segmentations made by cellpose as AVG_rT#_C#_ch#_slice#_seg.npy

4.) run extractResGalvoROIs.py again -- the logic will funnel analysis to compressing data into an expmtTraces.pkl

5.) once pickle is created, run readResGalvoTraces.py -- this will generate summary plots of your experiment in expmtName/figures directory


# References:

maskNMF: A denoise-sparsen-detect approach for extracting neural signals from dense imaging data. (2023). A. Pasarkar*, I. Kinsella, P. Zhou, M. Wu, D. Pan, J.L. Fan, Z. Wang, L. Abdeladim, D.S. Peterka, H. Adesnik, N. Ji, L. Paninski.

Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106

Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. Nature methods, 1-8.


