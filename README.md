# vagalProcessingPipeline
Processing pipeline that utilizes jnormcorre and cellpose to compress multi photon calcium imaging experiments of the vagal ganglia


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





