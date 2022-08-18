# ANCRNet
Code for ANCR-Net, a CNN for the segmentation and modelling of new multiple sclerosis lesions

Appearance adaptation and non-correspondence segmentation network as described in our Frontiers paper: 
Image Registration and Appearance Adaptation in Non-Correspondent Image Regions for New MS Lesions Detection
(Julia Andresen, Hristina Uzunova, Jan Ehrhardt, Timo Kepp and Heinz Handels) Accepted for publication.

To use the scripts in this repository you need to download the MSSeg-2 challenge data: 
https://portal.fli-iam.irisa.fr/msseg-2/data/
https://shanoir.irisa.fr/shanoir-ng/challenge-request

The network is used to segment new MS lesions, perform image registration and model the appearance of new lesions at the same time.


To train ANCR-Net use (pretraining.py and) training.py.
Before starting the scripts enter the paths to your data directory and the path to store the results into.
Pre-training is performed on artificially deformed images with synthetic new MS lesions. 
The main training then uses the pretrained network and the ground-truth training.


We will soon add a trained version of ANCR-Net together with a script showing how to use the trained network.
