#Preliminaries:
This project only runs with the additional moduls opencv, pylsd and imutils.

To install opencv either search in the anaconda navigator environment and install the module
or type into the cmd.exe / terminal:

conda install -c conda-forge-opencv

pip install opencv-python
pip3 install opencv-python

To use pylsd install it with
conda install -c

or less problematic

pip3 install ocrd-fork-pylsd
pip3 install pylsd
pip3 install pylsd-nova

To use imutils install it with

conda install -c conda-forge-imutils

or

pip3 install imutils

Also have a look at the folder "conda" which contains the package configuration
for anaconda and pip and the folder "additional_info" to see
which modules and packages where used during implementation and testing.

#Run code:
The files main.py, train.py and test.py include the main
parts of the project. 
We strongly recommend to use spyder or python3 for running the scripts.
The scripts either can be run line by line, block by block, or the whole script at once
in both spyder and python over the terminal.

#Folders:
additional_info: Decribed above
conda: Described above
data: Contains subfolders which contain train, test and process data. The subfolder 02_svm contains
the data of the lasted trained SVM.
examples: Contains sketches of circuits.
images: Contains images used for the report.
output: Contains output of the different stages for visualisation with mathplot.pyplot
stages: Contains images for the different recognition steps.
Do not delete the folders if not necessary (except additional_info, conda and images)! 
Otherwise the code won't work.




