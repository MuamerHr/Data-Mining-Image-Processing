#################################################################################
#Preliminaries:
All solutions require the opencv library. The solution to task 7 requires a newer version,
where the SIFT implementation is avalaible again. It should not matter wether you use the 
library opencv-python or opencv-contrib-python.

To install opencv either search in the anaconda navigator environment and install the module
or type into the cmd.exe / terminal:

conda install -c conda-forge-opencv

pip install opencv-python
pip3 install opencv-python


Also have a look at the folder "conda" which contains the package configuration
for anaconda and pip and the folder "additional_info" to see
which modules and packages where used during implementation and testing.

################################################################################
Structure of solution folders q1,...,q7:
-Every folder qx contains the input images and at least one folder with the results, 
that is the folder "out".

-Every folder qx contains exactly one file which has to be executed, that is:
folder	executable		runtime			
q1:		hw1_1.py		fast
q2:		hw1_2.py		fast
q3:		hw1_3.py		moderate - fast
q4:		q4.py			moderate -fast
q5:		q5.py			fast
q6:		hw1_6.py		fast
q7:		q5.py			slow
	
#Run code:
We strongly recommend to use spyder or python3 for running the scripts.
The scripts either can be run line by line, block by block, or the whole script at once
in both spyder and python over the terminal. But take care of the mentioned runtimes
(at least with q7.py).



