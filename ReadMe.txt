Instructions to execute the project:

Packages needed:
- sklearn
- skimage
- matplotlib
- scipy
- itertools (for ploting Confusion matrix)

-----------------------------------------------------------------------------------
Download the following dataset
Dataset link: http://jvgemert.github.io/pub/EgyptianHieroglyphDataset.tar.gz
------------------------------------------------------------------------------------

Files: 

1) bounding.py	: File used to read one image, segments the image and stores in a folder
2) resize.py	: File used to read the segmented hieroglyphs and resizes it
3) proj.py		: File used to train the dataset
4) testProj.py 	: File used to test the dataset
5) f_utils.py 	: Support File containing common functions for the previous files
------------------------------------------------------------------------------------

Note: Have all the files in the same directory

Detection:
	Part 1:
		-> To execute "bounding.py": Open the file and change the image path at line 10 and run ("python bounding.py"). This file will take a slab and segment it
			Note: slabs are present in "\EgyptianHieroglyphDataset\pictures\"
		-> It will create a folder called 'resize' in the current directory.
	
	Part 2:
		-> To execute 'resize.py': type the following command in the command line:
				"python resize.py"
		-> The size of all the images in the 'resize' folder will be resized (50x75)
	
Training:
	-> Go to the dataset folder "\EgyptianHieroglyphDataset\ExampleSet7\train"
	-> Copy and paste the 'train' folder in the current directory, so that current directory has
		a folder called 'train'; under which all the class folder are present. For example, to access an image called '050072_G17.jpg'
		
		Path to the a sample image would be: \train\G17\050072_G17.jpg
		
	-> To train this dataset, type the following command in the command line (make sure present working directory is the currect directory):
			"python proj.py -t train\"
	
	-> The entire dataset will take about 15-20 mins to be processed.
	-> Training the dataset is complete, you should be able to see a 'bag.pkl' in the current directory
	
Testing:
	-> Create a folder called 'test' in the current directory
	-> Create a subfolder called 'tt' (inside the 'test' folder)
	-> Go to the dataset folder "\EgyptianHieroglyphDataset\ExampleSet7\test"
	-> Copy and paste the content of 'test' folder inside "test\tt\". 
		Sample image path would be: \test\tt\070001_G1.jpg
	
	-> To test the dataset, type the following command in the command line (make sure present working directory is the currect directory):
			"python testProj.py -t test\ --v"
	-> The output can be seen on the command line: 
		- it will have two column of values: Test images and predicted class 
		- it will also create a 'confusion or error matrix': to display this matrix, uncomment lines 153-157 (testProj.py: by default it is commented)
			Note: To visualize the matrix clearly, make sure to use a very small number of class in the training (6-8)
		- Finally it will display the specifics of confusion matrix, like True positive, True Negative, False positive, False negative and others

Note: 	1) The result of the detection can be used in Testing. To do this:
			Instead of copying the content of "\EgyptianHieroglyphDataset\ExampleSet7\test"; copy the content of "resize" folder which was created by 'Detection' process
		2) Since the detected image has a lot of noise, the result might not be desirable (while testing the images from 'Detection' process). 
		