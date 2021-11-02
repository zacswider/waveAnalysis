# Instructions for use:	
• If you have never used Python before (that's you, Bill), follow the steps below.

• Click the green "Code" button and select "Download ZIP"

• Move the zipped file to your desktop and double click it to unzip.

• Go to https://www.anaconda.com/products/individual and download Anaconda (a distribution of the Python programming language).

• Once downloaded, open the package and follow the installation instructions.

• Open the terminal by pushing command + space bar, typing "terminal", and hitting enter.

• Paste the following code into the terminal and hit enter:

  	cd desktop

• Paste the following code into the terminal and hit enter:

 	 conda env update --prefix ./env --file environment.yml  --prune
			
• This should result in a number of packages being installed to your computer, it may take a minute.

• Paste the following code into the terminal and hit enter:

  	cd signalProcessing-main

• Paste the following code into the terminal and hit enter:

 	 python3 signalProcessing2C.py

• A window will open up requesting you to select your source workspace. Navigate to the test dataset to ensure that everything is working appropriately: 

			Desktop > signalProcessing-main > testDatasets

• The script should generate two new directories: 0_comparisons and 0_signalProcessing, and a new .csv file: 0_fileStats.csv




