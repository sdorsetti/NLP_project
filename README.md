# NLP_project
This repository is organized in the following way :     
- A configuration file : config.py
- A folder for preprocessing 
- A folder for labellizing the data
- A folder for the user location study, with the appropriate models and training. 
- A folder for the hashtag study. 

In order to use this repository, one needs to change the structure_dict, in the configuration file, in order to replace the path by the actual ones. 
One needs also the initial dataframe, which was pushed on github, and a configuration dictionnary in json format at the corresponding path in structure_dict, which stores the main arguments for the trainings.
Finally, the shapefiles of the world boundaries are mandatory. They are also pushed on github
Once these settings done, it is easy to use the repository on this colab. It will store the results into the path written in the structure dict.