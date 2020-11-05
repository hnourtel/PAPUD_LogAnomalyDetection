# Log Anomaly Detection Tool
This repository provides a tool detecting anomalies in logs.

It was developed at LORIA in the context of ITEA3/PAPUD project.

## Tool presentation
The tool aims to analyze system logs to detect log line outliers. These outliers are interpreted as anomalies in the logs,
in the sense that the system does not function normally because it does not output classic logs. It can be used to detect
cyber-attacks, software failures, hardware failures and all malfunctions of a system that have impact on the logs.

It uses deep learning techniques for NLP to analyze the logs stored in a flat file. It consists of two parts :
* A line encoder : a model is trained for each word of the line, it predicts the word given all other words as input
* A one-class classifier : a DeepSVDD model is trained for each word, using encoder's output as input.
All DeepSVDD scores are summed to determine total score for the line. 

## Installation
This tool is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your local machine in the directory of your choice:
```
git clone https://github.com/hnourtel/PAPUD_LogAnomalyDetection.git
```

## Getting started
The script `launchScript.py` provides an example of the pipeline execution of the tool.
For the example, it uses the LANL Cybersecurity dataset (https://csr.lanl.gov/data/cyber1/).
You need to download the `auth.txt` file and `redteam.txt` file to use example script.   
### Directory tree structure
The tool needs a specific data structure to work. Here is the example with LANL dataset.
If you are using another dataset, juste replace LANL by your corpus name in the directory tree.
You need to create the following tree whereever you want on your disk :

LANL_Data  
├── LANL_Corpus  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── train  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── dev  
│   &nbsp;&nbsp;&nbsp;&nbsp;└── test  
└─ LANL_Model  

`LANL_Data` contains all data needed for execution :
* Corpus data in LANL_Corpus
* Trained models in LANL_Model
* A file for saving vocabulary calculated during execution
* redteam file for testing the anomaly detection model

`train` directory contains training files with line extracted from `auth.txt` file  
`dev` directory contains files used to control learning during training part  
`test` directory contains files used to test the anomaly detection model  
  
### Running script
The script `launchScript` needs only one parameter to run : the data directory path (here the path to `LANL_Data`).  
Other parameters are directly sets in `launchScript.py` as the two different models need different parameter values.

Run the script using the following command :
`python3 launchScript.py /home/myFiles/LANL_Data`

## Contact
If you would like to get in touch on this subject, contact `hubert.nourtel@gmail.com` or `cerisara@loria.fr` 

## Copyright
Copyright LORIA, Hubert Nourtel, Christophe Cerisara, Samuel Cruz-Lara

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
