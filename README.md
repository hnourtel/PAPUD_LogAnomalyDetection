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
File tree structure
Lauching script

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
