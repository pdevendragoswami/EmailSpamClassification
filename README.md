# How to run the app?

Step1: conda create -n EmailSpamClassification python=3.8 -y 

Step2: conda activate EmailSpamClassification

Step3: pip install -r requirements.txt

Step4: python app.py

Step5: Now open up your local host 0.0.0.0:8080

# Workflows:

1. Create a folder for your project
2. Create environment for your project - conda create -n EmailSpamClassification python=3.8 -y
3. Activate your environment env - conda activate EmailSpamClassification
4. Create template.py file - echo. > template.py
5. Define the project template and run the file - python template.py
6. Define the requirements.txt and setup.py file
7. Run the requirements.txt file - pip install -r requirements.txt
8. Define the constant path in constants-init.py file
9. Define logger function in logging folder
10. Define the utils functions that are commonly used.
11. Define config.yaml and params.yaml file for stage-01-data_ingestion
12. Write code from stage 01 data ingestion in notebook
13. Define config_entity.py file and configuration.py for the stage 01
14. write modular code for stage 01 with referece from notebook for component and pipeline
15. Repeat step from 11 to 15 for each step.