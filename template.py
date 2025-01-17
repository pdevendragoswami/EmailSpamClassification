import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "EmailSpamClassification"



list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/stage_01_data_ingestion.py",
    f"src/{project_name}/components/stage_02_data_validation.py",
    f"src/{project_name}/components/stage_03_data_transformation.py",
    f"src/{project_name}/components/stage_04_model_trainer.py",
    f"src/{project_name}/components/stage_05_model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/utils.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_data_transformation.py",
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py", 
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "README.md",
    "research/trials.ipynb",
    "research/stage_01_data_ingestion.ipynb",
    "research/stage_02_data_validation.ipynb",
    "research/stage_03_data_transformation.ipynb",
    "research/stage_04_model_trainer.ipynb",
    "research/stage_05_model_evaluation.ipynb",
    f"research/{project_name}.ipynb",
    "templates/index.html",
    "templates/results.html"


]



for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")