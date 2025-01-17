from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from EmailSpamClassification.pipeline.prediction import PredictionPipeline
from EmailSpamClassification.components.stage_03_data_transformation import DataTransformationConfig


app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 




@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user

            text = str(request.form['email'])

            data = {'text':[text]}

            data_df = pd.DataFrame(data)
        

            predict_pipeline_obj = PredictionPipeline()

            pred_value = predict_pipeline_obj.predict_value(data_df,config=DataTransformationConfig)

            if pred_value[0] == 0:
                 output = "Not a spam"
            else:
                 output = "Spam"
                 
        
            return render_template('results.html', prediction = (output))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')



if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)