import numpy as np
import json
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')

@app.route('/',methods = ['GET'])
def home():
    return render_template('index2.html')

standard_to = StandardScaler()

@app.route('/predict', methods = ['POST'])

def predict():
    if request.method == 'POST':
       # Customer_ID = int(request.form['Customer_ID'])
        Age = int(request.form['Age'])
        #Country = request.form['Country']
                
       # GENDER = str(request.form['GENDER'])
        #Mode_Of_Payment = str(request.form['Mode_Of_Payment'])
        Amount_Spent = int(request.form['Amount_Spent'])
       # odds1 = float(request.form['odds1'])
        odds2 = float(request.form['odds2'])
        No_Transactions = int(request.form['No_Transactions'])
        No_Bettings = int(request.form['No_Bettings'])
        No_Payments = int(request.form['No_Payments'])

        prediction = model1.predict([[ Age, Amount_Spent,odds2, No_Transactions, No_Bettings, No_Payments]])
        
        output = round(prediction[0], 2)
        
        if output == 0:
            return render_template('index2.html', prediction_text = 'Is Fraud ')
        else:
            return render_template('index2.html', prediction_text = 'Is not a Fraud ')
        
        '''
        if prediction == 1:
            return render_template('index2.html', prediction_text = 'Not A Fraud')
        if prediction == 0:
            
            return render_template('index2.html', prediction_text = 'Is A Fraud')
    else:
        return render_template('index2.html')
        '''
if __name__ == "__main__":
    app.run(debug=True) 
#def predict():
    
    #For Rendering results on HTML GUI
    
    #data = [object(x) for x in request.form.values()]
   # final_features = [np.array(data)]
   # prediction = model.predict(final_features)
    
   # output = round(prediction[0], 2)
    
   # return render_template('index1.html', prediction_text = 'Is Fraud $ {}'.format(output))







