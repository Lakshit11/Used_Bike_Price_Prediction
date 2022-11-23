from operator import methodcaller
from flask import Flask,render_template, request
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods = ["GET","POST"])
def predict():
    Name = request.form["bike"].upper()
    KM = request.form["KM"]
    Age = request.form["Age"]
    owner = request.form["Owner"]
    form_array = np.array([[Name, KM, Age, owner]])
    
    # Importing data
    data = pd.read_csv('https://raw.githubusercontent.com/Lakshit11/Used-Bike-Price-Prediction/main/Used_Bikes.csv')
    data.head()
    
    # dropping redundant col
    data2 = data.drop(['power','brand','city'],axis = 1)
    
    # changing type to int of floats
    data2['price'] = data2['price'].astype('int32')
    data2['kms_driven'] = data2['kms_driven'].astype('int32')
    data2['age'] = data2['age'].astype('int32')
    
    # removing outliers for age of bike
    upper_limit=data2['age'].mean()+2.3*data2['age'].std() 
    data3 = data2.loc[(data2['age']<upper_limit)]
    # removing outliers for kms_driven
    upper_limit_1=data3['kms_driven'].mean()+3.4*data3['kms_driven'].std()
    data4 = data3.loc[(data3['kms_driven']<upper_limit_1)]
    # removing outliers for price
    upper_limit_2=data4['price'].mean()+0.4*data4['price'].std()
    data5 = data4.loc[(data4['price']<upper_limit_2)]
    
    # Encoding
    le_owner = LabelEncoder()
    data5['Owner_Type'] = le_owner.fit_transform(data5['owner'])
    data5.drop('owner',axis = 1, inplace=True)
    data5['bike_name'] = data5['bike_name'].str.upper()
    dum = pd.get_dummies(data5['bike_name'])
    data6 = pd.concat([data5, dum],axis = 1)
    final_data = data6.drop(['bike_name'],axis = 1)
    
    # splitting
    X = final_data.drop('price',axis = 1)
    y = final_data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # dumping the model
    pickle.dump(model, open("bike.pkl","wb"))
    if Name in X.columns:
        loc_index = np.where(X.columns==Name)[0][0]
        x = np.zeros(len(X.columns)) 
        x[0] = KM
        x[1] = Age
        x[2] = owner
        if loc_index >= 0:
            x[loc_index] = 1

        model = pickle.load(open("bike.pkl","rb"))
        pred_price = round(model.predict([x])[0],2)
        result = pred_price
        return render_template("result.html", result = result,name = Name, age = Age, km = KM)
    else:
        return render_template("error.html", name = Name)


if __name__ == "__main__":
    app.run(debug=True)