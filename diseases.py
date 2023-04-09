from tkinter import *
from tkinter import messagebox
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
window = Tk()
window.title("Team Prediction")  # to define the title
# window.geometry('400x400')

frame = Frame(window, bg="red")
frame.place(relx=0.25, relwidth=0.5, relheight=0.9)
var = StringVar(window)


def click():
    disease_code = {
        0: 'Paroymsal Positional Vertigo(vertigo)',
        1: 'Aids',
        2: 'Acne',
        3: 'Alcoholic Hepatitis',
        4: 'Allergy',
        5: 'Arthritis',
        6: 'Bronchial Asthma',
        7: 'Cervical Spondylosis',
        8: 'Chicken Pox',
        9: 'Chronic Cholestasis',
        10: 'Common Cold',
        11: 'Dengue',
        12: 'Diabetes',
        13: 'Dimorphic Hemmorhoids(Piles)',
        14: 'Drug Reaction',
        15: 'Fungal Infection',
        16: 'Gastroesophageal Reflux Disease',
        17: 'Gastroenteritis',
        18: 'Heart attack',
        19: 'Hepatitis B',
        20: 'Hepatitis C',
        21: 'Hepatitis D',
        22: 'Hepatitis E',
        23: 'Hypertension',
        24: 'Hyperthyroidism',
        25: 'Hypoglycemia',
        26: 'Hypothyroidism	',
        27: 'Impetigo',
        28: 'Jaundice',
        29: 'Malaria',
        30: 'Migraine',
        31: 'Osteoarthristis',
        32: 'Paralysis (Brain Hemorrhage)',
        33: 'Peptic Ulcer Diseases',
        34: 'Pneumonia',
        35: 'Psoriasis',
        36: 'Tuberculosis',
        37: 'Typhoid',
        38: 'Urinary Tract Infection',
        39: 'Varicose Veins',
        40: 'Hepatitis A',
        41: 'Blood Cancer'
    }

    url = 'https://prediction-system-backend-services.onrender.com/detailed-report/'
    testdata = pd.read_json(url+var.get(),
                            orient='records')
    testdata.drop(['case_id', 'symptoms_id'], axis=1, inplace=True)
    traindata = pd.read_csv("C:/Users/HP/Training.csv")
    traindata = traindata.loc[:, ~traindata.columns.str.contains('^Unnamed')]
    label = LabelEncoder()
    traindata["prognosis"] = label.fit_transform(traindata["prognosis"])
    ytrain = traindata['prognosis']
    xtrain = traindata.drop(['prognosis'], axis=1)
    lr = LogisticRegression(solver='newton-cg')
    lr.fit(xtrain, ytrain)
    dis_by_lr = lr.predict(testdata)
    # print(disease_code[dis_by_lr[0]])

    ins_url = 'https://prediction-system-backend-services.onrender.com/save-disease/'
    dis_insert = requests.post(ins_url+var.get()+"/"+disease_code[dis_by_lr[0]]);
    print(dis_insert.status_code)
    messagebox.showinfo("Team Prediction" , "Diagnosis Report Generated & Send to user id : "+var.get())


Label(frame, text="Team Prediction", fg="white", bg="red", font=("Arial", 35)).place(x=220, y=100)
Label(frame, text="Case Id :", fg="white", bg="red", font=("Arial", 25)).place(x=160, y=230)
Entry(frame, font=("Arial", 25), width=16, justify="center", textvariable=var).place(x=320, y=230)
Button(frame, text="Predict", command=click, fg="black", bg="white", font=("Arial", 15), activebackground="yellow").place(x=350, y=330)
window.mainloop()

