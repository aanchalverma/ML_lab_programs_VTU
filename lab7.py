import csv
import bayespy as bp
import numpy as np
from colorama import init

init()

ageEnum = {'SuperSeniorCitizen': 0, 'SeniorCitizen' : 1, 'MiddleAged' : 2, 'Youth' : 3, 'Teen' : 4}

genderEnum = {'Male':0, 'Female':1}

familyHistoryEnum = {'Yes' : 0, 'No' : 1}

dietEnum = {'High':0, 'Medium' : 1,'Low' : 2}

lifeStyleEnum = { 'Athlete' : 0, 'Active' : 1, 'Moderate' : 2, 'Sedetary' : 3 }

cholestrolEnum = {'High' : 0, 'BorderLine' : 1, 'Normal' : 2 }

heartDiseaseEnum = {'Yes':0, 'No':1 }

with open('heart_disease_data.csv') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    data =[]
    for x in dataset:
        data.append([ageEnum[x[0]],genderEnum[x[1]],familyHistoryEnum[x[2]],dietEnum[x[3]],lifeStyleEnum[x[4]],cholestrolEnum[x[5]],heartDiseaseEnum[x[6]]])
    data = np.array(data)
    N = len(data)

p_age = bp.nodes.Dirichlet(1.0 * np.ones(5))
age = bp.nodes.Categorical(p_age, plates=(N,))
age.observe(data[:,0])

p_gender = bp.nodes.Dirichlet(1.0*np.ones(2))
#you forgot
gender = bp.nodes.Categorical(p_gender, plates = (N,))
gender.observe(data[:,1])

p_family = bp.nodes.Dirichlet(1.0*np.ones(2))
family = bp.nodes.Categorical(p_family, plates = (N,))
family.observe(data[:,2])

p_diet = bp.nodes.Dirichlet(1.0*np.ones(3))
diet = bp.nodes.Categorical(p_diet, plates = (N,))
diet.observe(data[:,3])

p_life = bp.nodes.Dirichlet(1.0*np.ones(4))
life = bp.nodes.Categorical(p_life, plates = (N,))
life.observe(data[:,4])

p_chol = bp.nodes.Dirichlet(1.0*np.ones(3))
chol = bp.nodes.Categorical(p_chol, plates =(N,))
chol.observe(data[:,5])

p_hd = bp.nodes.Dirichlet(np.ones(2), plates = (5,2,2,3,4,3))
#you forgot
hd = bp.nodes.MultiMixture([age,gender,family,diet,life,chol],bp.nodes.Categorical,p_hd)
hd.observe(data[:,6])
hd.update()

m = 0
#wrong condition
while(m ==0):
    #you forgot
    age=int(input('Enter the age:'+str(ageEnum)))
    gender = int(input('Enter the gender:'+str(genderEnum)))
    family = int(input('Enter the family:'+str(familyHistoryEnum)))
    diet = int(input('Enter the diet:'+str(dietEnum)))
    life = int(input('Enter the lifestyle:'+str(lifeStyleEnum)))
    cholestrol = int(input('Enter the cholestrol:'+str(cholestrolEnum)))
    #res = bp.nodes.MultiMixture([int(input('Enter Age:'+str(ageEnum))),int(input('Enter Gender:'+str(genderEnum))),int(input('Enter Family History:'+str(familyHistoryEnum))),int(input('Enter diet:'+str(dietEnum))),int(input('Enter life style:'+str(lifeStyleEnum))),int(input('Enter cholesterol:'+str(cholesterolEnum)))],bp.nodes.Categorical,p_heartdisease).get_moments()[0][heartDiseaseEnum['Yes']]

    res = bp.nodes.MultiMixture([age, gender, family, diet, life, cholestrol],bp.nodes.Categorical,p_hd).get_moments()[0][heartDiseaseEnum["Yes"]]
    
    print("Prediction value:",str(res))
    m = int(input("Continue:0 Exit:1"))

