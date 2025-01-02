import numpy
import mysql.connector
import csv
import os
from dotenv import load_dotenv

load_dotenv()

CAREGIVERS_AND_SERVICES_CSV_PATH = 'caregivers_and_services.csv'
CAREGIVERS_AND_SERVICES_NPY_PATH = 'caregivers_and_services.npy'
DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# TODO: transferring save and retrieval of info to a different module.
def save_services(caregivers_and_services):
    tmp = [(j, i.get(j)) for i in caregivers_and_services for j in i]

    with open(CAREGIVERS_AND_SERVICES_CSV_PATH, mode='w', newline='') as fo:
        writer = csv.writer(fo)
        writer.writerows(tmp)

    numpy.save(CAREGIVERS_AND_SERVICES_NPY_PATH, 
               numpy.array(caregivers_and_services, dtype=object))

# TODO: transferring save and retrieval of info to a different module.
def load_services():
    pass

# checking the services offered by all the caregivers which could be time consuming.
# TODO: to break down for better filtering on needs
def check_services(save:bool=False):
    connection = mysql.connector.connect(
    host='localhost',
    user=DB_USERNAME,
    password=DB_PASSWORD,
    database='test'
    )
    cursor = connection.cursor()
    
    command = f"""
    SELECT 
        care_services.id, 
        care_services.name
    FROM care_services;
    """
    cursor.execute(command)
    care_services = cursor.fetchall()
    
    care_services_dict = dict(care_services)
    
    command = f"""
    SELECT 
        care_service_caregiver.caregiver_id, 
        care_service_caregiver.care_service_id
    FROM care_service_caregiver;
    """
    cursor.execute(command)
    caregiver_services = cursor.fetchall()
    
    command = f"""
    SELECT 
        caregivers.id, 
        caregivers.first_name,
        caregivers.last_name
    FROM caregivers;
    """
    cursor.execute(command)
    caregivers = cursor.fetchall()
    
    caregivers_and_services = []

    for i in caregivers:
        tmp = []
        for j in caregiver_services:
            if i[0] == j[0]:
                tmp.append(care_services_dict.get(j[1]))
        name = i[1] + ' ' + i[2]
        caregivers_and_services.append({name:tmp})
    
    if save:
        save_services(caregivers_and_services)
    
    return caregivers_and_services
    
    
    