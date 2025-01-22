# TODO: check the list of jobs with find jobs function predict labels based on that
# find the matching credentials and services with that job posting
# if matches found rank based on location

import os
from dotenv import load_dotenv
import mysql.connector
import re
import logging
import tensorflow
import json
import random
from webookcare.models.dl.patients.credentials_recommendation import main as credentials_recommender
from webookcare.models.dl.patients.service_recommendation import main as service_recommender
# from webookcare.models.dl.caregivers.careservices.retrieve_services_info import check_services
# from webookcare.models.dl.caregivers.qualifications.retrieve_credentials import check_credentials
# from webookcare.models.dl.patients.location_ranking.locations import sort_locations
# from webookcare.queries_api.patient.data_models import PatientReq
# from webookcare.queries_api.caregiver.data_models import CareGiverReq
# from webookcare.models.dl.patients.credentials_recommendation import main as credentials_recommender
from typing import Tuple, List

load_dotenv()

DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")



# numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# matching the services requested with care givers services
def get_services(caregiver_id) -> List[str]:
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            database=DB_NAME
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
        
        command = f"""
        SELECT 
            care_service_caregiver.caregiver_id, 
            care_service_caregiver.care_service_id
        FROM care_service_caregiver
        WHERE care_service_caregiver.caregiver_id = {caregiver_id};
        """
        cursor.execute(command)
        caregiver_services = cursor.fetchall()
        
        services = []
        for i in caregiver_services:
            for j in care_services:
                if i[1] == j[0]:
                    services.append(j[1])
        return services
        
    finally:
        cursor.close()
        connection.close()

# def find_jobs(database_state:bool=False):
def find_jobs(caregiver_id, 
              credentials=None, 
              services=None):
    potential_jobs = []
    # TODO: to be implemented later
    # if database_state:
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = connection.cursor(dictionary=True)  # Enables DictCursor
        command = """
        SELECT
            jobs.job_id,
            jobs.patient_id,
            jobs.job_description,
            jobs.credentials,
            jobs.services
        FROM jobs;
        """
        cursor.execute(command)
        jobs = cursor.fetchall()  # Each row is now a dictionary
        for job in jobs:
            job['credentials'] = json.loads(job['credentials'])
            job['services'] = json.loads(job['services'])
        # return jobs  # Directly returns a list of dictionaries
    finally:
        cursor.close()
        connection.close()
    
    services = get_services(caregiver_id)
    jobs_of_interest = []
    all_jobs = [random.choice(jobs) for i in range(20)]
    all_jobs = [(i['job_id'],
                 i['job_description'],
                 service_recommender.predict_data(i['job_description']),
                 credentials_recommender.predict_data(i['job_description'])) \
                     for i in all_jobs]
    
    # TODO: could be used as backup for services:
    # # list of bigram services required passed in
        # looping through the list of bigram services
    # for i in jobs:
        # tmp_sent = ' '.join(i['services'])
        # for z in services:
            # print(z)
            # # splitting the bigram into words
            # tmp_serv = z.split()
            # # for each word
            # for x in tmp_serv:
                # if re.findall(rf"\b{x}\b", tmp_sent, re.IGNORECASE):
                    # if i['job_id'] not in jobs_of_interest:
                        # jobs_of_interest.append(i['job_id'])
    
    
    for i in all_jobs:
        if i[2]:
            tmp_sent = ' '.join(i[2])
            for j in services:
                if re.findall(rf"\b{j}\b", tmp_sent, re.IGNORECASE):
                    if i[0] not in jobs_of_interest:
                        jobs_of_interest.append(i[0])
    return jobs_of_interest
    


# def match_credentials(credentials) -> List[str]:
#     """
#     Gives back the list of caregivers of interest.

#     This function takes a requested credentials and compares it against a dictionary
#     of available caregiver credentials to find matching caregivers.

#     Args:
#         credentials: The requested care credentials to match against available caregiver 
#             credentials. Expected to be a credentials enum value (e.g., credentials.CPR).

#     Returns:
#         list: A list of potential caregivers matching the requested credentials:
#             {
#                 'caregiver_id_1',
#                 'caregiver_id_2',
#                 ...
#             }

#     Raises:
#         ValueError: If the credentials parameter is not a valid Credentials enum value
#         KeyError: If the credentials is not found in the available credentials

#     Example:
#         >>> available_matches = match_credentials(credentials.CPR)
#         >>> print(available_matches)
#         [
#             'caregiver_1',
#             'caregiver_2',
#         ]

#     Note:
#         The check_credentials() function is called internally to get the current
#         dictionary of available caregiver credentials.
#     """
#     caregivers_credentials_dict = check_credentials()
#     # print(caregivers_credentials_dict)
#     # TODO: can add a logging method to detect if no credentials are passed in
#     # TODO: can set a threshold for specific number of credentials offered by the caregiver for matching
#     # even if one of the credentials is offered by the caregiver suggest them
#     # list of unigram credentials required passed in
#     if credentials:
#         caregivers_oi = []
#         # looping through the list of unigram credentials
#         for i in credentials:
#             # splitting the unigram into words
#             # looping through caregivers info of credentials
#             for j in caregivers_credentials_dict:
#                 # TODO: clean up potentially wrong codes below
#                 if i in caregivers_credentials_dict[j]:
#                     if j not in caregivers_oi:
#                         caregivers_oi.append(j)
#     else:
#         # raise Exception('credentials list is empty!')
#         # in case no credentials is found
#         # caregivers_oi=[None,]
#         caregivers_oi=None
#     return caregivers_oi

# # TODO: finish implementing
# def rank_locations(patient_location:Tuple[float, float], 
#                    targets:List[Tuple[float, float],]) -> List[str]:
#     sorted_locations = sort_locations(patient_location, 
#                                       targets)
#     sorted_locations = [i[0] for i in sorted_locations]
#     return sorted_locations


# def rank_patients(caregiver: caregiverReq) -> Tuple[List[str], List[str], List[str]]:
#     # Extract validated data from the Patient model
#     criteria = {
#         'patient_id': patient.patient_id,
#         'job_description': patient.job_description,
#         'rate': patient.rate,
#         'healthcare_setting': patient.healthcare_setting,
#         'property_type': patient.property_type,
#         'health_condition': patient.health_condition,
#         'caregiver_type': patient.caregiver_type,
#         'credentials': patient.credentials,
#         'careservices': patient.careservices,
#         'budget': patient.budget,
#         'care_date': patient.care_date,
#         'care_location': patient.care_location
#     }
#     credentials: List[str] = credentials_recommender.predict_data(
#         criteria['job_description']
#     )
#     services: List[str] = service_recommender.predict_data(
#         criteria['job_description']
#     )
#     potential_caregivers_cred: List[str] | None = match_credentials(credentials)
#     potential_caregivers_serv: List[str] | None = match_services(services)
#     potential_caregivers = []
#     count = 0
#     for i in (potential_caregivers_cred, 
#               potential_caregivers_serv):
#         if i is not None:
#             assert isinstance(i, list)
#             for j in i:
#                 if j not in potential_caregivers:
#                     potential_caregivers.append(j)
#         else:
#             count += 1
#         if count == 2:
#             print('No credentials or services found!')
    
#     # TODO: finish implementing
#     potential_caregivers = rank_locations(
#         patient.care_location,
#         potential_caregivers
#     )
    
#     # TODO: finish implementing
#     potential_caregivers = rank_reviews(criteria['patient_id'],
#                                         potential_caregivers)
    
#     return potential_caregivers