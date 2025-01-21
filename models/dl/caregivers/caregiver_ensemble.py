# TODO: check the list of jobs with find jobs function predict labels based on that
# find the matching credentials and services with that job posting
# if matches found rank based on location

import os
from dotenv import load_dotenv
import mysql.connector
import re
import logging
import tensorflow
# from webookcare.models.dl.patients.credentials_recommendation import main as credentials_recommender
# from webookcare.models.dl.patients.service_recommendation import main as service_recommender
# from webookcare.models.dl.caregivers.careservices.retrieve_services_info import check_services
# from webookcare.models.dl.caregivers.qualifications.retrieve_credentials import check_credentials
# from webookcare.models.dl.patients.location_ranking.locations import sort_locations
# from webookcare.queries_api.patient.data_models import PatientReq
from webookcare.queries_api.caregiver.data_models import CareGiverReq
from webookcare.models.dl.patients.credentials_recommendation import main as credentials_recommender
from webookcare.tools.NLP.misc import (
    determine_shapes,
    predict
)
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


def find_jobs(database_state:bool=False):
    # TODO: to be implemented later
    if database_state:
        connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_NAME
        )
        cursor = connection.cursor()
        command = f"""
        SELECT
            job.patient_id,
            jobs.job_description
        FROM jobs;
        """
        cursor.execute(command)
        jobs = cursor.fetchall()
    else:
        # TODO: determine if retrain options needs to be added
        # TODO: create ngrams based on labels available and do
        # predictions based on that to retrieve list of credentials
        # and services
        determine_shapes(credentials_recommender.current_dir,
                         save_req=False,
                         data_file_path=credentials_recommender.DATA_FILE_PATH,
                         labels_file_path=credentials_recommender.LABELS_FILE_PATH,
                         num_ngrams=credentials_recommender.NUM_NGRAMS)
        

# # matching the services requested with care givers services
# def match_services(service) -> List[str]:
#     """
#     Gives back the dictionary of caregivers and the services they offer.

#     This function takes a requested service and compares it against a dictionary
#     of available caregiver services to find matching caregivers.

#     Args:
#         service: The requested care service to match against available caregiver services.
#                 Expected to be a CareServices enum value (e.g., CareServices.WOUND_CARE).

#     Returns:
#         list: A list of matching caregivers based on the requested service:
#             [
#                 'caregiver_id_1',
#                 'caregiver_id_2',
#                 ...
#             ]

#     Raises:
#         ValueError: If the service parameter is not a valid CareServices enum value
#         KeyError: If the service is not found in the available services

#     Example:
#         >>> available_matches = match_services(CareServices.WOUND_CARE)
#         >>> print(available_matches)
#         [
#             'caregiver_1',
#             'caregiver_2',
#         ]

#     Note:
#         The check_services() function is called internally to get the current
#         dictionary of available caregiver services.
#     """
#     caregivers_services_dict = check_services()
    
#     # TODO: can add a logging method to detect if no services are passed in
#     # TODO: can set a threshold for specific number of services offered by the caregiver for matching
#     # even if one of the services is offered by the caregiver suggest them
#     # list of bigram services required passed in
#     if service:
#         caregivers_oi = []
#         # looping through the list of bigram services
#         for i in service:
#             # splitting the bigram into words
#             tmp_serv = i.split()
#             # for each word
#             for x in tmp_serv:
#                 # looping through caregivers info of services
#                 for y in caregivers_services_dict:
#                     for z in y:
#                         tmp_sent = ' '.join(y[z])
#                         if re.findall(rf"\b{x}\b", tmp_sent, re.IGNORECASE):
#                             if z not in caregivers_oi:
#                                 caregivers_oi.append(z)
#     else:
#         # raise Exception('Service list is empty!')
#         # in case no service is found
#         # caregivers_oi=[None,]
#         caregivers_oi=None
#     # print(caregivers_oi)
#     return caregivers_oi

# # matching the services requested with care givers services
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

# # TODO: finish implementing
# def rank_reviews(patient_id:int,
#                  patients_oi:List[str]) -> List[str]:
#     sorted_reviews = sort_reviews(patient_id,
#                                   patients_oi)
#     sorted_reviews = [i[0] for i in sorted_reviews]
#     return sorted_reviews

# # TODO: need to delegate job_description None value to make raw predictions if no value passed
# # to the recommenders
# def rank_caregivers(patient: PatientReq) -> Tuple[List[str], List[str], List[str]]:
#     """
#     Ranks caregivers based on patient requirements defined in the Patient model.

#     This function evaluates and ranks caregivers based on the criteria specified
#     in the Patient model, including qualifications, availability, location, and 
#     care requirements. The validation of inputs is handled automatically by the
#     Pydantic model.

#     Args:
#         patient (Patient): A validated Patient model instance containing all
#             patient requirements and preferences.

#     Returns:
#         Tuple[List[str], List[str], List[str]]: A tuple containing three lists:
#             - credentials: Caregiver IDs that perfectly match all criteria
#             - services: Caregiver IDs that match most important criteria
#             - potential_caregivers: Caregiver IDs that match some criteria

#     Raises:
#         ValueError: If patient model validation fails
#         TypeError: If input is not a Patient model instance

#     Example:
#         >>> patient_data = {
#         ...     "patient_id": 123,
#         ...     "healthcare_setting": "home",
#         ...     "caregiver_type": "registered_nurse",
#         ...     "credentials": ["registered_nurse", "cpr_certified"],
#         ...     "care_location": (40.7128, -74.0060)
#         ... }
#         >>> patient = Patient(**patient_data)
#         >>> credentials, services, potential_caregivers = rank_caregivers(patient)
#         >>> print(potential_caregivers[:3])  # Print top 3 perfect matches
#         ['caregiver_789', 'caregiver_456', 'caregiver_123']
#     """
#     # Type validation
#     # if not isinstance(patient, PatientReq):
#         # raise TypeError("Input must be an instance of Patient model")

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

#     # TODO: Implement the actual ranking logic here
#     # This would involve:
#     # 1. Querying available caregivers
#     # 2. Matching against criteria
#     # 3. Scoring and ranking
#     # 4. Sorting into appropriate match categories
#     # potential_caregivers = []
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