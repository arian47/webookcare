import os
from dotenv import load_dotenv
import re
import random
from webookcare.models.dl.patients.credentials_recommendation import main as credentials_recommender
from webookcare.models.dl.patients.service_recommendation import main as service_recommender
from webookcare.models.dl.caregivers.careservices.retrieve_services_info import check_services
from webookcare.models.dl.caregivers.qualifications.retrieve_credentials import check_credentials
from webookcare.models.dl.caregivers.jobs.jobs import retrieve_jobs
# from webookcare.models.dl.patients.location_ranking.locations import sort_locations
from webookcare.queries_api.caregiver.data_models import CareGiverReq
from typing import Tuple, List

load_dotenv()

DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")   

# # TODO: finish implementing locations sorting after 
# having appropriate data in the database
# def rank_locations(patient_location:Tuple[float, float], 
                #    targets:List[Tuple[float, float],]) -> List[str]:
    # sorted_locations = sort_locations(patient_location, 
                                    #   targets)
    # sorted_locations = [i[0] for i in sorted_locations]
    # return sorted_locations

# TODO: can later be delegated to the specific model for jobs
def rank_patients(caregiver: CareGiverReq) -> Tuple[List[str], 
                                                    List[str], 
                                                    List[str]]:
    # Extract validated data from the Patient model
    criteria = {
        'caregiver_id': caregiver.caregiver_id,
        'rate': caregiver.rate,
        'healthcare_setting': caregiver.healthcare_setting,
        'property_type': caregiver.property_type,
        'health_condition': caregiver.health_condition,
        'caregiver_type': caregiver.caregiver_type,
        'credentials': caregiver.credentials,
        'careservices': caregiver.careservices,
        'care_dates': caregiver.care_dates,
        'caregiver_location': caregiver.caregiver_location
    }
        
    # get all the jobs
    jobs = retrieve_jobs()
    
    # get caregivers services and credentials
    services = check_services(criteria['caregiver_id'])
    credentials = check_credentials(criteria['caregiver_id'])
    
    JIF = True
    CJIF = 0
    
    while JIF:
        # for efficiency only retain 20 jobs randomly
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
        
        jobs_of_interest = []
        for i in all_jobs:
            if i[2]:
                tmp_sent = ' '.join(i[2])
                for j in services:
                    if re.findall(rf"\b{j}\b", tmp_sent, re.IGNORECASE):
                        if i[0] not in jobs_of_interest:
                            jobs_of_interest.append(i[0])
            if i[3]:
                tmp_sent = ' '.join(i[3])
                for j in credentials:
                    for k in credentials[j]:
                        if re.findall(rf"\b{k}\b", tmp_sent, re.IGNORECASE):
                            if i[0] not in jobs_of_interest:
                                jobs_of_interest.append(i[0])
        if jobs_of_interest or CJIF <=5:
            JIF = False
        if not jobs_of_interest:
            CJIF += 1
    
    return jobs_of_interest
    
    # # TODO: finish implementing
    # potential_caregivers = rank_locations(
    #     caregiver.care_location,
    #     potential_caregivers
    # )
    
    # return potential_caregivers
