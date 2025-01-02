import re
import typer
import logging
import tensorflow
from webookcare.patients.credentials_recommendation import main as credentials_recommender
from webookcare.patients.service_recommendation import main as service_recommender
from webookcare.caregivers.careservices.retrieve_services_info import check_services
import webookcare.patients.service_recommendation
from webookcare.queries_api.patient.data_models import Patient
from typing import Tuple, List
from datetime import datetime

app = typer.Typer()

# numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def match_services(service):
    """
    Gives back the dictionary of caregivers and the services they offer.

    This function takes a requested service and compares it against a dictionary
    of available caregiver services to find matching caregivers.

    Args:
        service: The requested care service to match against available caregiver services.
                Expected to be a CareServices enum value (e.g., CareServices.WOUND_CARE).

    Returns:
        dict: A dictionary mapping caregiver IDs to their matching services:
            {
                'caregiver_id_1': ['service1', 'service2'],
                'caregiver_id_2': ['service1', 'service3'],
                ...
            }

    Raises:
        ValueError: If the service parameter is not a valid CareServices enum value
        KeyError: If the service is not found in the available services

    Example:
        >>> available_matches = match_services(CareServices.WOUND_CARE)
        >>> print(available_matches)
        {
            'caregiver_123': ['wound_care', 'medication_management'],
            'caregiver_456': ['wound_care', 'vital_signs_monitoring']
        }

    Note:
        The check_services() function is called internally to get the current
        dictionary of available caregiver services.
    """
    caregivers_services_dict = check_services()
    
    # list of bigram services required passed in
    if service:
        caregivers_oi = []
        for i in service:
            tmp_serv = i.split()
            # for each word in the bigram
            for x in tmp_serv:
                # looping through caregivers info of services
                for y in caregivers_services_dict:
                    for z in y:
                        tmp_sent = ' '.join(y[z])
                        if re.findall(rf"\b{x}\b", tmp_sent, re.IGNORECASE):
                            if z not in caregivers_oi:
                                caregivers_oi.append(z)
    else:
        # raise Exception('Service list is empty!')
        # in case no service is found
        caregivers_oi=[None,]
    # print(caregivers_oi)
    return caregivers_oi

# TODO: need to delegate job_description None value to make raw predictions if no value passed
# to the recommenders
def rank_caregivers(patient: Patient) -> Tuple[List[str], List[str], List[str]]:
    """
    Ranks caregivers based on patient requirements defined in the Patient model.

    This function evaluates and ranks caregivers based on the criteria specified
    in the Patient model, including qualifications, availability, location, and 
    care requirements. The validation of inputs is handled automatically by the
    Pydantic model.

    Args:
        patient (Patient): A validated Patient model instance containing all
            patient requirements and preferences.

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple containing three lists:
            - credentials: Caregiver IDs that perfectly match all criteria
            - services: Caregiver IDs that match most important criteria
            - potential_caregivers: Caregiver IDs that match some criteria

    Raises:
        ValueError: If patient model validation fails
        TypeError: If input is not a Patient model instance

    Example:
        >>> patient_data = {
        ...     "patient_id": 123,
        ...     "healthcare_setting": "home",
        ...     "caregiver_type": "registered_nurse",
        ...     "credentials": ["registered_nurse", "cpr_certified"],
        ...     "care_location": (40.7128, -74.0060)
        ... }
        >>> patient = Patient(**patient_data)
        >>> credentials, services, potential_caregivers = rank_caregivers(patient)
        >>> print(potential_caregivers[:3])  # Print top 3 perfect matches
        ['caregiver_789', 'caregiver_456', 'caregiver_123']
    """
    # Type validation
    if not isinstance(patient, Patient):
        raise TypeError("Input must be an instance of Patient model")

    # Extract validated data from the Patient model
    criteria = {
        'patient_id': patient.patient_id,
        'job_description': patient.job_description,
        'rate': patient.rate,
        'healthcare_setting': patient.healthcare_setting,
        'property_type': patient.property_type,
        'health_condition': patient.health_condition,
        'caregiver_type': patient.caregiver_type,
        'credentials': patient.credentials,
        'careservices': patient.careservices,
        'budget': patient.budget,
        'care_date': patient.care_date,
        'care_location': patient.care_location
    }

    # TODO: Implement the actual ranking logic here
    # This would involve:
    # 1. Querying available caregivers
    # 2. Matching against criteria
    # 3. Scoring and ranking
    # 4. Sorting into appropriate match categories
    # potential_caregivers = []
    credentials: List[str] = credentials_recommender.predict(
        criteria['job_description']
    )
    services: List[str] = service_recommender.predict(
        criteria['job_description']
    )
    potential_caregivers: List[str] = match_services(services)
    return (credentials, 
            services, 
            potential_caregivers)
