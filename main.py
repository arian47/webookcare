from fastapi import FastAPI
import webookcare
import json
from webookcare.models.dl.patients.patient_ensemble import rank_caregivers
from webookcare.queries_api.patient.data_models import PatientReq

#TODO: implementation of data model structure for CareGiver
# from webookcare.queries_api.caregiver.data_models import CareGiver

app = FastAPI()


# @app.post("/caregivers/")
# async def get_candidates(caregiver: CareGiver):
    # pass


@app.post("/patients/")
async def get_candidates(patient: PatientReq):
    patient_dict = patient.dict()
    caregivers_oi = rank_caregivers(
    patient_id = patient_dict.get('patient_id'),
    job_description = patient_dict.get('job_description'),
    rate = patient_dict.get('rate'),
    healthcare_setting = patient_dict.get('healthcare_setting'),
    property_type = patient_dict.get('property_type'),
    health_condition = patient_dict.get('health_condition'),
    caregiver_type = patient_dict.get('caregiver_type'),
    credentials = patient_dict.get('credentials'),
    careservices = patient_dict.get('careservices'),
    budget = patient_dict.get('budget'),
    care_date = patient_dict.get('care_date'),
    care_location = patient_dict.get('care_location'),)
    
    # response = dict(
        # credentials=credentials,
        # services=services,
        # potential_caregivers=potential_caregivers
    # )
    # return json.dumps(response)
    # return response
    return caregivers_oi
    
    # test case
    # return {"message": "Patient data received successfully.", 
            # "data": patient.dict()}


if __name__ == "__main__":
    app()