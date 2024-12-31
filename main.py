from fastapi import FastAPI
from pydantic import BaseModel
import webookcare
import json

import webookcare.patient_ensemble

app = FastAPI()


class Patient(BaseModel):
    patient_id: int
    job_description: str | None = None
    rate: float | None = None
    healthcare_setting: str | None = None
    care_location : tuple[float, float] | None = None # latitude and longitude of the carelocation
    property_type : str | None = None
    health_condition : str | None = None
    caregiver_type : str | None = None
    credentials : str | None = None
    careservices : list[str] | None = None
    budget : float | int | None = None
    care_date : float | None = None
    


@app.post("/patients/")
async def get_candidates(patient: Patient):
    patient_dict = patient.dict()
    (
        credentials,
        services,
        potential_caregivers
    ) = webookcare.patient_ensemble.rank_caregivers(
        patient_id = patient_dict.get('patient_id'),
        patient_req = patient_dict.get('job_description')
    )
    
    response = dict(
        credentials=credentials,
        services=services,
        potential_caregivers=potential_caregivers
    )
    # return json.dumps(response)
    return response
    
    
    # test case
    # return {"message": "Patient data received successfully.", 
            # "data": patient.dict()}


if __name__ == "__main__":
    app()