from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple
from datetime import datetime
from enum import Enum
from webookcare.data_structs.structs import (
    HealthcareSetting, 
    PropertyType, 
    HealthCondition, 
    CareWorkerType, 
    CareServices, 
    Qualifications
    )

# TODO: to be implemented after patient ensemble model
class CareGiverReq(BaseModel):
    """
    Pydantic model representing a patient care request with all necessary details.
    
    This model includes information about the patient, required care services,
    caregiver requirements, location details, and scheduling information.
    
    Attributes:
        patient_id (int): Unique identifier for the patient
        job_description (str, optional): Detailed description of care requirements
        rate (float, optional): Hourly rate for the care service
        healthcare_setting (HealthcareSetting, optional): Type of healthcare setting
        property_type (PropertyType, optional): Type of property where care is needed
        health_condition (HealthCondition, optional): Patient's health condition
        caregiver_type (CaregiverType, optional): Required type of caregiver
        credentials (List[Credentials], optional): Required caregiver credentials
        careservices (List[CareServices], optional): Required care services
        budget (float, optional): Total budget for the care service
        care_date (float, optional): Unix timestamp for when care is needed
        care_location (Tuple[float, float], optional): Geographic coordinates
    """
    caregiver_id: int = Field(
        gt=0, 
        description="Unique identifier for the patient"
    )
    rate: float | None = Field(
        default=None, 
        gt=0, 
        lt=1000,
        description="Hourly rate in dollars"
    )
    healthcare_setting: HealthcareSetting | None = Field(
        default=None,
        description="Type of healthcare setting HCW can work in"
    )
    property_type : PropertyType | None = Field(
        default=None,
        description="Type of property HCW can work in"
    )
    health_condition : HealthCondition | None = Field(
        default=None,
        description="health condition of the patients HCW can work with"
    )
    caregiver_type : CareWorkerType | None = Field(
        default=None,
        description="Type of caregiver"
    )
    credentials : list[Qualifications] | None = Field(
        default=None,
        description="Credentials of the caregiver"
    )
    careservices : list[CareServices] | None = Field(
        default=None,
        description="care services provided by the HCW",
        max_items=10
    )
    care_dates : float | None = Field(
        default=None,
        description="the dates HCW can work for."
    )
    caregiver_location : tuple[float, float] | None = Field(
        default=None,
        description="Geographic coordinates (latitude, longitude) of the HCW location"
    )

    @validator('credentials')
    def validate_credentials(cls, v):
        """
        Validates that at least one credential is specified when credentials are provided.
        
        Args:
            v: List of credentials to validate
            
        Returns:
            The validated credentials list
            
        Raises:
            ValueError: If the credentials list is empty
        """
        if v is not None and len(v) == 0:
            raise ValueError("At least one credential must be specified")
        return v

    @validator('careservices')
    def validate_careservices(cls, v):
        """
        Validates that at least one care service is specified when services are provided.
        
        Args:
            v: List of care services to validate
            
        Returns:
            The validated care services list
            
        Raises:
            ValueError: If the care services list is empty
        """
        if v is not None and len(v) == 0:
            raise ValueError("At least one care service must be specified")
        return v

    @validator('caregiver_location')
    def validate_coordinates(cls, v):
        """
        Validates that the provided coordinates are within valid ranges.
        
        Args:
            v: Tuple of (latitude, longitude) coordinates
            
        Returns:
            The validated coordinates tuple
            
        Raises:
            ValueError: If coordinates are outside valid ranges
        """
        if v is not None:
            lat, lon = v
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError("Invalid coordinates: latitude must be between -90 and 90, longitude between -180 and 180")
        return v
    
    @validator('care_dates')
    def validate_date(cls, v):
        """
        Validates that the care date is not in the past.
        
        Args:
            v: Unix timestamp to validate
            
        Returns:
            The validated timestamp
            
        Raises:
            ValueError: If the date is in the past
        """
        if v is not None:
            date = datetime.fromtimestamp(v)
            if date < datetime.now():
                raise ValueError("Care date cannot be in the past")
        return v

    @property
    def formatted_care_date(self) -> str | None:
        """
        Formats the care date timestamp as a human-readable string.
        
        Returns:
            str: Formatted date string in "YYYY-MM-DD HH:MM:SS" format,
                 or None if no date is set
        """
        if self.care_dates is None:
            return None
        return datetime.fromtimestamp(self.care_dates).strftime("%Y-%m-%d %H:%M:%S")

    class Config:
        validate_assignment = True
        extra = "forbid"
        json_schema_extra = {
            "examples": [
                {
                    "caregiver_id": 1,
                    "rate": 35.50,
                    "healthcare_setting": "home",
                    "property_type": "house",
                    "health_condition": "stable",
                    "caregiver_type": "certified_nursing_assistant",
                    "credentials": ["certified_nursing_assistant", "cpr_certified"],
                    "careservices": ["medication_management", "mobility_assistance", "personal_hygiene"],
                    "care_dates": 1735689600,  # Example future timestamp
                    "caregiver_location": (40.7128, -74.0060)  # Example coordinates for New York
                }
            ]
        }