import mysql.connector
from typing import Tuple, List
import os
from dotenv import load_dotenv
from webookcare.models.dl.patients.location_ranking.main import predict

load_dotenv()

DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def sort_locations(patient_location:Tuple[float, float],
                   care_givers:List[str]):
    """
    Sorts the caregiver locations based on the distance from the patient location.

    Parameters
    ----------
    patient_location : Tuple[float, float]
        The location of the patient.
    caregiver_locations : Tuple[Tuple[float, float]]
        The locations of the caregivers.

    Returns
    -------
    Tuple[Tuple[float, float]]
        The sorted locations of the caregivers.
    """
    assert isinstance(care_givers, list), "care_givers must be a list of strings"
    names = [(i.split()[0].lower(), 
              i.split()[1].lower()) for i in care_givers]
    # print(len(names))
    connection = mysql.connector.connect(
    host='localhost',
    user=DB_USERNAME,
    password=DB_PASSWORD,
    database='test'
    )
    cursor = connection.cursor()
    
    locations = []
    invalid_items = []
    for i in names:
        command = f"""
        SELECT
            caregivers.home_latitude,
            caregivers.home_longitude
        FROM caregivers
        WHERE caregivers.first_name = '{i[0]}' 
        AND caregivers.last_name = '{i[1]}';
        """
        # print(command)
        cursor.execute(command)
        try:
            caregivers = cursor.fetchall().pop()
            locations.append(caregivers)
        except IndexError:
            # when there is a null value in the database
            invalid_items.append(i)
    names = [i for i in names if i not in invalid_items]
    # print(len(names))
    # print(len(locations))
    # print(locations)
    invalid_indexs = [i for i in range(len(locations))if locations[i][0]==None or locations[i][1]==None]
    names = [names[i] for i in range(len(names)) if i not in invalid_indexs]
    locations = [locations[i] for i in range(len(locations)) if i not in invalid_indexs]
    locations = [tuple(map(float, i)) for i in locations]
    locations = [[patient_location[0], patient_location[1],
                 i[0], i[1]] for i in locations]
    # print(locations)
    estimated_distances = predict(locations, 
                                  'location_ranking')
    estimated_distances = estimated_distances.numpy().tolist()
    estimated_distances = [i[0] for i in estimated_distances]
    
    names_locations = []
    for i in range(len(names)):
        names_locations.append((names[i][0] + ' ' + names[i][1], 
                                estimated_distances[i]))
    # estimated_distance = []
    potential_caregivers = sorted(names_locations, 
                                  key=lambda x: x[1])
    return potential_caregivers