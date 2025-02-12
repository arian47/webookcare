import numpy
import csv
import os
import re
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

CAREGIVERS_CREDENTIALS_CSV_PATH = 'caregivers_qualifications.csv'
CAREGIVERS_CREDENTIALS_NPY_PATH = 'caregivers_qualifications.npy'
DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")

# # TODO: transferring save and retrieval of info to a different module.
# def save_credentials(caregivers_credentials):
#     tmp = [(j, i.get(j)) for i in caregivers_credentials for j in i]

#     with open(CAREGIVERS_CREDENTIALS_CSV_PATH, mode='w', newline='') as fo:
#         writer = csv.writer(fo)
#         writer.writerows(tmp)

#     numpy.save(CAREGIVERS_CREDENTIALS_NPY_PATH, 
#                numpy.array(caregivers_credentials, dtype=object))

# # TODO: transferring save and retrieval of info to a different module.
# def load_services():
#     pass

# checking the services offered by all the caregivers which could be time consuming.
# TODO: to break down for better filtering on needs
def check_credentials(caregiver_id:int=None,
                      save:bool=False):
    """
    Fetches and associates caregivers with their corresponding care 
    credentials and qualifications.

    This function queries the database for the list of care credentials, the caregivers associated 
    with each care credential, and the list of caregivers. It then associates each caregiver with
    the credentials they are responsible for. Optionally, the results can be saved to a persistent 
    storage (e.g., a file or another database).

    Parameters
    ----------
    save : bool, optional
        Whether to save the resulting caregiver-credentials associations. Default is False.
    
    Returns
    -------
    list of dict
        A list where each dictionary represents a caregiver and their associated credentials,
        in the form of `{caregiver_name: [credentials_name, ...]}`.
    """
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_NAME
        )
    cursor = connection.cursor()
    
    if not caregiver_id:
        command = f"""
        SELECT 
            caregivers.id, 
            caregivers.first_name,
            caregivers.last_name
        FROM caregivers;
        """
    else:
        command = f"""
        SELECT 
            caregivers.id, 
            caregivers.first_name,
            caregivers.last_name
        FROM caregivers
        WHERE caregivers.id = {caregiver_id};
        """
    cursor.execute(command)
    caregivers = cursor.fetchall()
    
    caregivers = [(i[0], i[1] + ' ' + i[2]) for i in caregivers]
    caregivers = dict(caregivers)
    
    keys = [
        'caregiver_id', 'owns_car', 'cpr_aed_certificate', 'first_aid_certificate', 'food_safe_certification', 
        'vaccination_records', 'medical_clearance', 'licensed_practical_nurse', 'registered_nurse', 
        'registered_care_aide', 'degree_certificate', 'repeat_clients', 'years_experience', 
        'status', 'verified', 'bc_care_registration_number', 
        ]
    
    tmp_txt = ''
    for i in keys:
        tmp_txt += f'qualifications.{i}, '
    tmp_txt = tmp_txt[:-2]

    query_txt = f'SELECT {tmp_txt} FROM qualifications;'
    
    cursor.execute(query_txt)
    qualifications = cursor.fetchall()
    val_len = len(qualifications[0])
    
    qualifications_data = []
    for i in qualifications:
        tmp = []
        for j in range(1, val_len):
            if i[j] in (0, 1):
                tmp.append(bool(i[j]))
            else:
                tmp.append(i[j])
        tmp = dict([(i, j) for i, j in zip(keys[1:], tmp)])
        name = caregivers.get(i[0])
        qualifications_data.append({name:tmp})
    
    # for each dict (user)
    tmp_dict = {}
    for i in qualifications_data:
        # for each user
        for j in i:
            tmp = []
            ntmp = []
            for x in i[j]:
                if i[j][x] == True:
                    tmp.append(x)
            if tmp:
                for i in tmp:
                    res = re.sub(r"_", "", i)
                    ntmp.append(res)
                tmp_dict[j] = ntmp
    
    if None in tmp_dict:
        tmp_dict.pop(None)
    return tmp_dict

                    
                
    
    
    
    