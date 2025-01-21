import os
import pathlib
import mysql.connector
from typing import List
from webookcare.tools.save_models import load_ml
from dotenv import load_dotenv

load_dotenv()

DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
HCW_REVIEWS_PATH = 'hcw_reviews.csv'
HCW_REVIEWS_NPY_PATH = 'hcw_reviews.npy'
DEFAULT_MODEL_PATH = 'svd_model.pkl'

current_dir = pathlib.Path(__file__).resolve().parent

HCW_REVIEWS_PATH = os.path.join(
    current_dir, 
    "hcw_reviews.csv"
    ).replace('\\', '/')
HCW_REVIEWS_NPY_PATH = os.path.join(
    current_dir, 
    "hcw_reviews.npy"
    ).replace('\\', '/')
DEFAULT_MODEL_PATH = os.path.join(
    current_dir, 
    "svd_model.pkl"
    ).replace('\\', '/')

def sort_reviews(patient_id:int,
                 caregivers_oi:List[str]):
    
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_NAME
        )
    cursor = connection.cursor()
    caregivers_reviews = []
    for i in caregivers_oi:
        command = f"""
        SELECT
            caregivers.first_name,
            caregivers.last_name, 
            caregivers.id
        FROM caregivers
        where caregivers.first_name='{i.split()[0]}'
        AND
        caregivers.last_name='{i.split()[1]}';
        """
        cursor.execute(command)
        res = cursor.fetchall().pop()
        caregivers_reviews.append(res)
    
    
    model = load_ml(DEFAULT_MODEL_PATH,
                    "pickle")
    reviews = [model.predict(patient_id, i[2]).est for i in caregivers_reviews]
    reviews = [(caregivers_oi[i], reviews[i]) for i in range(len(reviews))]
    reviews = sorted(reviews, 
                     key=lambda x: x[1])
    # prediction = model.predict(patient_id, 20)
    return reviews