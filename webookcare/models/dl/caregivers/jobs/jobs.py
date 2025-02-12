from dotenv import load_dotenv
import mysql.connector
import os
import json

load_dotenv()

DB_USERNAME = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")

# TODO: potentially not a good idea to return all the jobs list
def retrieve_jobs():
    jobs = []  # Initialize the jobs list in case of any error
    cursor = None  # Initialize cursor variable
    connection = None  # Initialize connection variable
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
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cursor:  # Only attempt to close cursor if it was created
            cursor.close()
        if connection:  # Only attempt to close connection if it was created
            connection.close()

    
    return jobs
        
