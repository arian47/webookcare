import os
import io
import pickle
import google.auth
import pathlib
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Define the scopes and the file paths
SCOPES = ['https://www.googleapis.com/auth/drive']
TOKEN_PATH = 'token.pickle'
CREDENTIALS_PATH = 'credentials.json'
saved_model_res_id = '1keHa5IpZT8SYQ1-hpAwnixaesK_YSUsV'
vocab_res_id = '1Mqa0Riy6AQvnaVoVPW14txpSi8XVDwcG'
labels_vocab_res_id = '1fLp0DllMAfecCsDYGIGhUpgmjnKrnE4b'

def authenticate():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            flow.redirect_uri = 'http://localhost:8080/'
            creds = flow.run_local_server(port=8080)
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)


def download_file(service, file_id, dest_path):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
            

def download_folder(service, folder_id, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    for item in items:
        file_id = item['id']
        file_name = item['name']
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # If the item is a folder, recursively download it
            download_folder(service, file_id, os.path.join(dest_path, file_name))
        else:
            # If the item is not a folder, download it
            download_file(service, file_id, os.path.join(dest_path, file_name))

 
