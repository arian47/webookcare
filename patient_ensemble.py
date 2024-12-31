import re
import typer
import logging
import tensorflow
import webookcare.patients.credentials_recommendation.main
import webookcare.patients.service_recommendation.main
import webookcare.caregivers.careservices.retrieve_services_info
import webookcare.patients.service_recommendation

app = typer.Typer()

# numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def match_services(service):
    caregivers_services_dict = webookcare.caregivers.careservices.retrieve_services_info.check_services()
    # # test case
    # for i, j in caregivers_services_dict.items():
        # if i>0:
            # print('there is data!')
        # break
    # print(caregivers_services_dict)
    
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

def rank_caregivers(patient_id, patient_req):
    # potential_caregivers = []
    cred = webookcare.patients.credentials_recommendation.main.predict(
        patient_req
    )
    service = webookcare.patients.service_recommendation.main.predict(
        patient_req
    )
    potential_caregivers = match_services(service)
    # print(cred, service, sep='\n\n')
    return (cred, service, potential_caregivers)

# rank_caregivers('Hi I need care mom is in ICU she will need extensive care!')
