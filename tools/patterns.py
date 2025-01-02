patterns = dict(
    service_recommendation=dict(
        data_patterns={
            'client care plan 1' :
                r'Duties\s*to\s*Perform\s*Notes(.*?)Terms\s*of\s*Plan',
            'occupational therapy services 2' : 
                r'Notes:(.*?)Please\s*make\s*invoices\s*for\s*above\s*items\s*payable\s*to:',
            'comunity therapists 3' : [[
                r'Vanessa,(.*?)Laila',
                r'Medical\s*orders/Instructions:(.*?)PERSONAL\s*ATTENDANT\s*CARE',
                ]],
            'care plan 4' :
                r'Condition(.*?)Characteristics',
            'client care plan 5' :
                r'CONDITION(.*?)HEALTHCARE\s*PROFESSIONALS',
            'care plan 6' :
                r'General\s*Notes(.*?)Extra\s*Information',
            'home healthcare worker care plan 7' : 
                r'Comments:(.*?)Safety/Risk:',
            'care plan 8': 
                r'Condition(.*?)Characteristic'
            
        },
        label_patterns={
            'comunity therapists 3' : [
                [r'HOME\s*MAKING\s*SERVICES(.*?)ADDITIONAL\s*ACTIVITIES',
                 r'HOME\s*MAKING\s*SERVICES(.*?)Vanessa',
                 r'HOME\s*MAKING\s*SERVICES(.*?)Laila',
                 r'HOME\s*MAKING\s*SERVICES.*'],
                [r'PERSONAL\s*ATTENDANT\s*CARE(.*?)ADDITIONAL\s*ACTIVITIES',
                r'PERSONAL\s*ATTENDANT\s*CARE(.*?)Hi\s*Vanessa',
                r'PERSONAL\s*ATTENDANT\s*CARE(.*?)Terms\s*of\s*Plan',
                r'PERSONAL\s*ATTENDANT\s*CARE(.*)',] # to be thought about
            ],
            'care plan 4' :
                r'Client\s*Care\s*Plan(.*?)Authorization\s*Care\s*Plans'
            ,
            'client care plan 5': [[
                r'DUTIES\s*TO\s*PERFORM(.*?)Duties\s*to\s*Perform\s*Notes'
                r'DUTIES\s*TO\s*PERFORM(.*?)AUTHORIZATIONS',
            ]],
            'home healthcare worker care plan 7': [[ 
                r'✓(.*?)(prn|✓|2X/wk|1X/wk|Daily)',
                r'✓(.*?)ADL’s'
            ]],
            'care plan 8' : 
                r'.*Duties\s*to\s*Perform(.*?)Current\s*Authorizations.*',
            'occupational therapy services 2' : 
                r'.*Duties:(.*?)(Client’s information).*'
        }
    ),
    credentials_recommendation=dict(
        data_patterns=r"""(?i)\b(?:RCA|Care\s*Aide|Care\s*aide|Careaide|HCA)[-:\s]*([A-Z]\.\s*)*([A-Za-z]+\s*)+""",
        label_patterns=None),
    
)