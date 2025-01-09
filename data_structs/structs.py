from pydantic import BaseModel, Field, validator, constr
from typing import Optional, List, Tuple
from datetime import datetime
from enum import Enum

class CareWorkerType(str, Enum):
    """
    Enumeration of care worker types, specifying the roles and certifications 
    commonly held by care workers.

    Attributes:
        RPN: Registered Practical Nurse
        RN: Registered Nurse
        LPN: Licensed Practical Nurse
        CNA: Certified Nursing Assistant
        HHA: Home Health Aide
        PT: Physical Therapist
        OT: Occupational Therapist
        LP: Licensed Practitioner
        PCA: Personal Care Assistant
    """
    RPN = "Registered Practical Nurse" 
    RN = "registered_nurse"
    LPN = "licensed_practical_nurse"
    CNA = "certified_nursing_assistant"
    HHA = "home_health_aide"
    PT = "physical_therapist"
    OT = "occupational_therapist"
    LP = "licensed_practitioner"
    PCA = "personal_care_assistant"


# Enum for Qualifications, Certifications, and Trainings
class Qualification(str, Enum):
    """
    Enumeration of qualifications, certifications, and trainings relevant to care workers.

    This class defines a set of qualifications commonly required or valued for roles in caregiving, 
    childcare, healthcare, and related fields. Each qualification is represented as a string value.

    Attributes:
        FIRST_AID_LEVEL_ONE: Certification for basic first aid.
        CPR_AED: Certification for cardiopulmonary resuscitation and automated external defibrillator use.
        FOOD_SAFE: Certification for safe food handling practices.
        MEDICATION_ADMINISTRATION_COURSE: Training in administering medication safely.
        FOOT_CARE: Certification or training in foot care.
        EARLY_CHILDHOOD_CERTIFICATE: Certificate in early childhood education.
        EARLY_CHILDHOOD_EDUCATION_DIPLOMA: Diploma in early childhood education.
        BC_ECE_LICENSE: British Columbia Early Childhood Educator License.
        INFANT_TODDLER_CERTIFICATE: Certification focused on infant and toddler care.
        RESPONSIBLE_ADULT_CERTIFICATION: Certification for working responsibly with children.
        BABYSITTER_COURSE: Training for babysitting.
        SACC_CERTIFICATION: School Age Child Care certification.
        CRITICAL_INCIDENT_MANAGEMENT: Training in managing critical incidents.
        NON_VIOLENT_CRISIS_INTERVENTION_TRAINING: Certification for non-violent crisis intervention.
        CONFLICT_RESOLUTION_TRAINING: Training in conflict resolution.
        SIVA_TRAINING: Supporting Individuals through Valued Attachments training.
        MANDT_CERTIFICATION: Certification in The Mandt System (behavioral crisis prevention and intervention).
        MINISTRY_OF_CHILDREN_AND_FAMILY_DEVELOPMENT_CLEARANCE: Clearance from the Ministry of Children and Family Development.
        ANTI_BULLYING_AND_HARASSMENT: Training in anti-bullying and harassment practices.
        SAFETALK: SafeTalk suicide prevention training.
        MENTAL_HEALTH_FIRST_AID: Certification in mental health first aid.
        WHMIS: Workplace Hazardous Materials Information Systems (WHMIS) training.
        PALLIATIVE_CARE_CERTIFICATE: Certification in palliative care.
        DEMENTIA_CARE_CERTIFICATE: Certification in dementia care.
        SAFETY_AND_MOBILITY_CERTIFICATE: Certification in safety and mobility practices.
        HCA_ACUTE_CARE: Health Care Assistant certification for acute care.
        OPEN_HEART_CARDIO_VASCULAR_INTENSIVE_CARE_WITH_BALLOON_PUMP: Certification in specialized cardiac care.
        NEONATAL_RESUSCITATION_PROGRAM: Neonatal resuscitation program certification.
        PEDIATRIC_ADVANCED_LIFE_SUPPORT: Pediatric advanced life support certification.
        EMERGENCY_NURSING_PEDIATRIC_COURSE: Training in emergency pediatric nursing.
        ADVANCED_CARDIAC_LIFE_SUPPORT: Certification in advanced cardiac life support.
        TRAUMA_NURSING_CORE_COURSE: Trauma nursing core course certification.
        ADVANCED_TRAUMA_LIFE_SUPPORT: Certification in advanced trauma life support.
        CANADIAN_TRIAGE_AND_ACUITY_SCORE: Training in Canadian triage and acuity scoring.
        PEDIATRIC_EARLY_WARNING_SYSTEM: Certification in the pediatric early warning system.
        ACQUIRED_BRAIN_INJURY_TRAINING: Training in managing acquired brain injuries.
        DOT_MEDICATION_TRAINING: Training for Directly Observed Therapy (DOT) medication administration.
        DIVERSITY_COMPETENCY: Certification or training in diversity and inclusion practices.
        SAFE_CLIENT_HANDLING: Certification in safe handling of clients.
        VIOLENCE_PREVENTION: Training in violence prevention strategies.
        INFECTION_PREVENTION_AND_CONTROL: Certification in infection prevention and control.
        DEMENTIA_CARE: Training in dementia care practices.
        MENTAL_HEALTH_RESPONSE_TRAINING: Certification in mental health response.
        VAC_WOUND_CARE: Certification in vacuum-assisted closure (VAC) wound care.
        RECREATION_DIPLOMA_PROGRAM: Diploma in recreation programming.
        PAYROLL_COMPLIANCE_PROFESSIONAL_CERTIFICATION: Certification as a Payroll Compliance Professional (PCP).
        GOOGLE_WORKSPACE_MICROSOFT_OFFICE: Knowledge and skills in Google Workspace and Microsoft Office.
        SCHEDULING_SOFTWARE: Proficiency in scheduling software.
        CPA_LEVEL_1: Certification in CPA (Chartered Professional Accountant) Level 1.
        CLASS_5_DRIVERS_LICENCE: Possession of a Class 5 driver's licence.
        CLASS_4_DRIVERS_LICENCE: Possession of a Class 4 driver's licence.
        OWNS_A_CAR: Indicates ownership of a personal vehicle.
    """
    FIRST_AID_LEVEL_ONE = "First Aid Level One"
    CPR_AED = "CPR/AED"
    FOOD_SAFE = "Food Safe"
    MEDICATION_ADMINISTRATION_COURSE = "Medication Administration Course"
    FOOT_CARE = "Foot Care"
    EARLY_CHILDHOOD_CERTIFICATE = "Early Childhood Certificate"
    EARLY_CHILDHOOD_EDUCATION_DIPLOMA = "Early Childhood Education Diploma"
    BC_ECE_LICENSE = "BC ECE License"
    INFANT_TODDLER_CERTIFICATE = "Infant Toddler Certificate"
    RESPONSIBLE_ADULT_CERTIFICATION = "Responsible Adult Certification"
    BABYSITTER_COURSE = "Babysitter Course"
    SACC_CERTIFICATION = "SACC Certification"
    CRITICAL_INCIDENT_MANAGEMENT = "Critical Incident Management"
    NON_VIOLENT_CRISIS_INTERVENTION_TRAINING = "Non-Violent Crisis Intervention Training"
    CONFLICT_RESOLUTION_TRAINING = "Conflict Resolution Training"
    SIVA_TRAINING = "SIVA Training"
    MANDT_CERTIFICATION = "MANDT Certification"
    MINISTRY_OF_CHILDREN_AND_FAMILY_DEVELOPMENT_CLEARANCE = "Ministry of Children and Family Development Clearance"
    ANTI_BULLYING_AND_HARASSMENT = "Anti-Bullying and Harassment"
    SAFETALK = "SafeTalk"
    MENTAL_HEALTH_FIRST_AID = "Mental Health First Aid"
    WHMIS = "Workplace Hazardous Materials Information Systems (WHMIS)"
    PALLIATIVE_CARE_CERTIFICATE = "Palliative Care Certificate"
    DEMENTIA_CARE_CERTIFICATE = "Dementia Care Certificate"
    SAFETY_AND_MOBILITY_CERTIFICATE = "Safety and Mobility Certificate"
    HCA_ACUTE_CARE = "HCA Acute Care"
    OPEN_HEART_CARDIO_VASCULAR_INTENSIVE_CARE_WITH_BALLOON_PUMP = "Open Heart/Cardio Vascular Intensive Care with Balloon Pump Certification"
    NEONATAL_RESUSCITATION_PROGRAM = "Neonatal Resuscitation Program"
    PEDIATRIC_ADVANCED_LIFE_SUPPORT = "Pediatric Advanced Life Support"
    EMERGENCY_NURSING_PEDIATRIC_COURSE = "Emergency Nursing Pediatric Course"
    ADVANCED_CARDIAC_LIFE_SUPPORT = "Advanced Cardiac Life Support"
    TRAUMA_NURSING_CORE_COURSE = "Trauma Nursing Core Course"
    ADVANCED_TRAUMA_LIFE_SUPPORT = "Advanced Trauma Life Support"
    CANADIAN_TRIAGE_AND_ACUITY_SCORE = "Canadian Triage and Acuity Score"
    PEDIATRIC_EARLY_WARNING_SYSTEM = "Pediatric Early Warning System"
    ACQUIRED_BRAIN_INJURY_TRAINING = "Acquired Brain Injury Training"
    DOT_MEDICATION_TRAINING = "DOT Medication Training"
    DIVERSITY_COMPETENCY = "Diversity Competency"
    SAFE_CLIENT_HANDLING = "Safe Client Handling"
    VIOLENCE_PREVENTION = "Violence Prevention"
    INFECTION_PREVENTION_AND_CONTROL = "Infection Prevention and Control"
    DEMENTIA_CARE = "Dementia Care"
    MENTAL_HEALTH_RESPONSE_TRAINING = "Mental Health Response Training"
    VAC_WOUND_CARE = "VAC Wound Care"
    RECREATION_DIPLOMA_PROGRAM = "Recreation Diploma Program"
    PAYROLL_COMPLIANCE_PROFESSIONAL_CERTIFICATION = "Payroll Compliance Professional (PCP) Certification"
    GOOGLE_WORKSPACE_MICROSOFT_OFFICE = "Google Workspace & Microsoft Office Knowledge & Skills"
    SCHEDULING_SOFTWARE = "Scheduling Software"
    CPA_LEVEL_1 = "CPA Level 1"
    CLASS_5_DRIVERS_LICENCE = "Class 5 Driver's Licence"
    CLASS_4_DRIVERS_LICENCE = "Class 4 Driver's Licence"
    OWNS_A_CAR = "Owns a car"

class HealthCareSkills(str, Enum):
    """
    Enumeration of healthcare-related skills, certifications, and responsibilities.
    
    Attributes:
        CRITICAL_INCIDENT_MANAGEMENT: Managing critical incidents in healthcare settings.
        COMPUTER_ASSISTANCE: Providing computer assistance and technical support.
        CASE_MANAGEMENT: Coordinating care and services for patients.
        CARE_PLAN_DEVELOPMENT: Developing care plans for patients.
        MEDICATION_INJECTION: Administering injectable medications.
        NARCOTIC_ADMINISTRATION: Administering narcotic medications.
        SUB_CUE_MEDICATIONS: Handling subcutaneous medications, including patches.
        ORAL_MEDICATION_ADMINISTRATION: Administering oral medications.
        BOWEL_ROUTINES: Managing bowel routines, including suppositories and enemas.
        CATHETERISATION: Performing catheterisation procedures.
        STOCKINGS: Assisting with compression stockings.
        CONFLICT_RESOLUTION: Resolving conflicts in healthcare settings.
        SCHEDULE_CREATION: Creating and managing schedules.
        CRISIS_MANAGEMENT: Handling crises effectively.
        PROGRAM_DEVELOPMENT: Developing healthcare programs.
        SAFE_CLIENT_HANDLING: Ensuring the safe handling of clients.
        INFECTION_CONTROL: Implementing infection prevention and control measures.
        VIOLENCE_PREVENTION: Addressing and preventing violence in healthcare.
        CHARTING: Maintaining accurate patient records and charts.
        DEMENTIA_CARE: Providing specialized care for dementia patients.
        CHILD_SUPERVISION: Supervising children in various care settings.
        EARLY_CHILDHOOD_EDUCATION: Educating young children with a focus on early learning.
        PHYSICAL_DISABILITIES: Supporting individuals with physical disabilities.
        NUTRITION: Providing nutritional guidance and care.
        INFANT_CARE: Caring for infants, including feeding and hygiene.
        SOCIALIZATION_PLAY_BASED_LEARNING: Encouraging socialization and play-based learning.
        SAFETY_PLANNING: Developing safety plans for clients.
        EMOTIONAL_SUPPORT: Offering emotional support to clients.
        ADVOCACY: Advocating for client needs and rights.
        BEHAVIOR_MANAGEMENT: Managing and intervening in behavioral issues.
        CHILD_ABUSE_PREVENTION: Preventing and responding to child abuse and neglect.
        ENVIRONMENTAL_HEALTH_SAFETY: Maintaining health and safety in the environment.
        LIFE_SKILLS: Teaching life skills to clients.
        SOCIAL_SUPPORT: Providing social support to individuals.
        COMMUNICATION_STRATEGIES: Employing effective communication strategies.
        CURRICULUM_PLANNING: Planning educational curricula.
        EARLY_CHILDHOOD_LITERACY: Promoting literacy in early childhood.
        LANGUAGE_DEVELOPMENT: Supporting language development in children.
        CPR: Administering cardiopulmonary resuscitation (CPR).
        ADVANCED_CERTIFICATIONS: Holding advanced certifications like ACLS, PALS, and TNCC.
    """
    CRITICAL_INCIDENT_MANAGEMENT = "Critical Incident Management"
    COMPUTER_ASSISTANCE = "Computer Assistance"
    CASE_MANAGEMENT = "Case Management"
    CARE_PLAN_DEVELOPMENT = "Developing Care Plans"
    MEDICATION_INJECTION = "Medication Injection"
    NARCOTIC_ADMINISTRATION = "Narcotic Administration"
    SUB_CUE_MEDICATIONS = "Sub Cue Medications (patches)"
    ORAL_MEDICATION_ADMINISTRATION = "Administration of Oral Medications"
    BOWEL_ROUTINES = "Bowel Routines"
    SUPPOSITORIES = "Suppositories"
    ENEMAS = "Enemas"
    CATHETERISATION = "Catheterisation"
    STOCKINGS = "Compression Stockings"
    CONFLICT_RESOLUTION = "Conflict Resolution"
    SCHEDULE_CREATION = "Create Schedules"
    CRISIS_MANAGEMENT = "Crisis Management"
    PROGRAM_DEVELOPMENT = "Develop Programs"
    OPEN_HEART_CARE = "Operating Room – Open Heart/Cardio Vascular Intensive Care with Balloon Pump Certification"
    NRP = "NRP- Neonatal Resuscitation Program"
    PALS = "PALS- Pediatric Advanced Life Support"
    ENPC = "ENPC- Emergency Nursing Pediatric Course"
    ACLS = "ACLS – Advanced Cardiac Life Support"
    TNCC = "TNCC- Trauma Nursing Core Course"
    ATLS = "ATLS- Advanced Trauma Life Support"
    BLS = "BLS – Basic Life Support"
    CTAS = "CTAS – Canadian Triage and Acuity Score"
    PEWS = "PEWS- Pediatric Early Warning System"
    STRANGER_IN_CRISIS = "Stranger in Crisis"
    CPR = "CPR-Cardiopulmonary Resuscitation"
    CRISIS_INTERVENTION = "Crisis Intervention"
    ACQUIRED_BRAIN_INJURY = "Acquired Brain Injury Training"
    DOT_MEDICATION = "DOT Medication Training"
    DOSE_INHALERS = "Dose Inhalers"
    INSULIN = "Insulin"
    EYE_CARE = "Eye Care"
    MEDICATED_PATCHES = "Medicated Patches, Creams, and Ointment Administration"
    BOWEL_CARE = "Bowel Care"
    DIVERSITY_COMPETENCY = "Diversity Competency"
    SAFE_CLIENT_HANDLING = "Safe Client Handling"
    CHARTING = "Charting"
    VIOLENCE_PREVENTION = "Violence Prevention"
    INFECTION_CONTROL = "Infection Prevention and Control"
    DEMENTIA_CARE = "Dementia Care"
    MENTAL_HEALTH_RESPONSE = "Mental Health Response Training"
    EDUCATOR = "Educator"
    CHILD_SUPERVISION = "Child Supervision"
    DAILY_CHILD_CARE = "Daily Child Care"
    CHILD_PROGRESS_DISCUSSION = "Discussion of Child's Progress"
    PARENT_OBSERVATIONS = "Discussion with Parents About Observations"
    ECA_DUTIES = "ECA Duties"
    GUIDANCE_DRESSING = "Guidance with Dressing"
    GUIDANCE_EATING = "Guidance with Eating"
    LEADING_ACTIVITIES = "Leading Activities"
    DAYCARE_MAINTENANCE = "Maintaining the Daycare"
    MENTORING = "Mentoring"
    OBSERVATION_TRACKING = "Observation Tracking"
    PREPARE_CRAFT_MATERIALS = "Prepare Craft Materials"
    CHILD_PROGRESS_TRACKING = "Progress Tracking of Children"
    RELATIONSHIP_BUILDING = "Relationship Building"
    SING_SONGS = "Sing Songs"
    STORYTELLING = "Storytelling"
    SUPERVISE_ECA = "Supervise an ECA"
    TEACH_SONGS = "Teach Songs"
    EARLY_CHILDHOOD_EDUCATION = "Early Childhood Education"
    PHYSICAL_DISABILITIES = "Physical Disabilities"
    CHILD_ABUSE_PREVENTION = "Child Abuse and Neglect Prevention"
    BEHAVIOR_MANAGEMENT = "Behavior Management"
    ENVIRONMENTAL_HEALTH_SAFETY = "Environmental Health and Safety"
    CHILD_DEVELOPMENT = "Child Development"
    CURRICULUM_PLANNING = "Curriculum Planning"
    CLASSROOM_MANAGEMENT = "Classroom Management"
    CHILD_PSYCHOLOGY = "Child Psychology"
    CHILD_GROWTH_DEVELOPMENT = "Child Growth and Development"
    EARLY_CHILDHOOD_LITERACY = "Early Childhood Literacy"
    SENSORY_INTEGRATION = "Sensory Integration"
    COMMUNICATION_STRATEGIES = "Communication Strategies"
    NUTRITION = "Nutrition"
    INFANT_CARE = "Infant Care"
    LANGUAGE_DEVELOPMENT = "Language Development"
    SOCIALIZATION_PLAY_BASED_LEARNING = "Socialization and Play-Based Learning"
    RECREATION_ASSISTANT = "Recreation Assistant"
    CONFLICT_RESOLUTION_SKILLS = "Conflict Resolution"
    CRISIS_PREVENTION = "Crisis Prevention"
    EMERGENCY_RESPONSE = "Emergency Response"
    SAFETY_PLANNING = "Safety Planning"
    GOAL_SETTING = "Goal Setting"
    CARE_PLANS = "Care Plans"
    EMOTIONAL_SUPPORT = "Emotional Support"
    BEHAVIORAL_SUPPORT = "Behavioral Support"
    BEHAVIORAL_INTERVENTION = "Behavioral Intervention"
    ANGER_MANAGEMENT = "Anger Management"
    LIFE_SKILLS = "Life Skills"
    SOCIAL_SUPPORT = "Social Support"
    LIAISON = "Liaison"
    HARM_REDUCTION = "Harm Reduction"
    ADVOCACY = "Advocacy"
    TEACHING = "Teaching"
    PROBLEM_SOLVING = "Problem-Solving"
    SUPERVISION = "Supervision"
    ADMINISTRATION_DUTIES = "Administration Duties"
    SCHEDULING = "Scheduling"
    SAFETY_SECURITY = "Safety and Security"


class HealthExpertise(str, Enum):
    ADDICTIONS = "Addictions"
    ALS = "ALS"
    ALZHEIMERS_DISEASE = "Alzheimer's Disease"
    ARTHRITIS = "Arthritis"
    AUTISM_SPECTRUM_DISORDER = "Autism Spectrum Disorder (ASD)"
    ADHD = "Attention Deficit Hyperactivity Disorder (ADHD)"
    BLOOD_DISORDERS = "Blood Disorders"
    BRAIN_INJURIES = "Brain Injuries"
    CANCER_RECOVERY = "Cancer Recovery"
    CARDIOVASCULAR_DISORDERS = "Cardiovascular Disorders"
    CEREBRAL_PALSY = "Cerebral Palsy"
    COPD = "COPD"
    DEMENTIA = "Dementia"
    DEPRESSION = "Depression"
    DEVELOPMENTAL_DISABILITIES = "Developmental Disabilities"
    DIABETES = "Diabetes"
    GASTROINTESTINAL_DISORDERS = "Gastrointestinal Disorders"
    HEARING_IMPAIRMENT = "Hearing Impairment"
    HIV_AIDS = "HIV/AIDS"
    HOSPICE_CARE = "Hospice Care"
    MEMORY_CARE = "Memory Care"
    MULTIPLE_SCLEROSIS = "Multiple Sclerosis"
    NEUROLOGICAL_DISORDERS = "Neurological Disorders"
    ORTHOPEDIC_CARE = "Orthopedic Care"
    PARKINSONS_DISEASE = "Parkinson's Disease"
    PALLIATIVE_CARE = "Palliative Care"
    PEDIATRIC_CARE = "Pediatric Care"
    POST_SURGERY_RECOVERY = "Post Surgery Recovery"
    RENAL_UROLOGICAL_DISORDERS = "Renal and Urological Disorders"
    RESPIRATORY_DISORDERS = "Respiratory Disorders"
    SKIN_DISORDERS = "Skin Disorders"
    STROKE = "Stroke"
    SPINAL_CORD_INJURIES = "Spinal Cord Injuries"
    TRACHEOTOMY_VENTILATION = "Tracheotomy/Ventilation"
    VISION_EYE_DISORDERS = "Vision and Eye Disorders"




# Enum for Care Services
class CareServices(str, Enum):
    """
    Enumeration of care services that can be provided by a care worker.

    Attributes:
        MEDICATION_REMINDERS: Assistance with medication schedules.
        MEALS: Preparation and serving of meals.
        DRESSING: Assistance with dressing.
        GROOMING: Assistance with grooming tasks.
        PERSONAL_CARE: General personal care tasks.
        COMPANIONSHIP: Providing social interaction and companionship.
        HOUSEKEEPING: General housekeeping duties, such as cleaning and tidying.
        TRANSPORTATION: Assisting with transportation needs.
        EXERCISE: Supporting physical exercise and mobility activities.
        PET_CARE: Assistance with pet-related responsibilities.
        SHOPPING: Helping with shopping and errands.
        BATHING_SHOWERING: Assistance with bathing or showering.
        BED_BATH: Providing bed baths for individuals with limited mobility.
        BLOOD_GLUCOSE_CHECKING: Monitoring blood glucose levels.
        FEEDING: Assistance with eating and feeding.
        BLOOD_PRESSURE: Monitoring and recording blood pressure.
        CUEING_COACHING: Cueing and coaching for tasks or routines.
        CATHETER_CARE: Assistance with catheter maintenance and care.
        OSTOMY_CARE: Ostomy care and support.
        OXYGEN_USE: Assistance with oxygen equipment and therapy.
        SHAVE: Shaving assistance.
        SHAMPOO: Hair washing and shampooing.
        SKINCARE: General skincare assistance.
        NAIL_CARE: Nail care and hygiene.
        ORAL_CARE: Assistance with oral hygiene.
        HAIR_CARE: Assistance with hair care and styling.
        EYE_CARE: Providing care for eyes and related needs.
        TRANSFERRING: Assistance with transferring between positions or locations.
        MOBILITY_ASSISTANCE: Supporting mobility and preventing falls.
        REPOSITIONING: Repositioning individuals for comfort or health reasons.
        CHARTING: Documenting and charting care activities.
        ELIMINATION: Assistance with elimination and toileting needs.
        PERINEAL_CARE: Perineal hygiene and care.
        PALLIATIVE_CARE: End-of-life care and support.
        STOCKINGS: Assisting with compression stockings.
        TOILETING: Providing toileting assistance.
        RESPITE: Respite care for caregivers.
        READY_FOR_BED: Assistance with bedtime routines.
        LIFTS_FLOOR_AND_OVERHEAD_CEILING: Using floor and ceiling lifts for transfers.
        BASIC_WOUND_CARE: Basic wound care and dressing changes.
        PAIN_MANAGEMENT: Pain management and relief support.
        EDUCATION: Providing education related to care.
        DELEGATION: Assisting with delegation of tasks.
        COMPUTER_ASSISTANCE: Help with computer and technology-related tasks.
        CASE_MANAGEMENT: Managing care cases and coordination.
        DEVELOP_CARE_PLANS: Developing personalized care plans.
        SPECIMEN_COLLECTION: Assisting with collection of medical specimens.
        MEDICATION_INJECTION: Administering medication through injections.
        NARCOTIC_ADMINISTRATION: Administering narcotic medications.
        SUB_CUE_MEDICATIONS: Applying subcutaneous medications like patches.
        MEDICATION_ADMINISTRATION: Administering prescribed medications.
        IV_THERAPY: Assisting with intravenous therapy.
        BOWEL_ROUTINES: Supporting bowel care routines.
        SUPPOSITORIES: Administering suppositories.
        ENEMAS: Administering enemas.
        CATHETERISATION: Assisting with catheter insertion and care.
        BOWEL_CARE: Comprehensive bowel care support.
        CLIENT_ASSESSMENT_AND_MONITORING: Monitoring and assessing clients' health.
        SUPERVISION: Providing oversight and supervision.
        DELEGATION_TRAINING: Training for task delegation.
        LEARNING_ACTIVITIES: Organizing and facilitating learning activities.
        RECREATIONAL_ACTIVITIES: Planning recreational and engaging activities.
        INFANT_DIAPERING: Diapering infants.
        AGE_APPROPRIATE_PLAY: Organizing age-appropriate play activities.
        STIMULATING_ACTIVITIES: Planning stimulating and educational activities.
        SOCIALISATION: Encouraging social interaction and engagement.
        NAP_TIME_ROUTINE: Supporting nap time routines for children.
        DISCUSSION_OF_CHILD_PROGRESS: Discussing progress and observations about children.
        DISCUSSION_WITH_PARENTS: Communicating with parents regarding child care.
        ECA_DUTIES: Early childhood assistant (ECA) responsibilities.
        LEADING_ACTIVITIES: Leading group activities and programs.
        MAINTAINING_DAYCARE: Maintaining a safe and clean daycare environment.
        MENTORING: Providing mentoring and guidance.
        OBSERVATION_AND_ASSESSMENT: Observing and assessing individual needs.
        PREPARE_CRAFT_MATERIALS: Preparing materials for arts and crafts.
        PROGRESS_TRACKING_CHILDREN: Tracking developmental progress of children.
        RELATIONSHIP_BUILDING: Fostering positive relationships.
        SING_SONGS: Singing songs with children or clients.
        STORYTELLING: Telling stories to engage and entertain.
        SUPERVISE_ECA: Supervising early childhood assistants.
        TEACH_SONGS: Teaching songs and music activities.
        PHYSICAL_DISABILITIES: Assistance for individuals with physical disabilities.
        CURRICULUM_PLANS: Designing curriculum plans for educational or care settings.
        ACTIVITY_PLANS: Creating activity plans for engagement and development.
        BEHAVIOUR_INTERVENTION_PLANS: Developing strategies for managing and improving behavior.
        APPLIED_BEHAVIOUR_ANALYSIS: Using ABA techniques for behavioral interventions.
        CLASSROOM_MANAGEMENT: Overseeing and managing classroom environments.
        PROGRAM_DEVELOPMENT: Designing and implementing developmental programs.
        CHILD_GROWTH_AND_DEVELOPMENT: Focusing on child growth and developmental milestones.
        EARLY_CHILDHOOD_LITERACY: Promoting literacy skills in early childhood.
        SENSORY_INTEGRATION: Activities to support sensory processing and integration.
        COMMUNICATION_STRATEGIES: Developing and implementing communication strategies.
        INFANT_CARE: Providing care and activities specific to infants.
        CHILD_DEVELOPMENT: Supporting physical, emotional, and cognitive child development.
        LANGUAGE_DEVELOPMENT: Encouraging and facilitating language acquisition and skills.
        SOCIALIZATION: Activities to promote interaction and social skills.
        PLAY_BASED_LEARNING: Using play as a method for teaching and learning.
        SAFETY_PLANNING: Creating and implementing safety plans.
        GOAL_SETTING: Assisting individuals with setting and achieving goals.
        EMOTIONAL_SUPPORT: Providing emotional support and reassurance.
        BEHAVIORAL_SUPPORT: Assisting with managing and improving behaviors.
        BEHAVIORAL_INTERVENTION: Implementing strategies for behavioral improvements.
        ANGER_MANAGEMENT: Providing strategies and support for managing anger.
        LIFE_SKILLS: Teaching and supporting daily living and life skills.
        SOCIAL_SUPPORT: Offering support to enhance social connections and interactions.
        LIAISON: Acting as an intermediary for communication and coordination.
        HARM_REDUCTION: Strategies for reducing harm and promoting safety.
        ADVOCACY: Supporting and advocating for individuals’ needs and rights.
        TEACHING: Providing educational lessons or guidance.
        PROBLEM_SOLVING: Assisting in identifying and resolving issues.
        RECEIVABLES: Managing payments or debts owed.
        GREETING: Welcoming individuals warmly and courteously.
        SCREENING: Conducting checks or evaluations.
        PORTER: Assisting with carrying or moving items.
        MEAL_PLANNING: Organizing meal menus and plans.
        CARE_COORDINATION: Managing care plans and activities for individuals.
        BARTENDING: Preparing and serving beverages.
        BUS_TABLES: Clearing and resetting tables.
        CLEANING_KITCHEN_EQUIPMENT: Cleaning and maintaining kitchen appliances.
        SANITISING: Ensuring areas and items are sanitized and clean.
        DINING_ROOM_SETUP: Arranging dining areas for service.
        DISHWASHING: Cleaning dishes and utensils.
        INDUSTRIAL_EQUIPMENT: Operating and maintaining large-scale equipment.
        FULFILLING_CUSTOMER_REQUESTS: Meeting customer needs and requests.
        WIPING: Cleaning surfaces with a cloth or sponge.
        VACUUMING: Cleaning floors with a vacuum.
        MOPPING: Cleaning floors with a mop.
        CLEAN_KITCHEN: Tidying and cleaning kitchen areas.
        CLEAN_BATHROOM: Tidying and cleaning bathroom areas.
        DISHES: Washing and cleaning dishes.
        LOAD_UNLOAD_DISHWASHER: Managing dishwashing machine usage.
        BEDDING_CHANGE: Replacing and organizing bed linens.
        GARBAGE_RECYCLING: Managing waste and recyclables.
        GENERAL_HOUSEKEEPING: Providing general cleaning and tidying services.
        LAUNDRY_DUTIES: Washing, drying, and folding laundry.
        LIGHT_TIDYING: Minor cleaning and organizing tasks.
        ORGANISING: Arranging items or spaces efficiently.
        EQUIPMENT_MAINTENANCE: Performing upkeep on tools and devices.
        SUPPLY_REPLENISH: Restocking needed supplies.
        DUSTING: Removing dust from surfaces.
        WINDOWS: Cleaning window panes.
        DEEP_CLEAN: Intensive cleaning of areas or items.
        OVEN: Cleaning ovens thoroughly.
        FRIDGE: Cleaning and organizing refrigerators.
        SPOILED_FOOD_REMOVAL: Disposing of expired or spoiled food items.
        UTENSIL_MAINTENANCE: Cleaning and maintaining utensils.
        SERVE_LARGE_GROUPS: Catering to large groups during meals or events.
        SET_TABLES: Preparing table settings.
        TAKE_ORDERS: Recording customer orders.
        SERVE: Delivering food and beverages to customers.
        ADMINISTRATION_DUTIES: Handling clerical and administrative tasks.
        SCHEDULING: Planning and organizing schedules.
        WELCOMING: Greeting and making individuals feel at ease.
        CUSTOMER_SERVICE: Assisting customers with inquiries or issues.
        CHECK_IN: Managing check-ins for individuals or groups.
        VISITOR_LOG: Maintaining a record of visitors.
        ENFORCE_SAFETY_PROTOCOLS: Ensuring compliance with safety measures.
        COVID_SCREENING: Conducting health checks related to COVID-19.
        FOOD_PREPARATION: Preparing ingredients and meals.
        ATTENDANCE_TRACKING: Monitoring and recording attendance.
        HANDLE_EMERGENCY_COVERAGE: Managing unforeseen staffing needs.
        COMPLIANCE_AND_REGULATIONS: Ensuring adherence to policies and regulations.
        PROCESS_PAYROLL: Managing employee payroll.
        DATA_ENTRY_AND_RECORD_KEEPING: Recording and maintaining data records.
        BENEFITS: Administering employee benefits.
        PAYROLL_REPORTING: Creating and maintaining payroll reports.
        MEAL_DELIVERY: Transporting meals to customers or locations.
        ASSISTANCE_WITH_MEAL_CHOICES: Helping individuals select appropriate meals.
        FOOD_SAFETY: Ensuring compliance with food safety standards.
        ORDERING: Managing food or supply orders.
        MENU_PLANNING: Creating meal menus.
        QUALITY_CONTROL: Ensuring products or services meet standards.
        DIETARY_GUIDELINE_ADHERENCE: Following nutritional guidelines in planning.
        ORDER_PICKUP: Collecting orders for customers.
        PAYMENT_HANDLING: Managing financial transactions.
    """
    MEDICATION_REMINDERS = "Medication Reminders"
    MEALS = "Meals"
    DRESSING = "Dressing"
    GROOMING = "Grooming"
    PERSONAL_CARE = "Personal Care"
    COMPANIONSHIP = "Companionship"
    HOUSEKEEPING = "Housekeeping"
    TRANSPORTATION = "Transportation"
    EXERCISE = "Exercise"
    PET_CARE = "Pet Care"
    SHOPPING = "Shopping"
    BATHING_SHOWERING = "Bathing/Showering"
    BED_BATH = "Bed Bath"
    BLOOD_GLUCOSE_CHECKING = "Blood Glucose Checking"
    FEEDING = "Feeding"
    BLOOD_PRESSURE = "Blood Pressure"
    CUEING_COACHING = "Cueing/Coaching"
    CATHETER_CARE = "Catheter Care"
    OSTOMY_CARE = "Ostomy Care"
    OXYGEN_USE = "Oxygen Use"
    SHAVE = "Shave"
    SHAMPOO = "Shampoo"
    SKINCARE = "Skincare"
    NAIL_CARE = "Nail Care"
    ORAL_CARE = "Oral Care"
    HAIR_CARE = "Hair Care"
    EYE_CARE = "Eye Care"
    TRANSFERRING = "Transferring"
    MOBILITY_ASSISTANCE = "Mobility Assistance"
    REPOSITIONING = "Repositioning"
    CHARTING = "Charting"
    ELIMINATION = "Elimination"
    PERINEAL_CARE = "Perineal Care"
    PALLIATIVE_CARE = "Palliative Care"
    STOCKINGS = "Stockings"
    TOILETING = "Toileting"
    RESPITE = "Respite"
    READY_FOR_BED = "Ready for Bed"
    LIFTS_FLOOR_AND_OVERHEAD_CEILING = "Lifts: Floor and Overhead Ceiling"
    BASIC_WOUND_CARE = "Basic Wound Care"
    PAIN_MANAGEMENT = "Pain Management"
    EDUCATION = "Education"
    DELEGATION = "Delegation"
    COMPUTER_ASSISTANCE = "Computer Assistance"
    CASE_MANAGEMENT = "Case Management"
    DEVELOP_CARE_PLANS = "Develop Care Plans"
    SPECIMEN_COLLECTION = "Specimen Collection"
    MEDICATION_INJECTION = "Medication Injection"
    NARCOTIC_ADMINISTRATION = "Narcotic Administration"
    SUB_CUE_MEDICATIONS = "Sub Cue Medications (Patches)"
    MEDICATION_ADMINISTRATION = "Medication Administration"
    IV_THERAPY = "IV Therapy"
    BOWEL_ROUTINES = "Bowel Routines"
    SUPPOSITORIES = "Suppositories"
    ENEMAS = "Enemas"
    CATHETERISATION = "Catheterisation"
    BOWEL_CARE = "Bowel Care"
    CLIENT_ASSESSMENT_AND_MONITORING = "Client Assessment and Monitoring"
    SUPERVISION = "Supervision"
    DELEGATION_TRAINING = "Delegation Training"
    LEARNING_ACTIVITIES = "Learning Activities"
    RECREATIONAL_ACTIVITIES = "Recreational Activities"
    INFANT_DIAPERING = "Infant Diapering"
    AGE_APPROPRIATE_PLAY = "Age-Appropriate Play"
    STIMULATING_ACTIVITIES = "Stimulating Activities"
    SOCIALISATION = "Socialisation"
    NAP_TIME_ROUTINE = "Nap Time Routine"
    DISCUSSION_OF_CHILD_PROGRESS = "Discussion of Child's Progress"
    DISCUSSION_WITH_PARENTS = "Discussion with Parents about Observations"
    ECA_DUTIES = "ECA Duties"
    LEADING_ACTIVITIES = "Leading Activities"
    MAINTAINING_DAYCARE = "Maintaining the Daycare"
    MENTORING = "Mentoring"
    OBSERVATION_AND_ASSESSMENT = "Observation and Assessment"
    PREPARE_CRAFT_MATERIALS = "Prepare Craft Materials"
    PROGRESS_TRACKING_CHILDREN = "Progress Tracking of Children"
    RELATIONSHIP_BUILDING = "Relationship Building"
    SING_SONGS = "Sing Songs"
    STORYTELLING = "Storytelling"
    SUPERVISE_ECA = "Supervise an ECA"
    TEACH_SONGS = "Teach Songs"
    PHYSICAL_DISABILITIES = "Physical Disabilities"
    CURRICULUM_PLANS = "Curriculum Plans"
    ACTIVITY_PLANS = "Activity Plans"
    BEHAVIOUR_INTERVENTION_PLANS = "Behaviour Intervention Plans"
    APPLIED_BEHAVIOUR_ANALYSIS = "Applied Behaviour Analysis"
    CLASSROOM_MANAGEMENT = "Classroom Management"
    PROGRAM_DEVELOPMENT = "Program Development"
    CHILD_GROWTH_AND_DEVELOPMENT = "Child Growth and Development"
    EARLY_CHILDHOOD_LITERACY = "Early Childhood Literacy"
    SENSORY_INTEGRATION = "Sensory Integration"
    COMMUNICATION_STRATEGIES = "Communication Strategies"
    INFANT_CARE = "Infant Care"
    CHILD_DEVELOPMENT = "Child Development"
    LANGUAGE_DEVELOPMENT = "Language Development"
    SOCIALIZATION = "Socialization"
    PLAY_BASED_LEARNING = "Play-Based Learning"
    SAFETY_PLANNING = "Safety Planning"
    GOAL_SETTING = "Goal Setting"
    EMOTIONAL_SUPPORT = "Emotional Support"
    BEHAVIORAL_SUPPORT = "Behavioral Support"
    BEHAVIORAL_INTERVENTION = "Behavioral Intervention"
    ANGER_MANAGEMENT = "Anger Management"
    LIFE_SKILLS = "Life Skills"
    SOCIAL_SUPPORT = "Social Support"
    LIAISON = "Liaison"
    HARM_REDUCTION = "Harm Reduction"
    ADVOCACY = "Advocacy"
    TEACHING = "Teaching"
    PROBLEM_SOLVING = "Problem Solving"
    RECEIVABLES = "Receivables"
    GREETING = "Greeting"
    SCREENING = "Screening"
    PORTER = "Porter"
    MEAL_PLANNING = "Meal Planning"
    CARE_COORDINATION = "Care Coordination"
    BARTENDING = "Bartending"
    BUS_TABLES = "Bus Tables"
    CLEANING_KITCHEN_EQUIPMENT = "Cleaning Kitchen Equipment"
    SANITISING = "Sanitising"
    DINING_ROOM_SETUP = "Dining Room Set Up"
    DISHWASHING = "Dishwashing"
    INDUSTRIAL_EQUIPMENT = "Industrial Equipment"
    FULFILLING_CUSTOMER_REQUESTS = "Fulfilling Customer Requests"
    WIPING = "Wiping"
    VACUUMING = "Vacuuming"
    MOPPING = "Mopping"
    CLEAN_KITCHEN = "Clean Kitchen"
    CLEAN_BATHROOM = "Clean Bathroom"
    DISHES = "Dishes"
    LOAD_UNLOAD_DISHWASHER = "Load/Unload Dishwasher"
    BEDDING_CHANGE = "Bedding Change"
    GARBAGE_RECYCLING = "Garbage/Recycling"
    GENERAL_HOUSEKEEPING = "General Housekeeping"
    LAUNDRY_DUTIES = "Laundry Duties"
    LIGHT_TIDYING = "Light Tidying"
    ORGANISING = "Organising"
    EQUIPMENT_MAINTENANCE = "Equipment Maintenance"
    SUPPLY_REPLENISH = "Supply Replenish"
    DUSTING = "Dusting"
    WINDOWS = "Windows"
    DEEP_CLEAN = "Deep Clean"
    OVEN = "Oven"
    FRIDGE = "Fridge"
    SPOILED_FOOD_REMOVAL = "Spoiled Food Removal"
    UTENSIL_MAINTENANCE = "Utensil Maintenance"
    SERVE_LARGE_GROUPS = "Serve Large Groups"
    SET_TABLES = "Set Tables"
    TAKE_ORDERS = "Take Orders"
    SERVE = "Serve"
    ADMINISTRATION_DUTIES = "Administration Duties"
    SCHEDULING = "Scheduling"
    WELCOMING = "Welcoming"
    CUSTOMER_SERVICE = "Customer Service"
    CHECK_IN = "Check-In"
    VISITOR_LOG = "Visitor Log"
    ENFORCE_SAFETY_PROTOCOLS = "Enforce Safety Protocols"
    COVID_SCREENING = "COVID Screening"
    FOOD_PREPARATION = "Food Preparation"
    ATTENDANCE_TRACKING = "Attendance Tracking"
    HANDLE_EMERGENCY_COVERAGE = "Handle Emergency Coverage"
    COMPLIANCE_AND_REGULATIONS = "Compliance and Regulations"
    PROCESS_PAYROLL = "Process Payroll"
    DATA_ENTRY_AND_RECORD_KEEPING = "Data Entry and Record Keeping"
    BENEFITS = "Benefits"
    PAYROLL_REPORTING = "Payroll Reporting"
    MEAL_DELIVERY = "Meal Delivery"
    ASSISTANCE_WITH_MEAL_CHOICES = "Assistance with Meal Choices"
    FOOD_SAFETY = "Food Safety"
    ORDERING = "Ordering"
    MENU_PLANNING = "Menu Planning"
    QUALITY_CONTROL = "Quality Control"
    DIETARY_GUIDELINE_ADHERENCE = "Dietary Guideline Adherence"
    ORDER_PICKUP = "Order Pickup"
    PAYMENT_HANDLING = "Payment Handling"

class HealthcareSetting(str, Enum):
    """
    Enumeration of possible healthcare settings where care can be provided.
    
    Attributes:
        HOSPITAL: Care provided in a hospital setting.
        HOME: Care provided in the patient's home.
        NURSING_FACILITY: Care provided in a nursing facility.
        ASSISTED_LIVING: Care provided in an assisted living facility.
        REHABILITATION_CENTER: Care provided in a rehabilitation center.
    """
    HOSPITAL = "hospital"
    HOME = "home"
    NURSING_FACILITY = "nursing_facility"
    ASSISTED_LIVING = "assisted_living"
    REHABILITATION_CENTER = "rehabilitation_center"

class PropertyType(str, Enum):
    DETACHED_SINGLE_FAMILY_HOME = "Detached single-family home"
    DUPLEX = "Duplex"
    TOWNHOUSE = "Townhouse"
    APARTMENT = "Apartment"
    CONDOMINIUM = "Condominium"
    MOBILE_HOME = "Mobile home"
    MODULAR_HOME = "Modular home"
    BASEMENT_SUITE = "Basement suite"
    HOTEL_ROOM = "Hotel room"

class HealthCondition(str, Enum):
    """
    Enumeration of possible patient health conditions.
    
    Attributes:
        GOOD: Patient in good health, minimal assistance needed
        STABLE: Patient condition is stable
        MODERATE: Patient requires moderate level of care
        SERIOUS: Patient requires serious medical attention
        CRITICAL: Patient requires critical care
        REQUIRES_MONITORING: Patient needs constant monitoring
    """
    GOOD = "good"
    STABLE = "stable"
    MODERATE = "moderate"
    SERIOUS = "serious"
    CRITICAL = "critical"
    REQUIRES_MONITORING = "requires monitoring"
    
class CareTime(str, Enum):
    DAY_TIME = "Day time"
    OVERNIGHT = "Overnight"
    FULL_DAY = "Full day"
    
    
class Person(BaseModel):
    """
    A person with basic information.

    Attributes:
        first_name: The person's first name.
        last_name: The person's last name.
        age: The person's age.
        date_of_birth: The person's date of birth (optional).
        gender: The person's gender (optional).
        phone_number: The person's phone number (optional).
        address: The person's address (optional).
        unit_suit_no: The unit or suite number (optional).
        postal_code: The person's postal code (optional).
        city: The person's city (optional).
        province_state: The person's province or state (optional).
        weight: The person's weight (optional).
        emergency_contact: The person's emergency contact (optional).
    """
    first_name: str = Field( 
        min_length=1, 
        max_length=100, 
        description="The person's first name"
        )
    last_name: str = Field(
        min_length=1, 
        max_length=100, 
        description="The person's last name"
        )
    age: int = Field( 
        gt=0, 
        le=150, 
        description="The person's age, must be between 0 and 120"
        )
    # to better implement later
    # date_of_birth: Optional[datetime.date] = Field( 
        # description="The person's date of birth"
        # )
    date_of_birth: str | None = Field(
        default=None, 
        description="The person's date of birth"
        )
    gender: Optional[str] = Field(
        None, 
        max_length=20, 
        description="The person's gender"
        )
    # Regex validation for phone number format
    phone_number: Optional[str] = Field(
        default=None, 
        max_length=15, 
        description="The person's phone number",
        regex=r'^\+?[0-9\s\-\(\)]{10,15}$'  # Corrected usage of regex inside Field
    )
    address: Optional[str] = Field(
        default=None, 
        max_length=255, 
        description="The person's address"
        )
    unit_suit_no: Optional[str] = Field(
        default=None, 
        max_length=20, 
        description="The unit or suite number"
        )
    postal_code: Optional[str] = Field(
        default=None, 
        max_length=10, 
        description="The person's postal code"
        )
    city: Optional[str] = Field(
        default=None, 
        max_length=100, 
        description="The person's city"
        )
    province_state: Optional[str] = Field(
        default=None, 
        max_length=100, 
        description="The person's province or state"
        )
    weight: Optional[float] = Field(
        default=None, 
        gt=0, 
        description="The person's weight in kilograms"
        )
    emergency_contact: Optional[str] = Field(
        default=None, 
        max_length=255, 
        description="The person's emergency contact"
        )
    
    location : Optional[Tuple[float, float]] = Field(
        default = (None, None),
        description="The person's location"
        )

    class Config:
        min_anystr_length = 1
        anystr_strip_whitespace = True
        # validate_assignment = True
        # extra = "forbid"
        # json_schema_extra = {
        #     "examples": [
        #         {
        #             "patient_id": 1,
        #             "job_description": "Full-time care needed for elderly patient with mobility issues. Assistance required with daily activities and medication management.",
        #             "rate": 35.50,
        #             "healthcare_setting": "home",
        #             "property_type": "house",
        #             "health_condition": "stable",
        #             "caregiver_type": "certified_nursing_assistant",
        #             "credentials": ["certified_nursing_assistant", "cpr_certified"],
        #             "careservices": ["medication_management", "mobility_assistance", "personal_hygiene"],
        #             "budget": 1500.00,
        #             "care_date": 1735689600,  # Example future timestamp
        #             "care_location": (40.7128, -74.0060)  # Example coordinates for New York
        #         }
        #     ]
        # }
    
    # @property
    # def formatted_care_date(self) -> str | None:
    #     """
    #     Formats the care date timestamp as a human-readable string.
        
    #     Returns:
    #         str: Formatted date string in "YYYY-MM-DD HH:MM:SS" format,
    #              or None if no date is set
    #     """
    #     if self.care_date is None:
    #         return None
    #     return datetime.fromtimestamp(self.care_date).strftime("%Y-%m-%d %H:%M:%S")
    
    # @validator('careservices')
    # def validate_careservices(cls, v):
    #     """
    #     Validates that at least one care service is specified when services are provided.
        
    #     Args:
    #         v: List of care services to validate
            
    #     Returns:
    #         The validated care services list
            
    #     Raises:
    #         ValueError: If the care services list is empty
    #     """
    #     if v is not None and len(v) == 0:
    #         raise ValueError("At least one care service must be specified")
    #     return v
    