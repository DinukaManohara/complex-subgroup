uci_datasets_info = {
    "sepsis_survival_primary_cohort": {
        "delim": ",",
        "size": 110204,
        "target_variable": ["hospital_outcome_1alive_0dead"],
        "categorical_variables": ["hospital_outcome_1alive_0dead"],
        "id_variables": []
    },
    "social_network_ads": {
        "delim": ",",
        "size": 400,
        "target_variable": ["Purchased"],
        "categorical_variables": ["Gender"],
        "id_variables": ["User ID"]
    },
    "uk_used_cars_ford": {
        "delim": ",",
        "size": 17965,
        "target_variable": ["price"],
        "categorical_variables": ["model", "transmission", "fuelType"],
        "id_variables": []
    },

    "combined_cycle_power_plant": {
        "delim": ",",
        "size": 9568,
        "target_variable": ["PE"],
        "categorical_variables": [],
        "id_variables": []
    },
    "protein_tertiary_structure": {
        "delim": ",",
        "size": 45730,
        "target_variable": ["RMSD"],
        "categorical_variables": [],
        "id_variables": []
    },
    "gas_turbine_emissions": {
        "delim": ",",
        "size": 36733,
        "target_variable": ["CO", "NOX"],
        "categorical_variables": [],
        "id_variables": []
    },
    # "blog_feedback": {
    #     "delim": ",",
    #     "size": 60021,
    #     "target_variable": ["target"],
    #     "categorical_variables": [],
    #     "id_variables": []
    # },
    "naval_propulsion_plants": {
        "delim": ",",
        "size": 11934,
        "target_variable": ["compressor_decay", "turbine_decay"],
        "categorical_variables": [],
        "id_variables": []
    },
    # "ct_slices_axial": {
    #     "delim": ",",
    #     "size": 53500,
    #     "target_variable": ["reference"],
    #     "categorical_variables": [],
    #     "id_variables": ["patientId"]
    # },
    "winequality-red": {
        "delim":";",
        "size":1599,
        "target_variable": ["quality"],
        "categorical_variables": [],
        "id_variables": []
    },
    "winequality-white": {
        "delim":";",
        "size":4898,
        "target_variable": ["quality"],
        "categorical_variables": [],
        "id_variables": []
    },
    "dry_bean": {
        "delim":",",
        "size":13611,
        "target_variable": ["Class"],
        "categorical_variables": ["Class"],
        "id_variables": []
    },
    "insurance": {
        "delim":",",
        "size":1338,
        "target_variable": ["charges"],
        "categorical_variables": ["smoker"],
        "id_variables": ["region", "sex", "children"]
    },
    # "covertype": {
    #     "delim":",",
    #     "size":581012,
    #     "target_variable": ["Cover_Type"],
    #     "categorical_variables": ["Wilderness_Area", "Soil_Type"], # Note: Provided as binary one-hot columns in the raw dataset
    #     "id_variables": []
    # },
    # "raisin": {
    #     "delim":",",
    #     "size":900,
    #     "target_variable": ["Class"],
    #     "categorical_variables": ["Class"],
    #     "id_variables": []
    # },
    "yeast": {
        "delim":",",
        "size":1484,
        "target_variable": ["localization_site"],
        "categorical_variables": ["localization_site"],
        "id_variables": ["Sequence_Name"]
    },
    # "mice_protein_expression": {
    #     "delim":",",
    #     "size":1080,
    #     "target_variable": ["class"], 
    #     "categorical_variables": ["Genotype", "Treatment", "Behavior", "class"],
    #     "id_variables": ["MouseID"]
    # },
    # "concrete_compressive_strength": {
    #     "delim":",",
    #     "size":1030,
    #     "target_variable": ["Concrete_compressive_strength"],
    #     "categorical_variables": [],
    #     "id_variables": []
    # },
    "steel_industry_energy_consumption": {
        "delim":",",
        "size":35040,
        "target_variable": ["Load_Type"],
        "categorical_variables": ["WeekStatus", "Day_of_week", "Load_Type"],
        "id_variables": ["date"]
    },
    "superconductivity": {
        "delim":",",
        "size":21263,
        "target_variable": ["critical_temp"],
        "categorical_variables": [],
        "id_variables": []
    },
    "airfoil_self_noise": {
        "delim":",",
        "size":1503,
        "target_variable": ["scaled_sound_pressure_level"],
        "categorical_variables": [],
        "id_variables": []
    },
    # "statlog_shuttle": {
    #     "target_variable": ["class"],
    #     "categorical_variables": [],
    #     "id_variables": []
    # },
    # "musk_version_2": {
    #     "delim":",",
    #     "size":6598,
    #     "target_variable": ["class"],
    #     "categorical_variables": [],
    #     "id_variables": ["molecule_name", "conformation_name"]
    # },
    # "estimation_of_obesity_levels": {
    #     "delim":",",
    #     "size":2111,
    #     "target_variable": ["NObeyesdad"],
    #     "categorical_variables": ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"],
    #     "id_variables": []
    # },
    # "diabetes_130_us_hospitals": {
    #     "delim":",",
    #     "size":101766,
    #     "target_variable": ["readmitted"],
    #     "categorical_variables": [
    #         "race", "gender", "age", "weight", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3", 
    #         "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide", "chlorpropamide", 
    #         "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
    #         "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", 
    #         "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", 
    #         "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed", "readmitted"
    #     ],
    #     "id_variables": ["encounter_id", "patient_nbr"]
    # },
    # "maternal_health_risk": {
    #     "delim":",",
    #     "size":1014,
    #     "target_variable": ["RiskLevel"],
    #     "categorical_variables": ["RiskLevel"],
    #     "id_variables": []
    # },
    # "support2": {
    #     "delim":",",
    #     "target_variable": ["death", "hospdead"], # Mortality targets
    #     "categorical_variables": ["sex", "dzgroup", "dzclass", "income", "race", "sfdm2"],
    #     "id_variables": ["index"]
    # },
    "eeg_eye_state": {
        "delim":",",
        "size":14980,
        "target_variable": ["eyeDetection"],
        "categorical_variables": ["eyeDetection"],
        "id_variables": []
    },
    "hcv_egyptian_patients": {
        "delim":",",
        "size":1385,
        "target_variable": ["Baseline histological Staging"],
        "categorical_variables": ["Gender", "Fever", "Nausea/Vomting", "Headache", "Diarrhea", "Fatigue & generalized bone ache", "Jaundice", "Epigastric pain"],
        "id_variables": []
    },
    # "contraceptive_method_choice": {
    #     "delim":",",
    #     "size":1473,
    #     "target_variable": ["Contraceptive method used"],
    #     "categorical_variables": ["Wife's education", "Husband's education", "Wife's religion", "Wife's now working?", "Husband's occupation", "Standard-of-living index", "Media exposure"],
    #     "id_variables": []
    # },
    # "bike_sharing": {
    #     "delim":",",
    #     "size":17379,
    #     "target_variable": ["cnt"], # Can also include "casual" and "registered"
    #     "categorical_variables": ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"],
    #     "id_variables": ["instant", "dteday"]
    # },
    # "drug_consumption_quantified": {
    #     "target_variable": [
    #         "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke", "Crack", 
    #         "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth", "Mushrooms", 
    #         "Nicotine", "Semer", "VSA"
    #     ], # 18 separate target drugs
    #     "categorical_variables": ["Age", "Gender", "Education", "Country", "Ethnicity"],
    #     "id_variables": ["ID"]
    # },
    "power_consumption_tetouan_city": {
        "delim":",",
        "size":52416,
        "target_variable": ["Zone 1 Power Consumption", "Zone 2 Power Consumption", "Zone 3 Power Consumption"],
        "categorical_variables": [],
        "id_variables": ["DateTime"]
    },
    "parkinsons": {
        "delim": ",",
        "size": 5875,
        "target_variable": ["motor_UPDRS", "total_UPDRS"],
        "categorical_variables": ["sex"],
        "id_variables": ["subject#"]
    },
    "california": {
        "delim": ",",
        "size": 20640,
        "target_variable": ["median_house_value"],
        "categorical_variables": ["ocean_proximity"],
        "id_variables": []
    },
    # "air_quality": {
    #     "delim": ";",
    #     "size": 9358,
    #     "target_variable": ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
    #     "categorical_variables": [],
    #     "id_variables": ["Date", "Time"]
    # },
    "beijing": {
        "delim": ",",
        "size": 43824,
        "target_variable": ["pm2.5"],
        "categorical_variables": ["cbwd"],
        "id_variables": ["No"]
    },
    "abalone": {
        "delim": ",",
        "size": 4177,
        "target_variable": ["Rings"],
        "categorical_variables": ["Sex"],
        "id_variables": []
    },
    # "sgemm": {
    #     "delim": ",",
    #     "size": 241600,
    #     "target_variable": ["Run1", "Run2", "Run3", "Run4"],
    #     "categorical_variables": ["STRM", "STRN", "SA", "SB"],
    #     "id_variables": []
    # },
    "appliance_energy": {
        "delim": ",",
        "size": 19735,
        "target_variable": ["Appliances"],
        "categorical_variables": [],
        "id_variables": ["date"]
    }
} 