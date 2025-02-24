import numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# Load CSV file
file_path = "road_accident_dataset.csv"
original_df = pd.read_csv(file_path)

def data_preprocess(original_df, year="categorical"):
    df = original_df.copy()
    le = LabelEncoder()
    # Country
    #['USA' 'UK' 'Canada' 'India' 'China' 'Japan' 'Russia' 'Brazil' 'Germany' 'Australia']
    # Store mappings for decoding
    df["Country_Encoded"] = le.fit_transform(df["Country"])
    country_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    country_remapping = {v: k for k, v in country_mapping.items()}

    # Year
    if (year == "numerical"):
        df["Year_Encoded"] = le.fit_transform(df["Year"])
        year_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        year_remapping = {v: k for k, v in year_mapping.items()}

    # Month - Ordinary Encoding
    # ['October' 'December' 'July' 'May' 'March' 'August' 'April' 'September' 'January' 'February' 'June' 'November']
    month_order = [['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]
    month_encoder = OrdinalEncoder(categories=month_order)
    df["Month_Encoded"] = month_encoder.fit_transform(df[["Month"]]) + 1

    # Day of Week - Ordinary Encoding
    # ['Tuesday' 'Saturday' 'Sunday' 'Monday' 'Friday' 'Thursday' 'Wednesday']
    week_order = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
    week_encoder = OrdinalEncoder(categories=week_order)
    df["Day_of_Week_Encoded"] = week_encoder.fit_transform(df[["Day of Week"]]) + 1


    # Time of Day - Ordinary Encoding
    # ['Evening' 'Afternoon' 'Night' 'Morning']
    day_order = [['Morning', 'Afternoon', 'Evening', 'Night']]
    day_encoder = OrdinalEncoder(categories=day_order)
    df["Time_of_Day_Encoded"] = day_encoder.fit_transform(df[["Time of Day"]])

    # Urban / Rural
    # ['Rural' 'Urban']
    df["Urban/Rural_Encoded"] = le.fit_transform(df["Urban/Rural"])
    urban_rural_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    urban_rural_mapping_remapping = {v: k for k, v in urban_rural_mapping.items()}

    # Road Type
    # ['Street' 'Highway' 'Main Road']
    df["Road_Type_Encoded"] = le.fit_transform(df["Road Type"])
    road_type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    road_type_remapping = {v: k for k, v in road_type_mapping.items()}

    # Weather Conditions
    # ['Windy' 'Snowy' 'Clear' 'Rainy' 'Foggy']
    df["Weather_Conditions_Encoded"] = le.fit_transform(df["Weather Conditions"])
    weather_conditions_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    weather_conditions_remapping = {v: k for k, v in weather_conditions_mapping.items()}

    # Driver Age Group - Ordinary Encoding
    # ['18-25' '41-60' '26-40' '<18' '61+']
    age_order = [['<18', '18-25', '26-40', '41-60', '61+']]
    age_encoder = OrdinalEncoder(categories=age_order)
    df["Driver_Age_Group_Encoded"] = age_encoder.fit_transform(df[["Driver Age Group"]])

    # Driver Gender
    # ['Male' 'Female']
    df["Driver_Gender_Encoded"] = le.fit_transform(df["Driver Gender"])
    driver_gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    driver_gender_remapping = {v: k for k, v in driver_gender_mapping.items()}

    # Vehicle Condition - Ordinary Encoding
    # ['Poor' 'Moderate' 'Good']
    vehicle_condition_order = [['Good', 'Moderate', 'Poor']]
    vehicle_condition_encoder = OrdinalEncoder(categories=vehicle_condition_order)
    df["Vehicle_Condition_Encoded"] = vehicle_condition_encoder.fit_transform(df[["Vehicle Condition"]])

    # Accident Severity - Ordinary Encoding
    # ['Moderate' 'Minor' 'Severe']
    accident_severity_order = [['Minor', 'Moderate', 'Severe']]
    accident_severity_encoder = OrdinalEncoder(categories=accident_severity_order)
    df["Accident_Severity_Encoded"] = accident_severity_encoder.fit_transform(df[["Accident Severity"]])

    # Road Condition - Ordinary Encoding
    # ['Wet' 'Snow-covered' 'Icy' 'Dry']
    road_condition_order = [['Dry', 'Wet', 'Snow-covered', 'Icy']]
    road_condition_encoder = OrdinalEncoder(categories=road_condition_order)
    df["Road_Condition_Encoded"] = road_condition_encoder.fit_transform(df[["Road Condition"]])

    # Accident Cause 
    # ['Weather' 'Mechanical Failure' 'Speeding' 'Distracted Driving' 'Drunk Driving']
    df["Accident_Cause_Encoded"] = le.fit_transform(df["Accident Cause"])
    accident_cause_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    accident_cause_remapping = {v: k for k, v in accident_cause_mapping.items()}

    # Region
    # ['Europe' 'North America' 'South America' 'Australia' 'Asia']
    df["Region_Encoded"] = le.fit_transform(df["Region"])
    region_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    region_remapping = {v: k for k, v in region_mapping.items()}


    # Remove original column
    df.drop(columns=["Country"], inplace=True)
    df.drop(columns=["Month"], inplace=True)
    df.drop(columns=["Day of Week"], inplace=True)
    df.drop(columns=["Time of Day"], inplace=True)
    df.drop(columns=["Urban/Rural"], inplace=True)
    df.drop(columns=["Road Type"], inplace=True)
    df.drop(columns=["Weather Conditions"], inplace=True)
    df.drop(columns=["Driver Age Group"], inplace=True)
    df.drop(columns=["Driver Gender"], inplace=True)
    df.drop(columns=["Vehicle Condition"], inplace=True)
    df.drop(columns=["Accident Severity"], inplace=True)
    df.drop(columns=["Road Condition"], inplace=True)
    df.drop(columns=["Accident Cause"], inplace=True)
    df.drop(columns=["Region"], inplace=True)
    if (year == "numerical"):
        df.drop(columns=["Year"], inplace=True)



    if (year == "numerical"):
        return df, year_remapping, country_remapping, month_encoder, week_encoder, day_encoder, urban_rural_mapping_remapping, road_type_remapping, weather_conditions_remapping, age_encoder, driver_gender_remapping, vehicle_condition_encoder, accident_severity_encoder, road_condition_encoder, accident_cause_remapping, region_remapping
    
    return df, country_remapping, month_encoder, week_encoder, day_encoder, urban_rural_mapping_remapping, road_type_remapping, weather_conditions_remapping, age_encoder, driver_gender_remapping, vehicle_condition_encoder, accident_severity_encoder, road_condition_encoder, accident_cause_remapping, region_remapping


def decode_label(encoded_series, reverse_mapping):
    return encoded_series.map(reverse_mapping)

def decode_ordinal(encoded_series, encoder, sub_one = "no"):
    if (sub_one == "yes"):
        corrected_series = encoded_series - 1  # Shift values back before decoding
        decoded_values = encoder.inverse_transform(corrected_series.to_frame())
    else:
        decoded_values = encoder.inverse_transform(encoded_series.to_frame())
    return pd.Series(decoded_values.flatten(), index=encoded_series.index)

def decoder(encoded_df, encoder, column_name):
    reversed_name = column_name.replace("_Encoded", "")

    
    if isinstance(encoder, OrdinalEncoder):
        if(column_name == "Month_Encoded" or column_name =="Day_of_Week_Encoded"):
            encoded_df[reversed_name] = decode_ordinal(encoded_df[column_name], encoder, sub_one = "yes")
        else: 
            encoded_df[reversed_name] = decode_ordinal(encoded_df[column_name], encoder, sub_one="no")
        return encoded_df
    else:
        encoded_df[reversed_name] = decode_label(encoded_df[column_name], encoder)
        return encoded_df


def decode_all(df, encoder_list, encoded_name_list):
    decoded_df = df
    for index, (encoder, encoded_name) in enumerate (zip(encoder_list, encoded_name_list)):
        decoded_df = decoder(decoded_df, encoder, encoded_name)
        decoded_df.drop(columns=[encoded_name], inplace=True)

    return decoded_df


# =====================================================================================================

print(original_df.head(5))

encoded_df, country_remapping, month_encoder, week_encoder, day_encoder, urban_rural_mapping_remapping, road_type_remapping, weather_conditions_remapping, age_encoder, driver_gender_remapping, vehicle_condition_encoder, accident_severity_encoder, road_condition_encoder, accident_cause_remapping, region_remapping = data_preprocess(original_df)
print(encoded_df.head(5))

encoder_list = [country_remapping, month_encoder, week_encoder, day_encoder, urban_rural_mapping_remapping, road_type_remapping, weather_conditions_remapping, age_encoder, driver_gender_remapping, vehicle_condition_encoder, accident_severity_encoder, road_condition_encoder, accident_cause_remapping, region_remapping]
encoded_name_list = ["Country_Encoded", "Month_Encoded", "Day_of_Week_Encoded", "Time_of_Day_Encoded", "Urban/Rural_Encoded", "Road_Type_Encoded", "Weather_Conditions_Encoded", "Driver_Age_Group_Encoded", "Driver_Gender_Encoded", "Vehicle_Condition_Encoded", "Accident_Severity_Encoded", "Road_Condition_Encoded", "Accident_Cause_Encoded", "Region_Encoded"]
decoded_df = decode_all(encoded_df, encoder_list, encoded_name_list)
print(decoded_df.head(5))



# Example of decode a single column: 
# decoded_df = decoder(encoded_df, month_encoder, "Month_Encoded")
# print(decoded_df)

