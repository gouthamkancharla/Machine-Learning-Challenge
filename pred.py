import numpy as np
import pandas as pd
import sys
import csv
import random

def preprocess_data(dataset):
    """
    Preprocesses the input dataset to generate a feature matrix for prediction.

    The function performs the following steps:
    1. Renames dataset columns to standardized names.
    2. Extracts and transforms features from the dataset:
        - F1: Food complexity (directly used as is).
        - F2: Number of ingredients (normalized after cleaning text).
        - F3: Occasion versatility (encoded as a one-hot vector).
        - F4: Price per serving (normalized after cleaning text).
        - F5: Reminiscent movie (mapped to predefined categories).
        - F6: Drink to pair with (mapped to predefined categories).
        - F7: Reminiscent persons versatility (encoded as a one-hot vector).
        - F8: Hot sauce level (mapped to predefined categories).
    3. Combines all processed features into a feature matrix.

    Args:
        dataset (pd.DataFrame): Input dataset containing raw data.

    Returns:
        np.ndarray: A 2D numpy array where each row represents a processed data point
                    and each column corresponds to a specific feature.
    """
    def generate_fet1_food_complexity():
        return dataset["P1"]

    def generate_fet2_number_of_ingredients():
        for i in range(len(dataset)):
            s = dataset["P2"][i]
            try:
                for j in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz;''][:,.<>?{}|=+-_)(*&^%$#@!~`\":":
                    s = s.replace(j, " ")
            except:
                pass
            q = 0
            try:
                s = s.split(" ")
                for j in s:
                    if j.isdigit():
                        q = int(j)
                        break
            except:
                pass
            dataset.loc[i, ("P2")] = q
        retlist = np.array(dataset["P2"])
        retlist = (retlist - 1) / (25 - 1)
        return retlist

    def generate_fet3_8_occasion_versatility():
        retlst = []
        for places in dataset["P3"]:
            x = [0,0,0,0,0,0]
            try:
                l = places.split(",")
                for i in l:
                    if i == "Week day lunch":
                        x[0] = 1
                    elif i == "Week day dinner":
                        x[1] = 1
                    elif i == "Weekend lunch":
                        x[2] = 1
                    elif i == "Weekend dinner":
                        x[3] = 1
                    elif i == "At a party":
                        x[4] = 1
                    elif i == "Late night snack":
                        x[5] = 1
            except:
                pass
            retlst.append(x)
        return np.array(retlst)

    def generate_fet9_price_per_serving():
        for i in range(len(dataset)):
            s = dataset["P4"][i]
            try:
                for j in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz;''][:,.<>?{}|=+-_)(*&^%$#@!~`\":":
                    s = s.replace(j, " ")
            except:
                pass
            q = 0
            try:
                s = s.split(" ")
                for j in s:
                    if j.isdigit():
                        q = int(j)
                        break
            except:
                pass
            dataset.loc[i, ("P4")] = q
        retlist = np.array(dataset["P4"])
        retlist = (retlist - 1) / (100 - 1)
        return retlist

    def generate_fet10_reminiscent_movie():
        retlst= []
        movie_map = {"Home Alone": 1, "Teenage Mutant Ninja Turtles": 1, "spiderman": 1, "Spider-Man 2": 1, "Spiderman": 1,
                     "Cloudy with a Chance of Meatballs": 1, "Cloudy with a chance of meatballs": 1, "The Godfather": 1,
                     "Home alone": 1, "Ratatoullie": 1,
                     "The Avengers": 2, "Avengers": 2, "avengers": 2, "The Avengers (2012)": 2, "none": 2, "Aladdin": 2,
                     "The Dictator": 2, "Borat": 2, "Dangal": 2,
                     "Jiro Dreams of Sushi": 3, "Finding Nemo": 3, "Spirited Away": 3, "Kill Bill": 3, "Your Name": 3,
                     "Kung Fu Panda": 3, "Monsters Inc.": 3, "Godzilla": 3, "The Wolverine": 3, "Big Hero 6": 3
                     }
        for movie in dataset["P5"]:
            if movie in movie_map:
                retlst.append(movie_map[movie])
            else:
                retlst.append(0)
        return retlst

    def generate_fet11_drink_to_pair_with():
        retlst= []
        drink_map = {"Water": 1, "water": 1, "Iced Tea": 1, "Nestea": 1, "Juice": 1,
                     "Coke": 2, "coke": 2, "Soda": 2, "soda": 2, "Coca Cola": 2, "Coca-Cola": 2,
                     "Sprite": 2, "Diet Coke": 2, "Pepsi": 2,
                     "Tea": 3, "Sake": 3, "tea": 3, "Green tea": 3, "Green Tea": 3, "sake": 3, "green tea": 3
                     }
        for drink in dataset["P6"]:
            if drink in drink_map:
                retlst.append(drink_map[drink])
            else:
                retlst.append(0)
        return retlst

    def generate_fet12_16_reminiscent_persons_versatility():
        retlst = [] #Parents,Siblings,Friends,Teachers,Strangers
        for persons in dataset["P7"]:
            x = [0,0,0,0,0]
            try:
                l = persons.split(",")
                for i in l:
                    if i == "Parents":
                        x[0] = 1
                    elif i == "Siblings":
                        x[1] = 1
                    elif i == "Friends":
                        x[2] = 1
                    elif i == "Teachers":
                        x[3] = 1
                    elif i == "Strangers":
                        x[4] = 1
            except:
                pass
            retlst.append(x)
        return np.array(retlst)

    def generate_fet17_hot_sauce_level():
        retlst = []
        hot_sauce_level_map = {"None": 0,
                               "A little (mild)": 1,
                               "A moderate amount (medium)": 2,
                               "A lot (hot)": 3,
                               "I will have some of this food item with my hot sauce": 4
                               }
        for hot_sauce_level in dataset["P8"]:
            retlst.append(hot_sauce_level_map[hot_sauce_level])
        return retlst

    peoples = generate_fet12_16_reminiscent_persons_versatility()
    places = generate_fet3_8_occasion_versatility()

    feature_matrix = np.stack([
        generate_fet1_food_complexity(),
        generate_fet2_number_of_ingredients(),
        places[:,0], places[:,1], places[:,2], places[:,3], places[:,4], places[:,5],
        generate_fet9_price_per_serving(),
        generate_fet10_reminiscent_movie(),
        generate_fet11_drink_to_pair_with(),
        peoples[:,0], peoples[:,1], peoples[:,2], peoples[:,3], peoples[:,4],
        generate_fet17_hot_sauce_level()
    ], axis=1)
    return feature_matrix

def make_prediction(F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17):
    """
    Makes a prediction based on the provided feature values. Using a decision tree model.

    The function uses a series of nested conditional statements to classify the input
    into one of three categories, represented as a one-hot encoded list:
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], or [0.0, 0.0, 1.0].
    """
    if F7 <= 0.5:
        if F17 <= 0.5:
            if F6 <= 0.5:
                if F12 <= 0.5:
                    if F2 <= 0.15:
                        if F9 <= 0.05:
                            return [1.0, 0.0, 0.0]
                        else:  # if F9 > 0.05
                            return [0.0, 1.0, 0.0]
                    else:  # if F2 > 0.15
                        if F16 <= 0.5:
                            return [0.0, 1.0, 0.0]
                        else:  # if F16 > 0.5
                            return [0.0, 1.0, 0.0]
                else:  # if F12 > 0.5
                    if F5 <= 0.5:
                        if F11 <= 0.5:
                            return [0.0, 0.0, 1.0]
                        else:  # if F11 > 0.5
                            return [0.0, 1.0, 0.0]
                    else:  # if F5 > 0.5
                        return [0.0, 0.0, 1.0]
            else:  # if F6 > 0.5
                if F5 <= 0.5:
                    if F2 <= 0.15:
                        return [0.0, 0.0, 1.0]
                    else:  # if F2 > 0.15
                        if F1 <= 4.5:
                            return [0.0, 0.0, 1.0]
                        else:  # if F1 > 4.5
                            return [0.0, 0.0, 1.0]
                else:  # if F5 > 0.5
                    if F11 <= 2.5:
                        if F10 <= 1.5:
                            return [0.0, 0.0, 1.0]
                        else:  # if F10 > 1.5
                            return [0.0, 0.0, 1.0]
                    else:  # if F11 > 2.5
                        return [0.0, 0.0, 1.0]
        else:  # if F17 > 0.5
            if F2 <= 0.19:
                if F11 <= 2.5:
                    if F9 <= 0.16:
                        if F10 <= 1.0:
                            return [0.0, 1.0, 0.0]
                        else:  # if F10 > 1.0
                            return [0.0, 1.0, 0.0]
                    else:  # if F9 > 0.16
                        if F11 <= 1.5:
                            return [0.0, 0.0, 1.0]
                        else:  # if F11 > 1.5
                            return [0.0, 1.0, 0.0]
                else:  # if F11 > 2.5
                    return [0.0, 0.0, 1.0]
            else:  # if F2 > 0.19
                if F9 <= 0.22:
                    if F1 <= 4.5:
                        if F10 <= 2.5:
                            return [0.0, 1.0, 0.0]
                        else:  # if F10 > 2.5
                            return [0.0, 0.0, 1.0]
                    else:  # if F1 > 4.5
                        if F17 <= 1.5:
                            return [0.0, 0.0, 1.0]
                        else:  # if F17 > 1.5
                            return [0.0, 1.0, 0.0]
                else:  # if F9 > 0.22
                    if F1 <= 3.0:
                        return [1.0, 0.0, 0.0]
                    else:  # if F1 > 3.0
                        return [0.0, 0.0, 1.0]
    else:  # if F7 > 0.5
        if F10 <= 1.5:
            if F11 <= 2.5:
                if F11 <= 1.5:
                    if F9 <= 0.06:
                        if F2 <= 0.23:
                            return [1.0, 0.0, 0.0]
                        else:  # if F2 > 0.23
                            return [1.0, 0.0, 0.0]
                    else:  # if F9 > 0.06
                        if F11 <= 0.5:
                            return [1.0, 0.0, 0.0]
                        else:  # if F11 > 0.5
                            return [0.0, 0.0, 1.0]
                else:  # if F11 > 1.5
                    if F17 <= 1.5:
                        if F9 <= 0.0:
                            return [0.0, 0.0, 1.0]
                        else:  # if F9 > 0.0
                            return [1.0, 0.0, 0.0]
                    else:  # if F17 > 1.5
                        if F9 <= 0.06:
                            return [1.0, 0.0, 0.0]
                        else:  # if F9 > 0.06
                            return [1.0, 0.0, 0.0]
            else:  # if F11 > 2.5
                return [0.0, 0.0, 1.0]
        else:  # if F10 > 1.5
            if F10 <= 2.5:
                if F9 <= 0.05:
                    if F2 <= 0.13:
                        return [0.0, 1.0, 0.0]
                    else:  # if F2 > 0.13
                        return [1.0, 0.0, 0.0]
                else:  # if F9 > 0.05
                    if F8 <= 0.5:
                        if F1 <= 1.5:
                            return [0.0, 0.0, 1.0]
                        else:  # if F1 > 1.5
                            return [0.0, 1.0, 0.0]
                    else:  # if F8 > 0.5
                        if F9 <= 0.14:
                            return [0.0, 1.0, 0.0]
                        else:  # if F9 > 0.14
                            return [0.0, 1.0, 0.0]
            else:  # if F10 > 2.5
                if F2 <= 0.27:
                    return [0.0, 0.0, 1.0]
                else:  # if F2 > 0.27
                    if F9 <= 0.08:
                        return [1.0, 0.0, 0.0]
                    else:  # if F9 > 0.08
                        return [0.0, 0.0, 1.0]

def predict(data):
    """
    This function is a wrapper around the make_prediction function, which takes the seventeen
    features of a data point as input and returns the predicted food category.
    """
    return make_prediction(
        data[0], data[1], data[2], data[3], data[4],
        data[5], data[6], data[7], data[8], data[9],
        data[10], data[11], data[12], data[13], data[14],
        data[15], data[16]
    )

def predict_all(csv_file_path):
    """
    Processes a CSV file containing input data, generates predictions for each row,
    and writes the predictions to an output CSV file.

    The function performs the following steps:
    1. Reads the input CSV file into a pandas DataFrame.
    2. Preprocesses the data using the `preprocess_data` function to generate a feature matrix.
    3. Iterates through the feature matrix, makes predictions for each row using the `make_prediction` function.
    4. Returns a list of predictions, where each prediction corresponds to a food category.

    Args:
        csv_file_path (str): Path to the input CSV file containing the raw data.

    Returns:
        list: A list of predictions for each row in the input CSV file.
    """
    try:
        # Read CSV, automatically detecting header. Keep default NA values for now.
        # Use keep_default_na=False to treat empty strings as strings, not NaN initially
        df = pd.read_csv(csv_file_path, keep_default_na=False)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Column Renaming ---
    # Assume the features P1-P8 (or Q1-Q8) are always in columns 1 through 8 (0-indexed)
    if df.shape[1] < 9:
        print(f"Error: CSV file has fewer than 9 columns ({df.shape[1]}). Expected at least ID + 8 features.", file=sys.stderr)
        sys.exit(1)

    # Get the names of the columns that *should* contain the features
    original_feature_cols = df.columns[1:9]

    # Define the target names used internally by preprocess_data
    target_feature_names = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

    # Create the renaming dictionary
    rename_map = dict(zip(original_feature_cols, target_feature_names))

    # Rename the columns in the DataFrame
    try:
        df.rename(columns=rename_map, inplace=True)
    except Exception as e:
        print(f"Error renaming columns: {e}", file=sys.stderr)
        print("Original columns:", df.columns, file=sys.stderr)
        print("Rename map:", rename_map, file=sys.stderr)
        sys.exit(1)

    data_fets = preprocess_data(df)
    results = []
    for i in data_fets:
        x = predict(i)
        if x[0] == 1.0:
            results.append("Pizza")
        elif x[1] == 1.0:
            results.append("Shawarma")
        elif x[2] == 1.0:
            results.append("Sushi")
    return results
