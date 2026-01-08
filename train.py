import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# // standardscalar stretches small values to be larger and larger values to be smaller to even out the playin field
# // onehotencoder converts categorical variables into a 0/1 format so that everything is YES or NO 

from sklearn.compose import ColumnTransformer
# // columntransformer lets us applies different transformations to different columns at the same time 

from sklearn.pipeline import Pipeline 
# // pipeline lets us do everything in one step in same order as every time as before 

def main():
    df = pd.read_csv('data.csv') ##df is a dataframe
    # df is full dataset and we are reading it using pandas library at this point 

    # X = Input (what we want to predict) , Y = output (what data we have to predict it)

    X = df.drop('median_house_value', axis=1) ##axis 1 = column, axis 0 = row 
    ##QUESTION:
    # why are we dropping median_house_value? because that is what we want to predict

    Y = df['median_house_value']
    ## just defining what we want to predict here 
    # =========================
    # 4. TRAIN / TEST SPLIT
    # =========================
    # next step;
    # Creating a training data and a testing data set from our full dataset in order to train the model 

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # test_size = 0.2 means 20% of data will be used for testing, 80% for training 
    # if youre wondering where the 42 comes from then its an inside joke because fuck programmers and data scientists
    # it only creates random split but exact value makes sure we split it in the same way every time even if its random

    # data leakage prevention

    # =========================
    # 5. IDENTIFY COLUMN TYPES
    # =========================

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns

    # Why detect automatically?
    # So code doesn't break if dataset changes.
    # Hardcoding column names is fragile.

    # =========================
    #  DEFINE PREPROCESSING RULES
    # =========================
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # if data is number then scale it if it is not a number then dont scale it thats basically our entire pipeline
    # or processing for now... if its a cloth then put into washing machine if its a plate then do NOT 

    # ---- CATEGORICAL PIPELINE ----

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) 

    #     # OneHotEncoder: //if its a plate then put 1 in plate column and 0 in all other columns, you dont need to know 
    # how 
    # - converts categories into binary columns
    # handle_unknown="ignore" prevents crashes
    # if unseen categories appear in test data 

    # =========================
    # 7. COMBINE PREPROCESSING
    # =========================

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # preprocessor is now a SINGLE object that knows:
    # - which columns are numeric
    # - which are categorical
    # - what to do with each

    # =========================
    # 8. APPLY PREPROCESSING (CORRECTLY)
    # =========================

    X_train_processed = preprocessor.fit_transform(X_train) ##Look at the training data 
    #learn how to scale and encode it, then apply those rules to it.
    X_test_processed = preprocessor.transform(X_test) ##Look at the testing data
    #apply the SAME rules to it (do NOT relearn from test data!)
    # =========================

    print("Train shape:", X_train_processed.shape) #just tells us how many rows and columns in training data after processing
    print("Test shape:", X_test_processed.shape)  #just tells us how many rows and colums in test data after processing 


if __name__ == "__main__":
    main()

## we do this so that it doesnt run on import, only runs when you run train.py. itd run on import otherwise
##which sucks 