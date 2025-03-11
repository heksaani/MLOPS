import pickle
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import pandas as pd

import great_expectations as gx
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import TargetEncoder

from thefuzz import fuzz, process

from great_expectations.data_context import FileDataContext
from great_expectations.datasource.fluent import BatchRequest
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult


# Random seed for making the assignments reproducible
RANDOM_SEED = 42



def file_reader(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read files of deals and house information into DataFrames
    Args:
        path (Path): Path to the folder where the files exist.
    Returns:
        a tuple consisting of a Pandas DataFrame of deals and a Pandas DataFrame of house information
    """
    
    price_file_path = path / "deals.csv"
    house_file_path = path / "house_info.json" 

    # Read the files into DataFrames
    deals = pd.read_csv(price_file_path)
    house_info = pd.read_json(house_file_path, orient="records")
    
    return deals, house_info

def dataframe_merger(prices: pd.DataFrame, house_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the two DataFrames given as inputs
    Args:
        prices (DataFrame): A pandas DataFrame holding the price information
        house_data (DataFrame): A pandas DataFrame holding the detailed information about the sold houses
    Returns:
        df: The merged pandas DataFrame
    """
    import pandas as pd
    # first convert the datesold column to datetime fromt the prices DataFrame
    prices["date"] = pd.to_datetime(prices["datesold"])

    #rename the building_year column in the price DataFrame to yr_built
    prices.rename(columns={"building_year": "yr_built"}, inplace=True)

    # merge the two DataFrames
    df = pd.merge(prices, house_data, how="inner", on=["postcode", "area", "date", "yr_built", "bedrooms"])

    return df

def drop_futile_columns(df: pd.DataFrame) -> None:
    """
    Removes unneeded columns from the argument DataFrame
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data
    """

    # drop the columns that are not needed ['yr_renovated', 'prev_owner', 'datesold', 'sqft_above']
    df.drop(columns=["yr_renovated", "prev_owner", "datesold", "sqft_above"], inplace=True)

def correct_distance_unit(df: pd.DataFrame) -> None:
    """
    Correct the falsely input values in column 'distance'
    Args:
        df (DataFrame): A Pandas DataFrame holding all of the housing data
    """

    # convert the distance unit from meters to kilometers
    # these rows are defined by ['distance']>=100
    df.loc[df["distance"] >= 100, "distance"] = df["distance"] / 1000

def string_transformer(df: pd.DataFrame) -> None:
    """
    Lowercases all values in the column 'condition' and removes trailing white space.
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data
    """
    # convert all values in the column 'condition' to lowercase and remove trailing white space
    df['condition'] = df['condition'].str.lower().str.strip()

def typo_fixer(df: pd.DataFrame, threshold: float, correct_condition_values: List[str]) -> None:
    """
    Uses fuzzy string matching to fix typos in the column 'condition'. It loops through each entry in
    the column and replaces them with suggested corrections if the similarity score is high enough. 
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data
        threshold (int): A number between 0-100. Only the entries with score above this number are replaced.
        correct_condition_values (List): A list of correct strings that we hope the condition column to include.
        For example, correct_ones=['excellent', 'good', 'satisfactory'] in the case of the training dataset.
    """
    scores = []
    conditions = []
    # fol loop over values in the column 'condition'
    for value in df['condition']:
        best_match, score = process.extractOne(value, correct_condition_values, scorer=fuzz.ratio)
        scores.append(score)
        if score >= threshold:
            conditions.append(best_match)
        else:
            conditions.append(value)
    df['condition'] = conditions
    df['similarity_scores'] = scores

def data_extraction(path: Path, correct_condition_values: List[str]) -> pd.DataFrame:
    """
    The entire data extraction/cleaning pipeline wrapped inside a single function.
    Args:
        path (Path): Path to the folder where the files exist.
        correct_condition_values (List): A list of correct strings that we hope the condition column to include.
    Returns:
        df (DataFrame): A pandas DataFrame holding all of the (cleaned) housing data.
    """
    threshold = 80
    prices, house_data = file_reader(path)
    df = dataframe_merger(prices, house_data)
    drop_futile_columns(df)
    correct_distance_unit(df)
    string_transformer(df)
    typo_fixer(df, threshold, correct_condition_values)
    return df

def batch_creator(df: pd.DataFrame, context: FileDataContext, data_source_name: str) -> BatchRequest:
    """
    Creates a new Batch Request using the given DataFrame
    Args:
        df (DataFrame): A pandas DataFrame holding the cleaned housing data. 
        context (GX FileDataContext): The current active GX Data Context 
        data_source_name (str): Name of the GX Data Source, to which the DataFrame is added
    Returns:
        new_batch_request (GX BatchRequest): The GX batch request created using df
    """
    datasource = context.get_datasource(data_source_name)
    test_asset_name = "test_data"
    try:
        #Create a new DataFrame Data Asset
        test_data_asset = datasource.add_dataframe_asset(name=test_asset_name)
    except ValueError:
        # The Data Asset already exists
        test_data_asset = datasource.get_asset(test_asset_name)
    
    # Single batch request for all data in the DataFrame
    test_data_request = test_data_asset.build_batch_request(dataframe=df)

    return test_data_request

def create_checkpoint(context: FileDataContext, batch_request: BatchRequest, checkpoint_name: str, expectation_suite_name: str, run_name: str) -> CheckpointResult:
    """
    Creates a new GX Checkpoint from the argument batch_request
    Args:
        context (GX FileDataContext): The current active context 
        new_batch_request (GX BatchRequest): A GX batch request used to create the Checkpoint
        checkpoint_name (str): Name of the Checkpoint
        expectation_suite_name (str): Name of the Expectation Suite used to validate the data
        run_name (str): Name of the validation running
    Returns:
        checkpoint_result (GX CheckpointResult): 
    """

    checkpoint = context.add_or_update_checkpoint(
        name = checkpoint_name,
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name,
    )
    # Run the Checkpoint
    checkpoint_result = checkpoint.run(run_name=run_name)
    return checkpoint_result

def separate_X_and_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the features and targets.
    Args:
        df (DataFrame): A pandas DataFrame holding the cleaned housing data
        target (str): Name of the target column
    Returns:
        X (DataFrame): A Pandas DataFrame holding the cleaned housing data without the target column
        y (Series): A pandas Series with the target values
    """
    df = df.copy()  # By copying the DataFrame we protect the original DataFrame in case we make errors in the following assignments. This way, we can always restart from this step.
    y = df[target].astype('float64')
    X = df.loc[:, df.columns != target]    
    return (X, y)

def impute_missing(df: pd.DataFrame) -> None:
    """
    Imputes missing numerical values using MICE.
    Args:
        df (DataFrame): A pandas DataFrame holding the features
    """
    imp = IterativeImputer(random_state=RANDOM_SEED, add_indicator=True) 

    # A list of column names where missing values need to be imputed.
    # We'll not include categorical or datetime variables in the calculations
    included_columns = [True if x not in ['postcode', 'area', 'date', 'condition'] else False for x in df.columns]
    df_to_impute = df.loc[:, included_columns]

    # Fit the imputer on the data using fit_transform() method
    imputed_array = imp.fit_transform(df_to_impute)

    # Convert the imputed array back to a DataFrame using get_feature_names_out() method
    feature_names = imp.get_feature_names_out(input_features=df_to_impute.columns)
    imputed_df = pd.DataFrame(imputed_array, columns=feature_names, index=df_to_impute.index)
    df[feature_names] = imputed_df

def datetime_decomposer(df: pd.DataFrame, dt_column_name: str) -> None:
    """
    Decomposes datetime values into year, quarter, and weekday.
    Args:
        df (DataFrame): A pandas DataFrame holding the features
        dt_column_name(str): The name of the datetime column
    """

    df["year"] = df[dt_column_name].dt.year
    df["quarter"] = df[dt_column_name].dt.quarter
    df["weekday"] = df[dt_column_name].dt.weekday

    df.drop(columns=dt_column_name, inplace=True)

def condition_encoder(df: pd.DataFrame) -> None:
    """
    Encodes conditions to numerical range 1-5
    Args:
        df (DataFrame): A pandas DataFrame holding the features
    """
    # using pandas Series map() method
    df['condition'] = df['condition'].map({'poor': 1, 'tolerable': 2, 'satisfactory': 3, 'good': 4, 'excellent': 5})

def target_encode(df: pd.DataFrame, columns: List[str], target: Optional[pd.Series]=None, encoder: Optional[TargetEncoder]=None) -> TargetEncoder:
    """
    Encodes postcode and area to numerical format using a target encoder
    Args:
        df (DataFrame): A pandas DataFrame holding the features
        columns (list of strings): Names of the categorical columns to be encoded
        target (Series|None): A pandas Series with the target values. This is required only when fitting the encoder.
        encoder(TargetEncoder|None): An already fitted encoder. This is required when we want to apply an encoder, which has already been fitted during training.
    Returns:
        encoder(TargetEncoder): The fitted TargetEncoder. Either a new fitted one or the one passed as an argument.
    """
    if encoder is None:
        encoder = TargetEncoder(target_type='continuous', smooth='auto', random_state=RANDOM_SEED)
    # If the target column is specified, the methods `fit` and `transform` should both be used (separately, don't use `fit_transform` to disable cross-fitting). 
    if target is not None:
        encoder.fit(df[columns], target)
    
    df[columns] = encoder.transform(df[columns])

    return encoder 

def store_features(X: pd.DataFrame, feature_file_path: str) -> None:    
    """
    Stores a set of features to a specified location
    Args:
        X (DataFrame): A pandas DataFrame holding the features
        feature_file_path (Path): Path for the stored features, e.g., feature_store/housing_train_X.parquet
    """
    X.to_parquet(feature_file_path)
    print(f"Features stored in {feature_file_path}")



def store_targets(y: pd.Series, target_file_path: Path) -> None:
    """
    Stores a set of features to a specified location
    Args:
        y (Series): A pandas Series holding the target values
        target_file_path (Path): Path for the stored targets, e.g., feature_store/housing_train_y.csv
    """
    y.to_csv(target_file_path, index=False)
    print(f"Targets stored in {target_file_path}")




def store_encoder(encoder: TargetEncoder, encoder_file_path: Path) -> None:
    """
    Stores a targetEncoder to a specified location
    Args:
        encoder (TargetEncoder): A fitted scikit-learn TargetEncoder object
        encoder_file_path (Path): Path of the stored target encoder.
    """
    with open(encoder_file_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Encoder stored in {encoder_file_path}")

def drop_unimportant_features(df: pd.DataFrame):
    """
    df (DataFrame): DataFrame from which the unimportant features should be dropped.
    """
    unimportant_features = ['floors', 'similarity_scores', 'missingindicator_sqft_living15', 'missingindicator_sqft_lot15', 'quarter', 'weekday']
    df.drop(columns=unimportant_features, inplace=True)

def feature_engineering_pipeline(df: pd.DataFrame, 
                                    feature_store_path: Path,
                                    feature_file_name: str, 
                                    encoder_file_name: str, 
                                    target_file_name: Optional[str]=None, 
                                    fit_encoder: bool=False, 
                                    targets_included: bool=True) -> None:
    """
    Converts a given (merged) housing data DataFrame into features and targets, performs feature engineering, and 
    stores the features along with possible targets and a fitted encoder
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data, or just the features (see targets_included)
        feature_store_path (Path): Path of the feature store
        feature_file_name (str): Filename for the stored features.
        encoder_file_name (str): Filename for the stored encoder.
        target_file_name (str|None): Filename for the stored targets.
        fit_encoder (bool): Whether a new target encoder should be fitted. If False, uses a previously stored encoder
        targets_included (bool):  If True, df has all of the housing data including targets. If False, df has only the features.
    """
    if targets_included:
        X, y = separate_X_and_y(df, target='price')   
    else:
        if fit_encoder:
            raise ValueError("Target encoder can not be trained without targets.")
        X = df.copy()

    impute_missing(X)
    datetime_decomposer(X, dt_column_name='date')
    condition_encoder(X)
    drop_unimportant_features(X)

    feature_file_path = feature_store_path / feature_file_name
    target_file_path = feature_store_path / target_file_name
    encoder_file_path = feature_store_path / "encoders" / encoder_file_name

    if fit_encoder:
        t_encoder = target_encode(X, columns=['postcode', 'area'], target=y)
        store_features(X, feature_file_path)
        store_targets(y, target_file_path)
        store_encoder(t_encoder, encoder_file_path)

    else:
        with open(encoder_file_path, 'rb') as enc_f:
            t_encoder = pickle.load(enc_f)
        target_encode(X, columns=['postcode', 'area'], encoder=t_encoder)
        store_features(X, feature_file_path)    
        if targets_included:
            store_targets(y, target_file_path)

def etl(path: Path, 
        gx_context_root_dir: Path, 
        gx_datasource_name: str, 
        gx_checkpoint_name: str,
        gx_expectation_suite_name: str,
        gx_run_name: str,
        feature_store_path: Path, 
        feature_file_name: str,
        encoder_file_name: str, 
        target_file_name: Optional[str]=None, 
        fit_encoder: bool=False, 
        targets_included: bool=True
    ):
    """
    This function loads, merges, cleans, and validates the specified data, extract features, and save the features in the feature store.
    Args:
        path (Path): Path to the folder where the files "deals.csv" and "housing_info.json" exist
        gx_context_root_dir (Path): The directory containing the Great Expectations configs.
        gx_datasource_name (str): Great Expectations data source name
        gx_checkpoint_name (str): Name of the Great Expectations checkpoint that runs the validation
        gx_expectation_suite_name (str): Name of the Expectation Suite used to validate the data
        gx_run_name (str): Name of the validation running
        feature_store_path (Path): Path of the feature store
        feature_file_name (str): Filename for the stored features
        encoder_file_name (str): Filename for the stored target encoder
        target_file_name (None|str): Filename for the stored targets
        fit_encoder (bool): Whether a new target encoder should be fitted. If False, uses a previously stored encoder
        targets_included (bool): If True, df has all of the housing data including targets. If False, df has only the features.
    """
    correct_condition_values=['poor', 'tolerable', 'satisfactory', 'good', 'excellent']
    context = gx.get_context(context_root_dir=gx_context_root_dir)
    df = data_extraction(path, correct_condition_values=correct_condition_values)

    # Create the batch request and checkpoint
    context = gx.get_context(gx_context_root_dir)
    test_batch_request = batch_creator(
        df=df,
        context=context,
        data_source_name=gx_datasource_name)

    test_data_checkpoint_result = create_checkpoint(context=context,
                                                    batch_request=test_batch_request,
                                                    checkpoint_name=gx_checkpoint_name,
                                                    expectation_suite_name=gx_expectation_suite_name,
                                                    run_name=gx_run_name)

    if not test_data_checkpoint_result.success:
        print("Some GX validations failed.")

    feature_engineering_pipeline(df, feature_store_path, feature_file_name, encoder_file_name, target_file_name, fit_encoder, targets_included)



