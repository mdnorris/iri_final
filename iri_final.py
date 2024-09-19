import streamlit as st
import pandas as pd
import os
import re
import shutil
import tempfile
from openai import AzureOpenAI
from io import StringIO
import tiktoken
# from azure.identity import DefaultAzureCredential, get_bearer_token_provider
# import time
import itertools
import sys
# from jarowinkler import jarowinkler_similarity
import numpy as np

# from datawig import SimpleImputer
# from mxnet.base import MXNetError
# testing automatic push
# Set up tiktoken encoding
encoding = tiktoken.encoding_for_model("gpt-4o")

# Set the streamlit page to wide format for easier viewing
st.set_page_config(layout="wide")

# token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

# API info needed to call on OpenAI model
deployment_name = 'gpt-4o-2'

# API info needed to call on OpenAI model
client = AzureOpenAI(
    api_version='2024-06-01',  # This is a version that allows you to ask for logprobs
    azure_endpoint='https://adv-datascience-west.openai.azure.com/',
    api_key='d56952cf138e4c09a5f113682ce1b540'
)


def column_Mapping_backend(column_mapping):
    data_out = pd.DataFrame()
    retries = 3
    while retries > 0:
        try:
            column_mapping_copy = column_mapping.copy()
            column_mapping_copy_str = column_mapping_copy.astype('str')
            prompt = (
                f"The following dataset contains a list of column names:\n\n{column_mapping_copy_str}  "
                f"Some column names with the ADV_ prefix are matching columns to the columns that do not contain the ADV_ prefix.  "
                f"Can you attempt to pair up the ADV_ prefixed columns with the non ADV_ columns?  Please format your response into tabular dataset that will be easy to read back in python."
                f"Please only respond with the completed table, with no other commentary. ")

            response = client.chat.completions.create(
                model='gpt-4o-2',
                # engine = deployment_name,
                max_tokens=1500,
                temperature=1,
                messages=[
                    {"role": "system",
                     "content": "You are a model that is used to assist in analyzing datasets."},
                    {"role": "user", "content": prompt}
                ]
            )
            data_out = response.choices[0].message.content
            break
        except Exception as e:
            print(e)
            retries -= 1
            if retries > 0:
                print('Timeout error, retrying...')
                # time.sleep(5)
        else:
            print('API is not responding, all retries exhausted. Raising exception...')
            raise ValueError(
                "Time out error - try restarting the script and checking your connection to OpenAI Studio")

    # rows = data_out.split('\n')
    # data_out_series = pd.Series(rows[1:])

    # Cleaning the string to make it a valid CSV
    data = data_out.replace("|", "").strip()
    data = "\n".join([line.strip() for line in data.split("\n") if "---" not in line])

    # Use StringIO to read the string as a CSV
    df = pd.read_csv(StringIO(data), sep="\s\s+", engine='python')

    # Convert the list of lists into a Pandas DataFrame
    # data_out_series = pd.DataFrame(data[1:], columns=data[0])

    # data_out_series = pd.Series(data_out)
    df_output = df

    return df_output


# Secondary backend function for additional imputation needs
# OpenAI is used here
def backend_plus(df_input, column_mapping, issues=['?', 'NA', '-']):
    """
    Adds nulls where user-defined or ai detected issues are found
    Once these are added, they are imputed away using OpenAI imputation

    Parameters
    ----------
    df_input : Pandas dataframe. Data to be imputed in.
    issues : List of data issues found by the user or AI assistant. Hardcode these for now

    Returns
    -------
    Dataframe with no null values or data inconsistencies

    """
    # Now that the original nulls are filled, create new nulls where there are data issues like missing categories
    # Do not attempt to fill UPC or Item Code columns if there are issues or if
    # the issue is in a Circana-sourced column
    # Iterate over each column in the DataFrame

    # TESTING - print out df_input's data issues before they get nulled out
    # print(df_input.loc[df_input['Category Name'] == '?'])

    for column in df_input.columns:
        # Replace data issues if they're not in the excluded columns
        if "UPC" not in column and "ADV_" in column:
            df_input[column] = df_input[column].replace(to_replace=issues, value=np.nan)

    # If there are no issues present, then skip the rest of the function and return the df
    if df_input.isnull().sum().sum() == 0:
        print(
            "No data issues were detected. All null values have been filled using column mapping.")
        return df_input, None  # Return no metrics if there were no addtional imputations performed

    # Reset index to hopefully avoid issues and create an additional index column
    df_input.reset_index(inplace=True, drop=False)

    # TESTING - print out df_input nulls
    # print(df_input[df_input.isna().any(axis=1)])

    # Model prompt - use the column mapping from the frontend instead of manually writing the string out and pull in the entire dataset post column mapping
    prompt = f"""
    The following data is a sample of a dataset of product attributes called the AWG Item List: {df_input[df_input.notna().all(axis=1)].head(50).to_string(sparsify=False, justify="center")} 
    The missing item attribution values need to be filled in. Here's how the columns map to each other:
    {column_mapping.to_string(sparsify=False, justify="center")}

    The dataset contained several issues which have been replaced with nulls. Here are the rows that contain null values:
    {df_input[df_input.isna().any(axis=1)].to_string(sparsify=False, justify="center")}

    Impute these null values using the mapping formula above and the sample data as context and format your answer into a table using | to separate columns with no leading or trailing whitespaces. Do not just put NaN in the empty cells or placeholder values.
    Only output the rows that contained null values and do not drop the index column. Start your response as a table where the first column is the 'index' column.
    Only impute nulls and data issues in the columns that start with the prefix ADV_
    Include the logprobs for the tokens imputed in your response - do not make up hypothetical values and do not convert these values to probabilities. Format these logprobs as a table consisting of only these 3 columns - index, imputed field, and logprob. 
    The imputed field column should contain the name of the column the imputed value was under, and the index should contain the index number of the imputed value. 
    Start this section of your response with the heading "### Logprobs Table". Do not add any additional columns to this Logprobs Table.
    """
    # Followup prompt in case the output token limit is hit by the initial prompt
    # Should be a rare occurence
    followup_prompt = """
    Continue the output from the previous prompt.
    Include the logprobs for the tokens imputed in the model response - do not make up hypothetical values.
    Format these logprobs as a table consisting of the 3 columns - index, imputed field, and logprob. Start this section of your response with the heading "### Logprobs Table". Do not add any additional columns to this Logprobs Table.
    """
    # Main prompt loop - repeat until data quality is assured
    complete = False
    while complete == False:
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=1.1,
            # Try raising this to suppress the "I can't generate data" response. Don't tune the top_p param while tuning this one
            logprobs=True,
            # Can't get the logprobs for this imputation from a followup, so just inlcude them in the same call
            # seed=42, # Attempt to rein in unpredictability using a random seed
            messages=[
                {"role": "system",
                 "content": "You are a model used to impute missing data."},
                # Try out different language to avoid imputation tactics on the part of the model
                {"role": "user", "content": prompt}
            ]

        )

        # Process output from AI
        output = response.choices[0].message.content

        # Make sure output isn't complete nonsense
        if '### Logprobs Table' not in output or '| index |' not in output:
            continue  # Restart the while loop to call the api and try again

        # Make sure there aren't null values still in the output
        # This will prevent StringIO errors from being thrown
        if any(issue in output for issue in ['N/A', 'n/a', 'null', 'NaN', 'NAN']):
            continue

        # If the model refuses to work for whatever reason - restart as well
        if 'cannot be provided' in output:
            continue

        # Make sure the output isn't just empty rows
        if output.find('0') == -1:
            continue

        # TESTING - print output to terminal
        # print(output)

        # Split the string so the metric extraction doesn't interfere with imputation extraction
        output_strings = output.split('### Logprobs Table', maxsplit=1)

        # TESTING - print split outputs
        # print(f"first string:  {output_strings[0]} ------------ second string: {output_strings[1]}")

        # Now pull the metrics out of the appropiate output string and get it into a useful format using StringIO again
        output_start_metrics = output_strings[1].find('| index |')
        output_end_metrics = output_strings[1].rfind('|')

        # StringIO converts the trimmed response string to a file pandas read_csv can convert to a dataframe
        # Add try-except logic here to try the api call again in case of unpredictable results in getting logprobs
        try:
            output_df_metrics = pd.read_csv(
                StringIO(output_strings[1][output_start_metrics:output_end_metrics]),
                delimiter='|', index_col=[0, -1], skiprows=[
                    1])  # Skip the first row where the model keeps putting dash marks, and the last row to avoid duplicates

            # Strip the whitespaces from the metrics column names
            output_df_metrics.rename(columns=lambda x: x.strip(), inplace=True)

            # Add a column where the logprobs are converted to probabilities
            output_df_metrics['prob'] = round(
                np.exp(output_df_metrics['logprob'].astype(float)), 2)
        except:
            continue

        # Check the token length of the output
        # If the limit of 5000 tokens was hit, use the followup prompt to complete the output
        if len(encoding.encode(output)) >= 5000:
            print(
                f'Token limit hit ({encoding.encode(output)} > 5000). Followup response:')
            response_followup = client.chat.completions.create(
                model=deployment_name,
                temperature=1.1,
                # Try raising this to suppress the "I can't generate data" response. Don't tune the top_p param while tuning this one
                messages=[
                    {"role": "system",
                     "content": "You are a model used to impute missing data."},
                    # Try out different language to avoid imputation tactics on the part of the model
                    {"role": "user", "content": followup_prompt}
                ]

            )
            # Process output from AI
            # First, split the string so the metric extraction doesn't interfere with imputation extraction
            output_followup = response_followup.choices[0].message.content
            output_strings_followup = output_followup.split('### Logprobs Table',
                                                            maxsplit=1)

            # Now pull the metrics out of the appropiate output string and get it into a useful format using StringIO again
            # Again, use the try-except logic here too
            try:
                output_start_metrics_followup = output_strings_followup[1].find(
                    '| index |')
                output_end_metrics_followup = output_strings_followup[1].rfind('|')

                # StringIO converts the trimmed response string to a file pandas read_csv can convert to a dataframe
                output_df_metrics_followup = pd.read_csv(
                    StringIO(output_strings_followup[1][
                             output_start_metrics_followup:output_end_metrics_followup]),
                    delimiter='|', index_col=[0, -1], skiprows=[
                        1])  # Skip the first row where the model keeps putting dash marks, and the last row to avoid duplicates

                # Strip the whitespaces from the metrics column names
                output_df_metrics_followup.rename(columns=lambda x: x.strip(),
                                                  inplace=True)

                # Add a column where the logprobs are converted to probabilities
                output_df_metrics_followup['prob'] = round(
                    np.exp(output_df_metrics_followup['logprob'].astype(float)),
                    2)
            except:
                continue

            # Paste the trimmed output from the followup into the main output string
            output_followup_cleaned = re.search(pattern='| index |(.*)| INCLUDE    |',
                                                string=output_followup).group(1)
            output = output + output_followup_cleaned

            # Append the additional metrics data to the main metrics table
            pd.concat([output_df_metrics, output_df_metrics_followup])

        # Trim out the non-tabular data
        output_start = output_strings[0].find('| index |')
        output_end = output_strings[0].rfind('|INCLUDE    |')

        # StringIO converts the trimmed response string to a file pandas read_csv can convert to a dataframe
        df_response = pd.read_csv(StringIO(output_strings[0][output_start:output_end]),
                                  delimiter='|',
                                  index_col=[0, -1], skiprows=[
                1])  # Skip the first row where the model keeps putting dash marks, and the last row to avoid duplicates

        # Check if the output has no nulls. If so, redo the api call
        if df_response.isnull().values.any():
            continue
        else:
            complete = True

    # Merge the response df with the input data to remove nulls. First, do some cleanup
    df_response = df_response.rename(columns=lambda x: x.strip())
    df_response.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df_response.dropna(axis=0, inplace=True, thresh=2)
    df_response.dropna(axis=1, inplace=True)
    df_response['index'] = df_response['index'].astype('int64')
    # df_response['Item Code'] = df_response['Item Code'].astype('int64')
    df_response.reset_index(inplace=True, drop=True)
    df_response.set_index(df_response['index'], inplace=True)

    # TESTING - print the response df after cleaning
    # print(df_response)

    # Then update and return filled data
    df_output = df_input.combine_first(df_response)
    # df_output = df_input.copy()

    # Finally, drop the index column and return the data plus the metrics
    df_output.drop(columns='index', inplace=True)
    return df_output, output_df_metrics


# Function for calling the backend code
# Input csv will be df_input after adding nulls
def backend_main(df_input, column_mapping):
    """
    Backend code for the Item Recommender placed into a single function for easy insertion into frontend code.
    df_input should be the raw data directly from the user input on the frontend
    column_mapping is the user-inputted mappings converted into a Pandas dataframe
    -------
    Returns:
    dataframe with no null values
    """
    # Map the source columns to their intended target columns using frontend column mapping
    for ind in column_mapping.index:
        df_input[column_mapping['To'][ind]] = df_input[column_mapping['To'][ind]].fillna(
            df_input[column_mapping['From'][ind]])

    df_output = df_input.copy()

    return df_output


##### FRONT END CODE #####
# creates a session state for the column pairs
if 'columnsList' not in st.session_state:
    st.session_state["columnsList"] = []

# creates a session state for buttons, allows them to be grayed out so they can't be used once imputation has started
if "button1" not in st.session_state:
    st.session_state.button1 = False

if "button2" not in st.session_state:
    st.session_state.button2 = False

if "button3" not in st.session_state:
    st.session_state.button3 = False

if 'columnMapping_AI' not in st.session_state:
    st.session_state.columnMapping_AI = True

if 'columnsList_AI' not in st.session_state:
    st.session_state["columnsList_AI"] = []

# Flag to indicate if imputation has started
imputation_preview = False
imputation_preview_button_flag = False
imputation_started = False


# this callback unfreezes the Add Columns to SAved Column Pairs? button in the event that someone clicks the reset saved column pairs button
def reset_saved_columns_pair_callback():
    st.session_state.button2 = False  # not st.session_state.button2


# this callback unfreezes all the buttons
def cancel_callback():
    st.session_state.button1 = not st.session_state.button1


# this callback allows the user to select yes in the imputation process
# def yes_callback():
#   imputation_started = True


# Function to rename the file with "_new" added to the filename
def rename_file(filename):
    root, ext = os.path.splitext(filename)
    return root + '_new' + ext


# Frontend app implementation - functionalize the page setup
def frontend_main():
    """
    Function that contains most of the frontend code outside of some helper functions.

    Returns
    -------
    None. Only runs the code to set up the web page.
    """
    global imputation_started
    # global column_mapping_OpenAI_Flag
    # Streamlit app with the new title
    st.title("Item Auto Attribution Tool")

    # File upload widget
    uploaded_file = st.file_uploader("**Upload a CSV file**", type=["csv"],
                                     disabled=st.session_state.button1)  # , on_change=saved_columns.clear())

    # Initialize an empty dataframe to store saved columns
    saved_columns = []
    saved_template_columns = []
    columnsDisplayed = []
    savedColumnsDisplayed = []
    inputDF_Nulls = []
    columnsDisplayed_openAI = []

    # Initialize the instruction text block so it can be deleted after imputation starts
    instructions = st.empty()

    # Fill the empty object with text
    instructions.markdown("""
                    ## Getting Started
                    Welcome to the Item Recommender application homepage. To use this application, follow these steps:
                    1. Upload a file to the application by dragging one to the "Drag and drop" section above or by hitting the "Browse files" button.
                    2. Check the auto-generated column mappings on the left side of the screen. These will tell the program how to fill in nulls in the ADV columns based on the Circana columns.
                    3. Hit the "Add these Pairs to the Saved Column Pairs?" button and check under the Saved Column Pairs section to make sure the column pairs are correct.
                    4. If the pairs are not correct, click the checkbox column on the left to delete a row or click in a cell to alter the text.
                    5. If needed, you can download the column mapping to use again later using the "Download Column Mapping Template" button.
                    6. Hit the "Begin Imputation" button to get started!
                    """)

    finalDF = pd.DataFrame()
    # df_input = pd.read_csv(uploaded_file, header=0)
    # finalDF_Nulls = pd.DataFrame()

    # Once a file has been uploaded, triggers this portion of the code
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file, header=0)  # Try specifying header=0
        finalDF = df_input

        df_input_copy = df_input.copy()
        df_input_copy.replace(['NA', 'N/A', 'n/a', 'null', np.nan, 'NAN'], 'NaN',
                              inplace=True)

        inputDF_Nulls = df_input.isna()

        # finalDF_withFormatting = finalDF

        def color_null(val):
            background_color = '#d65f5f' if val == 'NaN' else 'white'
            color = 'white' if val == 'NaN' else 'black'
            return f"background-color: {background_color}; color: {color}"

        with st.expander("**Imported Dataset**"):
            st.dataframe(df_input_copy.style.applymap(color_null), column_config={
                "Item Code": st.column_config.TextColumn(),
                "UPCItem": st.column_config.TextColumn(),
                "UPCCase": st.column_config.TextColumn(),
                "ADV_ItemUPC": st.column_config.TextColumn(),
                "ADV_CaseUPC10": st.column_config.TextColumn(),
            },
                         hide_index=True,
                         )

        # st.session_state["columnsList"].clear()
        columnsDisplayed = pd.DataFrame(df_input.columns.tolist())
        # if (column_mapping_OpenAI_Flag):

        expander = st.sidebar.expander("**Column Names Detected**", expanded=False)
        expander.table(columnsDisplayed)

        if (len(st.session_state["columnsList_AI"]) == 0):
            st.session_state["columnMapping_AI"] = True

        if (st.session_state["columnMapping_AI"]):
            # st.session_state["columnsList_AI"] = column_Mapping_backend(columnsDisplayed)
            columnsDisplayed_openAI = column_Mapping_backend(columnsDisplayed)
            st.session_state["columnsList_AI"] = columnsDisplayed_openAI
            st.session_state["columnMapping_AI"] = False

        if (st.session_state["columnMapping_AI"] is not None):
            st.sidebar.write(
                "**Below are the Column Mappings that were Auto Generated.**")
            st.sidebar.write(st.session_state["columnsList_AI"])

            # saved_columns.append(df_template_input)
            if (st.sidebar.button("**Add these Pairs to the Saved Column Pairs?**",
                                  disabled=st.session_state.button2)):
                st.sidebar.write("If any are not correct, you can remove them below.")
                # st.session_state.button2 = True
                for i in st.session_state["columnsList_AI"].index:
                    saved_template_columns.append(
                        [st.session_state["columnsList_AI"].iloc[i, 0],
                         st.session_state["columnsList_AI"].iloc[i, 1]])
                    st.session_state["columnsList"].append(
                        [st.session_state["columnsList_AI"].iloc[i, 0],
                         st.session_state["columnsList_AI"].iloc[i, 1]])
                st.sidebar.success("**Selected columns saved!**")

        # Check if the DataFrame is not empty and contains columns
        if not df_input.empty:
            # Add a "Begin Imputation" button
            if st.button("**Begin Imputation**", key=1,
                         disabled=st.session_state.button1):  # , on_click=btn1_callback()):
                imputation_started = True
                st.session_state.button1 = True
                st.session_state.button2 = True
            else:
                st.session_state.button1 = False

            columnsDisplayed.columns = ["Column Names"]

            # section where the user can upload a template file of column mappings
            template_file = st.sidebar.file_uploader("**Upload a template file**",
                                                     type=["csv"],
                                                     disabled=st.session_state.button1)  # , on_change=saved_columns.clear())
            if (template_file is not None):
                df_template_input = pd.read_csv(template_file, header=0)
                st.sidebar.write(df_template_input)

                # saved_columns.append(df_template_input)
                if (st.sidebar.button("**Add columns to Saved Column Pairs?**",
                                      disabled=st.session_state.button2)):
                    # st.session_state.button2 = True
                    for i in df_template_input.index:
                        saved_template_columns.append(
                            [df_template_input.iloc[i, 0], df_template_input.iloc[i, 1]])
                        st.session_state["columnsList"].append(
                            [df_template_input.iloc[i, 0], df_template_input.iloc[i, 1]])
                    st.sidebar.success("**Selected columns saved!**")

            # Add two dropdown lists for column selection
            st.sidebar.header("Select Columns")
            column1 = st.sidebar.selectbox("**Select First Column**",
                                           df_input.columns.tolist(), key="column1",
                                           disabled=st.session_state.button1)
            column2 = st.sidebar.selectbox("**Select Second Column**",
                                           df_input.columns.tolist(), key="column2",
                                           disabled=st.session_state.button1)

            # Add a button to save selected columns
            if st.sidebar.button("**Save Selected Columns**", key=3,
                                 disabled=st.session_state.button1):
                if column1 == column2:
                    st.sidebar.warning(
                        "The columns selected must be different. Try again.")
                else:
                    # Save the selected columns
                    saved_columns.append([column1, column2])
                    st.session_state["columnsList"].append([column1, column2])
                    st.sidebar.success("**Selected columns saved!**")

            # button to allow the user to reset the saved column pairs
            if st.sidebar.button('Reset Saved Column Pairs', key=4,
                                 disabled=st.session_state.button1,
                                 on_click=reset_saved_columns_pair_callback()):
                saved_columns.clear()
                # st.session_state.button2 = False
                st.session_state["columnsList"].clear()

            # makes sure there are a unique set of column pairs for the user
            st.session_state["columnsList"].sort()
            st.session_state["columnsList"] = list(
                k for k, _ in itertools.groupby(st.session_state["columnsList"]))

            # Display saved columns at the bottom
            st.sidebar.subheader("**Saved Column Pairs**")
            savedColumnsDisplayed = pd.DataFrame(st.session_state["columnsList"])

            if (len(savedColumnsDisplayed) > 0):
                savedColumnsDisplayed.columns = ['From', 'To']
                # st.sidebar.data_editor(savedColumnsDisplayed, num_rows= "dynamic")
                savedColumnsDisplayed = st.sidebar.data_editor(savedColumnsDisplayed,
                                                               num_rows="dynamic")
                # st.sidebar.dataframe(savedColumnsDisplayed)
                # st.sidebar.dataframe(savedColumnsDisplayed)

                # st.session_state["columnsList"] =

                # Provide a download link for the column mapping
                st.sidebar.download_button(
                    label="**Download Column Mapping Template**",
                    data=savedColumnsDisplayed.to_csv(index=False),
                    file_name=f"column_mapping_template_{uploaded_file.name}",
                    key="download_button_cm",
                )

        else:
            st.warning("**The uploaded CSV file is empty or has no columns.**")

    # Check if imputation has started
    if imputation_started:
        # Perform imputation or any desired action here
        # Remove the intro instruction guide now
        instructions.empty()

        originalDS = df_input

        ##pulling in the columns pairs list from the front end section above
        savedColumnMapping = pd.DataFrame(st.session_state["columnsList"])
        # st.dataframe(savedColumnsDisplayed)

        # cancels imputation and let's the user start over
        if (st.button('**Cancel**', on_click=cancel_callback())):
            st.session_state.button1 = False
            # kills the imputation process
            # However, you'll see the "Imputed in.." message pop up again once
            # This is because the API call already went out
            sys.exit('User canceled imputation. Resetting...')

        # Imputation process Kicks off
        # Prints a message letting the user know imputation has started
        with st.spinner("**Imputing...**"):
            # Column mapping imputation
            finalOutput = backend_main(df_input=finalDF,
                                       column_mapping=savedColumnsDisplayed)

            # OpenAI Imputation
            finalOutput, finalOutputMetrics = backend_plus(df_input=finalOutput,
                                                           column_mapping=savedColumnsDisplayed)

            # kick off the preview section
            # st.write("Here is a quick preview of what the results will look like when finished:")
            # st.write("Do you want to continue?")

            # kicks off the full imputation
            # if (st.button("Yes", on_click=yes_callback())):
            #   st.session_state.button1 = True

            # kicks off the full imputation
            #  with st.spinner("**Imputation process started....**"):
            #     time.sleep(5)
            st.success('Imputation Complete!')
        originalDS = originalDS.where(~inputDF_Nulls)

        # Create a temporary directory to save the renamed file
        temp_dir = tempfile.mkdtemp()
        new_filename = os.path.join(temp_dir, rename_file(uploaded_file.name))

        # Save the renamed file to the temporary directory
        finalOutput.to_csv(new_filename, index=False)

        finalOutput["Item Code"] = finalOutput["Item Code"].astype('int')

        finalOutputMerged = pd.merge(originalDS, finalOutput, how="left",
                                     on=["Item Code"])

        st.success(f"**File saved as {new_filename}**")

        # Provide a download link for the new file
        st.download_button(
            label="**Download Processed File**",
            data=open(new_filename, "rb").read(),
            file_name=os.path.basename(new_filename),
            key="download_button",
        )

        ## Display a color coding legend for the completed data display
        st.markdown('### Color Coding Key')
        ## st.image("https://gist.github.com/mdnorris/865d30dfc324cea5a7cc44260d05751f.js")
        st.text("Blue good, green pretty good, yellow iffy, red not to be trusted")

        # Converts the metrics table to a dataframe
        # st.dataframe(finalOutputMetrics)
        finalOutputMetrics['imputed field'] = finalOutputMetrics[
            'imputed field'].str.strip()

        # function to apply the coloring for the metrics
        def apply_background_color(row, formatting_df):
            color = [''] * len(row)  # Default no color
            for _, fmt_row in formatting_df.iterrows():
                if row.name == fmt_row["index"] and (fmt_row["prob"] >= 0.90):
                    col_idx = row.index.get_loc(fmt_row["imputed field"])
                    color[col_idx] = 'background-color: lightgreen'
                elif row.name == fmt_row["index"] and (
                        fmt_row["prob"] >= 0.75 and fmt_row["prob"] < 0.90):
                    col_idx = row.index.get_loc(fmt_row["imputed field"])
                    color[col_idx] = 'background-color: lightyellow'
                elif row.name == fmt_row["index"] and fmt_row["prob"] < 0.75:
                    col_idx = row.index.get_loc(fmt_row["imputed field"])
                    color[col_idx] = 'background-color: lightcoral'
            return color

        # function to apply the coloring for the metrics detail section
        def apply_background_color_accuracy_data(value):
            if value == 0:
                color = 'background-color: darkgray; color: darkgray'  # Hide these cells
            elif value < 75:
                color = 'background-color: lightcoral'
            elif 75 <= value < 90:
                color = 'background-color: lightyellow'
            elif 90 <= value < 100:
                color = 'background-color: lightgreen'
            elif value == 100:
                color = 'background-color: lightblue'
            else:
                color = 'background-color: lightgreen'  # If the model returns values above 100% (rare), return light green coloring
            return color

        ## Builds out the pandas data frame that identifies where the values were null and have been filled in
        inputDFNulls = inputDF_Nulls
        # inputDFNulls.replace(['?', 'NA', '-','N/A', 'n/a', 'null', 'NaN', 'NAN'], np.nan, inplace = True)
        finalDFNulls = finalOutput.isna()
        finalDFNullDiff = inputDFNulls.astype(int) - finalDFNulls.astype(int)

        # create a copy of finalDFNullDiff so we can insert the accuracy metrics into it
        df_accuracy_metrics = finalDFNullDiff
        # Iterate over the rows of df_accuracy
        for _, row in finalOutputMetrics.iterrows():
            # Extract the index, imputed field (column name), and accuracy value
            index_value = row["index"]
            column_name = row["imputed field"]
            accuracy_value = row["prob"]

            # Set the accuracy value in the original dataframe
            df_accuracy_metrics.at[index_value, column_name] = accuracy_value

        finalOutputDisplayed = finalOutput

        # Function to apply the light green background for the missing values that were filled in
        def apply_styles(df, mask):
            # Create a styled DataFrame by copying the styles from the mask
            style = pd.DataFrame('', index=df.index,
                                 columns=df.columns)  # Initialize an empty style DataFrame
            for col in df.columns:
                style[col] = np.where(mask[col] == 1, 'background-color: lightblue', '')
            return style

        # Apply the style to the pandas dataframe
        # This should be ok to leave in since most item attribution data will have this feature
        # finalOutputDisplayed['ADV_ItemUPC'] = finalOutputDisplayed['ADV_ItemUPC'].astype(str)
        # finalOutputDisplayed.style.apply(lambda x: apply_styles(finalOutputDisplayed, finalDFNullDiff),
        #                                             axis=None)
        # Applies the coloring to the metrics table
        # finalOutputDisplayed.style.apply(lambda row: apply_background_color(row, finalOutputMetrics), axis=1)

        # Apply both styles to the DataFrame and create a new styled df to display
        styled_df = finalOutputDisplayed.style \
            .apply(lambda x: apply_styles(finalOutputDisplayed, finalDFNullDiff),
                   axis=None) \
            .apply(lambda row: apply_background_color(row, finalOutputMetrics), axis=1)

        # st.dataframe(df_accuracy_metrics)

        # ADV_Brand_Accuracy_test = df_accuracy_metrics["ADV_Brand"][df_accuracy_metrics["ADV_Brand"] != 0]
        # ADV_Brand_Accuracy_test_value = ADV_Brand_Accuracy_test.mean()
        # st.write(ADV_Brand_Accuracy_test_value)
        # st.dataframe(inputDFNulls)

        ## This section outputs the final model data with the missing values filled in and colored green in the background to highlight where the data was filled in
        # Same here - but maybe add a try/except block in the future just in case
        with st.expander("**Completed Model Data**"):
            st.dataframe(styled_df, column_config={
                "Item Code": st.column_config.TextColumn(),
                "UPCItem": st.column_config.TextColumn(),
                "UPCCase": st.column_config.TextColumn(),
                "ADV_ItemUPC": st.column_config.TextColumn(),
                "ADV_CaseUPC10": st.column_config.TextColumn(),
            },
                         hide_index=True,
                         )

        ## This section outputs the final model accuracy metrics data with the missing values filled in and colored green in the background to highlight where the data was filled in
        # Same here - but maybe add a try/except block in the future just in case
        df_accuracy_metrics_as_percents = df_accuracy_metrics * 100

        # Color code this dataframe as well, just like the one above
        df_accuracy_metrics_as_percents_output = df_accuracy_metrics_as_percents.style.applymap(
            apply_background_color_accuracy_data)

        # df_accuracy_metrics_as_percents_output = df_accuracy_metrics_as_percents.applymap(lambda x: f'{int(x)}%' if x != 0 else '--') # Don't display untouched data as zero to avoid confusion

        with st.expander("**Model Accuracy Data**"):
            st.dataframe(df_accuracy_metrics_as_percents_output, column_config={
                "Item Code": st.column_config.TextColumn(),
                "UPCItem": st.column_config.TextColumn(),
                "UPCCase": st.column_config.TextColumn(),
                "ADV_ItemUPC": st.column_config.TextColumn(),
                "ADV_CaseUPC10": st.column_config.TextColumn(),
            },
                         hide_index=False,
                         )

        ## Calculate the accuracy metrics dynamically using Jaro-Winkler text similarity
        ##Should probably keep this jaro winkler stuff in case we have to revert back to it.
        # def calculate_Jaro(x, y):
        #   return jarowinkler_similarity(str(x), str(y))

        # finalAccuracyMetrics = pd.DataFrame()

        # Function to calculate mean excluding zeros
        def mean_excluding_zeros(column):
            return column[column != 0].mean()

        # Apply the function to each column and multiply by 100 to get percentages
        accuracy_percentages = df_accuracy_metrics.apply(mean_excluding_zeros)
        filtered_accuracy_percentages = accuracy_percentages[
            accuracy_percentages.index.isin(savedColumnsDisplayed.iloc[:, 1])]
        filtered_accuracy_percentages = filtered_accuracy_percentages.to_frame().T
        filtered_accuracy_percentages.columns = [f"{col} Column Accuracy" for col in
                                                 filtered_accuracy_percentages.columns]

        # st.write(filtered_accuracy_percentages)

        # These for loops do the actual calculation by comparing the score pre-imputation to post-imputation (x vs y)
        # for i in range(len(savedColumnsDisplayed)):
        #  finalAccuracyMetrics[savedColumnsDisplayed.iloc[i, 1] + " Column Accuracy"] = np.nan
        # finalOutputMerged[savedColumnsDisplayed.iloc[i, 1] + "_LogProbs"] = finalOutputMerged.apply(
        #      lambda row: calculate_Jaro(row[savedColumnsDisplayed.iloc[i, 0] + '_x'],
        #                                 row[savedColumnsDisplayed.iloc[i, 1] + '_y']), axis=1)

        # Output the metrics in a dataframe to be displayed on the frontend
        # for i in range(len(savedColumnsDisplayed)):
        #    finalAccuracyMetrics.loc[0, savedColumnsDisplayed.iloc[i, 1] + " Column Accuracy"] = \
        #        finalOutputMerged[finalOutputMerged[savedColumnsDisplayed.iloc[i, 1] + '_x'].isnull()][
        #           savedColumnsDisplayed.iloc[i, 1] + '_LogProbs'].mean()

        ## Second, display the accuracy metrics
        # Set up the section header and the number of columns for the metrics to be outputted in
        st.subheader("Model Metrics")
        col_num = 0  # This is used to count iteration number later
        col1, col2, col3, col4 = st.columns(4)
        output_columns = [col1, col2, col3,
                          col4]  # Needed an iterable list of these variables

        # Dynamically loop through each column that gets imputed in and use the col list above to output metrics
        # The n col variables can be reused
        for i, n in enumerate(filtered_accuracy_percentages.columns):
            output_columns[col_num].metric(f"**{n} %**", "{:.00%}".format(
                filtered_accuracy_percentages.loc[0, n]))
            # Check to see if i has reached 4 or a number divisible by 4
            # This means a new output row needs to be started on the frontend and the col variables can be overwritten
            if (i + 1) % 4 == 0:
                col_num = 0
                pass
            # Otherwise keep going through the loop
            else:
                col_num += 1

        # Clean up the temporary directory when the app is done
        shutil.rmtree(temp_dir)

        # Save the renamed file to the temporary directory
        # finaldf.to_csv(finalFileName, index=False)

    else:
        print('waiting')
    return  # Don't need to return anything here, just run the script to create the webpage


# Run the frontend script via the above function
frontend_main()
