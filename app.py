import streamlit as st
import pandas as pd
import joblib
import base64
from io import BytesIO

# Load the pre-trained models
clf_cracking = joblib.load('cracking_model.pkl')
clf_pothole = joblib.load('pothole_model.pkl')
clf_ravelling = joblib.load('ravelling_model.pkl')

# Function to download the output as a CSV
def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        towrite = BytesIO()
        object_to_download.to_csv(towrite, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Streamlit application
st.title('Pavement Condition Prediction Based on IRI')
st.write('Upload a CSV file containing only the IRI column.')

# Upload CSV
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    if 'IRI' in df.columns:
        st.write('Input Data')
        st.write(df)

        # Add Original_IRI column and copy the original IRI values
        df.insert(0, 'Original_IRI', df['IRI'])

        # Replace blank cells with 0 and apply min-max scaling
        df['IRI'].fillna(0, inplace=True)
        min_value = df['IRI'].min()
        max_value = df['IRI'].max()
        df['IRI'] = (df['IRI'] - min_value) / (max_value - min_value)

        st.write('Processed Data')
        st.write(df)

        # Predict cracking, pothole, and ravelling values
        df['Cracking'] = clf_cracking.predict(df[['IRI']])
        df['Pothole'] = clf_pothole.predict(df[['IRI']])
        df['Ravelling'] = clf_ravelling.predict(df[['IRI']])

        st.write('Predicted Values')
        st.write(df)

        # Button to download the output
        if st.button('Download Output CSV'):
            tmp_download_link = download_link(df, 'predicted_pavement_conditions.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
    else:
        st.error('The CSV file must contain the IRI column.')
