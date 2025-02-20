import streamlit as st
import pandas as pd
# import pickle

# Variables initialization
df = pd.DataFrame()
'''
with open('best_svc_model.pkl', 'rb') as file:
    data = pickle.load(file)
print("Type of loaded object:", type(data))
'''
def Home_screen():
    # Main function
    title, space, logoutBtn = st.columns([1, 1, 0.25], vertical_alignment="bottom", border=False)
    with title:
        st.title('ModelXpert')
    with space:
        st.write(" ")  
    with logoutBtn:
        st.button("", on_click=logout, icon=":material/logout:")
    
    uploaded_file = st.file_uploader(
        'Upload your dataset here',
        type=['csv', 'xlsx'],
        accept_multiple_files=False,
        label_visibility="visible",
    )

    if uploaded_file:
        checkDoc(uploaded_file)

    if not df.empty:
        labels = radioButton()
        submitButton(labels)

def logout():
    st.session_state.logged_in = False

def submitButton(labelSelection):
    # Target variable, independent variables, and Submit Button
    if labelSelection == "Manual":
        targetVariable = st.selectbox("Choose Target Variable", df.columns)
        options = st.multiselect("Choose the columns you want to set as independent variable", df.columns)
        if st.button("Submit"):
            dataSubmit(targetVariable, options)
    elif labelSelection == "Automatic":
        targetVariable = st.selectbox("Choose Target Variable", df.columns, index=len(df.columns)-1, disabled=True)
        options = st.multiselect(
            "Choose the columns you want to set as independent variable",
            df.columns,  # Available options
            default=df.columns[:-1],  # Preselected values
            disabled=True
        )  # Disable interaction
        if st.button("Submit"):
            dataSubmit(targetVariable, options)

def dataSubmit(targetVariable, options):
    # Handle submitted data
    st.write("Submitted Target Variable:", targetVariable)
    st.write("Submitted Independent Variables:", options)

def radioButton():
    # Label selection radio button
    labelSelection = st.radio(
        "Select Labels",
        ["Automatic", "Manual"], 
        captions=["Completely automatic", "Manually select the labels"]
    )
    return labelSelection

def checkDoc(uploaded_file):
    # Check the document
    global df  
    st.write(f"Filename: {uploaded_file.name}")

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write("CSV file loaded successfully!")

    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        st.write("Excel file loaded successfully!")

    else:
        st.error("File type not supported for preview.")
    
    st.write(df.head())

if __name__ == "__main__":
    Home_screen()
