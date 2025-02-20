import streamlit as st

from home import Home_screen
from signup import signup_page

# Simulated user credentials (for demonstration purposes only)
USER_CREDENTIALS = {
    "admin": "admin",
    "user": "admin"
}

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.session_state.logged_in = True
        st.session_state.reload = True
    # else:
    #     st.error("Invalid username or password")

# Login Page
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    loggingin,space,Signup = st.columns([0.25,1,0.25],vertical_alignment="bottom",gap="large")
    with loggingin: st.button("Login",on_click=login(username, password))
    with space: st.write(" ")
    # with Signup: st.button("Signup", on_click=signup_page)
    with Signup: st.button("Signup")
# Main App
if st.session_state.logged_in:
    Home_screen()
else:
    login_page()
