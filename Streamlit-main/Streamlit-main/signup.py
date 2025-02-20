import streamlit as st

USER_CREDENTIALS = {
    "admin": "admin",
    "user": "admin"
}
def signup(username, password):
    if username in USER_CREDENTIALS:
        st.error("Username already exists")
    else:
        USER_CREDENTIALS[username] = password
        st.success("Account created successfully")
        st.session_state.logged_in = True
        st.session_state.reload = True

def signup_page():
    st.title("Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    # loggingin,space,Signup = st.columns([0.25,1,0.25],vertical_alignment="bottom",gap="large")
    st.button("Signup", on_click=lambda: signup(username, password))
    # with loggingin: st.button("Signup",on_click=signup(username, password))
    # with space: st.write(" ")
    # with Signup: st.button("Login",on_click=login_page())