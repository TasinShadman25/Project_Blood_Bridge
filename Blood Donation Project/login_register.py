import streamlit as st
import pandas as pd
import hashlib
from pathlib import Path

USERS_CSV = Path("users.csv")

def rerun():
    """Cross-version safe rerun."""
    st.session_state["trigger_rerun"] = True
    st.stop()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_register_page():
    st.title("Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # --- LOGIN ---
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if USERS_CSV.exists():
                    users_df = pd.read_csv(USERS_CSV)
                    user = users_df[users_df["username"] == username]
                    if not user.empty and user.iloc[0]["password"] == hash_password(password):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["menu"] = "🏠 Home"
                        rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("No users found. Please register first.")

    # --- REGISTER ---
    with tab2:
        new_username = st.text_input("New Username", key="reg_username")
        new_password = st.text_input("New Password", type="password", key="reg_password")
        if st.button("Register"):
            if USERS_CSV.exists():
                users_df = pd.read_csv(USERS_CSV)
            else:
                users_df = pd.DataFrame(columns=["username", "password"])

            if new_username in users_df["username"].values:
                st.error("Username already exists")
            else:
                new_user = pd.DataFrame({
                    "username": [new_username],
                    "password": [hash_password(new_password)]
                })
                users_df = pd.concat([users_df, new_user], ignore_index=True)
                users_df.to_csv(USERS_CSV, index=False)
                st.success("Registered successfully!")

                # Log in new user immediately
                st.session_state["logged_in"] = True
                st.session_state["username"] = new_username
                st.session_state["menu"] = "🏠 Home"
                st.stop()
