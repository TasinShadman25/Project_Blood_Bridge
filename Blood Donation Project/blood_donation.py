# blood_donation_full.py
import pandas as pd
import streamlit as st
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import plotly.express as px
from login_register import login_register_page

st.set_page_config(page_title="Blood Bridge Dashboard", layout="wide")

# ----------------------
# SESSION STATE INIT
# ----------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None

# ----------------------
# SHOW LOGIN IF NOT LOGGED IN
# ----------------------
if not st.session_state["logged_in"]:
    login_register_page()
    st.stop()  # Stop the rest of the app until login

# ----------------------
# LOGOUT BUTTON
# ----------------------
if st.sidebar.button("🚪 Log Out"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()  # instantly restarts the app and returns to login



# ----------------------
# APP CONFIGURATION AFTER LOGIN
# ----------------------
#st.set_page_config(page_title="Blood Donation Dashboard", layout="wide")
st.sidebar.title("🩸 Blood Bridge")

# Sidebar menu
# Sidebar menu
menu_options = [
    "🏠 Home", 
    "👥 Donors List", 
    "🏥 Hospitals List", 
    "📝 Register as Donor", 
    "💉 Find Matching Donors & Nearest Hospital", 
    "📊 Donation Statistics",
    "🩸 Request Blood",  
    "📋 Request Blood List"
]

# Set default menu if not set
if "menu" not in st.session_state:
    st.session_state["menu"] = menu_options[0]

# Safe index
try:
    default_index = menu_options.index(st.session_state["menu"])
except ValueError:
    default_index = 0
    st.session_state["menu"] = menu_options[0]

menu = st.sidebar.radio(
    "Navigation",
    menu_options,
    index=default_index
)
st.session_state["menu"] = menu


# ===============================
# 📂 CSV & DATA SETUP
# ===============================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DONORS_CSV = DATA_DIR / "donors.csv"
REQUESTS_CSV = DATA_DIR / "requests.csv"
AB_POOL_CSV = DATA_DIR / "ab_negative_pool.csv"
TIPS_CSV = DATA_DIR / "tips.csv"
HOSPITALS_CSV = DATA_DIR / "hospital.csv"

# Create CSVs with headers if missing
for file, columns, sample in [
    (DONORS_CSV, ["name","phone","location","blood_group","last_donation_date","eligible_status","anonymous_mode"], None),
    (REQUESTS_CSV, ["patient_name","phone","location","required_blood_group","emergency","hospital_name","verified"], None),
    (AB_POOL_CSV, ["name","phone","location","last_donation_date"], None),
    (TIPS_CSV, ["tip"], ["Stay hydrated before donating blood.","Eat a healthy meal before donation."]),
    (HOSPITALS_CSV, ["Name","Address","Mobile"], None)
]:
    if not file.exists():
        if sample:
            pd.DataFrame({col:[val] for col,val in zip(columns,sample)}).to_csv(file,index=False)
        else:
            pd.DataFrame(columns=columns).to_csv(file,index=False)

# ===============================
# 🌐 Geocoder
# ===============================
geolocator = Nominatim(user_agent="blood_donation_app")
def get_coordinates(address):
    try:
        loc = geolocator.geocode(address)
        if loc:
            return (loc.latitude, loc.longitude)
    except:
        return None
    return None

# ===============================
# 🧬 Abstract Classes
# ===============================
class User(ABC):
    def __init__(self, name, phone, location):
        self.__name = name
        self.__phone = phone
        self.__location = location

    def get_name(self): return self.__name
    def set_name(self, name): self.__name = name
    def get_phone(self): return self.__phone
    def set_phone(self, phone): self.__phone = phone
    def get_location(self): return self.__location
    def set_location(self, location): self.__location = location

    @abstractmethod
    def display_info(self):
        pass

class Donor(User):
    def __init__(self, name, phone, location, blood_group, last_donation_date=None, eligible_status=True, anonymous_mode=False):
        super().__init__(name, phone, location)
        self.__blood_group = blood_group
        self.__last_donation_date = last_donation_date or datetime.now().strftime("%Y-%m-%d")
        self.__eligible_status = eligible_status
        self.__anonymous_mode = anonymous_mode

    def get_blood_group(self): return self.__blood_group
    def set_blood_group(self, bg): self.__blood_group = bg
    def get_last_donation_date(self): return self.__last_donation_date
    def set_last_donation_date(self, date): self.__last_donation_date = date
    def get_eligible_status(self): return self.__eligible_status
    def set_eligible_status(self, status): self.__eligible_status = status
    def get_anonymous_mode(self): return self.__anonymous_mode
    def set_anonymous_mode(self, anon): self.__anonymous_mode = anon

    def display_info(self):
        return f"{self.get_name()} ({self.get_blood_group()}) - {self.get_location()}"

class Requester(User):
    def __init__(self, patient_name, phone, location, required_blood_group, emergency=False, hospital_name=None):
        super().__init__(patient_name, phone, location)
        self.__required_blood_group = required_blood_group
        self.__emergency = emergency
        self.__hospital_name = hospital_name
        self.__verified = False

    def get_required_blood_group(self): return self.__required_blood_group
    def set_required_blood_group(self, bg): self.__required_blood_group = bg
    def display_info(self):
        return f"Patient {self.get_name()} needs {self.get_required_blood_group()} at {self.get_location()}"

# ===============================
# 📦 Data Handlers
# ===============================
def save_donor(donor: Donor):
    df = pd.DataFrame([{
        "name": donor.get_name(),
        "phone": donor.get_phone(),
        "location": donor.get_location(),
        "blood_group": donor.get_blood_group(),
        "last_donation_date": donor.get_last_donation_date(),
        "eligible_status": donor.get_eligible_status(),
        "anonymous_mode": donor.get_anonymous_mode()
    }])
    if DONORS_CSV.exists():
        df.to_csv(DONORS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(DONORS_CSV, index=False)
    return True

def save_request(request: Requester):
    df = pd.DataFrame([{
        "patient_name": request.get_name(),
        "phone": request.get_phone(),
        "location": request.get_location(),
        "required_blood_group": request.get_required_blood_group(),
        "emergency": getattr(request,"_Requester__emergency",False),
        "hospital_name": getattr(request,"_Requester__hospital_name",None),
        "verified": getattr(request,"_Requester__verified",False)
    }])
    if REQUESTS_CSV.exists():
        df.to_csv(REQUESTS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(REQUESTS_CSV, index=False)
    return True

def read_donors():
    if not DONORS_CSV.exists():
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=["name","phone","location","blood_group","last_donation_date","eligible_status","anonymous_mode"])
    
    df = pd.read_csv(DONORS_CSV)
    
    # Ensure column exists
    if "last_donation_date" in df.columns:
        df["last_donation_date"] = pd.to_datetime(df["last_donation_date"], errors="coerce", infer_datetime_format=True)
        df["eligible_status"] = (datetime.utcnow() - df["last_donation_date"]).dt.days >= 90
    else:
        df["last_donation_date"] = pd.NaT
        df["eligible_status"] = False
    if "anonymous_mode" not in df.columns:
        df["anonymous_mode"] = False
    return df


def update_eligibility(df):
    df["days_since"] = (datetime.utcnow() - df["last_donation_date"]).dt.days
    df["eligible_status"] = df["days_since"] >= 90
    df.drop(columns=["days_since"], inplace=True)
    return df

def get_matching_donors(blood_group, location):
    df = read_donors()
    if df.empty: return pd.DataFrame()
    df = df[df["blood_group"].str.upper() == blood_group.upper()]
    df = df[df["location"].str.contains(location, case=False, na=False)]
    df = df[(df["anonymous_mode"] != True) & (df["eligible_status"] == True)]
    return df

def find_closest_donor(request_location, blood_group_required):
    df = get_matching_donors(blood_group_required, request_location)
    if df.empty: return None
    request_coords = get_coordinates(request_location)
    if not request_coords: return None
    df['Coordinates'] = df['location'].apply(get_coordinates)
    df = df[df['Coordinates'].notnull()]
    if df.empty: return None
    df['Distance_km'] = df['Coordinates'].apply(lambda x: geodesic(request_coords, x).km)
    closest = df.loc[df['Distance_km'].idxmin()]
    return closest

def find_closest_hospital(user_location):
    user_coords = get_coordinates(user_location)
    if not user_coords: return None
    hospitals_df = pd.read_csv(HOSPITALS_CSV)
    hospitals_df['Coordinates'] = hospitals_df['Address'].apply(get_coordinates)
    hospitals_df = hospitals_df[hospitals_df['Coordinates'].notnull()]
    if hospitals_df.empty: return None
    hospitals_df['Distance_km'] = hospitals_df['Coordinates'].apply(lambda x: geodesic(user_coords, x).km)
    closest = hospitals_df.loc[hospitals_df['Distance_km'].idxmin()]
    return closest

def load_random_tip():
    tips = pd.read_csv(TIPS_CSV)
    return tips.sample(1).iloc[0,0] if not tips.empty else ""

# ===============================
# 📊 Charts
# ===============================
def donors_per_blood_group():
    df = read_donors()
    if df.empty: fig,ax=plt.subplots(); ax.text(0.5,0.5,"No donor data",ha="center",va="center"); ax.axis("off"); return fig
    counts = df['blood_group'].value_counts()
    fig,ax=plt.subplots(); counts.plot.bar(ax=ax); ax.set_title("Donors per Blood Group"); return fig

def eligible_vs_ineligible():
    df = read_donors()
    if df.empty: fig,ax=plt.subplots(); ax.text(0.5,0.5,"No donor data",ha="center",va="center"); ax.axis("off"); return fig
    counts = df['eligible_status'].value_counts()
    fig,ax=plt.subplots(); counts.plot.pie(ax=ax, autopct='%1.1f%%'); ax.set_ylabel(''); ax.set_title("Eligible vs Ineligible Donors"); return fig

def requests_per_blood_group():
    df = pd.read_csv(REQUESTS_CSV) if REQUESTS_CSV.exists() else pd.DataFrame()
    if df.empty: fig,ax=plt.subplots(); ax.text(0.5,0.5,"No requests yet",ha="center",va="center"); ax.axis("off"); return fig
    counts = df['required_blood_group'].value_counts()
    fig,ax=plt.subplots(); counts.plot.pie(ax=ax, autopct='%1.1f%%'); ax.set_ylabel(''); ax.set_title("Requests per Blood Group"); return fig




# ===============================
# 👥 Donors List Page (PASTE HERE)
# ===============================
if menu == "👥 Donors List":
    st.title("👥 Donors List")
    
    # Read donors.csv
    donors_df = read_donors()  # Make sure this reads donors.csv
    
    if donors_df.empty:
        st.info("No donor data available.")
    else:
        display_cols = ['name', 'blood_group', 'location', 'phone', 'email', 'eligible_status']
        available_cols = [col for col in display_cols if col in donors_df.columns]
        
        st.dataframe(donors_df[available_cols], use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Filter Donors by Blood Group")
        blood_groups = donors_df['blood_group'].dropna().unique().tolist()
        selected_bg = st.selectbox("Select Blood Group", ["All"] + blood_groups)
        
        if selected_bg != "All":
            filtered_df = donors_df[donors_df['blood_group'] == selected_bg]
        else:
            filtered_df = donors_df
        
        st.dataframe(filtered_df[available_cols], use_container_width=True)


blood_colors = {"A+":"red","A-":"darkred","B+":"blue","B-":"darkblue","O+":"green","O-":"darkgreen","AB+":"purple","AB-":"indigo"}
# ================= HOME =================
if menu == "🏠 Home":
    st.title("🩸 Blood Bridge Dashboard - Home")
    st.markdown("Welcome to the central hub for blood donation statistics and updates.\n You can Request Blood and Register as a Donor.")
    st.markdown("---")

    # --- Define Layout ---
    col0, col1, col2 = st.columns(3)

    # ===============================
    # 🩸 COLUMN 0: Metrics + Charts
    # ===============================
    with col0:
        donors_df = read_donors()
        requests_df = pd.read_csv(REQUESTS_CSV) if REQUESTS_CSV.exists() else pd.DataFrame()

        total_donors = len(donors_df)
        eligible_donors = donors_df["eligible_status"].sum() if not donors_df.empty else 0
        total_requests = len(requests_df)

        # --- Metrics in one row ---
        st.markdown("### 📊 Key Features")
        m1, m2, m3 = st.columns(3)
        m1.metric("🧍 Total Donors", total_donors)
        m2.metric("✅ Eligible Donors", eligible_donors)
        m3.metric("📨 Total Requests", total_requests)

        # --- Bar Chart: Donors by Blood Group ---
        if not donors_df.empty:
            st.markdown("### 🩸 Donors by Blood Group")
            blood_counts = donors_df["blood_group"].value_counts()
            st.bar_chart(blood_counts)

        # --- Bar Chart: Area-wise Requests ---
        if not requests_df.empty and "location" in requests_df.columns:
            st.markdown("### 📍 Blood Requests by Area")
            area_counts = requests_df["location"].value_counts()
            st.bar_chart(area_counts)

    # --- COLUMN 1: Buttons + Recent Requests ---
    with col1:
        st.markdown("### ⚙️ Quick Actions")

        # --- Make a Request Button ---
        if st.button("📝 Make a Request", key="make_request", use_container_width=True):
            st.session_state["menu"] = "🩸 Request Blood"
            st.rerun()

        # --- Register as Donor Button ---
        if st.button("💉 Register as Donor", key="register_donor", use_container_width=True):
            st.session_state["menu"] = "📝 Register as Donor"
            st.rerun()

        st.markdown("---")

        # --- Recent Requests Table ---
        st.markdown("### 📢 Recent Blood Requests")
        if not requests_df.empty:
            st.dataframe(
                requests_df[['patient_name', 'required_blood_group', 'phone', 'location', 'emergency', 'hospital_name', 'condition', 'allergies']].tail(5), 
                use_container_width=True
            )
        else:
            st.info("No recent requests found.")

        st.markdown("---")

        # ===== Line Chart: Blood Donations Over Time =====
        st.subheader("📈 Blood Donations Over Time")
        if not donors_df.empty:
            donors_df['last_donation_date'] = pd.to_datetime(donors_df['last_donation_date'], errors='coerce')
            donations_per_month = donors_df.groupby(donors_df['last_donation_date'].dt.to_period('M')) \
                                        .size().reset_index(name='donations')
            donations_per_month['last_donation_date'] = donations_per_month['last_donation_date'].dt.to_timestamp()

            import plotly.express as px
            fig = px.line(
                donations_per_month, 
                x='last_donation_date', 
                y='donations', 
                title='Blood Donations Over Time',
                markers=True
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Donations",
                xaxis=dict(tickformat="%b %Y")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No donation data available yet.")


    # ===============================
    # 💡 COLUMN 2: Donor Locations Map
    # ===============================
    with col2:
        st.subheader("📍 Donor Locations Map (Dhaka)")

        # Copy donors dataframe
        donors_df_dhaka = donors_df.copy()

        # Keep only donors whose location is inside Dhaka city
        donors_df_dhaka = donors_df_dhaka[
            donors_df_dhaka['location'].str.contains(
                "Dhaka|Malibagh|Moghbazar|Mirpur|Dhanmondi|Gulshan|Banani|Uttara", 
                case=False, na=False
            )
        ]

        # Make sure latitude and longitude columns are numeric
        if 'latitude' in donors_df_dhaka.columns and 'longitude' in donors_df_dhaka.columns:
            donors_df_dhaka['latitude'] = pd.to_numeric(donors_df_dhaka['latitude'], errors='coerce')
            donors_df_dhaka['longitude'] = pd.to_numeric(donors_df_dhaka['longitude'], errors='coerce')

            # Drop rows with missing or invalid coordinates
            map_df = donors_df_dhaka.dropna(subset=['latitude', 'longitude'])

            # Show map
            if not map_df.empty:
                st.map(map_df[['latitude', 'longitude']])
            else:
                st.info("No donor location data available in Dhaka.")
        else:
            st.warning("Latitude/Longitude missing. Run geocode_donors.py first!")

        st.markdown("---")

        # --- Blood Group Requests Bar Chart ---
        st.markdown("### 📈 Requests by Blood Group")
        if not requests_df.empty and "required_blood_group" in requests_df.columns:
            group_counts = requests_df["required_blood_group"].value_counts()
            st.bar_chart(group_counts)
        else:
            st.info("No blood group requests data available yet.")


# ================= HOSPITALS LIST =================
elif menu=="🏥 Hospitals List":
    st.title("🏥 Hospitals List")
    hospitals_df = pd.read_csv(HOSPITALS_CSV)
    search_area = st.text_input("Search hospitals by area")
    filtered = hospitals_df
    if search_area: filtered = hospitals_df[hospitals_df['Address'].str.contains(search_area, case=False, na=False)]
    st.dataframe(filtered)

# ================= REGISTER DONOR =================
elif menu=="📝 Register as Donor":
    st.title("➕ Register as Donor")
    with st.form("donor_form", clear_on_submit=True):
        name = st.text_input("Name")
        phone = st.text_input("Phone")
        location = st.text_input("Location")
        blood_group = st.selectbox("Blood Group", ["A+","A-","B+","B-","AB+","AB-","O+","O-"])
        last_date = st.date_input("Last Donation Date")
        anon = st.checkbox("Anonymous Mode")
        submit = st.form_submit_button("Save Donor")
        if submit:
            donor = Donor(name, phone, location, blood_group, last_date.isoformat(), True, anon)
            save_donor(donor)
            st.success("Donor saved successfully!")

# ================= FIND DONORS & HOSPITALS =================
elif menu == "💉 Find Matching Donors & Nearest Hospital":
    st.title("🔍 Find Donors & Nearest Hospital")

    # --- Input Fields ---
    bg = st.text_input("Blood Group (optional)")
    loc = st.text_input("Location (optional)")

    # --- Matching Donors ---
    st.subheader("🧍 Matching Donors")
    donors_df = read_donors()
    if donors_df.empty:
        st.info("No donors available.")
    else:
        filtered = donors_df.copy()
        if bg:
            filtered = filtered[filtered['blood_group'].str.upper() == bg.upper()]
        if loc:
            filtered = filtered[filtered['location'].str.contains(loc, case=False, na=False)]

        if filtered.empty:
            st.info("No matching donors found.")
        else:
            display_cols = ['name', 'blood_group', 'phone', 'location', 'eligible_status']
            available_cols = [col for col in display_cols if col in filtered.columns]
            st.dataframe(filtered[available_cols], use_container_width=True)

    st.markdown("---")
    st.subheader("🏥 Closest Donor & Hospital Search")

    # --- Closest Donor & Hospital ---
    if st.button("Search Closest Donor & Hospital"):
        search_loc = loc if loc else "Dhaka"
        search_bg = bg if bg else "A+"

        closest_donor = find_closest_donor(search_loc, search_bg)
        closest_hospital = find_closest_hospital(search_loc)

        # --- Display Closest Donor ---
        if closest_donor is not None:
            st.write({
                "Name": closest_donor['name'],
                "Blood Group": closest_donor['blood_group'],
                "Phone": closest_donor['phone'],
                "Location": closest_donor['location'],
                "Distance (km)": round(closest_donor['Distance_km'], 2)
            })
        else:
            st.info("No closest donor found.")

        # --- Display Closest Hospital ---
        if closest_hospital is not None:
            st.write({
                "Hospital Name": closest_hospital['Name'],
                "Address": closest_hospital['Address'],
                "Distance (km)": round(closest_hospital['Distance_km'], 2)
            })
        else:
            st.info("No nearby hospital found.")


# ================= REQUEST BLOOD (Add / Remove) =================
elif menu=="🩸 Request Blood":
    st.title("🩹 Blood Requests - Add / Remove")
    
    # --- Add new blood request ---
    st.subheader("➕ Add New Blood Request")
    with st.form("request_form", clear_on_submit=True):
        patient_name = st.text_input("Patient Name")
        phone = st.text_input("Phone")
        location = st.text_input("Location")
        required_blood_group = st.selectbox(
            "Required Blood Group", 
            ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
        )
        emergency = st.checkbox("Emergency")
        hospital_name = st.text_input("Hospital Name")

        # ===== New fields =====
        condition = st.text_area("💊 Condition of the Patient", placeholder="e.g., Injury, Operation, Accident, etc.")
        allergies = st.text_area("⚠️ Allergies (if any)", placeholder="e.g., Dust, Peanuts, None")
        # =====================

        submit_request = st.form_submit_button("Submit Request")
        if submit_request:
            # Save the new request to CSV
            new_data = pd.DataFrame([{
                "patient_name": patient_name,
                "phone": phone,
                "location": location,
                "required_blood_group": required_blood_group,
                "emergency": emergency,
                "hospital_name": hospital_name,
                "verified": False,
                "condition": condition if condition else "Not specified",
                "allergies": allergies if allergies else "None"
            }])

            if REQUESTS_CSV.exists():
                new_data.to_csv(REQUESTS_CSV, mode="a", header=False, index=False)
            else:
                new_data.to_csv(REQUESTS_CSV, index=False)
            
            st.success("✅ Blood request added successfully!")

            # --- Show the latest few entries as confirmation ---
            if REQUESTS_CSV.exists():
                recent_df = pd.read_csv(REQUESTS_CSV).tail(5)
                st.subheader("📋 Latest Blood Requests")
                st.dataframe(recent_df)

    # --- Remove existing blood request ---
    st.subheader("🗑 Remove Existing Request")
    if REQUESTS_CSV.exists():
        df = pd.read_csv(REQUESTS_CSV)
        if not df.empty:
            # Build options as patient + blood group + location
            options = df.apply(lambda x: f"{x['patient_name']} | {x['required_blood_group']} | {x['location']}", axis=1).tolist()
            to_remove = st.selectbox("Select a request to remove", options)
            if st.button("Remove Request"):
                # Remove selected request
                df = df[~df.apply(lambda x: f"{x['patient_name']} | {x['required_blood_group']} | {x['location']}" == to_remove, axis=1)]
                df.to_csv(REQUESTS_CSV, index=False)
                st.success("Request removed successfully!")
        else:
            st.info("No blood requests available to remove.")
    else:
        st.info("No blood requests available.")
          
elif menu=="📋 Request Blood List":
    st.title("📋 Blood Requests List")
    
    if REQUESTS_CSV.exists():
        df = pd.read_csv(REQUESTS_CSV)
        if df.empty:
            st.warning("No blood requests available.")
        else:
            # Filters
            blood_filter = st.selectbox(
                "Filter by Blood Group", 
                options=["All"] + df['required_blood_group'].dropna().unique().tolist()
            )
            location_filter = st.text_input("Filter by Location (optional)")

            filtered = df.copy()
            if blood_filter != "All":
                filtered = filtered[filtered['required_blood_group'] == blood_filter]
            if location_filter:
                filtered = filtered[filtered['location'].str.contains(location_filter, case=False, na=False)]

            # Display table with condition and allergies
            st.dataframe(
                filtered[['patient_name','required_blood_group','phone','location','emergency','hospital_name','condition','allergies']],
                use_container_width=True
            )
    else:
        st.info("No blood requests yet.")

# ================= CHARTS =================

elif menu == "📊 Donation Statistics":
    import plotly.express as px
    import pandas as pd

    st.title("📊 Blood Donation Statistics Dashboard")
    st.markdown("View insights from live donor and blood request data.")
    st.markdown("---")

    # --- Load Data Safely ---
    donors_file = DONORS_CSV
    requests_file = REQUESTS_CSV

    if donors_file.exists():
        donors_df = pd.read_csv(donors_file)
    else:
        st.error("❌ donors.csv not found")
        st.stop()

    if requests_file.exists():
        requests_df = pd.read_csv(requests_file)
    else:
        st.error("❌ requests.csv not found")
        st.stop()

# --- Summary Metrics ---
    st.subheader("📋 Summary Statistics")
    c1,c2,c3 = st.columns(3)
    total_donors = len(donors_df)
    total_requests = len(requests_df)
    unique_areas = donors_df['location'].nunique() if 'location' in donors_df.columns else 0

    c1.metric("Total Donors", total_donors)
    c2.metric("Total Requests", total_requests)
    c3.metric("Areas Covered", unique_areas)

    # --- Donors per Location Bar Chart ---
    st.subheader("📍 Donors per Area")
    if 'location' in donors_df.columns:
        area_counts = donors_df['location'].value_counts().reset_index()
        area_counts.columns = ['Location','Number of Donors']
        fig_area = px.bar(
            area_counts,
            x='Location',
            y='Number of Donors',
            text='Number of Donors',
            color='Location',
            title='Donors by Area',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_area.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.warning("⚠️ donors.csv does not have a 'location' column")

    st.markdown("---")

    # --- Requests per Blood Group Bar Chart ---
    st.subheader("💉 Blood Requests by Blood Group")
    if 'required_blood_group' in requests_df.columns:
        bg_counts = requests_df['required_blood_group'].value_counts().reset_index()
        bg_counts.columns = ['Blood Group','Requests']
        fig_bar = px.bar(
            bg_counts,
            x='Blood Group',
            y='Requests',
            text='Requests',
            color='Blood Group',
            title='Blood Group Requests',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_bar.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ requests.csv does not have 'required_blood_group' column")

    st.markdown("---")

    # --- Requests per Blood Group Pie Chart ---
    st.subheader("🩸 Most Requested Blood Groups")
    if 'required_blood_group' in requests_df.columns:
        fig_pie = px.pie(
            bg_counts,
            names='Blood Group',
            values='Requests',
            color_discrete_sequence=px.colors.qualitative.Safe,
            hole=0.3,
            title='Blood Group Request Distribution'
        )
        fig_pie.update_layout(title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # --- Line Chart: Blood Donations Over Time ---
    st.subheader("📈 Blood Donations Over Time")
    if 'last_donation_date' in donors_df.columns:
        donors_df['last_donation_date'] = pd.to_datetime(donors_df['last_donation_date'], errors='coerce')
        donations_per_month = donors_df.groupby(donors_df['last_donation_date'].dt.to_period('M')).size().reset_index(name='donations')
        donations_per_month['last_donation_date'] = donations_per_month['last_donation_date'].dt.to_timestamp()

        fig_line = px.line(
            donations_per_month,
            x='last_donation_date',
            y='donations',
            markers=True,
            title='Blood Donations Over Time'
        )
        fig_line.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Donations",
            xaxis=dict(tickformat="%b %Y"),
            title_x=0.5
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("⚠️ donors.csv does not have a 'last_donation_date' column")
        
    st.markdown("---")