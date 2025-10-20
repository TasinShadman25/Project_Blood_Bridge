import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Load donors CSV
df = pd.read_csv("data/donors.csv")

# Initialize geocoder
geolocator = Nominatim(user_agent="blood_donation_app", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Ensure lat/lon columns exist
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    df['latitude'] = None
    df['longitude'] = None

# Add Bangladesh to make the search more accurate
df['full_location'] = df['location'].astype(str) + ', Bangladesh'

# Fill missing latitude and longitude
for i, row in df.iterrows():
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        location = geocode(row['full_location'])
        if location:
            df.at[i, 'latitude'] = location.latitude
            df.at[i, 'longitude'] = location.longitude
        else:
            print(f"Could not find location for {row['location']}")

# Save back to CSV
df.to_csv("data/donors.csv", index=False)
print("Lat/Lon added to donors.csv successfully!")
