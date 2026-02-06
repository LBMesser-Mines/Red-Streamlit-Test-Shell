import json
import pandas as pd

#-----------------Loading Locations---------------------#
with open("lstx-reasonable.json") as f:
	data = json.load(f)

# Extract location IDs
locations = [loc["id"] for loc in data["locations"]]

# Create DataFrame with location and Symbol columns
df = pd.DataFrame({
	"location": locations,
	"Symbol": "SupplyCompany"
})

# Write to CSV
df.to_csv("symbol-injector-output.csv", index=False)

print(f"Created CSV with {len(df)} locations and Symbol column set to 'SupplyCompany'")

