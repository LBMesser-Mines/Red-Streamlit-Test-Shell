# Route Mapper

A Python script that visualizes shipping routes between locations on an interactive map. The script reads a CSV file with location pairs and displays arcs between them that follow actual road routes, with arc thickness proportional to the weight shipped.

## Features

- Reads CSV files with `location`, `location_prime`, and `weight_shipped` columns
- Geocodes location names to coordinates
- Gets actual routes between locations using OpenStreetMap routing (OSRM)
- Displays routes on an interactive map using Folium
- Arc thickness scales with weight (1-1000 lbs mapped to 2-15 pixels)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python route_mapper.py demo_routes.csv -o output_map.html
```

## CSV Format

The CSV file should have three columns:
- `location`: Starting location name
- `location_prime`: Destination location name  
- `weight_shipped`: Weight in lbs (1-1000)

Example:
```csv
location,location_prime,weight_shipped
Times Square,Central Park,250
Brooklyn Bridge,Statue of Liberty,500
Empire State Building,Yankee Stadium,750
```

## Demo

The repository includes `demo_routes.csv` with 3 routes in the New York City area. Run:

```bash
python route_mapper.py demo_routes.csv
```

This will create `route_map.html` which you can open in a web browser to view the interactive map.

## Notes

- The script uses a public OSRM server for routing. For production use, consider running your own OSRM instance.
- Geocoding uses Nominatim (OpenStreetMap's geocoding service) with rate limiting.
- If routing fails, the script falls back to straight lines between locations.







