import os
import folium
from ev_ai_agents.carbon_agent.utils.data_loader import load_green_logistics

def map_high_emission_routes(output_path: str = "route_emissions_map.html") -> str:
    """Generate an interactive Folium map showing routes colored by emissions and priority markers.
    
    Args:
        output_path (str): File path to save the HTML map.
        
    Returns:
        str: Absolute path to the saved map file.
    """
    df = load_green_logistics()
    
    # Center map around California / Southwest US where many routes are located
    m = folium.Map(location=[37.5, -119.0], zoom_start=6, tiles="cartodbpositron")
    
    # Calculate annual emissions in MT
    df['annual_emissions_mt'] = (df['carbon_emissions_kg'] * df['annual_trips']) / 1000.0
    
    # Add routes as polylines
    for _, row in df.iterrows():
        start_coord = [row['start_lat'], row['start_lon']]
        end_coord = [row['end_lat'], row['end_lon']]
        
        annual_emissions = row['annual_emissions_mt']
        is_diesel = 'Diesel' in str(row['vehicle_type'])
        
        # Color coding by emissions volume
        if annual_emissions >= 50.0:
            color = "#ef4444"  # Premium Red
            weight = 5
        elif annual_emissions >= 15.0:
            color = "#f59e0b"  # Premium Orange
            weight = 4
        else:
            color = "#10b981"  # Premium Green
            weight = 3
            
        # Draw the route path
        tooltip_text = (
            f"<b>Route:</b> {row['route_id']} ({row['start_location']} &rarr; {row['end_location']})<br>"
            f"<b>Vehicle:</b> {row['vehicle_type']}<br>"
            f"<b>Distance:</b> {row['distance_km']} km<br>"
            f"<b>Annual Trips:</b> {row['annual_trips']}<br>"
            f"<b>Annual CO2:</b> {round(annual_emissions, 2)} MT"
        )
        
        folium.PolyLine(
            locations=[start_coord, end_coord],
            color=color,
            weight=weight,
            opacity=0.8,
            tooltip=tooltip_text
        ).add_to(m)
        
        # Add markers for start and end points
        folium.CircleMarker(
            location=start_coord,
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"Start: {row['start_location']}"
        ).add_to(m)
        
        folium.CircleMarker(
            location=end_coord,
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"End: {row['end_location']}"
        ).add_to(m)
        
        # If it is a high-priority diesel route (e.g. annual emissions > 40 MT), highlight it as electrification priority
        if is_diesel and annual_emissions > 40.0:
            folium.Marker(
                location=start_coord,
                icon=folium.Icon(color='blue', icon='flash', prefix='fa'),
                popup=f"⚡ <b>Electrification Priority</b><br>Route: {row['route_id']}<br>Convert {row['vehicle_type']} to EV."
            ).add_to(m)
            
    # Save the map
    m.save(output_path)
    return os.path.abspath(output_path)
