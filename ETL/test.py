import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
# This is our first testing exploration of the data and possible implementations
# Configure spark session
spark = SparkSession\
    .builder\
    .master('local[2]')\
    .appName('accidents_etl')\
    .config("spark.mongodb.input.uri", 'mongodb://127.0.0.1/Accident.us_accidents?readPreference=primaryPreferred')\
    .config('spark.mongodb.output.uri', 'mongodb://127.0.0.1/Accident.us_accidents')\
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
    .getOrCreate()

accidents_schema = StructType([
    StructField('ID', StringType()),
    StructField('Severity', DoubleType()),
    StructField('Start_Time', StringType()),
    StructField('End_Time', StringType()),
    StructField('Start_Lat', DoubleType()),
    StructField('Start_Lng', DoubleType()),
    StructField('End_Lat', DoubleType()),
    StructField('End_Lng', DoubleType()),
    StructField('Distance(mi)', DoubleType()),
    StructField('Description', StringType()),
    StructField('Number', DoubleType()),
    StructField('Street', StringType()),
    StructField('Side', StringType()),
    StructField('City', StringType()),
    StructField('County', StringType()),
    StructField('State', StringType()),
    StructField('Zipcode', StringType()),
    StructField('Country', StringType()),
    StructField('Timezone', StringType()),
    StructField('Airport_Code', StringType()),
    StructField('Weather_Timestamp', StringType()),
    StructField('Temperature(F)', DoubleType()),
    StructField('Wind_Chill(F)', DoubleType()),
    StructField('Humidity(%)', DoubleType()),
    StructField('Pressure(in)', DoubleType()),
    StructField('Visibility(mi)', DoubleType()),
    StructField('Wind_Direction', StringType()),
    StructField('Wind_Speed(mph)', DoubleType()),
    StructField('Precipitation(in)', DoubleType()),
    StructField('Weather_Condition', StringType()),
    StructField('Amenity', StringType()),
    StructField('Bump', StringType()),
    StructField('Crossing', StringType()),
    StructField('Give_Way', StringType()),
    StructField('Junction', StringType()),
    StructField('No_Exit', StringType()),
    StructField('Railway', StringType()),
    StructField('Roundabout', StringType()),
    StructField('Station', StringType()),
    StructField('Stop', StringType()),
    StructField('Traffic_Calming', StringType()),
    StructField('Traffic_Signal', StringType()),
    StructField('Turning_Loop', StringType()),
    StructField('Sunrise_Sunset', StringType()),
    StructField('Civil_Twilight', StringType()),
    StructField('Nautical_Twilight', StringType()),
    StructField('Astronomical_Twiligh', StringType()),
])

# Load the dataset
df_load = spark.read.csv(r"Accident_No_NA.csv", schema=accidents_schema)

# Drop fields we don't need from df_load
lst_dropped_columns = ['ID','Description','Turning_Loop','Country','Weather_Timestamp','Number','Wind_Chill(F)']
df_load = df_load.drop(*lst_dropped_columns).cache()
# Preview df_load
df_load.show(5)

#df_clean1 = df_load.select('Wind_Direction').distinct()
#print(df_load.collect())
df_load = df_load.withColumn('Wind_Direction', when((df_load['Wind_Direction'] == 'WSW') | (df_load['Wind_Direction'] == 'WNW') | (df_load['Wind_Direction'] == 'W'), 'West')
                                                    .when((df_load['Wind_Direction'] == 'SSW') | (df_load['Wind_Direction'] == 'SSE') | (df_load['Wind_Direction'] == 'SW') | (df_load['Wind_Direction'] == 'S') | (df_load['Wind_Direction'] == 'SE'), 'South')
                                                    .when((df_load['Wind_Direction'] == 'NNW') | (df_load['Wind_Direction'] == 'NNE') | (df_load['Wind_Direction'] == 'NW') | (df_load['Wind_Direction'] == 'NE') | (df_load['Wind_Direction'] == 'N'), 'North')
                                                    .when((df_load['Wind_Direction'] == 'ESE') | (df_load['Wind_Direction'] == 'ENE') | (df_load['Wind_Direction'] == 'E'), 'East')
                                                    .when(df_load['Wind_Direction'] == 'CALM', 'Clam')
                                                    .when(df_load['Wind_Direction'] == 'VAR', 'Variable')
                                                    .otherwise(df_load['Wind_Direction']))

#df_load = df_load.select('Weather_Condition').distinct()
#print(df_load.collect())
df_load = df_load.withColumn('Weather_Condition', when(df_load['Weather_Condition'].rlike('Fog|Overcast|Haze|Mist|Smoke'), 'Fog')
                                                       .when(df_load['Weather_Condition'].rlike('Clear|Fair'), 'Clear')
                                                       .when(df_load['Weather_Condition'].rlike('Rain|Showers|Drizzle|Thunder'), 'Rain')
                                                       .when(df_load['Weather_Condition'].rlike('Ice|Snow|Sleet|Hail'), 'Snow')
                                                       .when(df_load['Weather_Condition'].rlike('Storm|storm|Tornado'), 'Storm')
                                                       .when(df_load['Weather_Condition'].rlike('Stand|Dust'), 'Sand')
                                                       .when(df_load['Weather_Condition'].rlike('Cloudy|Clouds|Cloud'), 'Cloudy')
                                                       .otherwise('Other'))

# Create a year field and add it to the dataframe
df_load = df_load.withColumn('Year', year(to_timestamp('Start_Time')))
df_load.show(5)

# Build the accidents frequency dataframe using the year field and counts for each year
df_accidents_freq = df_load.groupBy('Year').count().withColumnRenamed('count', 'Counts').sort('Year')
df_accidents_freq.show(5)

# Write df_quake_freq to mongodb
df_accidents_freq.write.format('mongo')\
    .mode('overwrite')\
    .option('spark.mongodb.output.uri', 'mongodb://127.0.0.1:27017Accident.us_accidents').save()



"""
Section: Data visulization
"""

import pandas as pd
from bokeh.io import output_notebook, output_file
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models.tools import HoverTool
import math
from math import pi
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.themes import built_in_themes
from bokeh.io import curdoc
from pymongo import MongoClient


# Create a custom read function to read data from mongodb into a dataframe
def read_mongo(host='127.0.0.1', port=27017, username=None, password=None, db='Quake', collection='pred_results'):
    mongo_uri = 'mongodb://{}:{}/{}.{}'.format(host, port, db, collection)

    # Connect to mongodb
    conn = MongoClient(mongo_uri)
    db = conn[db]

    # Select all records from the collection
    cursor = db[collection].find()

    # Create the dataframe
    df = pd.DataFrame(list(cursor))

    # Delete the _id field
    del df['_id']

    return df

# Load the datasets from mongodb
df_quakes = read_mongo(collection='quakes')
df_quake_freq = read_mongo(collection='quake_freq')
df_quake_pred = read_mongo(collection='pred_results')

df_quakes_2016 = df_quakes[df_quakes['Year'] == 2016]
# Preview df_quakes_2016
df_quakes_2016.head()

# Show plots embedded in jupyter notebook
output_notebook()


# Create custom style function to style our plots
def style(p):
    # Title
    p.title.align = 'center'
    p.title.text_font_size = '20pt'
    p.title.text_font = 'serif'

    # Axis titles
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_style = 'bold'
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_style = 'bold'

    # Tick labels
    p.xaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'

    # Plot the legend in the top left corner
    p.legend.location = 'top_left'

    return p


# Create the Geo Map plot
def plotMap():
    lat = df_quakes_2016['Latitude'].values.tolist()
    lon = df_quakes_2016['Longitude'].values.tolist()

    pred_lat = df_quake_pred['Latitude'].values.tolist()
    pred_lon = df_quake_pred['Longitude'].values.tolist()

    lst_lat = []
    lst_lon = []
    lst_pred_lat = []
    lst_pred_lon = []

    i = 0
    j = 0

    # Convert Lat and Long values into merc_projection format
    for i in range(len(lon)):
        r_major = 6378137.000
        x = r_major * math.radians(lon[i])
        scale = x / lon[i]
        y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 +
                                                lat[i] * (math.pi / 180.0) / 2.0)) * scale

        lst_lon.append(x)
        lst_lat.append(y)
        i += 1

    # Convert predicted lat and long values into merc_projection format
    for j in range(len(pred_lon)):
        r_major = 6378137.000
        x = r_major * math.radians(pred_lon[j])
        scale = x / pred_lon[j]
        y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 +
                                                pred_lat[j] * (math.pi / 180.0) / 2.0)) * scale

        lst_pred_lon.append(x)
        lst_pred_lat.append(y)
        j += 1

    df_quakes_2016['coords_x'] = lst_lat
    df_quakes_2016['coords_y'] = lst_lon
    df_quake_pred['coords_x'] = lst_pred_lat
    df_quake_pred['coords_y'] = lst_pred_lon

    # Scale the circles
    df_quakes_2016['Mag_Size'] = df_quakes_2016['Magnitude'] * 4
    df_quake_pred['Mag_Size'] = df_quake_pred['Pred_Magnitude'] * 4

    # create datasources for our ColumnDataSource object
    lats = df_quakes_2016['coords_x'].tolist()
    longs = df_quakes_2016['coords_y'].tolist()
    mags = df_quakes_2016['Magnitude'].tolist()
    years = df_quakes_2016['Year'].tolist()
    mag_size = df_quakes_2016['Mag_Size'].tolist()

    pred_lats = df_quake_pred['coords_x'].tolist()
    pred_longs = df_quake_pred['coords_y'].tolist()
    pred_mags = df_quake_pred['Pred_Magnitude'].tolist()
    pred_year = df_quake_pred['Year'].tolist()
    pred_mag_size = df_quake_pred['Mag_Size'].tolist()

    # Create column datasource
    cds = ColumnDataSource(
        data=dict(
            lat=lats,
            lon=longs,
            mag=mags,
            year=years,
            mag_s=mag_size
        )
    )

    pred_cds = ColumnDataSource(
        data=dict(
            pred_lat=pred_lats,
            pred_long=pred_longs,
            pred_mag=pred_mags,
            year=pred_year,
            pred_mag_s=pred_mag_size
        )
    )

    # Tooltips
    TOOLTIPS = [
        ("Year", " @year"),
        ("Magnitude", " @mag"),
        ("Predicted Magnitude", " @pred_mag")
    ]

    # Create figure
    p = figure(title='Earthquake Map',
               plot_width=2300, plot_height=450,
               x_range=(-2000000, 6000000),
               y_range=(-1000000, 7000000),
               tooltips=TOOLTIPS)

    p.circle(x='lon', y='lat', size='mag_s', fill_color='#cc0000', fill_alpha=0.7,
             source=cds, legend='Quakes 2016')

    # Add circles for our predicted earthquakes
    p.circle(x='pred_long', y='pred_lat', size='pred_mag_s', fill_color='#ccff33', fill_alpha=7.0,
             source=pred_cds, legend='Predicted Quakes 2017')

    p.add_tile(CARTODBPOSITRON)

    # Style the map plot
    # Title
    p.title.align = 'center'
    p.title.text_font_size = '20pt'
    p.title.text_font = 'serif'

    # Legend
    p.legend.location = 'bottom_right'
    p.legend.background_fill_color = 'black'
    p.legend.background_fill_alpha = 0.8
    p.legend.click_policy = 'hide'
    p.legend.label_text_color = 'white'
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    # show(p)

    return p

# plotMap()

# Create the Bar Chart
def plotBar():
    # Load the datasource
    cds = ColumnDataSource(data=dict(
        yrs=df_quake_freq['Year'].values.tolist(),
        numQuakes=df_quake_freq['Counts'].values.tolist()
    ))

    # Tooltip
    TOOLTIPS = [
        ('Year', ' @yrs'),
        ('Number of earthquakes', ' @numQuakes')
    ]

    # Create a figure
    barChart = figure(title='Frequency of Earthquakes by Year',
                      plot_height=400,
                      plot_width=1150,
                      x_axis_label='Years',
                      y_axis_label='Number of Occurances',
                      x_minor_ticks=2,
                      y_range=(0, df_quake_freq['Counts'].max() + 100),
                      toolbar_location=None,
                      tooltips=TOOLTIPS)

    # Create a vertical bar
    barChart.vbar(x='yrs', bottom=0, top='numQuakes',
                  color='#cc0000', width=0.75,
                  legend='Year', source=cds)

    # Style the bar chart
    barChart = style(barChart)

    # show(barChart)

    return barChart

# plotBar()


# Create a magnitude plot
def plotMagnitude():
    # Load the datasource
    cds = ColumnDataSource(data=dict(
        yrs=df_quake_freq['Year'].values.tolist(),
        avg_mag=df_quake_freq['Avg_Magnitude'].round(1).values.tolist(),
        max_mag=df_quake_freq['Max_Magnitude'].values.tolist()
    ))

    # Tooltip
    TOOLTIPS = [
        ('Year', ' @yrs'),
        ('Average Magnitude', ' @avg_mag'),
        ('Maximum Magnitude', ' @max_mag')
    ]

    # Create the figure
    mp = figure(title='Maximum and Average Magnitude by Year',
                plot_width=1150, plot_height=400,
                x_axis_label='Years',
                y_axis_label='Magnitude',
                x_minor_ticks=2,
                y_range=(5, df_quake_freq['Max_Magnitude'].max() + 1),
                toolbar_location=None,
                tooltips=TOOLTIPS)

    # Max Magnitude
    mp.line(x='yrs', y='max_mag', color='#cc0000', line_width=2, legend='Max Magnitude', source=cds)
    mp.circle(x='yrs', y='max_mag', color='#cc0000', size=8, fill_color='#cc0000', source=cds)

    # Average Magnitude
    mp.line(x='yrs', y='avg_mag', color='yellow', line_width=2, legend='Avg Magnitude', source=cds)
    mp.circle(x='yrs', y='avg_mag', color='yellow', size=8, fill_color='yellow', source=cds)

    mp = style(mp)

    # show(mp)

    return mp

# plotMagnitude()

# Display the visuals directly in the browser
output_file('dashboard.html')
# Change to a dark theme
curdoc().theme = 'dark_minimal'

# Build the grid plot
from bokeh.layouts import gridplot

# Make the grid
grid = gridplot([[plotMap()], [plotBar(), plotMagnitude()]])

# Shor the grid
show(grid)