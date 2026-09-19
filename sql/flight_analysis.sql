CREATE DATABASE aviation_analysis;

USE aviation_analysis;

CREATE TABLE flight_data (
FL_DATE DATE,
AIRLINE VARCHAR(100),
AIRLINE_CODE VARCHAR(10),
FL_NUMBER INT,
ORIGIN VARCHAR(10),
ORIGIN_CITY VARCHAR(100),
DEST VARCHAR(10),
DEST_CITY VARCHAR(100),
CRS_DEP_TIME INT,
DEP_TIME FLOAT,
DEP_DELAY FLOAT,
TAXI_OUT FLOAT,
WHEELS_OFF FLOAT,
WHEELS_ON FLOAT,
TAXI_IN FLOAT,
CRS_ARR_TIME INT,
ARR_TIME FLOAT,
ARR_DELAY FLOAT,
CANCELLED INT,
DIVERTED INT,
CRS_ELAPSED_TIME FLOAT,
ELAPSED_TIME FLOAT,
AIR_TIME FLOAT,
DISTANCE FLOAT,
DELAY_DUE_CARRIER FLOAT,
DELAY_DUE_WEATHER FLOAT,
DELAY_DUE_NAS FLOAT,
DELAY_DUE_SECURITY FLOAT,
DELAY_DUE_LATE_AIRCRAFT FLOAT
);

 SET GLOBAL local_infile = 1;

SHOW VARIABLES LIKE 'local_infile';

LOAD DATA LOCAL INFILE "C:/Users/Shreyas/OneDrive/Documents/DA_proj/Global Flight Delay & Airport Operations Analysis/clean_flight_data.csv"
INTO TABLE flight_data
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

SELECT COUNT(*) FROM flight_data; # checking whether all the data  are imported

# Total Flights
SELECT COUNT(*) AS total_flights
FROM flight_data;

# Total Flights by Airline
SELECT AIRLINE,
COUNT(*) AS total_flights
FROM flight_data
GROUP BY AIRLINE
ORDER BY total_flights DESC;

# Average Arrival Delay by Airline
SELECT AIRLINE,
AVG(ARR_DELAY) AS avg_arrival_delay
FROM flight_data
WHERE ARR_DELAY IS NOT NULL
GROUP BY AIRLINE
ORDER BY avg_arrival_delay DESC;

# Airports with Highest Departure Delays
SELECT ORIGIN,
AVG(DEP_DELAY) AS avg_departure_delay
FROM flight_data
WHERE DEP_DELAY IS NOT NULL
GROUP BY ORIGIN
ORDER BY avg_departure_delay DESC
LIMIT 10;

# Top 10 Most Frequent Routes
SELECT ORIGIN, DEST,
COUNT(*) AS total_flights
FROM flight_data
GROUP BY ORIGIN, DEST
ORDER BY total_flights DESC
LIMIT 10;

# Route With Highest Average Delay
SELECT ORIGIN, DEST,
AVG(ARR_DELAY) AS avg_delay
FROM flight_data
WHERE ARR_DELAY IS NOT NULL
GROUP BY ORIGIN, DEST
ORDER BY avg_delay DESC
LIMIT 10;

# Flights Cancelled by Airline
SELECT AIRLINE,
COUNT(*) AS cancelled_flights
FROM flight_data
WHERE CANCELLED = 1
GROUP BY AIRLINE
ORDER BY cancelled_flights DESC;

SET GLOBAL local_infile = 0; 
SHOW VARIABLES LIKE 'local_infile';

