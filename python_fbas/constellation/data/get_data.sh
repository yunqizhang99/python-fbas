#!/bin/bash

# exit if no argument is provided
if [ -z "$1" ]
then
  echo "Please provide a date in YYYY-MM-DD format."
  exit 1
fi

DATE=$1

curl -X 'GET' "https://api.stellarbeat.io/v1/node?at=${DATE}"  -H 'accept: application/json' > ./network-${DATE}.json
curl -X 'GET' "https://api.stellarbeat.io/v1/organization?at=${DATE}"  -H 'accept: application/json' > ./organizations-${DATE}.json
