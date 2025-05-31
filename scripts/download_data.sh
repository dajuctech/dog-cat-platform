#!/bin/bash

# Script to run the data ingestion module from Google Drive

echo "Starting data ingestion process..."
python3 src/data_pipeline/data_ingestion.py
echo "Data ingestion completed successfully!"
