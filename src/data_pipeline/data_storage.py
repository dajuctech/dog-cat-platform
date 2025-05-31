import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro
from google.cloud import bigquery
import boto3

def generate_image_metadata(directory, output_format='parquet', output_path='data/metadata/'):
    """
    Scan images in a directory and save metadata as Parquet or Avro.
    """
    print(f"Scanning images in {directory}...")

    metadata = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, filename)
                label = os.path.basename(os.path.dirname(full_path))
                metadata.append({'filepath': full_path, 'label': label})

    df = pd.DataFrame(metadata)
    os.makedirs(output_path, exist_ok=True)
    
    if output_format == 'parquet':
        parquet_file = os.path.join(output_path, 'image_metadata.parquet')
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_file)
        print(f"Metadata saved to {parquet_file}")
    elif output_format == 'avro':
        avro_file = os.path.join(output_path, 'image_metadata.avro')
        schema = {
            'doc': 'Image metadata',
            'name': 'Image',
            'namespace': 'example',
            'type': 'record',
            'fields': [
                {'name': 'filepath', 'type': 'string'},
                {'name': 'label', 'type': 'string'}
            ]
        }
        with open(avro_file, 'wb') as out:
            fastavro.writer(out, schema, df.to_dict('records'))
        print(f"Metadata saved to {avro_file}")
    else:
        raise ValueError("Unsupported format. Use 'parquet' or 'avro'.")

def upload_to_bigquery(parquet_path, dataset_id, table_id):
    """
    Upload Parquet file to Google BigQuery.
    """
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    with open(parquet_path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
    job.result()
    print(f"Data uploaded to BigQuery table {dataset_id}.{table_id}")

def upload_to_redshift(parquet_path, bucket_name, s3_key, redshift_table, iam_role):
    """
    Upload Parquet file to S3 and load into Redshift.
    """
    s3 = boto3.client('s3')
    s3.upload_file(parquet_path, bucket_name, s3_key)
    print(f"File uploaded to s3://{bucket_name}/{s3_key}")
    
    # Use Redshift COPY command (requires setting up Redshift connection)
    copy_command = f"""
    COPY {redshift_table}
    FROM 's3://{bucket_name}/{s3_key}'
    IAM_ROLE '{iam_role}'
    FORMAT AS PARQUET;
    """
    print(f"Execute this COPY command in Redshift:\n{copy_command}")

if __name__ == "__main__":
    directory = 'data/processed/dogs-vs-cats-vvsmall/train'
    generate_image_metadata(directory, output_format='parquet')

    # Optional BigQuery upload
    # upload_to_bigquery('data/metadata/image_metadata.parquet', 'your_dataset', 'your_table')

    # Optional Redshift upload
    # upload_to_redshift('data/metadata/image_metadata.parquet', 'your-s3-bucket', 'path/image_metadata.parquet', 'your_redshift_table', 'your-iam-role')
