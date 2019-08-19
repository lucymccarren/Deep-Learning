import os
import boto3
import re
import copy
import time
from time import gmtime, strftime
from sagemaker import get_execution_role

role = get_execution_role()

region = boto3.Session().region_name

bucket='sagemaker-lucy-1' 
prefix = 'sagemaker/klarna'
bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket) 

import sagemaker

train_data = 's3://{}/{}/{}'.format(bucket, prefix, 'dataset.csv')

s3_output_location = 's3://{}/{}/{}'.format(bucket, prefix, 'model')
print(train_data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json
import sagemaker.amazon.common as smac

df = pd.read_csv('s3://sagemaker-lucy-1/sagemaker/klarna/dataset.csv',sep = ';')
df.columns = ["uuid","default","account_amount_added_12_24m","account_days_in_dc_12_24m","account_days_in_rem_12_24m","account_days_in_term_12_24m","account_incoming_debt_vs_paid_0_24m","account_status","account_worst_status_0_3m","account_worst_status_12_24m","account_worst_status_3_6m","account_worst_status_6_12m","age","avg_payment_span_0_12m","avg_payment_span_0_3m","merchant_category","merchant_group","has_paid","max_paid_inv_0_12m","max_paid_inv_0_24m","name_in_email","num_active_div_by_paid_inv_0_12m","num_active_inv","num_arch_dc_0_12m","num_arch_dc_12_24m","num_arch_ok_0_12m","num_arch_ok_12_24m","num_arch_rem_0_12m","num_arch_written_off_0_12m","num_arch_written_off_12_24m","num_unpaid_bills","status_last_archived_0_24m","status_2nd_last_archived_0_24m","status_3rd_last_archived_0_24m","status_max_archived_0_6_months","status_max_archived_0_12_months","status_max_archived_0_24_months","recovery_debt","sum_capital_paid_account_0_12m","sum_capital_paid_account_12_24m","sum_paid_inv_0_12m","time_hours","worst_status_active_inv"]

print(df.shape)
display(df.describe())
display(df.default.value_counts())

# Data Conversion
# Split the Data into 90% Training dataset (default rate known) and 10% Testing dataset (default rate unknown)

df_train = df.head(89976)
df_test = df.tail(10000)

#Y-axis (labels) needs to have vector containing probability of default
train_y = np.array(df_train.iloc[:, 1]).astype('float32')
test_y = np.array(df_test.iloc[:, 1]).astype('float32')

display(test_y)
test_y.shape

#X-axis (vectors) should have all other data from the dataset
#taking only numeric columns
numeric_columns = [2,3,4,5,6,12,13,14,18,19,21,22,23,24,25,26,27,28,29,30,37,38,39,40,41]
train_x = np.array(df_train.iloc[:, numeric_columns]).astype('float32')
test_x = np.array(df_test.iloc[:, numeric_columns]).astype('float32')

display(train_x)
print('Train_x shape is') 
print(train_x.shape)

#Replace null values with mean of column
import numpy.ma as ma
train_x_no_null = np.where(np.isnan(train_x), ma.array(train_x, mask=np.isnan(train_x)).mean(axis=0), train_x)
train_x_no_null = train_x_no_null.astype('float32')
test_x_no_null = np.where(np.isnan(test_x), ma.array(test_x, mask=np.isnan(test_x)).mean(axis=0), test_x)
test_x_no_null = test_x_no_null.astype('float32')


display(train_x_no_null)
print('Train_x_no_null shape is')
print(train_x_no_null.shape)

#Convert datasets to recordIO-wrapped protobuf format

train_file = 'linear_train.data'

sess = sagemaker.Session()

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_x_no_null.astype('float32'), train_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f)

s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, train_file)
print('uploaded training data location: {}'.format(s3_train_data))

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))

#Set the parameters for the model
containers = {
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:latest'
                }

linear = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                       role=role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.m4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)

linear.set_hyperparameters(feature_dim=25,
                           mini_batch_size=1000,
                           predictor_type='binary_classifier')

linear.fit({'train': s3_train_data})

#Deploy the model
linear_predictor = linear.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')

#Access the Results
from sagemaker.predictor import csv_serializer, json_deserializer

linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer

import numpy as np

#Testing predictions for a single data point
result = linear_predictor.predict(test_x_no_null[3887])
print(result)

#Predictions for all data
predictions = []
for array in test_x_no_null:
    result = linear_predictor.predict(array)
    predictions += [r['predicted_label'] for r in result['predictions']]
predictions = np.array(predictions)

display(predictions)
predictions.shape

#Check cases where probability of default is 1
x = all(i == 0 for i in predictions)
y = np.argwhere(predictions > 0)
print(x)
print(y)
print("Values bigger than 0 =", predictions[predictions>0])
print("Their indices are ", np.nonzero(predictions > 0))

#Write UUID and corresponding probability of default to file

uuid = np.array(df_test.iloc[:, 0])

Output = []

for i in range(len(uuid)):
        Output.append([uuid[i],predictions[i]])
        
import boto3
import csv

predictions_df = pd.DataFrame
predictions_df = pd.DataFrame(Output, columns=['uuid','probability of default']) 
predictions_df.to_csv('predictions.csv')

# Create an S3 Client
s3 = boto3.client('s3')
s3.upload_file('predictions.csv', bucket, '{}/{}'.format(prefix, 'predictions.csv'))

