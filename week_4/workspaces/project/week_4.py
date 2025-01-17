from datetime import datetime
from typing import List

from dagster import (
    AssetSelection,
    Nothing,
    OpExecutionContext,
    ScheduleDefinition,
    String,
    asset,
    define_asset_job,
    load_assets_from_current_module,
)
from workspaces.types import Aggregation, Stock


@asset(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
)
def get_s3_data(context) -> List[Stock]:
    
    """
    from CoRise: This op reads a file from S3 (provided as a config schema) 
    and converts the contents into a list of our custom data type Stock. 
    Last week we used the csv module to read the contents of a local file 
    and return an iterator. We will replace that functionality with our S3 resource 
    and use the S3 client method get_data to read the contents a file 
    from remote storage 
    (in this case our localstack version of S3 within Docker).

    my comments: this could be an overly long one-liner but imo it's more legible as
    broken down into parts. could also be a list comprehension with '[]' instead of list() 
    """

    s3_key = context.op_config["s3_key"]
    s3_data = context.resources.s3.get_data(s3_key)
    return list(Stock.from_list(row) for row in s3_data)


@asset
def process_data(get_s3_data) -> Aggregation:
    """
    using context from previous op and the returned stock_list, 
    this op takes the stock list converting from Stock class 
    then takes the max from the stock list using key and a lambda function
    , pulling out the required Aggregation class attributes 
    and to return an Aggregation class object
    """

    highest = max(get_s3_data, key=lambda x: x.high)

    return Aggregation(date=highest.date, high=highest.high)


@asset(
    required_resource_keys={"redis"},
)
def put_redis_data(context, process_data):
    """
    from CoRise: This op relies on the redis_resource. 
    In week one, our op did not do anything besides accept the output 
    from the processing app. Now we want to take that output
    (our Aggregation custom type) and upload it to Redis. 
    Luckily, our wrapped Redis client has a method to do just that. 
    If you look at the put_data method, it takes in a name and a value 
    and uploads them to our cache. Our Aggregation types has two properties to it, 
    a date and a high. The date should be the name 
    and the high should be our value, but be careful because the put_data method 
    expects those values as strings.

    my comments: ridiculously straightforward - using context to get resources and the redis resource, 
    it uses the put_data method defined in the redis class resource and puts the defined values for date
    and highest value as strings (forced with str())
    """
    context.resources.redis.put_data(
        name=str(process_data.date), value=str(process_data.high))


@asset(
    required_resource_keys={"s3"},
)
def put_s3_data(context, process_data):
    """
    from CoRise: This op also relies on the same S3 resource as get_s3_data. 
    For the sake of this project we will use the same bucket 
    so we can leverage the same configuration. 
    As with the redis op we will take in the aggregation from 
    and write to it into a file in S3. 
    The key name for this file should not be set in a config 
    (as it is with the get_s3_data op) 
    but should be generated within the op itself.

    my comments: very similar to the Redis.put_data method but we have a data parameter 
    defined for s3 which should accept the aggregation which is inferred(?) from the method inputs
    """

    context.resources.s3.put_data(
        key_name=str(process_data.date), data=process_data)


project_assets = load_assets_from_current_module()

local_config = {
    "ops": {
        "get_s3_data": {
            "config": {"s3_key": "prefix/stock_9.csv"},
        },
    },
}

machine_learning_asset_job = define_asset_job(
    name="machine_learning_asset_job",
    selection=project_assets,
    config=local_config
)

machine_learning_schedule = ScheduleDefinition(job=machine_learning_asset_job, cron_schedule="*/15 * * * *")
