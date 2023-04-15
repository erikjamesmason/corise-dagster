from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
    tags={"kind": "s3"},
    out = {"stocks": Out(dagster_type = List[Stock])},
    description = "Get a list of stock data from s3_key"
)
def get_s3_data(context: OpExecutionContext):
    """
    This op reads a file from S3 (provided as a config schema) 
    and converts the contents into a list of our custom data type Stock. 
    Last week we used the csv module to read the contents of a local file 
    and return an iterator. We will replace that functionality with our S3 resource 
    and use the S3 client method get_data to read the contents a file 
    from remote storage 
    (in this case our localstack version of S3 within Docker).
    """

    s3_key = context.op_config["s3_key"]
    s3_data = context.resources.s3.get_data(s3_key)
    return list(Stock.from_list(row) for row in s3_data)


@op(
    ins={"stock_list": In(dagster_type=List[Stock], description="List of Stock")},
    out={"aggregation": Out(dagster_type=Aggregation,
                            description="Aggregated value from data")}
)
def process_data(context, stock_list) -> Aggregation:
    """
    using context from previous op and the returned stock_list, 
    this op takes the stock list converting from Stock class 
    and turning into a dictionary to turn into a Pandas DataFrame. 
    Then it's just a matter of finding the max 
    and pulling out the required Aggregation class attributes 
    and readjusting their datatype
    """

    highest = max(stock_list, key = lambda x: x.high)

    return Aggregation(date=highest.date, high=highest.high)


@op(
    required_resource_keys={"redis"},
    tags={"kind": "Redis"}
)
def put_redis_data(context: OpExecutionContext, aggregation: Aggregation):
    """
    This op relies on the redis_resource. 
    In week one, our op did not do anything besides accept the output 
    from the processing app. Now we want to take that output
    (our Aggregation custom type) and upload it to Redis. 
    Luckily, our wrapped Redis client has a method to do just that. 
    If you look at the put_data method, it takes in a name and a value 
    and uploads them to our cache. Our Aggregation types has two properties to it, 
    a date and a high. The date should be the name 
    and the high should be our value, but be careful because the put_data method 
    expects those values as strings.
    """
    context.resources.redis.put_data(name=str(aggregation.date), value=str(aggregation.high))

@op(
    required_resource_keys={"s3"},
    tags={"kind": "s3"}
)
def put_s3_data(context: OpExecutionContext, aggregation: Aggregation):
    """
    This op also relies on the same S3 resource as get_s3_data. 
    For the sake of this project we will use the same bucket 
    so we can leverage the same configuration. 
    As with the redis op we will take in the aggregation from 
    and write to it into a file in S3. 
    The key name for this file should not be set in a config 
    (as it is with the get_s3_data op) 
    but should be generated within the op itself.
    """
    
    context.resources.s3.put_data(key_name=str(aggregation.date), data=aggregation)


@graph
def machine_learning_graph():
    

    a = get_s3_data()

    b = process_data(a)

    redis_line = put_redis_data(b)

    s3_line = put_s3_data(b)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3":mock_s3_resource,
    "redis":ResourceDefinition.mock_resource()}
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker,
    resource_defs={'s3': s3_resource,
    'redis':redis_resource}
)