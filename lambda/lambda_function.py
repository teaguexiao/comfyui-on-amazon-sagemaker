import json
import boto3
import logging
import random
import base64
import io
import os
import time
import uuid

# Define Logger
logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)

sagemaker_runtime_client = boto3.client("sagemaker-runtime")
sagemaker_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")


def update_seed(prompt_dict, seed=None):
    """
    Update the seed value for the KSampler node in the prompt dictionary.

    Args:
        prompt_dict (dict): The prompt dictionary containing the node information.
        seed (int, optional): The seed value to set for the KSampler node. If not provided, a random seed will be generated.

    Returns:
        dict: The updated prompt dictionary with the seed value set for the KSampler node.
    """
    # set seed for KSampler node
    for i in prompt_dict:
        if "inputs" in prompt_dict[i]:
            if (
                prompt_dict[i]["class_type"] == "KSampler"
                and "seed" in prompt_dict[i]["inputs"]
            ):
                if seed is None:
                    prompt_dict[i]["inputs"]["seed"] = random.randint(0, int(1e10))
                else:
                    prompt_dict[i]["inputs"]["seed"] = int(seed)
    return prompt_dict


def update_prompt_text(prompt_dict, positive_prompt, negative_prompt):
    """
    Update the prompt text in the given prompt dictionary.

    Args:
        prompt_dict (dict): The dictionary containing the prompt information.
        positive_prompt (str): The new text to replace the positive prompt placeholder.
        negative_prompt (str): The new text to replace the negative prompt placeholder.

    Returns:
        dict: The updated prompt dictionary.
    """
    # replace prompt text for CLIPTextEncode node
    for i in prompt_dict:
        if "inputs" in prompt_dict[i]:
            if (
                prompt_dict[i]["class_type"] == "CLIPTextEncode"
                and "text" in prompt_dict[i]["inputs"]
            ):
                if prompt_dict[i]["inputs"]["text"] == "POSITIVE_PROMT_PLACEHOLDER":
                    prompt_dict[i]["inputs"]["text"] = positive_prompt
                elif prompt_dict[i]["inputs"]["text"] == "NEGATIVE_PROMPT_PLACEHOLDER":
                    prompt_dict[i]["inputs"]["text"] = negative_prompt
    return prompt_dict


def wait_for_async_job(output_location, max_attempts=60):
    """
    Wait for the async inference job to complete by checking if the output file exists in S3.

    Args:
        output_location (str): The S3 URI where the inference result will be stored
        max_attempts (int): Maximum number of attempts to check file existence

    Returns:
        dict: The job status response containing the output location
    """
    # Parse the S3 URI
    bucket = output_location.split('/')[2]
    key = '/'.join(output_location.split('/')[3:])
    
    for attempt in range(max_attempts):
        try:
            # Check if the file exists in S3
            s3_client.head_object(Bucket=bucket, Key=key)
            return {"OutputLocation": output_location}
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                # File doesn't exist yet, wait and try again
                time.sleep(5)
            else:
                # Other error occurred
                raise Exception(f"Error checking S3 file: {str(e)}")
    
    raise Exception("Async inference job timed out - output file not found")


def invoke_from_prompt(prompt_file, positive_prompt, negative_prompt, seed=None):
    """
    Invokes the SageMaker endpoint asynchronously with the provided prompt data.

    Args:
        prompt_file (str): The path to the JSON file in ./workflow/ containing the prompt data.
        positive_prompt (str): The new text to replace the positive prompt placeholder.
        negative_prompt (str): The negative prompt to be used in the prompt data.
        seed (int, optional): The seed value for randomization. Defaults to None.

    Returns:
        dict: The response from the SageMaker endpoint containing the async inference job details.
    """
    logger.info("prompt: %s", prompt_file)

    # read the prompt data from json file
    with open("./workflow/" + prompt_file) as prompt_file:
        prompt_text = prompt_file.read()

    prompt_dict = json.loads(prompt_text)
    prompt_dict = update_seed(prompt_dict, seed)
    prompt_dict = update_prompt_text(prompt_dict, positive_prompt, negative_prompt)
    prompt_text = json.dumps(prompt_dict)

    endpoint_name = os.environ["ENDPOINT_NAME"]
    deployment_bucket = os.environ["DEPLOYMENT_BUCKET"]
    
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Upload the input data to S3
    input_key = f"async-input/{request_id}/input.json"
    s3_client.put_object(
        Bucket=deployment_bucket,
        Key=input_key,
        Body=prompt_text,
        ContentType="application/json"
    )
    
    # Create the S3 URI for the input data
    input_location = f"s3://{deployment_bucket}/{input_key}"
    
    # Start async inference
    response = sagemaker_runtime_client.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="*/*",
        InputLocation=input_location
    )
    
    # Add the request ID to the response for tracking
    response["RequestId"] = request_id
    response["InputLocation"] = input_location
    
    return response


def get_async_result(output_location):
    """
    Retrieves the result from the S3 output location of an async inference job.

    Args:
        output_location (str): The S3 URI where the inference result is stored.

    Returns:
        bytes: The image data from the inference result.
    """
    # Parse the S3 URI
    bucket = output_location.split('/')[2]
    key = '/'.join(output_location.split('/')[3:])
    
    # Get the result from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    
    # Read the binary data directly
    return response['Body'].read()


def lambda_handler(event: dict, context: dict):
    """
    Lambda function handler for processing events.

    Args:
        event (dict): The event from lambda function URL.
        context (dict): The runtime information of the Lambda function.

    Returns:
        dict: The response data for lambda function URL.
    """
    logger.info("Event:")
    logger.info(json.dumps(event, indent=2))
    request = json.loads(event["body"])

    try:
        prompt_file = request.get("prompt_file", "workflow_api.json")
        positive_prompt = request["positive_prompt"]
        negative_prompt = request.get("negative_prompt", "")
        seed = request.get("seed")
        
        # Start async inference
        response = invoke_from_prompt(
            prompt_file=prompt_file,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
        )
        
        # Wait for the output file to exist
        job_status = wait_for_async_job(response["OutputLocation"])
        
        # Get the result
        image_data = get_async_result(job_status["OutputLocation"])
        
        # Return the image data
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "image/png"
            },
            "body": base64.b64encode(image_data).decode("utf-8"),
            "isBase64Encoded": True
        }
        
    except KeyError as e:
        logger.error(f"Error: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "Missing required parameter",
            })
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    event = {
        "body": "{\"positive_prompt\": \"hill happy dog\",\"negative_prompt\": \"hill\",\"prompt_file\": \"workflow_api.json\",\"seed\": 123}"
    }
    lambda_handler(event, None)
