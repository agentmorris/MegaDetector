# Documentation for the legacy MD batch API

## API

### API endpoints

Once configured to run on a live instance, the endpoints of this API are available at

```
http://URL/v4/camera-trap/detection-batch
```

#### `/request_detections`

To submit a request for batch processing, make a POST call to this endpoint with a json body containing input fields defined below. The API will return with a json response very quickly to give you a RequestID (UUID4 hex) representing the request you have submitted, for example:
```json
{
  "request_id": "f940ecd58c7746b1bde89bd6ba5a5202"
}
```
or an error message, if your inputs are not acceptable:
```json
{
  "error": "error message."
}
```
In particular the endpoint will return a 503 error if the queue of requests is full. Please re-try later in that case.


#### `/task`

Check the status of your request by calling the `/task` endpoint via a GET call, passing in your RequestID:

```http://URL/v4/camera-trap/detection-batch/task/RequestID```

This returns a json with the fields `Status`, `TaskId` (which is the `request_id` in this document), and a few others. The `Status` field is a json object with the following fields: 

- `request_status`: one of `running`, `failed`, `problem`, `completed`, and `canceled`. 
    - The status `failed` indicates that the images have not been submitted to the cluster for processing, and so you can go ahead and call the `\request_detections` endpoint again, correcting your inputs according to the error message returned with the status. 
    - The status `problem` indicates that the images have already been submitted for processing but the API encountered an error while monitoring progress; in this case, please contact us to retrieve your results so that no unnecessary processing would occupy the cluster (`message` field will mention "please contact us").
    - `canceled` if your call to the `/cancel_request` endpoint took effect.

- `message`: a longer string describing the `request_status` and any errors; when the request is completed, the URLs to the output files will also be here (see [Outputs](#23-outputs) section below).


#### `/supported_model_versions`
Check which versions of MegaDetector are supported by this API by making a GET call to this endpoint.

#### `/default_model_version`
Check which version of MegaDetector is used by default by making a GET call to this endpoint.

#### `/cancel_request`
If you have submitted a request by mistake, you can make a POST call to this endpoint to cancel it.

The body should contain the `caller` (see next section on _API inputs_) and `request_id` fields. You should get back a response immediately with status code 200 if the signal was successfully sent. You can verify that the request has been canceled using the `/task` endpoint. 


### API inputs

| Parameter                | Is required | Type | Explanation                 |
|--------------------------|-------------|-------|----------------------------|
| input_container_sas      | Yes<sup>1</sup>         | string | SAS URL with list and read permissions to the Blob Storage container where the images are stored. |
| images_requested_json_sas | No<sup>1</sup>        | string | SAS URL with list and read permissions to a json file in Blob Storage. See below for explanation of the content of the json to provide. |
| image_path_prefix        | No          | string | Only process images whose full path starts with `image_path_prefix` (case-_sensitive_). Note that any image paths specified in `images_requested_json_sas` will need to be the full path from the root of the container, regardless whether `image_path_prefix` is provided. |
| first_n                  | No          | int | Only process the first `first_n` images. Order of images is not guaranteed, but is likely to be alphabetical. Set this to a small number to avoid taking time to fully list all images in the blob (about 15 minutes for 1 million images) if you just want to try this API. |
| sample_n                | No          | int | Randomly select `sample_n` images to process. |
| model_version           | No          | string | Version of the MegaDetector model to use. Default is the most updated stable version (check using the `/default_model_version` endpoint). Supported versions are available at the `/supported_model_versions` endpoint.|
| request_name            | No          | string | A string (letters, digits, `_`, `-` allowed, max length 92 characters) that will be appended to the output file names to help you identify the resulting files. A timestamp in UTC (`%Y%m%d%H%M%S`) of the time of submission will be appended to the resulting files automatically. |
| use_url                  | No         | bool | Set to `true` if you are providing public image URLs. |
| caller                  | Yes         | string | An identifier that we use to whitelist users for now. |
| country                  | No (but recommended) | string | Country where the majority of the images in this batch are taken. Preferably use an [ISO 3166-1 alpha-3 code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3#Officially_assigned_code_elements), such as "BWA" for Botswana and "USA" for the United States |
| organization_name | No (but recommended) | string | Organization conducting the survey. |


<sup>1</sup> There are two ways of giving the API access to your images. 

1 - If you have all your images in a container in Azure Blob Storage, provide the parameter `input_container_sas` as described above. This means that your images do not have to be at publicly accessible URLs. In this case, the json pointed to by `images_requested_json_sas` should look like:
```json
[
  "Season1/Location1/Camera1/image1.jpg", 
  "Season1/Location1/Camera1/image2.jpg"
]
```
Only images whose paths are listed here will be processed if you provide this list.

2 - If your images are stored elsewhere and you can provide a publicly accessible URL to each, you do not need to specify `input_container_sas`. Instead, list the URLs to all the images (instead of their paths) you&rsquo;d like to process in the json at `images_requested_json_sas`.


#### Attaching metadata

We can store a (short) string of metadata with each image path or URL. The json at `images_requested_json_sas` should then look like:
```json
[
  ["Season1/Location1/Camera1/image1.jpg", "metadata_string1"], 
  ["Season1/Location1/Camera1/image2.jpg", "metadata_string2"]
]
``` 
The metadata string will be copied to the `meta` field in the image's entry in the output file (format see below).


#### Other notes and example  

- Only images with file name ending in ".jpg", ".jpeg" or ".png" (case insensitive) will be processed, so please make sure the file names are compliant before you upload them to the container (you cannot rename a blob without copying it entirely once it is in Blob Storage). 

- By default we process all such images in the specified container. You can choose to only process a subset of them by specifying the other input parameters. The images will be filtered out accordingly in this order:
    - `images_requested_json_sas`
    - `image_path_prefix`
    - `first_n`
    - `sample_n`
    
    - For example, if you specified both `images_requested_json_sas` and `first_n`, only images that are in your provided list at `images_requested_json_sas` will be considered, and then we process the `first_n` of those.

Example body of the POST request:
```json
{
  "input_container_sas": "https://storageaccountname.blob.core.windows.net/container-name?se=2019-04-23T01%3A30%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=A_LONG_STRING",
  "images_requested_json_sas": "https://storageaccountname2.blob.core.windows.net/container-name2/possibly_in_a_folder/my_list_of_images.json?se=2019-04-19T20%3A31%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=ANOTHER_LONG_STRING",
  "image_path_prefix": "2020/Alberta",
  "first_n": 100000,
  "request_name": "Alberta_2020",
  "model_version": "4.1",
  "caller": "allowlisted_user_x",
  "country": "CAN",
  "organization_name": "Name of Organization"
}
```

You can manually call the API using applications such as Postman:

![Screenshot of Azure Storage Explorer used for generating SAS tokens with read and list permissions](./images/Postman_screenshot.png)


#### How to obtain a SAS token

You can easily generate a [SAS token](https://docs.microsoft.com/en-us/azure/storage/common/storage-dotnet-shared-access-signature-part-1) to a container using the desktop app [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/) (available on Windows, macOS and Linux). You can also issue SAS tokens programmatically by using the [Azure Storage SDK](https://azure-storage.readthedocs.io/ref/azure.storage.blob.baseblobservice.html#azure.storage.blob.baseblobservice.BaseBlobService.generate_blob_shared_access_signature).


Using Storage Explorer, right click on the container or blob you&rsquo;d like to grant access for, and choose &ldquo;Get Shared Access Signature...&rdquo;. On the dialog window that appears, 
- cross out the &ldquo;Start time&rdquo; field if you will be using the SAS token right away
- set the &ldquo;Expiry time&rdquo; to a date in the future, about a month ahead is reasonable. The SAS token needs to be valid for the duration of the batch processing request. 
- make sure &ldquo;Read&rdquo; and &ldquo;List&rdquo; are checked under &ldquo;Permissions&rdquo; (see screenshot) 

Click &ldquo;Create&rdquo;, and the &ldquo;URL&rdquo; field on the next screen is the value required for `input_container_sas` or `images_requested_json_sas`. 

![Screenshot of Azure Storage Explorer used for generating SAS tokens with read and list permissions](./images/SAS_screenshot.png)


### API outputs

Once your request is submitted and parameters validated, the API divides all images into shards of about 2000 images each, and send them to an [Azure Batch](https://azure.microsoft.com/en-us/services/batch/) node pool to be scored by the model. Another process will monitor how many shards have been evaluated, checking every 15 minutes, and update the status of the request, which you can check via the `/task` endpoint. 

When all shards have finished processing, the `status` returned by the `/task` endpoint will have the `request_status` field as `completed`, and the `message` field will contain a URL to the output file. The returned body looks like

```json
{
    "Status": {
        "request_status": "completed",
        "message": {
            "num_failed_shards": 0,
            "output_file_urls": {
                "detections": "https://cameratrap.blob.core.windows.net/async-api-internal/ee26326e-7e0d-4524-a9ea-f57a5799d4ba/ee26326e-7e0d-4524-a9ea-f57a5799d4ba_detections_4_1_on_test_images_20200709211752.json?sv=2019-02-02&sr=b&sig=key1"
            }
        },
        "time": "2020-07-09 21:27:17"
    },
    "Timestamp": "2020-07-09 21:27:17",
    "Endpoint": "/v3/camera-trap/detection-batch/request_detections",
    "TaskId": "ea26326e-7e0d-4524-a9ea-f57a5799d4ba"
}
```
 
To obtain the URL of the output file:
```python
task_status = body['Status']
assert task_status['request_status'] == 'completed'
message = task_status['message']
assert message['num_failed_shards'] == 0

url_to_results_file = message['output_file_urls']['detections']
```
Note that the field `Status` in the returned body is capitalized (since July 2020).

The URL to the output file is valid for 180 days from the time the request has finished. If you neglected to retrieve them before the link expired, contact us with the RequestID and we can send the results to you. 

The output file is a JSON in the format described below.
