# Let's just use the local mongod instance. Edit as needed.

# Please note that MONGO_HOST and MONGO_PORT could very well be left
# out as they already default to a bare bones local 'mongod' instance.
#MONGO_HOST = 'localhost'
MONGO_PORT = 29019
MONGO_USERNAME = 'user'
MONGO_PASSWORD = 'user'
MONGO_DBNAME = 'apitest'

URL_PREFIX="api"

# Enable reads (GET), inserts (POST) and DELETE for resources/collections
# (if you omit this line, the API will default to ['GET'] and provide
# read-only access to the endpoint).
RESOURCE_METHODS = ['GET', 'POST', 'DELETE']

# Enable reads (GET), edits (PATCH), replacements (PUT) and deletes of
# individual items  (defaults to read-only item access).
ITEM_METHODS = ['GET', 'PATCH', 'PUT', 'DELETE']

schema_image = {
    'name': {'type': 'string'},
    'file': {'type': 'media'},
    'path': {'type': 'string'},
    'randomized_location': {'type': 'string'},
    'randomized_scale': {'type': 'string'},
}

image = {
    'resource_methods': ['GET', 'POST'],
    'schema': schema_image
}

result = {
    'resource_methods': ['GET'],
    'schema' : {
        'oids'  : {'type': 'list'},
        'status': {'type': 'string'},
        'result': {'type': 'list'}
    }
}

DOMAIN = {
    'image': image,
    'result': result,
}
