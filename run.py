from eve import Eve
from gridfs import GridFS
from bson.objectid import ObjectId
import json

import sys
#sys.path.insert(0, '../simg-task') # Assume task is a sibling folder as api
import tasks as Tasks

app = Eve()


#get connection to db
#it's probably better to use  app.data.driver.db
#but we know the following will work for now
from pymongo import MongoClient
db = MongoClient('localhost', 29019).apitest
fs = GridFS(db, collection='fs')
resultCollectionName = 'result'

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# add query_id to the response payload, and
# fire up a celery task
def post_image_post_cb(request, payload):
    cargo = json.JSONDecoder().decode(payload.data)

    # create a document in the result collection
    doc = db.result.insert_one({
        'oids' : [cargo[u'_id']]
    })

    # fire up a celery task
    #Tasks.do_roi.delay(str(doc.inserted_id))
    #run task without celery
    Tasks.do_roi(str(doc.inserted_id))

    cargo['result'] = {
        'href' : resultCollectionName + '/' + str(doc.inserted_id)
    }
    payload.data = json.JSONEncoder().encode(cargo) 

# add extra fields to the response payload
# * result
# * query_id
def post_results_get_cb(request, payload):
    cargo = json.JSONDecoder().decode(payload.data)

    if len(cargo['_items']) == 0: # no matches
        return 

    item = db.results.find_one(ObjectId(cargo['_items'][0][u'_id'])) # XXX: use the first item

    if item and 'query_id' in item:
        queryOid = item['query_id']
        if queryOid:
            cargo['query_id'] = str(queryOid)

        result = item['result']
        if result:
            cargo['result'] = result

        payload.data = json.JSONEncoder().encode(cargo) 

if __name__ == '__main__':
    #app.on_inserted_image += db_image_insert_cb
    app.on_post_POST_image += post_image_post_cb
    app.on_post_GET_results += post_results_get_cb
    app.run(host='0.0.0.0', port=5050, debug=True)

