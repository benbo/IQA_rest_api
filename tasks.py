from __future__ import print_function
from pymongo import MongoClient
from gridfs import GridFS
from Composites import Composites
from ImageSearch import ImageSearchAnnoy
from bson.objectid import ObjectId
import numpy as np
import cv2
import os 
import sys
import tempfile
from google.protobuf import text_format as proto_text

CAFFE_ROOT = '/mnt/xfsdata/simg/caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
from caffe.proto import caffe_pb2

c = lambda s: os.path.join(CAFFE_ROOT, s)
CAFFENET_PROTO = c('models/bvlc_reference_caffenet/deploy.prototxt')
CAFFENET_MODEL = c(
            'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
MEAN_FILE = c('python/caffe/imagenet/ilsvrc_2012_mean.npy')
MEAN = np.load(MEAN_FILE).mean(axis=(1, 2))

caffe.set_mode_gpu()
caffe.set_device(1)

#Specify mongodb host and datababse to connect to
# format: transport://userid:password@hostname:port/virtual_host
BROKER_URL = 'mongodb://user:user@localhost:29019/apitest'

#get connection to db
db = MongoClient('localhost', 29019).apitest
fs = GridFS(db, collection='fs')



def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_net(prototxt=CAFFENET_PROTO, model=CAFFENET_MODEL, layers=['pool5'],
             data_mean=MEAN, batch_size=100,set_raw_scale=True,set_channel_swap=True,set_transpose=True):
    param = caffe_pb2.NetParameter()
    with open(prototxt, 'r') as f:
        proto_text.Parse(f.read(), param)
    try:
        param.layer[0].input_param.shape[0].dim[0] = batch_size
    except:
        print("WARNING: old prototxt found. Trying to use previous model definition.")
        param.input_shape[0].dim[0] = batch_size

    # assume that everything after the last layer is garbage
    last_layer = max(i for i, l in enumerate(param.layer) if l.name in layers)
    del param.layer[last_layer + 1:]

    with tempfile.NamedTemporaryFile() as f:
        f.write(proto_text.MessageToString(param))
        f.flush()
        net = caffe.Net(f.name, model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', data_mean) # mean pixel
    if set_raw_scale:
        transformer.set_raw_scale('data', 255)  # model works on [0,255], not [0,1]
    if set_channel_swap:
        transformer.set_channel_swap('data', (2,1,0))  # model has channels in BGR

    return net, transformer

def preprocess_images(images, transformer):
    for image in images:
        if isinstance(image, (str, bytes)):
            image = caffe.io.load_image(image)
        yield transformer.preprocess('data', image)

def group(things, batch_size):
    out = []
    for x in things:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if len(out)>0:
        yield out

def get_features(images):
    image_stream = preprocess_images(images, transformer)
    for batch in group(image_stream, batch_size):
        batch = np.asarray(batch)
        if len(batch) != batch_size:  # last group
            net.blobs['data'].reshape(*batch.shape)
            net.blobs['data'].data[...] = batch
            net.forward()
            all_out = np.hstack([
                net.blobs[l].data.reshape(batch.shape[0], -1) for l in layers])
            for row in all_out:
                yield row
            #reset shape
            s = list(batch.shape)
            s[0]=batch_size
            net.blobs['data'].reshape(*tuple(s))
        else:
            net.blobs['data'].data[...] = batch
            net.forward()
            all_out = np.hstack([
                net.blobs[l].data.reshape(batch.shape[0], -1) for l in layers])
            for row in all_out:
                yield row




#takes a string polygon path, converts it to a list of tuples
def get_path(stringpath):
    path = stringpath.split(',')
    return [tuple(map(int,path[i:i+2])) for i in xrange(0,len(path),2)]


# simg ROI task
# @param result_oid - string, e.g. 57226be994eb93da56175ce1
# TODO: 
def do_roi(result_oid):
    '''
    fetch images and polygons from mongodb, create composite images, featurize them, marginalize, 
    and then run a similarity search against an image feature database
    '''
    eprint(result_oid)
    # fetch the result object from the database 
    result_obj = db.result.find_one(ObjectId(result_oid))
    # retrieve image and path info
    oids = db.image.find({'_id': {'$in': [ObjectId(i) for i in result_obj['oids']]}})
    queries= []
    for obj in oids:
        if obj:
            spath = obj['path']
            rand_loc = False
            rand_scale = False
            dsc = db.fs.files.find_one(obj['file'])
            key = dsc['md5']+spath
            if obj['randomized_location']== 'true':
                key+='rl'
                eprint('randomized_location')
                rand_loc = True
            if obj['randomized_scale'] == 'true':
                key+='rs'
                eprint('randomized_scale')
                rand_scale = True
            query = db.queries.find_one({'key':key})
            if query:
                #query exists, use it
                eprint('found query in db')
                query = np.array(query['query'])
            else:
                #fetch chunked image
                grid = fs.get(obj['file'])
                nparr = np.fromstring(grid.read(), np.uint8)
                src = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
                if spath == 'undefined':
                    #do query with original image, no marginalization:
                    eprint('single image, get query vector.')
                    query = list(get_features([src]))[0]
                    eprint('query vector done')

                else:
                    #generate query
                    path = get_path(spath)
                    poly = np.array(path)
                    composites = [img for img in C.generate_composites(src,poly,randomize_loc = rand_loc ,randomize_scale = rand_scale,scale_lower=1.0, scale_upper=1.5)]
                    eprint('get query vector for composites')
                    query =np.mean(list(get_features(composites)), axis=0)
                    inserted = db.queries.insert_one({'key':key,'query':query.tolist()})
            queries.append(query)
        else:
            db.result.update_one({'_id':ObjectId(result_oid)}, {'$set':{'result':results,'status':'fail'}})
            return False

    eprint('run annoy')
    results = I.run_query(np.mean(queries, axis=0),n=100,accuracy_factor = 6)
    eprint('annoy done')
    #db.results.insert_one({'query_id':queryOid,'result':results})
    db.result.update_one({'_id':ObjectId(result_oid)}, {'$set':{'result':results,'status':'success'}})


layers = ['fc6','fc7']
batch_size=2000
#init caffe
net, transformer = load_net(layers=layers, batch_size=batch_size,set_raw_scale=False,set_channel_swap=False)

####Instances needed to run ROI algorithm
#TEST

sample_size = 2000
num_features = 8192
#test
#filelist = '/mnt/xfsdata/simg/opt/simg-task/TJ2015_test.txt'
#annoy_file = '/mnt/xfsdata/simg/TJ2015_test.ann'
#TJ
#filelist = '/mnt/xfsdata/simg/opt/simg-task/TJ2015_12_11_filenames.txt'
#annoy_file = '/mnt/xfsdata/simg/X_TJ2015_12_11.ann'
#ILSVRC
#filelist = '/mnt/xfsdata/simg/image_db/filelists/ILSVRC2015_filelist.txt'
#annoy_file = '/mnt/xfsdata/simg/image_db/annoy/ILSVRC2015_256.ann'
#mirflickr
#filelist = '/mnt/xfsdata/simg/image_db/filelists/flickr_filelist.txt'
#annoy_file = '/mnt/xfsdata/simg/image_db/annoy/mirflickr.ann'
#TJ 2016 
filelist = '/mnt/xfsdata/simg/image_db/filelists/TJ2016_6_5_4_3_filenames.txt'
annoy_file = '/mnt/xfsdata/simg/image_db/annoy/TJ2016_6_5_4_3.ann'


C = Composites(sample_size = sample_size,filelist=filelist)
I = ImageSearchAnnoy(num_features,annf=annoy_file,imageListPath =filelist,prefix = '/xfsauton/tj_usa/backpages/BackpageOutput/')


