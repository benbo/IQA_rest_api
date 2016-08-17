# IQA_rest_api
Rest api for the Image Query Adaptation (IQA) tool 

# Virtualenv setup

```bash
# Install virtualenv
$ yum install python-virtualenv.noarch 

# Create virtualenv root folder
$ virtualenv /mnt/xfsdata/simg

# To activate
$ cd /mnt/xfsdata/simg
$ source bin/activate

# To deactivate
$ deactivate
```

## NPM/bower setup
package management tools for front-end development 
```bash
# yum install npm
# npm install -g bower
```

#  Eve setup
```bash
$ pip install eve
```

# MongoDB setup
[Instruction on installing & launching MongoDB Community edition on RHEL 6/7](https://docs.mongodb.org/manual/tutorial/install-mongodb-on-red-hat/#install-mongodb-community-edition)


# Open CV setup

```bash
$ yum install cmake
$ yum install gtk2-devel
$ yum install libdc1394-devel
$ yum install libv4l-devel
$ yum install ffmpeg-devel
$ yum install gstreamer-plugins-base-devel
$ yum install lapack-devel blas-devel libgfortran python-gconf
$ yum install protobuf-devel leveldb-devel snappy-devel boost-devel hdf5-devel opencv-devel
$ yum install gflags-devel glog-devel lmdb-devel atlas-devel openblas-devel
$ yum install libjpeg-turbo-devel



$ git clone https://github.com/Itseez/opencv.git
$ cd opencv/
$ mkdir build
$ cd build/

# if cuda is causing problems, build without.
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/mnt/xfsdata/simg -D WITH_CUDA=OFF ..

$ make -j10
$ make install

```

# Caffe setup
This installation assumes that cuda is already installed
```bash
$ git clone https://github.com/BVLC/caffe.git
$ cd caffe
$ # install cuDNN. Register online and download, then unpack and place files in cuda folder
$ # cd into the folder where you extracted the cuDNN content
$ sudo cp include/cudnn.h /usr/local/cuda/include
$ sudo cp lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

#use cmake
mkdir build
cd build
cmake -D USE_OPENCV=OFF -D BLAS=open ..
make all
make install
#optional: run tests
make runtest


$ pip install pillow
$ pip install --upgrade setuptools pip
$ pip install -r python/requirements.txt
$ make -j20 pycaffe

# download required model
$ scripts/download_model_binary.py models/bvlc_reference_caffenet

# to use caffe, we need to do
$ export PYTHONPATH=/mnt/xfsdata/simg/caffe/python:$PYTHONPATH

```


## run MongoDB locally
This doesn't require root privilege.

```bash
$ mongod --dbpath=<PATH-TO>/data/db/

# e.g. on gpu1, use port 29019
$ mongod --dbpath=/mnt/xfsdata/simg/db --port=29019
```

## Crete db and user and assign the roles
Open a MondoDb shell by typeing ```mongo --port 29019```
```bash
> use apitest
> db.createUser({user:'user', pwd:'user', roles:['readWrite', 'dbAdmin']})
```

## EVE DB configuration
Specify the DB info in the setting.py 

```python
#MONGO_HOST = 'localhost'
MONGO_PORT = 29019
MONGO_USERNAME = 'user'
MONGO_PASSWORD = 'user'
MONGO_DBNAME = 'apitest'
```

# File Storage
* Media files(images, pdfs, etc.) are stored in MongoDB [GridFS](http://docs.mongodb.org/manual/core/gridfs/)

# Launch Eve
```bash
$ python run.py
```

# APIs
## test api
To test if the API server is up and running, do
```bash
curl http://127.0.0.1:5000/api
{"_links": {"child": [{"href": "image", "title": "image"}]}}
```
Eve should spit out something like this
```
127.0.0.1 - - [19/Apr/2016 14:53:38] "GET /api HTTP/1.1" 200 -
```

## image
```bash
# upload an image
curl -X POST -F "name=angry_cat" -F "file=@/mnt/xfsdata/simg/opt/api/example/cat1.jpg" -F "path=295,75,389,86,302,198" -F "randomized_location=true" -F "randomized_scale=false"  http://127.0.0.1:5000/api/image

{"_updated": "Thu, 14 Apr 2016 01:17:09 GMT", "_links": {"self": {"href": "img/570eef95e3c45314416a2410", "title": "Img"}}, "_created": "Thu, 14 Apr 2016 01:17:09 GMT", "_status": "OK", "_id": "570eef95e3c45314416a2410", "_etag": "7b9176a5ee1db78bd397d424d088e45d2b11721f"}

# list all images in the db
curl -X GET http://127.0.0.1:5000/api/image

# get all results
curl -X GET -g http://127.0.0.1:5000/api/result

# get the reuslt for \_id=5744645ee3c45379ae86f2cc from result table
curl -X GET -g http://127.0.0.1:5000/api/result?where={%22_id%22:%20%225744645ee3c45379ae86f2cc%22}
{"_items": [{"_updated": "Thu, 01 Jan 1970 00:00:00 GMT", "_created": "Thu, 01 Jan 1970 00:00:00 GMT", "_id": "5744645ee3c45379ae86f2cc", "_links": {"self": {"href": "result/5744645ee3c45379ae86f2cc", "title": "Result"}}, "_etag": "768d74bd842e091f5699af4c96419baad6d00854"}], "_links": {"self": {"href": "result?where={\"_id\":\"5744645ee3c45379ae86f2cc\"}", "title": "result"}, "parent": {"href": "/", "title": "home"}}, "_meta": {"max_results": 25, "total": 1, "page": 1}}
```
