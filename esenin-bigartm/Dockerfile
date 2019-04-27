FROM ubuntu:16.04

RUN apt-get -y update
RUN apt-get -y install git make cmake build-essential libboost-all-dev

RUN apt-get -y install python-numpy python-pandas python-scipy
RUN apt-get -y install python-pip
RUN pip install protobuf tqdm wheel flask

RUN git clone --branch v0.9.0 https://github.com/bigartm/bigartm.git
WORKDIR bigartm
RUN mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr .. && make -j && make install
RUN cd 3rdparty/protobuf-3.0.0/python && python2 setup.py build && python2 setup.py install
RUN cd python && python2 setup.py install

ENV ARTM_SHARED_LIBRARY=/usr/lib/libartm.so

EXPOSE 9000
COPY main.py /app/main.py
CMD ["/app/main.py"]

