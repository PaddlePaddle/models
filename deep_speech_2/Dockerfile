FROM paddlepaddle/paddle:latest-gpu
ADD ./setup.sh ./requirements.txt /
RUN bash -x ./setup.sh && rm -rf ./setup.sh && ldconfig
CMD ["paddle", "version"]
