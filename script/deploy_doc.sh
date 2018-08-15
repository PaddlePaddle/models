exit_code=0

if [[ "$TRAVIS_PULL_REQUEST" != "false" ]]; then exit $exit_code; fi;
    
# Deploy to the the content server if its a "develop" or "release/version" branch
# The "develop_doc" branch is reserved to test full deploy process without impacting the real content.
if [ "$TRAVIS_BRANCH" == "develop_doc" ]; then
    PPO_SCRIPT_BRANCH=develop
elif [[ "$TRAVIS_BRANCH" == "develop"  ||  "$TRAVIS_BRANCH" =~ ^v|release/[[:digit:]]+\.[[:digit:]]+(\.[[:digit:]]+)?(-\S*)?$ ]]; then
    PPO_SCRIPT_BRANCH=master
else
    # Early exit, this branch doesn't require documentation build
    exit $exit_code;
fi

export DEPLOY_DOCS_SH=https://raw.githubusercontent.com/PaddlePaddle/PaddlePaddle.org/$PPO_SCRIPT_BRANCH/scripts/deploy/deploy_docs.sh

docker run -it \
    -e CONTENT_DEC_PASSWD=$CONTENT_DEC_PASSWD \
    -e TRAVIS_BRANCH=$TRAVIS_BRANCH \
    -e DEPLOY_DOCS_SH=$DEPLOY_DOCS_SH \
    -e TRAVIS_PULL_REQUEST=$TRAVIS_PULL_REQUEST \
    -e PPO_SCRIPT_BRANCH=$PPO_SCRIPT_BRANCH \
    -e PADDLE_ROOT=/models \
    -v "$PWD:/models" \
    -w /models \
    paddlepaddle/paddle:latest-dev \
    /bin/bash -c 'curl $DEPLOY_DOCS_SH | bash -s $CONTENT_DEC_PASSWD $TRAVIS_BRANCH /models models/build/doc/ $PPO_SCRIPT_BRANCH' || exit_code=$(( exit_code | $? ))
