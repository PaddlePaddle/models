import os


def dist_env():
    """
    Return a dict of all variable that distributed training may use.
    NOTE: you may rewrite this function to suit your cluster environments.
    """
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    num_trainers = 1
    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    assert(training_role == "PSERVER" or training_role == "TRAINER")

    # - PADDLE_TRAINER_ENDPOINTS means nccl2 mode.
    # - PADDLE_PSERVER_ENDPOINTS means pserver mode.
    # - PADDLE_CURRENT_ENDPOINT means current process endpoint.
    worker_endpoints = []
    port = os.getenv("PADDLE_PORT", "8701")
    if os.getenv("PADDLE_TRAINER_ENDPOINTS"):
        trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    else:# for paddlecloud
        worker_ips = os.getenv("PADDLE_TRAINERS", "")
        for ip in worker_ips.split(","):
            worker_endpoints.append(':'.join([ip, port]))
        trainer_endpoints = ",".join(worker_endpoints)

    pserver_ips = os.getenv("PADDLE_PSERVERS", "")
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)

    if os.getenv("PADDLE_CURRENT_ENDPOINT"):
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
    else:# for paddlecloud
        current_endpoint = os.getenv("POD_IP", "") + ":" + port
    if trainer_endpoints:
        trainer_endpoints = trainer_endpoints.split(",")
        num_trainers = len(trainer_endpoints)
    elif pserver_endpoints:
        num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    
    return {
        "trainer_id": trainer_id,
        "num_trainers": num_trainers,
        "current_endpoint": current_endpoint,
        "training_role": training_role,
        "pserver_endpoints": pserver_endpoints,
        "trainer_endpoints": trainer_endpoints
    }
