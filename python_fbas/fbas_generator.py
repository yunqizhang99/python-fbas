import json
import os
import random
import logging
from python_fbas.fbas import FBAS, QSet

def gen_symmetric_fbas(num_orgs:int, output=None) -> FBAS:
    """
    Generates a symmetric FBAS with num_orgs organizations, each with 3 validators. The inter-org threshold is set to 2/3 of the total number of organizations, and the intra-org threshold is set to 2 out of 3.
    """
    org_threshold = int(2*num_orgs/3)+1
    def org_validators(o):
        return ["v-"+str(o)+"-"+str(i) for i in range(1, 4)]
    def org_qset(o):
        return QSet.make(2, org_validators(o), [])
    meta = {v : {'homeDomain' : 'o-'+str(o)} for o in range(1, num_orgs+1) for v in org_validators(o)}
    qset = QSet.make(org_threshold, [], [org_qset(o) for o in range(1, num_orgs+1)])
    fbas = FBAS(
        {v : qset for o in range(1, num_orgs+1) for v in org_validators(o)},
        meta)
    if output:
        # checkout that output does not exist in the current directory:
        if os.path.exists(output):
            raise FileExistsError
        # write to file output in the current directory:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(fbas.to_json()))
    return fbas

def gen_asymmetric_fbas(num_orgs:int, output=None) -> FBAS:
    """
    Generates an asymmetric FBAS with num_orgs organizations, each with 3 validators. The inter-org threshold varies randomly between 1/2 and 1, and the intra-org threshold is set to 2 out of 3.
    """
    def org_validators(o):
        return ["v-"+str(o)+"-"+str(i) for i in range(1, 4)]
    def org_qset(o):
        return QSet.make(int(len(org_validators(o))/2)+1, org_validators(o), [])
    meta = {v : {'homeDomain' : 'o-'+str(o)} for o in range(1, num_orgs+1) for v in org_validators(o)}
    qset = {o : QSet.make(random.randint(int(num_orgs/2)+1, num_orgs), [], [org_qset(o) for o in range(1, num_orgs+1)]) for o in range(1, num_orgs+1)}
    thresholds = [qset[o].threshold for o in range(1, num_orgs+1)]
    logging.info(f"threshold are {thresholds}")
    fbas = FBAS(
        {v : qset[o] for o in range(1, num_orgs+1) for v in org_validators(o)},
        meta)
    if output:
        # checkout that output does not exist in the current directory:
        if os.path.exists(output):
            raise FileExistsError
        # write to file output in the current directory:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(fbas.to_json()))
    return fbas