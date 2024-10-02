import json
import os
from python_fbas.fbas import FBAS, QSet

def gen_symmetric_fbas(num_orgs:int, output=None) -> FBAS:
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
        with open(output, 'w') as f:
            f.write(json.dumps(fbas.to_json()))
    return fbas