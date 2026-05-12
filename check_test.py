import sys, json
sys.path.insert(0, 'consensus_lab')
import app as a

with a.app.test_client() as c:
    scenarios = ['gaussian','overlap','elongated','density','highdim','imbalance']
    print('PREVIEW ENDPOINT')
    for sc in scenarios:
        r = c.post('/api/generate-preview',
            data=json.dumps({'scenario':sc,'n_samples':300,'n_clusters':3,'dim':2,'difficulty':'medium','seed':19}),
            content_type='application/json')
        d = json.loads(r.data)
        err = d.get('error')
        if err:
            print('  ' + sc + ': ERROR - ' + str(err))
        else:
            print('  ' + sc + ': OK  n=' + str(d['n']) + ' k=' + str(d['k']) + ' dim=' + str(d['dim']) + ' projected=' + str(d['projected']))
    print('')
    print('EXPERIMENT RUN')
    algos = ['hierarchical_baseline','hierarchical_weighted','sdgca','sdgca_modified']
    for algo in algos:
        r = c.post('/api/experiment/run',
            data=json.dumps({'dataset':'analysis_densired_compact','algorithm':algo,'method':'average','seed':19,'m':5,'runs':1}),
            content_type='application/json')
        d = json.loads(r.data)
        if d.get('error'):
            print('  ' + algo + ': ERROR - ' + str(d['error']))
        else:
            print('  ' + algo + ': OK  NMI=' + str(round(d['nmi_mean'],3)) + ' ARI=' + str(round(d['ari_mean'],3)))
