import sas_modeling.calc_discrepancy
import sas_modeling.calc_i0
import sas_modeling.compare
import sas_modeling.file_io
import sas_modeling.geometric
import sas_modeling.make_figures

sasmol_error = ('sasmol source: https://github.com/madscatt/sasmol')

try:
    import sas_modeling.cluster_ensemble
except ImportError as e:
    print('WARNING: sas_modeling.cluster_ensemble unavailable')
    print('ImportError: {}'.format(e.message))
    if 'sasmol' in e.message:
        print(sasmol_error)

try:
    import sas_modeling.calc_pr
except ImportError as e:
    print('WARNING: sas_modeling.calc_pr unavailable')
    print('ImportError: {}'.format(e.message))
    if 'sasmol' in e.message:
        print(sasmol_error)

del(sasmol_error)
