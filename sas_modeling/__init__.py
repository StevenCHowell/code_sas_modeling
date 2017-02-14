import sas_modeling.calc_discrepancy
import sas_modeling.calc_i0
import sas_modeling.compare
import sas_modeling.file_io
import sas_modeling.geometric
import sas_modeling.make_figures

sasmol_error = ('not available, depends on sasmol: '
        'https://github.com/madscatt/sasmol')

try:
    import sas_modeling.cluster_ensemble
except ImportError:
    print('cluster_ensemble {}'.format(sasmol_error))

try:
    import sas_modeling.calc_pr
except ImportError:
    print('calc_pr {}'.format(sasmol_error))

del(sasmol_error)
