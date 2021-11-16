import numpy as np
import pandas as pd
import os.path
# Run in the master directory

def lc18_example(filename='tab_yieldstot_iso_exp_pd.dec', loc='input/yields/snii/lc18/tab_R', 
                 outfilename='lc18_pandas.csv', split_size=12, columns=['elemSymb', 'elemZ',
                  'elemA', 'initial', '013a000', '015a000', '020a000', '025a000', '030a000',
                   '040a000', '060a000', '080a000', '120a000']):
    '''
    Example of how to melt a custom yield with consecutive even tables
    into a GalCEM-friendly input format.

        split_size      in this case I know there are 12 tables, for 4 initial metallicities and 3 initial velocities
    '''
    output_name = f'{loc}/{outfilename}'
    print(f'output name\n{output_name}')
    if not os.path.exists(output_name):
        print('Processing yields')
        df = pd.read_table(f'{loc}/{filename}', sep=',  ', dtype={'ID': object}, header=None)
        print(f'Table import. \n {df}')
        df = np.array_split(df, split_size) 
        print(f'Table split. \n {df}')
        new_headers = [d.iloc[0].values for d in df]
        f = [pd.DataFrame(d[1:]) for d in df]
        for i,val in f:
            val.columns = new_headers[i]

        f.to_csv(output_name, header=True, index=False)
        print(f'Processed yields saved in {output_name}')
    return None

def k10_example(filename=['Z0.0001.dat', 'Z0.008.dat', 'Z0.004.dat', 'Z0.02.dat'], 
                loc='input/yields/lims/k10', outfilename='k10_pandas.csv',
                columns=['Mini','Zini','Mfin','elemSymb','elemZ','elemA','Yield',
                'Mi_windloss','Mi_ini','Xi_avg','Xi_ini','ProdFact']):
    '''
    Example of how to merge multiple yield tables
    into a GalCEM-friendly input format.
    '''
    output_name = f'{loc}/{outfilename}'
    print(f'output name\n{output_name}')
    if not os.path.exists(output_name):
        print('Processing yields')
        df = [pd.read_csv(f'{loc}/{name}', comment='#') for name in filename] 
        print(f'Table import. \n {df}')
        # !!!!!!! careful! If you use this function, uncomment the header, otherwise the reading will skip the first line for the 'g' element
        df = pd.DataFrame(np.vstack(df), columns=columns)
        print(f'Processed table. \n {df}')
        df.to_csv(output_name, header=True, index=False)
        print(f'Processed yields saved in {output_name}')
    return None