import numpy as np
import pandas as pd
import os.path
import itertools

import prep.inputs as INp
IN = INp.Inputs()
# Run in the master directory after having imported the new yields in the appropriate 'loc'

def lc18_example(filename='tab_yieldstot_iso_exp_pd.dec', loc='input/yields/snii/lc18/tab_R', 
                 outfilename='lc18_pandas.csv'):
    '''
    Example of how to melt a custom yield with consecutive even tables
    into a GalCEM-friendly input format.

        split_size      in this case I know there are 12 tables, for 4 initial metallicities and 3 initial velocities
    '''
    output_name = loc + outfilename
    print(f'output name\n',output_name)
    if not os.path.exists(output_name):
        Zini = np.power(10, [0., -1., -2., -3.])
        vel = [0., 150., 300.]
        mass = ['13.', '15.', '20.', '25.', '30.', '40.', '60.', '80.', '120.']
        header = ['elemSymb', 'elemZ', 'elemA', 'mass_i_ini'] + mass
        numerics = header[1:]
        print('Processing yields')
        df = pd.read_table(f'{loc}/{filename}', sep=',  ', dtype={'ID': object}, header=None)
        print(f'Table import. \n ',df,'\n\n')
        d = np.array_split(df, len(Zini))
        df = [np.array_split(f, len(vel)) for f in d]
        print(f'Table split. \n ',df,'\n\n')
        # If you want to rename the headers
        #new_headers = [d.iloc[0].values for d in df]
        #f = [pd.DataFrame(d[1:]) for d in df]
        #for i, val in enumerate(f):
        #    val.columns = new_headers[i]
        #print(f'Table rename. \n {f}')
        for i, d in enumerate(df):
            for j, f in enumerate(d):
                df[i][j].columns = header
                df[i][j] = pd.concat([df[i][j], 
                           pd.DataFrame({'Z_ini': [IN.solar_metallicity * Zini[i]]}, index=df[i][j].index),
                           pd.DataFrame({'vel_ini': [vel[j]]}, index=df[i][j].index)], axis=1)
                df[i][j] = df[i][j][1:]
                df[i][j][numerics] = df[i][j][numerics].apply(pd.to_numeric)
                df[i][j] = df[i][j][['vel_ini', 'Z_ini']+header]
        df = pd.concat(list(itertools.chain(*df)), axis=0)
        print(f'Saving table \n ',df,'\n\n')
        with open(output_name, 'w') as f:
            f.write(f'# From file: {loc}/{filename}\n')
        df.to_csv(output_name, mode='a', header=True, index=False)
        print(f'Processed yields saved in \n',output_name)
    return None

def k10_example(filename=['Z0.0001.dat', 'Z0.008.dat', 'Z0.004.dat', 'Z0.02.dat'], 
                loc='input/yields/lims/k10', outfilename='k10_pandas.csv',
                columns=['mass_ini','Z_ini','mass_fin','elemSymb','elemZ','elemA','yields',
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
        # !!!!!!! careful! If you use this function, uncomment the header in each yield file, otherwise the reading will skip the first line for the 'g' element
        df = pd.DataFrame(np.vstack(df), columns=columns)
        print(f'Processed table. \n {df}')
        with open(output_name, 'w') as f:
            f.write(f'# From file: {loc}/{filename}\n')
        df.to_csv(output_name, mode='a', header=True, index=False)
        print(f'Processed yields saved in {output_name}')
    return None