import plots as plts

def plots():
    tic.append(time.process_time())
    plts.elem_abundance()
    plts.iso_evolution()
    plts.iso_abundance()
    aux.tic_count(string="Plots saved in", tic=tic)
    return None