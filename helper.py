import os
import warnings

if __name__ == "__main__":
    #parameters
    rs = [2,1]
    Ls = [2,3,4,5,6,7,8,9]
    wds = [1e-3,1e-4,1e-5]
    datasetsizes = [2048,1024,512,256,128,64]
    labelnoise = [0,0.25,0.5,1]
    epochs = 60_100
    jobname = "new_targets"

    #print out current queue status
    os.system('squeue --format="%.10i %.9P %.75j %.8u %.8T %.10M %.9l %.6D %R" -u sueparkinson')
    #check on the status of the jobs
    print("num log files created:")
    numlogfiles = os.system(f'ls -1 log/{jobname}*/*.out | wc -l')
    print("errors in .err files:")
    os.system(f'cat log/{jobname}*/*.err')
    print('num jobs run on GPU:')
    gpujobs = int(os.popen(f'grep -irm 1 cuda log/{jobname}*/*.out | wc -l').read())
    print(gpujobs)
    print('num jobs run on CPU:')
    cpujobs = int(os.popen(f'grep -irm 1 cpu log/{jobname}*/*.out | wc -l').read())
    print(cpujobs)
    if cpujobs > 0: 
        warnings.warn(f"!!!WARNING!!! {cpujobs} ran on CPU")

    #see which files are missing
    missingcount = 0
    for r in rs:
        for ln in labelnoise:
            foldername = jobname + f"_labelnoise{ln}"
            print(f"\n\ntrained models missing for r={r},ln={ln}:")
            for datasetsize in datasetsizes:
                print(f"\nN={datasetsize}:",end="")
                for L in Ls:
                    print(f"\tL={L}:",end=" ")
                    for weight_decay in wds:
                        paramname=f"N{datasetsize}_L{L}_r{r}_wd{weight_decay}_epochs{epochs}"
                        filename = f'{foldername}/{paramname}'
                        if not os.path.isfile(f'{filename}model.pt'):
                            print(f"wd={weight_decay}",end=" ") 
                            missingcount += 1
    totalmodels = len(rs)*len(Ls)*len(wds)*len(datasetsizes)*len(labelnoise)
    print(f"\n\ntrying to train {totalmodels} total models")
    print(f"{gpujobs} have (at least) started training")
    print(f"{missingcount} models have not finished training or are missing")
    print(f"{gpujobs+missingcount-totalmodels} models are currently training")
