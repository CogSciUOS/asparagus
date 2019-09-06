import os
import sys


def submit_script(scriptpath, args, mem = "4G", cuda_cores = 0,jobname = "job"):
    args_string = ""
    for a in args:
        args_string += a
        args_string += " "
    args_string = args_string[:-1]


    os.system('qsub -l mem='+ mem +' -l cuda_cores='+ str(cuda_cores)+" <<'toggle_here_document'\n"#use here document instead of loading a shell script from file
                + "#!/bin/bash \n"
                + "#$ -N " + jobname +"\n"
                + "#$ -l h_rt=01:30:00" + "\n"
                + "echo 'Start-time' \n"
                + "date \n"
                + "echo 'Host' \n"
                + "hostname \n"
                + 'source /net/projects/scratch/summer/valid_until_31_January_2020/ann4auto/env/gpuenv/bin/activate \n'
                + 'python3 ' + scriptpath + " " + args_string + "\n"
                + "echo 'End-time'\n"
                + "date \n"
                + '\n'+'toggle_here_document')
    
if __name__ == "__main__":
    submit_script(os.getcwd()+"/"+sys.argv[1],sys.argv[2:])

