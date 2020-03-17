import os
import sys
import ast

def submit_script(scriptpath, args, environment, mem = "16G", cuda_cores = 1, jobname = "job"):
    """ Submits the specified script as a gridjob and passes arguments as command line parameters.
        Use the command line tools qstat to check the status of your gridjobs. Use qdel - u [YOUR_USERNAME] to delete
        all your running jobs. The output and error files are saved to your home directory (cd ~).
    Args:
        scriptpath: The path to the python script
        args: A list of arguments that is converted to strings and passed to the script executed remotely in the grid
        environment: The shell command used to activate the python/conda etc. environment.
        mem: A string that specifies the amount of memory the gridjob requires (e.g. 4G for four gigabytes)
        cuda_cores: The number of cuda_cores one wishes to use for computation.
        jobname: The name of the gridjob.
    """
    args = [str(a) for a in args]
    args_string = ""
    for a in args:
        args_string += a
        args_string += " "
    args_string = args_string[:-1]

    os.system('qsub -l mem='+ mem +' -l cuda_cores='+ str(cuda_cores)+" <<'toggle_here_document'\n"#use heredocument instead of loading a shell script from file
                + "#!/bin/bash \n"
                + "#$ -N " + jobname +"\n"
                + "#$ -l h_rt=24:00:00" + "\n"
                + "echo 'Start-time' \n"
                + "date \n"
                + "echo 'Host' \n"
                + "hostname \n"
                + environment + " \n"
                + 'python3 ' + scriptpath + " " + args_string + "\n"
                + "echo 'End-time'\n"
                + "date \n"
                + '\n'+'toggle_here_document')

def typecast(args):
    """ Casts strings contained in args to the automatically detected type
    Args:
        args: List of arguments where each element is a strings
    Returns:
        List of arguments parsed to the detected variable type.
    """
    typecasted = []
    for a in args:
        try:
            typecasted.append(ast.literal_eval(a))
        except:
            typecasted.append(a)
    return typecasted
