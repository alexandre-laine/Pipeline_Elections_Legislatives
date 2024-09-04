import subprocess
import os 

rapport_path = os.path.join(os.getcwd(), "rapport")

def choose_directory(rapport_path):

    os.chdir(rapport_path)

def run_pdflatex(tex_file):

    command = ["pdflatex", tex_file+".tex"]
    process = subprocess.Popen(command)
    process.communicate()  # Wait for the process to finish

def run_biber(tex_file):

    command = ["biber", tex_file]
    process = subprocess.Popen(command)
    process.communicate()  # Wait for the process to finish

def run_cleaning(file, end):
    command = ["del", f"{file}.{end}"]
    process = subprocess.Popen(command, shell=True)
    process.communicate()  # Wait for the process to finish


file_name = "LAINE_Alexandre_Memoire_DESU_2024"

choose_directory(rapport_path)
run_pdflatex(file_name)
run_biber(file_name)
run_pdflatex(file_name)
run_pdflatex(file_name)
for end in  ["bcf","bbl","blg", "log", "aux", "out", "run.xml", "toc"]:
    run_cleaning(file_name, end)