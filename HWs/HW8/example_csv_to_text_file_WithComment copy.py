#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:35:01 2023

@author: kendrickshepherd
"""

# this file will read a CSV file of student grades and 
# output a student-specific file of where they received and 
# missed points
def OutputStudentEvaluations(filename,input_folder,output_folder):
    full_filename = input_folder + "/" + filename # It's a common method used to create a file
                                                  # path by combining folder paths and filenames
                                                  # in Python."path/to/folder/file.txt"
    feedback_filename = "Homework_8_Feedback.txt"
    
    # First name, last name, and net id must be in the 
    # resulting file name for Learning Suite to be able to 
    # automatically upload the information
    first_name_line = 0 # line index of the student first names
    last_name_line = 1 # line index of the student last names
    net_id_line = 2 # line index of the student netids
    
    # the indices of the lines that start new problems
    problem_start_lines = [3]
    full_score_line = 20 # index of the line with the student score
    first_student_column = 3 # index of column with first student
    
    lines = []
    counter = 0
    with open(full_filename, 'r') as file: #This line is used to open a file named by the 
                                           #variable full_filename in read mode ('r'). 
                                           #The with statement is a context manager that 
                                           #ensures proper handling of resources, like files, 
                                           #by automatically closing the file when it's no 
                                           #longer needed
        for line in file:
            no_return_symbol = line.strip()
            # The strip() method in Python is used to remove leading and trailing whitespace 
            #characters (such as spaces, tabs, or newline characters) from a string.
            
            #For instance, if line contains a string like " Hello " (with spaces before and
            #after "Hello"), line.strip() will return "Hello" (without any leading or 
            #trailing spaces).
            if '"' in no_return_symbol:
                no_return_symbol = no_return_symbol.replace('"','',1) #replace " with empty
                # string. 
                #The 1 as the third argument specifies to perform this replacement only once.
                split_lines = no_return_symbol.split('"') #For instance, if no_return_symbol 
                #contains a string like "Hello" "world", applying no_return_symbol.split('"') 
                #would result in a list of substrings: ['', 'Hello', ' ', 'world', '']. 
                split_lines[1] = split_lines[1].replace(",","",1) # If there was a comma in 
                #the second element, it would replace the first comma encountered with an 
                #empty string due to the 1
                problem_statement = split_lines[0]
                results = split_lines[1].split(",") # For instance, if split_lines[1] contains
                # a string like "apple,orange,banana", executing split_lines[1].split(",") 
                # would result in a list of substrings: ['apple', 'orange', 'banana'].
                
                results.insert(0,problem_statement) # in this case, it's inserting 
                #problem_statement at the beginning of the results list, shifting the existing 
                #elements one position to the right.
                lines.append(results)
                split_index = no_return_symbol.find('"',1) # .find('"', 1): The find() method 
                #searches for the specified substring (in this case, ") within the string, 
                #starting the search from the given index (1 in this case). It returns the 
                #index of the first occurrence of the substring within the string.
                
                print("Problem line with column:",counter,split_index)
            else:
                lines.append(no_return_symbol.split(","))    # ?
            counter += 1
    
    # iterate through each student (the columns of the file)
    for i in range(first_student_column,len(lines[0])):
        first_name = lines[first_name_line][i]
        last_name = lines[last_name_line][i]
        net_id = lines[net_id_line][i]
        # output format that can be read by Learning Suite
        identifier = last_name + "_" + first_name + "_" + net_id    #output file name 
                                                                    #Ramin_Pahnabi_rpahnabi
        out_file = output_folder + "/" + identifier + "_" + feedback_filename
                      #for ex:path/to/folder/Ramin_Pahnabi_rpahnabi_Homework_4_Feedback.txt
                                           
        with open(out_file, 'w') as f:              #the same as the previous one for read
            f.write("======================\n")
            f.write("Feedback on Homework 8\n")
            f.write("======================\n")
            
            problem_found = False
            new_problem = False
            total_found = False
            for j in range(0,len(lines)):
                if j in problem_start_lines:
                    new_problem = True
                    problem_found = True
                    f.write("------------------\n")   
                    f.write(lines[j][0]+"\n")        # writing number of problem 
                    f.write("------------------\n")
                    f.write("\n")
                elif j == full_score_line:
                    total_found = True
                    f.write("\n")
                    f.write("******************\n")
                    f.write("Total Score: " + lines[j][i] + "/" + lines[j][1] + "\n") 
   # can i write lines[j][i] mean is total score of each student
   
   # Note: in the python everything will be considered from 0, Ex lines[1][2]=row:2 column:3
                    f.write("******************\n")
                elif not problem_found:
                    continue
                else: # problem has been found... write the ouptput         # ?
                    if lines[j][0] == "": #line of each Q section describtion   
                        f.write("\n")
                        continue
                    elif lines[j][i] == "":
                        f.write("\n")
                        f.write(lines[j][0] + ":\n")
                        f.write("\n")
                        continue
                    f.write(lines[j][0] + ":\n")
                    f.write(lines[j][i] + "/" + lines[j][1] + "\n")
    
    return lines
    

input_folder = "Solution_H_8"  # this folder(Solution_H_4) should be created in the
                               # same folder as this  python file folder
student_evaluations = "Homework_8.csv" # should be lacated in the input_folder

output_folder = "Student_Feedback" # this folder(Solution_Feedback)will be created 
                                   # in the same folder as this  python file folder
lines = OutputStudentEvaluations(student_evaluations,input_folder,output_folder)