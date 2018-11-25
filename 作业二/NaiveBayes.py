
# coding: utf-8

# Naive Bayes



import pandas as pd
import numpy as np


# ### Read the Dataset file

#打开训练文件
df = pd.read_csv("Saving_file/Training Word Count.csv")


# ### Print the data file


df


# ### 聚合字数
Labels=[]
for label in df["Label"]:
    if label not in Labels:
        Labels.append(label)


Labels


headers = list(df)
unique_words = headers[1:]


Aggregate_Count_Matrix = pd.DataFrame(unique_words,columns=["Word"])


Aggregate_Count_Matrix



Counter=0
for Label in Labels:
        
    Current_Word_Count=np.array([0]*len(unique_words))
    
    i=0
    while i<len(df):
        if df["Label"][i]==Label:
            if Counter%200==0:
                print(str(int(Counter*100/len(df))) + "% Complete")
            Counter+=1
            Current_Document = df[i:i+1]
            Current_Document = Current_Document.values.tolist()
            Current_Document = Current_Document[0][1:]
            Current_Document = np.array(Current_Document)
            Current_Word_Count = Current_Word_Count + Current_Document
        i+=1
    Aggregate_Count_Matrix[Label] = Current_Word_Count


# In[10]:

Aggregate_Count_Matrix = Aggregate_Count_Matrix.set_index("Word")
Aggregate_Count_Matrix = Aggregate_Count_Matrix.transpose()


# In[11]:

Aggregate_Count_Matrix


# In[12]:

Aggregate_Count_Matrix.to_csv("Saving_file/Aggregate Word Count.csv")


# Probability Count With Laplacian Correction
# In[13]:

Probability_Count_Matrix = pd.DataFrame(unique_words,columns=["Word"])


# In[14]:

i=0
for Label in Labels:
    Current_Class = Aggregate_Count_Matrix[i:i+1]
    Current_Class = Current_Class.values.tolist()
    Current_Class = Current_Class[0]
    Current_Class = np.array(Current_Class)
    Current_Class = Current_Class+1
    
    Total = np.sum(Current_Class)
    Current_Class = Current_Class*5000/(Total)
    Probability_Count_Matrix[Label] = Current_Class
    i+=1
    


# In[15]:

Probability_Count_Matrix


# In[16]:

Probability_Count_Matrix = Probability_Count_Matrix.set_index("Word")
Probability_Count_Matrix = Probability_Count_Matrix.transpose()

Probability_Count_Matrix.to_csv("Saving_file/Probability Matrix.csv")  #保存


# ## Run a Test on Testing Data

# In[17]:

import pandas as pd
import numpy as np


# In[18]:

Probability_Count_Matrix = pd.read_csv("Saving_file/Probability Matrix.csv")  #打开


# In[19]:

Probability_Count_Matrix=Probability_Count_Matrix.rename(columns = {'Unnamed: 0':'Label'})


# In[20]:

Probability_Count_Matrix


# In[21]:

unique_words = list(Probability_Count_Matrix)[1:]


# In[22]:

Labels = Probability_Count_Matrix["Label"]


# In[23]:

Test_Frame = pd.read_csv("Saving_file/Testing Word Count.csv")
Test_Frame


# In[24]:

Actual_Results = Test_Frame["Label"].values.tolist()
Final_Results = pd.DataFrame(Actual_Results,columns=["Actual Labels"])
Predicted_Results = []


# In[29]:

i=0
while i<len(Test_Frame):
    
    
    Current_Document = Test_Frame[i:i+1]
    Current_Document = Current_Document.values.tolist()
    Current_Document = Current_Document[0][1:]
    Current_Document = np.array(Current_Document)
    
    Training_File_Count = pd.read_csv("Saving_file/Training File Count.csv")["Files Count"]
    Training_File_Count = Training_File_Count.values.tolist()
    Label_Probability = np.array(Training_File_Count)
    
    Without_Label_probability = []
    j=0
    while j<len(Probability_Count_Matrix):
        
        
        Current_Class = Probability_Count_Matrix[j:j+1]
        Current_Class = Current_Class.values.tolist()
        Current_Class = Current_Class[0][1:]
        Current_Class = np.array(Current_Class)
        
        Current_Class_wise_probability = Current_Class**Current_Document
        
        Without_Label_probability.append(np.prod(Current_Class_wise_probability))

        j+=1
    Without_Label_probability = np.array(Without_Label_probability)
    
    With_Label_Probability = Without_Label_probability*Label_Probability
    
    index_with_max_value = np.argmax(With_Label_Probability)
    
    Predicted_Results.append(Labels[index_with_max_value])
    
    
    i+=1


# In[33]:

Final_Results["Predicted Labels"] = Predicted_Results
Final_Results = Final_Results.set_index("Actual Labels")
Final_Results.to_csv("Saving_file/Results.csv")


# ## Confusion Matrix混合矩阵

# In[43]:

Labels


# In[40]:

print(Labels)


# In[39]:

import pandas as pd
from tabulate import tabulate

results = pd.read_csv("Saving_file/Results(0).csv")

actual = results["Actual Labels"]
predicted = results["Predicted Labels"]

conf_matrix=[]
footer=["Total"]
classes = Labels #Labels是测试文件，准确率还是：77%

for c in classes:
    footer.append(0)
footer.append(0)

i=0
for c in classes:
    
    current=[c]
    for c1 in classes:
        current.append(0)
    current.append(0)
    
    if i==0:
        header=["X"]
        for c2 in classes:
            header.append(c2)
        header.append("Total")
        conf_matrix.append(header)
        i+=1
        
    
    current_pt=0
    for label in actual:
        if label==c:
            current_class=0
            for c3 in classes:
                if predicted[current_pt]==c3:
                    current[current_class+1]+=1
                    footer[current_class+1]+=1
                current_class+=1
            current[-1]+=1
            footer[-1]+=1
                    
                
        current_pt+=1
        
    conf_matrix.append(current)
            
conf_matrix.append(footer)
print (tabulate(conf_matrix))


# Efficiency

Truthness = (actual==predicted)

Rights=0
for k in Truthness:
    if k==True:
        Rights+=1

print("Efficiency : " + str(int(Rights*100/len(actual))) + "%")





